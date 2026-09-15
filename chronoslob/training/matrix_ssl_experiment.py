"""Train-only pretraining loop and checkpointing for matrix SSL.

This module wires :class:`~chronoslob.models.matrix_ssl.MatrixSSLModel` into a
small, deterministic CPU-first pretraining loop and provides encoder
checkpoint save/load helpers used by the FI-2010 SSL runner.

The pretraining loop reports the train loss and a *train-carved* validation
pretraining loss. The validation loader, when supplied, must be built from a
partition carved out of the official training rows; this module never inspects
labels and never touches test rows.

Nothing here makes a market-performance, profitability or benchmark-ranking
claim.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chronoslob.models.matrix_ssl import (
    MatrixSSLConfig,
    MatrixSSLModel,
    create_matrix_ssl_model,
)
from chronoslob.training.torch_training import set_torch_deterministic

try:  # pragma: no cover - exercised when torch is unavailable
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    import torch as _torch_typing  # noqa: F401

__all__ = [
    "MATRIX_SSL_CHECKPOINT_VERSION",
    "MatrixSSLEpochResult",
    "MatrixSSLTrainingConfig",
    "evaluate_matrix_ssl",
    "fit_matrix_ssl",
    "load_pretrained_encoder_state",
    "save_pretrained_encoder",
    "train_matrix_ssl_one_epoch",
]

MATRIX_SSL_CHECKPOINT_VERSION = "fi2010-matrix-ssl/encoder/v1"


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the matrix SSL experiment. Install the "
            "'torch' optional dependency: pip install -e '.[torch]'"
        )
    return torch


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_positive_float(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float")
    numeric = float(value)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    return numeric


def _validate_non_negative_float(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float")
    numeric = float(value)
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _resolve_device(device: str) -> Any:
    torch_module = _require_torch()
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty string")
    normalised = device.strip().lower()
    if normalised == "cpu":
        return torch_module.device("cpu")
    if normalised.startswith("cuda"):
        if not torch_module.cuda.is_available():
            raise RuntimeError(
                f"requested device {device!r} but CUDA is not available; "
                "ChronosLOB matrix SSL training defaults to CPU"
            )
        return torch_module.device(normalised)
    raise ValueError(
        f"unsupported device {device!r}; matrix SSL training supports 'cpu' or "
        "'cuda'-prefixed devices"
    )


@dataclass(frozen=True)
class MatrixSSLTrainingConfig:
    """Configuration for the matrix SSL pretraining loop."""

    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = 1.0
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self) -> None:
        _validate_positive_int(self.epochs, name="epochs")
        _validate_positive_float(self.learning_rate, name="learning_rate")
        _validate_non_negative_float(self.weight_decay, name="weight_decay")
        if self.gradient_clip_norm is not None:
            _validate_positive_float(
                self.gradient_clip_norm, name="gradient_clip_norm"
            )
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        _validate_non_negative_int(self.seed, name="seed")


@dataclass(frozen=True)
class MatrixSSLEpochResult:
    """Per-epoch summary returned by :func:`fit_matrix_ssl`."""

    epoch: int
    train_loss: float
    train_loss_components: dict[str, float] = field(default_factory=dict)
    validation_loss: float | None = None
    validation_loss_components: dict[str, float] = field(default_factory=dict)


def _move_batch_to_device(batch: Mapping[str, Any], device_obj: Any) -> dict[str, Any]:
    torch_module = _require_torch()
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device_obj) if torch_module.is_tensor(value) else value
    return moved


def _assert_finite_gradients(model: Any) -> None:
    torch_module = _require_torch()
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if torch_module.isnan(parameter.grad).any().item():
            raise RuntimeError(f"gradient for parameter {name!r} contains NaN")
        if torch_module.isinf(parameter.grad).any().item():
            raise RuntimeError(
                f"gradient for parameter {name!r} contains infinite values"
            )


def _accumulate_components(
    accumulator: dict[str, float],
    counts: dict[str, int],
    components: Mapping[str, Any],
    weight: int,
) -> None:
    for name, value in components.items():
        accumulator[name] = accumulator.get(name, 0.0) + float(value.item()) * weight
        counts[name] = counts.get(name, 0) + weight


def _average_components(
    accumulator: Mapping[str, float],
    counts: Mapping[str, int],
) -> dict[str, float]:
    averaged: dict[str, float] = {}
    for name, total in accumulator.items():
        count = counts.get(name, 0)
        if count == 0:
            continue
        averaged[name] = total / count
    return averaged


def _forward_batch(model: MatrixSSLModel, batch: Mapping[str, Any]) -> Any:
    return model(
        batch["x"],
        mask=batch.get("mask"),
        masked_target=batch.get("masked_target"),
        next_bucket_labels=batch.get("next_bucket_labels"),
    )


def train_matrix_ssl_one_epoch(
    model: MatrixSSLModel,
    dataloader: Iterable[Any],
    optimizer: Any,
    *,
    device: str = "cpu",
    gradient_clip_norm: float | None = None,
) -> tuple[float, dict[str, float]]:
    """Run one matrix SSL training epoch."""
    torch_module = _require_torch()
    if model is None:
        raise ValueError("model must not be None")
    if optimizer is None:
        raise ValueError("optimizer must not be None")
    device_obj = _resolve_device(device)
    model.train()
    model.to(device_obj)

    total_loss = 0.0
    total_samples = 0
    component_totals: dict[str, float] = {}
    component_counts: dict[str, int] = {}

    for raw_batch in dataloader:
        batch = _move_batch_to_device(raw_batch, device_obj)
        batch_size = int(batch["x"].shape[0])
        if batch_size == 0:
            continue
        optimizer.zero_grad()
        output = _forward_batch(model, batch)
        if output.loss is None:
            raise RuntimeError(
                "matrix SSL forward did not produce a loss; ensure mask, "
                "masked_target or next_bucket_labels are present"
            )
        loss = output.loss
        if not torch_module.isfinite(loss).item():
            raise RuntimeError("matrix SSL training loss became non-finite")
        loss.backward()
        _assert_finite_gradients(model)
        if gradient_clip_norm is not None:
            if gradient_clip_norm <= 0:
                raise ValueError("gradient_clip_norm must be positive when provided")
            torch_module.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(gradient_clip_norm),
            )
        optimizer.step()
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        _accumulate_components(
            component_totals,
            component_counts,
            output.loss_components,
            batch_size,
        )

    if total_samples == 0:
        raise ValueError("train_matrix_ssl_one_epoch received an empty dataloader")
    averaged = _average_components(component_totals, component_counts)
    return total_loss / total_samples, averaged


def evaluate_matrix_ssl(
    model: MatrixSSLModel,
    dataloader: Iterable[Any],
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate a matrix SSL model without updating its parameters."""
    torch_module = _require_torch()
    if model is None:
        raise ValueError("model must not be None")
    device_obj = _resolve_device(device)
    model.eval()
    model.to(device_obj)

    total_loss = 0.0
    total_samples = 0
    component_totals: dict[str, float] = {}
    component_counts: dict[str, int] = {}

    with torch_module.no_grad():
        for raw_batch in dataloader:
            batch = _move_batch_to_device(raw_batch, device_obj)
            batch_size = int(batch["x"].shape[0])
            if batch_size == 0:
                continue
            output = _forward_batch(model, batch)
            if output.loss is None:
                raise RuntimeError("matrix SSL evaluation forward produced no loss")
            if not torch_module.isfinite(output.loss).item():
                raise RuntimeError("matrix SSL evaluation loss became non-finite")
            total_loss += float(output.loss.item()) * batch_size
            total_samples += batch_size
            _accumulate_components(
                component_totals,
                component_counts,
                output.loss_components,
                batch_size,
            )

    if total_samples == 0:
        raise ValueError("evaluate_matrix_ssl received an empty dataloader")
    return {
        "loss": total_loss / total_samples,
        "loss_components": _average_components(component_totals, component_counts),
        "n_samples": int(total_samples),
    }


def fit_matrix_ssl(
    model: MatrixSSLModel,
    train_loader: Iterable[Any],
    validation_loader: Iterable[Any] | None = None,
    config: MatrixSSLTrainingConfig | None = None,
) -> list[MatrixSSLEpochResult]:
    """Pretrain a matrix SSL model and return per-epoch results.

    ``validation_loader`` must be built from rows carved out of the official
    training partition. It is used only to report a validation pretraining
    loss; no labels or test rows are consulted.
    """
    _require_torch()
    if model is None:
        raise ValueError("model must not be None")
    training_config = config if config is not None else MatrixSSLTrainingConfig()
    if not isinstance(training_config, MatrixSSLTrainingConfig):
        raise TypeError("config must be a MatrixSSLTrainingConfig instance")
    device_obj = _resolve_device(training_config.device)
    set_torch_deterministic(training_config.seed)

    model.to(device_obj)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    history: list[MatrixSSLEpochResult] = []
    for epoch in range(1, training_config.epochs + 1):
        train_loss, train_components = train_matrix_ssl_one_epoch(
            model,
            train_loader,
            optimizer,
            device=training_config.device,
            gradient_clip_norm=training_config.gradient_clip_norm,
        )
        validation_loss: float | None = None
        validation_components: dict[str, float] = {}
        if validation_loader is not None:
            evaluation = evaluate_matrix_ssl(
                model,
                validation_loader,
                device=training_config.device,
            )
            validation_loss = float(evaluation["loss"])
            validation_components = dict(evaluation["loss_components"])
        history.append(
            MatrixSSLEpochResult(
                epoch=epoch,
                train_loss=train_loss,
                train_loss_components=train_components,
                validation_loss=validation_loss,
                validation_loss_components=validation_components,
            )
        )
    return history


def save_pretrained_encoder(
    model: MatrixSSLModel,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save the transferable encoder weights and SSL config to ``path``.

    The checkpoint stores only the shared encoder parameters (not the SSL
    reconstruction heads and not any classification head), the architecture
    fields needed to rebuild a matching supervised classifier, and optional
    metadata. Returns the resolved checkpoint path.
    """
    torch_module = _require_torch()
    if not isinstance(model, MatrixSSLModel):
        raise TypeError("model must be a MatrixSSLModel instance")
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    config = model.config
    payload: dict[str, Any] = {
        "checkpoint_version": MATRIX_SSL_CHECKPOINT_VERSION,
        "encoder_state_dict": model.encoder_state_dict(),
        "architecture": {
            "input_features": int(config.input_features),
            "model_dim": int(config.model_dim),
            "num_heads": int(config.num_heads),
            "num_layers": int(config.num_layers),
            "feedforward_dim": int(config.feedforward_dim),
            "dropout": float(config.dropout),
            "max_sequence_length": int(config.max_sequence_length),
        },
        "ssl_objectives": list(config.enabled_objectives()),
        "metadata": dict(metadata or {}),
    }
    torch_module.save(payload, resolved)
    return resolved


def load_pretrained_encoder_state(path: str | Path) -> dict[str, Any]:
    """Load and return the encoder state dict saved by :func:`save_pretrained_encoder`."""
    torch_module = _require_torch()
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"pretrained encoder checkpoint not found: {resolved}")
    payload = torch_module.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "encoder_state_dict" not in payload:
        raise ValueError(
            "checkpoint does not contain an 'encoder_state_dict'; it was not "
            "written by save_pretrained_encoder"
        )
    encoder_state = payload["encoder_state_dict"]
    if not isinstance(encoder_state, Mapping):
        raise ValueError("encoder_state_dict must be a mapping of tensors")
    return {str(key): value for key, value in encoder_state.items()}


# Reference imports that document the contract this loop relies on. The
# pretraining model is always constructed via ``create_matrix_ssl_model`` so
# the encoder parameter names stay aligned with the supervised classifier.
_RESERVED_MODEL_FACTORY = create_matrix_ssl_model
_RESERVED_CONFIG = MatrixSSLConfig
