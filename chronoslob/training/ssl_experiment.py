"""Tiny CPU-only training and evaluation utilities for SSL pretraining.

This module wires the Phase-11 tokeniser, the Phase-12 transformer encoder
and the Phase-13 SSL wrapper into a small, deterministic SSL training and
evaluation path. It includes:

* :class:`SSLTrainingConfig`
* :class:`SSLEpochResult`
* :func:`train_ssl_one_epoch`
* :func:`evaluate_ssl`
* :func:`fit_ssl_model`
* :func:`run_ssl_smoke_from_event_log`

Nothing in this module implements market labels, supervised multi-task
fine-tuning, calibration, execution simulation, backtesting, mixed
precision, checkpointing or any market-performance claim. The smoke
runner emits an explicit warning that the produced losses are a plumbing
check and not market evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chronoslob.models.ssl import (
    DEFAULT_MASKED_FIELDS,
    DEFAULT_NEXT_FIELDS,
    MarketSSLTransformer,
    MaskingConfig,
    SSLTransformerConfig,
    create_ssl_transformer,
)
from chronoslob.models.tokenisation import (
    TokenisationConfig,
    tokenise_event_log,
)
from chronoslob.models.transformer import MarketTransformerConfig
from chronoslob.training.ssl_datasets import (
    DEFAULT_IGNORE_INDEX,
    MaskedTokenBatch,
    MaskingPolicy,
    SSLTokenSequenceDataset,
    collate_ssl_token_windows,
)
from chronoslob.training.token_datasets import (
    TokenSequenceDataset,
    TokenWindowConfig,
    build_token_window_indices,
)
from chronoslob.training.torch_training import set_torch_deterministic

try:  # pragma: no cover - exercised when torch is unavailable
    import torch
    from torch.utils.data import DataLoader

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]
    _TORCH_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    import torch as _torch_typing  # noqa: F401

__all__ = [
    "SSLEpochResult",
    "SSLTrainingConfig",
    "evaluate_ssl",
    "fit_ssl_model",
    "run_ssl_smoke_from_event_log",
    "train_ssl_one_epoch",
]

_SYNTHETIC_SSL_WARNING = (
    "Synthetic SSL plumbing only; losses do not measure market signal, alpha "
    "or benchmark performance."
)


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the SSL experiment runner. Install the "
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
                "ChronosLOB SSL training defaults to CPU"
            )
        return torch_module.device(normalised)
    raise ValueError(
        f"unsupported device {device!r}; ChronosLOB SSL training currently "
        "supports 'cpu' or 'cuda'-prefixed devices"
    )


@dataclass(frozen=True)
class SSLTrainingConfig:
    """Configuration for the self-supervised pretraining loop."""

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
                self.gradient_clip_norm,
                name="gradient_clip_norm",
            )
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        _validate_non_negative_int(self.seed, name="seed")


@dataclass(frozen=True)
class SSLEpochResult:
    """Per-epoch summary returned by :func:`fit_ssl_model`."""

    epoch: int
    train_loss: float
    train_loss_components: dict[str, float] = field(default_factory=dict)
    validation_loss: float | None = None
    validation_loss_components: dict[str, float] = field(default_factory=dict)


def _move_inputs_to_device(
    inputs: Mapping[str, Any],
    device_obj: Any,
) -> dict[str, Any]:
    torch_module = _require_torch()
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if torch_module.is_tensor(value):
            moved[key] = value.to(device_obj)
        else:
            moved[key] = value
    return moved


def _move_label_dict_to_device(
    labels: Mapping[str, Any],
    device_obj: Any,
) -> dict[str, Any]:
    torch_module = _require_torch()
    return {
        key: value.to(device_obj) if torch_module.is_tensor(value) else value
        for key, value in labels.items()
    }


def _ensure_masked_token_batch(batch: Any) -> MaskedTokenBatch:
    if not isinstance(batch, MaskedTokenBatch):
        raise TypeError(
            "SSL training loop expects MaskedTokenBatch payloads from "
            "collate_ssl_token_windows; got "
            f"{type(batch).__name__}"
        )
    return batch


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


def train_ssl_one_epoch(
    model: MarketSSLTransformer,
    dataloader: Iterable[Any],
    optimizer: Any,
    *,
    device: str = "cpu",
    gradient_clip_norm: float | None = None,
) -> tuple[float, dict[str, float]]:
    """Run one SSL training epoch.

    Returns the sample-weighted mean combined loss and the sample-weighted
    mean per-objective losses.
    """
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
        batch = _ensure_masked_token_batch(raw_batch)
        inputs = _move_inputs_to_device(batch.inputs, device_obj)
        masked_labels = (
            _move_label_dict_to_device(batch.masked_labels, device_obj)
            if batch.masked_labels
            else None
        )
        next_labels = (
            _move_label_dict_to_device(batch.next_labels, device_obj)
            if batch.next_labels
            else None
        )
        batch_size = int(inputs["attention_mask"].shape[0])
        if batch_size == 0:
            continue

        optimizer.zero_grad()
        output = model(
            inputs,
            masked_labels=masked_labels,
            next_labels=next_labels,
        )
        if output.loss is None:
            raise RuntimeError(
                "SSL forward did not produce a loss; ensure masked_labels or "
                "next_labels are provided"
            )
        loss = output.loss
        if not torch_module.isfinite(loss).item():
            raise RuntimeError("SSL training loss became non-finite")
        loss.backward()
        _assert_finite_gradients(model)
        if gradient_clip_norm is not None:
            if gradient_clip_norm <= 0:
                raise ValueError(
                    "gradient_clip_norm must be positive when provided"
                )
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
        raise ValueError("train_ssl_one_epoch received an empty dataloader")
    averaged_components = _average_components(component_totals, component_counts)
    return total_loss / total_samples, averaged_components


def evaluate_ssl(
    model: MarketSSLTransformer,
    dataloader: Iterable[Any],
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate an SSL model without updating its parameters."""
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
            batch = _ensure_masked_token_batch(raw_batch)
            inputs = _move_inputs_to_device(batch.inputs, device_obj)
            masked_labels = (
                _move_label_dict_to_device(batch.masked_labels, device_obj)
                if batch.masked_labels
                else None
            )
            next_labels = (
                _move_label_dict_to_device(batch.next_labels, device_obj)
                if batch.next_labels
                else None
            )
            batch_size = int(inputs["attention_mask"].shape[0])
            if batch_size == 0:
                continue
            output = model(
                inputs,
                masked_labels=masked_labels,
                next_labels=next_labels,
            )
            if output.loss is None:
                raise RuntimeError(
                    "SSL evaluation forward did not produce a loss"
                )
            if not torch_module.isfinite(output.loss).item():
                raise RuntimeError("SSL evaluation loss became non-finite")
            total_loss += float(output.loss.item()) * batch_size
            total_samples += batch_size
            _accumulate_components(
                component_totals,
                component_counts,
                output.loss_components,
                batch_size,
            )

    if total_samples == 0:
        raise ValueError("evaluate_ssl received an empty dataloader")
    return {
        "loss": total_loss / total_samples,
        "loss_components": _average_components(component_totals, component_counts),
        "n_samples": int(total_samples),
    }


def fit_ssl_model(
    model: MarketSSLTransformer,
    train_loader: Iterable[Any],
    validation_loader: Iterable[Any] | None = None,
    config: SSLTrainingConfig | None = None,
) -> list[SSLEpochResult]:
    """Train an SSL model and return per-epoch results."""
    _require_torch()
    if model is None:
        raise ValueError("model must not be None")
    training_config = config if config is not None else SSLTrainingConfig()
    if not isinstance(training_config, SSLTrainingConfig):
        raise TypeError("config must be a SSLTrainingConfig instance")
    device_obj = _resolve_device(training_config.device)
    set_torch_deterministic(training_config.seed)

    model.to(device_obj)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    history: list[SSLEpochResult] = []
    for epoch in range(1, training_config.epochs + 1):
        train_loss, train_components = train_ssl_one_epoch(
            model,
            train_loader,
            optimizer,
            device=training_config.device,
            gradient_clip_norm=training_config.gradient_clip_norm,
        )
        validation_loss: float | None = None
        validation_components: dict[str, float] = {}
        if validation_loader is not None:
            evaluation = evaluate_ssl(
                model,
                validation_loader,
                device=training_config.device,
            )
            validation_loss = float(evaluation["loss"])
            validation_components = dict(evaluation["loss_components"])
        history.append(
            SSLEpochResult(
                epoch=epoch,
                train_loss=train_loss,
                train_loss_components=train_components,
                validation_loss=validation_loss,
                validation_loss_components=validation_components,
            )
        )
    return history


def _build_default_ssl_config(
    vocab_sizes: Mapping[str, int],
    *,
    window_length: int,
    masked_fields: tuple[str, ...] = DEFAULT_MASKED_FIELDS,
    next_fields: tuple[str, ...] = DEFAULT_NEXT_FIELDS,
    field_embedding_dim: int = 8,
    model_dim: int = 32,
    num_heads: int = 4,
    num_layers: int = 2,
    feedforward_dim: int = 64,
    dropout: float = 0.1,
    mask_probability: float = 0.15,
) -> SSLTransformerConfig:
    transformer_config = MarketTransformerConfig(
        vocab_sizes=dict(vocab_sizes),
        field_embedding_dim=field_embedding_dim,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
        max_sequence_length=window_length,
        num_classes=2,
        pooling="mean",
        activation="gelu",
        use_layer_norm=True,
    )
    masking_config = MaskingConfig(mask_probability=mask_probability)
    return SSLTransformerConfig(
        transformer=transformer_config,
        masked_fields=masked_fields,
        next_fields=next_fields,
        enable_masked_field_loss=True,
        enable_next_field_loss=True,
        enable_contrastive_loss=False,
        masking=masking_config,
    )


def _ssl_dataloader(
    ssl_dataset: SSLTokenSequenceDataset,
    *,
    batch_size: int,
) -> Any:
    _require_torch()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return DataLoader(
        ssl_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_ssl_token_windows,
    )


def run_ssl_smoke_from_event_log(
    path: str | Path,
    *,
    symbol: str | None = None,
    window_length: int = 4,
    batch_size: int = 4,
    epochs: int = 1,
    seed: int = 42,
    max_levels_per_side: int = 2,
    learning_rate: float = 1e-3,
    mask_probability: float = 0.15,
    masked_fields: tuple[str, ...] = DEFAULT_MASKED_FIELDS,
    next_fields: tuple[str, ...] = DEFAULT_NEXT_FIELDS,
    ssl_config: SSLTransformerConfig | None = None,
) -> dict[str, Any]:
    """Run a tiny CPU SSL smoke experiment from a local canonical event log.

    The function tokenises the event log with Phase-11 defaults, builds
    fixed-length token windows, applies deterministic field masking, builds
    next-field targets and trains a small SSL transformer for ``epochs``
    epochs on CPU. The returned payload includes loss components and an
    explicit ``notes`` field labelling the run as synthetic SSL plumbing
    only.
    """
    _require_torch()
    file_path = Path(path)
    tokenisation_config = TokenisationConfig(
        max_levels_per_side=max_levels_per_side,
    )
    sequence = tokenise_event_log(file_path, tokenisation_config, symbol=symbol)
    window_config = TokenWindowConfig(
        window_length=window_length,
        stride=1,
        drop_incomplete=False,
        padding_side="left",
    )
    window_indices = build_token_window_indices(sequence, window_config)
    if not window_indices:
        raise ValueError(
            "SSL smoke produced zero windows; the event log may be empty or "
            "every record may have been filtered out by --symbol"
        )

    base_dataset = TokenSequenceDataset(
        sequence,
        window_config,
        window_indices=window_indices,
    )

    if ssl_config is None:
        resolved_ssl_config = _build_default_ssl_config(
            sequence.field_sizes,
            window_length=window_length,
            masked_fields=tuple(masked_fields),
            next_fields=tuple(next_fields),
            mask_probability=mask_probability,
        )
    else:
        if not isinstance(ssl_config, SSLTransformerConfig):
            raise TypeError("ssl_config must be a SSLTransformerConfig instance")
        if ssl_config.transformer.max_sequence_length < window_length:
            raise ValueError(
                "ssl_config.transformer.max_sequence_length is smaller than "
                f"window_length: {ssl_config.transformer.max_sequence_length} < "
                f"{window_length}"
            )
        resolved_ssl_config = ssl_config

    set_torch_deterministic(seed)
    model = create_ssl_transformer(resolved_ssl_config)
    policy = MaskingPolicy(
        fields=resolved_ssl_config.masked_fields,
        vocab_sizes={
            field_name: int(resolved_ssl_config.transformer.vocab_sizes[field_name])
            for field_name in resolved_ssl_config.masked_fields
        },
        config=resolved_ssl_config.masking,
        ignore_index=int(resolved_ssl_config.ignore_index),
    )
    ssl_dataset = SSLTokenSequenceDataset(
        base_dataset,
        policy,
        resolved_ssl_config.next_fields,
        base_seed=seed,
        enable_masking=resolved_ssl_config.enable_masked_field_loss,
        enable_next_targets=resolved_ssl_config.enable_next_field_loss,
    )
    train_loader = _ssl_dataloader(ssl_dataset, batch_size=batch_size)

    training_config = SSLTrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        device="cpu",
    )
    history = fit_ssl_model(model, train_loader, None, training_config)

    eval_loader = _ssl_dataloader(ssl_dataset, batch_size=batch_size)
    evaluation = evaluate_ssl(model, eval_loader, device="cpu")

    final_history = history[-1] if history else None
    return {
        "path": str(file_path),
        "symbol_filter": symbol,
        "input_record_count": int(sequence.input_record_count),
        "tokenised_record_count": len(sequence.records),
        "window_length": int(window_length),
        "window_count": len(base_dataset),
        "enabled_objectives": list(resolved_ssl_config.enabled_objectives()),
        "masked_fields": list(resolved_ssl_config.masked_fields),
        "next_fields": list(resolved_ssl_config.next_fields),
        "vocab_sizes": dict(sequence.field_sizes),
        "model_parameter_count": int(model.n_parameters()),
        "training_history": [
            {
                "epoch": int(item.epoch),
                "train_loss": float(item.train_loss),
                "train_loss_components": dict(item.train_loss_components),
            }
            for item in history
        ],
        "final_train_loss": (
            float(final_history.train_loss) if final_history is not None else None
        ),
        "final_train_loss_components": (
            dict(final_history.train_loss_components)
            if final_history is not None
            else {}
        ),
        "synthetic_smoke_metrics": {
            "loss": float(evaluation["loss"]),
            "loss_components": dict(evaluation["loss_components"]),
            "n_samples": int(evaluation["n_samples"]),
        },
        "ignore_index": int(resolved_ssl_config.ignore_index),
        "mask_probability": float(resolved_ssl_config.masking.mask_probability),
        "notes": _SYNTHETIC_SSL_WARNING,
        "write_outputs": False,
    }


# Reference imports that document the contract this runner relies on:
# ``DEFAULT_IGNORE_INDEX`` is exported by the dataset module so callers can
# pick the same sentinel as the SSL loss helpers.
_RESERVED_IGNORE_INDEX = DEFAULT_IGNORE_INDEX
