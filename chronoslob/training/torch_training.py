"""Generic supervised classification training utilities for PyTorch models.

These helpers are deliberately small. They cover deterministic seeding,
a single-epoch training step, a no-gradient evaluation step and a tiny
``fit`` loop that returns a per-epoch history. Nothing in this module
implements:

- transformer architectures;
- self-supervised pretraining objectives;
- model checkpointing;
- distributed or multi-GPU training;
- execution backtests or PnL.

PyTorch is treated as an optional dependency and a clear ``ImportError``
is raised if the ``[torch]`` extra is not installed. Tests should guard
imports with ``pytest.importorskip("torch")``.
"""

from __future__ import annotations

import contextlib
import os
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from chronoslob.training.metrics import (
    compute_classification_metrics,
    confusion_matrix_as_dict,
)

try:  # pragma: no cover - exercised when torch is unavailable
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    import torch as _torch_typing  # noqa: F401

__all__ = [
    "TorchEpochResult",
    "TorchTrainingConfig",
    "evaluate_torch_classifier",
    "fit_torch_classifier",
    "set_torch_deterministic",
    "train_one_epoch",
]


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for torch training utilities. Install the "
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
                "ChronosLOB training utilities default to CPU"
            )
        return torch_module.device(normalised)
    raise ValueError(
        f"unsupported device {device!r}; ChronosLOB torch utilities currently "
        "support 'cpu' or 'cuda'-prefixed devices"
    )


@dataclass(frozen=True)
class TorchTrainingConfig:
    """Configuration for the supervised torch classification fit loop."""

    epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = 1.0
    device: str = "cpu"
    seed: int = 42
    log_every: int = 0

    def __post_init__(self) -> None:
        """Validate epoch counts, optimiser hyperparameters and device."""
        _validate_positive_int(self.epochs, name="epochs")
        _validate_positive_float(self.learning_rate, name="learning_rate")
        _validate_non_negative_float(self.weight_decay, name="weight_decay")
        if self.gradient_clip_norm is not None:
            _validate_positive_float(self.gradient_clip_norm, name="gradient_clip_norm")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        _validate_non_negative_int(self.seed, name="seed")
        _validate_non_negative_int(self.log_every, name="log_every")


@dataclass(frozen=True)
class TorchEpochResult:
    """Per-epoch summary returned by :func:`fit_torch_classifier`."""

    epoch: int
    train_loss: float
    validation_loss: float | None = None
    validation_metrics: dict[str, object] | None = field(default=None)


def set_torch_deterministic(seed: int) -> None:
    """Seed Python, NumPy and PyTorch and set deterministic flags.

    The function avoids assuming CUDA is available. When CUDA is present
    its seeds are set as well, but no CUDA-only operations are required to
    succeed. Determinism is best-effort: hardware, library versions and
    parallelism can still introduce small variations across machines.
    """
    torch_module = _require_torch()
    _validate_non_negative_int(seed, name="seed")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    if hasattr(torch_module, "use_deterministic_algorithms"):
        with contextlib.suppress(RuntimeError, TypeError):
            torch_module.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.benchmark = False
        torch_module.backends.cudnn.deterministic = True


def _iterate_batches(dataloader: Iterable[Any]) -> Iterable[dict[str, Any]]:
    for batch in dataloader:
        if not isinstance(batch, dict):
            raise TypeError(
                "torch training utilities expect dict batches with 'x' and 'y' "
                f"keys; got {type(batch).__name__}"
            )
        if "x" not in batch or "y" not in batch:
            raise KeyError("batch is missing required keys 'x' and 'y'")
        yield batch


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


def train_one_epoch(
    model: Any,
    dataloader: Iterable[Any],
    optimizer: Any,
    loss_fn: Any,
    *,
    device: str = "cpu",
    gradient_clip_norm: float | None = None,
) -> float:
    """Run one training epoch and return the sample-weighted mean loss."""
    torch_module = _require_torch()
    if model is None:
        raise ValueError("model must not be None")
    if optimizer is None:
        raise ValueError("optimizer must not be None")
    if loss_fn is None:
        raise ValueError("loss_fn must not be None")
    device_obj = _resolve_device(device)
    model.train()
    model.to(device_obj)

    total_loss = 0.0
    total_samples = 0
    for batch in _iterate_batches(dataloader):
        x = batch["x"].to(device_obj)
        y = batch["y"].to(device_obj)
        if x.shape[0] == 0:
            continue
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        if not torch_module.isfinite(loss).item():
            raise RuntimeError("training loss became non-finite")
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
        batch_size = int(x.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("train_one_epoch received an empty dataloader")
    return total_loss / total_samples


def _softmax_probabilities(logits: Any) -> np.ndarray:
    torch_module = _require_torch()
    probabilities = torch_module.softmax(logits, dim=-1)
    numpy_probabilities = np.asarray(
        probabilities.detach().cpu().numpy(),
        dtype=float,
    )
    # Renormalise rows so float32 round-off does not trigger sklearn's
    # "y_prob values do not sum to one" warning during log_loss.
    row_sums = numpy_probabilities.sum(axis=-1, keepdims=True)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    return numpy_probabilities / row_sums


def evaluate_torch_classifier(
    model: Any,
    dataloader: Iterable[Any],
    *,
    device: str = "cpu",
    labels: Sequence[object] | None = None,
) -> dict[str, Any]:
    """Evaluate a fitted classifier and return metrics plus confusion matrix.

    The model is put into ``eval`` mode and gradients are disabled. The
    function returns a dictionary compatible with the rest of
    ``chronoslob.training`` containing ``metrics``, ``confusion_matrix``,
    ``loss`` and the raw collected ``logits``, ``probabilities``,
    ``predictions`` and ``targets``.
    """
    torch_module = _require_torch()
    if model is None:
        raise ValueError("model must not be None")
    device_obj = _resolve_device(device)
    model.eval()
    model.to(device_obj)

    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    all_logits: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    all_predictions: list[int] = []
    all_targets: list[int] = []
    total_loss = 0.0
    total_samples = 0

    with torch_module.no_grad():
        for batch in _iterate_batches(dataloader):
            x = batch["x"].to(device_obj)
            y = batch["y"].to(device_obj)
            if x.shape[0] == 0:
                continue
            logits = model(x)
            loss = loss_fn(logits, y)
            if not torch_module.isfinite(loss).item():
                raise RuntimeError("evaluation loss became non-finite")
            probabilities = _softmax_probabilities(logits)
            predictions = np.asarray(
                logits.argmax(dim=-1).detach().cpu().numpy(),
                dtype=np.int64,
            )
            targets = np.asarray(y.detach().cpu().numpy(), dtype=np.int64)
            all_logits.append(
                np.asarray(logits.detach().cpu().numpy(), dtype=float)
            )
            all_probabilities.append(probabilities)
            all_predictions.extend(int(value) for value in predictions.tolist())
            all_targets.extend(int(value) for value in targets.tolist())
            total_loss += float(loss.item())
            total_samples += int(x.shape[0])

    if total_samples == 0:
        raise ValueError("evaluate_torch_classifier received an empty dataloader")

    logits_array = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, 0))
    probabilities_array = (
        np.concatenate(all_probabilities, axis=0)
        if all_probabilities
        else np.zeros((0, 0))
    )
    metric_labels: Sequence[object] | None = (
        sorted({*all_targets, *all_predictions})
        if labels is None
        else list(labels)
    )

    metrics = compute_classification_metrics(
        all_targets,
        all_predictions,
        y_proba=probabilities_array if probabilities_array.size > 0 else None,
        labels=metric_labels,
    )
    confusion = confusion_matrix_as_dict(
        all_targets,
        all_predictions,
        labels=metric_labels,
    )
    return {
        "loss": total_loss / total_samples,
        "metrics": metrics.to_dict(),
        "confusion_matrix": confusion,
        "logits": logits_array,
        "probabilities": probabilities_array,
        "predictions": all_predictions,
        "targets": all_targets,
    }


def fit_torch_classifier(
    model: Any,
    train_loader: Iterable[Any],
    validation_loader: Iterable[Any] | None = None,
    config: TorchTrainingConfig | None = None,
) -> list[TorchEpochResult]:
    """Train a supervised classifier and return per-epoch results.

    The function uses Adam with the configured learning rate and weight
    decay, ``torch.nn.CrossEntropyLoss`` and a deterministic seed. It does
    not write checkpoints, log to disk or produce fake metrics. The
    returned list has one :class:`TorchEpochResult` per epoch.
    """
    _require_torch()
    if model is None:
        raise ValueError("model must not be None")
    training_config = config if config is not None else TorchTrainingConfig()
    if not isinstance(training_config, TorchTrainingConfig):
        raise TypeError("config must be a TorchTrainingConfig instance")
    device_obj = _resolve_device(training_config.device)

    set_torch_deterministic(training_config.seed)

    model.to(device_obj)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    history: list[TorchEpochResult] = []
    for epoch in range(1, training_config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device=training_config.device,
            gradient_clip_norm=training_config.gradient_clip_norm,
        )
        validation_loss: float | None = None
        validation_metrics: dict[str, object] | None = None
        if validation_loader is not None:
            validation_result = evaluate_torch_classifier(
                model,
                validation_loader,
                device=training_config.device,
            )
            validation_loss = float(validation_result["loss"])
            validation_metrics = {
                "metrics": validation_result["metrics"],
                "confusion_matrix": validation_result["confusion_matrix"],
            }
        history.append(
            TorchEpochResult(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                validation_metrics=validation_metrics,
            )
        )
    return history
