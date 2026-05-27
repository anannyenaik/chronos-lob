"""Supervised transformer encoder training/evaluation plumbing.

This module wires the Phase-11 field-wise tokenisation and token-window
dataset to the Phase-12 transformer encoder defined in
:mod:`chronoslob.models.transformer`. It provides a tiny, deterministic
training/evaluation loop and a synthetic smoke command for CLI plumbing.

The implementation deliberately stays minimal: it covers config, one-epoch
training, evaluation, a small fit loop and an end-to-end smoke runner from a
local canonical event log. Nothing here implements masked event modelling,
next-event prediction, contrastive learning, calibration, execution
simulation, backtesting or any form of self-supervised pretraining. Smoke
labels are synthetic and must never be reported as market signal or
benchmark performance.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from chronoslob.models.tokenisation import (
    TokenisationConfig,
    tokenise_event_log,
)
from chronoslob.models.transformer import (
    MarketTransformerConfig,
    MarketTransformerEncoder,
    MarketTransformerOutput,
    create_market_transformer,
)
from chronoslob.training.metrics import (
    compute_classification_metrics,
    confusion_matrix_as_dict,
)
from chronoslob.training.token_batching import collate_token_windows
from chronoslob.training.token_datasets import (
    TOKEN_WINDOW_FIELD_NAMES,
    TokenSequenceDataset,
    TokenWindowConfig,
    build_token_window_indices,
)
from chronoslob.training.torch_training import set_torch_deterministic

try:  # pragma: no cover - exercised when torch is unavailable
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset as _TorchDataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]
    _TorchDataset = object  # type: ignore[assignment,misc]
    _TORCH_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    import torch as _torch_typing  # noqa: F401

__all__ = [
    "TransformerEpochResult",
    "TransformerTrainingConfig",
    "evaluate_transformer_classifier",
    "fit_transformer_classifier",
    "run_transformer_smoke_from_event_log",
    "train_transformer_one_epoch",
]

_SYNTHETIC_SMOKE_WARNING = (
    "Synthetic smoke labels only; no market signal, alpha or benchmark is implied."
)


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the transformer experiment runner. "
            "Install the 'torch' optional dependency: pip install -e '.[torch]'"
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
                "ChronosLOB transformer training defaults to CPU"
            )
        return torch_module.device(normalised)
    raise ValueError(
        f"unsupported device {device!r}; ChronosLOB transformer training "
        "currently supports 'cpu' or 'cuda'-prefixed devices"
    )


@dataclass(frozen=True)
class TransformerTrainingConfig:
    """Configuration for the supervised transformer fit loop."""

    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = 1.0
    device: str = "cpu"
    seed: int = 42
    early_stopping_patience: int | None = None
    early_stopping_metric: str = "validation_loss"
    checkpoint_path: Path | None = None
    save_best_checkpoint: bool = True

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
        if self.early_stopping_patience is not None:
            _validate_non_negative_int(
                self.early_stopping_patience,
                name="early_stopping_patience",
            )
        if self.early_stopping_metric not in {
            "validation_loss",
            "validation_macro_f1",
        }:
            raise ValueError(
                "early_stopping_metric must be 'validation_loss' or "
                "'validation_macro_f1'"
            )


@dataclass(frozen=True)
class TransformerEpochResult:
    """Per-epoch summary returned by :func:`fit_transformer_classifier`."""

    epoch: int
    train_loss: float
    validation_loss: float | None = None
    validation_metrics: dict[str, object] | None = field(default=None)
    monitored_value: float | None = None
    is_best: bool = False
    early_stop: bool = False


def _move_batch_to_device(
    batch: Mapping[str, Any],
    device_obj: Any,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for field_name in TOKEN_WINDOW_FIELD_NAMES:
        if field_name not in batch:
            raise KeyError(f"batch is missing required token field {field_name!r}")
        moved[field_name] = batch[field_name].to(device_obj)
    if "attention_mask" not in batch:
        raise KeyError("batch is missing required 'attention_mask'")
    moved["attention_mask"] = batch["attention_mask"].to(device_obj)
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


def train_transformer_one_epoch(
    model: MarketTransformerEncoder,
    dataloader: Iterable[Any],
    optimizer: Any,
    loss_fn: Any,
    *,
    device: str = "cpu",
    gradient_clip_norm: float | None = None,
) -> float:
    """Run one supervised training epoch and return the mean cross-entropy loss."""
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
    for batch in dataloader:
        if not isinstance(batch, Mapping):
            raise TypeError(
                "transformer training expects mapping batches with token "
                f"fields and 'y'; got {type(batch).__name__}"
            )
        if "y" not in batch:
            raise KeyError("batch is missing required key 'y'")
        inputs = _move_batch_to_device(batch, device_obj)
        targets = batch["y"].to(device_obj)
        batch_size = int(targets.shape[0])
        if batch_size == 0:
            continue
        optimizer.zero_grad()
        output = model(inputs)
        logits = output.logits
        loss = loss_fn(logits, targets)
        if not torch_module.isfinite(loss).item():
            raise RuntimeError("transformer training loss became non-finite")
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

    if total_samples == 0:
        raise ValueError("train_transformer_one_epoch received an empty dataloader")
    return total_loss / total_samples


def _softmax_probabilities(logits: Any) -> np.ndarray:
    torch_module = _require_torch()
    probabilities = torch_module.softmax(logits, dim=-1)
    numpy_probabilities = np.asarray(
        probabilities.detach().cpu().numpy(),
        dtype=float,
    )
    row_sums = numpy_probabilities.sum(axis=-1, keepdims=True)
    row_sums = np.where(row_sums == 0.0, 1.0, row_sums)
    return numpy_probabilities / row_sums


def evaluate_transformer_classifier(
    model: MarketTransformerEncoder,
    dataloader: Iterable[Any],
    *,
    device: str = "cpu",
    labels: Sequence[object] | None = None,
) -> dict[str, Any]:
    """Evaluate a fitted transformer encoder on a labelled dataloader."""
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
        for batch in dataloader:
            if not isinstance(batch, Mapping):
                raise TypeError(
                    "transformer evaluation expects mapping batches with "
                    f"token fields and 'y'; got {type(batch).__name__}"
                )
            if "y" not in batch:
                raise KeyError("batch is missing required key 'y'")
            inputs = _move_batch_to_device(batch, device_obj)
            targets = batch["y"].to(device_obj)
            batch_size = int(targets.shape[0])
            if batch_size == 0:
                continue
            output: MarketTransformerOutput = model(inputs)
            logits = output.logits
            loss = loss_fn(logits, targets)
            if not torch_module.isfinite(loss).item():
                raise RuntimeError(
                    "transformer evaluation loss became non-finite"
                )
            probabilities = _softmax_probabilities(logits)
            predictions = np.asarray(
                logits.argmax(dim=-1).detach().cpu().numpy(),
                dtype=np.int64,
            )
            target_array = np.asarray(targets.detach().cpu().numpy(), dtype=np.int64)
            all_logits.append(
                np.asarray(logits.detach().cpu().numpy(), dtype=float)
            )
            all_probabilities.append(probabilities)
            all_predictions.extend(int(value) for value in predictions.tolist())
            all_targets.extend(int(value) for value in target_array.tolist())
            total_loss += float(loss.item())
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "evaluate_transformer_classifier received an empty dataloader"
        )

    logits_array = (
        np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, 0))
    )
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
        "notes": _SYNTHETIC_SMOKE_WARNING,
    }


def _monitored_metric_value(
    result: TransformerEpochResult,
    *,
    metric: str,
) -> tuple[float | None, bool]:
    """Return ``(value, higher_is_better)`` for early stopping."""
    if metric == "validation_loss":
        return result.validation_loss, False
    if metric == "validation_macro_f1":
        metrics = result.validation_metrics
        if not isinstance(metrics, Mapping):
            return None, True
        nested = metrics.get("metrics")
        if not isinstance(nested, Mapping):
            return None, True
        value = nested.get("macro_f1")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, True
        return float(value), True
    raise ValueError(
        "early_stopping_metric must be 'validation_loss' or "
        "'validation_macro_f1'"
    )


def _is_improvement(
    value: float,
    best_value: float | None,
    *,
    higher_is_better: bool,
) -> bool:
    if best_value is None:
        return True
    if higher_is_better:
        return value > best_value
    return value < best_value


def _clone_model_state(model: Any) -> dict[str, Any]:
    return {
        str(name): tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _write_checkpoint(
    *,
    path: Path,
    model: Any,
    epoch: int,
    metric: str,
    value: float,
) -> None:
    torch_module = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": int(epoch),
            "metric": metric,
            "metric_value": float(value),
        },
        path,
    )


def fit_transformer_classifier(
    model: MarketTransformerEncoder,
    train_loader: Iterable[Any],
    validation_loader: Iterable[Any] | None = None,
    config: TransformerTrainingConfig | None = None,
) -> list[TransformerEpochResult]:
    """Train a supervised transformer classifier and return per-epoch results."""
    _require_torch()
    if model is None:
        raise ValueError("model must not be None")
    training_config = config if config is not None else TransformerTrainingConfig()
    if not isinstance(training_config, TransformerTrainingConfig):
        raise TypeError("config must be a TransformerTrainingConfig instance")
    device_obj = _resolve_device(training_config.device)
    set_torch_deterministic(training_config.seed)

    model.to(device_obj)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    history: list[TransformerEpochResult] = []
    best_value: float | None = None
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    for epoch in range(1, training_config.epochs + 1):
        train_loss = train_transformer_one_epoch(
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
            evaluation = evaluate_transformer_classifier(
                model,
                validation_loader,
                device=training_config.device,
            )
            validation_loss = float(evaluation["loss"])
            validation_metrics = {
                "metrics": evaluation["metrics"],
                "confusion_matrix": evaluation["confusion_matrix"],
            }
        epoch_result = TransformerEpochResult(
            epoch=epoch,
            train_loss=train_loss,
            validation_loss=validation_loss,
            validation_metrics=validation_metrics,
        )
        monitored_value, higher_is_better = _monitored_metric_value(
            epoch_result,
            metric=training_config.early_stopping_metric,
        )
        is_best = False
        should_stop = False
        if monitored_value is not None:
            if _is_improvement(
                monitored_value,
                best_value,
                higher_is_better=higher_is_better,
            ):
                best_value = monitored_value
                best_state = _clone_model_state(model)
                epochs_without_improvement = 0
                is_best = True
                if (
                    training_config.checkpoint_path is not None
                    and training_config.save_best_checkpoint
                ):
                    _write_checkpoint(
                        path=training_config.checkpoint_path,
                        model=model,
                        epoch=epoch,
                        metric=training_config.early_stopping_metric,
                        value=monitored_value,
                    )
            else:
                epochs_without_improvement += 1
                should_stop = (
                    training_config.early_stopping_patience is not None
                    and epochs_without_improvement
                    >= training_config.early_stopping_patience
                )
        history.append(
            TransformerEpochResult(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                validation_metrics=validation_metrics,
                monitored_value=monitored_value,
                is_best=is_best,
                early_stop=should_stop,
            )
        )
        if should_stop:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def _synthetic_smoke_label(
    sample_index: int,
    last_side_id: int,
    num_classes: int,
) -> int:
    """Return a deterministic synthetic label for the smoke command.

    The label combines the sample index with the final token's ``side`` ID so
    that adjacent windows do not always produce the same class. This is a
    plumbing label only; it carries no market information and must never be
    interpreted as a forecast target.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    return (sample_index + int(last_side_id)) % int(num_classes)


_SAMPLE_INDEX_KEY = "_smoke_sample_index"


def _collate_with_synthetic_labels(
    samples: Sequence[Mapping[str, Any]],
    *,
    num_classes: int,
) -> dict[str, Any]:
    torch_module = _require_torch()
    base_samples = [
        {key: value for key, value in sample.items() if key != _SAMPLE_INDEX_KEY}
        for sample in samples
    ]
    base = collate_token_windows(base_samples)
    side_tokens = base["side"]
    attention_mask = base["attention_mask"]
    real_token_counts = attention_mask.to(torch_module.long).sum(dim=1)
    last_real_indices = (real_token_counts - 1).clamp(min=0)
    row_idx = torch_module.arange(side_tokens.shape[0])
    last_side_ids = side_tokens[row_idx, last_real_indices]
    targets = torch_module.tensor(
        [
            _synthetic_smoke_label(
                int(samples[position][_SAMPLE_INDEX_KEY]),
                int(last_side_ids[position].item()),
                num_classes,
            )
            for position in range(side_tokens.shape[0])
        ],
        dtype=torch_module.long,
    )
    base["y"] = targets
    return base


class _SmokeLabelledDataset(_TorchDataset):
    """Tiny in-memory wrapper that tags each window with its sample index."""

    def __init__(self, dataset: TokenSequenceDataset) -> None:
        n_samples = len(dataset)
        if n_samples == 0:
            raise ValueError(
                "token-window dataset is empty; cannot build dataloader"
            )
        self._items: list[dict[str, Any]] = []
        for index in range(n_samples):
            window = dict(dataset[index])
            window[_SAMPLE_INDEX_KEY] = int(index)
            self._items.append(window)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, item: int) -> dict[str, Any]:
        return self._items[item]


def _build_smoke_dataloader(
    dataset: TokenSequenceDataset,
    *,
    batch_size: int,
    num_classes: int,
) -> Any:
    _require_torch()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    labelled = _SmokeLabelledDataset(dataset)

    def _collate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return _collate_with_synthetic_labels(samples, num_classes=num_classes)

    return DataLoader(
        labelled,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_collate,
    )


def _default_smoke_model_config(
    vocab_sizes: Mapping[str, int],
    *,
    num_classes: int,
    max_sequence_length: int,
    field_embedding_dim: int = 8,
    model_dim: int = 32,
    num_heads: int = 4,
    num_layers: int = 2,
    feedforward_dim: int = 64,
    dropout: float = 0.1,
) -> MarketTransformerConfig:
    return MarketTransformerConfig(
        vocab_sizes=dict(vocab_sizes),
        field_embedding_dim=field_embedding_dim,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
        max_sequence_length=max_sequence_length,
        num_classes=num_classes,
        pooling="mean",
        activation="gelu",
        use_layer_norm=True,
    )


def run_transformer_smoke_from_event_log(
    path: str | Path,
    *,
    symbol: str | None = None,
    window_length: int = 4,
    batch_size: int = 4,
    epochs: int = 1,
    seed: int = 42,
    num_classes: int = 3,
    max_levels_per_side: int = 2,
    learning_rate: float = 1e-3,
    model_config: MarketTransformerConfig | None = None,
) -> dict[str, Any]:
    """Run a tiny supervised transformer smoke experiment from an event log.

    The function tokenises a local canonical event log using Phase-11
    defaults, builds fixed-length token windows, generates deterministic
    synthetic labels and trains a small transformer encoder for ``epochs``
    epochs on CPU. The returned payload includes a ``notes`` field that
    states the labels are synthetic and a ``label_source`` field that names
    the synthetic label rule. Nothing here is a benchmark or alpha result.
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
            "transformer smoke produced zero windows; the event log may be "
            "empty or every record may have been filtered out by --symbol"
        )

    dataset = TokenSequenceDataset(
        sequence,
        window_config,
        window_indices=window_indices,
    )

    if model_config is None:
        resolved_model_config = _default_smoke_model_config(
            sequence.field_sizes,
            num_classes=num_classes,
            max_sequence_length=window_length,
        )
    else:
        if not isinstance(model_config, MarketTransformerConfig):
            raise TypeError("model_config must be a MarketTransformerConfig instance")
        # ``num_classes`` in the supplied model config takes precedence over
        # the ``num_classes`` keyword argument so the model and label space
        # stay aligned with the explicit configuration.
        resolved_num_classes = int(model_config.num_classes)
        if resolved_num_classes != num_classes:
            num_classes = resolved_num_classes
        if model_config.max_sequence_length < window_length:
            raise ValueError(
                "model_config.max_sequence_length is smaller than "
                f"window_length: {model_config.max_sequence_length} < {window_length}"
            )
        resolved_model_config = model_config

    training_config = TransformerTrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        device="cpu",
    )
    set_torch_deterministic(seed)
    model = create_market_transformer(resolved_model_config)
    train_loader = _build_smoke_dataloader(
        dataset,
        batch_size=batch_size,
        num_classes=num_classes,
    )
    history = fit_transformer_classifier(model, train_loader, None, training_config)

    eval_loader = _build_smoke_dataloader(
        dataset,
        batch_size=batch_size,
        num_classes=num_classes,
    )
    evaluation = evaluate_transformer_classifier(model, eval_loader, device="cpu")

    return {
        "path": str(file_path),
        "symbol_filter": symbol,
        "input_record_count": int(sequence.input_record_count),
        "tokenised_record_count": len(sequence.records),
        "window_length": int(window_length),
        "window_count": len(dataset),
        "num_classes": int(num_classes),
        "vocab_sizes": dict(sequence.field_sizes),
        "model_parameter_count": int(model.n_parameters()),
        "training_history": [
            {
                "epoch": int(item.epoch),
                "train_loss": float(item.train_loss),
            }
            for item in history
        ],
        "synthetic_smoke_metrics": {
            "loss": float(evaluation["loss"]),
            "accuracy": float(evaluation["metrics"]["accuracy"]),
            "n_samples": int(evaluation["metrics"]["n_samples"]),
        },
        "label_source": (
            "synthetic_plumbing:(sample_index + last_real_token_side_id) "
            "mod num_classes"
        ),
        "notes": _SYNTHETIC_SMOKE_WARNING,
        "write_outputs": False,
    }
