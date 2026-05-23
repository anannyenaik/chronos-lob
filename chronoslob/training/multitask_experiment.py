"""Multi-task supervised fine-tuning utilities for tokenised sequences.

This Phase 14 module wires the Phase-11 token windows, Phase-12
transformer encoder backbone and Phase-14 multi-task heads into a tiny,
deterministic CPU-compatible training path. It is infrastructure only:
no calibration, confidence filtering, execution simulation, backtesting,
checkpointing, dashboards or market-performance claims are implemented.
The event-log smoke runner uses a local synthetic fixture and labels the
result as supervised plumbing only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chronoslob.book.event_replay import replay_event_log_to_label_frame
from chronoslob.data.event_store import read_event_log_jsonl
from chronoslob.labels.pipeline import LabelPipelineConfig
from chronoslob.models.multitask import (
    DEFAULT_TASK_HEADS,
    MultiTaskTransformer,
    MultiTaskTransformerConfig,
    TaskHeadConfig,
    create_multitask_transformer,
)
from chronoslob.models.tokenisation import (
    TokenisationConfig,
    tokenise_event_log,
)
from chronoslob.models.transformer import MarketTransformerConfig
from chronoslob.training.multitask_datasets import (
    MultiTaskLabelSpec,
    MultiTaskTokenDataset,
    MultiTaskWindowConfig,
    collate_multitask_token_windows,
)
from chronoslob.training.token_datasets import (
    TOKEN_WINDOW_FIELD_NAMES,
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
    "MultiTaskEpochResult",
    "MultiTaskTrainingConfig",
    "evaluate_multitask_classifier",
    "fit_multitask_model",
    "run_multitask_smoke_from_event_log",
    "train_multitask_one_epoch",
]

_SYNTHETIC_MULTITASK_WARNING = (
    "Synthetic supervised plumbing only; losses and accuracies are not market "
    "evidence and imply no alpha, tradability or benchmark performance."
)

_DIRECTION_LABEL_IDS: Mapping[str, int] = {
    "down": 0,
    "stationary": 1,
    "up": 2,
}


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the multi-task experiment runner. "
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
                "ChronosLOB multi-task training defaults to CPU"
            )
        return torch_module.device(normalised)
    raise ValueError(
        f"unsupported device {device!r}; ChronosLOB multi-task training "
        "currently supports 'cpu' or 'cuda'-prefixed devices"
    )


@dataclass(frozen=True)
class MultiTaskTrainingConfig:
    """Configuration for the supervised multi-task fit loop."""

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
class MultiTaskEpochResult:
    """Per-epoch summary returned by :func:`fit_multitask_model`."""

    epoch: int
    train_loss: float
    train_loss_components: dict[str, float] = field(default_factory=dict)
    train_valid_counts: dict[str, int] = field(default_factory=dict)
    validation_loss: float | None = None
    validation_loss_components: dict[str, float] = field(default_factory=dict)
    validation_metrics: dict[str, object] | None = field(default=None)


def _move_tensor_mapping_to_device(
    values: Mapping[str, Any],
    device_obj: Any,
) -> dict[str, Any]:
    torch_module = _require_torch()
    return {
        key: value.to(device_obj) if torch_module.is_tensor(value) else value
        for key, value in values.items()
    }


def _split_batch(
    batch: Mapping[str, Any],
    device_obj: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if "targets" not in batch or "target_mask" not in batch:
        raise KeyError("multi-task batch is missing 'targets' or 'target_mask'")
    inputs: dict[str, Any] = {}
    for field_name in TOKEN_WINDOW_FIELD_NAMES:
        if field_name not in batch:
            raise KeyError(f"batch is missing required token field {field_name!r}")
        inputs[field_name] = batch[field_name].to(device_obj)
    if "attention_mask" not in batch:
        raise KeyError("batch is missing required 'attention_mask'")
    inputs["attention_mask"] = batch["attention_mask"].to(device_obj)
    targets = batch["targets"]
    target_mask = batch["target_mask"]
    if not isinstance(targets, Mapping):
        raise TypeError("batch['targets'] must be a mapping")
    if not isinstance(target_mask, Mapping):
        raise TypeError("batch['target_mask'] must be a mapping")
    return (
        inputs,
        _move_tensor_mapping_to_device(targets, device_obj),
        _move_tensor_mapping_to_device(target_mask, device_obj),
    )


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
        if count > 0:
            averaged[name] = total / count
    return averaged


def train_multitask_one_epoch(
    model: MultiTaskTransformer,
    dataloader: Iterable[Any],
    optimizer: Any,
    *,
    device: str = "cpu",
    gradient_clip_norm: float | None = None,
) -> tuple[float, dict[str, float], dict[str, int]]:
    """Run one supervised multi-task training epoch.

    Returns the sample-weighted mean combined loss, sample-weighted mean
    per-task loss components and per-task valid label counts.
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
    valid_counts: dict[str, int] = dict.fromkeys(model.task_names, 0)

    for raw_batch in dataloader:
        if not isinstance(raw_batch, Mapping):
            raise TypeError(
                "multi-task training expects mapping batches; got "
                f"{type(raw_batch).__name__}"
            )
        inputs, targets, target_mask = _split_batch(raw_batch, device_obj)
        batch_size = int(inputs["attention_mask"].shape[0])
        if batch_size == 0:
            continue

        optimizer.zero_grad()
        output = model(inputs, targets=targets, target_mask=target_mask)
        if output.loss is None:
            raise RuntimeError("multi-task forward did not produce a loss")
        loss = output.loss
        if not torch_module.isfinite(loss).item():
            raise RuntimeError("multi-task training loss became non-finite")
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
        for name, count in output.valid_counts.items():
            valid_counts[name] = valid_counts.get(name, 0) + int(count)

    if total_samples == 0:
        raise ValueError("train_multitask_one_epoch received an empty dataloader")
    return (
        total_loss / total_samples,
        _average_components(component_totals, component_counts),
        valid_counts,
    )


def evaluate_multitask_classifier(
    model: MultiTaskTransformer,
    dataloader: Iterable[Any],
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate a multi-task classifier without updating parameters."""
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
    valid_counts: dict[str, int] = dict.fromkeys(model.task_names, 0)
    correct_counts: dict[str, int] = dict.fromkeys(model.task_names, 0)

    with torch_module.no_grad():
        for raw_batch in dataloader:
            if not isinstance(raw_batch, Mapping):
                raise TypeError(
                    "multi-task evaluation expects mapping batches; got "
                    f"{type(raw_batch).__name__}"
                )
            inputs, targets, target_mask = _split_batch(raw_batch, device_obj)
            batch_size = int(inputs["attention_mask"].shape[0])
            if batch_size == 0:
                continue
            output = model(inputs, targets=targets, target_mask=target_mask)
            if output.loss is None:
                raise RuntimeError("multi-task evaluation did not produce a loss")
            if not torch_module.isfinite(output.loss).item():
                raise RuntimeError("multi-task evaluation loss became non-finite")

            total_loss += float(output.loss.item()) * batch_size
            total_samples += batch_size
            _accumulate_components(
                component_totals,
                component_counts,
                output.loss_components,
                batch_size,
            )
            for task in model.config.tasks:
                name = task.name
                if name not in targets or name not in output.logits:
                    continue
                valid_mask = targets[name] != int(task.ignore_index)
                if name in target_mask:
                    valid_mask = valid_mask & target_mask[name]
                valid = int(valid_mask.sum().item())
                valid_counts[name] = valid_counts.get(name, 0) + valid
                if valid == 0:
                    continue
                predictions = output.logits[name].argmax(dim=-1)
                correct = int((predictions[valid_mask] == targets[name][valid_mask]).sum().item())
                correct_counts[name] = correct_counts.get(name, 0) + correct

    if total_samples == 0:
        raise ValueError("evaluate_multitask_classifier received an empty dataloader")

    task_metrics: dict[str, dict[str, float | int | None]] = {}
    for name in model.task_names:
        valid = int(valid_counts.get(name, 0))
        correct = int(correct_counts.get(name, 0))
        task_metrics[name] = {
            "valid_count": valid,
            "correct": correct,
            "accuracy": (correct / valid if valid > 0 else None),
        }
    return {
        "loss": total_loss / total_samples,
        "loss_components": _average_components(component_totals, component_counts),
        "valid_counts": valid_counts,
        "task_metrics": task_metrics,
        "n_samples": int(total_samples),
    }


def fit_multitask_model(
    model: MultiTaskTransformer,
    train_loader: Iterable[Any],
    validation_loader: Iterable[Any] | None = None,
    config: MultiTaskTrainingConfig | None = None,
) -> list[MultiTaskEpochResult]:
    """Train a supervised multi-task model and return per-epoch results."""
    _require_torch()
    if model is None:
        raise ValueError("model must not be None")
    training_config = config if config is not None else MultiTaskTrainingConfig()
    if not isinstance(training_config, MultiTaskTrainingConfig):
        raise TypeError("config must be a MultiTaskTrainingConfig instance")
    device_obj = _resolve_device(training_config.device)
    set_torch_deterministic(training_config.seed)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("multi-task model has no trainable parameters")
    model.to(device_obj)
    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    history: list[MultiTaskEpochResult] = []
    for epoch in range(1, training_config.epochs + 1):
        train_loss, train_components, train_valid_counts = train_multitask_one_epoch(
            model,
            train_loader,
            optimizer,
            device=training_config.device,
            gradient_clip_norm=training_config.gradient_clip_norm,
        )
        validation_loss: float | None = None
        validation_components: dict[str, float] = {}
        validation_metrics: dict[str, object] | None = None
        if validation_loader is not None:
            evaluation = evaluate_multitask_classifier(
                model,
                validation_loader,
                device=training_config.device,
            )
            validation_loss = float(evaluation["loss"])
            validation_components = dict(evaluation["loss_components"])
            validation_metrics = {
                "valid_counts": evaluation["valid_counts"],
                "task_metrics": evaluation["task_metrics"],
            }
        history.append(
            MultiTaskEpochResult(
                epoch=epoch,
                train_loss=train_loss,
                train_loss_components=train_components,
                train_valid_counts=train_valid_counts,
                validation_loss=validation_loss,
                validation_loss_components=validation_components,
                validation_metrics=validation_metrics,
            )
        )
    return history


def _default_label_specs(
    tasks: Sequence[TaskHeadConfig] = DEFAULT_TASK_HEADS,
) -> tuple[MultiTaskLabelSpec, ...]:
    return tuple(
        MultiTaskLabelSpec(
            name=task.name,
            num_classes=int(task.num_classes),
            ignore_index=int(task.ignore_index),
        )
        for task in tasks
    )


def _default_smoke_model_config(
    vocab_sizes: Mapping[str, int],
    *,
    window_length: int,
    tasks: tuple[TaskHeadConfig, ...] = DEFAULT_TASK_HEADS,
) -> MultiTaskTransformerConfig:
    backbone = MarketTransformerConfig(
        vocab_sizes=dict(vocab_sizes),
        field_embedding_dim=8,
        model_dim=32,
        num_heads=4,
        num_layers=2,
        feedforward_dim=64,
        dropout=0.1,
        max_sequence_length=window_length,
        num_classes=2,
        pooling="mean",
        activation="gelu",
        use_layer_norm=True,
    )
    return MultiTaskTransformerConfig(
        backbone=backbone,
        tasks=tasks,
        dropout=0.1,
        freeze_backbone=False,
    )


def _to_python_datetime(value: Any) -> Any:
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    return value


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return value != value
    return False


def _bool_int(value: Any) -> int | None:
    if _missing(value):
        return None
    return int(bool(value))


def _direction_int(value: Any) -> int | None:
    if _missing(value):
        return None
    label = str(value)
    if label not in _DIRECTION_LABEL_IDS:
        raise ValueError(f"unsupported direction label {label!r}")
    return int(_DIRECTION_LABEL_IDS[label])


def _integer_label(value: Any) -> int | None:
    if _missing(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    return int(value)


def _volatility_regime_labels(values: Sequence[Any]) -> list[int | None]:
    indexed_values: list[tuple[float, int]] = []
    output: list[int | None] = [None] * len(values)
    for position, value in enumerate(values):
        if _missing(value):
            continue
        indexed_values.append((float(value), position))
    if not indexed_values:
        return output
    ordered = sorted(indexed_values)
    n_values = len(ordered)
    for rank, (_, position) in enumerate(ordered):
        output[position] = min(2, int(rank * 3 / max(n_values, 1)))
    return output


def _smoke_label_table_from_frame(
    label_frame: Any,
    label_specs: Sequence[MultiTaskLabelSpec],
) -> dict[Any, dict[str, int | None]]:
    if not hasattr(label_frame, "iterrows"):
        raise TypeError("label_frame must be a pandas-like DataFrame")
    volatility_values = (
        list(label_frame["future_volatility_1"])
        if "future_volatility_1" in label_frame.columns
        else [None] * len(label_frame)
    )
    volatility_labels = _volatility_regime_labels(volatility_values)
    table: dict[Any, dict[str, int | None]] = {}
    for position, row in label_frame.reset_index(drop=True).iterrows():
        labels: dict[str, int | None] = {}
        for spec in label_specs:
            if spec.name == "direction":
                labels[spec.name] = _direction_int(row.get("direction_1"))
            elif spec.name == "return_quantile":
                labels[spec.name] = _integer_label(row.get("return_quantile_1"))
            elif spec.name == "volatility_regime":
                labels[spec.name] = volatility_labels[int(position)]
            elif spec.name == "spread_widening":
                labels[spec.name] = _bool_int(row.get("spread_widening_1"))
            elif spec.name == "fill_probability":
                bid = _bool_int(row.get("bid_fill_proxy_1"))
                ask = _bool_int(row.get("ask_fill_proxy_1"))
                labels[spec.name] = (
                    None
                    if bid is None and ask is None
                    else int(bool(bid) or bool(ask))
                )
            elif spec.name == "adverse_selection":
                bid = _bool_int(row.get("bid_adverse_selection_proxy_2"))
                ask = _bool_int(row.get("ask_adverse_selection_proxy_2"))
                labels[spec.name] = (
                    None
                    if bid is None and ask is None
                    else int(bool(bid) or bool(ask))
                )
            else:
                labels[spec.name] = None
        key = (row["symbol"], _to_python_datetime(row["timestamp"]))
        table[key] = labels
    return table


def _multitask_dataloader(
    dataset: MultiTaskTokenDataset,
    *,
    batch_size: int,
) -> Any:
    _require_torch()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_multitask_token_windows,
    )


def run_multitask_smoke_from_event_log(
    path: str | Path,
    *,
    symbol: str | None = None,
    window_length: int = 4,
    batch_size: int = 4,
    epochs: int = 1,
    seed: int = 42,
    max_levels_per_side: int = 2,
    learning_rate: float = 1e-3,
    multitask_config: MultiTaskTransformerConfig | None = None,
) -> dict[str, Any]:
    """Run a tiny CPU multi-task smoke experiment from a local event log.

    The runner tokenises a local canonical event log, builds token windows,
    generates tiny supervised labels through the existing label pipeline
    where available and uses a deterministic volatility-regime plumbing
    label for the synthetic fixture. It trains for ``epochs`` on CPU and
    returns a structured payload marked synthetic. Nothing is written to
    disk and no market-performance claim is made.
    """
    _require_torch()
    file_path = Path(path)
    tokenisation_config = TokenisationConfig(
        max_levels_per_side=max_levels_per_side,
    )
    sequence = tokenise_event_log(file_path, tokenisation_config, symbol=symbol)
    token_window_config = TokenWindowConfig(
        window_length=window_length,
        stride=1,
        drop_incomplete=False,
        padding_side="left",
    )
    token_windows = build_token_window_indices(sequence, token_window_config)
    if not token_windows:
        raise ValueError(
            "multi-task smoke produced zero windows; the event log may be "
            "empty or every record may have been filtered out by --symbol"
        )

    records = read_event_log_jsonl(file_path)
    if symbol is not None:
        records = [record for record in records if record.symbol == symbol]
    label_config = LabelPipelineConfig(
        horizons=(1,),
        include_direction=True,
        include_return=True,
        include_return_quantiles=True,
        include_volatility=True,
        include_spread_widening=True,
        include_fill_probability=True,
        include_adverse_selection=True,
        fill_horizon=1,
        adverse_evaluation_horizon=2,
        missing="none",
    )
    label_frame = replay_event_log_to_label_frame(
        records,
        label_config=label_config,
    )

    if multitask_config is None:
        resolved_model_config = _default_smoke_model_config(
            sequence.field_sizes,
            window_length=window_length,
        )
    else:
        if not isinstance(multitask_config, MultiTaskTransformerConfig):
            raise TypeError(
                "multitask_config must be a MultiTaskTransformerConfig instance"
            )
        if multitask_config.backbone.max_sequence_length < window_length:
            raise ValueError(
                "multitask_config.backbone.max_sequence_length is smaller than "
                f"window_length: {multitask_config.backbone.max_sequence_length} "
                f"< {window_length}"
            )
        resolved_model_config = multitask_config

    label_specs = _default_label_specs(resolved_model_config.tasks)
    label_table = _smoke_label_table_from_frame(label_frame, label_specs)
    window_config = MultiTaskWindowConfig(
        window_length=window_length,
        stride=1,
        drop_incomplete=False,
        respect_symbol_boundaries=True,
        respect_split_boundaries=True,
        drop_all_missing_samples=True,
    )
    dataset = MultiTaskTokenDataset(
        sequence,
        window_config,
        label_specs,
        label_table=label_table,
    )
    if len(dataset) == 0:
        raise ValueError(
            "multi-task smoke produced no supervised windows with valid labels"
        )

    set_torch_deterministic(seed)
    model = create_multitask_transformer(resolved_model_config)
    train_loader = _multitask_dataloader(dataset, batch_size=batch_size)
    training_config = MultiTaskTrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        device="cpu",
    )
    history = fit_multitask_model(model, train_loader, None, training_config)

    eval_loader = _multitask_dataloader(dataset, batch_size=batch_size)
    evaluation = evaluate_multitask_classifier(model, eval_loader, device="cpu")
    final_history = history[-1] if history else None

    task_metrics = dict(evaluation["task_metrics"])
    task_accuracy = {
        task_name: metrics["accuracy"]
        for task_name, metrics in task_metrics.items()
        if metrics["accuracy"] is not None
    }
    return {
        "path": str(file_path),
        "symbol_filter": symbol,
        "input_record_count": int(sequence.input_record_count),
        "tokenised_record_count": len(sequence.records),
        "window_length": int(window_length),
        "window_count": len(token_windows),
        "supervised_window_count": len(dataset),
        "enabled_tasks": list(resolved_model_config.task_names),
        "valid_labels_per_task": dataset.task_label_counts(),
        "vocab_sizes": dict(sequence.field_sizes),
        "model_parameter_count": int(model.n_parameters()),
        "training_history": [
            {
                "epoch": int(item.epoch),
                "train_loss": float(item.train_loss),
                "train_loss_components": dict(item.train_loss_components),
                "train_valid_counts": dict(item.train_valid_counts),
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
            "task_accuracy": task_accuracy,
            "valid_counts": dict(evaluation["valid_counts"]),
            "n_samples": int(evaluation["n_samples"]),
        },
        "label_source": (
            "existing_label_pipeline_plus_deterministic_synthetic_volatility_"
            "regime_for_plumbing"
        ),
        "notes": _SYNTHETIC_MULTITASK_WARNING,
        "write_outputs": False,
    }
