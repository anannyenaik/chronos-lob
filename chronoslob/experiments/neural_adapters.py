"""Neural adapters for the FI-2010 paper experiment runner.

The adapters in this module keep neural baselines behind the same
artefact contract as the classical runner. They receive the runner's
temporal train/validation/test partitions, build split-contained
windows, fit preprocessing on training rows only and return in-memory
predictions and metrics for the held-out test split.

``deeplob_style`` uses the existing numeric sequence dataset, DeepLOB-style
model and generic PyTorch classifier loop. ``transformer`` uses the
existing snapshot tokenisation layer, supervised market transformer and
transformer training loop. ``ssl_transformer`` is intentionally absent
until a train-only self-supervised pretraining and fine-tuning path is
implemented end to end.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chronoslob.experiments.fi2010_benchmark import (
    FI2010BenchmarkConfig,
    PaperNeuralSettings,
)
from chronoslob.experiments.neural_benchmarking import resolve_neural_device
from chronoslob.models.deeplob import DeepLOBConfig, create_deeplob_model
from chronoslob.models.matrix_transformer import (
    MatrixTransformerConfig,
    create_matrix_transformer,
)
from chronoslob.models.preprocessing import (
    TrainOnlyStandardScaler,
    build_target_vector,
)
from chronoslob.models.tokenisation import (
    TokenisationConfig,
    fit_tokenisation_state,
    tokenise_records,
)
from chronoslob.models.transformer import (
    MarketTransformerConfig,
    create_market_transformer,
)
from chronoslob.training.dataloaders import (
    DataLoaderConfig,
    create_sequence_dataloader,
)
from chronoslob.training.datasets import (
    SequenceDataset,
    SequenceWindowConfig,
    encode_target_values,
)
from chronoslob.training.metrics import (
    compute_classification_metrics,
    confusion_matrix_as_dict,
)
from chronoslob.training.splitters import SplitIndices
from chronoslob.training.token_batching import collate_token_windows
from chronoslob.training.token_datasets import (
    TokenSequenceDataset,
    TokenWindowConfig,
    TokenWindowIndex,
    build_token_window_indices,
)
from chronoslob.training.torch_training import (
    TorchTrainingConfig,
    evaluate_torch_classifier,
    fit_torch_classifier,
    set_torch_deterministic,
)
from chronoslob.training.transformer_experiment import (
    TransformerTrainingConfig,
    evaluate_transformer_classifier,
    fit_transformer_classifier,
)

try:  # pragma: no cover - exercised when torch is unavailable
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset as _TorchDataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]
    _TorchDataset = object  # type: ignore[assignment,misc]
    _TORCH_AVAILABLE = False

__all__ = [
    "NeuralPaperModelResult",
    "effective_matrix_window_length",
    "run_matrix_transformer_finetune",
    "run_neural_paper_model",
]


@dataclass(frozen=True)
class NeuralPaperModelResult:
    """In-memory outcome returned by a neural paper-runner adapter."""

    model_name: str
    model_type: str
    metrics: dict[str, float]
    confusion_matrix: dict[str, Any]
    n_test_rows: int
    prediction_rows: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for neural paper-runner models. Install the "
            "'torch' optional dependency: pip install -e '.[torch]'"
        )
    return torch


def _normalise_class_value(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    return value


def _to_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _select_metric_subset(metric_dict: Mapping[str, Any]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for key, value in metric_dict.items():
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            numeric = float(value)
            if math.isfinite(numeric):
                cleaned[key] = numeric
    return cleaned


def _best_epoch(history: Sequence[Any]) -> int | None:
    for item in reversed(list(history)):
        if bool(item.is_best):
            return int(item.epoch)
    if not history:
        return None
    return int(history[-1].epoch)


def _history_early_stopped(history: Sequence[Any]) -> bool:
    return any(bool(item.early_stop) for item in history)


def _history_rows(history: Sequence[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in history:
        monitored_value = item.monitored_value
        row: dict[str, Any] = {
            "epoch": int(item.epoch),
            "train_loss": float(item.train_loss),
            "validation_loss": (
                None if item.validation_loss is None else float(item.validation_loss)
            ),
            "monitored_value": (
                None
                if monitored_value is None
                else float(monitored_value)
            ),
            "is_best": bool(item.is_best),
            "early_stop": bool(item.early_stop),
        }
        validation_metrics = item.validation_metrics
        if isinstance(validation_metrics, Mapping):
            metrics = validation_metrics.get("metrics")
            if isinstance(metrics, Mapping):
                row["validation_macro_f1"] = metrics.get("macro_f1")
                row["validation_accuracy"] = metrics.get("accuracy")
                row["validation_mcc"] = metrics.get("matthews_corrcoef")
        rows.append(row)
    return rows


def _expected_calibration_error(
    *,
    y_true: Sequence[Any],
    probabilities: np.ndarray,
    labels: Sequence[Any],
    n_bins: int = 10,
) -> float:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be 2D for ECE")
    if probabilities.shape[0] != len(y_true):
        raise ValueError("probability row count must match y_true")
    confidences = probabilities.max(axis=1)
    predicted_indices = probabilities.argmax(axis=1)
    label_array = np.asarray(list(labels), dtype=object)
    predicted_labels = label_array[predicted_indices]
    accuracies = (predicted_labels == np.asarray(list(y_true), dtype=object)).astype(
        float
    )
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(len(y_true))
    if total <= 0:
        return 0.0
    ece = 0.0
    for index in range(n_bins):
        lo = bin_edges[index]
        hi = bin_edges[index + 1]
        if index == 0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi)
        if not bool(in_bin.any()):
            continue
        bin_size = float(in_bin.sum())
        bin_acc = float(accuracies[in_bin].mean())
        bin_conf = float(confidences[in_bin].mean())
        ece += abs(bin_acc - bin_conf) * bin_size / total
    return ece


def _multiclass_brier_score(
    *,
    y_true: Sequence[Any],
    probabilities: np.ndarray,
    labels: Sequence[Any],
) -> float:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be 2D for Brier score")
    if probabilities.shape[0] != len(y_true):
        raise ValueError("probability row count must match y_true")
    label_to_index = {label: position for position, label in enumerate(labels)}
    one_hot = np.zeros_like(probabilities, dtype=float)
    for row_position, label in enumerate(y_true):
        class_index = label_to_index.get(label)
        if class_index is None:
            continue
        one_hot[row_position, class_index] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _timestamps_for_rows(
    frame: pd.DataFrame,
    *,
    config: FI2010BenchmarkConfig,
) -> pd.Series:
    if config.timestamp_column is not None and config.timestamp_column in frame.columns:
        return pd.to_datetime(frame[config.timestamp_column], utc=True, errors="raise")
    base = pd.Timestamp("2000-01-01T00:00:00Z")
    return pd.Series(
        [base + pd.Timedelta(seconds=index) for index in range(len(frame))],
        index=frame.index,
    )


def _build_sequence_frames(
    frame: pd.DataFrame,
    *,
    config: FI2010BenchmarkConfig,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = _timestamps_for_rows(frame, config=config).reset_index(drop=True)
    metadata_frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["FI2010"] * len(frame),
        }
    )
    matrix_frame = pd.DataFrame(
        {
            column: frame[column].reset_index(drop=True)
            for column in feature_columns
        }
    )
    feature_frame = pd.concat([metadata_frame, matrix_frame], axis=1)

    label_frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["FI2010"] * len(frame),
            config.label_name: frame[config.label_name].reset_index(drop=True),
        }
    )
    return feature_frame, label_frame


def _standardise_feature_frame(
    feature_frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    train_indices: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_values = (
        feature_frame.iloc[list(train_indices)]
        .loc[:, list(feature_columns)]
        .to_numpy(dtype=float, copy=True)
    )
    scaler = TrainOnlyStandardScaler()
    scaler.fit(train_values)
    all_values = feature_frame.loc[:, list(feature_columns)].to_numpy(
        dtype=float,
        copy=True,
    )
    scaled = feature_frame.copy()
    transformed = scaler.transform(all_values)
    for position, column in enumerate(feature_columns):
        scaled[column] = transformed[:, position].astype(float, copy=False)
    return scaled, {
        "standardise": True,
        "feature_columns": list(feature_columns),
        "fit_split": "train",
        "mean": [float(value) for value in scaler.mean_.tolist()],
        "scale": [float(value) for value in scaler.scale_.tolist()],
    }


def _train_class_mapping(
    frame: pd.DataFrame,
    *,
    config: FI2010BenchmarkConfig,
    split: SplitIndices,
) -> dict[object, int]:
    train_target = build_target_vector(
        frame,
        target_column=config.label_name,
        row_indices=split.train,
    )
    _, mapping = encode_target_values(train_target.y.tolist())
    if len(mapping) < 2:
        raise ValueError(
            "training split has fewer than two classes for neural classification"
        )
    return mapping


def _make_sequence_dataset(
    *,
    feature_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
    config: FI2010BenchmarkConfig,
    feature_columns: Sequence[str],
    lookback: int,
    class_to_index: Mapping[object, int],
    allowed_indices: Sequence[int],
    partition_name: str,
    required: bool,
) -> tuple[SequenceDataset | None, str | None]:
    if not allowed_indices:
        reason = f"{partition_name} split contains no rows"
        if required:
            raise ValueError(reason)
        return None, reason
    sequence_config = SequenceWindowConfig(
        lookback=lookback,
        target_column=config.label_name,
        feature_columns=list(feature_columns),
        require_contiguous_indices=True,
    )
    try:
        dataset = SequenceDataset(
            feature_frame=feature_frame,
            label_frame=label_frame,
            config=sequence_config,
            feature_columns=list(feature_columns),
            class_to_index=class_to_index,
            allowed_target_indices=list(allowed_indices),
        )
    except ValueError as exc:
        reason = f"{partition_name} sequence dataset could not be built: {exc}"
        if required:
            raise ValueError(reason) from exc
        return None, reason
    if len(dataset) == 0:
        reason = (
            f"{partition_name} split has no usable sequence samples after "
            f"lookback={lookback} filtering"
        )
        if required:
            raise ValueError(reason)
        return None, reason
    return dataset, None


def effective_matrix_window_length(
    *,
    configured_window_length: int,
    split: SplitIndices,
) -> int:
    """Public wrapper so the SSL runner reuses the exact window-length rule.

    The SSL encoder's ``max_sequence_length`` must equal the supervised
    fine-tuning classifier's window length for the encoder transfer to be
    architecturally valid. Both therefore derive the effective window from this
    single function.
    """
    return _effective_matrix_window_length(
        configured_window_length=configured_window_length,
        split=split,
    )


def _effective_matrix_window_length(
    *,
    configured_window_length: int,
    split: SplitIndices,
) -> int:
    """Return a split-contained window length for small smoke partitions."""
    partition_lengths = [
        len(split.train),
        len(split.test),
    ]
    usable_lengths = [length for length in partition_lengths if length > 0]
    if not usable_lengths:
        raise ValueError("matrix transformer requires train and test rows")
    return max(1, min(int(configured_window_length), min(usable_lengths)))


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("probabilities must be a 2D matrix")
    if bool(np.isnan(matrix).any()) or bool(np.isinf(matrix).any()):
        raise ValueError("probabilities must be finite")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("probability rows must sum to one")
    return matrix


def _align_probabilities_to_all_labels(
    probabilities: np.ndarray,
    *,
    class_to_index: Mapping[object, int],
    all_labels: Sequence[Any],
) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    aligned = np.zeros((matrix.shape[0], len(all_labels)), dtype=float)
    normalised_mapping = {
        _normalise_class_value(label): int(index)
        for label, index in class_to_index.items()
    }
    for column_position, label in enumerate(all_labels):
        source_index = normalised_mapping.get(_normalise_class_value(label))
        if source_index is None or source_index >= matrix.shape[1]:
            continue
        aligned[:, column_position] = matrix[:, source_index]
    return _validate_probabilities(aligned)


def _encoded_to_original(
    values: Sequence[int],
    *,
    index_to_class: Mapping[int, object],
) -> list[Any]:
    output: list[Any] = []
    for value in values:
        class_value = index_to_class[int(value)]
        output.append(_to_json_value(class_value))
    return output


def _build_prediction_rows(
    *,
    row_indices: Sequence[int],
    true_labels: Sequence[Any],
    predictions: Sequence[Any],
    probabilities: np.ndarray,
    all_labels: Sequence[Any],
    timestamps: Sequence[str] | None,
    model_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    label_order = [str(label) for label in all_labels]
    for position, row_index in enumerate(row_indices):
        row_probabilities = probabilities[position]
        row: dict[str, Any] = {
            "row_index": int(row_index),
            "split": "test",
            "model_name": model_name,
            "label": _to_json_value(true_labels[position]),
            "prediction": _to_json_value(predictions[position]),
            "confidence": float(row_probabilities.max()),
        }
        if timestamps is not None:
            row["timestamp"] = timestamps[position]
        for class_position, class_label in enumerate(label_order):
            row[f"probability_{class_label}"] = float(
                row_probabilities[class_position]
            )
        rows.append(row)
    return rows


def _result_from_evaluation(
    *,
    model_name: str,
    model_type: str,
    evaluation: Mapping[str, Any],
    row_indices: Sequence[int],
    class_to_index: Mapping[object, int],
    all_labels: Sequence[Any],
    class_count_train: int,
    class_count_test: int,
    test_timestamps: Sequence[str] | None,
    metadata: Mapping[str, Any],
) -> NeuralPaperModelResult:
    index_to_class = {int(index): label for label, index in class_to_index.items()}
    targets = [int(value) for value in evaluation["targets"]]
    predicted_indices = [int(value) for value in evaluation["predictions"]]
    true_labels = _encoded_to_original(targets, index_to_class=index_to_class)
    predicted_labels = _encoded_to_original(
        predicted_indices,
        index_to_class=index_to_class,
    )
    probabilities = _align_probabilities_to_all_labels(
        np.asarray(evaluation["probabilities"], dtype=float),
        class_to_index=class_to_index,
        all_labels=all_labels,
    )
    metrics = compute_classification_metrics(
        true_labels,
        predicted_labels,
        y_proba=probabilities,
        labels=all_labels,
    )
    metric_subset = _select_metric_subset(metrics.to_dict())
    metric_subset["class_count_train"] = float(class_count_train)
    metric_subset["class_count_test"] = float(class_count_test)
    if "brier_score" not in metric_subset:
        metric_subset["brier_score"] = _multiclass_brier_score(
            y_true=true_labels,
            probabilities=probabilities,
            labels=all_labels,
        )
    metric_subset["mean_confidence"] = float(probabilities.max(axis=1).mean())
    metric_subset["expected_calibration_error"] = float(
        _expected_calibration_error(
            y_true=true_labels,
            probabilities=probabilities,
            labels=all_labels,
        )
    )
    confusion = confusion_matrix_as_dict(
        y_true=true_labels,
        y_pred=predicted_labels,
        labels=all_labels,
    )
    rows = _build_prediction_rows(
        row_indices=row_indices,
        true_labels=true_labels,
        predictions=predicted_labels,
        probabilities=probabilities,
        all_labels=all_labels,
        timestamps=test_timestamps,
        model_name=model_name,
    )
    return NeuralPaperModelResult(
        model_name=model_name,
        model_type=model_type,
        metrics=metric_subset,
        confusion_matrix=confusion,
        n_test_rows=len(row_indices),
        prediction_rows=rows,
        metadata=dict(metadata),
    )


def _run_deeplob_style(
    *,
    frame: pd.DataFrame,
    config: FI2010BenchmarkConfig,
    split: SplitIndices,
    feature_columns: Sequence[str],
    all_labels: Sequence[Any],
    class_count_train: int,
    class_count_test: int,
    test_timestamps: Sequence[str] | None,
    settings: PaperNeuralSettings,
) -> NeuralPaperModelResult:
    _require_torch()
    feature_frame, label_frame = _build_sequence_frames(
        frame,
        config=config,
        feature_columns=feature_columns,
    )
    scaled_feature_frame, scaler_metadata = _standardise_feature_frame(
        feature_frame,
        feature_columns=feature_columns,
        train_indices=split.train,
    )
    class_mapping = _train_class_mapping(frame, config=config, split=split)
    train_dataset, _ = _make_sequence_dataset(
        feature_frame=scaled_feature_frame,
        label_frame=label_frame,
        config=config,
        feature_columns=feature_columns,
        lookback=settings.lookback,
        class_to_index=class_mapping,
        allowed_indices=split.train,
        partition_name="train",
        required=True,
    )
    assert train_dataset is not None
    validation_dataset, validation_skip = _make_sequence_dataset(
        feature_frame=scaled_feature_frame,
        label_frame=label_frame,
        config=config,
        feature_columns=feature_columns,
        lookback=settings.lookback,
        class_to_index=class_mapping,
        allowed_indices=split.validation,
        partition_name="validation",
        required=False,
    )
    test_dataset, _ = _make_sequence_dataset(
        feature_frame=scaled_feature_frame,
        label_frame=label_frame,
        config=config,
        feature_columns=feature_columns,
        lookback=settings.lookback,
        class_to_index=class_mapping,
        allowed_indices=split.test,
        partition_name="test",
        required=True,
    )
    assert test_dataset is not None

    loader_config = DataLoaderConfig(
        batch_size=settings.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    train_loader = create_sequence_dataloader(train_dataset, loader_config)
    validation_loader = (
        create_sequence_dataloader(validation_dataset, loader_config)
        if validation_dataset is not None
        else None
    )
    test_loader = create_sequence_dataloader(test_dataset, loader_config)
    model_config = DeepLOBConfig(
        input_features=train_dataset.n_features,
        n_classes=len(class_mapping),
        conv_channels=settings.deeplob_conv_channels,
        lstm_hidden_size=settings.deeplob_lstm_hidden_size,
        dropout=settings.dropout,
        use_batch_norm=settings.deeplob_use_batch_norm,
    )
    device_resolution = resolve_neural_device(settings.device)
    checkpoint_path = (
        Path(settings.checkpoint_path)
        if settings.checkpoint_enabled and settings.checkpoint_path is not None
        else None
    )
    training_config = TorchTrainingConfig(
        epochs=settings.max_epochs,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        gradient_clip_norm=settings.gradient_clip_norm,
        device=device_resolution.resolved,
        seed=config.seed,
        early_stopping_patience=settings.early_stopping_patience,
        early_stopping_metric=settings.early_stopping_metric,
        checkpoint_path=checkpoint_path,
    )
    if settings.deterministic:
        set_torch_deterministic(config.seed)
    model = create_deeplob_model(model_config)
    training_started = time.perf_counter()
    history = fit_torch_classifier(
        model,
        train_loader,
        validation_loader,
        training_config,
    )
    training_seconds = time.perf_counter() - training_started
    evaluation = evaluate_torch_classifier(
        model,
        test_loader,
        device=device_resolution.resolved,
        labels=sorted(class_mapping.values()),
    )
    row_indices = [sample.target_index for sample in test_dataset.sample_indices]
    metadata: dict[str, Any] = {
        "neural_family": "deeplob_style",
        "sample_counts": {
            "train": len(train_dataset),
            "validation": 0
            if validation_dataset is None
            else len(validation_dataset),
            "test": len(test_dataset),
        },
        "window_policy": {
            "lookback": int(settings.lookback),
            "require_contiguous_indices": True,
            "windows_stay_inside_split": True,
        },
        "validation_skip_reason": validation_skip,
        "training_history": _history_rows(history),
        "max_epochs": int(settings.max_epochs),
        "epochs_ran": len(history),
        "best_epoch": _best_epoch(history),
        "early_stopped": _history_early_stopped(history),
        "training_seconds": float(training_seconds),
        "early_stopping": {
            "metric": settings.early_stopping_metric,
            "patience": settings.early_stopping_patience,
        },
        "device": {
            "requested": device_resolution.requested,
            "resolved": device_resolution.resolved,
            "cuda_available": device_resolution.cuda_available,
        },
        "deterministic_seed": {
            "enabled": bool(settings.deterministic),
            "seed": int(config.seed),
        },
        "model_parameter_count": int(model.n_parameters()),
        "parameter_count": int(model.n_parameters()),
        "checkpoint": {
            "enabled": bool(settings.checkpoint_enabled),
            "path": None if checkpoint_path is None else str(checkpoint_path),
            "written": bool(checkpoint_path is not None and checkpoint_path.is_file()),
        },
        "class_mapping": {str(label): int(index) for label, index in class_mapping.items()},
        "standardisation": scaler_metadata,
    }
    return _result_from_evaluation(
        model_name="deeplob_style",
        model_type="deeplob_style",
        evaluation=evaluation,
        row_indices=row_indices,
        class_to_index=class_mapping,
        all_labels=all_labels,
        class_count_train=class_count_train,
        class_count_test=class_count_test,
        test_timestamps=test_timestamps,
        metadata=metadata,
    )


def _run_matrix_transformer(
    *,
    model_name: str,
    frame: pd.DataFrame,
    config: FI2010BenchmarkConfig,
    split: SplitIndices,
    feature_columns: Sequence[str],
    all_labels: Sequence[Any],
    class_count_train: int,
    class_count_test: int,
    test_timestamps: Sequence[str] | None,
    settings: PaperNeuralSettings,
    pretrained_encoder_state: Mapping[str, Any] | None = None,
    init_source: str = "random_init",
    model_type: str = "normalised_matrix_transformer",
) -> NeuralPaperModelResult:
    _require_torch()
    feature_frame, label_frame = _build_sequence_frames(
        frame,
        config=config,
        feature_columns=feature_columns,
    )
    scaled_feature_frame, scaler_metadata = _standardise_feature_frame(
        feature_frame,
        feature_columns=feature_columns,
        train_indices=split.train,
    )
    class_mapping = _train_class_mapping(frame, config=config, split=split)
    effective_window_length = _effective_matrix_window_length(
        configured_window_length=settings.transformer_window_length,
        split=split,
    )
    train_dataset, _ = _make_sequence_dataset(
        feature_frame=scaled_feature_frame,
        label_frame=label_frame,
        config=config,
        feature_columns=feature_columns,
        lookback=effective_window_length,
        class_to_index=class_mapping,
        allowed_indices=split.train,
        partition_name="train",
        required=True,
    )
    assert train_dataset is not None
    validation_dataset, validation_skip = _make_sequence_dataset(
        feature_frame=scaled_feature_frame,
        label_frame=label_frame,
        config=config,
        feature_columns=feature_columns,
        lookback=effective_window_length,
        class_to_index=class_mapping,
        allowed_indices=split.validation,
        partition_name="validation",
        required=False,
    )
    test_dataset, _ = _make_sequence_dataset(
        feature_frame=scaled_feature_frame,
        label_frame=label_frame,
        config=config,
        feature_columns=feature_columns,
        lookback=effective_window_length,
        class_to_index=class_mapping,
        allowed_indices=split.test,
        partition_name="test",
        required=True,
    )
    assert test_dataset is not None

    loader_config = DataLoaderConfig(
        batch_size=settings.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    train_loader = create_sequence_dataloader(train_dataset, loader_config)
    validation_loader = (
        create_sequence_dataloader(validation_dataset, loader_config)
        if validation_dataset is not None
        else None
    )
    test_loader = create_sequence_dataloader(test_dataset, loader_config)
    model_config = MatrixTransformerConfig(
        input_features=train_dataset.n_features,
        n_classes=len(class_mapping),
        model_dim=settings.transformer_model_dim,
        num_heads=settings.transformer_num_heads,
        num_layers=settings.transformer_num_layers,
        feedforward_dim=settings.transformer_feedforward_dim,
        dropout=settings.dropout,
        max_sequence_length=effective_window_length,
    )
    device_resolution = resolve_neural_device(settings.device)
    checkpoint_path = (
        Path(settings.checkpoint_path)
        if settings.checkpoint_enabled and settings.checkpoint_path is not None
        else None
    )
    training_config = TorchTrainingConfig(
        epochs=settings.max_epochs,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        gradient_clip_norm=settings.gradient_clip_norm,
        device=device_resolution.resolved,
        seed=config.seed,
        early_stopping_patience=settings.early_stopping_patience,
        early_stopping_metric=settings.early_stopping_metric,
        checkpoint_path=checkpoint_path,
    )
    if settings.deterministic:
        set_torch_deterministic(config.seed)
    model = create_matrix_transformer(model_config)
    transferred_encoder_keys: list[str] = []
    if pretrained_encoder_state is not None:
        from chronoslob.models.matrix_ssl import load_encoder_state_into_classifier

        transferred_encoder_keys = load_encoder_state_into_classifier(
            model,
            pretrained_encoder_state,
        )
    training_started = time.perf_counter()
    history = fit_torch_classifier(
        model,
        train_loader,
        validation_loader,
        training_config,
    )
    training_seconds = time.perf_counter() - training_started
    evaluation = evaluate_torch_classifier(
        model,
        test_loader,
        device=device_resolution.resolved,
        labels=sorted(class_mapping.values()),
    )
    row_indices = [sample.target_index for sample in test_dataset.sample_indices]
    metadata: dict[str, Any] = {
        "neural_family": "normalised_fi2010_matrix_transformer",
        "baseline_kind": "supervised transformer baseline",
        "pretraining": {
            "init_source": init_source,
            "encoder_initialised_from_pretraining": bool(
                pretrained_encoder_state is not None
            ),
            "transferred_encoder_key_count": len(transferred_encoder_keys),
        },
        "matrix_path": "normalised FI-2010 matrix path",
        "raw_snapshot_construction": False,
        "sample_counts": {
            "train": len(train_dataset),
            "validation": 0
            if validation_dataset is None
            else len(validation_dataset),
            "test": len(test_dataset),
        },
        "window_policy": {
            "configured_window_length": int(settings.transformer_window_length),
            "effective_window_length": int(effective_window_length),
            "require_contiguous_indices": True,
            "windows_stay_inside_split": True,
        },
        "validation_skip_reason": validation_skip,
        "training_history": _history_rows(history),
        "max_epochs": int(settings.max_epochs),
        "epochs_ran": len(history),
        "best_epoch": _best_epoch(history),
        "early_stopped": _history_early_stopped(history),
        "training_seconds": float(training_seconds),
        "early_stopping": {
            "metric": settings.early_stopping_metric,
            "patience": settings.early_stopping_patience,
        },
        "device": {
            "requested": device_resolution.requested,
            "resolved": device_resolution.resolved,
            "cuda_available": device_resolution.cuda_available,
        },
        "deterministic_seed": {
            "enabled": bool(settings.deterministic),
            "seed": int(config.seed),
        },
        "model_parameter_count": int(model.n_parameters()),
        "parameter_count": int(model.n_parameters()),
        "checkpoint": {
            "enabled": bool(settings.checkpoint_enabled),
            "path": None if checkpoint_path is None else str(checkpoint_path),
            "written": bool(checkpoint_path is not None and checkpoint_path.is_file()),
        },
        "class_mapping": {str(label): int(index) for label, index in class_mapping.items()},
        "standardisation": scaler_metadata,
    }
    return _result_from_evaluation(
        model_name=model_name,
        model_type=model_type,
        evaluation=evaluation,
        row_indices=row_indices,
        class_to_index=class_mapping,
        all_labels=all_labels,
        class_count_train=class_count_train,
        class_count_test=class_count_test,
        test_timestamps=test_timestamps,
        metadata=metadata,
    )


def run_matrix_transformer_finetune(
    *,
    model_name: str,
    frame: pd.DataFrame,
    config: FI2010BenchmarkConfig,
    split: SplitIndices,
    feature_columns: Sequence[str],
    all_labels: Sequence[Any],
    class_count_train: int,
    class_count_test: int,
    test_timestamps: Sequence[str] | None,
    settings: PaperNeuralSettings,
    pretrained_encoder_state: Mapping[str, Any] | None,
    init_source: str,
    model_type: str,
) -> NeuralPaperModelResult:
    """Fit a matrix transformer, optionally from a pretrained encoder.

    This is the shared supervised path used by both the SSL fine-tune (with a
    pretrained encoder) and the supervised baseline (random init). Routing both
    through one function guarantees identical preprocessing, windowing, splits,
    seeds and training for a fair comparison.
    """
    return _run_matrix_transformer(
        model_name=model_name,
        frame=frame,
        config=config,
        split=split,
        feature_columns=feature_columns,
        all_labels=all_labels,
        class_count_train=class_count_train,
        class_count_test=class_count_test,
        test_timestamps=test_timestamps,
        settings=settings,
        pretrained_encoder_state=pretrained_encoder_state,
        init_source=init_source,
        model_type=model_type,
    )


class _LabelledTokenDataset(_TorchDataset):
    """Token-window dataset wrapper that adds train-encoded labels."""

    def __init__(
        self,
        base_dataset: TokenSequenceDataset,
        frame: pd.DataFrame,
        *,
        label_name: str,
        class_to_index: Mapping[object, int],
    ) -> None:
        _require_torch()
        if not isinstance(base_dataset, TokenSequenceDataset):
            raise TypeError("base_dataset must be a TokenSequenceDataset")
        self._base_dataset = base_dataset
        self._row_indices: list[int] = []
        self._targets: list[int] = []
        mapping = {
            _normalise_class_value(label): int(index)
            for label, index in class_to_index.items()
        }
        for index in base_dataset.window_indices:
            source_row = base_dataset.sequence.records[
                index.anchor_index
            ].source_record_index
            if source_row is None:
                raise ValueError("token window anchor is not tied to a source row")
            label_value = _normalise_class_value(frame.iloc[source_row][label_name])
            if label_value not in mapping:
                raise ValueError(
                    f"label value {label_value!r} at row {source_row} was not "
                    "observed in the training label encoder"
                )
            self._row_indices.append(int(source_row))
            self._targets.append(int(mapping[label_value]))

    def __len__(self) -> int:
        return len(self._base_dataset)

    def __getitem__(self, item: int) -> dict[str, Any]:
        torch_module = _require_torch()
        sample = dict(self._base_dataset[item])
        sample["y"] = torch_module.tensor(
            self._targets[item],
            dtype=torch_module.long,
        )
        sample["target_index"] = int(self._row_indices[item])
        return sample

    @property
    def row_indices(self) -> list[int]:
        return list(self._row_indices)


def _collate_labelled_token_windows(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    torch_module = _require_torch()
    base = collate_token_windows(samples)
    base["y"] = torch_module.stack([sample["y"] for sample in samples], dim=0)
    base["target_index"] = torch_module.tensor(
        [int(sample["target_index"]) for sample in samples],
        dtype=torch_module.long,
    )
    return base


def _partition_by_row(split: SplitIndices) -> dict[int, str]:
    output: dict[int, str] = {}
    for name, indices in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        for row_index in indices:
            output[int(row_index)] = name
    return output


def _last_token_index_by_source_row(
    sequence: Any,
) -> dict[int, int]:
    anchors: dict[int, int] = {}
    for token_index, record in enumerate(sequence.records):
        if record.source_record_index is None:
            continue
        anchors[int(record.source_record_index)] = int(token_index)
    return anchors


def _token_split_ids(sequence: Any, split: SplitIndices) -> list[str | None]:
    partitions = _partition_by_row(split)
    split_ids: list[str | None] = []
    for record in sequence.records:
        if record.source_record_index is None:
            split_ids.append(None)
        else:
            split_ids.append(partitions[int(record.source_record_index)])
    return split_ids


def _select_token_windows_for_rows(
    *,
    all_windows: Sequence[TokenWindowIndex],
    anchors_by_row: Mapping[int, int],
    rows: Sequence[int],
    partition_name: str,
    required: bool,
) -> tuple[list[TokenWindowIndex] | None, str | None]:
    anchor_set = {
        anchors_by_row[int(row)]
        for row in rows
        if int(row) in anchors_by_row
    }
    selected = [window for window in all_windows if window.anchor_index in anchor_set]
    if selected:
        return selected, None
    reason = f"{partition_name} split has no usable token windows"
    if required:
        raise ValueError(reason)
    return None, reason


def _make_labelled_token_dataset(
    *,
    sequence: Any,
    window_config: TokenWindowConfig,
    windows: Sequence[TokenWindowIndex],
    frame: pd.DataFrame,
    config: FI2010BenchmarkConfig,
    class_to_index: Mapping[object, int],
) -> _LabelledTokenDataset:
    base_dataset = TokenSequenceDataset(
        sequence,
        window_config,
        window_indices=list(windows),
    )
    return _LabelledTokenDataset(
        base_dataset,
        frame,
        label_name=config.label_name,
        class_to_index=class_to_index,
    )


def _make_token_loader(dataset: _LabelledTokenDataset, *, batch_size: int) -> Any:
    _require_torch()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_collate_labelled_token_windows,
    )


def _load_fi2010_snapshots(
    *,
    data_path: Path,
    config: FI2010BenchmarkConfig,
) -> list[Any]:
    from chronoslob.data.fi2010 import FI2010Config, load_fi2010

    dataset = load_fi2010(
        FI2010Config(
            path=data_path,
            timestamp_column=config.timestamp_column,
            split_column=config.split_column,
            label_columns=list(config.label_columns),
            price_level_count=config.price_level_count,
        )
    )
    snapshots = dataset.to_snapshot_rows()
    if len(snapshots) != dataset.n_rows:
        raise ValueError(
            "FI-2010 snapshot conversion did not preserve one snapshot per row"
        )
    return snapshots


def _run_transformer(
    *,
    frame: pd.DataFrame,
    config: FI2010BenchmarkConfig,
    data_path: Path,
    split: SplitIndices,
    all_labels: Sequence[Any],
    class_count_train: int,
    class_count_test: int,
    test_timestamps: Sequence[str] | None,
    settings: PaperNeuralSettings,
) -> NeuralPaperModelResult:
    _require_torch()
    class_mapping = _train_class_mapping(frame, config=config, split=split)
    snapshots = _load_fi2010_snapshots(data_path=data_path, config=config)
    token_config = TokenisationConfig(
        max_levels_per_side=settings.transformer_max_levels_per_side,
        fit_source_tokens=True,
        sort_records=True,
        include_bos=False,
        include_eos=False,
    )
    fit_state = fit_tokenisation_state(
        snapshots,
        token_config,
        split_indices=split,
    )
    sequence = tokenise_records(snapshots, fit_state.vocabulary, fit_state.config)
    window_config = TokenWindowConfig(
        window_length=settings.transformer_window_length,
        stride=1,
        drop_incomplete=False,
        respect_symbol_boundaries=True,
        respect_split_boundaries=True,
        padding_side="left",
    )
    split_ids = _token_split_ids(sequence, split)
    all_windows = build_token_window_indices(
        sequence,
        window_config,
        split_ids=split_ids,
    )
    anchors_by_row = _last_token_index_by_source_row(sequence)
    train_windows, _ = _select_token_windows_for_rows(
        all_windows=all_windows,
        anchors_by_row=anchors_by_row,
        rows=split.train,
        partition_name="train",
        required=True,
    )
    validation_windows, validation_skip = _select_token_windows_for_rows(
        all_windows=all_windows,
        anchors_by_row=anchors_by_row,
        rows=split.validation,
        partition_name="validation",
        required=False,
    )
    test_windows, _ = _select_token_windows_for_rows(
        all_windows=all_windows,
        anchors_by_row=anchors_by_row,
        rows=split.test,
        partition_name="test",
        required=True,
    )
    assert train_windows is not None
    assert test_windows is not None

    train_dataset = _make_labelled_token_dataset(
        sequence=sequence,
        window_config=window_config,
        windows=train_windows,
        frame=frame,
        config=config,
        class_to_index=class_mapping,
    )
    validation_dataset: _LabelledTokenDataset | None = None
    if validation_windows is not None:
        try:
            validation_dataset = _make_labelled_token_dataset(
                sequence=sequence,
                window_config=window_config,
                windows=validation_windows,
                frame=frame,
                config=config,
                class_to_index=class_mapping,
            )
        except ValueError as exc:
            validation_skip = f"validation token dataset could not be built: {exc}"
    test_dataset = _make_labelled_token_dataset(
        sequence=sequence,
        window_config=window_config,
        windows=test_windows,
        frame=frame,
        config=config,
        class_to_index=class_mapping,
    )

    model_config = MarketTransformerConfig(
        vocab_sizes=sequence.field_sizes,
        field_embedding_dim=settings.transformer_field_embedding_dim,
        model_dim=settings.transformer_model_dim,
        num_heads=settings.transformer_num_heads,
        num_layers=settings.transformer_num_layers,
        feedforward_dim=settings.transformer_feedforward_dim,
        dropout=settings.dropout,
        max_sequence_length=settings.transformer_window_length,
        num_classes=len(class_mapping),
        pooling="mean",
        activation="gelu",
        use_layer_norm=True,
    )
    device_resolution = resolve_neural_device(settings.device)
    checkpoint_path = (
        Path(settings.checkpoint_path)
        if settings.checkpoint_enabled and settings.checkpoint_path is not None
        else None
    )
    training_config = TransformerTrainingConfig(
        epochs=settings.max_epochs,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        gradient_clip_norm=settings.gradient_clip_norm,
        device=device_resolution.resolved,
        seed=config.seed,
        early_stopping_patience=settings.early_stopping_patience,
        early_stopping_metric=settings.early_stopping_metric,
        checkpoint_path=checkpoint_path,
    )
    if settings.deterministic:
        set_torch_deterministic(config.seed)
    model = create_market_transformer(model_config)
    train_loader = _make_token_loader(train_dataset, batch_size=settings.batch_size)
    validation_loader = (
        _make_token_loader(validation_dataset, batch_size=settings.batch_size)
        if validation_dataset is not None
        else None
    )
    test_loader = _make_token_loader(test_dataset, batch_size=settings.batch_size)
    training_started = time.perf_counter()
    history = fit_transformer_classifier(
        model,
        train_loader,
        validation_loader,
        training_config,
    )
    training_seconds = time.perf_counter() - training_started
    evaluation = evaluate_transformer_classifier(
        model,
        test_loader,
        device=device_resolution.resolved,
        labels=sorted(class_mapping.values()),
    )
    metadata: dict[str, Any] = {
        "neural_family": "transformer",
        "sample_counts": {
            "train": len(train_dataset),
            "validation": 0
            if validation_dataset is None
            else len(validation_dataset),
            "test": len(test_dataset),
        },
        "window_policy": {
            "window_length": int(settings.transformer_window_length),
            "respect_split_boundaries": True,
            "windows_stay_inside_split": True,
        },
        "validation_skip_reason": validation_skip,
        "token_window_length": int(settings.transformer_window_length),
        "tokenised_record_count": len(sequence.records),
        "fitted_source_tokens": list(fit_state.fitted_source_tokens),
        "training_history": _history_rows(history),
        "max_epochs": int(settings.max_epochs),
        "epochs_ran": len(history),
        "best_epoch": _best_epoch(history),
        "early_stopped": _history_early_stopped(history),
        "training_seconds": float(training_seconds),
        "early_stopping": {
            "metric": settings.early_stopping_metric,
            "patience": settings.early_stopping_patience,
        },
        "device": {
            "requested": device_resolution.requested,
            "resolved": device_resolution.resolved,
            "cuda_available": device_resolution.cuda_available,
        },
        "deterministic_seed": {
            "enabled": bool(settings.deterministic),
            "seed": int(config.seed),
        },
        "model_parameter_count": int(model.n_parameters()),
        "parameter_count": int(model.n_parameters()),
        "checkpoint": {
            "enabled": bool(settings.checkpoint_enabled),
            "path": None if checkpoint_path is None else str(checkpoint_path),
            "written": bool(checkpoint_path is not None and checkpoint_path.is_file()),
        },
        "class_mapping": {str(label): int(index) for label, index in class_mapping.items()},
        "tokenisation_fit_split": "train",
    }
    return _result_from_evaluation(
        model_name="transformer",
        model_type="market_transformer",
        evaluation=evaluation,
        row_indices=test_dataset.row_indices,
        class_to_index=class_mapping,
        all_labels=all_labels,
        class_count_train=class_count_train,
        class_count_test=class_count_test,
        test_timestamps=test_timestamps,
        metadata=metadata,
    )


def run_neural_paper_model(
    *,
    model_name: str,
    frame: pd.DataFrame,
    config: FI2010BenchmarkConfig,
    data_path: Path,
    split: SplitIndices,
    feature_columns: Sequence[str],
    all_labels: Sequence[Any],
    class_count_train: int,
    class_count_test: int,
    test_timestamps: Sequence[str] | None,
    settings: PaperNeuralSettings,
) -> NeuralPaperModelResult:
    """Fit one supported neural paper-runner model and return test artefacts."""
    cleaned = model_name.strip().lower()
    if cleaned == "deeplob_style":
        return _run_deeplob_style(
            frame=frame,
            config=config,
            split=split,
            feature_columns=feature_columns,
            all_labels=all_labels,
            class_count_train=class_count_train,
            class_count_test=class_count_test,
            test_timestamps=test_timestamps,
            settings=settings,
        )
    if cleaned in {"transformer", "matrix_transformer"}:
        return _run_matrix_transformer(
            model_name=cleaned,
            frame=frame,
            config=config,
            split=split,
            feature_columns=feature_columns,
            all_labels=all_labels,
            class_count_train=class_count_train,
            class_count_test=class_count_test,
            test_timestamps=test_timestamps,
            settings=settings,
        )
    raise ValueError(
        f"unsupported neural paper model {model_name!r}; supported: "
        "['deeplob_style', 'transformer', 'matrix_transformer']"
    )
