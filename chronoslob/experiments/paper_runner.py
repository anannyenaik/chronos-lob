"""Paper experiment runner for the FI-2010 mid-price direction task.

This module turns a user-supplied local FI-2010-style file into a
validated experiment artefact directory. It composes the existing
benchmark preparation logic, baseline infrastructure and
artefact contract so that one CLI invocation produces a complete,
validated experiment record covering classical and neural baselines.

The runner is deliberately scoped:

* It runs the paper model registry (see
  :mod:`chronoslob.experiments.model_registry`) including classical
  baselines plus traceable DeepLOB-style and transformer neural paths.
* It writes only the required artefacts plus row-level predictions and
  a confusion-matrix artefact.
* It never downloads data, never fits preprocessing or model-selection
  choices on validation or test data and never performs network calls.

The benchmark suite is the predictive-quality evidence stream.
Calibration evidence (reliability bins, ECE recomputation
from stored predictions), execution-sensitivity evidence and plot
generation remain tracked under later phases.
"""

from __future__ import annotations

import platform
import shutil
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator

from chronoslob import __version__
from chronoslob.experiments.artifacts import (
    validate_experiment_directory,
    write_json_model,
)
from chronoslob.experiments.evidence import (
    CalibrationBinRow,
    CalibrationSummary,
    ExecutionRow,
    ExecutionSummary,
    build_calibration_bins,
    build_execution_sensitivity,
    build_return_proxy_series,
    summarise_calibration,
    summarise_execution,
    write_calibration_bins_csv,
    write_execution_sensitivity_csv,
)
from chronoslob.experiments.fi2010_benchmark import (
    FI2010BenchmarkConfig,
    FI2010PreparationResult,
    load_benchmark_config,
    prepare_fi2010_benchmark,
)
from chronoslob.experiments.manifests import stable_json_dumps
from chronoslob.experiments.model_registry import (
    REQUIRED_PAPER_MODELS,
    SUPPORTED_PAPER_MODELS,
    PaperModelSpec,
    build_paper_baseline_config,
    get_paper_model_spec,
    normalise_paper_model_names,
)
from chronoslob.experiments.neural_adapters import run_neural_paper_model
from chronoslob.experiments.plots import (
    PaperPlotSummary,
    build_paper_experiment_plots,
)
from chronoslob.experiments.schemas import (
    EvidenceStreams,
    ExperimentConfigSummary,
    ExperimentResults,
    ExperimentValidationReport,
    ModelResult,
)
from chronoslob.models.preprocessing import (
    TrainOnlyStandardScaler,
    build_feature_matrix,
    build_target_vector,
    select_feature_columns,
)
from chronoslob.training.experiment import get_git_commit
from chronoslob.training.metrics import (
    compute_classification_metrics,
    confusion_matrix_as_dict,
)
from chronoslob.training.splitters import (
    SplitIndices,
    TemporalSplitConfig,
    temporal_train_validation_test_split,
)

__all__ = [
    "PAPER_RUNNER_VERSION",
    "REQUIRED_PAPER_MODELS",
    "SUPPORTED_PAPER_MODELS",
    "PaperExperimentSummary",
    "PaperModelOutcome",
    "PaperModelSkip",
    "run_paper_experiment",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)

PAPER_RUNNER_VERSION = "phase-e/paper-experiment-runner/v1"

_FIXTURE_PATH_MARKERS = ("tests", "fixtures")

_PREDICTIVE_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "matthews_corrcoef",
        "n_samples",
        "class_count_train",
        "class_count_test",
    }
)
_CALIBRATION_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "brier_score",
        "log_loss",
        "expected_calibration_error",
        "mean_confidence",
    }
)


class PaperModelOutcome(BaseModel):
    """Per-model evaluation outcome produced by the paper runner."""

    model_config = _MODEL_CONFIG

    model_name: str
    model_type: str
    split: str
    horizon: int
    metrics: dict[str, float]
    confusion_matrix: dict[str, Any]
    n_test_rows: int
    emits_probabilities: bool
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model_name", "model_type", "split")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("paper model outcome strings must be non-empty")
        return value.strip()

    @field_validator("horizon", "n_test_rows")
    @classmethod
    def _validate_non_negative_int(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("integer fields must be int")
        if value < 0:
            raise ValueError("integer fields must be non-negative")
        return value


class PaperModelSkip(BaseModel):
    """Record describing a requested model that could not be run."""

    model_config = _MODEL_CONFIG

    model_name: str
    reason: str

    @field_validator("model_name", "reason")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("skipped-model strings must be non-empty")
        return value.strip()


class PaperExperimentSummary(BaseModel):
    """Top-level summary returned by :func:`run_paper_experiment`."""

    model_config = _MODEL_CONFIG

    experiment_name: str
    task_name: str
    horizon: int
    split_name: str
    data_path: str
    output_dir: str
    requested_models: list[str]
    models_run: list[str]
    skipped_models: list[PaperModelSkip] = Field(default_factory=list)
    metric_names: list[str]
    predictive_metric_names: list[str]
    calibration_metric_names: list[str]
    execution_metric_names: list[str] = Field(default_factory=list)
    calibration_models: list[str] = Field(default_factory=list)
    execution_models: list[str] = Field(default_factory=list)
    artefacts: dict[str, str]
    is_fixture: bool
    runner_version: str
    created_at: datetime
    validation: ExperimentValidationReport
    outcomes: list[PaperModelOutcome] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    plot_summary: PaperPlotSummary | None = None

    @field_validator(
        "experiment_name",
        "task_name",
        "split_name",
        "data_path",
        "output_dir",
        "runner_version",
    )
    @classmethod
    def _validate_required_string(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("summary field must be a non-empty string")
        return value.strip()

    @field_validator("horizon")
    @classmethod
    def _validate_horizon(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("horizon must be an integer")
        if value <= 0:
            raise ValueError("horizon must be positive")
        return value


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_fixture_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return all(marker in parts for marker in _FIXTURE_PATH_MARKERS)


def _resolve_code_commit() -> str | None:
    return get_git_commit()


def _build_combined_frame(
    preparation: FI2010PreparationResult,
    *,
    config: FI2010BenchmarkConfig,
    data_path: Path,
) -> pd.DataFrame:
    """Reload the local FI-2010 file directly so that feature/label rows align.

    The preparation step already validated the file. We re-read here to
    build feature and label matrices for the model run, keeping the
    loader configuration identical to preparation.
    """
    from chronoslob.data.fi2010 import FI2010Config, load_fi2010

    fi2010_config = FI2010Config(
        path=data_path,
        timestamp_column=config.timestamp_column,
        split_column=config.split_column,
        label_columns=list(config.label_columns),
        price_level_count=config.price_level_count,
    )
    dataset = load_fi2010(fi2010_config)
    frame = dataset.frame.copy()
    if len(frame) != preparation.split_summary.n_rows:
        raise ValueError(
            "FI-2010 row count changed between preparation and runner load",
        )
    return frame


def _feature_columns_from_frame(
    frame: pd.DataFrame,
    *,
    label_columns: Sequence[str],
    extra_exclude: Sequence[str],
) -> list[str]:
    exclude: set[str] = set(label_columns)
    exclude.update(extra_exclude)
    candidate = frame.drop(columns=[col for col in exclude if col in frame.columns])
    return select_feature_columns(candidate, reject_label_like=True)


def _safe_prepare_in_subdir(
    *,
    config: FI2010BenchmarkConfig,
    config_source_path: Path,
    data_path: Path,
    out_dir: Path,
) -> tuple[FI2010PreparationResult, Path]:
    preparation_dir = out_dir / "preparation"
    preparation_dir.mkdir(parents=True, exist_ok=True)
    result = prepare_fi2010_benchmark(
        config,
        data_path=data_path,
        output_dir=preparation_dir,
        config_source_path=config_source_path,
    )
    return result, preparation_dir


def _build_split(
    *,
    n_rows: int,
    config: FI2010BenchmarkConfig,
) -> SplitIndices:
    split_config = TemporalSplitConfig(
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        min_train_size=1,
        min_validation_size=1,
        min_test_size=0,
    )
    return temporal_train_validation_test_split(n_rows, split_config)


def _select_metric_subset(metric_dict: Mapping[str, Any]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for key, value in metric_dict.items():
        if value is None:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            cleaned[key] = float(value)
    return cleaned


def _model_known_classes(model: Any) -> list[Any] | None:
    if hasattr(model, "classes_"):
        try:
            classes = list(model.classes_)
        except (AttributeError, TypeError, ValueError):
            return None
        return classes
    estimator = getattr(model, "estimator", None)
    if estimator is not None and hasattr(estimator, "classes_"):
        try:
            return list(estimator.classes_)
        except (AttributeError, TypeError, ValueError):
            return None
    return None


def _align_probabilities_to_labels(
    probabilities: np.ndarray,
    *,
    model_classes: Sequence[Any] | None,
    target_labels: Sequence[Any],
) -> np.ndarray:
    """Project per-row probabilities onto ``target_labels``.

    Unknown target labels receive zero probability for that row. This
    keeps the predictions CSV schema stable across models that have only
    observed a subset of the full label set during training.
    """
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    n_rows = matrix.shape[0]
    aligned = np.zeros((n_rows, len(target_labels)), dtype=float)
    if model_classes is None:
        if matrix.shape[1] == len(target_labels):
            return matrix
        return aligned
    class_to_column = {value: position for position, value in enumerate(model_classes)}
    for column_position, target_label in enumerate(target_labels):
        source_column = class_to_column.get(target_label)
        if source_column is None:
            continue
        if source_column >= matrix.shape[1]:
            continue
        aligned[:, column_position] = matrix[:, source_column]
    return aligned


def _predictions_for_split(
    *,
    model: Any,
    x: np.ndarray,
    y_true: np.ndarray,
    row_indices: Sequence[int],
    split_name: str,
    model_name: str,
    labels: Sequence[Any],
    timestamps: Sequence[str] | None,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray | None]:
    predictions = np.asarray(model.predict(x))
    proba = model.predict_proba(x)
    label_order = [str(label) for label in labels]
    model_classes = _model_known_classes(model)
    aligned_probabilities: np.ndarray | None = None
    if proba is not None:
        aligned_probabilities = _align_probabilities_to_labels(
            np.asarray(proba),
            model_classes=model_classes,
            target_labels=labels,
        )
    rows: list[dict[str, Any]] = []
    for position, row_index in enumerate(row_indices):
        row: dict[str, Any] = {
            "row_index": int(row_index),
            "split": split_name,
            "model_name": model_name,
            "label": _to_json_value(y_true[position]),
            "prediction": _to_json_value(predictions[position]),
        }
        if timestamps is not None:
            row["timestamp"] = timestamps[position]
        if aligned_probabilities is not None:
            row_probabilities = aligned_probabilities[position]
            for class_position, class_label in enumerate(label_order):
                value = float(row_probabilities[class_position])
                row[f"probability_{class_label}"] = value
            row["confidence"] = float(np.max(row_probabilities))
        rows.append(row)
    return rows, predictions, aligned_probabilities


def _to_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _timestamps_for_indices(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None,
    indices: Sequence[int],
) -> list[str] | None:
    if timestamp_column is None or timestamp_column not in frame.columns:
        return None
    series = pd.to_datetime(
        frame.iloc[list(indices)][timestamp_column],
        utc=True,
        errors="raise",
    )
    return [ts.isoformat() for ts in series]


def _write_predictions_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("predictions must contain at least one row")
    frame = pd.DataFrame(list(rows))
    ordered_columns = [
        column
        for column in (
            "row_index",
            "split",
            "timestamp",
            "label",
            "prediction",
            "model_name",
            "confidence",
        )
        if column in frame.columns
    ]
    probability_columns = sorted(
        [column for column in frame.columns if column.startswith("probability_")]
    )
    remaining = [
        column
        for column in frame.columns
        if column not in ordered_columns and column not in probability_columns
    ]
    final_columns = ordered_columns + probability_columns + remaining
    frame = frame.loc[:, final_columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _expected_calibration_error(
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: Sequence[Any],
    n_bins: int = 10,
) -> float:
    """Compute equal-width expected calibration error.

    The bin width is fixed at ``1 / n_bins`` over the unit interval and
    bin assignment is performed on the per-row top-class probability.
    """
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be 2D for ECE")
    confidences = probabilities.max(axis=1)
    predicted_indices = probabilities.argmax(axis=1)
    label_array = np.asarray(list(labels))
    predicted_labels = label_array[predicted_indices]
    accuracies = (predicted_labels == np.asarray(y_true)).astype(float)
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


def _confusion_matrix_payload(
    outcomes: Sequence[PaperModelOutcome],
) -> dict[str, Any]:
    return {
        "models": [
            {
                "model_name": outcome.model_name,
                "model_type": outcome.model_type,
                "split": outcome.split,
                "horizon": outcome.horizon,
                "labels": outcome.confusion_matrix["labels"],
                "matrix": outcome.confusion_matrix["matrix"],
            }
            for outcome in outcomes
        ]
    }


def _resolve_execution_direction_map(
    *,
    config: FI2010BenchmarkConfig,
    all_labels: Sequence[Any],
) -> dict[str, int]:
    """Resolve the predicted-class to direction-sign mapping for execution.

    A config-provided map wins; otherwise the FI-2010 convention is used
    as a sensible default for integer-coded labels.
    """
    from chronoslob.experiments.evidence import _default_direction_map

    label_strings = [str(label) for label in all_labels]
    config_map = config.execution_sensitivity.class_direction_map
    if config_map is not None:
        return {str(key): int(value) for key, value in config_map.items()}
    return _default_direction_map(label_strings)


def _build_environment_payload() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_version": __version__,
    }


def _write_model_card(
    *,
    path: Path,
    config: FI2010BenchmarkConfig,
    outcomes: Sequence[PaperModelOutcome],
    requested_models: Sequence[str],
    skipped_models: Sequence[PaperModelSkip],
    data_path: Path,
    data_source_kind: str,
    data_sha256: str | None,
    is_fixture: bool,
    seed: int,
    metric_names: Sequence[str],
    predictive_metric_names: Sequence[str],
    calibration_metric_names: Sequence[str],
    execution_metric_names: Sequence[str],
    artefacts: Mapping[str, str],
    code_commit: str | None,
    split_counts: Mapping[str, int],
    neural_settings: Mapping[str, Any],
    calibration_models: Sequence[str],
    execution_models: Sequence[str],
    calibration_warning: str | None,
    execution_warning: str | None,
) -> None:
    lines: list[str] = []
    lines.append(f"# Model Card: {config.experiment_name}")
    lines.append("")
    if is_fixture:
        lines.append(
            "Status: synthetic fixture smoke run of the paper "
            "benchmark suite. This artefact set exercises the paper "
            "experiment runner on a tiny synthetic fixture and is not "
            "benchmark evidence, market evidence or execution evidence.",
        )
    else:
        lines.append(
            "Status: locally executed paper experiment run of the "
            "benchmark suite. Inspect the data manifest, "
            "split summary and limitations before treating the metrics "
            "as benchmark evidence.",
        )
    lines.append("")
    lines.append("## Experiment")
    lines.append("")
    lines.append(f"- name: `{config.experiment_name}`")
    lines.append(f"- task: `{config.task_name}`")
    lines.append(f"- horizon: {config.horizon}")
    lines.append(f"- label column: `{config.label_name}`")
    lines.append(f"- split: `{config.split_name}` (temporal train/validation/test)")
    lines.append(f"- seed: {seed}")
    if code_commit:
        lines.append(f"- code commit: `{code_commit}`")
    lines.append(f"- runner version: `{PAPER_RUNNER_VERSION}`")
    lines.append("")
    lines.append("## Data")
    lines.append("")
    lines.append(f"- dataset: `{config.dataset_name}`")
    lines.append(f"- data source kind: `{data_source_kind}`")
    lines.append(f"- local source path: `{data_path}`")
    if data_sha256 is not None:
        lines.append(f"- source SHA-256: `{data_sha256}`")
    if is_fixture:
        lines.append(
            "- fixture flag: this path lives under `tests/fixtures/` and "
            "is a synthetic FI-2010-like file used only for plumbing "
            "checks.",
        )
    lines.append("")
    lines.append("## Split Design")
    lines.append("")
    total_rows = int(split_counts.get("n_rows", 0))
    train_rows = int(split_counts.get("n_train", 0))
    validation_rows = int(split_counts.get("n_validation", 0))
    test_rows = int(split_counts.get("n_test", 0))
    lines.append(f"- total rows loaded: {total_rows}")
    lines.append(f"- train rows: {train_rows}")
    lines.append(f"- validation rows: {validation_rows}")
    lines.append(f"- test rows: {test_rows}")
    lines.append(
        "- split is constructed by the deterministic temporal splitter; "
        "no shuffling, no stratification and no test-row use during "
        "preprocessing or model fitting.",
    )
    lines.append("")
    lines.append("## Models")
    lines.append("")
    lines.append("- requested:")
    for name in requested_models:
        lines.append(f"  - `{name}`")
    lines.append("- successfully run:")
    if outcomes:
        for outcome in outcomes:
            lines.append(
                f"  - `{outcome.model_name}` (type `{outcome.model_type}`) "
                f"on the `{outcome.split}` split with {outcome.n_test_rows} "
                "test rows",
            )
    else:
        lines.append("  - none")
    lines.append("- skipped:")
    if skipped_models:
        for skip in skipped_models:
            lines.append(f"  - `{skip.model_name}`: {skip.reason}")
    else:
        lines.append("  - none")
    lines.append("")
    lines.append("## Neural Settings")
    lines.append("")
    neural_keys = (
        "supported_models",
        "lookback",
        "transformer_window_length",
        "batch_size",
        "max_epochs",
        "learning_rate",
        "weight_decay",
        "gradient_clip_norm",
        "device",
        "deterministic",
        "dropout",
        "deeplob_conv_channels",
        "deeplob_lstm_hidden_size",
        "deeplob_use_batch_norm",
        "transformer_field_embedding_dim",
        "transformer_model_dim",
        "transformer_num_heads",
        "transformer_num_layers",
        "transformer_feedforward_dim",
        "transformer_max_levels_per_side",
    )
    for key in neural_keys:
        if key in neural_settings:
            value = neural_settings[key]
            lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Metric Groups")
    lines.append("")
    lines.append("- predictive metrics emitted:")
    if predictive_metric_names:
        for name in predictive_metric_names:
            lines.append(f"  - {name}")
    else:
        lines.append("  - none")
    lines.append("- calibration metrics emitted:")
    if calibration_metric_names:
        for name in calibration_metric_names:
            lines.append(f"  - {name}")
    else:
        lines.append("  - none (no model emitted probabilities on this run)")
    lines.append("- execution-aware sensitivity metrics emitted:")
    if execution_metric_names and execution_metric_names != ["not_computed"]:
        for name in execution_metric_names:
            lines.append(f"  - {name}")
    else:
        lines.append("  - none on this run")
    lines.append("- all emitted metrics:")
    for metric in metric_names:
        lines.append(f"  - {metric}")
    if not metric_names:
        lines.append("  - none")
    lines.append("")
    lines.append("## Artefacts")
    lines.append("")
    for key, relative_path in artefacts.items():
        lines.append(f"- `{relative_path}` ({key})")
    lines.append("")
    lines.append("## Calibration Evidence")
    lines.append("")
    if calibration_models:
        lines.append(
            "- reliability bins computed from held-out test predictions "
            "and written to `calibration_bins.csv`."
        )
        lines.append(
            "- models with calibration bins: "
            + ", ".join(f"`{name}`" for name in calibration_models)
        )
        lines.append(
            "- calibration artefacts are derived from stored prediction "
            "rows; no calibrator is fitted on test data."
        )
    elif calibration_warning is not None:
        lines.append(f"- not produced this run: {calibration_warning}")
    else:
        lines.append(
            "- no model with usable probability output was available, so "
            "no calibration bins were written."
        )
    lines.append("")
    lines.append("## Execution-Aware Sensitivity")
    lines.append("")
    if execution_models:
        lines.append(
            "- cost-aware signal sensitivity rows are computed from "
            "stored prediction rows under explicit simple assumptions "
            "and are written to `execution_sensitivity.csv`."
        )
        lines.append(
            "- models with sensitivity rows: "
            + ", ".join(f"`{name}`" for name in execution_models)
        )
        lines.append(
            "- this is a simplified proxy analysis, not a production "
            "backtest, and does not claim tradable profitability or "
            "live execution."
        )
    elif execution_warning is not None:
        lines.append(f"- not produced this run: {execution_warning}")
    else:
        lines.append(
            "- execution-aware sensitivity was not produced for this run."
        )
    lines.append("")
    lines.append("## Leakage Controls")
    lines.append("")
    lines.append(
        "- Train, validation and test indices come from the deterministic "
        "temporal splitter; no random or stratified shuffling is used.",
    )
    lines.append(
        "- Per-model train-only feature standardisation is applied for "
        "models that require it; standardisation statistics are never "
        "fit on validation or test rows.",
    )
    lines.append(
        "- No model-selection choice, calibrator, bucket boundary or "
        "threshold is fitted on validation or test rows in this phase.",
    )
    lines.append(
        "- Label, split and timestamp columns are excluded from the "
        "feature matrix.",
    )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- This phase supports `majority`, `logistic`, `ridge`, "
        "`elastic_net`, `random_forest`, `gradient_boosting`, "
        "`deeplob_style` and `transformer`. The DeepLOB-style path is "
        "not an exact external-paper reproduction.",
    )
    lines.append(
        "- Plot generation remains tracked under a later phase; this "
        "runner emits calibration and execution-sensitivity evidence "
        "as CSV artefacts only.",
    )
    lines.append(
        "- Reported numbers are run-specific and must not be interpreted "
        "as profitability, deployability or live-trading evidence.",
    )
    if is_fixture:
        lines.append(
            "- The data source is a synthetic fixture; metrics describe "
            "fixture plumbing only and are not benchmark evidence.",
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _copy_config(
    *,
    config: FI2010BenchmarkConfig,
    config_source_path: Path,
    destination: Path,
) -> None:
    if config_source_path.is_file():
        shutil.copyfile(config_source_path, destination)
        return
    destination.write_text(
        yaml.safe_dump(
            config.model_dump(mode="json"),
            sort_keys=True,
            allow_unicode=True,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )


def _build_results(
    *,
    config: FI2010BenchmarkConfig,
    outcomes: Sequence[PaperModelOutcome],
    created_at: datetime,
    code_commit: str | None,
    calibration_models: Sequence[str],
    execution_models: Sequence[str],
    calibration_emitted: bool,
    execution_emitted: bool,
) -> tuple[ExperimentResults, list[str], list[str], list[str], list[str]]:
    metric_names: list[str] = []
    seen: set[str] = set()
    for outcome in outcomes:
        for name in outcome.metrics:
            if name not in seen:
                metric_names.append(name)
                seen.add(name)
    if not metric_names:
        metric_names = ["accuracy", "macro_f1"]

    primary_metric = "macro_f1" if "macro_f1" in seen else metric_names[0]
    config_summary = ExperimentConfigSummary(
        experiment_name=config.experiment_name,
        task_name=config.task_name,
        horizon=config.horizon,
        split_name=config.split_name,
        seed=config.seed,
        model_names=[outcome.model_name for outcome in outcomes],
        primary_metric=primary_metric,
        created_at=created_at,
        code_commit=code_commit,
    )

    calibration_lookup = set(calibration_models)
    execution_lookup = set(execution_models)
    model_results: list[ModelResult] = []
    for outcome in outcomes:
        artefacts: dict[str, str] = {
            "predictions": "predictions.csv",
            "confusion_matrix": "confusion_matrix.json",
        }
        if outcome.model_name in calibration_lookup:
            artefacts["calibration_bins"] = "calibration_bins.csv"
        if outcome.model_name in execution_lookup:
            artefacts["execution_sensitivity"] = "execution_sensitivity.csv"
        model_results.append(
            ModelResult(
                model_name=outcome.model_name,
                split=outcome.split,
                horizon=outcome.horizon,
                metrics=outcome.metrics,
                artefacts=artefacts,
            )
        )

    predictive_metrics = [
        name for name in metric_names if name in _PREDICTIVE_METRIC_NAMES
    ]
    calibration_metrics = [
        name for name in metric_names if name in _CALIBRATION_METRIC_NAMES
    ]
    if not predictive_metrics:
        predictive_metrics = ["accuracy"]
    if not calibration_metrics:
        calibration_metrics = ["not_computed"]

    if calibration_emitted and "calibration_bins" not in calibration_metrics:
        calibration_metrics = [*calibration_metrics, "calibration_bins"]
    execution_metrics: list[str]
    if execution_emitted:
        execution_metrics = [
            "gross_signal_return_proxy",
            "net_signal_return_proxy",
            "turnover_proxy",
            "hit_rate_proxy",
        ]
    else:
        execution_metrics = ["not_computed"]

    evidence_streams = EvidenceStreams(
        predictive=predictive_metrics,
        calibration=calibration_metrics,
        execution=execution_metrics,
        robustness=[],
        systems=[],
    )

    results = ExperimentResults(
        experiment_name=config.experiment_name,
        task_name=config.task_name,
        created_at=created_at,
        config_summary=config_summary,
        model_results=model_results,
        evidence_streams=evidence_streams,
    )
    return (
        results,
        metric_names,
        predictive_metrics,
        calibration_metrics,
        execution_metrics,
    )


def _evaluate_model(
    *,
    spec: PaperModelSpec,
    seed: int,
    train_features_raw: np.ndarray,
    test_features_raw: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    all_labels: Sequence[Any],
    class_count_train: int,
    class_count_test: int,
    test_indices: Sequence[int],
    horizon: int,
    test_timestamps: Sequence[str] | None,
) -> tuple[PaperModelOutcome, list[dict[str, Any]]]:
    """Fit one classical model and produce its outcome and prediction rows."""
    baseline_config = build_paper_baseline_config(spec.name, seed=seed)
    # Late import keeps the baseline factory in one place.
    from chronoslob.models.baselines import create_baseline_model

    model = create_baseline_model(baseline_config)

    if spec.requires_standardisation:
        scaler = TrainOnlyStandardScaler()
        train_x = scaler.fit_transform(train_features_raw)
        inference_x = scaler.transform(test_features_raw)
    else:
        train_x = train_features_raw
        inference_x = test_features_raw

    model.fit(train_x, train_labels)
    predictions = np.asarray(model.predict(inference_x))
    proba_raw = model.predict_proba(inference_x)
    proba: np.ndarray | None = None
    if proba_raw is not None:
        proba = _align_probabilities_to_labels(
            np.asarray(proba_raw, dtype=float),
            model_classes=_model_known_classes(model),
            target_labels=all_labels,
        )

    metrics = compute_classification_metrics(
        y_true=test_labels.tolist(),
        y_pred=predictions.tolist(),
        y_proba=proba,
        labels=all_labels,
    )
    confusion = confusion_matrix_as_dict(
        y_true=test_labels.tolist(),
        y_pred=predictions.tolist(),
        labels=all_labels,
    )

    metric_subset = _select_metric_subset(metrics.to_dict())
    metric_subset["class_count_train"] = float(class_count_train)
    metric_subset["class_count_test"] = float(class_count_test)
    if proba is not None:
        confidences = proba.max(axis=1)
        if "brier_score" not in metric_subset:
            metric_subset["brier_score"] = _multiclass_brier_score(
                y_true=test_labels.tolist(),
                probabilities=proba,
                labels=all_labels,
            )
        metric_subset["mean_confidence"] = float(confidences.mean())
        metric_subset["expected_calibration_error"] = float(
            _expected_calibration_error(
                y_true=test_labels,
                probabilities=proba,
                labels=all_labels,
            )
        )

    rows, _, _ = _predictions_for_split(
        model=model,
        x=inference_x,
        y_true=test_labels,
        row_indices=test_indices,
        split_name="test",
        model_name=spec.name,
        labels=all_labels,
        timestamps=test_timestamps,
    )

    outcome = PaperModelOutcome(
        model_name=spec.name,
        model_type=spec.model_type,
        split="test",
        horizon=horizon,
        metrics=metric_subset,
        confusion_matrix=confusion,
        n_test_rows=len(test_labels),
        emits_probabilities=spec.emits_probabilities and proba is not None,
    )
    return outcome, rows


def _evaluate_neural_model(
    *,
    spec: PaperModelSpec,
    frame: pd.DataFrame,
    config: FI2010BenchmarkConfig,
    data_path: Path,
    split: SplitIndices,
    feature_columns: Sequence[str],
    all_labels: Sequence[Any],
    class_count_train: int,
    class_count_test: int,
    test_timestamps: Sequence[str] | None,
) -> tuple[PaperModelOutcome, list[dict[str, Any]]]:
    """Fit one neural paper model and produce its outcome and prediction rows."""
    result = run_neural_paper_model(
        model_name=spec.name,
        frame=frame,
        config=config,
        data_path=data_path,
        split=split,
        feature_columns=feature_columns,
        all_labels=all_labels,
        class_count_train=class_count_train,
        class_count_test=class_count_test,
        test_timestamps=test_timestamps,
        settings=config.neural_settings,
    )
    outcome = PaperModelOutcome(
        model_name=result.model_name,
        model_type=result.model_type,
        split="test",
        horizon=config.horizon,
        metrics=result.metrics,
        confusion_matrix=result.confusion_matrix,
        n_test_rows=result.n_test_rows,
        emits_probabilities=True,
        metadata=result.metadata,
    )
    return outcome, result.prediction_rows


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_paper_experiment(
    config_path: Path,
    data_path: Path,
    out_dir: Path,
    *,
    models: Sequence[str] | None = None,
    overwrite: bool = False,
    build_plots: bool = False,
) -> PaperExperimentSummary:
    """Run the paper experiment runner and write a validated artefact directory.

    Parameters
    ----------
    config_path:
        Path to a Phase B FI-2010 benchmark preparation config.
    data_path:
        Local FI-2010-style file path. Required; never downloaded.
    out_dir:
        Directory where experiment artefacts will be written.
    models:
        Optional sequence of model short names from
        :data:`SUPPORTED_PAPER_MODELS`. Defaults to ``("majority",)``.
        Names are case-folded and de-duplicated. The ``majority``
        baseline must always be present.
    overwrite:
        When ``False``, refuse to write into an existing directory that
        already contains artefacts. When ``True``, the target directory
        is replaced before writing.
    build_plots:
        When ``True``, deterministic plots are generated from the stored
        artefacts after evidence files have been written. Plot generation
        failures for optional inputs are recorded as warnings and do not
        invalidate the experiment.

    Returns
    -------
    PaperExperimentSummary
        Summary object with metrics, artefact paths and a validation
        report from :func:`validate_experiment_directory`.
    """
    resolved_config_path = Path(config_path)
    resolved_data_path = Path(data_path)
    resolved_out_dir = Path(out_dir)

    if not resolved_config_path.is_file():
        raise FileNotFoundError(
            f"paper experiment config not found: {resolved_config_path}",
        )
    if not resolved_data_path.exists():
        raise FileNotFoundError(
            f"local FI-2010 data path does not exist: {resolved_data_path}",
        )
    if not resolved_data_path.is_file():
        raise FileNotFoundError(
            f"local FI-2010 data path is not a regular file: {resolved_data_path}",
        )

    config = load_benchmark_config(resolved_config_path)
    requested_models = normalise_paper_model_names(models)
    created_at = datetime.now(UTC)

    if resolved_out_dir.exists():
        if not resolved_out_dir.is_dir():
            raise FileExistsError(
                f"output path exists and is not a directory: {resolved_out_dir}",
            )
        contains_anything = any(resolved_out_dir.iterdir())
        if contains_anything and not overwrite:
            raise FileExistsError(
                "refusing to write into a non-empty output directory; "
                "pass overwrite=True to replace it: "
                f"{resolved_out_dir}",
            )
        if contains_anything:
            shutil.rmtree(resolved_out_dir)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    preparation, _ = _safe_prepare_in_subdir(
        config=config,
        config_source_path=resolved_config_path,
        data_path=resolved_data_path,
        out_dir=resolved_out_dir,
    )

    combined_frame = _build_combined_frame(
        preparation,
        config=config,
        data_path=resolved_data_path,
    )

    label_columns = list(config.label_columns)
    extra_exclude: list[str] = []
    if config.timestamp_column is not None:
        extra_exclude.append(config.timestamp_column)
    if config.split_column is not None:
        extra_exclude.append(config.split_column)

    feature_columns = _feature_columns_from_frame(
        combined_frame,
        label_columns=label_columns,
        extra_exclude=extra_exclude,
    )
    if not feature_columns:
        raise ValueError("paper experiment requires at least one feature column")

    n_rows = len(combined_frame)
    split = _build_split(n_rows=n_rows, config=config)
    if split.n_train == 0:
        raise ValueError("temporal split produced no training rows")
    if split.n_test == 0:
        raise ValueError(
            "temporal split produced no test rows; "
            "the fixture or config is too small for evaluation",
        )

    train_features = build_feature_matrix(
        combined_frame,
        feature_columns=feature_columns,
        row_indices=split.train,
    )
    test_features = build_feature_matrix(
        combined_frame,
        feature_columns=feature_columns,
        row_indices=split.test,
    )

    train_target = build_target_vector(
        combined_frame,
        target_column=config.label_name,
        row_indices=split.train,
    )
    test_target = build_target_vector(
        combined_frame,
        target_column=config.label_name,
        row_indices=split.test,
    )
    all_target = build_target_vector(
        combined_frame,
        target_column=config.label_name,
    )

    test_timestamps = _timestamps_for_indices(
        combined_frame,
        timestamp_column=config.timestamp_column,
        indices=split.test,
    )

    all_labels = list(all_target.classes or [])
    class_count_train = len(train_target.classes or [])
    class_count_test = len(test_target.classes or [])

    outcomes: list[PaperModelOutcome] = []
    skipped: list[PaperModelSkip] = []
    prediction_rows: list[dict[str, Any]] = []

    for model_token in requested_models:
        spec = get_paper_model_spec(model_token)
        try:
            if spec.model_family == "classical":
                outcome, rows = _evaluate_model(
                    spec=spec,
                    seed=config.seed,
                    train_features_raw=train_features.x,
                    test_features_raw=test_features.x,
                    train_labels=train_target.y,
                    test_labels=test_target.y,
                    all_labels=all_labels,
                    class_count_train=class_count_train,
                    class_count_test=class_count_test,
                    test_indices=split.test,
                    horizon=config.horizon,
                    test_timestamps=test_timestamps,
                )
            elif spec.model_family == "neural":
                outcome, rows = _evaluate_neural_model(
                    spec=spec,
                    frame=combined_frame,
                    config=config,
                    data_path=resolved_data_path,
                    split=split,
                    feature_columns=feature_columns,
                    all_labels=all_labels,
                    class_count_train=class_count_train,
                    class_count_test=class_count_test,
                    test_timestamps=test_timestamps,
                )
            else:
                raise ValueError(
                    f"unsupported paper model family {spec.model_family!r}"
                )
        except (ImportError, ValueError, RuntimeError, TypeError) as exc:
            if spec.name in REQUIRED_PAPER_MODELS:
                raise
            skipped.append(
                PaperModelSkip(
                    model_name=spec.name,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        outcomes.append(outcome)
        prediction_rows.extend(rows)

    if not outcomes:
        raise RuntimeError(
            "all requested paper-runner models were skipped; "
            f"skipped: {[skip.model_name for skip in skipped]}",
        )

    # Move data_manifest.json out of preparation/ into the experiment root.
    manifest_source = preparation.output_dir / "data_manifest.json"
    manifest_destination = resolved_out_dir / "data_manifest.json"
    if not manifest_source.is_file():
        raise FileNotFoundError(
            f"expected preparation manifest is missing: {manifest_source}",
        )
    shutil.copyfile(manifest_source, manifest_destination)

    config_destination = resolved_out_dir / "config.yaml"
    _copy_config(
        config=config,
        config_source_path=resolved_config_path,
        destination=config_destination,
    )

    predictions_path = resolved_out_dir / "predictions.csv"
    _write_predictions_csv(prediction_rows, predictions_path)

    predictions_frame = pd.DataFrame(prediction_rows)

    calibration_bin_rows: list[CalibrationBinRow] = []
    calibration_summaries: list[CalibrationSummary] = []
    calibration_models: list[str] = []
    calibration_warning: str | None = None
    if config.calibration.enabled:
        for outcome in outcomes:
            if not outcome.emits_probabilities:
                continue
            model_rows = predictions_frame[
                predictions_frame["model_name"] == outcome.model_name
            ]
            if model_rows.empty or "confidence" not in model_rows.columns:
                continue
            bins = build_calibration_bins(
                model_rows,
                model_name=outcome.model_name,
                split=outcome.split,
                n_bins=config.calibration.n_bins,
            )
            if not bins:
                continue
            calibration_bin_rows.extend(bins)
            summary = summarise_calibration(bins)
            if summary is not None:
                calibration_summaries.append(summary)
            calibration_models.append(outcome.model_name)
        if not calibration_models:
            calibration_warning = (
                "no model in this run emitted usable probability output, "
                "so calibration_bins.csv was not written"
            )
    else:
        calibration_warning = "calibration.enabled is false in the config"

    calibration_emitted = bool(calibration_bin_rows)
    if calibration_emitted:
        write_calibration_bins_csv(
            calibration_bin_rows,
            resolved_out_dir / "calibration_bins.csv",
        )

    execution_rows: list[ExecutionRow] = []
    execution_summaries: list[ExecutionSummary] = []
    execution_models: list[str] = []
    execution_warning: str | None = None
    if config.execution_sensitivity.enabled:
        return_proxy = build_return_proxy_series(
            combined_frame,
            horizon=config.horizon,
            config=config.execution_sensitivity.return_proxy,
        )
        if return_proxy is None:
            execution_warning = (
                "execution-sensitivity skipped: forward mid-price proxy "
                "could not be built from the supplied frame"
            )
        else:
            direction_map = _resolve_execution_direction_map(
                config=config,
                all_labels=all_labels,
            )
            for outcome in outcomes:
                if not outcome.emits_probabilities:
                    continue
                model_rows = predictions_frame[
                    predictions_frame["model_name"] == outcome.model_name
                ]
                if model_rows.empty:
                    continue
                sensitivity_rows = build_execution_sensitivity(
                    model_rows,
                    model_name=outcome.model_name,
                    split=outcome.split,
                    return_proxy=return_proxy,
                    config=config.execution_sensitivity,
                    direction_map=direction_map,
                )
                if not sensitivity_rows:
                    continue
                execution_rows.extend(sensitivity_rows)
                execution_summaries.append(
                    summarise_execution(
                        sensitivity_rows,
                        model_name=outcome.model_name,
                        split=outcome.split,
                    )
                )
                execution_models.append(outcome.model_name)
            if not execution_models and execution_warning is None:
                execution_warning = (
                    "execution-sensitivity skipped: no model emitted "
                    "probabilities that could be combined with the "
                    "forward-return proxy on this run"
                )
            if (
                execution_models
                and execution_warning is None
                and all(
                    int(row.eligible_predictions) == 0 for row in execution_rows
                )
            ):
                execution_warning = (
                    "execution-sensitivity rows recorded but every "
                    "parameter combination has zero eligible predictions; "
                    "the forward-return proxy is undefined on this "
                    "split (typically because horizon exceeds the "
                    "available rows beyond the test window)"
                )
    else:
        execution_warning = (
            "execution_sensitivity.enabled is false in the config"
        )

    execution_emitted = bool(execution_rows)
    if execution_emitted:
        write_execution_sensitivity_csv(
            execution_rows,
            resolved_out_dir / "execution_sensitivity.csv",
        )

    code_commit = _resolve_code_commit()
    (
        results,
        metric_names,
        predictive_metric_names,
        calibration_metric_names,
        execution_metric_names,
    ) = _build_results(
        config=config,
        outcomes=outcomes,
        created_at=created_at,
        code_commit=code_commit,
        calibration_models=calibration_models,
        execution_models=execution_models,
        calibration_emitted=calibration_emitted,
        execution_emitted=execution_emitted,
    )
    results_path = resolved_out_dir / "results.json"
    write_json_model(results_path, results)

    confusion_path = resolved_out_dir / "confusion_matrix.json"
    confusion_path.write_text(
        stable_json_dumps(_confusion_matrix_payload(outcomes)),
        encoding="utf-8",
    )

    data_sha256 = preparation.data_manifest.source_sha256
    data_source_kind = preparation.data_manifest.source_kind.value

    artefacts: dict[str, str] = {
        "config": "config.yaml",
        "data_manifest": "data_manifest.json",
        "results": "results.json",
        "predictions": "predictions.csv",
        "model_card": "model_card.md",
        "confusion_matrix": "confusion_matrix.json",
        "runner_summary": "runner_summary.json",
    }
    if calibration_emitted:
        artefacts["calibration_bins"] = "calibration_bins.csv"
    if execution_emitted:
        artefacts["execution_sensitivity"] = "execution_sensitivity.csv"

    is_fixture = _is_fixture_path(resolved_data_path.resolve())
    split_counts = {
        "n_rows": n_rows,
        "n_train": split.n_train,
        "n_validation": split.n_validation,
        "n_test": split.n_test,
    }
    model_card_path = resolved_out_dir / "model_card.md"
    _write_model_card(
        path=model_card_path,
        config=config,
        outcomes=outcomes,
        requested_models=requested_models,
        skipped_models=skipped,
        data_path=resolved_data_path,
        data_source_kind=data_source_kind,
        data_sha256=data_sha256,
        is_fixture=is_fixture,
        seed=config.seed,
        metric_names=metric_names,
        predictive_metric_names=predictive_metric_names,
        calibration_metric_names=calibration_metric_names,
        execution_metric_names=execution_metric_names,
        artefacts=artefacts,
        code_commit=code_commit,
        split_counts=split_counts,
        neural_settings=config.neural_settings.model_dump(mode="json"),
        calibration_models=calibration_models,
        execution_models=execution_models,
        calibration_warning=calibration_warning,
        execution_warning=execution_warning,
    )

    evidence_summary: dict[str, Any] = {
        "calibration_artefact_written": calibration_emitted,
        "calibration_models": list(calibration_models),
        "execution_artefact_written": execution_emitted,
        "execution_models": list(execution_models),
        "calibration_warning": calibration_warning,
        "execution_warning": execution_warning,
        "calibration_summaries": [
            summary.model_dump() for summary in calibration_summaries
        ],
        "execution_summaries": [
            summary.model_dump() for summary in execution_summaries
        ],
        "calibration_config": config.calibration.model_dump(),
        "execution_sensitivity_config": config.execution_sensitivity.model_dump(),
    }

    plot_summary: PaperPlotSummary | None = None
    plot_warnings: list[str] = []
    if build_plots:
        try:
            plot_summary = build_paper_experiment_plots(
                resolved_out_dir,
                overwrite=True,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            plot_warnings.append(
                f"plot generation failed: {type(exc).__name__}: {exc}"
            )
        if plot_summary is not None:
            plot_warnings.extend(plot_summary.warnings)
            if plot_summary.plots_written:
                artefacts["plots"] = "plots/"

    runner_summary_payload: dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "task_name": config.task_name,
        "horizon": config.horizon,
        "split_name": config.split_name,
        "data_path": str(resolved_data_path),
        "output_dir": str(resolved_out_dir),
        "requested_models": list(requested_models),
        "models_run": [outcome.model_name for outcome in outcomes],
        "skipped_models": [skip.model_dump() for skip in skipped],
        "metric_names": metric_names,
        "predictive_metric_names": predictive_metric_names,
        "calibration_metric_names": calibration_metric_names,
        "execution_metric_names": execution_metric_names,
        "runner_version": PAPER_RUNNER_VERSION,
        "created_at": created_at.isoformat(),
        "is_fixture": is_fixture,
        "data_source_kind": data_source_kind,
        "environment": _build_environment_payload(),
        "split_counts": split_counts,
        "neural_settings": config.neural_settings.model_dump(mode="json"),
        "evidence": evidence_summary,
        "model_metadata": {
            outcome.model_name: outcome.metadata
            for outcome in outcomes
            if outcome.metadata
        },
        "plots": {
            "requested": bool(build_plots),
            "plots_written": (
                list(plot_summary.plots_written) if plot_summary is not None else []
            ),
            "plots_skipped": (
                list(plot_summary.plots_skipped) if plot_summary is not None else []
            ),
            "warnings": list(plot_warnings),
            "builder_version": (
                plot_summary.builder_version if plot_summary is not None else None
            ),
        },
    }
    (resolved_out_dir / "runner_summary.json").write_text(
        stable_json_dumps(runner_summary_payload),
        encoding="utf-8",
    )

    validation = validate_experiment_directory(resolved_out_dir, include_plots=True)
    if validation.missing_required:
        missing = ", ".join(validation.missing_required)
        raise RuntimeError(
            f"paper experiment artefacts failed validation; missing: {missing}",
        )

    warnings: list[str] = list(validation.warnings)
    for skip in skipped:
        warnings.append(
            f"requested model {skip.model_name!r} was skipped: {skip.reason}"
        )
    if calibration_warning is not None:
        warnings.append(f"calibration evidence note: {calibration_warning}")
    if execution_warning is not None:
        warnings.append(f"execution sensitivity note: {execution_warning}")
    for plot_warning in plot_warnings:
        warnings.append(f"plot note: {plot_warning}")

    return PaperExperimentSummary(
        experiment_name=config.experiment_name,
        task_name=config.task_name,
        horizon=config.horizon,
        split_name=config.split_name,
        data_path=str(resolved_data_path),
        output_dir=str(resolved_out_dir),
        requested_models=list(requested_models),
        models_run=[outcome.model_name for outcome in outcomes],
        skipped_models=skipped,
        metric_names=metric_names,
        predictive_metric_names=predictive_metric_names,
        calibration_metric_names=calibration_metric_names,
        execution_metric_names=execution_metric_names,
        calibration_models=list(calibration_models),
        execution_models=list(execution_models),
        artefacts=artefacts,
        is_fixture=is_fixture,
        runner_version=PAPER_RUNNER_VERSION,
        created_at=created_at,
        validation=validation,
        outcomes=outcomes,
        warnings=warnings,
        plot_summary=plot_summary,
    )


def _module_smoke() -> int:  # pragma: no cover - convenience entry point
    """Tiny smoke entry point for manual usage; not part of the public CLI."""
    from chronoslob.utils.paths import project_root

    root = project_root()
    summary = run_paper_experiment(
        config_path=root / "configs" / "experiments" / "fi2010_midprice_h10.yaml",
        data_path=root / "tests" / "fixtures" / "fi2010" / "tiny_fi2010_like.csv",
        out_dir=root / "runs" / "paper_experiment_smoke",
        models=("majority",),
        overwrite=True,
    )
    sys.stdout.write(f"paper experiment summary: {summary.experiment_name}\n")
    return 0
