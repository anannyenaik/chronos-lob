"""Paper experiment runner for the FI-2010 mid-price direction task.

This module turns a user-supplied local FI-2010-style file into a
validated experiment artefact directory. It composes the existing
benchmark preparation logic, classical baseline infrastructure and
artefact contract so that one CLI invocation produces a complete,
validated experiment record.

The runner is deliberately scoped:

* It supports only a small initial set of classifiers (``majority`` and
  optionally ``logistic``) so it can be exercised on the tiny synthetic
  fixture without any neural infrastructure.
* It writes only the required artefacts plus row-level predictions and
  (when computed) a confusion-matrix artefact.
* It never downloads data, never fits preprocessing or model-selection
  choices on validation or test data and never performs network calls.

Stronger model families (random forest, DeepLOB-style, transformer,
SSL) are out of scope for this phase; they are tracked under Phase D
and Phase E of the empirical upgrade plan.
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
from chronoslob.experiments.fi2010_benchmark import (
    FI2010BenchmarkConfig,
    FI2010PreparationResult,
    load_benchmark_config,
    prepare_fi2010_benchmark,
)
from chronoslob.experiments.manifests import stable_json_dumps
from chronoslob.experiments.schemas import (
    EvidenceStreams,
    ExperimentConfigSummary,
    ExperimentResults,
    ExperimentValidationReport,
    ModelResult,
)
from chronoslob.models.baselines import (
    BaselineModelConfig,
    create_baseline_model,
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
    "SUPPORTED_PAPER_MODELS",
    "PaperExperimentSummary",
    "PaperModelOutcome",
    "run_paper_experiment",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)

PAPER_RUNNER_VERSION = "phase-c/paper-experiment-runner/v1"

_MAJORITY_MODEL = "majority"
_LOGISTIC_MODEL = "logistic"

SUPPORTED_PAPER_MODELS: tuple[str, ...] = (
    _MAJORITY_MODEL,
    _LOGISTIC_MODEL,
)

_FIXTURE_PATH_MARKERS = ("tests", "fixtures")


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


class PaperExperimentSummary(BaseModel):
    """Top-level summary returned by :func:`run_paper_experiment`."""

    model_config = _MODEL_CONFIG

    experiment_name: str
    task_name: str
    horizon: int
    split_name: str
    data_path: str
    output_dir: str
    models_run: list[str]
    metric_names: list[str]
    artefacts: dict[str, str]
    is_fixture: bool
    runner_version: str
    created_at: datetime
    validation: ExperimentValidationReport
    outcomes: list[PaperModelOutcome] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

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


def _normalise_models(models: Sequence[str] | None) -> tuple[str, ...]:
    if models is None:
        return (_MAJORITY_MODEL,)
    cleaned: list[str] = []
    seen: set[str] = set()
    for model in models:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model names must be non-empty strings")
        token = model.strip().lower()
        if token not in SUPPORTED_PAPER_MODELS:
            raise ValueError(
                "unsupported model name: "
                f"{model!r}; supported: {list(SUPPORTED_PAPER_MODELS)}"
            )
        if token in seen:
            continue
        cleaned.append(token)
        seen.add(token)
    if not cleaned:
        raise ValueError("models must contain at least one supported entry")
    if _MAJORITY_MODEL not in cleaned:
        raise ValueError(
            "the majority baseline must be included; "
            "add 'majority' to --models",
        )
    return tuple(cleaned)


def _is_fixture_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return all(marker in parts for marker in _FIXTURE_PATH_MARKERS)


def _resolve_code_commit() -> str | None:
    return get_git_commit()


def _split_indices_to_lookup(split: SplitIndices) -> dict[str, list[int]]:
    return {
        "train": list(split.train),
        "validation": list(split.validation),
        "test": list(split.test),
    }


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


def _baseline_config_for(model_name: str, *, seed: int) -> BaselineModelConfig:
    if model_name == _MAJORITY_MODEL:
        return BaselineModelConfig(
            name="majority_class",
            model_type="majority_class",
            random_state=seed,
        )
    if model_name == _LOGISTIC_MODEL:
        return BaselineModelConfig(
            name="logistic_regression",
            model_type="logistic_regression",
            random_state=seed,
        )
    raise ValueError(f"unsupported model name in runner: {model_name!r}")


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
) -> list[dict[str, Any]]:
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
    return rows


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


def _build_confusion_matrix_payload(
    outcomes: Sequence[PaperModelOutcome],
) -> dict[str, Any]:
    return {
        outcome.model_name: outcome.confusion_matrix for outcome in outcomes
    }


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
    data_path: Path,
    data_sha256: str | None,
    is_fixture: bool,
    seed: int,
    metric_names: Sequence[str],
    code_commit: str | None,
) -> None:
    lines: list[str] = []
    lines.append(f"# Model Card: {config.experiment_name}")
    lines.append("")
    if is_fixture:
        lines.append(
            "Status: synthetic fixture smoke run. "
            "This artefact set exercises the paper experiment runner on a "
            "tiny synthetic fixture and is not benchmark evidence, market "
            "evidence or execution evidence.",
        )
    else:
        lines.append(
            "Status: locally executed paper experiment run. "
            "Inspect the data manifest, split summary and limitations "
            "before treating the metrics as benchmark evidence.",
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
    lines.append(f"- local source path: `{data_path}`")
    if data_sha256 is not None:
        lines.append(f"- source SHA-256: `{data_sha256}`")
    if is_fixture:
        lines.append(
            "- fixture flag: this path lives under `tests/fixtures/` and "
            "is a synthetic FI-2010-like file used only for plumbing checks.",
        )
    lines.append("")
    lines.append("## Models")
    lines.append("")
    for outcome in outcomes:
        lines.append(
            f"- `{outcome.model_name}` (type `{outcome.model_type}`) on "
            f"the `{outcome.split}` split with {outcome.n_test_rows} rows",
        )
    lines.append("")
    lines.append("## Metrics Emitted")
    lines.append("")
    for metric in metric_names:
        lines.append(f"- {metric}")
    if not metric_names:
        lines.append("- none")
    lines.append("")
    lines.append("## Artefacts")
    lines.append("")
    lines.append("- `config.yaml`")
    lines.append("- `data_manifest.json`")
    lines.append("- `results.json`")
    lines.append("- `predictions.csv`")
    lines.append("- `model_card.md`")
    lines.append("")
    lines.append("## Leakage Controls")
    lines.append("")
    lines.append(
        "- Train, validation and test indices come from the deterministic "
        "temporal splitter; no random or stratified shuffling is used.",
    )
    lines.append(
        "- Train-only standardisation is applied for logistic regression; "
        "the majority baseline never inspects feature values.",
    )
    lines.append(
        "- No model-selection choice, calibrator or bucket boundary is "
        "fitted on validation or test rows in this phase.",
    )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- Only the majority-class baseline (and optionally logistic "
        "regression) is supported in this phase. Random forest, DeepLOB "
        "and transformer baselines are tracked under later phases.",
    )
    lines.append(
        "- Calibration, execution-sensitivity, robustness and systems "
        "evidence streams are not produced in this phase.",
    )
    lines.append(
        "- Reported numbers are run-specific and must not be interpreted "
        "as profitability, deployability or live-trading evidence.",
    )
    if is_fixture:
        lines.append(
            "- The data source is a synthetic fixture; metrics describe "
            "fixture plumbing only.",
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
) -> ExperimentResults:
    metric_names: list[str] = []
    seen: set[str] = set()
    for outcome in outcomes:
        for name in outcome.metrics:
            if name not in seen:
                metric_names.append(name)
                seen.add(name)
    if not metric_names:
        metric_names = ["accuracy", "macro_f1"]

    config_summary = ExperimentConfigSummary(
        experiment_name=config.experiment_name,
        task_name=config.task_name,
        horizon=config.horizon,
        split_name=config.split_name,
        seed=config.seed,
        model_names=[outcome.model_name for outcome in outcomes],
        primary_metric="macro_f1" if "macro_f1" in seen else metric_names[0],
        created_at=created_at,
        code_commit=code_commit,
    )

    model_results: list[ModelResult] = []
    for outcome in outcomes:
        model_results.append(
            ModelResult(
                model_name=outcome.model_name,
                split=outcome.split,
                horizon=outcome.horizon,
                metrics=outcome.metrics,
                artefacts={"predictions": "predictions.csv"},
            )
        )

    predictive_metric_names: frozenset[str] = frozenset(
        {
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "balanced_accuracy",
            "matthews_corrcoef",
            "n_samples",
        }
    )
    predictive_metrics = [
        name for name in metric_names if name in predictive_metric_names
    ]
    calibration_metrics = [
        name for name in metric_names if name in {"brier_score", "log_loss"}
    ]
    if not predictive_metrics:
        predictive_metrics = ["accuracy"]
    if not calibration_metrics:
        calibration_metrics = ["not_computed"]

    evidence_streams = EvidenceStreams(
        predictive=predictive_metrics,
        calibration=calibration_metrics,
        execution=["not_computed"],
        robustness=[],
        systems=[],
    )

    return ExperimentResults(
        experiment_name=config.experiment_name,
        task_name=config.task_name,
        created_at=created_at,
        config_summary=config_summary,
        model_results=model_results,
        evidence_streams=evidence_streams,
    )


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
        Optional sequence of model names. Defaults to ``("majority",)``.
    overwrite:
        When ``False``, refuse to write into an existing directory that
        already contains artefacts. When ``True``, the target directory
        is replaced before writing.

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
    model_tokens = _normalise_models(models)
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

    standardise_required = _LOGISTIC_MODEL in model_tokens
    if standardise_required:
        scaler = TrainOnlyStandardScaler()
        x_train_scaled = scaler.fit_transform(train_features.x)
        x_test_scaled = scaler.transform(test_features.x)
    else:
        x_train_scaled = train_features.x
        x_test_scaled = test_features.x

    test_timestamps = _timestamps_for_indices(
        combined_frame,
        timestamp_column=config.timestamp_column,
        indices=split.test,
    )

    outcomes: list[PaperModelOutcome] = []
    prediction_rows: list[dict[str, Any]] = []

    for model_token in model_tokens:
        baseline_config = _baseline_config_for(model_token, seed=config.seed)
        model = create_baseline_model(baseline_config)
        if model_token == _LOGISTIC_MODEL:
            model.fit(x_train_scaled, train_target.y)
            inference_x = x_test_scaled
        else:
            model.fit(train_features.x, train_target.y)
            inference_x = test_features.x

        metrics = compute_classification_metrics(
            y_true=test_target.y.tolist(),
            y_pred=np.asarray(model.predict(inference_x)).tolist(),
            y_proba=model.predict_proba(inference_x),
            labels=all_target.classes,
        )
        confusion = confusion_matrix_as_dict(
            y_true=test_target.y.tolist(),
            y_pred=np.asarray(model.predict(inference_x)).tolist(),
            labels=all_target.classes,
        )
        outcome = PaperModelOutcome(
            model_name=baseline_config.name,
            model_type=baseline_config.model_type,
            split="test",
            horizon=config.horizon,
            metrics=_select_metric_subset(metrics.to_dict()),
            confusion_matrix=confusion,
            n_test_rows=len(test_target.y),
        )
        outcomes.append(outcome)

        rows = _predictions_for_split(
            model=model,
            x=inference_x,
            y_true=test_target.y,
            row_indices=split.test,
            split_name="test",
            model_name=baseline_config.name,
            labels=all_target.classes or [],
            timestamps=test_timestamps,
        )
        prediction_rows.extend(rows)

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

    code_commit = _resolve_code_commit()
    results = _build_results(
        config=config,
        outcomes=outcomes,
        created_at=created_at,
        code_commit=code_commit,
    )
    results_path = resolved_out_dir / "results.json"
    write_json_model(results_path, results)

    confusion_path = resolved_out_dir / "confusion_matrix.json"
    confusion_path.write_text(
        stable_json_dumps(_build_confusion_matrix_payload(outcomes)),
        encoding="utf-8",
    )

    metric_names: list[str] = []
    seen_metrics: set[str] = set()
    for outcome in outcomes:
        for name in outcome.metrics:
            if name not in seen_metrics:
                metric_names.append(name)
                seen_metrics.add(name)

    data_sha256 = preparation.data_manifest.source_sha256

    model_card_path = resolved_out_dir / "model_card.md"
    is_fixture = _is_fixture_path(resolved_data_path.resolve())
    _write_model_card(
        path=model_card_path,
        config=config,
        outcomes=outcomes,
        data_path=resolved_data_path,
        data_sha256=data_sha256,
        is_fixture=is_fixture,
        seed=config.seed,
        metric_names=metric_names,
        code_commit=code_commit,
    )

    runner_summary_payload: dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "task_name": config.task_name,
        "horizon": config.horizon,
        "split_name": config.split_name,
        "data_path": str(resolved_data_path),
        "output_dir": str(resolved_out_dir),
        "models_run": list(model_tokens),
        "metric_names": metric_names,
        "runner_version": PAPER_RUNNER_VERSION,
        "created_at": created_at.isoformat(),
        "is_fixture": is_fixture,
        "environment": _build_environment_payload(),
        "split_counts": {
            "n_rows": n_rows,
            "n_train": split.n_train,
            "n_validation": split.n_validation,
            "n_test": split.n_test,
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

    artefacts: dict[str, str] = {
        "config": "config.yaml",
        "data_manifest": "data_manifest.json",
        "results": "results.json",
        "predictions": "predictions.csv",
        "model_card": "model_card.md",
        "confusion_matrix": "confusion_matrix.json",
        "runner_summary": "runner_summary.json",
    }

    return PaperExperimentSummary(
        experiment_name=config.experiment_name,
        task_name=config.task_name,
        horizon=config.horizon,
        split_name=config.split_name,
        data_path=str(resolved_data_path),
        output_dir=str(resolved_out_dir),
        models_run=list(model_tokens),
        metric_names=metric_names,
        artefacts=artefacts,
        is_fixture=is_fixture,
        runner_version=PAPER_RUNNER_VERSION,
        created_at=created_at,
        validation=validation,
        outcomes=outcomes,
        warnings=list(validation.warnings),
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
