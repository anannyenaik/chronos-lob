"""Multi-fold classical FI-2010 benchmark runner.

The runner consumes split-aware CSV files produced by
``prepare-fi2010-multifold`` and evaluates classical baselines on each
requested fold. It preserves the official train/test split, carves
validation from official train rows and writes lightweight aggregate
artefacts without storing row-level predictions by default.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from chronoslob import __version__
from chronoslob.experiments.evidence import (
    CalibrationBinRow,
    CalibrationConfig,
    ExecutionRow,
    ExecutionSensitivityConfig,
    build_calibration_bins,
    build_execution_sensitivity,
    build_return_proxy_series,
    summarise_calibration,
    write_calibration_bins_csv,
    write_execution_sensitivity_csv,
)
from chronoslob.experiments.fi2010_benchmark import (
    BenchmarkSplitConfig,
    FI2010BenchmarkConfig,
    build_benchmark_split,
    effective_split_summary,
)
from chronoslob.experiments.fi2010_multifold import load_multifold_config
from chronoslob.experiments.manifests import sha256_file, stable_json_dumps
from chronoslob.experiments.model_registry import (
    PaperModelSpec,
    build_paper_baseline_config,
    get_paper_model_spec,
)
from chronoslob.experiments.paper_runner import (
    _align_probabilities_to_labels,
    _model_known_classes,
    _multiclass_brier_score,
)
from chronoslob.models.baselines import BaseBaselineModel, create_baseline_model
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

__all__ = [
    "CLASSICAL_MULTIFOLD_MODELS",
    "FI2010_MULTIFOLD_CLASSICAL_VERSION",
    "MultifoldClassicalConfig",
    "MultifoldClassicalSummary",
    "load_multifold_classical_config",
    "normalise_classical_model_names",
    "run_fi2010_multifold_classical",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)

FI2010_MULTIFOLD_CLASSICAL_VERSION = "phase-d/fi2010-multifold-classical/v1"

CLASSICAL_MULTIFOLD_MODELS: tuple[str, ...] = (
    "majority",
    "logistic",
    "ridge",
    "elastic_net",
    "random_forest",
    "gradient_boosting",
)

_UNSUPPORTED_IN_CLASSICAL_RUNNER: frozenset[str] = frozenset(
    {
        "deeplob_style",
        "transformer",
        "matrix_transformer",
        "ssl_transformer",
    }
)

RESULTS_BY_FOLD_COLUMNS: tuple[str, ...] = (
    "fold_id",
    "model_name",
    "split",
    "accuracy",
    "macro_f1",
    "mcc",
    "brier_score",
    "ece",
    "n_train",
    "n_validation",
    "n_test",
    "seed",
    "status",
    "error",
)

RESULTS_SUMMARY_COLUMNS: tuple[str, ...] = (
    "model_name",
    "split",
    "fold_count",
    "run_count",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "mcc_mean",
    "mcc_std",
    "brier_score_mean",
    "brier_score_std",
    "ece_mean",
    "ece_std",
)

CALIBRATION_SUMMARY_COLUMNS: tuple[str, ...] = (
    "model_name",
    "split",
    "fold_count",
    "run_count",
    "n_samples_mean",
    "mean_confidence_mean",
    "mean_confidence_std",
    "ece_mean",
    "ece_std",
)

EXECUTION_SUMMARY_COLUMNS: tuple[str, ...] = (
    "model_name",
    "split",
    "confidence_threshold",
    "cost_bps",
    "latency_steps",
    "fold_count",
    "run_count",
    "eligible_predictions",
    "eligible_predictions_std",
    "trade_count_proxy",
    "trade_count_proxy_std",
    "turnover_proxy",
    "turnover_proxy_std",
    "gross_signal_return_proxy",
    "gross_signal_return_proxy_std",
    "cost_proxy",
    "cost_proxy_std",
    "net_signal_return_proxy",
    "net_signal_return_proxy_std",
    "hit_rate_proxy",
    "hit_rate_proxy_std",
)


class AggregationConfig(BaseModel):
    """Aggregation options for top-level CSV summaries."""

    model_config = _MODEL_CONFIG

    metrics: tuple[str, ...] = ("accuracy", "macro_f1", "mcc", "brier_score", "ece")
    std_ddof: int = 0

    @field_validator("metrics")
    @classmethod
    def _validate_metrics(cls, value: Sequence[str]) -> tuple[str, ...]:
        cleaned: list[str] = []
        allowed = {"accuracy", "macro_f1", "mcc", "brier_score", "ece"}
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("aggregation metrics must be non-empty strings")
            metric = item.strip()
            if metric not in allowed:
                raise ValueError(
                    f"unsupported aggregation metric {metric!r}; "
                    f"supported: {sorted(allowed)}",
                )
            if metric not in cleaned:
                cleaned.append(metric)
        if not cleaned:
            raise ValueError("aggregation metrics must not be empty")
        return tuple(cleaned)

    @field_validator("std_ddof")
    @classmethod
    def _validate_std_ddof(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("std_ddof must be an integer")
        if value < 0:
            raise ValueError("std_ddof must be non-negative")
        return value


class OutputArtefactConfig(BaseModel):
    """Configured output artefact filenames."""

    model_config = _MODEL_CONFIG

    summary: str = "summary.json"
    results_by_fold: str = "results_by_fold.csv"
    results_summary: str = "results_summary.csv"
    calibration_summary: str = "calibration_summary.csv"
    execution_summary: str = "execution_summary.csv"
    model_failures: str = "model_failures.json"
    folds_dir: str = "folds"
    fold_subdir_template: str = "fold_{fold}"

    @field_validator(
        "summary",
        "results_by_fold",
        "results_summary",
        "calibration_summary",
        "execution_summary",
        "model_failures",
        "folds_dir",
        "fold_subdir_template",
    )
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("artefact names must be non-empty strings")
        return value.strip()


class MultifoldClassicalConfig(BaseModel):
    """Validated config consumed by the multi-fold classical runner."""

    model_config = _MODEL_CONFIG

    study_name: str
    dataset_name: str
    task_name: str = "midprice_direction"
    folds: tuple[int, ...]
    label_columns: tuple[str, ...]
    target_horizon: int
    label_name: str
    split_column: str
    train_value: str
    test_value: str
    validation_fraction_within_train: float
    processed_root_placeholder: str
    combined_csv_filename_template: str
    classical_models: tuple[str, ...]
    seeds: tuple[int, ...] = (0,)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    output_artefacts: OutputArtefactConfig = Field(default_factory=OutputArtefactConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    execution_sensitivity: ExecutionSensitivityConfig = Field(
        default_factory=ExecutionSensitivityConfig
    )

    @field_validator(
        "study_name",
        "dataset_name",
        "task_name",
        "label_name",
        "split_column",
        "train_value",
        "test_value",
        "processed_root_placeholder",
        "combined_csv_filename_template",
    )
    @classmethod
    def _validate_required_string(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("multi-fold classical config strings must be non-empty")
        return value.strip()

    @field_validator("folds")
    @classmethod
    def _validate_folds(cls, value: Sequence[int]) -> tuple[int, ...]:
        cleaned: list[int] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError("folds must be integers")
            if item <= 0:
                raise ValueError("folds must be positive")
            if item not in cleaned:
                cleaned.append(int(item))
        if not cleaned:
            raise ValueError("folds must not be empty")
        return tuple(cleaned)

    @field_validator("label_columns", "classical_models")
    @classmethod
    def _validate_string_tuple(cls, value: Sequence[str]) -> tuple[str, ...]:
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("config lists must contain non-empty strings")
            token = item.strip()
            if token not in cleaned:
                cleaned.append(token)
        if not cleaned:
            raise ValueError("config lists must not be empty")
        return tuple(cleaned)

    @field_validator("target_horizon")
    @classmethod
    def _validate_horizon(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("target_horizon must be an integer")
        if value <= 0:
            raise ValueError("target_horizon must be positive")
        return int(value)

    @field_validator("validation_fraction_within_train")
    @classmethod
    def _validate_validation_fraction(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("validation_fraction_within_train must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0 or numeric >= 1.0:
            raise ValueError(
                "validation_fraction_within_train must satisfy 0 <= value < 1",
            )
        return numeric

    @field_validator("seeds")
    @classmethod
    def _validate_seeds(cls, value: Sequence[int]) -> tuple[int, ...]:
        cleaned: list[int] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError("seeds must be integers")
            if item < 0:
                raise ValueError("seeds must be non-negative")
            if item not in cleaned:
                cleaned.append(int(item))
        if not cleaned:
            raise ValueError("seeds must contain at least one entry")
        return tuple(cleaned)


class MultifoldClassicalSummary(BaseModel):
    """Top-level return value from the multi-fold classical runner."""

    model_config = _MODEL_CONFIG

    study_name: str
    dataset_name: str
    task_name: str
    target_horizon: int
    config_path: str
    processed_root: str
    output_dir: str
    folds_requested: list[int]
    folds_completed: list[int]
    models_requested: list[str]
    seeds: list[int]
    fold_count: int
    model_count: int
    result_rows: int
    failure_count: int
    full_predictions_written: bool
    artefacts: dict[str, str]
    runner_version: str
    created_at: datetime
    git_commit: str | None = None
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _FoldInput:
    fold_id: int
    path: Path
    sha256: str


@dataclass(frozen=True)
class _PreparedModel:
    model: BaseBaselineModel
    scaler: TrainOnlyStandardScaler | None
    standardisation: dict[str, Any] | None


@dataclass(frozen=True)
class _SplitEvaluation:
    result_row: dict[str, Any]
    prediction_frame: pd.DataFrame
    calibration_bins: list[CalibrationBinRow]
    calibration_summary: dict[str, Any] | None
    execution_rows: list[ExecutionRow]
    confusion_entry: dict[str, Any] | None


def _as_mapping(payload: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a YAML mapping")
    return payload


def _sequence_from_mapping(
    mapping: Mapping[str, Any],
    key: str,
    default: Sequence[Any],
) -> tuple[Any, ...]:
    value = mapping.get(key, default)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{key} must be a sequence")
    return tuple(value)


def _normalise_seeds(payload: Mapping[str, Any]) -> tuple[int, ...]:
    runner = payload.get("classical_runner")
    if isinstance(runner, Mapping) and "seeds" in runner:
        raw = runner["seeds"]
    else:
        raw_seeds = payload.get("seeds", (0,))
        if isinstance(raw_seeds, Sequence) and not isinstance(raw_seeds, (str, bytes)):
            raw = tuple(raw_seeds[:1]) if raw_seeds else (0,)
        else:
            raw = (0,)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("seed list must be a sequence of integers")
    return tuple(int(item) for item in raw)


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _as_mapping(payload, name=str(path))


def load_multifold_classical_config(path: str | Path) -> MultifoldClassicalConfig:
    """Load and validate the multi-fold classical runner config."""
    config_path = Path(path)
    base = load_multifold_config(config_path)
    payload = _load_yaml_mapping(config_path)

    target = _as_mapping(payload.get("target"), name="target")
    official_split = _as_mapping(payload.get("official_split"), name="official_split")
    preparation = _as_mapping(payload.get("preparation"), name="preparation")
    models_mapping = _as_mapping(payload.get("models", {}), name="models")
    runner_mapping = payload.get("classical_runner", {})
    if runner_mapping is None:
        runner_mapping = {}
    runner_mapping = _as_mapping(runner_mapping, name="classical_runner")

    configured_models = runner_mapping.get("models", models_mapping.get("classical"))
    if configured_models is None:
        configured_models = CLASSICAL_MULTIFOLD_MODELS
    if isinstance(configured_models, (str, bytes)) or not isinstance(
        configured_models, Sequence
    ):
        raise ValueError("classical model list must be a sequence")

    calibration_payload = dict(
        _as_mapping(
            payload.get("calibration", {}),
            name="calibration",
        )
    )
    calibration_payload = {
        key: value
        for key, value in calibration_payload.items()
        if key in {"enabled", "n_bins"}
    }
    execution_payload = dict(
        _as_mapping(
            payload.get("execution_sensitivity", {}),
            name="execution_sensitivity",
        )
    )
    target_direction_map = target.get("class_direction_map")
    if (
        "class_direction_map" not in execution_payload
        and isinstance(target_direction_map, Mapping)
    ):
        execution_payload["class_direction_map"] = dict(target_direction_map)

    aggregation_payload = _as_mapping(
        payload.get("aggregation", {}),
        name="aggregation",
    )
    artefact_payload_raw = payload.get("output_artefacts")
    if artefact_payload_raw is None:
        artefact_payload_raw = {}
    artefact_payload = _as_mapping(artefact_payload_raw, name="output_artefacts")

    return MultifoldClassicalConfig.model_validate(
        {
            "study_name": base.study_name,
            "dataset_name": base.dataset_name,
            "task_name": str(payload.get("task_name", "midprice_direction")),
            "folds": base.folds,
            "label_columns": base.label_columns,
            "target_horizon": base.target_horizon,
            "label_name": str(target.get("label_column", f"label_{base.target_horizon}")),
            "split_column": base.split_column,
            "train_value": base.train_value,
            "test_value": base.test_value,
            "validation_fraction_within_train": official_split.get(
                "validation_fraction_within_train",
                0.15,
            ),
            "processed_root_placeholder": str(
                preparation.get(
                    "processed_output_root_placeholder",
                    payload.get("local_data_root_path", "data/processed/fi2010"),
                )
            ),
            "combined_csv_filename_template": str(
                preparation.get(
                    "combined_csv_filename_template",
                    payload.get("combined_csv_filename_template", "fold{fold}_combined.csv"),
                )
            ),
            "classical_models": tuple(str(item) for item in configured_models),
            "seeds": _normalise_seeds(payload),
            "aggregation": dict(aggregation_payload),
            "output_artefacts": dict(artefact_payload),
            "calibration": dict(calibration_payload),
            "execution_sensitivity": execution_payload,
        }
    )


def normalise_classical_model_names(
    models: Sequence[str] | None,
    *,
    default: Sequence[str] = CLASSICAL_MULTIFOLD_MODELS,
) -> tuple[str, ...]:
    """Normalise and validate model names for this classical-only runner."""
    selected = tuple(default) if models is None else tuple(models)
    if not selected:
        raise ValueError("models must contain at least one entry")

    cleaned: list[str] = []
    for raw in selected:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("model names must be non-empty strings")
        token = raw.strip().lower()
        if token in _UNSUPPORTED_IN_CLASSICAL_RUNNER:
            raise ValueError(
                f"model {token!r} is not supported by the multi-fold "
                "classical runner; supported classical models: "
                f"{list(CLASSICAL_MULTIFOLD_MODELS)}",
            )
        spec = get_paper_model_spec(token)
        if spec.model_family != "classical":
            raise ValueError(
                f"model {spec.name!r} is not a classical baseline for this runner",
            )
        if spec.name not in CLASSICAL_MULTIFOLD_MODELS:
            raise ValueError(
                f"model {spec.name!r} is not enabled for this runner; "
                f"supported: {list(CLASSICAL_MULTIFOLD_MODELS)}",
            )
        if spec.name not in cleaned:
            cleaned.append(spec.name)
    return tuple(cleaned)


def _filter_folds(configured: Sequence[int], requested: Sequence[int] | None) -> tuple[int, ...]:
    if requested is None:
        return tuple(configured)
    configured_set = set(configured)
    cleaned: list[int] = []
    for fold in requested:
        if isinstance(fold, bool) or not isinstance(fold, int):
            raise ValueError("requested folds must be integers")
        if fold not in configured_set:
            raise ValueError(
                f"requested fold {fold} is not configured; configured folds "
                f"are {sorted(configured_set)}",
            )
        if fold not in cleaned:
            cleaned.append(int(fold))
    if not cleaned:
        raise ValueError("requested folds must not be empty")
    return tuple(cleaned)


def _fold_filename(template: str, fold_id: int) -> str:
    if "{fold}" not in template:
        raise ValueError("combined_csv_filename_template must contain '{fold}'")
    return template.replace("{fold}", str(fold_id))


def _resolve_fold_inputs(
    config: MultifoldClassicalConfig,
    *,
    processed_root: Path,
    folds: Sequence[int],
) -> list[_FoldInput]:
    inputs: list[_FoldInput] = []
    for fold_id in folds:
        path = processed_root / _fold_filename(
            config.combined_csv_filename_template,
            fold_id,
        )
        if not path.is_file():
            raise FileNotFoundError(f"fold {fold_id} processed CSV is missing: {path}")
        _validate_fold_csv_headers_and_split(config, path=path, fold_id=fold_id)
        inputs.append(_FoldInput(fold_id=fold_id, path=path, sha256=sha256_file(path)))
    return inputs


def _validate_fold_csv_headers_and_split(
    config: MultifoldClassicalConfig,
    *,
    path: Path,
    fold_id: int,
) -> None:
    header = pd.read_csv(path, nrows=0)
    missing = [
        column
        for column in (config.split_column, config.label_name)
        if column not in header.columns
    ]
    if missing:
        raise ValueError(f"fold {fold_id} CSV is missing required columns: {missing}")

    split_frame = pd.read_csv(path, usecols=[config.split_column])
    if split_frame.empty:
        raise ValueError(f"fold {fold_id} CSV contains no rows: {path}")
    values: list[str] = []
    for position, raw in enumerate(split_frame[config.split_column].tolist()):
        if pd.isna(raw):
            raise ValueError(
                f"fold {fold_id} split column has a missing value at row {position}",
            )
        text = str(raw).strip().casefold()
        if not text:
            raise ValueError(
                f"fold {fold_id} split column has an empty value at row {position}",
            )
        values.append(text)
    allowed = {config.train_value.casefold(), config.test_value.casefold()}
    unexpected = sorted({value for value in values if value not in allowed})
    if unexpected:
        raise ValueError(
            f"fold {fold_id} split column contains invalid labels {unexpected}; "
            f"expected only {sorted(allowed)}",
        )
    if config.train_value.casefold() not in values:
        raise ValueError(f"fold {fold_id} split column contains no train rows")
    if config.test_value.casefold() not in values:
        raise ValueError(f"fold {fold_id} split column contains no test rows")


def _ensure_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {path}")
        if any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    "refusing to write into a non-empty output directory; "
                    f"pass overwrite=True to replace it: {path}",
                )
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _benchmark_config(config: MultifoldClassicalConfig, *, seed: int) -> FI2010BenchmarkConfig:
    return FI2010BenchmarkConfig(
        experiment_name=f"{config.study_name}_classical",
        dataset_name=config.dataset_name,
        task_name=config.task_name,
        horizon=config.target_horizon,
        split_name="official_column",
        seed=seed,
        local_data_path="<multi-fold-processed-root>",
        output_dir="experiments/fi2010_multifold_classical",
        label_name=config.label_name,
        label_columns=config.label_columns,
        timestamp_column=None,
        split_column=config.split_column,
        price_level_count=10,
        train_fraction=0.7,
        validation_fraction=0.15,
        test_fraction=0.15,
        split=BenchmarkSplitConfig(
            method="official_column",
            split_column=config.split_column,
            validation_fraction_within_train=config.validation_fraction_within_train,
            train_value=config.train_value,
            test_value=config.test_value,
        ),
        calibration=config.calibration,
        execution_sensitivity=config.execution_sensitivity,
    )


def _feature_columns(frame: pd.DataFrame, config: MultifoldClassicalConfig) -> list[str]:
    excluded = {
        config.split_column,
        config.label_name,
        *config.label_columns,
    }
    excluded.update(
        column
        for column in frame.columns
        if str(column).lower().startswith("label")
    )
    candidate = frame.drop(columns=[column for column in excluded if column in frame.columns])
    columns = select_feature_columns(candidate, reject_label_like=True)
    if not columns:
        raise ValueError("multi-fold classical runner requires at least one feature column")
    return columns


def _fit_model(
    *,
    spec: PaperModelSpec,
    seed: int,
    train_features_raw: np.ndarray,
    train_labels: np.ndarray,
    train_indices: Sequence[int],
) -> _PreparedModel:
    model = create_baseline_model(build_paper_baseline_config(spec.name, seed=seed))
    if spec.requires_standardisation:
        scaler = TrainOnlyStandardScaler()
        train_x = scaler.fit_transform(train_features_raw)
        standardisation: dict[str, Any] | None = {
            "fit_split": "train",
            "fit_row_count": len(train_indices),
            "fit_row_start": int(train_indices[0]) if train_indices else None,
            "fit_row_end": int(train_indices[-1]) if train_indices else None,
            "feature_count": int(train_features_raw.shape[1]),
        }
    else:
        train_x = train_features_raw
        scaler = None
        standardisation = None
    model.fit(train_x, train_labels)
    return _PreparedModel(
        model=model,
        scaler=scaler,
        standardisation=standardisation,
    )


def _transform_features(
    *,
    prepared: _PreparedModel,
    spec: PaperModelSpec,
    inference_features_raw: np.ndarray,
) -> np.ndarray:
    if not spec.requires_standardisation:
        return inference_features_raw
    if prepared.scaler is None:
        raise ValueError("standardisation metadata is missing for scaled model")
    return prepared.scaler.transform(inference_features_raw)


def _prediction_frame(
    *,
    row_indices: Sequence[int],
    split_name: str,
    model_name: str,
    labels_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for position, row_index in enumerate(row_indices):
        row: dict[str, Any] = {
            "row_index": int(row_index),
            "split": split_name,
            "model_name": model_name,
            "label": _json_value(labels_true[position]),
            "prediction": _json_value(predictions[position]),
        }
        if probabilities is not None:
            row["confidence"] = float(np.max(probabilities[position]))
        rows.append(row)
    return pd.DataFrame(rows)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _evaluate_split(
    *,
    config: MultifoldClassicalConfig,
    fold_id: int,
    seed: int,
    split_name: str,
    spec: PaperModelSpec,
    prepared: _PreparedModel,
    inference_features_raw: np.ndarray,
    labels_true: np.ndarray,
    row_indices: Sequence[int],
    all_labels: Sequence[Any],
    n_train: int,
    n_validation: int,
    n_test: int,
    return_proxy: pd.Series | None,
) -> _SplitEvaluation:
    if len(row_indices) == 0:
        return _SplitEvaluation(
            result_row={
                "fold_id": fold_id,
                "model_name": spec.name,
                "split": split_name,
                "accuracy": None,
                "macro_f1": None,
                "mcc": None,
                "brier_score": None,
                "ece": None,
                "n_train": n_train,
                "n_validation": n_validation,
                "n_test": n_test,
                "seed": seed,
                "status": "skipped",
                "error": "split has no rows",
            },
            prediction_frame=pd.DataFrame(),
            calibration_bins=[],
            calibration_summary=None,
            execution_rows=[],
            confusion_entry=None,
        )

    inference_x = _transform_features(
        prepared=prepared,
        spec=spec,
        inference_features_raw=inference_features_raw,
    )
    predictions = np.asarray(prepared.model.predict(inference_x))
    proba_raw = prepared.model.predict_proba(inference_x)
    probabilities: np.ndarray | None = None
    if proba_raw is not None:
        probabilities = _align_probabilities_to_labels(
            np.asarray(proba_raw, dtype=float),
            model_classes=_model_known_classes(prepared.model),
            target_labels=all_labels,
        )

    metrics = compute_classification_metrics(
        labels_true.tolist(),
        predictions.tolist(),
        y_proba=probabilities,
        labels=all_labels,
    )
    brier = metrics.brier_score
    if brier is None and probabilities is not None:
        brier = _multiclass_brier_score(
            y_true=labels_true.tolist(),
            probabilities=probabilities,
            labels=all_labels,
        )

    prediction_frame = _prediction_frame(
        row_indices=row_indices,
        split_name=split_name,
        model_name=spec.name,
        labels_true=labels_true,
        predictions=predictions,
        probabilities=probabilities,
    )

    calibration_bins: list[CalibrationBinRow] = []
    calibration_payload: dict[str, Any] | None = None
    ece: float | None = None
    if config.calibration.enabled and probabilities is not None:
        calibration_bins = build_calibration_bins(
            prediction_frame,
            model_name=spec.name,
            split=split_name,
            n_bins=config.calibration.n_bins,
        )
        calibration_summary = summarise_calibration(calibration_bins)
        if calibration_summary is not None:
            ece = float(calibration_summary.expected_calibration_error)
            calibration_payload = {
                "fold_id": fold_id,
                "model_name": spec.name,
                "split": split_name,
                "seed": seed,
                "n_bins": calibration_summary.n_bins,
                "n_samples": calibration_summary.n_samples,
                "mean_confidence": calibration_summary.mean_confidence,
                "ece": calibration_summary.expected_calibration_error,
            }

    execution_rows: list[ExecutionRow] = []
    if (
        split_name == "test"
        and config.execution_sensitivity.enabled
        and return_proxy is not None
        and probabilities is not None
    ):
        direction_map = config.execution_sensitivity.class_direction_map
        if direction_map is None:
            direction_map = {str(label): 0 for label in all_labels}
        execution_rows = build_execution_sensitivity(
            prediction_frame,
            model_name=spec.name,
            split=split_name,
            return_proxy=return_proxy,
            config=config.execution_sensitivity,
            direction_map=direction_map,
        )

    confusion = confusion_matrix_as_dict(
        y_true=labels_true.tolist(),
        y_pred=predictions.tolist(),
        labels=all_labels,
    )
    confusion_entry = {
        "fold_id": fold_id,
        "model_name": spec.name,
        "split": split_name,
        "seed": seed,
        "labels": confusion["labels"],
        "matrix": confusion["matrix"],
    }

    return _SplitEvaluation(
        result_row={
            "fold_id": fold_id,
            "model_name": spec.name,
            "split": split_name,
            "accuracy": float(metrics.accuracy),
            "macro_f1": float(metrics.macro_f1),
            "mcc": float(metrics.matthews_corrcoef),
            "brier_score": None if brier is None else float(brier),
            "ece": ece,
            "n_train": n_train,
            "n_validation": n_validation,
            "n_test": n_test,
            "seed": seed,
            "status": "ok",
            "error": "",
        },
        prediction_frame=prediction_frame,
        calibration_bins=calibration_bins,
        calibration_summary=calibration_payload,
        execution_rows=execution_rows,
        confusion_entry=confusion_entry,
    )


def _failure_rows(
    *,
    fold_id: int,
    model_name: str,
    seed: int,
    n_train: int,
    n_validation: int,
    n_test: int,
    error: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ("validation", "test"):
        rows.append(
            {
                "fold_id": fold_id,
                "model_name": model_name,
                "split": split_name,
                "accuracy": None,
                "macro_f1": None,
                "mcc": None,
                "brier_score": None,
                "ece": None,
                "n_train": n_train,
                "n_validation": n_validation,
                "n_test": n_test,
                "seed": seed,
                "status": "failed",
                "error": error,
            }
        )
    return rows


def _run_fold(
    *,
    config: MultifoldClassicalConfig,
    fold_input: _FoldInput,
    models: Sequence[str],
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[ExecutionRow]]:
    frame = pd.read_csv(fold_input.path)
    benchmark_config = _benchmark_config(config, seed=config.seeds[0])
    split = build_benchmark_split(benchmark_config, frame)
    split_summary = effective_split_summary(
        config=benchmark_config,
        frame=frame,
        indices=split,
    )

    feature_columns = _feature_columns(frame, config)
    train_features = build_feature_matrix(
        frame,
        feature_columns=feature_columns,
        row_indices=split.train,
    )
    validation_features = build_feature_matrix(
        frame,
        feature_columns=feature_columns,
        row_indices=split.validation,
    )
    test_features = build_feature_matrix(
        frame,
        feature_columns=feature_columns,
        row_indices=split.test,
    )
    train_target = build_target_vector(
        frame,
        target_column=config.label_name,
        row_indices=split.train,
    )
    validation_target = build_target_vector(
        frame,
        target_column=config.label_name,
        row_indices=split.validation,
    )
    test_target = build_target_vector(
        frame,
        target_column=config.label_name,
        row_indices=split.test,
    )
    all_target = build_target_vector(frame, target_column=config.label_name)
    all_labels = list(all_target.classes or [])

    return_proxy = None
    execution_warning: str | None = None
    if config.execution_sensitivity.enabled:
        return_proxy = build_return_proxy_series(
            frame,
            horizon=config.target_horizon,
            config=config.execution_sensitivity.return_proxy,
        )
        if return_proxy is None:
            execution_warning = (
                "execution proxy unavailable because the configured price "
                "columns could not build a forward mid-price series"
            )

    fold_dir = out_dir / config.output_artefacts.folds_dir / (
        config.output_artefacts.fold_subdir_template.replace(
            "{fold}",
            str(fold_input.fold_id),
        )
    )
    fold_dir.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    calibration_summaries: list[dict[str, Any]] = []
    calibration_bins: list[CalibrationBinRow] = []
    execution_rows: list[ExecutionRow] = []
    confusion_entries: list[dict[str, Any]] = []
    model_metadata: dict[str, Any] = {}

    for seed in config.seeds:
        for model_name in models:
            spec = get_paper_model_spec(model_name)
            try:
                prepared = _fit_model(
                    spec=spec,
                    seed=seed,
                    train_features_raw=train_features.x,
                    train_labels=train_target.y,
                    train_indices=split.train,
                )
                if prepared.standardisation is not None:
                    model_metadata[f"{model_name}/seed_{seed}"] = {
                        "standardisation": prepared.standardisation
                    }
                for split_name, features, target, indices in (
                    (
                        "validation",
                        validation_features.x,
                        validation_target.y,
                        split.validation,
                    ),
                    ("test", test_features.x, test_target.y, split.test),
                ):
                    evaluation = _evaluate_split(
                        config=config,
                        fold_id=fold_input.fold_id,
                        seed=seed,
                        split_name=split_name,
                        spec=spec,
                        prepared=prepared,
                        inference_features_raw=features,
                        labels_true=target,
                        row_indices=indices,
                        all_labels=all_labels,
                        n_train=split.n_train,
                        n_validation=split.n_validation,
                        n_test=split.n_test,
                        return_proxy=return_proxy,
                    )
                    result_rows.append(evaluation.result_row)
                    calibration_bins.extend(evaluation.calibration_bins)
                    if evaluation.calibration_summary is not None:
                        calibration_summaries.append(evaluation.calibration_summary)
                    execution_rows.extend(evaluation.execution_rows)
                    if evaluation.confusion_entry is not None:
                        confusion_entries.append(evaluation.confusion_entry)
            except (ImportError, ValueError, RuntimeError, TypeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                failures.append(
                    {
                        "fold_id": fold_input.fold_id,
                        "model_name": model_name,
                        "seed": seed,
                        "error": error,
                    }
                )
                result_rows.extend(
                    _failure_rows(
                        fold_id=fold_input.fold_id,
                        model_name=model_name,
                        seed=seed,
                        n_train=split.n_train,
                        n_validation=split.n_validation,
                        n_test=split.n_test,
                        error=error,
                    )
                )

    write_calibration_bins_csv(calibration_bins, fold_dir / "calibration_bins.csv")
    write_execution_sensitivity_csv(
        execution_rows,
        fold_dir / "execution_sensitivity.csv",
    )
    (fold_dir / "confusion_matrix.json").write_text(
        stable_json_dumps({"fold_id": fold_input.fold_id, "models": confusion_entries}),
        encoding="utf-8",
    )

    fold_payload: dict[str, Any] = {
        "fold_id": fold_input.fold_id,
        "fold_csv": str(fold_input.path),
        "fold_csv_sha256": fold_input.sha256,
        "runner_version": FI2010_MULTIFOLD_CLASSICAL_VERSION,
        "split_summary": split_summary.model_dump(mode="json"),
        "feature_columns": feature_columns,
        "models": result_rows,
        "model_metadata": model_metadata,
        "failures": failures,
        "warnings": [execution_warning] if execution_warning is not None else [],
        "full_predictions_written": False,
    }
    (fold_dir / "results.json").write_text(
        stable_json_dumps(fold_payload),
        encoding="utf-8",
    )
    _write_fold_model_card(
        path=fold_dir / "model_card.md",
        config=config,
        fold_input=fold_input,
        models=models,
        split_summary=split_summary.model_dump(mode="json"),
        failures=failures,
        execution_warning=execution_warning,
    )
    return result_rows, failures, calibration_summaries, execution_rows


def _write_fold_model_card(
    *,
    path: Path,
    config: MultifoldClassicalConfig,
    fold_input: _FoldInput,
    models: Sequence[str],
    split_summary: Mapping[str, Any],
    failures: Sequence[Mapping[str, Any]],
    execution_warning: str | None,
) -> None:
    lines = [
        f"# FI-2010 Multi-Fold Classical Fold {fold_input.fold_id}",
        "",
        "## Scope",
        "",
        "- runner: classical baselines only",
        f"- dataset: `{config.dataset_name}`",
        f"- task: `{config.task_name}`",
        f"- target: `{config.label_name}`",
        f"- fold CSV: `{fold_input.path}`",
        "- full predictions: not written",
        "",
        "## Split",
        "",
        f"- train rows: {split_summary.get('n_train')}",
        f"- validation rows: {split_summary.get('n_validation')}",
        f"- official test rows: {split_summary.get('n_test')}",
        "- validation is carved only from official train rows",
        "- preprocessing is fitted only on train rows",
        "",
        "## Models",
        "",
    ]
    for model_name in models:
        lines.append(f"- `{model_name}`")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- No neural or self-supervised models are run by this runner.",
            "- Execution-sensitivity rows are proxy diagnostics under configured assumptions.",
            "- No profitability or deployment claim is made.",
        ]
    )
    if execution_warning is not None:
        lines.append(f"- Execution proxy note: {execution_warning}.")
    if failures:
        lines.append("- Model failures are recorded in `model_failures.json`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        frame = pd.DataFrame(list(rows), columns=list(columns))
    else:
        frame = pd.DataFrame(columns=list(columns))
    frame.to_csv(path, index=False)


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _std(series: pd.Series, *, ddof: int) -> float:
    numeric = _safe_numeric(series).dropna()
    if numeric.empty:
        return 0.0
    if len(numeric) <= ddof:
        return 0.0
    value = float(numeric.std(ddof=ddof))
    return value if math.isfinite(value) else 0.0


def _mean(series: pd.Series) -> float | None:
    numeric = _safe_numeric(series).dropna()
    if numeric.empty:
        return None
    value = float(numeric.mean())
    return value if math.isfinite(value) else None


def _build_results_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    ddof: int,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return []
    ok = frame[frame["status"] == "ok"].copy()
    if ok.empty:
        return []
    summary_rows: list[dict[str, Any]] = []
    for (model_name, split_name), group in ok.groupby(["model_name", "split"], sort=True):
        row: dict[str, Any] = {
            "model_name": str(model_name),
            "split": str(split_name),
            "fold_count": int(group["fold_id"].nunique()),
            "run_count": len(group),
        }
        for metric in ("accuracy", "macro_f1", "mcc", "brier_score", "ece"):
            row[f"{metric}_mean"] = _mean(group[metric])
            row[f"{metric}_std"] = _std(group[metric], ddof=ddof)
        summary_rows.append(row)
    return summary_rows


def _build_calibration_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    ddof: int,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return []
    summary_rows: list[dict[str, Any]] = []
    for (model_name, split_name), group in frame.groupby(["model_name", "split"], sort=True):
        summary_rows.append(
            {
                "model_name": str(model_name),
                "split": str(split_name),
                "fold_count": int(group["fold_id"].nunique()),
                "run_count": len(group),
                "n_samples_mean": _mean(group["n_samples"]),
                "mean_confidence_mean": _mean(group["mean_confidence"]),
                "mean_confidence_std": _std(group["mean_confidence"], ddof=ddof),
                "ece_mean": _mean(group["ece"]),
                "ece_std": _std(group["ece"], ddof=ddof),
            }
        )
    return summary_rows


def _build_execution_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    ddof: int,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return []
    group_columns = [
        "model_name",
        "split",
        "confidence_threshold",
        "cost_bps",
        "latency_steps",
    ]
    metrics = [
        "eligible_predictions",
        "trade_count_proxy",
        "turnover_proxy",
        "gross_signal_return_proxy",
        "cost_proxy",
        "net_signal_return_proxy",
        "hit_rate_proxy",
    ]
    summary_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, sort=True):
        model_name, split_name, threshold, cost, latency = keys
        row: dict[str, Any] = {
            "model_name": str(model_name),
            "split": str(split_name),
            "confidence_threshold": float(threshold),
            "cost_bps": float(cost),
            "latency_steps": int(latency),
            "fold_count": int(group["fold_id"].nunique()) if "fold_id" in group else 0,
            "run_count": len(group),
        }
        for metric in metrics:
            row[metric] = _mean(group[metric])
            row[f"{metric}_std"] = _std(group[metric], ddof=ddof)
        summary_rows.append(row)
    return summary_rows


def _execution_rows_with_fold_id(
    rows: Sequence[ExecutionRow],
    *,
    fold_id_lookup: Mapping[tuple[str, str], int],
) -> list[dict[str, Any]]:
    materialised: list[dict[str, Any]] = []
    for row in rows:
        payload = row.to_csv_row()
        key = (row.model_name, row.split)
        payload["fold_id"] = int(fold_id_lookup.get(key, 0))
        materialised.append(payload)
    return materialised


def _build_environment_payload() -> dict[str, str]:
    import platform

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_version": __version__,
    }


def run_fi2010_multifold_classical(
    config_path: str | Path,
    *,
    processed_root: str | Path | None = None,
    out_dir: str | Path,
    models: Sequence[str] | None = None,
    folds: Sequence[int] | None = None,
    overwrite: bool = False,
) -> MultifoldClassicalSummary:
    """Run classical FI-2010 benchmarks across prepared folds."""
    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_file():
        raise FileNotFoundError(
            f"multi-fold classical config not found: {resolved_config_path}",
        )
    config = load_multifold_classical_config(resolved_config_path)
    selected_models = normalise_classical_model_names(
        models,
        default=config.classical_models,
    )
    selected_folds = _filter_folds(config.folds, folds)
    resolved_processed = Path(
        processed_root
        if processed_root is not None
        else config.processed_root_placeholder
    )
    fold_inputs = _resolve_fold_inputs(
        config,
        processed_root=resolved_processed,
        folds=selected_folds,
    )

    resolved_out_dir = Path(out_dir)
    _ensure_output_dir(resolved_out_dir, overwrite=overwrite)

    created_at = datetime.now(UTC)
    all_result_rows: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []
    all_calibration_summaries: list[dict[str, Any]] = []
    all_execution_rows: list[ExecutionRow] = []
    all_execution_summary_inputs: list[dict[str, Any]] = []

    for fold_input in fold_inputs:
        fold_rows, failures, calibration_rows, execution_rows = _run_fold(
            config=config,
            fold_input=fold_input,
            models=selected_models,
            out_dir=resolved_out_dir,
        )
        all_result_rows.extend(fold_rows)
        all_failures.extend(failures)
        all_calibration_summaries.extend(calibration_rows)
        all_execution_rows.extend(execution_rows)
        for row in execution_rows:
            payload = row.to_csv_row()
            payload["fold_id"] = fold_input.fold_id
            all_execution_summary_inputs.append(payload)

    artefacts = {
        "summary": config.output_artefacts.summary,
        "results_by_fold": config.output_artefacts.results_by_fold,
        "results_summary": config.output_artefacts.results_summary,
        "calibration_summary": config.output_artefacts.calibration_summary,
        "execution_summary": config.output_artefacts.execution_summary,
        "model_failures": config.output_artefacts.model_failures,
        "folds": f"{config.output_artefacts.folds_dir}/",
    }

    _write_csv(
        all_result_rows,
        resolved_out_dir / config.output_artefacts.results_by_fold,
        RESULTS_BY_FOLD_COLUMNS,
    )
    result_summary_rows = _build_results_summary(
        all_result_rows,
        ddof=config.aggregation.std_ddof,
    )
    _write_csv(
        result_summary_rows,
        resolved_out_dir / config.output_artefacts.results_summary,
        RESULTS_SUMMARY_COLUMNS,
    )
    calibration_summary_rows = _build_calibration_summary(
        all_calibration_summaries,
        ddof=config.aggregation.std_ddof,
    )
    _write_csv(
        calibration_summary_rows,
        resolved_out_dir / config.output_artefacts.calibration_summary,
        CALIBRATION_SUMMARY_COLUMNS,
    )
    execution_summary_rows = _build_execution_summary(
        all_execution_summary_inputs,
        ddof=config.aggregation.std_ddof,
    )
    _write_csv(
        execution_summary_rows,
        resolved_out_dir / config.output_artefacts.execution_summary,
        EXECUTION_SUMMARY_COLUMNS,
    )

    model_failures_payload: dict[str, Any] = {
        "failure_count": len(all_failures),
        "failures": all_failures,
    }
    (resolved_out_dir / config.output_artefacts.model_failures).write_text(
        stable_json_dumps(model_failures_payload),
        encoding="utf-8",
    )

    summary = MultifoldClassicalSummary(
        study_name=config.study_name,
        dataset_name=config.dataset_name,
        task_name=config.task_name,
        target_horizon=config.target_horizon,
        config_path=str(resolved_config_path),
        processed_root=str(resolved_processed),
        output_dir=str(resolved_out_dir),
        folds_requested=list(selected_folds),
        folds_completed=[item.fold_id for item in fold_inputs],
        models_requested=list(selected_models),
        seeds=list(config.seeds),
        fold_count=len(fold_inputs),
        model_count=len(selected_models),
        result_rows=len(all_result_rows),
        failure_count=len(all_failures),
        full_predictions_written=False,
        artefacts=artefacts,
        runner_version=FI2010_MULTIFOLD_CLASSICAL_VERSION,
        created_at=created_at,
        git_commit=get_git_commit(),
        warnings=[],
    )

    summary_payload = summary.model_dump(mode="json")
    summary_payload["config_sha256"] = sha256_file(resolved_config_path)
    summary_payload["fold_inputs"] = [
        {"fold_id": item.fold_id, "path": str(item.path), "sha256": item.sha256}
        for item in fold_inputs
    ]
    summary_payload["environment"] = _build_environment_payload()
    summary_payload["aggregation"] = config.aggregation.model_dump(mode="json")
    summary_payload["calibration"] = config.calibration.model_dump(mode="json")
    summary_payload["execution_sensitivity"] = config.execution_sensitivity.model_dump(
        mode="json"
    )
    (resolved_out_dir / config.output_artefacts.summary).write_text(
        stable_json_dumps(summary_payload),
        encoding="utf-8",
    )
    return summary
