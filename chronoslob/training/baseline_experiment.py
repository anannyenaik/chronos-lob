"""Baseline experiment runner using train-only preprocessing and temporal splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from chronoslob.labels.leakage import validate_no_lookahead
from chronoslob.models.baselines import BaselineModelConfig, create_baseline_model
from chronoslob.models.preprocessing import (
    TrainOnlyStandardScaler,
    align_feature_label_frames,
    build_feature_matrix,
    build_target_vector,
    select_feature_columns,
)
from chronoslob.training.artifacts import write_json
from chronoslob.training.evaluate import evaluate_classifier
from chronoslob.training.experiment import initialise_experiment_run
from chronoslob.training.splitters import (
    PurgedEmbargoConfig,
    TemporalSplitConfig,
    apply_purge_and_embargo,
    make_label_horizon_end_indices_from_frame,
    temporal_train_validation_test_split,
)
from chronoslob.utils.paths import project_root

__all__ = [
    "BaselineExperimentConfig",
    "BaselinePreprocessingConfig",
    "BaselineSplitConfig",
    "create_default_baseline_configs",
    "run_baseline_experiment",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)


class BaselineSplitConfig(BaseModel):
    """Temporal split settings for a baseline run."""

    model_config = _MODEL_CONFIG

    type: str = "temporal"
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    embargo: int = 0
    purge: bool = False

    @field_validator("type")
    @classmethod
    def _validate_type(cls, value: str) -> str:
        if value != "temporal":
            raise ValueError("baseline experiments currently only support temporal splits")
        return value

    @field_validator("train_fraction", "validation_fraction", "test_fraction")
    @classmethod
    def _validate_fraction(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("split fractions must be numeric")
        numeric = float(value)
        if numeric < 0.0:
            raise ValueError("split fractions must be non-negative")
        return numeric

    @field_validator("embargo")
    @classmethod
    def _validate_embargo(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("embargo must be an integer")
        if value < 0:
            raise ValueError("embargo must be non-negative")
        return value

    @model_validator(mode="after")
    def _validate_fraction_sum(self) -> BaselineSplitConfig:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1.0")
        return self


class BaselinePreprocessingConfig(BaseModel):
    """Preprocessing settings for a baseline run."""

    model_config = _MODEL_CONFIG

    standardise: bool = True


class BaselineExperimentConfig(BaseModel):
    """Configuration for one classical baseline experiment."""

    model_config = _MODEL_CONFIG

    run_name: str
    seed: int = 42
    target_column: str
    feature_columns: list[str] | None = None
    split: BaselineSplitConfig = Field(default_factory=BaselineSplitConfig)
    preprocessing: BaselinePreprocessingConfig = Field(
        default_factory=BaselinePreprocessingConfig
    )
    models: list[BaselineModelConfig] = Field(default_factory=list)

    @field_validator("run_name", "target_column")
    @classmethod
    def _validate_non_empty_string(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("run_name and target_column must be non-empty strings")
        return value

    @field_validator("seed")
    @classmethod
    def _validate_seed(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("seed must be an integer")
        if value < 0:
            raise ValueError("seed must be non-negative")
        return value

    @field_validator("models")
    @classmethod
    def _validate_models(cls, value: list[BaselineModelConfig]) -> list[BaselineModelConfig]:
        if not value:
            raise ValueError("models must contain at least one baseline")
        return list(value)


def create_default_baseline_configs(seed: int = 42) -> list[BaselineModelConfig]:
    """Return a fast, deterministic default set of classical baselines."""
    return [
        BaselineModelConfig(
            name="majority_class",
            model_type="majority_class",
            random_state=seed,
        ),
        BaselineModelConfig(
            name="logistic_regression",
            model_type="logistic_regression",
            random_state=seed,
        ),
        BaselineModelConfig(
            name="ridge_classifier",
            model_type="ridge_classifier",
            random_state=seed,
        ),
        BaselineModelConfig(
            name="random_forest",
            model_type="random_forest",
            random_state=seed,
            hyperparameters={"n_estimators": 20},
        ),
    ]


def _assert_temporal_order(frame: pd.DataFrame) -> None:
    if "timestamp" not in frame.columns or frame.empty:
        return
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("aligned baseline frame must be sorted by timestamp")


def _resolve_feature_columns(
    feature_frame: pd.DataFrame,
    configured_columns: list[str] | None,
) -> list[str]:
    if configured_columns is None:
        columns = select_feature_columns(feature_frame)
    else:
        columns = list(configured_columns)
        missing = [column for column in columns if column not in feature_frame.columns]
        if missing:
            raise ValueError(f"feature columns are missing from frame: {missing}")
        select_feature_columns(
            feature_frame.loc[:, columns],
            reject_label_like=True,
        )
    if not columns:
        raise ValueError("baseline experiment requires at least one feature column")
    return columns


def _make_temporal_split(
    n_rows: int,
    config: BaselineSplitConfig,
    aligned_frame: pd.DataFrame,
) -> Any:
    split = temporal_train_validation_test_split(
        n_rows,
        TemporalSplitConfig(
            train_fraction=config.train_fraction,
            validation_fraction=config.validation_fraction,
            test_fraction=config.test_fraction,
            min_train_size=1,
            min_validation_size=1,
            min_test_size=0,
        ),
    )
    if config.purge or config.embargo > 0:
        if "horizon_end" not in aligned_frame.columns:
            raise ValueError(
                "purge or embargo was requested but label horizon metadata is absent"
            )
        horizon_end_by_index = make_label_horizon_end_indices_from_frame(aligned_frame)
        split = apply_purge_and_embargo(
            split,
            horizon_end_by_index,
            PurgedEmbargoConfig(purge=config.purge, embargo=config.embargo),
            purge_against="validation_test",
        )
    return split


def _transform_with_train_only_scaler(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    test_x: np.ndarray | None,
    *,
    standardise: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    if not standardise:
        return train_x, validation_x, test_x, {"standardise": False}

    scaler = TrainOnlyStandardScaler()
    transformed_train = scaler.fit_transform(train_x)
    transformed_validation = scaler.transform(validation_x)
    transformed_test = None if test_x is None else scaler.transform(test_x)
    return (
        transformed_train,
        transformed_validation,
        transformed_test,
        {
            "standardise": True,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
    )


def _split_sizes(split: Any) -> dict[str, int]:
    return {
        "train": split.n_train,
        "validation": split.n_validation,
        "test": split.n_test,
    }


def _evaluate_models(
    config: BaselineExperimentConfig,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray | None,
    y_test: np.ndarray | None,
    labels: list[str | int | float] | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for model_config in config.models:
        model = create_baseline_model(model_config)
        model.fit(x_train, y_train)
        model_result: dict[str, Any] = {
            "name": model_config.name,
            "model_type": model_config.model_type,
            "validation": evaluate_classifier(
                model,
                x_validation,
                y_validation,
                labels=labels,
            ),
        }
        if x_test is not None and y_test is not None:
            model_result["test"] = evaluate_classifier(
                model,
                x_test,
                y_test,
                labels=labels,
            )
        results.append(model_result)
    return results


def _write_baseline_outputs(
    result: dict[str, Any],
    config: BaselineExperimentConfig,
    *,
    output_root: str | Path | None,
) -> Path:
    root = Path(output_root) if output_root is not None else project_root() / "runs"
    metadata, run_dir = initialise_experiment_run(
        root=root,
        run_name=config.run_name,
        phase="phase-6",
        seed=config.seed,
        notes="classical baseline experiment",
    )
    result["run_id"] = metadata.run_id
    result["output_path"] = str(run_dir)
    write_json(run_dir / "configs" / "baseline_config.json", config.model_dump(mode="json"))
    write_json(run_dir / "metrics.json", result)
    return run_dir


def run_baseline_experiment(
    feature_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
    config: BaselineExperimentConfig,
    *,
    output_root: str | Path | None = None,
    write_outputs: bool = False,
) -> dict[str, Any]:
    """Run classical baselines under train-only preprocessing and temporal splits."""
    leakage_result = validate_no_lookahead(feature_frame, label_frame)
    leakage_result.raise_if_errors()

    feature_columns = _resolve_feature_columns(feature_frame, config.feature_columns)
    aligned = align_feature_label_frames(feature_frame, label_frame)
    if aligned.empty:
        raise ValueError("feature and label frames have no aligned rows")
    _assert_temporal_order(aligned)

    split = _make_temporal_split(len(aligned), config.split, aligned)
    if split.n_train == 0:
        raise ValueError("temporal split produced no training rows")
    if split.n_validation == 0:
        raise ValueError("temporal split produced no validation rows")

    train_features = build_feature_matrix(
        aligned,
        feature_columns=feature_columns,
        row_indices=split.train,
    )
    validation_features = build_feature_matrix(
        aligned,
        feature_columns=feature_columns,
        row_indices=split.validation,
    )
    test_features = (
        None
        if split.n_test == 0
        else build_feature_matrix(
            aligned,
            feature_columns=feature_columns,
            row_indices=split.test,
        )
    )

    train_target = build_target_vector(
        aligned,
        target_column=config.target_column,
        row_indices=split.train,
    )
    validation_target = build_target_vector(
        aligned,
        target_column=config.target_column,
        row_indices=split.validation,
    )
    test_target = (
        None
        if split.n_test == 0
        else build_target_vector(
            aligned,
            target_column=config.target_column,
            row_indices=split.test,
        )
    )
    all_target = build_target_vector(aligned, target_column=config.target_column)

    x_train, x_validation, x_test, preprocessing_result = _transform_with_train_only_scaler(
        train_features.x,
        validation_features.x,
        None if test_features is None else test_features.x,
        standardise=config.preprocessing.standardise,
    )
    model_results = _evaluate_models(
        config,
        x_train=x_train,
        y_train=train_target.y,
        x_validation=x_validation,
        y_validation=validation_target.y,
        x_test=x_test,
        y_test=None if test_target is None else test_target.y,
        labels=all_target.classes,
    )

    result: dict[str, Any] = {
        "run_name": config.run_name,
        "seed": config.seed,
        "target_column": config.target_column,
        "split_sizes": _split_sizes(split),
        "feature_columns": list(feature_columns),
        "preprocessing": preprocessing_result,
        "models": model_results,
        "write_outputs": write_outputs,
    }
    if write_outputs:
        _write_baseline_outputs(result, config, output_root=output_root)
    return result
