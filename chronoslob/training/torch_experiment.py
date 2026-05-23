"""DeepLOB-style supervised neural smoke experiment runner.

This module wires the existing PyTorch sequence data layer to the
DeepLOB-style baseline introduced in
:mod:`chronoslob.models.deeplob`. The experiment runner aligns feature
and label frames, validates leakage controls, builds a temporal split,
fits train-only standardisation, constructs sequence dataloaders, trains
a small CNN-LSTM classifier and returns an in-memory result dictionary.
Output files are only written when ``write_outputs=True``; no model
checkpoints are written in this phase.

Nothing in this module implements transformers, self-supervised
pretraining, execution backtests, PnL or trading logic. The defaults
target deterministic CPU smoke tests on the bundled synthetic FI-2010
fixture.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from chronoslob.labels.leakage import validate_no_lookahead
from chronoslob.models.deeplob import (
    DeepLOBConfig,
    DeepLOBModel,
    create_deeplob_model,
)
from chronoslob.models.preprocessing import (
    align_feature_label_frames,
    select_feature_columns,
)
from chronoslob.training.artifacts import write_json
from chronoslob.training.dataloaders import (
    DataLoaderConfig,
    build_dataloaders_for_split,
)
from chronoslob.training.datasets import (
    SequenceWindowConfig,
    encode_target_values,
)
from chronoslob.training.experiment import initialise_experiment_run
from chronoslob.training.splitters import (
    TemporalSplitConfig,
    temporal_train_validation_test_split,
)
from chronoslob.training.torch_training import (
    TorchEpochResult,
    TorchTrainingConfig,
    evaluate_torch_classifier,
    fit_torch_classifier,
)
from chronoslob.utils.paths import project_root

try:  # pragma: no cover - exercised when torch is unavailable
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    import torch as _torch_typing  # noqa: F401

__all__ = [
    "DeepLOBExperimentConfig",
    "run_deeplob_experiment",
    "run_deeplob_smoke_from_fi2010_fixture",
]

_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=False,
    validate_assignment=True,
    arbitrary_types_allowed=True,
)

_SYNTHETIC_SMOKE_WARNING = (
    "Synthetic fixture smoke test only; not benchmark performance."
)


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the DeepLOB experiment runner. Install "
            "the 'torch' optional dependency: pip install -e '.[torch]'"
        )
    return torch


class DeepLOBExperimentConfig(BaseModel):
    """Configuration for a DeepLOB-style supervised neural experiment."""

    model_config = _MODEL_CONFIG

    run_name: str
    seed: int = 42
    target_column: str
    lookback: int = 10
    batch_size: int = 32
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    standardise: bool = True
    feature_columns: list[str] | None = None
    model: DeepLOBConfig | None = None
    training: TorchTrainingConfig = Field(default_factory=TorchTrainingConfig)
    write_outputs: bool = False

    @field_validator("run_name", "target_column")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("seed")
    @classmethod
    def _validate_seed(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("seed must be an integer")
        if value < 0:
            raise ValueError("seed must be non-negative")
        return value

    @field_validator("lookback", "batch_size")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("must be an integer")
        if value <= 0:
            raise ValueError("must be positive")
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

    @model_validator(mode="after")
    def _validate_fraction_sum(self) -> DeepLOBExperimentConfig:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1.0")
        return self


def _resolve_feature_columns(
    feature_frame: pd.DataFrame,
    configured: Sequence[str] | None,
) -> list[str]:
    if configured is None:
        columns = select_feature_columns(feature_frame)
    else:
        columns = list(configured)
        missing = [column for column in columns if column not in feature_frame.columns]
        if missing:
            raise ValueError(f"feature columns missing from frame: {missing}")
        select_feature_columns(
            feature_frame.loc[:, columns],
            reject_label_like=True,
        )
    if not columns:
        raise ValueError("DeepLOB experiment requires at least one feature column")
    return columns


def _make_temporal_split(
    n_rows: int,
    config: DeepLOBExperimentConfig,
) -> Any:
    return temporal_train_validation_test_split(
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


def _fit_train_only_mean_std(
    aligned_frame: pd.DataFrame,
    train_indices: Sequence[int],
    feature_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_values = (
        aligned_frame.iloc[list(train_indices)]
        .loc[:, list(feature_columns)]
        .to_numpy(dtype=np.float64, copy=True)
    )
    if train_values.shape[0] == 0:
        raise ValueError("train indices produced no rows for standardisation")
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0, ddof=0)
    std = np.where(std == 0.0, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)


def _apply_standardisation(
    feature_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> pd.DataFrame:
    transformed = feature_frame.copy()
    values = transformed.loc[:, list(feature_columns)].to_numpy(
        dtype=np.float64, copy=True
    )
    values = (values - mean) / std
    transformed.loc[:, list(feature_columns)] = values
    return transformed


def _epoch_result_to_dict(result: TorchEpochResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "epoch": int(result.epoch),
        "train_loss": float(result.train_loss),
    }
    if result.validation_loss is not None:
        payload["validation_loss"] = float(result.validation_loss)
    if result.validation_metrics is not None:
        payload["validation_metrics"] = dict(result.validation_metrics)
    return payload


def _serialise_evaluation(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "loss": float(result["loss"]),
        "metrics": dict(result["metrics"]),
        "confusion_matrix": dict(result["confusion_matrix"]),
        "n_samples": int(result["metrics"]["n_samples"]),
    }


def _serialise_training_config(config: TorchTrainingConfig) -> dict[str, Any]:
    return {
        "epochs": int(config.epochs),
        "learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "gradient_clip_norm": (
            None if config.gradient_clip_norm is None else float(config.gradient_clip_norm)
        ),
        "device": str(config.device),
        "seed": int(config.seed),
        "log_every": int(config.log_every),
    }


def _serialise_model_config(config: DeepLOBConfig) -> dict[str, Any]:
    return dict(asdict(config).items())


def _serialise_class_mapping(mapping: Mapping[Any, int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in mapping.items()}


def _resolve_model_config(
    requested: DeepLOBConfig | None,
    *,
    n_features: int,
    n_classes: int,
) -> DeepLOBConfig:
    if requested is None:
        return DeepLOBConfig(input_features=n_features, n_classes=n_classes)
    if requested.input_features != n_features:
        raise ValueError(
            "DeepLOBConfig.input_features does not match dataset feature count: "
            f"expected {n_features}, got {requested.input_features}"
        )
    if requested.n_classes != n_classes:
        raise ValueError(
            "DeepLOBConfig.n_classes does not match observed class count: "
            f"expected {n_classes}, got {requested.n_classes}"
        )
    return requested


def _evaluate_loader_if_present(
    model: DeepLOBModel,
    loaders: Mapping[str, Any],
    name: str,
    *,
    device: str,
    metric_labels: Sequence[int],
) -> dict[str, Any] | None:
    if name not in loaders:
        return None
    raw = evaluate_torch_classifier(
        model,
        loaders[name],
        device=device,
        labels=list(metric_labels),
    )
    return _serialise_evaluation(raw)


def _write_outputs(
    *,
    result: dict[str, Any],
    config: DeepLOBExperimentConfig,
    output_root: str | Path | None,
) -> Path:
    root = Path(output_root) if output_root is not None else project_root() / "runs"
    metadata, run_dir = initialise_experiment_run(
        root=root,
        run_name=config.run_name,
        phase="phase-7b",
        seed=config.seed,
        notes="DeepLOB-style supervised neural smoke experiment",
    )
    result["run_id"] = metadata.run_id
    result["output_path"] = str(run_dir)
    write_json(
        run_dir / "configs" / "deeplob_config.json",
        config.model_dump(mode="json"),
    )
    write_json(run_dir / "metrics.json", result)
    return run_dir


def run_deeplob_experiment(
    feature_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
    config: DeepLOBExperimentConfig,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run a DeepLOB-style supervised smoke experiment end to end.

    Returns an in-memory result dictionary. Set ``config.write_outputs``
    to ``True`` to additionally write metadata, config and metrics under
    ``output_root`` (defaults to the project-root ``runs`` directory). No
    model checkpoints are written in any case.
    """
    _require_torch()
    if not isinstance(feature_frame, pd.DataFrame):
        raise TypeError("feature_frame must be a pandas DataFrame")
    if not isinstance(label_frame, pd.DataFrame):
        raise TypeError("label_frame must be a pandas DataFrame")
    if not isinstance(config, DeepLOBExperimentConfig):
        raise TypeError("config must be a DeepLOBExperimentConfig instance")

    leakage_result = validate_no_lookahead(feature_frame, label_frame)
    leakage_result.raise_if_errors()

    feature_columns = _resolve_feature_columns(feature_frame, config.feature_columns)

    aligned = align_feature_label_frames(feature_frame, label_frame)
    if aligned.empty:
        raise ValueError("feature and label frames have no aligned rows")

    split = _make_temporal_split(len(aligned), config)
    if split.n_train == 0:
        raise ValueError("temporal split produced no training rows")
    if split.n_validation == 0:
        raise ValueError("temporal split produced no validation rows")

    if config.standardise:
        mean, std = _fit_train_only_mean_std(aligned, split.train, feature_columns)
        scaled_feature_frame = _apply_standardisation(
            feature_frame, feature_columns, mean, std
        )
        scaler_metadata: dict[str, Any] = {
            "standardise": True,
            "mean": [float(value) for value in mean.tolist()],
            "std": [float(value) for value in std.tolist()],
            "feature_columns": list(feature_columns),
        }
    else:
        scaled_feature_frame = feature_frame
        scaler_metadata = {"standardise": False}

    train_target_values = (
        aligned.iloc[list(split.train)].loc[:, config.target_column].tolist()
    )
    _, class_mapping = encode_target_values(train_target_values)
    sequence_config = SequenceWindowConfig(
        lookback=config.lookback,
        target_column=config.target_column,
        feature_columns=list(feature_columns),
    )
    loader_config = DataLoaderConfig(
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )
    loaders = build_dataloaders_for_split(
        scaled_feature_frame,
        label_frame,
        split,
        sequence_config,
        loader_config,
        class_to_index=class_mapping,
    )

    train_loader = loaders["train"]
    validation_loader = loaders.get("validation")
    test_loader = loaders.get("test")

    n_features = train_loader.dataset.n_features
    n_classes = max(2, len(class_mapping))
    model_config = _resolve_model_config(
        config.model,
        n_features=n_features,
        n_classes=n_classes,
    )
    model = create_deeplob_model(model_config)

    training_config = config.training
    if training_config.device.strip().lower() != "cpu":
        # CPU is the only path exercised by smoke tests. Other devices are
        # validated lazily inside the torch training utilities.
        warnings.warn(
            "DeepLOB smoke experiment runs on CPU by default; non-CPU devices "
            "are best-effort and not exercised by automated tests.",
            RuntimeWarning,
            stacklevel=2,
        )

    history = fit_torch_classifier(
        model,
        train_loader,
        validation_loader,
        training_config,
    )

    metric_labels = sorted(class_mapping.values())
    validation_metrics = _evaluate_loader_if_present(
        model,
        loaders,
        "validation",
        device=training_config.device,
        metric_labels=metric_labels,
    )
    test_metrics = _evaluate_loader_if_present(
        model,
        loaders,
        "test",
        device=training_config.device,
        metric_labels=metric_labels,
    )

    split_sizes = {
        "train": int(split.n_train),
        "validation": int(split.n_validation),
        "test": int(split.n_test),
    }
    sample_counts = {
        "train": len(train_loader.dataset),
        "validation": len(validation_loader.dataset)
        if validation_loader is not None
        else 0,
        "test": len(test_loader.dataset) if test_loader is not None else 0,
    }

    result: dict[str, Any] = {
        "run_name": config.run_name,
        "seed": config.seed,
        "target_column": config.target_column,
        "lookback": int(config.lookback),
        "batch_size": int(config.batch_size),
        "feature_columns": list(feature_columns),
        "feature_count": int(n_features),
        "class_mapping": _serialise_class_mapping(class_mapping),
        "n_classes": int(n_classes),
        "split_sizes": split_sizes,
        "sample_counts": sample_counts,
        "standardisation": scaler_metadata,
        "model_config": _serialise_model_config(model_config),
        "model_parameter_count": int(model.n_parameters()),
        "training_config": _serialise_training_config(training_config),
        "training_history": [_epoch_result_to_dict(item) for item in history],
        "final_validation_metrics": validation_metrics,
        "final_test_metrics": test_metrics,
        "write_outputs": bool(config.write_outputs),
        "notes": _SYNTHETIC_SMOKE_WARNING,
    }

    if config.write_outputs:
        _write_outputs(result=result, config=config, output_root=output_root)

    return result


def run_deeplob_smoke_from_fi2010_fixture(
    path: str | Path,
    *,
    lookback: int = 2,
    seed: int = 42,
    epochs: int = 1,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Run a tiny DeepLOB smoke experiment on the bundled FI-2010 fixture.

    The fixture is synthetic and intentionally small. The result is a
    pipeline smoke test only and must never be reported as benchmark
    performance — the returned dictionary carries a ``notes`` field that
    states this explicitly.
    """
    _require_torch()
    from chronoslob.data.fi2010 import FI2010Config, load_fi2010
    from chronoslob.features.pipeline import (
        FeaturePipelineConfig,
        build_feature_frame_from_fi2010,
    )
    from chronoslob.labels.pipeline import build_label_frame_from_fi2010

    label_columns = ["label_10", "label_50", "label_100"]
    dataset = load_fi2010(
        FI2010Config(
            path=Path(path),
            timestamp_column="timestamp",
            split_column="split",
            label_columns=label_columns,
            price_level_count=2,
        )
    )
    feature_frame = build_feature_frame_from_fi2010(
        dataset,
        FeaturePipelineConfig(
            include_order_flow=False,
            include_volatility=False,
        ),
    )
    labels = build_label_frame_from_fi2010(dataset, prefer_existing_labels=True)
    label_frame = labels.loc[:, ["timestamp", "symbol", "label_10"]]

    # Fixture is too small to satisfy the default 70/15/15 split with a
    # sensible test partition, so the smoke command uses larger train and
    # validation fractions while keeping a test fraction of zero.
    config = DeepLOBExperimentConfig(
        run_name="synthetic-fi2010-deeplob-smoke",
        seed=seed,
        target_column="label_10",
        lookback=lookback,
        batch_size=batch_size,
        train_fraction=0.5,
        validation_fraction=0.5,
        test_fraction=0.0,
        training=TorchTrainingConfig(
            epochs=epochs,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            device="cpu",
            seed=seed,
        ),
        write_outputs=False,
    )

    # The fixture's train partition does not see every class. Reuse the
    # smoke pattern from inspect-torch-dataset and pre-populate the class
    # mapping from the full aligned frame so the smoke command demonstrates
    # the data layer end to end without claiming train-only fitting.
    aligned = align_feature_label_frames(feature_frame, label_frame)
    _, full_mapping = encode_target_values(aligned.loc[:, "label_10"].tolist())
    n_classes = max(2, len(full_mapping))
    feature_columns = _resolve_feature_columns(feature_frame, config.feature_columns)
    config_with_model = config.model_copy(
        update={
            "model": DeepLOBConfig(
                input_features=len(feature_columns),
                n_classes=n_classes,
            )
        }
    )

    # We deliberately bypass run_deeplob_experiment for this smoke path so
    # the synthetic class mapping can come from the full aligned frame
    # without contaminating the production experiment runner.
    return _run_smoke_with_explicit_mapping(
        feature_frame=feature_frame,
        label_frame=label_frame,
        config=config_with_model,
        class_mapping=full_mapping,
        feature_columns=feature_columns,
    )


def _run_smoke_with_explicit_mapping(
    *,
    feature_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
    config: DeepLOBExperimentConfig,
    class_mapping: Mapping[Any, int],
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    leakage_result = validate_no_lookahead(feature_frame, label_frame)
    leakage_result.raise_if_errors()

    aligned = align_feature_label_frames(feature_frame, label_frame)
    if aligned.empty:
        raise ValueError("feature and label frames have no aligned rows")
    split = _make_temporal_split(len(aligned), config)
    if split.n_train == 0 or split.n_validation == 0:
        raise ValueError("synthetic fixture produced an empty split partition")

    if config.standardise:
        mean, std = _fit_train_only_mean_std(aligned, split.train, feature_columns)
        scaled_feature_frame = _apply_standardisation(
            feature_frame, feature_columns, mean, std
        )
        scaler_metadata: dict[str, Any] = {
            "standardise": True,
            "mean": [float(value) for value in mean.tolist()],
            "std": [float(value) for value in std.tolist()],
            "feature_columns": list(feature_columns),
        }
    else:
        scaled_feature_frame = feature_frame
        scaler_metadata = {"standardise": False}

    sequence_config = SequenceWindowConfig(
        lookback=config.lookback,
        target_column=config.target_column,
        feature_columns=list(feature_columns),
    )
    loader_config = DataLoaderConfig(
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )
    loaders = build_dataloaders_for_split(
        scaled_feature_frame,
        label_frame,
        split,
        sequence_config,
        loader_config,
        class_to_index=class_mapping,
    )

    train_loader = loaders["train"]
    validation_loader = loaders.get("validation")

    n_features = train_loader.dataset.n_features
    n_classes = max(2, len(class_mapping))
    model_config = _resolve_model_config(
        config.model,
        n_features=n_features,
        n_classes=n_classes,
    )
    model = create_deeplob_model(model_config)

    history = fit_torch_classifier(
        model,
        train_loader,
        validation_loader,
        config.training,
    )

    metric_labels = sorted(class_mapping.values())
    validation_metrics = _evaluate_loader_if_present(
        model,
        loaders,
        "validation",
        device=config.training.device,
        metric_labels=metric_labels,
    )

    return {
        "run_name": config.run_name,
        "seed": config.seed,
        "target_column": config.target_column,
        "lookback": int(config.lookback),
        "batch_size": int(config.batch_size),
        "feature_columns": list(feature_columns),
        "feature_count": int(n_features),
        "class_mapping": _serialise_class_mapping(class_mapping),
        "n_classes": int(n_classes),
        "split_sizes": {
            "train": int(split.n_train),
            "validation": int(split.n_validation),
            "test": int(split.n_test),
        },
        "sample_counts": {
            "train": len(train_loader.dataset),
            "validation": len(validation_loader.dataset)
            if validation_loader is not None
            else 0,
            "test": 0,
        },
        "standardisation": scaler_metadata,
        "model_config": _serialise_model_config(model_config),
        "model_parameter_count": int(model.n_parameters()),
        "training_config": _serialise_training_config(config.training),
        "training_history": [_epoch_result_to_dict(item) for item in history],
        "final_validation_metrics": validation_metrics,
        "final_test_metrics": None,
        "write_outputs": False,
        "notes": _SYNTHETIC_SMOKE_WARNING,
        "class_mapping_source": "full_aligned_frame_for_synthetic_fixture",
    }
