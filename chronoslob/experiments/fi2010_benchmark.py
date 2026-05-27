"""FI-2010 local benchmark preparation layer.

This module turns a user-supplied local FI-2010-style file into a
documented, validated benchmark input. It produces a data manifest,
label-distribution summary, split summary and a leakage-aware
validation summary in a preparation output directory.

The layer is deliberately read-only with respect to the source data and
never downloads, uploads or commits market data. It is also not a model
runner: it does not produce ``results.json`` or any prediction artefact.
Those belong to the later paper experiment runner phase.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from chronoslob.data.fi2010 import FI2010Config, FI2010Dataset, load_fi2010
from chronoslob.data.schemas import ensure_utc_datetime
from chronoslob.data.validation import validate_fi2010_dataset
from chronoslob.experiments.evidence import (
    CalibrationConfig,
    ExecutionSensitivityConfig,
)
from chronoslob.experiments.manifests import (
    build_local_file_manifest,
    stable_json_dumps,
)
from chronoslob.experiments.schemas import DataManifest
from chronoslob.training.splitters import (
    SplitIndices,
    TemporalSplitConfig,
    temporal_train_validation_test_split,
)

__all__ = [
    "BenchmarkSplitConfig",
    "FI2010BenchmarkConfig",
    "FI2010PreparationResult",
    "FI2010PreparationSummary",
    "LabelSummary",
    "PaperNeuralSettings",
    "SplitSummary",
    "ValidationSummary",
    "build_benchmark_split",
    "effective_split_column",
    "effective_split_name",
    "effective_split_summary",
    "load_benchmark_config",
    "prepare_fi2010_benchmark",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)
_LOCAL_PATH_PLACEHOLDER = "<path-to-local-fi2010-file>"
_DEFAULT_LABEL_COLUMNS = ("label_10", "label_50", "label_100")
_PREPARATION_VERSION = "phase-b/fi2010-benchmark-preparation/v1"
_TEMPORAL_SPLIT_METHOD = "temporal"
_OFFICIAL_COLUMN_SPLIT_METHOD = "official_column"
_OFFICIAL_ROW_ORDER_NOTE = (
    "no timestamp column configured; row order is preserved within "
    "official split partitions"
)


# ---------------------------------------------------------------------------
# Typed schemas
# ---------------------------------------------------------------------------


class LabelSummary(BaseModel):
    """Label-distribution summary for the chosen target column."""

    model_config = _MODEL_CONFIG

    label_name: str
    horizon: int
    row_count: int
    distinct_classes: list[str]
    class_counts: dict[str, int]
    class_proportions: dict[str, float]
    label_source: str

    @field_validator("label_name", "label_source")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("label summary field must be a non-empty string")
        return value.strip()

    @field_validator("horizon")
    @classmethod
    def _validate_horizon(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("horizon must be an integer")
        if value <= 0:
            raise ValueError("horizon must be positive")
        return value

    @field_validator("row_count")
    @classmethod
    def _validate_row_count(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("row_count must be an integer")
        if value < 0:
            raise ValueError("row_count must be non-negative")
        return value

    @field_validator("distinct_classes")
    @classmethod
    def _validate_distinct_classes(cls, value: list[str]) -> list[str]:
        cleaned = [str(item) for item in value]
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("distinct_classes must not contain duplicates")
        return cleaned

    @model_validator(mode="after")
    def _validate_counts_and_proportions(self) -> LabelSummary:
        expected = set(self.distinct_classes)
        if set(self.class_counts) != expected:
            raise ValueError(
                "class_counts keys must match distinct_classes",
            )
        if set(self.class_proportions) != expected:
            raise ValueError(
                "class_proportions keys must match distinct_classes",
            )
        total = sum(self.class_counts.values())
        if total != self.row_count:
            raise ValueError(
                "class_counts must sum to row_count",
            )
        for proportion in self.class_proportions.values():
            if proportion < 0.0 or proportion > 1.0:
                raise ValueError("class proportions must lie in [0, 1]")
        return self


class SplitSummary(BaseModel):
    """Temporal split summary for the prepared benchmark inputs."""

    model_config = _MODEL_CONFIG

    split_name: str
    split_method: str = _TEMPORAL_SPLIT_METHOD
    n_rows: int
    n_train: int
    n_validation: int
    n_test: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    split_column: str | None = None
    official_train_value: str | None = None
    official_test_value: str | None = None
    official_train_rows: int | None = None
    official_test_rows: int | None = None
    official_train_start_index: int | None = None
    official_train_end_index: int | None = None
    official_test_start_index: int | None = None
    official_test_end_index: int | None = None
    validation_fraction_within_train: float | None = None
    train_start_index: int | None = None
    train_end_index: int | None = None
    validation_start_index: int | None = None
    validation_end_index: int | None = None
    test_start_index: int | None = None
    test_end_index: int | None = None

    @field_validator("split_name")
    @classmethod
    def _validate_split_name(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("split_name must be a non-empty string")
        return value.strip()

    @field_validator("split_method")
    @classmethod
    def _validate_split_method(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("split_method must be a non-empty string")
        return value.strip().lower()

    @field_validator(
        "n_rows",
        "n_train",
        "n_validation",
        "n_test",
    )
    @classmethod
    def _validate_counts(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("split counts must be integers")
        if value < 0:
            raise ValueError("split counts must be non-negative")
        return value

    @field_validator("official_train_rows", "official_test_rows")
    @classmethod
    def _validate_optional_counts(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("official split counts must be integers when provided")
        if value < 0:
            raise ValueError("official split counts must be non-negative")
        return value

    @field_validator(
        "train_fraction",
        "validation_fraction",
        "test_fraction",
    )
    @classmethod
    def _validate_fraction(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("split fractions must be finite numbers")
        numeric = float(value)
        if numeric < 0.0 or numeric > 1.0:
            raise ValueError("split fractions must lie in [0, 1]")
        return numeric

    @field_validator("validation_fraction_within_train")
    @classmethod
    def _validate_optional_fraction(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("validation_fraction_within_train must be finite")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0 or numeric >= 1.0:
            raise ValueError(
                "validation_fraction_within_train must satisfy 0 <= value < 1"
            )
        return numeric

    @model_validator(mode="after")
    def _validate_consistency(self) -> SplitSummary:
        if self.n_train + self.n_validation + self.n_test != self.n_rows:
            raise ValueError("split counts must sum to n_rows")
        return self


class ValidationSummary(BaseModel):
    """Leakage and data-quality validation summary."""

    model_config = _MODEL_CONFIG

    fi2010_validation_ok: bool
    fi2010_error_count: int
    fi2010_warning_count: int
    label_validation_ok: bool
    label_error_count: int
    label_warning_count: int
    notes: list[str] = Field(default_factory=list)

    @field_validator(
        "fi2010_error_count",
        "fi2010_warning_count",
        "label_error_count",
        "label_warning_count",
    )
    @classmethod
    def _validate_counts(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("validation counts must be integers")
        if value < 0:
            raise ValueError("validation counts must be non-negative")
        return value

    @field_validator("notes")
    @classmethod
    def _validate_notes(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for note in value:
            if not isinstance(note, str) or not note.strip():
                raise ValueError("notes must be non-empty strings")
            cleaned.append(note.strip())
        return cleaned


class FI2010PreparationSummary(BaseModel):
    """Top-level preparation summary written to ``preparation_summary.json``."""

    model_config = _MODEL_CONFIG

    experiment_name: str
    dataset_name: str
    task_name: str
    horizon: int
    split_name: str
    seed: int
    label_name: str
    data_path: str
    output_dir: str
    preparation_version: str
    created_at: datetime
    artefacts: dict[str, str]
    warnings: list[str] = Field(default_factory=list)

    @field_validator(
        "experiment_name",
        "dataset_name",
        "task_name",
        "split_name",
        "label_name",
        "data_path",
        "output_dir",
        "preparation_version",
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

    @field_validator("seed")
    @classmethod
    def _validate_seed(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("seed must be an integer")
        if value < 0:
            raise ValueError("seed must be non-negative")
        return value

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        return ensure_utc_datetime(value)

    @field_validator("artefacts")
    @classmethod
    def _validate_artefacts(cls, value: dict[str, str]) -> dict[str, str]:
        cleaned: dict[str, str] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("artefact keys must be non-empty strings")
            if not isinstance(item, str) or not item.strip():
                raise ValueError("artefact paths must be non-empty strings")
            cleaned[key.strip()] = item.strip()
        return cleaned

    @field_validator("warnings")
    @classmethod
    def _validate_warnings(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for warning in value:
            if not isinstance(warning, str) or not warning.strip():
                raise ValueError("warnings must be non-empty strings")
            cleaned.append(warning.strip())
        return cleaned


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class BenchmarkSplitConfig(BaseModel):
    """Split policy for FI-2010 benchmark preparation and paper runs.

    ``temporal`` preserves the existing row-order 70/15/15 behaviour.
    ``official_column`` uses the configured split column to keep official
    FI-2010 test rows out of all fitting and validation decisions, carving
    validation from the tail of the official training rows.
    """

    model_config = _MODEL_CONFIG

    method: str = _TEMPORAL_SPLIT_METHOD
    split_column: str | None = None
    validation_fraction_within_train: float = 0.15
    train_value: str = "train"
    test_value: str = "test"

    @field_validator("method")
    @classmethod
    def _validate_method(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("split method must be a non-empty string")
        cleaned = value.strip().lower()
        if cleaned not in {_TEMPORAL_SPLIT_METHOD, _OFFICIAL_COLUMN_SPLIT_METHOD}:
            raise ValueError(
                "split method must be one of "
                f"{[_TEMPORAL_SPLIT_METHOD, _OFFICIAL_COLUMN_SPLIT_METHOD]}"
            )
        return cleaned

    @field_validator("split_column")
    @classmethod
    def _validate_split_column(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("split_column must be non-empty when provided")
        return value.strip()

    @field_validator("validation_fraction_within_train")
    @classmethod
    def _validate_validation_fraction(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                "validation_fraction_within_train must be a finite number"
            )
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0 or numeric >= 1.0:
            raise ValueError(
                "validation_fraction_within_train must satisfy 0 <= value < 1"
            )
        return numeric

    @field_validator("train_value", "test_value")
    @classmethod
    def _validate_split_value(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("official split values must be non-empty strings")
        return value.strip()

    @model_validator(mode="after")
    def _validate_distinct_values(self) -> BenchmarkSplitConfig:
        if self.train_value.casefold() == self.test_value.casefold():
            raise ValueError("train_value and test_value must be distinct")
        return self


class PaperNeuralSettings(BaseModel):
    """Config-driven settings for neural paper-runner baselines."""

    model_config = _MODEL_CONFIG

    supported_models: tuple[str, ...] = (
        "deeplob_style",
        "transformer",
        "matrix_transformer",
    )
    planned_models: tuple[str, ...] = ("ssl_transformer",)
    lookback: int = 1
    transformer_window_length: int = 4
    batch_size: int = 4
    max_epochs: int = 1
    early_stopping_patience: int | None = None
    early_stopping_metric: str = "validation_loss"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = 1.0
    device: str = "cpu"
    deterministic: bool = True
    checkpoint_enabled: bool = False
    checkpoint_path: str | None = None
    dropout: float = 0.0
    deeplob_conv_channels: int = 4
    deeplob_lstm_hidden_size: int = 8
    deeplob_use_batch_norm: bool = False
    transformer_field_embedding_dim: int = 4
    transformer_model_dim: int = 16
    transformer_num_heads: int = 2
    transformer_num_layers: int = 1
    transformer_feedforward_dim: int = 32
    transformer_max_levels_per_side: int = 2

    @field_validator("supported_models", "planned_models")
    @classmethod
    def _validate_model_names(cls, value: Sequence[str]) -> tuple[str, ...]:
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("neural model names must be non-empty strings")
            normalised = item.strip().lower()
            if normalised != item.strip():
                raise ValueError("neural model names must be lower-case")
            cleaned.append(normalised)
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("neural model names must not contain duplicates")
        return tuple(cleaned)

    @field_validator(
        "lookback",
        "transformer_window_length",
        "batch_size",
        "max_epochs",
        "deeplob_conv_channels",
        "deeplob_lstm_hidden_size",
        "transformer_field_embedding_dim",
        "transformer_model_dim",
        "transformer_num_heads",
        "transformer_num_layers",
        "transformer_feedforward_dim",
        "transformer_max_levels_per_side",
    )
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("neural integer settings must be integers")
        if value <= 0:
            raise ValueError("neural integer settings must be positive")
        return value

    @field_validator("learning_rate")
    @classmethod
    def _validate_learning_rate(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("learning_rate must be a finite number")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError("learning_rate must be positive")
        return numeric

    @field_validator("weight_decay")
    @classmethod
    def _validate_weight_decay(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("weight_decay must be a finite number")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError("weight_decay must be non-negative")
        return numeric

    @field_validator("gradient_clip_norm")
    @classmethod
    def _validate_gradient_clip_norm(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("gradient_clip_norm must be a finite number or null")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError("gradient_clip_norm must be positive when provided")
        return numeric

    @field_validator("early_stopping_patience")
    @classmethod
    def _validate_early_stopping_patience(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("early_stopping_patience must be an integer or null")
        if value < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        return int(value)

    @field_validator("early_stopping_metric")
    @classmethod
    def _validate_early_stopping_metric(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("early_stopping_metric must be a non-empty string")
        cleaned = value.strip().lower()
        if cleaned not in {"validation_loss", "validation_macro_f1"}:
            raise ValueError(
                "early_stopping_metric must be one of "
                "['validation_loss', 'validation_macro_f1']"
            )
        return cleaned

    @field_validator("dropout")
    @classmethod
    def _validate_dropout(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("dropout must be a finite number")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0 or numeric >= 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        return numeric

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("device must be a non-empty string")
        cleaned = value.strip().lower()
        if cleaned not in {"auto", "cpu", "cuda"} and not cleaned.startswith("cuda:"):
            raise ValueError("device must be auto, cpu, cuda or cuda:<index>")
        return cleaned

    @field_validator("checkpoint_path")
    @classmethod
    def _validate_checkpoint_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("checkpoint_path must be non-empty when provided")
        return value.strip()

    @model_validator(mode="after")
    def _validate_transformer_heads(self) -> PaperNeuralSettings:
        if self.transformer_model_dim % self.transformer_num_heads != 0:
            raise ValueError(
                "transformer_model_dim must be divisible by transformer_num_heads"
            )
        return self


class FI2010BenchmarkConfig(BaseModel):
    """Validated FI-2010 benchmark preparation configuration."""

    model_config = _MODEL_CONFIG

    experiment_name: str
    dataset_name: str = "FI-2010"
    task_name: str = "midprice_direction"
    horizon: int
    split_name: str = "temporal"
    seed: int = 0
    local_data_path: str = _LOCAL_PATH_PLACEHOLDER
    output_dir: str
    label_name: str
    label_columns: tuple[str, ...] = _DEFAULT_LABEL_COLUMNS
    timestamp_column: str | None = "timestamp"
    split_column: str | None = "split"
    price_level_count: int = 10
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    split: BenchmarkSplitConfig = Field(default_factory=BenchmarkSplitConfig)
    notes: str | None = None
    classical_models_supported: tuple[str, ...] = ()
    models_planned: tuple[str, ...] = ()
    metrics_planned: tuple[str, ...] = ()
    execution_assumptions_planned: dict[str, Any] = Field(default_factory=dict)
    feature_patterns: tuple[str, ...] | None = None
    neural_settings: PaperNeuralSettings = Field(
        default_factory=PaperNeuralSettings
    )
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    execution_sensitivity: ExecutionSensitivityConfig = Field(
        default_factory=ExecutionSensitivityConfig
    )

    @field_validator(
        "experiment_name",
        "dataset_name",
        "task_name",
        "split_name",
        "label_name",
        "local_data_path",
        "output_dir",
    )
    @classmethod
    def _validate_required_string(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("config field must be a non-empty string")
        return value.strip()

    @field_validator("horizon")
    @classmethod
    def _validate_horizon(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("horizon must be an integer")
        if value <= 0:
            raise ValueError("horizon must be positive")
        return value

    @field_validator("seed")
    @classmethod
    def _validate_seed(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("seed must be an integer")
        if value < 0:
            raise ValueError("seed must be non-negative")
        return value

    @field_validator("price_level_count")
    @classmethod
    def _validate_price_level_count(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("price_level_count must be an integer")
        if value <= 0:
            raise ValueError("price_level_count must be positive")
        return value

    @field_validator("label_columns")
    @classmethod
    def _validate_label_columns(cls, value: Sequence[str]) -> tuple[str, ...]:
        cleaned: list[str] = []
        for column in value:
            if not isinstance(column, str) or not column.strip():
                raise ValueError("label_columns must contain non-empty strings")
            cleaned.append(column.strip())
        if not cleaned:
            raise ValueError("label_columns must contain at least one entry")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("label_columns must not contain duplicates")
        return tuple(cleaned)

    @field_validator(
        "train_fraction",
        "validation_fraction",
        "test_fraction",
    )
    @classmethod
    def _validate_fraction(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("split fractions must be finite numbers")
        numeric = float(value)
        if numeric < 0.0 or numeric > 1.0:
            raise ValueError("split fractions must lie in [0, 1]")
        return numeric

    @field_validator("notes")
    @classmethod
    def _validate_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("notes must be a non-empty string when provided")
        return value.strip()

    @field_validator(
        "classical_models_supported",
        "models_planned",
        "metrics_planned",
    )
    @classmethod
    def _validate_planned_lists(cls, value: Sequence[str]) -> tuple[str, ...]:
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("planned entries must be non-empty strings")
            cleaned.append(item.strip())
        return tuple(cleaned)

    @field_validator("feature_patterns")
    @classmethod
    def _validate_feature_patterns(
        cls, value: Sequence[str] | None
    ) -> tuple[str, ...] | None:
        if value is None:
            return None
        if isinstance(value, str):
            raise TypeError("feature_patterns must be a sequence of strings, not a string")
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("feature_patterns entries must be non-empty strings")
            cleaned.append(item.strip())
        if not cleaned:
            raise ValueError(
                "feature_patterns must contain at least one entry when provided"
            )
        return tuple(cleaned)

    @model_validator(mode="after")
    def _validate_label_membership_and_fractions(self) -> FI2010BenchmarkConfig:
        if self.label_name not in self.label_columns:
            raise ValueError(
                "label_name must be one of label_columns",
            )
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1.0")
        return self

    @property
    def data_path_is_placeholder(self) -> bool:
        """Return ``True`` when ``local_data_path`` is the safe placeholder."""
        return self.local_data_path == _LOCAL_PATH_PLACEHOLDER


def load_benchmark_config(path: Path) -> FI2010BenchmarkConfig:
    """Load and validate an FI-2010 benchmark preparation YAML config."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"benchmark config not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"benchmark config must be a YAML mapping: {config_path}")
    return FI2010BenchmarkConfig.model_validate(payload)


def effective_split_name(config: FI2010BenchmarkConfig) -> str:
    """Return the split name used by manifests, runners and reports."""
    if config.split.method == _OFFICIAL_COLUMN_SPLIT_METHOD:
        return _OFFICIAL_COLUMN_SPLIT_METHOD
    return config.split_name


def effective_split_column(config: FI2010BenchmarkConfig) -> str | None:
    """Return the split column selected by the active split policy."""
    return config.split.split_column or config.split_column


def _bounds(values: Sequence[int]) -> tuple[int | None, int | None]:
    if not values:
        return None, None
    return int(values[0]), int(values[-1])


def _fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _official_validation_count(
    official_train_count: int,
    validation_fraction: float,
) -> int:
    if validation_fraction <= 0.0 or official_train_count <= 1:
        return 0
    raw_count = math.ceil(float(official_train_count) * validation_fraction)
    return min(max(raw_count, 1), official_train_count - 1)


def _normalised_split_values(frame: pd.DataFrame, split_column: str) -> list[str]:
    values: list[str] = []
    for position, raw in enumerate(frame[split_column].tolist()):
        if pd.isna(raw):
            raise ValueError(
                f"split column {split_column!r} contains a missing value "
                f"at row {position}"
            )
        text = str(raw).strip()
        if not text:
            raise ValueError(
                f"split column {split_column!r} contains an empty value "
                f"at row {position}"
            )
        values.append(text.casefold())
    return values


def build_benchmark_split(
    config: FI2010BenchmarkConfig,
    frame: pd.DataFrame,
) -> SplitIndices:
    """Build train/validation/test indices for the configured split policy."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    method = config.split.method
    if method == _TEMPORAL_SPLIT_METHOD:
        split_config = TemporalSplitConfig(
            train_fraction=config.train_fraction,
            validation_fraction=config.validation_fraction,
            test_fraction=config.test_fraction,
            min_train_size=1,
            min_validation_size=1,
            min_test_size=0,
        )
        return temporal_train_validation_test_split(len(frame), split_config)

    if method != _OFFICIAL_COLUMN_SPLIT_METHOD:
        raise ValueError(f"unsupported split method {method!r}")

    split_column = effective_split_column(config)
    if split_column is None:
        raise ValueError(
            "official_column split requires split.split_column or split_column"
        )
    if split_column not in frame.columns:
        raise ValueError(
            f"official_column split column {split_column!r} is missing"
        )

    train_value = config.split.train_value.casefold()
    test_value = config.split.test_value.casefold()
    values = _normalised_split_values(frame, split_column)
    allowed = {train_value, test_value}
    unexpected = sorted({value for value in values if value not in allowed})
    if unexpected:
        raise ValueError(
            "official_column split contains values outside the configured "
            f"train/test set: {unexpected}"
        )

    official_train = [
        row_index for row_index, value in enumerate(values) if value == train_value
    ]
    official_test = [
        row_index for row_index, value in enumerate(values) if value == test_value
    ]
    if not official_train:
        raise ValueError("official_column split contains no training rows")
    if not official_test:
        raise ValueError("official_column split contains no test rows")

    validation_count = _official_validation_count(
        len(official_train),
        config.split.validation_fraction_within_train,
    )
    if validation_count > 0:
        train = official_train[:-validation_count]
        validation = official_train[-validation_count:]
    else:
        train = list(official_train)
        validation = []
    if not train:
        raise ValueError(
            "official_column split left no training rows after validation carve-out"
        )
    return SplitIndices(train=train, validation=validation, test=official_test)


def effective_split_summary(
    *,
    config: FI2010BenchmarkConfig,
    frame: pd.DataFrame,
    indices: SplitIndices,
) -> SplitSummary:
    """Return a serialisable summary for the active split policy."""
    n_rows = len(frame)
    train_start, train_end = _bounds(indices.train)
    val_start, val_end = _bounds(indices.validation)
    test_start, test_end = _bounds(indices.test)

    split_column = effective_split_column(config)
    method = config.split.method
    official_train: list[int] = []
    official_test: list[int] = []
    if method == _OFFICIAL_COLUMN_SPLIT_METHOD:
        if split_column is None or split_column not in frame.columns:
            raise ValueError("official split summary requires a valid split column")
        values = _normalised_split_values(frame, split_column)
        train_value = config.split.train_value.casefold()
        test_value = config.split.test_value.casefold()
        official_train = [
            row_index
            for row_index, value in enumerate(values)
            if value == train_value
        ]
        official_test = [
            row_index
            for row_index, value in enumerate(values)
            if value == test_value
        ]

    official_train_start, official_train_end = _bounds(official_train)
    official_test_start, official_test_end = _bounds(official_test)
    return SplitSummary(
        split_name=effective_split_name(config),
        split_method=method,
        n_rows=n_rows,
        n_train=indices.n_train,
        n_validation=indices.n_validation,
        n_test=indices.n_test,
        train_fraction=_fraction(indices.n_train, n_rows),
        validation_fraction=_fraction(indices.n_validation, n_rows),
        test_fraction=_fraction(indices.n_test, n_rows),
        split_column=split_column if method == _OFFICIAL_COLUMN_SPLIT_METHOD else None,
        official_train_value=(
            config.split.train_value if method == _OFFICIAL_COLUMN_SPLIT_METHOD else None
        ),
        official_test_value=(
            config.split.test_value if method == _OFFICIAL_COLUMN_SPLIT_METHOD else None
        ),
        official_train_rows=(
            len(official_train) if method == _OFFICIAL_COLUMN_SPLIT_METHOD else None
        ),
        official_test_rows=(
            len(official_test) if method == _OFFICIAL_COLUMN_SPLIT_METHOD else None
        ),
        official_train_start_index=official_train_start,
        official_train_end_index=official_train_end,
        official_test_start_index=official_test_start,
        official_test_end_index=official_test_end,
        validation_fraction_within_train=(
            config.split.validation_fraction_within_train
            if method == _OFFICIAL_COLUMN_SPLIT_METHOD
            else None
        ),
        train_start_index=train_start,
        train_end_index=train_end,
        validation_start_index=val_start,
        validation_end_index=val_end,
        test_start_index=test_start,
        test_end_index=test_end,
    )


# ---------------------------------------------------------------------------
# Preparation result container
# ---------------------------------------------------------------------------


class FI2010PreparationResult(BaseModel):
    """Aggregated artefacts written by :func:`prepare_fi2010_benchmark`."""

    model_config = _MODEL_CONFIG

    summary: FI2010PreparationSummary
    data_manifest: DataManifest
    label_summary: LabelSummary
    split_summary: SplitSummary
    validation_summary: ValidationSummary
    output_dir: Path
    written_files: list[Path] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Preparation logic
# ---------------------------------------------------------------------------


def _resolve_data_path(data_path: str | Path, *, config_path: Path | None) -> Path:
    candidate = Path(data_path)
    if str(candidate) == _LOCAL_PATH_PLACEHOLDER:
        raise FileNotFoundError(
            "local_data_path is the safe placeholder; supply a local FI-2010 "
            "file path with --data-path",
        )
    if not candidate.is_absolute() and config_path is not None:
        resolved = (config_path.parent / candidate).resolve()
        if resolved.is_file():
            return resolved
    return candidate


def _build_dataset(
    config: FI2010BenchmarkConfig,
    data_path: Path,
) -> FI2010Dataset:
    label_columns = list(config.label_columns)
    fi2010_config = FI2010Config(
        path=data_path,
        timestamp_column=config.timestamp_column,
        split_column=effective_split_column(config),
        label_columns=label_columns,
        price_level_count=config.price_level_count,
    )
    return load_fi2010(fi2010_config)


def _label_summary_from_dataset(
    dataset: FI2010Dataset,
    *,
    label_name: str,
    horizon: int,
) -> LabelSummary:
    if label_name not in dataset.frame.columns:
        raise ValueError(
            f"label column {label_name!r} is missing from the FI-2010 dataset",
        )
    series = dataset.frame[label_name]
    finite_series = series.dropna()
    counts = finite_series.value_counts(dropna=True).sort_index()
    total = int(counts.sum())
    distinct = [str(value) for value in counts.index.tolist()]
    class_counts = {str(key): int(value) for key, value in counts.items()}
    if total > 0:
        class_proportions = {
            key: float(value) / float(total) for key, value in class_counts.items()
        }
    else:
        class_proportions = dict.fromkeys(class_counts, 0.0)
    return LabelSummary(
        label_name=label_name,
        horizon=horizon,
        row_count=total,
        distinct_classes=distinct,
        class_counts=class_counts,
        class_proportions=class_proportions,
        label_source="fi2010_existing_labels",
    )


def _split_summary_from_indices(
    *,
    split_name: str,
    n_rows: int,
    indices: SplitIndices,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> SplitSummary:
    def _bounds(values: Sequence[int]) -> tuple[int | None, int | None]:
        if not values:
            return None, None
        return int(values[0]), int(values[-1])

    train_start, train_end = _bounds(indices.train)
    val_start, val_end = _bounds(indices.validation)
    test_start, test_end = _bounds(indices.test)
    return SplitSummary(
        split_name=split_name,
        n_rows=n_rows,
        n_train=indices.n_train,
        n_validation=indices.n_validation,
        n_test=indices.n_test,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        train_start_index=train_start,
        train_end_index=train_end,
        validation_start_index=val_start,
        validation_end_index=val_end,
        test_start_index=test_start,
        test_end_index=test_end,
    )


def _validation_summary_from_dataset(
    dataset: FI2010Dataset,
    label_frame: pd.DataFrame,
    *,
    split_method: str,
) -> ValidationSummary:
    from chronoslob.labels.pipeline import validate_label_frame

    fi2010_result = validate_fi2010_dataset(dataset)
    label_result = validate_label_frame(label_frame, allow_nan=False)
    notes: list[str] = []
    if dataset.config.timestamp_column is None:
        if split_method == _OFFICIAL_COLUMN_SPLIT_METHOD:
            notes.append(_OFFICIAL_ROW_ORDER_NOTE)
        else:
            notes.append(
                "no timestamp column configured; temporal split is row-order only",
            )
    if dataset.config.split_column is None:
        notes.append(
            "no native split column; temporal split derived from row order",
        )
    return ValidationSummary(
        fi2010_validation_ok=fi2010_result.ok,
        fi2010_error_count=fi2010_result.error_count,
        fi2010_warning_count=fi2010_result.warning_count,
        label_validation_ok=label_result.ok,
        label_error_count=label_result.error_count,
        label_warning_count=label_result.warning_count,
        notes=notes,
    )


def _label_frame_for_validation(
    dataset: FI2010Dataset,
    *,
    label_name: str,
) -> pd.DataFrame:
    frame = dataset.frame.loc[:, [label_name]].copy()
    timestamps = pd.to_datetime(
        dataset.frame[dataset.config.timestamp_column],
        utc=True,
        errors="raise",
    ) if dataset.config.timestamp_column is not None else None
    if timestamps is not None:
        frame["timestamp"] = timestamps.reset_index(drop=True)
    else:
        base = pd.Timestamp("2000-01-01T00:00:00Z")
        frame["timestamp"] = pd.Series(
            [base + pd.Timedelta(seconds=idx) for idx in range(len(frame))]
        )
    frame["symbol"] = dataset.config.symbol
    frame["label_source"] = "fi2010_existing_labels"
    return frame


def _write_json(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json_dumps(model), encoding="utf-8")


def _stable_yaml_dumps(payload: Mapping[str, Any]) -> str:
    return yaml.safe_dump(
        dict(payload),
        sort_keys=True,
        allow_unicode=True,
        default_flow_style=False,
    )


def prepare_fi2010_benchmark(
    config: FI2010BenchmarkConfig,
    *,
    data_path: Path,
    output_dir: Path,
    config_source_path: Path | None = None,
    created_at: datetime | None = None,
) -> FI2010PreparationResult:
    """Run FI-2010 benchmark preparation and write preparation artefacts.

    The function never downloads or modifies the source data file. It
    reads the supplied local file, validates it through the existing
    FI-2010 loader and validator, constructs a label distribution and a
    deterministic split summary, builds a data manifest and
    writes a small set of preparation artefacts to ``output_dir``.

    Preparation is not a finished experiment: ``results.json``,
    ``predictions.csv`` and model artefacts are explicitly out of scope
    for this phase and are not written.
    """
    resolved_data_path = _resolve_data_path(
        data_path,
        config_path=config_source_path,
    )
    if not resolved_data_path.exists():
        raise FileNotFoundError(
            f"local FI-2010 data path does not exist: {resolved_data_path}",
        )
    if not resolved_data_path.is_file():
        raise FileNotFoundError(
            f"local FI-2010 data path is not a regular file: {resolved_data_path}",
        )

    dataset = _build_dataset(config, resolved_data_path)
    label_summary = _label_summary_from_dataset(
        dataset,
        label_name=config.label_name,
        horizon=config.horizon,
    )

    n_rows = dataset.n_rows
    indices = build_benchmark_split(config, dataset.frame)
    split_summary = effective_split_summary(
        config=config,
        frame=dataset.frame,
        indices=indices,
    )

    label_frame = _label_frame_for_validation(dataset, label_name=config.label_name)
    validation_summary = _validation_summary_from_dataset(
        dataset,
        label_frame,
        split_method=split_summary.split_method,
    )

    manifest_timestamp = created_at or datetime.now(UTC)
    manifest = build_local_file_manifest(
        resolved_data_path,
        dataset_name=config.dataset_name,
        dataset_variant=f"{config.task_name}_h{config.horizon}",
        label_name=config.label_name,
        horizon=config.horizon,
        split_name=effective_split_name(config),
        row_count=n_rows,
        feature_count=dataset.n_features,
        notes=config.notes,
        created_at=manifest_timestamp,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_path = output_path / "data_manifest.json"
    label_summary_path = output_path / "label_summary.json"
    split_summary_path = output_path / "split_summary.json"
    validation_summary_path = output_path / "validation_summary.json"
    config_snapshot_path = output_path / "config.yaml"
    summary_path = output_path / "preparation_summary.json"

    _write_json(manifest_path, manifest)
    _write_json(label_summary_path, label_summary)
    _write_json(split_summary_path, split_summary)
    _write_json(validation_summary_path, validation_summary)

    if config_source_path is not None and Path(config_source_path).is_file():
        shutil.copyfile(config_source_path, config_snapshot_path)
    else:
        config_snapshot_path.write_text(
            _stable_yaml_dumps(config.model_dump(mode="json")),
            encoding="utf-8",
        )

    warnings: list[str] = []
    if not validation_summary.fi2010_validation_ok:
        warnings.append(
            "FI-2010 validation reported errors; inspect validation_summary.json",
        )
    if not validation_summary.label_validation_ok:
        warnings.append(
            "label validation reported errors; inspect validation_summary.json",
        )
    if config.timestamp_column is None:
        if split_summary.split_method == _OFFICIAL_COLUMN_SPLIT_METHOD:
            warnings.append(_OFFICIAL_ROW_ORDER_NOTE)
        else:
            warnings.append(
                "no timestamp column configured; temporal split uses row order only",
            )

    artefacts = {
        "config": "config.yaml",
        "data_manifest": "data_manifest.json",
        "label_summary": "label_summary.json",
        "split_summary": "split_summary.json",
        "validation_summary": "validation_summary.json",
    }

    summary = FI2010PreparationSummary(
        experiment_name=config.experiment_name,
        dataset_name=config.dataset_name,
        task_name=config.task_name,
        horizon=config.horizon,
        split_name=effective_split_name(config),
        seed=config.seed,
        label_name=config.label_name,
        data_path=str(resolved_data_path),
        output_dir=str(output_path),
        preparation_version=_PREPARATION_VERSION,
        created_at=manifest_timestamp,
        artefacts=artefacts,
        warnings=warnings,
    )
    _write_json(summary_path, summary)

    written_files = [
        manifest_path,
        label_summary_path,
        split_summary_path,
        validation_summary_path,
        config_snapshot_path,
        summary_path,
    ]
    return FI2010PreparationResult(
        summary=summary,
        data_manifest=manifest,
        label_summary=label_summary,
        split_summary=split_summary,
        validation_summary=validation_summary,
        output_dir=output_path,
        written_files=written_files,
    )
