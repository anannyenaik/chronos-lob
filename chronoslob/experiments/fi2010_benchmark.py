"""FI-2010 local benchmark preparation layer.

This module turns a user-supplied local FI-2010-style file into a
documented, validated benchmark input. It produces a data manifest,
label-distribution summary, temporal split summary and a leakage-aware
validation summary in a preparation output directory.

The layer is deliberately read-only with respect to the source data and
never downloads, uploads or commits market data. It is also not a model
runner: it does not produce ``results.json`` or any prediction artefact.
Those belong to the later paper experiment runner phase.
"""

from __future__ import annotations

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
    "FI2010BenchmarkConfig",
    "FI2010PreparationResult",
    "FI2010PreparationSummary",
    "LabelSummary",
    "SplitSummary",
    "ValidationSummary",
    "load_benchmark_config",
    "prepare_fi2010_benchmark",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)
_LOCAL_PATH_PLACEHOLDER = "<path-to-local-fi2010-file>"
_DEFAULT_LABEL_COLUMNS = ("label_10", "label_50", "label_100")
_PREPARATION_VERSION = "phase-b/fi2010-benchmark-preparation/v1"


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
    n_rows: int
    n_train: int
    n_validation: int
    n_test: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
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
    notes: str | None = None
    classical_models_supported: tuple[str, ...] = ()
    models_planned: tuple[str, ...] = ()
    metrics_planned: tuple[str, ...] = ()
    execution_assumptions_planned: dict[str, Any] = Field(default_factory=dict)

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
        split_column=config.split_column,
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
) -> ValidationSummary:
    from chronoslob.labels.pipeline import validate_label_frame

    fi2010_result = validate_fi2010_dataset(dataset)
    label_result = validate_label_frame(label_frame, allow_nan=False)
    notes: list[str] = []
    if dataset.config.timestamp_column is None:
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
    deterministic temporal split summary, builds a data manifest and
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
    split_config = TemporalSplitConfig(
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        min_train_size=1,
        min_validation_size=1,
        min_test_size=0,
    )
    indices = temporal_train_validation_test_split(n_rows, split_config)
    split_summary = _split_summary_from_indices(
        split_name=config.split_name,
        n_rows=n_rows,
        indices=indices,
        train_fraction=split_config.train_fraction,
        validation_fraction=split_config.validation_fraction,
        test_fraction=split_config.test_fraction,
    )

    label_frame = _label_frame_for_validation(dataset, label_name=config.label_name)
    validation_summary = _validation_summary_from_dataset(dataset, label_frame)

    manifest_timestamp = created_at or datetime.now(UTC)
    manifest = build_local_file_manifest(
        resolved_data_path,
        dataset_name=config.dataset_name,
        dataset_variant=f"{config.task_name}_h{config.horizon}",
        label_name=config.label_name,
        horizon=config.horizon,
        split_name=config.split_name,
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
        split_name=config.split_name,
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
