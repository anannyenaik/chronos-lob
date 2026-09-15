"""Typed contracts for persisted experiment artefacts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from chronoslob.data.schemas import ensure_utc_datetime

__all__ = [
    "ArtifactKind",
    "DataManifest",
    "EvidenceStreams",
    "ExperimentArtifactExpectation",
    "ExperimentArtifactStatus",
    "ExperimentConfigSummary",
    "ExperimentManifest",
    "ExperimentResults",
    "ExperimentValidationReport",
    "ModelResult",
    "SourceKind",
    "validate_local_path_text",
    "validate_relative_artifact_path",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)
_NETWORK_URI_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_SECRET_MARKERS = (
    "api_key",
    "apikey",
    "access_key",
    "secret",
    "password",
    "credential",
)


class SourceKind(StrEnum):
    """Supported local data provenance categories."""

    LOCAL_FILE = "local_file"
    LOCAL_DIRECTORY = "local_directory"
    SYNTHETIC_FIXTURE = "synthetic_fixture"


class ArtifactKind(StrEnum):
    """Experiment artefact categories used by the directory contract."""

    CONFIG = "config"
    DATA_MANIFEST = "data_manifest"
    RESULTS = "results"
    PREDICTIONS = "predictions"
    CALIBRATION = "calibration"
    EXECUTION = "execution"
    MODEL_CARD = "model_card"
    PLOT = "plot"


def _validate_non_empty_string(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validate_string_list(value: list[str], *, field_name: str) -> list[str]:
    if not value:
        raise ValueError(f"{field_name} must contain at least one entry")
    return [
        _validate_non_empty_string(item, field_name=field_name) for item in value
    ]


def _contains_parent_reference(value: str) -> bool:
    normalised = value.replace("\\", "/")
    return any(part == ".." for part in PurePosixPath(normalised).parts)


def validate_local_path_text(value: str, *, field_name: str = "path") -> str:
    """Validate a local path-like string without requiring it to exist."""
    cleaned = _validate_non_empty_string(value, field_name=field_name)
    if _NETWORK_URI_PATTERN.match(cleaned):
        raise ValueError(f"{field_name} must be a local path, not a URI")
    if cleaned.startswith(("\\\\", "//")):
        raise ValueError(f"{field_name} must not be a network share path")
    lowered = cleaned.lower()
    if any(marker in lowered for marker in _SECRET_MARKERS):
        raise ValueError(f"{field_name} must not contain credential-like markers")
    return cleaned


def validate_relative_artifact_path(
    value: str,
    *,
    field_name: str = "path",
) -> str:
    """Validate a portable relative artefact path."""
    cleaned = validate_local_path_text(value, field_name=field_name)
    windows_path = PureWindowsPath(cleaned)
    posix_path = PurePosixPath(cleaned.replace("\\", "/"))
    if windows_path.is_absolute() or windows_path.drive or posix_path.is_absolute():
        raise ValueError(f"{field_name} must be relative to the experiment directory")
    if cleaned.startswith("~"):
        raise ValueError(f"{field_name} must not depend on a user home directory")
    if _contains_parent_reference(cleaned):
        raise ValueError(f"{field_name} must not contain parent-directory references")
    return cleaned.replace("\\", "/")


def _validate_optional_count(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer when provided")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _validate_positive_horizon(value: int, *, field_name: str = "horizon") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _validate_non_negative_seed(value: int, *, field_name: str = "seed") -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _validate_metric_mapping(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("metrics must be a mapping")
    if not value:
        raise ValueError("metrics must contain at least one metric")

    cleaned: dict[str, float] = {}
    for key, item in value.items():
        metric_name = _validate_non_empty_string(str(key), field_name="metric name")
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"metric {metric_name!r} must be a finite number")
        metric_value = float(item)
        if not math.isfinite(metric_value):
            raise ValueError(f"metric {metric_name!r} must be finite")
        cleaned[metric_name] = metric_value
    return cleaned


def _validate_artefact_mapping(value: object) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("artefacts must be a mapping when provided")
    cleaned: dict[str, str] = {}
    for key, item in value.items():
        artefact_name = _validate_non_empty_string(str(key), field_name="artefact name")
        if not isinstance(item, str):
            raise ValueError(f"artefact {artefact_name!r} path must be a string")
        cleaned[artefact_name] = validate_relative_artifact_path(
            item,
            field_name=f"artefact {artefact_name!r}",
        )
    return cleaned


class DataManifest(BaseModel):
    """Portable provenance summary for data used by an experiment."""

    model_config = _MODEL_CONFIG

    dataset_name: str
    dataset_version: str | None = None
    dataset_variant: str | None = None
    source_kind: SourceKind
    source_path: str
    source_sha256: str | None = None
    created_at: datetime
    row_count: int | None = None
    event_count: int | None = None
    feature_count: int | None = None
    label_name: str
    horizon: int
    split_name: str
    notes: str | None = None

    @field_validator("dataset_name", "label_name", "split_name")
    @classmethod
    def _validate_required_strings(cls, value: str) -> str:
        return _validate_non_empty_string(value, field_name="manifest field")

    @field_validator("dataset_version", "dataset_variant", "notes")
    @classmethod
    def _validate_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_non_empty_string(value, field_name="manifest field")

    @field_validator("source_path")
    @classmethod
    def _validate_source_path(cls, value: str) -> str:
        return validate_local_path_text(value, field_name="source_path")

    @field_validator("source_sha256")
    @classmethod
    def _validate_source_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = _validate_non_empty_string(value, field_name="source_sha256")
        if len(cleaned) != 64 or any(char not in "0123456789abcdef" for char in cleaned):
            raise ValueError("source_sha256 must be a lowercase 64-character hex digest")
        return cleaned

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        return ensure_utc_datetime(value)

    @field_validator("row_count", "event_count", "feature_count")
    @classmethod
    def _validate_counts(cls, value: int | None) -> int | None:
        return _validate_optional_count(value, field_name="manifest count")

    @field_validator("horizon")
    @classmethod
    def _validate_horizon(cls, value: int) -> int:
        return _validate_positive_horizon(value)

    @model_validator(mode="after")
    def _validate_version_or_variant(self) -> DataManifest:
        if self.dataset_version is None and self.dataset_variant is None:
            raise ValueError("dataset_version or dataset_variant must be provided")
        return self


ExperimentManifest = DataManifest


class ExperimentConfigSummary(BaseModel):
    """Small, machine-readable summary of the experiment configuration."""

    model_config = _MODEL_CONFIG

    experiment_name: str
    task_name: str
    horizon: int
    split_name: str
    seed: int
    model_names: list[str]
    primary_metric: str
    created_at: datetime
    code_commit: str | None = None

    @field_validator("experiment_name", "task_name", "split_name", "primary_metric")
    @classmethod
    def _validate_required_strings(cls, value: str) -> str:
        return _validate_non_empty_string(value, field_name="config summary field")

    @field_validator("code_commit")
    @classmethod
    def _validate_code_commit(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_non_empty_string(value, field_name="code_commit")

    @field_validator("horizon")
    @classmethod
    def _validate_horizon(cls, value: int) -> int:
        return _validate_positive_horizon(value)

    @field_validator("seed")
    @classmethod
    def _validate_seed(cls, value: int) -> int:
        return _validate_non_negative_seed(value)

    @field_validator("model_names")
    @classmethod
    def _validate_model_names(cls, value: list[str]) -> list[str]:
        return _validate_string_list(value, field_name="model_names")

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        return ensure_utc_datetime(value)


class ModelResult(BaseModel):
    """Metric record for one model on one split and horizon."""

    model_config = _MODEL_CONFIG

    model_name: str
    split: str
    horizon: int
    metrics: dict[str, float]
    artefacts: dict[str, str] | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("model_name", "split")
    @classmethod
    def _validate_required_strings(cls, value: str) -> str:
        return _validate_non_empty_string(value, field_name="model result field")

    @field_validator("horizon")
    @classmethod
    def _validate_horizon(cls, value: int) -> int:
        return _validate_positive_horizon(value)

    @field_validator("metrics", mode="before")
    @classmethod
    def _validate_metrics(cls, value: object) -> dict[str, float]:
        return _validate_metric_mapping(value)

    @field_validator("artefacts", mode="before")
    @classmethod
    def _validate_artefacts(cls, value: object) -> dict[str, str] | None:
        return _validate_artefact_mapping(value)

    @field_validator("warnings")
    @classmethod
    def _validate_warnings(cls, value: list[str]) -> list[str]:
        return [
            _validate_non_empty_string(item, field_name="warning") for item in value
        ]


class EvidenceStreams(BaseModel):
    """Metric groups expected for benchmark evidence review."""

    model_config = _MODEL_CONFIG

    predictive: list[str]
    calibration: list[str]
    execution: list[str]
    robustness: list[str] = Field(default_factory=list)
    systems: list[str] = Field(default_factory=list)

    @field_validator("predictive", "calibration", "execution")
    @classmethod
    def _validate_required_streams(cls, value: list[str]) -> list[str]:
        return _validate_string_list(value, field_name="evidence stream")

    @field_validator("robustness", "systems")
    @classmethod
    def _validate_optional_streams(cls, value: list[str]) -> list[str]:
        return [
            _validate_non_empty_string(item, field_name="evidence stream")
            for item in value
        ]


class ExperimentResults(BaseModel):
    """Top-level machine-readable results contract for an experiment."""

    model_config = _MODEL_CONFIG

    experiment_name: str
    task_name: str
    created_at: datetime
    config_summary: ExperimentConfigSummary
    model_results: list[ModelResult] = Field(min_length=1)
    evidence_streams: EvidenceStreams

    @field_validator("experiment_name", "task_name")
    @classmethod
    def _validate_required_strings(cls, value: str) -> str:
        return _validate_non_empty_string(value, field_name="experiment results field")

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        return ensure_utc_datetime(value)


class ExperimentArtifactExpectation(BaseModel):
    """One required or optional artefact expectation."""

    model_config = _MODEL_CONFIG

    path: str
    required: bool
    kind: ArtifactKind
    alternatives: list[str] = Field(default_factory=list)
    description: str | None = None

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return validate_relative_artifact_path(value)

    @field_validator("alternatives")
    @classmethod
    def _validate_alternatives(cls, value: list[str]) -> list[str]:
        return [validate_relative_artifact_path(path) for path in value]

    @field_validator("description")
    @classmethod
    def _validate_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_non_empty_string(value, field_name="description")

    @model_validator(mode="after")
    def _validate_required_alternatives(self) -> ExperimentArtifactExpectation:
        if self.required and self.alternatives:
            raise ValueError("required artefacts must not use alternatives")
        if self.path in self.alternatives:
            raise ValueError("primary path must not repeat in alternatives")
        return self

    @property
    def candidate_paths(self) -> tuple[str, ...]:
        """Return all path candidates that satisfy this expectation."""
        return (self.path, *self.alternatives)


class ExperimentArtifactStatus(BaseModel):
    """Presence and validation status for one artefact expectation."""

    model_config = _MODEL_CONFIG

    path: str
    exists: bool
    required: bool
    kind: ArtifactKind
    message: str

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return validate_relative_artifact_path(value)

    @field_validator("message")
    @classmethod
    def _validate_message(cls, value: str) -> str:
        return _validate_non_empty_string(value, field_name="message")


class ExperimentValidationReport(BaseModel):
    """Structured validation report for an experiment directory."""

    model_config = _MODEL_CONFIG

    experiment_dir: str
    is_valid: bool
    missing_required: list[str] = Field(default_factory=list)
    present_optional: list[str] = Field(default_factory=list)
    artefact_statuses: list[ExperimentArtifactStatus] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("experiment_dir")
    @classmethod
    def _validate_experiment_dir(cls, value: str) -> str:
        return validate_local_path_text(value, field_name="experiment_dir")

    @field_validator("missing_required", "present_optional", "warnings")
    @classmethod
    def _validate_string_items(cls, value: list[str]) -> list[str]:
        return [_validate_non_empty_string(item, field_name="report item") for item in value]
