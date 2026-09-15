"""Read, write and inspect experiment artefact contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from chronoslob.experiments.manifests import stable_json_dumps
from chronoslob.experiments.schemas import (
    DataManifest,
    ExperimentArtifactExpectation,
    ExperimentResults,
    ExperimentValidationReport,
)
from chronoslob.experiments.validation import (
    expected_experiment_artifacts as _expected_experiment_artifacts,
)
from chronoslob.experiments.validation import (
    validate_experiment_directory as _validate_experiment_directory,
)

__all__ = [
    "expected_experiment_artifacts",
    "load_data_manifest",
    "load_results",
    "read_json_model",
    "validate_experiment_directory",
    "write_json_model",
]

ModelT = TypeVar("ModelT", bound=BaseModel)


def _resolve_artefact_path(path: Path, default_filename: str) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        return candidate / default_filename
    return candidate


def expected_experiment_artifacts(
    include_plots: bool = True,
) -> list[ExperimentArtifactExpectation]:
    """Return the standard expected artefacts for an experiment directory."""
    return _expected_experiment_artifacts(include_plots=include_plots)


def read_json_model(path: Path, model_type: type[ModelT]) -> ModelT:
    """Read a JSON object from ``path`` and validate it as ``model_type``."""
    file_path = Path(path)
    try:
        payload: Any = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"failed to parse {file_path} as JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{file_path} must contain a JSON object")
    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid {model_type.__name__} in {file_path}: {exc}") from exc


def write_json_model(path: Path, model: BaseModel) -> None:
    """Write a Pydantic model as stable JSON."""
    if not isinstance(model, BaseModel):
        raise TypeError("model must be a Pydantic BaseModel")
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(stable_json_dumps(model), encoding="utf-8")


def load_results(path: Path) -> ExperimentResults:
    """Load ``results.json`` from an experiment directory or direct file path."""
    return read_json_model(
        _resolve_artefact_path(Path(path), "results.json"),
        ExperimentResults,
    )


def load_data_manifest(path: Path) -> DataManifest:
    """Load ``data_manifest.json`` from an experiment directory or direct file path."""
    return read_json_model(
        _resolve_artefact_path(Path(path), "data_manifest.json"),
        DataManifest,
    )


def validate_experiment_directory(
    path: Path,
    *,
    include_plots: bool = True,
) -> ExperimentValidationReport:
    """Validate an experiment directory against the persisted artefact contract."""
    return _validate_experiment_directory(path, include_plots=include_plots)
