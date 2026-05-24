"""Validation helpers for experiment artefact directories."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from chronoslob.experiments.schemas import (
    ArtifactKind,
    DataManifest,
    ExperimentArtifactExpectation,
    ExperimentArtifactStatus,
    ExperimentResults,
    ExperimentValidationReport,
)

__all__ = [
    "expected_experiment_artifacts",
    "validate_experiment_directory",
]


def expected_experiment_artifacts(
    *,
    include_plots: bool = True,
) -> list[ExperimentArtifactExpectation]:
    """Return the standard experiment artefact expectations."""
    expectations = [
        ExperimentArtifactExpectation(
            path="config.yaml",
            required=True,
            kind=ArtifactKind.CONFIG,
            description="Complete experiment configuration.",
        ),
        ExperimentArtifactExpectation(
            path="data_manifest.json",
            required=True,
            kind=ArtifactKind.DATA_MANIFEST,
            description="Local data provenance and split metadata.",
        ),
        ExperimentArtifactExpectation(
            path="results.json",
            required=True,
            kind=ArtifactKind.RESULTS,
            description="Machine-readable metrics and evidence streams.",
        ),
        ExperimentArtifactExpectation(
            path="model_card.md",
            required=True,
            kind=ArtifactKind.MODEL_CARD,
            description="Methodological summary and claim boundaries.",
        ),
        ExperimentArtifactExpectation(
            path="predictions.csv",
            alternatives=["predictions.parquet"],
            required=False,
            kind=ArtifactKind.PREDICTIONS,
            description="Row-level predictions in CSV or Parquet form.",
        ),
        ExperimentArtifactExpectation(
            path="calibration_bins.csv",
            required=False,
            kind=ArtifactKind.CALIBRATION,
            description="Reliability-bin calibration records.",
        ),
        ExperimentArtifactExpectation(
            path="execution_sensitivity.csv",
            required=False,
            kind=ArtifactKind.EXECUTION,
            description="Execution-assumption sensitivity records.",
        ),
    ]
    if include_plots:
        expectations.extend(
            [
                ExperimentArtifactExpectation(
                    path="plots/reliability_curve.png",
                    required=False,
                    kind=ArtifactKind.PLOT,
                    description="Reliability-curve plot derived from stored data.",
                ),
                ExperimentArtifactExpectation(
                    path="plots/cost_sensitivity.png",
                    required=False,
                    kind=ArtifactKind.PLOT,
                    description="Cost-sensitivity plot derived from stored data.",
                ),
                ExperimentArtifactExpectation(
                    path="plots/confusion_matrix.png",
                    required=False,
                    kind=ArtifactKind.PLOT,
                    description="Confusion-matrix plot derived from stored data.",
                ),
                ExperimentArtifactExpectation(
                    path="plots/regime_breakdown.png",
                    required=False,
                    kind=ArtifactKind.PLOT,
                    description="Regime-breakdown plot derived from stored data.",
                ),
            ]
        )
    return expectations


def _candidate_file(experiment_dir: Path, relative_path: str) -> Path:
    return experiment_dir / Path(relative_path)


def _first_present_candidate(
    experiment_dir: Path,
    expectation: ExperimentArtifactExpectation,
) -> str | None:
    for relative_path in expectation.candidate_paths:
        if _candidate_file(experiment_dir, relative_path).is_file():
            return relative_path
    return None


def _format_candidates(candidates: Sequence[str]) -> str:
    return " or ".join(candidates)


def _read_json_model(path: Path, model_type: type[BaseModel]) -> BaseModel:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"failed to parse {path.name} as JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"{path.name} failed schema validation: {exc}") from exc


def _validate_schema_file(
    experiment_dir: Path,
    relative_path: str,
    model_type: type[BaseModel],
) -> str | None:
    try:
        _read_json_model(_candidate_file(experiment_dir, relative_path), model_type)
    except (OSError, ValueError) as exc:
        return str(exc)
    return None


def _status_for_expectation(
    experiment_dir: Path,
    expectation: ExperimentArtifactExpectation,
) -> ExperimentArtifactStatus:
    present_path = _first_present_candidate(experiment_dir, expectation)
    if present_path is not None:
        return ExperimentArtifactStatus(
            path=present_path,
            exists=True,
            required=expectation.required,
            kind=expectation.kind,
            message="present",
        )

    candidates = _format_candidates(expectation.candidate_paths)
    label = "required" if expectation.required else "optional"
    return ExperimentArtifactStatus(
        path=expectation.path,
        exists=False,
        required=expectation.required,
        kind=expectation.kind,
        message=f"{label} artefact missing: {candidates}",
    )


def validate_experiment_directory(
    path: Path,
    *,
    include_plots: bool = True,
) -> ExperimentValidationReport:
    """Validate an experiment directory against the artefact contract."""
    experiment_dir = Path(path)
    expectations = expected_experiment_artifacts(include_plots=include_plots)
    statuses: list[ExperimentArtifactStatus] = []
    missing_required: list[str] = []
    present_optional: list[str] = []
    warnings: list[str] = []
    invalid_schema_paths: list[str] = []

    if not experiment_dir.is_dir():
        warnings.append("experiment directory is missing or is not a directory")

    for expectation in expectations:
        status = _status_for_expectation(experiment_dir, expectation)
        if status.exists and not status.required:
            present_optional.append(status.path)
        if not status.exists:
            if expectation.required:
                missing_required.append(expectation.path)
            else:
                warnings.append(status.message)
        statuses.append(status)

    schema_checks: tuple[tuple[str, type[BaseModel]], ...] = (
        ("data_manifest.json", DataManifest),
        ("results.json", ExperimentResults),
    )
    updated_statuses: list[ExperimentArtifactStatus] = []
    for status in statuses:
        schema_error: str | None = None
        for relative_path, model_type in schema_checks:
            if status.exists and status.path == relative_path:
                schema_error = _validate_schema_file(
                    experiment_dir,
                    relative_path,
                    model_type,
                )
                break
        if schema_error is None:
            updated_statuses.append(status)
            continue

        invalid_schema_paths.append(status.path)
        updated_statuses.append(
            status.model_copy(update={"message": f"invalid schema: {schema_error}"})
        )

    is_valid = not missing_required and not invalid_schema_paths
    return ExperimentValidationReport(
        experiment_dir=str(experiment_dir),
        is_valid=is_valid,
        missing_required=missing_required,
        present_optional=present_optional,
        artefact_statuses=updated_statuses,
        warnings=warnings,
    )
