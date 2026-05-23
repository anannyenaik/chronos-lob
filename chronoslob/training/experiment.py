"""Experiment registry skeleton for reproducible research runs."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from chronoslob.training.artifacts import (
    create_run_directory,
    safe_run_name,
    write_json,
)
from chronoslob.utils.paths import project_root

__all__ = [
    "ExperimentMetadata",
    "create_experiment_metadata",
    "get_git_commit",
    "initialise_experiment_run",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)


class ExperimentMetadata(BaseModel):
    """Metadata captured when a reproducible experiment run is initialised."""

    model_config = _MODEL_CONFIG

    run_id: str
    run_name: str
    created_at: datetime
    project: str = "ChronosLOB"
    phase: str
    seed: int
    git_commit: str | None = None
    config_path: str | None = None
    input_paths: list[str] = Field(default_factory=list)
    output_path: str | None = None
    notes: str | None = None

    @field_validator("run_id", "run_name", "phase")
    @classmethod
    def _validate_non_empty_string(cls, value: str, info: ValidationInfo) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return value

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("seed")
    @classmethod
    def _validate_seed(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("seed must be an integer")
        if value < 0:
            raise ValueError("seed must be non-negative")
        return value


def get_git_commit() -> str | None:
    """Return the current git commit hash, or ``None`` if unavailable."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root(),
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None


def _stringify_optional_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path))


def _stringify_paths(paths: Sequence[str | Path]) -> list[str]:
    return [str(Path(path)) for path in paths]


def create_experiment_metadata(
    *,
    run_name: str,
    phase: str,
    seed: int,
    config_path: str | Path | None = None,
    input_paths: Sequence[str | Path] = (),
    output_path: str | Path | None = None,
    notes: str | None = None,
) -> ExperimentMetadata:
    """Create metadata for an experiment run without writing files."""
    created_at = datetime.now(UTC)
    safe_name = safe_run_name(run_name)
    timestamp = created_at.strftime("%Y%m%dt%H%M%S%fz")
    return ExperimentMetadata(
        run_id=f"{safe_name}-{timestamp}",
        run_name=run_name,
        created_at=created_at,
        phase=phase,
        seed=seed,
        git_commit=get_git_commit(),
        config_path=_stringify_optional_path(config_path),
        input_paths=_stringify_paths(input_paths),
        output_path=_stringify_optional_path(output_path),
        notes=notes,
    )


def _copy_config_if_present(config_path: str | Path, run_dir: Path) -> str:
    source = Path(config_path).expanduser()
    if not source.is_file():
        return str(Path(config_path))
    destination = run_dir / "configs" / source.name
    shutil.copy2(source, destination)
    return str(destination)


def initialise_experiment_run(
    *,
    root: str | Path,
    run_name: str,
    phase: str,
    seed: int,
    config_path: str | Path | None = None,
    input_paths: Sequence[str | Path] = (),
    notes: str | None = None,
) -> tuple[ExperimentMetadata, Path]:
    """Create a metadata-only experiment run directory."""
    metadata = create_experiment_metadata(
        run_name=run_name,
        phase=phase,
        seed=seed,
        config_path=config_path,
        input_paths=input_paths,
        notes=notes,
    )
    run_dir = create_run_directory(root, metadata.run_id)

    recorded_config_path = metadata.config_path
    if config_path is not None:
        recorded_config_path = _copy_config_if_present(config_path, run_dir)

    metadata = metadata.model_copy(
        update={
            "config_path": recorded_config_path,
            "output_path": str(run_dir),
        }
    )
    write_json(run_dir / "metadata.json", metadata.model_dump(mode="json"))
    return metadata, run_dir
