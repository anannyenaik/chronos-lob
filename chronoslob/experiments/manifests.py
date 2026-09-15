"""Manifest helpers for local experiment inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from chronoslob.experiments.schemas import DataManifest, SourceKind

__all__ = [
    "build_directory_manifest",
    "build_local_file_manifest",
    "sha256_file",
    "stable_json_dumps",
]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of a local file, reading in chunks."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_dumps(payload: BaseModel | Mapping[str, Any]) -> str:
    """Serialise a model or mapping as stable, finite JSON."""
    if isinstance(payload, BaseModel):
        serialisable: Mapping[str, Any] = payload.model_dump(mode="json")
    else:
        serialisable = payload
    return (
        json.dumps(
            serialisable,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def build_local_file_manifest(
    path: Path,
    *,
    dataset_name: str,
    label_name: str,
    horizon: int,
    split_name: str,
    dataset_version: str | None = None,
    dataset_variant: str | None = None,
    row_count: int | None = None,
    event_count: int | None = None,
    feature_count: int | None = None,
    notes: str | None = None,
    created_at: datetime | None = None,
) -> DataManifest:
    """Build a manifest for a local data file without reading beyond hashing."""
    file_path = Path(path)
    if file_path.exists() and not file_path.is_file():
        raise ValueError(f"path is not a file: {file_path}")
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    return DataManifest(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        dataset_variant=dataset_variant,
        source_kind=SourceKind.LOCAL_FILE,
        source_path=str(file_path),
        source_sha256=sha256_file(file_path),
        created_at=created_at or datetime.now(UTC),
        row_count=row_count,
        event_count=event_count,
        feature_count=feature_count,
        label_name=label_name,
        horizon=horizon,
        split_name=split_name,
        notes=notes,
    )


def build_directory_manifest(
    path: Path,
    *,
    dataset_name: str,
    label_name: str,
    horizon: int,
    split_name: str,
    dataset_version: str | None = None,
    dataset_variant: str | None = None,
    row_count: int | None = None,
    event_count: int | None = None,
    feature_count: int | None = None,
    notes: str | None = None,
    created_at: datetime | None = None,
) -> DataManifest:
    """Build a manifest for a local directory without recursively hashing it."""
    directory_path = Path(path)
    if not directory_path.is_dir():
        raise NotADirectoryError(directory_path)

    return DataManifest(
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        dataset_variant=dataset_variant,
        source_kind=SourceKind.LOCAL_DIRECTORY,
        source_path=str(directory_path),
        source_sha256=None,
        created_at=created_at or datetime.now(UTC),
        row_count=row_count,
        event_count=event_count,
        feature_count=feature_count,
        label_name=label_name,
        horizon=horizon,
        split_name=split_name,
        notes=notes,
    )
