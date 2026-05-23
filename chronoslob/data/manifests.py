"""Reproducibility manifests for canonical event logs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from chronoslob.data.event_store import DEFAULT_EVENT_LOG_SCHEMA_VERSION, read_event_log_jsonl
from chronoslob.data.schemas import (
    BookEvent,
    MetadataDict,
    OrderBookSnapshot,
    ensure_utc_datetime,
    validate_metadata,
)


class EventLogManifest(BaseModel):
    """Deterministic metadata summary for a local event-log file."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    path: str
    schema_version: str = DEFAULT_EVENT_LOG_SCHEMA_VERSION
    n_records: int
    n_book_events: int
    n_snapshots: int
    symbols: list[str]
    start_timestamp: datetime | None
    end_timestamp: datetime | None
    min_sequence_id: int | None
    max_sequence_id: int | None
    sha256: str
    created_at: datetime
    metadata: MetadataDict = Field(default_factory=dict)

    @field_validator("path", "schema_version", "sha256")
    @classmethod
    def _validate_non_empty_string(cls, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("manifest string fields must be non-empty")
        return value

    @field_validator("n_records", "n_book_events", "n_snapshots")
    @classmethod
    def _validate_non_negative_count(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("manifest counts must be integers")
        if value < 0:
            raise ValueError("manifest counts must be non-negative")
        return value

    @field_validator("min_sequence_id", "max_sequence_id")
    @classmethod
    def _validate_optional_sequence_id(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("sequence ids must be integers when present")
        return value

    @field_validator("start_timestamp", "end_timestamp")
    @classmethod
    def _validate_optional_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return ensure_utc_datetime(value)

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        return ensure_utc_datetime(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value: object) -> MetadataDict:
        if value is None:
            return {}
        try:
            return validate_metadata(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError(str(exc)) from exc


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 content hash for a local file."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_event_log_manifest(
    path: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> EventLogManifest:
    """Create an in-memory manifest for a canonical event-log JSONL file."""
    file_path = Path(path)
    records = read_event_log_jsonl(file_path)
    timestamps = [record.timestamp for record in records]
    sequence_ids = [
        record.sequence_id for record in records if record.sequence_id is not None
    ]
    symbols = sorted({record.symbol for record in records})
    n_book_events = sum(isinstance(record, BookEvent) for record in records)
    n_snapshots = sum(isinstance(record, OrderBookSnapshot) for record in records)
    try:
        manifest_metadata = {} if metadata is None else validate_metadata(metadata)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc

    return EventLogManifest(
        path=str(file_path),
        n_records=len(records),
        n_book_events=n_book_events,
        n_snapshots=n_snapshots,
        symbols=symbols,
        start_timestamp=min(timestamps) if timestamps else None,
        end_timestamp=max(timestamps) if timestamps else None,
        min_sequence_id=min(sequence_ids) if sequence_ids else None,
        max_sequence_id=max(sequence_ids) if sequence_ids else None,
        sha256=sha256_file(file_path),
        created_at=datetime.now(UTC),
        metadata=manifest_metadata,
    )


def write_manifest(
    path: str | Path,
    manifest: EventLogManifest,
    *,
    overwrite: bool = False,
) -> Path:
    """Write ``manifest`` as stable JSON to ``path``."""
    if not isinstance(manifest, EventLogManifest):
        raise TypeError("manifest must be an EventLogManifest")
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"manifest already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def read_manifest(path: str | Path) -> EventLogManifest:
    """Read and validate an event-log manifest from JSON."""
    file_path = Path(path)
    try:
        payload: Any = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"failed to parse {file_path} as JSON: {exc.msg} at line {exc.lineno}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{file_path} must contain a JSON object")
    try:
        return EventLogManifest.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid event-log manifest {file_path}: {exc}") from exc


__all__ = [
    "EventLogManifest",
    "create_event_log_manifest",
    "read_manifest",
    "sha256_file",
    "write_manifest",
]
