"""Canonical local JSONL event-log storage.

The event log format stores one schema-preserving JSON object per line.
Each record wraps either a :class:`BookEvent` or an
:class:`OrderBookSnapshot` payload together with a record type and schema
version. Helpers in this module are local-file only: they do not perform
network access, ingestion from venues or any model training.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from chronoslob.data.schemas import BookEvent, OrderBookSnapshot

DEFAULT_EVENT_LOG_SCHEMA_VERSION = "1.0"


class EventLogRecordType(StrEnum):
    """Supported canonical event-log record types."""

    BOOK_EVENT = "book_event"
    ORDER_BOOK_SNAPSHOT = "order_book_snapshot"


_SUPPORTED_RECORD_TYPES = {record_type.value for record_type in EventLogRecordType}
_SUPPORTED_SCHEMA_VERSIONS = {DEFAULT_EVENT_LOG_SCHEMA_VERSION}


class EventLogRecord(BaseModel):
    """A single schema-wrapped event-log record."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    record_type: str
    payload: dict[str, object] = Field(default_factory=dict)
    schema_version: str = DEFAULT_EVENT_LOG_SCHEMA_VERSION

    @field_validator("record_type")
    @classmethod
    def _validate_record_type(cls, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("record_type must be a non-empty string")
        if value not in _SUPPORTED_RECORD_TYPES:
            raise ValueError(f"unsupported event-log record_type {value!r}")
        return value

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("schema_version must be a non-empty string")
        if value not in _SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(f"unsupported event-log schema_version {value!r}")
        return value

    @field_validator("payload", mode="before")
    @classmethod
    def _validate_payload(cls, value: object) -> dict[str, object]:
        if not isinstance(value, Mapping):
            raise ValueError("payload must be a non-empty mapping")
        cleaned: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("payload keys must be non-empty strings")
            cleaned[key] = item
        if not cleaned:
            raise ValueError("payload must be non-empty")
        return cleaned


EventLogObject = BookEvent | OrderBookSnapshot


def serialise_book_event(event: BookEvent) -> dict[str, object]:
    """Return a JSON-serialisable event-log record for ``event``."""
    if not isinstance(event, BookEvent):
        raise TypeError("event must be a BookEvent")
    return {
        "record_type": EventLogRecordType.BOOK_EVENT.value,
        "schema_version": DEFAULT_EVENT_LOG_SCHEMA_VERSION,
        "payload": event.model_dump(mode="json"),
    }


def serialise_order_book_snapshot(snapshot: OrderBookSnapshot) -> dict[str, object]:
    """Return a JSON-serialisable event-log record for ``snapshot``."""
    if not isinstance(snapshot, OrderBookSnapshot):
        raise TypeError("snapshot must be an OrderBookSnapshot")
    return {
        "record_type": EventLogRecordType.ORDER_BOOK_SNAPSHOT.value,
        "schema_version": DEFAULT_EVENT_LOG_SCHEMA_VERSION,
        "payload": snapshot.model_dump(mode="json"),
    }


def deserialise_event_log_record(
    record: Mapping[str, object],
) -> BookEvent | OrderBookSnapshot:
    """Reconstruct a canonical schema object from an event-log record."""
    try:
        event_log_record = EventLogRecord.model_validate(record)
    except ValidationError as exc:
        raise ValueError(f"invalid event-log record: {exc}") from exc

    try:
        if event_log_record.record_type == EventLogRecordType.BOOK_EVENT.value:
            return BookEvent.model_validate(event_log_record.payload)
        if (
            event_log_record.record_type
            == EventLogRecordType.ORDER_BOOK_SNAPSHOT.value
        ):
            return OrderBookSnapshot.model_validate(event_log_record.payload)
    except ValidationError as exc:
        raise ValueError(
            f"invalid {event_log_record.record_type} payload: {exc}"
        ) from exc

    raise ValueError(
        f"unsupported event-log record_type {event_log_record.record_type!r}"
    )


def _serialise_record(record: EventLogObject) -> dict[str, object]:
    if isinstance(record, BookEvent):
        return serialise_book_event(record)
    if isinstance(record, OrderBookSnapshot):
        return serialise_order_book_snapshot(record)
    raise TypeError(
        "records must contain BookEvent or OrderBookSnapshot objects; "
        f"got {type(record).__name__}"
    )


def write_event_log_jsonl(
    path: str | Path,
    records: Sequence[BookEvent | OrderBookSnapshot],
    *,
    overwrite: bool = False,
) -> Path:
    """Write ``records`` to a local canonical JSONL event log.

    Empty logs are rejected so callers cannot accidentally create an
    apparently valid artefact without market-state content.
    """
    if not records:
        raise ValueError("event log records must be non-empty")

    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"event log already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            serialised = _serialise_record(record)
            handle.write(
                json.dumps(
                    serialised,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            handle.write("\n")
    return output_path


def _load_json_line(path: Path, line_number: int, line: str) -> Mapping[str, object]:
    try:
        payload: Any = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"malformed JSON in {path} on line {line_number}: {exc.msg}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} line {line_number} must be a JSON object")
    return payload


def iter_event_log_jsonl(path: str | Path) -> Iterator[BookEvent | OrderBookSnapshot]:
    """Yield validated event-log records from ``path`` without reordering."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            raw_record = _load_json_line(file_path, line_number, stripped)
            try:
                yield deserialise_event_log_record(raw_record)
            except ValueError as exc:
                raise ValueError(
                    f"invalid event-log record in {file_path} on line "
                    f"{line_number}: {exc}"
                ) from exc


def read_event_log_jsonl(path: str | Path) -> list[BookEvent | OrderBookSnapshot]:
    """Return all validated event-log records from ``path`` in file order."""
    return list(iter_event_log_jsonl(path))


def _record_type_for_object(record: EventLogObject) -> str:
    if isinstance(record, BookEvent):
        return EventLogRecordType.BOOK_EVENT.value
    if isinstance(record, OrderBookSnapshot):
        return EventLogRecordType.ORDER_BOOK_SNAPSHOT.value
    raise TypeError(
        "records must contain BookEvent or OrderBookSnapshot objects; "
        f"got {type(record).__name__}"
    )


def filter_event_log_records(
    records: Sequence[BookEvent | OrderBookSnapshot],
    *,
    symbol: str | None = None,
    record_type: str | None = None,
) -> list[BookEvent | OrderBookSnapshot]:
    """Return a filtered copy of ``records`` by symbol and/or record type."""
    if record_type is not None and record_type not in _SUPPORTED_RECORD_TYPES:
        raise ValueError(f"unsupported event-log record_type {record_type!r}")
    filtered: list[BookEvent | OrderBookSnapshot] = []
    for record in records:
        if symbol is not None and record.symbol != symbol:
            continue
        if record_type is not None and _record_type_for_object(record) != record_type:
            continue
        filtered.append(record)
    return filtered


def _sequence_sort_key(record: EventLogObject) -> tuple[bool, int]:
    sequence_id = record.sequence_id
    return (sequence_id is None, 0 if sequence_id is None else sequence_id)


def sort_event_log_records(
    records: Sequence[BookEvent | OrderBookSnapshot],
) -> list[BookEvent | OrderBookSnapshot]:
    """Return a copy sorted by timestamp and sequence id where available.

    This function intentionally changes record order. Python's stable
    sorting preserves input order among records with the same timestamp
    and no comparable sequence id.
    """
    return sorted(records, key=lambda record: (record.timestamp, _sequence_sort_key(record)))


__all__ = [
    "DEFAULT_EVENT_LOG_SCHEMA_VERSION",
    "EventLogObject",
    "EventLogRecord",
    "EventLogRecordType",
    "deserialise_event_log_record",
    "filter_event_log_records",
    "iter_event_log_jsonl",
    "read_event_log_jsonl",
    "serialise_book_event",
    "serialise_order_book_snapshot",
    "sort_event_log_records",
    "write_event_log_jsonl",
]
