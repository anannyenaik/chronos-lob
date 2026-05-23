"""Tests for canonical event-log JSONL storage."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from chronoslob.data.event_store import (
    EventLogRecordType,
    deserialise_event_log_record,
    filter_event_log_records,
    iter_event_log_jsonl,
    read_event_log_jsonl,
    serialise_book_event,
    serialise_order_book_snapshot,
    sort_event_log_records,
    write_event_log_jsonl,
)
from chronoslob.data.schemas import (
    BookEvent,
    EventType,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _snapshot(
    sequence_id: int,
    *,
    timestamp: datetime | None = None,
    symbol: str = "TESTUSDT",
) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        timestamp=T0 + timedelta(seconds=sequence_id)
        if timestamp is None
        else timestamp,
        symbol=symbol,
        venue="synthetic",
        bids=[
            OrderBookLevel(price=100.0 + sequence_id * 0.01, quantity=1.0),
            OrderBookLevel(price=99.5, quantity=2.0),
        ],
        asks=[
            OrderBookLevel(price=100.5 + sequence_id * 0.01, quantity=1.5),
            OrderBookLevel(price=101.0, quantity=2.5),
        ],
        sequence_id=sequence_id,
        metadata={"source": "unit_test"},
    )


def _event(
    sequence_id: int,
    *,
    timestamp: datetime | None = None,
    symbol: str = "TESTUSDT",
) -> BookEvent:
    return BookEvent(
        timestamp=T0 + timedelta(seconds=sequence_id)
        if timestamp is None
        else timestamp,
        event_type=EventType.ADD,
        symbol=symbol,
        side=Side.BID,
        price=100.0,
        quantity=1.0,
        sequence_id=sequence_id,
        metadata={"source": "unit_test"},
    )


def test_serialise_deserialise_book_event() -> None:
    event = _event(1)

    record = serialise_book_event(event)
    restored = deserialise_event_log_record(record)

    assert record["record_type"] == EventLogRecordType.BOOK_EVENT.value
    assert isinstance(restored, BookEvent)
    assert restored.model_dump(mode="json") == event.model_dump(mode="json")


def test_serialise_deserialise_order_book_snapshot() -> None:
    snapshot = _snapshot(1)

    record = serialise_order_book_snapshot(snapshot)
    restored = deserialise_event_log_record(record)

    assert record["record_type"] == EventLogRecordType.ORDER_BOOK_SNAPSHOT.value
    assert isinstance(restored, OrderBookSnapshot)
    assert restored.model_dump(mode="json") == snapshot.model_dump(mode="json")


def test_write_read_jsonl_round_trip(tmp_path: Path) -> None:
    records = [_snapshot(1), _event(2), _snapshot(3)]
    path = tmp_path / "event_log.jsonl"

    written = write_event_log_jsonl(path, records)
    restored = read_event_log_jsonl(written)

    assert written == path
    assert [item.model_dump(mode="json") for item in restored] == [
        item.model_dump(mode="json") for item in records
    ]


def test_write_empty_log_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        write_event_log_jsonl(tmp_path / "empty.jsonl", [])


def test_overwrite_false_raises(tmp_path: Path) -> None:
    path = tmp_path / "event_log.jsonl"
    write_event_log_jsonl(path, [_snapshot(1)])

    with pytest.raises(FileExistsError):
        write_event_log_jsonl(path, [_snapshot(2)])


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_event_log_jsonl(tmp_path / "missing.jsonl")


def test_malformed_json_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(serialise_order_book_snapshot(_snapshot(1))) + "\nnot-json\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2"):
        read_event_log_jsonl(path)


def test_unsupported_record_type_raises() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        deserialise_event_log_record(
            {
                "record_type": "trade",
                "schema_version": "1.0",
                "payload": {"symbol": "TESTUSDT"},
            }
        )


def test_iter_event_log_jsonl_streams_records(tmp_path: Path) -> None:
    path = tmp_path / "event_log.jsonl"
    write_event_log_jsonl(path, [_snapshot(1), _event(2)])

    streamed = list(iter_event_log_jsonl(path))

    assert len(streamed) == 2
    assert isinstance(streamed[0], OrderBookSnapshot)
    assert isinstance(streamed[1], BookEvent)


def test_filter_by_symbol() -> None:
    records = [_snapshot(1), _event(2, symbol="OTHERUSDT"), _snapshot(3)]

    filtered = filter_event_log_records(records, symbol="TESTUSDT")

    assert [record.sequence_id for record in filtered] == [1, 3]


def test_filter_by_record_type() -> None:
    records = [_snapshot(1), _event(2), _snapshot(3)]

    filtered = filter_event_log_records(
        records,
        record_type=EventLogRecordType.BOOK_EVENT.value,
    )

    assert len(filtered) == 1
    assert isinstance(filtered[0], BookEvent)


def test_sort_event_log_records_deterministic_and_non_mutating() -> None:
    same_timestamp = T0 + timedelta(seconds=10)
    later = _snapshot(3, timestamp=T0 + timedelta(seconds=11))
    first = _event(1, timestamp=same_timestamp)
    second = _snapshot(2, timestamp=same_timestamp)
    records = [later, second, first]

    sorted_records = sort_event_log_records(records)

    assert [record.sequence_id for record in sorted_records] == [1, 2, 3]
    assert [record.sequence_id for record in records] == [3, 2, 1]
