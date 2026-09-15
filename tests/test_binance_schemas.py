"""Tests for offline Binance-style depth schemas and parsers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from chronoslob.data.binance import (
    BinanceDepthSnapshot,
    load_binance_diff_events_jsonl,
    load_binance_snapshot_json,
    parse_binance_diff_event,
    parse_binance_snapshot,
    to_order_book_snapshot,
)
from chronoslob.data.schemas import OrderBookSnapshot


def test_parse_snapshot_with_string_price_quantity_sorts_sides() -> None:
    payload = {
        "symbol": "TESTUSDT",
        "lastUpdateId": "100",
        "bids": [["99.50", "2.0"], ["100.00", "1.0"]],
        "asks": [["101.00", "2.5"], ["100.50", "1.5"]],
    }

    snapshot = parse_binance_snapshot(payload)

    assert snapshot.symbol == "TESTUSDT"
    assert snapshot.last_update_id == 100
    assert [level.price for level in snapshot.bids] == [100.0, 99.5]
    assert [level.quantity for level in snapshot.bids] == [1.0, 2.0]
    assert [level.price for level in snapshot.asks] == [100.5, 101.0]


def test_parse_diff_event_with_binance_style_keys() -> None:
    event = parse_binance_diff_event(
        {
            "e": "depthUpdate",
            "E": 1700000001000,
            "T": 1700000002000,
            "s": "TESTUSDT",
            "U": 101,
            "u": 105,
            "pu": 100,
            "b": [["100.00", "1.25"]],
            "a": [["100.50", "0"]],
        }
    )

    assert event.event_type == "depthUpdate"
    assert event.symbol == "TESTUSDT"
    assert event.first_update_id == 101
    assert event.final_update_id == 105
    assert event.previous_final_update_id == 100
    assert event.event_time is not None
    assert event.event_time.tzinfo is not None
    assert event.event_time.utcoffset() == UTC.utcoffset(event.event_time)
    assert event.transaction_time is not None
    assert event.bids[0].quantity == 1.25
    assert event.asks[0].quantity == 0.0


def test_parse_diff_event_with_fixture_alias_keys() -> None:
    event_time = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    event = parse_binance_diff_event(
        {
            "event_type": "depthUpdate",
            "event_time": event_time.isoformat(),
            "transaction_time": event_time,
            "symbol": "TESTUSDT",
            "first_update_id": "106",
            "final_update_id": "110",
            "previous_final_update_id": "105",
            "bids": [{"price": "99.50", "quantity": "0"}],
            "asks": [{"price": "101.00", "quantity": "3.0"}],
            "received_timestamp": event_time.isoformat(),
        }
    )

    assert event.symbol == "TESTUSDT"
    assert event.first_update_id == 106
    assert event.final_update_id == 110
    assert event.previous_final_update_id == 105
    assert event.event_time == event_time
    assert event.transaction_time == event_time
    assert event.received_timestamp == event_time
    assert event.bids[0].quantity == 0.0
    assert event.asks[0].price == 101.0


def test_duplicate_price_levels_raise() -> None:
    with pytest.raises(ValueError, match="duplicate price"):
        parse_binance_snapshot(
            {
                "symbol": "TESTUSDT",
                "lastUpdateId": 100,
                "bids": [["100.00", "1.0"], ["100.00", "2.0"]],
                "asks": [["101.00", "1.0"]],
            }
        )

    with pytest.raises(ValueError, match="duplicate price"):
        parse_binance_diff_event(
            {
                "s": "TESTUSDT",
                "U": 101,
                "u": 105,
                "b": [["100.00", "1.0"], ["100.00", "2.0"]],
                "a": [],
            }
        )


def test_invalid_update_id_order_raises() -> None:
    with pytest.raises(ValueError, match="first_update_id"):
        parse_binance_diff_event(
            {
                "s": "TESTUSDT",
                "U": 110,
                "u": 105,
                "b": [],
                "a": [],
            }
        )


def test_jsonl_loader_gives_line_numbered_error_on_bad_json(tmp_path: Path) -> None:
    path = tmp_path / "bad_updates.jsonl"
    path.write_text('{"s": "TESTUSDT", "U": 1\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_binance_diff_events_jsonl(path)


def test_jsonl_loader_gives_line_numbered_error_on_invalid_event(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad_event.jsonl"
    path.write_text('{"s": "TESTUSDT", "U": 2, "u": 1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_binance_diff_events_jsonl(path)


def test_snapshot_conversion_to_order_book_snapshot() -> None:
    timestamp = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    snapshot = parse_binance_snapshot(
        {
            "symbol": "TESTUSDT",
            "lastUpdateId": 100,
            "bids": [["100.00", "1.0"], ["99.50", "2.0"]],
            "asks": [["100.50", "1.5"], ["101.00", "2.5"]],
        },
        timestamp=timestamp,
    )

    canonical = to_order_book_snapshot(snapshot, depth=1)

    assert isinstance(canonical, OrderBookSnapshot)
    assert canonical.timestamp == timestamp
    assert canonical.symbol == "TESTUSDT"
    assert canonical.venue == "binance"
    assert canonical.sequence_id == 100
    assert [level.price for level in canonical.bids] == [100.0]
    assert [level.price for level in canonical.asks] == [100.5]
    assert canonical.metadata["source"] == "binance_snapshot"


def test_snapshot_conversion_marks_synthetic_timestamp_when_missing() -> None:
    snapshot = parse_binance_snapshot(
        {
            "symbol": "TESTUSDT",
            "lastUpdateId": 100,
            "bids": [["100.00", "1.0"]],
            "asks": [["100.50", "1.5"]],
        }
    )

    canonical = to_order_book_snapshot(snapshot)

    assert canonical.timestamp == datetime(1970, 1, 1, tzinfo=UTC)
    assert canonical.metadata["synthetic_time"] is True
    assert canonical.metadata["source"] == "binance_snapshot"


def test_fixture_snapshot_loads_from_local_file() -> None:
    path = Path(__file__).parent / "fixtures" / "binance" / "synthetic_snapshot.json"

    snapshot = load_binance_snapshot_json(path)

    assert path.exists()
    assert isinstance(snapshot, BinanceDepthSnapshot)
    assert snapshot.symbol == "TESTUSDT"
