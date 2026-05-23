"""Tests for deterministic snapshot-plus-diff reconstruction."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from chronoslob.book.reconstruction import (
    ReconstructionStatus,
    has_update_gap,
    is_stale_event,
    reconstruct_order_book,
    should_apply_first_diff,
)
from chronoslob.data.binance import (
    BinanceDepthLevel,
    BinanceDepthSnapshot,
    BinanceDiffDepthEvent,
    load_binance_diff_events_jsonl,
    load_binance_snapshot_json,
)

T0 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
T1 = datetime(2024, 1, 1, 12, 0, 1, tzinfo=UTC)
FIXTURES = Path(__file__).parent / "fixtures" / "binance"


def _level(price: float, quantity: float) -> BinanceDepthLevel:
    return BinanceDepthLevel(price=price, quantity=quantity)


def _snapshot() -> BinanceDepthSnapshot:
    return BinanceDepthSnapshot(
        symbol="TESTUSDT",
        last_update_id=100,
        bids=[_level(100.0, 1.0), _level(99.5, 2.0)],
        asks=[_level(100.5, 1.5), _level(101.0, 2.5)],
        timestamp=T0,
    )


def _event(
    *,
    first_update_id: int,
    final_update_id: int,
    previous_final_update_id: int | None = None,
    bids: list[tuple[float, float]] | None = None,
    asks: list[tuple[float, float]] | None = None,
) -> BinanceDiffDepthEvent:
    return BinanceDiffDepthEvent(
        symbol="TESTUSDT",
        first_update_id=first_update_id,
        final_update_id=final_update_id,
        previous_final_update_id=previous_final_update_id,
        bids=[_level(price, quantity) for price, quantity in (bids or [])],
        asks=[_level(price, quantity) for price, quantity in (asks or [])],
        transaction_time=T1,
    )


def test_should_apply_first_diff_condition() -> None:
    assert should_apply_first_diff(
        100,
        _event(first_update_id=99, final_update_id=101),
    )
    assert not should_apply_first_diff(
        100,
        _event(first_update_id=98, final_update_id=100),
    )
    assert not should_apply_first_diff(
        100,
        _event(first_update_id=102, final_update_id=105),
    )


def test_stale_events_skipped() -> None:
    result = reconstruct_order_book(
        _snapshot(),
        [
            _event(first_update_id=95, final_update_id=100),
            _event(
                first_update_id=101,
                final_update_id=101,
                previous_final_update_id=100,
                bids=[(100.0, 1.25)],
            ),
        ],
    )

    assert result.ok
    assert result.n_snapshots == 1
    assert len(result.issues) == 1
    assert result.issues[0].status == ReconstructionStatus.STALE_EVENT_SKIPPED


def test_gap_detection_via_pu() -> None:
    first = _event(
        first_update_id=101,
        final_update_id=101,
        previous_final_update_id=100,
        bids=[(100.0, 1.25)],
    )
    second = _event(
        first_update_id=102,
        final_update_id=102,
        previous_final_update_id=999,
        asks=[(100.5, 2.0)],
    )

    assert has_update_gap(101, second)
    result = reconstruct_order_book(_snapshot(), [first, second])

    assert not result.ok
    assert result.gap_count == 1
    assert result.n_snapshots == 1
    assert result.final_update_id == 101


def test_gap_detection_via_u_when_pu_absent() -> None:
    first = _event(first_update_id=101, final_update_id=101, bids=[(100.0, 1.25)])
    second = _event(first_update_id=103, final_update_id=103, asks=[(100.5, 2.0)])

    assert has_update_gap(101, second)
    result = reconstruct_order_book(_snapshot(), [first, second])

    assert not result.ok
    assert result.gap_count == 1
    assert result.n_snapshots == 1


def test_valid_reconstruction_produces_expected_number_of_snapshots() -> None:
    snapshot = load_binance_snapshot_json(FIXTURES / "synthetic_snapshot.json")
    events = load_binance_diff_events_jsonl(FIXTURES / "synthetic_diff_updates.jsonl")

    result = reconstruct_order_book(snapshot, events)

    assert result.ok
    assert result.n_snapshots == 3
    assert result.final_update_id == 115
    assert result.snapshots[-1].best_bid is not None
    assert result.snapshots[-1].best_bid.price == 100.0


def test_stop_on_gap_stops() -> None:
    snapshot = load_binance_snapshot_json(FIXTURES / "synthetic_snapshot.json")
    events = load_binance_diff_events_jsonl(FIXTURES / "synthetic_gap_updates.jsonl")

    result = reconstruct_order_book(snapshot, events, stop_on_gap=True)

    assert not result.ok
    assert result.gap_count == 1
    assert result.n_snapshots == 1
    assert result.final_update_id == 105


def test_crossed_update_records_issue() -> None:
    snapshot = load_binance_snapshot_json(FIXTURES / "synthetic_snapshot.json")
    events = load_binance_diff_events_jsonl(
        FIXTURES / "synthetic_crossed_updates.jsonl"
    )

    result = reconstruct_order_book(snapshot, events)

    assert not result.ok
    assert result.crossed_count == 1
    assert result.n_snapshots == 0
    assert result.final_update_id == 100
    assert result.issues[0].status == ReconstructionStatus.CROSSED_BOOK


def test_final_update_id_correct() -> None:
    snapshot = _snapshot()
    events = [
        _event(first_update_id=101, final_update_id=101, bids=[(100.0, 1.25)]),
        _event(first_update_id=102, final_update_id=102, asks=[(100.5, 2.0)]),
    ]

    result = reconstruct_order_book(snapshot, events)

    assert result.final_update_id == 102


def test_deterministic_repeated_reconstruction() -> None:
    snapshot = load_binance_snapshot_json(FIXTURES / "synthetic_snapshot.json")
    events = load_binance_diff_events_jsonl(FIXTURES / "synthetic_diff_updates.jsonl")

    first = reconstruct_order_book(snapshot, events)
    second = reconstruct_order_book(snapshot, events)

    first_snapshots = [item.model_dump(mode="json") for item in first.snapshots]
    second_snapshots = [item.model_dump(mode="json") for item in second.snapshots]

    assert first.ok == second.ok
    assert first.final_update_id == second.final_update_id
    assert first_snapshots == second_snapshots
    assert first.issues == second.issues


def test_is_stale_event() -> None:
    assert is_stale_event(100, _event(first_update_id=95, final_update_id=100))
    assert not is_stale_event(100, _event(first_update_id=101, final_update_id=101))
