"""Tests for chronoslob.features.order_flow."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from chronoslob.data.schemas import (
    BookEvent,
    EventType,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
)
from chronoslob.features.order_flow import (
    compute_order_flow_imbalance_from_snapshots,
    compute_order_flow_imbalance_series,
    compute_trade_imbalance_from_events,
)

T0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)


def _snapshot(
    t: datetime,
    bid_price: float,
    bid_qty: float,
    ask_price: float,
    ask_qty: float,
) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        timestamp=t,
        symbol="TEST",
        bids=[OrderBookLevel(price=bid_price, quantity=bid_qty)],
        asks=[OrderBookLevel(price=ask_price, quantity=ask_qty)],
    )


# ---------------------------------------------------------------------------
# OFI between two snapshots
# ---------------------------------------------------------------------------


def test_ofi_when_bid_price_improves_uses_current_bid_quantity() -> None:
    previous = _snapshot(T0, bid_price=100.0, bid_qty=2.0, ask_price=101.0, ask_qty=2.0)
    current = _snapshot(
        T0 + timedelta(seconds=1),
        bid_price=100.5,
        bid_qty=4.0,
        ask_price=101.0,
        ask_qty=2.0,
    )
    ofi = compute_order_flow_imbalance_from_snapshots(previous, current)
    # ask same: contribution = -(2 - 2) = 0; bid better: contribution = 4
    assert ofi == pytest.approx(4.0)


def test_ofi_when_bid_quantity_grows_at_same_price() -> None:
    previous = _snapshot(T0, bid_price=100.0, bid_qty=2.0, ask_price=101.0, ask_qty=2.0)
    current = _snapshot(
        T0 + timedelta(seconds=1),
        bid_price=100.0,
        bid_qty=5.0,
        ask_price=101.0,
        ask_qty=2.0,
    )
    ofi = compute_order_flow_imbalance_from_snapshots(previous, current)
    # bid same price: contribution = 5 - 2 = 3; ask same: contribution = 0
    assert ofi == pytest.approx(3.0)


def test_ofi_when_bid_price_worsens_uses_negative_previous_bid_quantity() -> None:
    previous = _snapshot(T0, bid_price=100.0, bid_qty=2.0, ask_price=101.0, ask_qty=2.0)
    current = _snapshot(
        T0 + timedelta(seconds=1),
        bid_price=99.5,
        bid_qty=10.0,
        ask_price=101.0,
        ask_qty=2.0,
    )
    ofi = compute_order_flow_imbalance_from_snapshots(previous, current)
    # bid worse: contribution = -2; ask same: 0
    assert ofi == pytest.approx(-2.0)


def test_ofi_when_ask_price_improves_uses_negative_current_ask_quantity() -> None:
    previous = _snapshot(T0, bid_price=100.0, bid_qty=2.0, ask_price=101.0, ask_qty=2.0)
    current = _snapshot(
        T0 + timedelta(seconds=1),
        bid_price=100.0,
        bid_qty=2.0,
        ask_price=100.5,
        ask_qty=3.0,
    )
    ofi = compute_order_flow_imbalance_from_snapshots(previous, current)
    # bid same: 0; ask better (down): -3
    assert ofi == pytest.approx(-3.0)


def test_ofi_when_ask_price_worsens_uses_positive_previous_ask_quantity() -> None:
    previous = _snapshot(T0, bid_price=100.0, bid_qty=2.0, ask_price=101.0, ask_qty=2.0)
    current = _snapshot(
        T0 + timedelta(seconds=1),
        bid_price=100.0,
        bid_qty=2.0,
        ask_price=101.5,
        ask_qty=10.0,
    )
    ofi = compute_order_flow_imbalance_from_snapshots(previous, current)
    # bid same: 0; ask worse (up): +2 (previous ask quantity)
    assert ofi == pytest.approx(2.0)


def test_ofi_rejects_backwards_timestamps() -> None:
    previous = _snapshot(T0 + timedelta(seconds=1), 100.0, 2.0, 101.0, 2.0)
    current = _snapshot(T0, 100.0, 2.0, 101.0, 2.0)
    with pytest.raises(ValueError, match="not be before"):
        compute_order_flow_imbalance_from_snapshots(previous, current)


def test_ofi_rejects_crossed_snapshots_by_default() -> None:
    previous = _snapshot(T0, 100.0, 2.0, 101.0, 2.0)
    crossed_current = OrderBookSnapshot(
        timestamp=T0 + timedelta(seconds=1),
        symbol="TEST",
        bids=[OrderBookLevel(price=102.0, quantity=2.0)],
        asks=[OrderBookLevel(price=101.0, quantity=2.0)],
    )
    with pytest.raises(ValueError):
        compute_order_flow_imbalance_from_snapshots(previous, crossed_current)


# ---------------------------------------------------------------------------
# OFI series
# ---------------------------------------------------------------------------


def test_ofi_series_first_value_defaults_to_zero() -> None:
    snaps = [
        _snapshot(T0, 100.0, 2.0, 101.0, 2.0),
        _snapshot(T0 + timedelta(seconds=1), 100.5, 4.0, 101.0, 2.0),
    ]
    series = compute_order_flow_imbalance_series(snaps)
    assert series[0] == 0.0
    assert series[1] == pytest.approx(4.0)


def test_ofi_series_first_value_can_be_nan() -> None:
    snaps = [_snapshot(T0, 100.0, 2.0, 101.0, 2.0)]
    series = compute_order_flow_imbalance_series(snaps, first_value_is_nan=True)
    assert math.isnan(series[0])


def test_ofi_series_rejects_unordered_timestamps() -> None:
    snaps = [
        _snapshot(T0 + timedelta(seconds=2), 100.0, 2.0, 101.0, 2.0),
        _snapshot(T0, 100.0, 2.0, 101.0, 2.0),
    ]
    with pytest.raises(ValueError, match="non-decreasing"):
        compute_order_flow_imbalance_series(snaps)


def test_ofi_series_handles_empty_input() -> None:
    assert compute_order_flow_imbalance_series([]) == []


# ---------------------------------------------------------------------------
# Trade imbalance
# ---------------------------------------------------------------------------


def _trade(side: Side | None, quantity: float | None) -> BookEvent:
    return BookEvent(
        timestamp=T0,
        event_type=EventType.TRADE,
        symbol="TEST",
        side=side,
        quantity=quantity,
        price=100.0,
    )


def test_trade_imbalance_with_bid_and_ask_trades() -> None:
    events = [
        _trade(Side.BID, 3.0),
        _trade(Side.ASK, 1.0),
    ]
    result = compute_trade_imbalance_from_events(events)
    assert result == pytest.approx((3.0 - 1.0) / 4.0)


def test_trade_imbalance_ignores_unknown_side() -> None:
    events = [
        _trade(None, 5.0),
        _trade(Side.BID, 1.0),
    ]
    result = compute_trade_imbalance_from_events(events)
    assert result == pytest.approx(1.0)


def test_trade_imbalance_ignores_non_trade_events() -> None:
    events = [
        BookEvent(
            timestamp=T0,
            event_type=EventType.ADD,
            symbol="TEST",
            side=Side.BID,
            price=100.0,
            quantity=10.0,
        ),
        _trade(Side.BID, 2.0),
    ]
    result = compute_trade_imbalance_from_events(events)
    assert result == pytest.approx(1.0)


def test_trade_imbalance_raises_when_no_usable_trades() -> None:
    with pytest.raises(ValueError, match="usable trade"):
        compute_trade_imbalance_from_events([])


def test_trade_imbalance_allow_empty_returns_nan() -> None:
    result = compute_trade_imbalance_from_events([], allow_empty=True)
    assert math.isnan(result)
