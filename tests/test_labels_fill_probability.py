"""Tests for passive-fill proxy labels."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from chronoslob.data.schemas import OrderBookLevel, OrderBookSnapshot, Side
from chronoslob.labels.fill_probability import (
    compute_passive_fill_proxy,
    compute_passive_fill_proxy_series,
)

T0 = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)


def _snapshot(
    offset: int,
    bid_price: float,
    bid_quantity: float,
    ask_price: float,
    ask_quantity: float,
) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        timestamp=T0 + timedelta(seconds=offset),
        symbol="TEST",
        bids=[OrderBookLevel(price=bid_price, quantity=bid_quantity)],
        asks=[OrderBookLevel(price=ask_price, quantity=ask_quantity)],
    )


def test_passive_bid_fill_when_best_bid_quantity_decreases() -> None:
    snapshots = [
        _snapshot(0, 100.0, 10.0, 101.0, 10.0),
        _snapshot(1, 100.0, 9.0, 101.0, 10.0),
    ]
    assert compute_passive_fill_proxy(snapshots, 0, 1, Side.BID) is True


def test_passive_bid_fill_when_best_bid_price_improves() -> None:
    snapshots = [
        _snapshot(0, 100.0, 10.0, 101.0, 10.0),
        _snapshot(1, 100.2, 10.0, 101.2, 10.0),
    ]
    assert compute_passive_fill_proxy(snapshots, 0, 1, Side.BID) is True


def test_passive_ask_fill_when_best_ask_quantity_decreases() -> None:
    snapshots = [
        _snapshot(0, 100.0, 10.0, 101.0, 10.0),
        _snapshot(1, 100.0, 10.0, 101.0, 9.0),
    ]
    assert compute_passive_fill_proxy(snapshots, 0, 1, Side.ASK) is True


def test_passive_ask_fill_when_best_ask_price_improves_lower() -> None:
    snapshots = [
        _snapshot(0, 100.0, 10.0, 101.0, 10.0),
        _snapshot(1, 99.8, 10.0, 100.8, 10.0),
    ]
    assert compute_passive_fill_proxy(snapshots, 0, 1, Side.ASK) is True


def test_no_passive_fill_case() -> None:
    snapshots = [
        _snapshot(0, 100.0, 10.0, 101.0, 10.0),
        _snapshot(1, 100.0, 11.0, 101.0, 11.0),
    ]
    assert compute_passive_fill_proxy(snapshots, 0, 1, Side.BID) is False
    assert compute_passive_fill_proxy(snapshots, 0, 1, Side.ASK) is False


def test_insufficient_future_horizon_raises() -> None:
    snapshots = [_snapshot(0, 100.0, 10.0, 101.0, 10.0)]
    with pytest.raises(IndexError, match="insufficient future data"):
        compute_passive_fill_proxy(snapshots, 0, 1, Side.BID)


def test_crossed_book_raises() -> None:
    snapshots = [
        _snapshot(0, 101.0, 10.0, 100.0, 10.0),
        _snapshot(1, 101.0, 9.0, 100.0, 10.0),
    ]
    with pytest.raises(ValueError, match="crossed"):
        compute_passive_fill_proxy(snapshots, 0, 1, Side.BID)


def test_series_output_with_missing_drop() -> None:
    snapshots = [
        _snapshot(0, 100.0, 10.0, 101.0, 10.0),
        _snapshot(1, 100.0, 9.0, 101.0, 10.0),
        _snapshot(2, 100.0, 9.0, 101.0, 9.0),
    ]
    labels = compute_passive_fill_proxy_series(snapshots, horizon=1, side=Side.BID)
    assert labels == [True, False]
