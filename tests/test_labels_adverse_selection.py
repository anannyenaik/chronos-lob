"""Tests for adverse-selection proxy labels."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from chronoslob.data.schemas import OrderBookLevel, OrderBookSnapshot, Side
from chronoslob.labels.adverse_selection import (
    compute_adverse_selection_after_fill_proxy,
    compute_adverse_selection_proxy_series,
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


def test_bid_adverse_selection_when_price_falls_after_proxy_fill() -> None:
    snapshots = [
        _snapshot(0, 100.0, 10.0, 101.0, 10.0),
        _snapshot(1, 100.0, 9.0, 101.0, 10.0),
        _snapshot(2, 99.0, 10.0, 100.0, 10.0),
    ]
    assert (
        compute_adverse_selection_after_fill_proxy(
            snapshots,
            0,
            fill_horizon=1,
            evaluation_horizon=2,
            side=Side.BID,
        )
        is True
    )


def test_ask_adverse_selection_when_price_rises_after_proxy_fill() -> None:
    snapshots = [
        _snapshot(0, 100.0, 10.0, 101.0, 10.0),
        _snapshot(1, 100.0, 10.0, 101.0, 9.0),
        _snapshot(2, 101.0, 10.0, 102.0, 10.0),
    ]
    assert (
        compute_adverse_selection_after_fill_proxy(
            snapshots,
            0,
            fill_horizon=1,
            evaluation_horizon=2,
            side=Side.ASK,
        )
        is True
    )


def test_no_adverse_selection_when_no_proxy_fill() -> None:
    snapshots = [
        _snapshot(0, 100.0, 10.0, 101.0, 10.0),
        _snapshot(1, 100.0, 11.0, 101.0, 11.0),
        _snapshot(2, 99.0, 10.0, 100.0, 10.0),
    ]
    assert (
        compute_adverse_selection_after_fill_proxy(
            snapshots,
            0,
            fill_horizon=1,
            evaluation_horizon=2,
            side=Side.BID,
        )
        is False
    )


def test_evaluation_horizon_before_fill_horizon_raises() -> None:
    snapshots = [
        _snapshot(0, 100.0, 10.0, 101.0, 10.0),
        _snapshot(1, 100.0, 9.0, 101.0, 10.0),
    ]
    with pytest.raises(ValueError, match=">="):
        compute_adverse_selection_after_fill_proxy(
            snapshots,
            0,
            fill_horizon=2,
            evaluation_horizon=1,
            side=Side.BID,
        )


def test_adverse_selection_series_output() -> None:
    snapshots = [
        _snapshot(0, 100.0, 10.0, 101.0, 10.0),
        _snapshot(1, 100.0, 9.0, 101.0, 10.0),
        _snapshot(2, 99.0, 10.0, 100.0, 10.0),
    ]
    labels = compute_adverse_selection_proxy_series(
        snapshots,
        fill_horizon=1,
        evaluation_horizon=2,
        side=Side.BID,
    )
    assert labels == [True]
