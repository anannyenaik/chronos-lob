"""Tests for chronoslob.features.imbalance."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from chronoslob.data.schemas import OrderBookLevel, OrderBookSnapshot, Side
from chronoslob.features.imbalance import (
    compute_depth,
    compute_depth_imbalance,
    compute_depth_slope,
    compute_level_imbalances,
    compute_liquidity_concentration,
    compute_queue_imbalance,
)

T0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)


def _snapshot(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        timestamp=T0,
        symbol="TEST",
        bids=[OrderBookLevel(price=p, quantity=q) for p, q in bids],
        asks=[OrderBookLevel(price=p, quantity=q) for p, q in asks],
    )


# ---------------------------------------------------------------------------
# compute_depth
# ---------------------------------------------------------------------------


def test_compute_depth_sums_all_levels_by_default() -> None:
    levels = [
        OrderBookLevel(price=100.0, quantity=1.0),
        OrderBookLevel(price=99.0, quantity=2.0),
        OrderBookLevel(price=98.0, quantity=3.0),
    ]
    assert compute_depth(levels) == pytest.approx(6.0)


def test_compute_depth_respects_depth_limit() -> None:
    levels = [
        OrderBookLevel(price=100.0, quantity=1.0),
        OrderBookLevel(price=99.0, quantity=2.0),
        OrderBookLevel(price=98.0, quantity=3.0),
    ]
    assert compute_depth(levels, depth=2) == pytest.approx(3.0)


def test_compute_depth_rejects_zero_or_negative_depth() -> None:
    levels = [OrderBookLevel(price=100.0, quantity=1.0)]
    with pytest.raises(ValueError):
        compute_depth(levels, depth=0)
    with pytest.raises(ValueError):
        compute_depth(levels, depth=-1)


def test_compute_depth_handles_more_depth_than_levels() -> None:
    levels = [OrderBookLevel(price=100.0, quantity=1.0)]
    assert compute_depth(levels, depth=5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Depth imbalance
# ---------------------------------------------------------------------------


def test_depth_imbalance_formula() -> None:
    bids = [OrderBookLevel(price=100.0, quantity=3.0)]
    asks = [OrderBookLevel(price=101.0, quantity=1.0)]
    expected = (3.0 - 1.0) / (3.0 + 1.0)
    assert compute_depth_imbalance(bids, asks) == pytest.approx(expected)


def test_depth_imbalance_zero_denominator_raises() -> None:
    with pytest.raises(ValueError, match="> 0"):
        compute_depth_imbalance(
            [OrderBookLevel(price=100.0, quantity=0.0)],
            [OrderBookLevel(price=101.0, quantity=0.0)],
        )


# ---------------------------------------------------------------------------
# Queue imbalance
# ---------------------------------------------------------------------------


def test_queue_imbalance_balanced() -> None:
    assert compute_queue_imbalance(2.0, 2.0) == pytest.approx(0.0)


def test_queue_imbalance_bid_heavy_positive() -> None:
    assert compute_queue_imbalance(3.0, 1.0) > 0.0


def test_queue_imbalance_ask_heavy_negative() -> None:
    assert compute_queue_imbalance(1.0, 3.0) < 0.0


def test_queue_imbalance_rejects_zero_denominator() -> None:
    with pytest.raises(ValueError, match="> 0"):
        compute_queue_imbalance(0.0, 0.0)


def test_queue_imbalance_rejects_negative_quantities() -> None:
    with pytest.raises(ValueError):
        compute_queue_imbalance(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Level imbalances
# ---------------------------------------------------------------------------


def test_level_imbalances_emits_keys_for_requested_depths() -> None:
    snap = _snapshot(
        bids=[(100.0, 1.0), (99.0, 2.0)],
        asks=[(101.0, 2.0), (102.0, 4.0)],
    )
    out = compute_level_imbalances(snap, depths=(1, 5, 10))
    for d in (1, 5, 10):
        assert f"bid_depth_{d}" in out
        assert f"ask_depth_{d}" in out
        assert f"depth_imbalance_{d}" in out
    # Depth 1 should equal top quantities.
    assert out["bid_depth_1"] == pytest.approx(1.0)
    assert out["ask_depth_1"] == pytest.approx(2.0)
    # Depth 5 should saturate at the available 2 levels.
    assert out["bid_depth_5"] == pytest.approx(3.0)
    assert out["ask_depth_5"] == pytest.approx(6.0)
    assert "queue_imbalance" in out


def test_level_imbalances_omits_depth_imbalance_for_empty_sides() -> None:
    snap = OrderBookSnapshot(
        timestamp=T0,
        symbol="TEST",
        bids=[OrderBookLevel(price=100.0, quantity=0.0)],
        asks=[OrderBookLevel(price=101.0, quantity=0.0)],
    )
    out = compute_level_imbalances(snap, depths=(1,))
    assert "bid_depth_1" in out
    assert "ask_depth_1" in out
    assert "depth_imbalance_1" not in out
    assert "queue_imbalance" not in out


# ---------------------------------------------------------------------------
# Depth slope
# ---------------------------------------------------------------------------


def test_depth_slope_positive_for_growing_quantities() -> None:
    snap = _snapshot(
        bids=[(100.0, 1.0), (99.0, 2.0), (98.0, 3.0)],
        asks=[(101.0, 1.0), (102.0, 2.0), (103.0, 3.0)],
    )
    slope_bid = compute_depth_slope(snap, side=Side.BID)
    slope_ask = compute_depth_slope(snap, side=Side.ASK)
    assert slope_bid > 0.0
    assert slope_ask > 0.0


def test_depth_slope_raises_with_insufficient_levels() -> None:
    snap = _snapshot(
        bids=[(100.0, 1.0)],
        asks=[(101.0, 1.0)],
    )
    with pytest.raises(ValueError):
        compute_depth_slope(snap, side=Side.BID)


def test_depth_slope_respects_depth_limit() -> None:
    snap = _snapshot(
        bids=[(100.0, 1.0), (99.0, 2.0), (98.0, 3.0)],
        asks=[(101.0, 1.0), (102.0, 2.0)],
    )
    slope = compute_depth_slope(snap, side=Side.BID, depth=2)
    assert slope > 0.0


# ---------------------------------------------------------------------------
# Liquidity concentration
# ---------------------------------------------------------------------------


def test_liquidity_concentration_is_share_in_top_levels() -> None:
    snap = _snapshot(
        bids=[(100.0, 4.0), (99.0, 1.0), (98.0, 5.0)],
        asks=[(101.0, 2.0), (102.0, 2.0)],
    )
    result = compute_liquidity_concentration(snap, side=Side.BID, top_n=2)
    assert result == pytest.approx((4.0 + 1.0) / 10.0)


def test_liquidity_concentration_requires_positive_top_n() -> None:
    snap = _snapshot(bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])
    with pytest.raises(ValueError):
        compute_liquidity_concentration(snap, side=Side.BID, top_n=0)


def test_liquidity_concentration_requires_positive_total() -> None:
    snap = _snapshot(bids=[(100.0, 0.0)], asks=[(101.0, 1.0)])
    with pytest.raises(ValueError):
        compute_liquidity_concentration(snap, side=Side.BID, top_n=1)
