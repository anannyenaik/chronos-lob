"""Tests for chronoslob.features.microprice."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from chronoslob.data.schemas import OrderBookLevel, OrderBookSnapshot
from chronoslob.features.microprice import (
    compute_microprice,
    compute_mid_price,
    compute_relative_spread,
    compute_snapshot_price_features,
    compute_spread,
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
# Mid-price / spread / relative spread
# ---------------------------------------------------------------------------


def test_mid_price_is_midpoint_of_bid_and_ask() -> None:
    assert compute_mid_price(100.0, 101.0) == pytest.approx(100.5)


def test_spread_is_ask_minus_bid() -> None:
    assert compute_spread(100.0, 101.0) == pytest.approx(1.0)


def test_relative_spread_is_spread_over_mid() -> None:
    assert compute_relative_spread(100.0, 101.0) == pytest.approx(1.0 / 100.5)


@pytest.mark.parametrize("bid,ask", [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0)])
def test_mid_price_rejects_non_positive(bid: float, ask: float) -> None:
    with pytest.raises(ValueError):
        compute_mid_price(bid, ask)


def test_mid_price_rejects_crossed_book_by_default() -> None:
    with pytest.raises(ValueError, match="strictly less than"):
        compute_mid_price(101.0, 100.0)


def test_spread_rejects_crossed_book_by_default() -> None:
    with pytest.raises(ValueError):
        compute_spread(101.0, 100.0)


def test_relative_spread_rejects_crossed_book_by_default() -> None:
    with pytest.raises(ValueError):
        compute_relative_spread(101.0, 100.0)


def test_mid_price_allows_crossed_when_opted_in() -> None:
    result = compute_mid_price(101.0, 100.0, allow_crossed=True)
    assert result == pytest.approx(100.5)


def test_mid_price_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        compute_mid_price(math.nan, 101.0)
    with pytest.raises(ValueError):
        compute_mid_price(100.0, math.inf)


# ---------------------------------------------------------------------------
# Microprice
# ---------------------------------------------------------------------------


def test_microprice_formula_matches_spec() -> None:
    bid = 100.0
    ask = 101.0
    bid_qty = 1.0
    ask_qty = 3.0
    expected = (ask * bid_qty + bid * ask_qty) / (bid_qty + ask_qty)
    assert compute_microprice(bid, ask, bid_qty, ask_qty) == pytest.approx(expected)


def test_microprice_with_equal_sizes_equals_mid() -> None:
    assert compute_microprice(100.0, 101.0, 2.0, 2.0) == pytest.approx(100.5)


def test_microprice_tilts_towards_thin_side() -> None:
    # When the bid is thin (small bid_qty) and ask is heavy, the microprice
    # should sit closer to the bid because the formula puts ask_qty on bid.
    bid = 100.0
    ask = 101.0
    micro = compute_microprice(bid, ask, bid_quantity=0.1, ask_quantity=10.0)
    assert bid < micro < (bid + ask) / 2.0


def test_microprice_rejects_zero_denominator() -> None:
    with pytest.raises(ValueError, match="> 0"):
        compute_microprice(100.0, 101.0, 0.0, 0.0)


def test_microprice_rejects_negative_quantities() -> None:
    with pytest.raises(ValueError):
        compute_microprice(100.0, 101.0, -1.0, 1.0)


def test_microprice_rejects_crossed_by_default() -> None:
    with pytest.raises(ValueError):
        compute_microprice(101.0, 100.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Snapshot price features
# ---------------------------------------------------------------------------


def test_snapshot_price_features_contain_expected_keys() -> None:
    snap = _snapshot(bids=[(100.0, 2.0)], asks=[(101.0, 3.0)])
    features = compute_snapshot_price_features(snap)
    assert features["mid_price"] == pytest.approx(100.5)
    assert features["spread"] == pytest.approx(1.0)
    assert features["relative_spread"] == pytest.approx(1.0 / 100.5)
    assert features["best_bid_price"] == pytest.approx(100.0)
    assert features["best_ask_price"] == pytest.approx(101.0)
    assert features["best_bid_quantity"] == pytest.approx(2.0)
    assert features["best_ask_quantity"] == pytest.approx(3.0)
    assert "microprice" in features


def test_snapshot_price_features_skip_microprice_when_no_top_quantity() -> None:
    snap = _snapshot(bids=[(100.0, 0.0)], asks=[(101.0, 0.0)])
    features = compute_snapshot_price_features(snap)
    assert "microprice" not in features
    assert features["spread"] == pytest.approx(1.0)


def test_snapshot_price_features_require_both_sides() -> None:
    empty_asks = OrderBookSnapshot(
        timestamp=T0,
        symbol="TEST",
        bids=[OrderBookLevel(price=100.0, quantity=1.0)],
        asks=[],
    )
    with pytest.raises(ValueError, match="bid and one ask"):
        compute_snapshot_price_features(empty_asks)
