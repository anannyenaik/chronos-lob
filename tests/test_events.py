"""Tests for order book event helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from chronoslob.book.events import (
    has_duplicate_prices,
    sort_levels_for_side,
    top_of_book,
    validate_book_side_order,
)
from chronoslob.data.schemas import OrderBookLevel, OrderBookSnapshot, Side

T0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)


def _level(price: float, quantity: float = 1.0) -> OrderBookLevel:
    return OrderBookLevel(price=price, quantity=quantity)


# ---------------------------------------------------------------------------
# sort_levels_for_side
# ---------------------------------------------------------------------------


def test_sort_levels_for_side_bid_is_descending() -> None:
    levels = [_level(99.0), _level(100.0), _level(98.5)]
    result = sort_levels_for_side(levels, Side.BID)
    assert [level.price for level in result] == [100.0, 99.0, 98.5]


def test_sort_levels_for_side_ask_is_ascending() -> None:
    levels = [_level(101.0), _level(100.5), _level(102.0)]
    result = sort_levels_for_side(levels, Side.ASK)
    assert [level.price for level in result] == [100.5, 101.0, 102.0]


def test_sort_levels_for_side_does_not_mutate_input() -> None:
    levels = [_level(99.0), _level(100.0), _level(98.5)]
    snapshot_prices = [level.price for level in levels]

    sort_levels_for_side(levels, Side.BID)

    assert [level.price for level in levels] == snapshot_prices


def test_sort_levels_for_side_returns_new_list_instance() -> None:
    levels: list[OrderBookLevel] = [_level(100.0)]
    result = sort_levels_for_side(levels, Side.BID)
    assert result is not levels


def test_sort_levels_for_side_handles_empty_list() -> None:
    assert sort_levels_for_side([], Side.BID) == []
    assert sort_levels_for_side([], Side.ASK) == []


def test_sort_levels_for_side_rejects_non_side_argument() -> None:
    with pytest.raises(TypeError):
        sort_levels_for_side([_level(100.0)], "bid")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_book_side_order
# ---------------------------------------------------------------------------


def test_validate_book_side_order_accepts_correct_bid_order() -> None:
    levels = [_level(100.0), _level(99.0), _level(98.0)]
    validate_book_side_order(levels, Side.BID)


def test_validate_book_side_order_accepts_correct_ask_order() -> None:
    levels = [_level(100.0), _level(101.0), _level(102.0)]
    validate_book_side_order(levels, Side.ASK)


def test_validate_book_side_order_rejects_unsorted_bids() -> None:
    levels = [_level(99.0), _level(100.0)]
    with pytest.raises(ValueError, match="bid"):
        validate_book_side_order(levels, Side.BID)


def test_validate_book_side_order_rejects_unsorted_asks() -> None:
    levels = [_level(101.0), _level(100.0)]
    with pytest.raises(ValueError, match="ask"):
        validate_book_side_order(levels, Side.ASK)


def test_validate_book_side_order_rejects_equal_prices() -> None:
    levels = [_level(100.0), _level(100.0)]
    with pytest.raises(ValueError):
        validate_book_side_order(levels, Side.BID)
    with pytest.raises(ValueError):
        validate_book_side_order(levels, Side.ASK)


def test_validate_book_side_order_accepts_short_lists() -> None:
    validate_book_side_order([], Side.BID)
    validate_book_side_order([_level(100.0)], Side.ASK)


# ---------------------------------------------------------------------------
# has_duplicate_prices
# ---------------------------------------------------------------------------


def test_has_duplicate_prices_detects_duplicates() -> None:
    assert has_duplicate_prices([_level(100.0), _level(100.0)]) is True


def test_has_duplicate_prices_unique_returns_false() -> None:
    assert has_duplicate_prices([_level(100.0), _level(101.0)]) is False


def test_has_duplicate_prices_handles_empty_and_single() -> None:
    assert has_duplicate_prices([]) is False
    assert has_duplicate_prices([_level(100.0)]) is False


# ---------------------------------------------------------------------------
# top_of_book
# ---------------------------------------------------------------------------


def test_top_of_book_returns_best_bid_and_ask() -> None:
    snap = OrderBookSnapshot(
        timestamp=T0,
        symbol="X",
        bids=[_level(100.0), _level(99.0)],
        asks=[_level(101.0), _level(102.0)],
    )
    bid, ask = top_of_book(snap)
    assert bid is not None and bid.price == 100.0
    assert ask is not None and ask.price == 101.0


def test_top_of_book_returns_none_for_empty_sides() -> None:
    snap = OrderBookSnapshot(timestamp=T0, symbol="X")
    bid, ask = top_of_book(snap)
    assert bid is None
    assert ask is None


def test_top_of_book_returns_none_for_missing_side() -> None:
    snap = OrderBookSnapshot(
        timestamp=T0,
        symbol="X",
        bids=[_level(100.0)],
    )
    bid, ask = top_of_book(snap)
    assert bid is not None and bid.price == 100.0
    assert ask is None
