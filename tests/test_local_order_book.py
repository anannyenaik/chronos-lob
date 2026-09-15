"""Tests for local offline order book state management."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from chronoslob.book.local_order_book import LocalOrderBook, LocalOrderBookConfig
from chronoslob.data.binance import (
    BinanceDepthLevel,
    BinanceDepthSnapshot,
    BinanceDiffDepthEvent,
)
from chronoslob.data.schemas import Side

T0 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
T1 = datetime(2024, 1, 1, 12, 0, 1, tzinfo=UTC)


def _level(price: float, quantity: float) -> BinanceDepthLevel:
    return BinanceDepthLevel(price=price, quantity=quantity)


def _snapshot(
    *,
    symbol: str = "TESTUSDT",
    last_update_id: int = 100,
    bids: list[tuple[float, float]] | None = None,
    asks: list[tuple[float, float]] | None = None,
) -> BinanceDepthSnapshot:
    return BinanceDepthSnapshot(
        symbol=symbol,
        last_update_id=last_update_id,
        bids=[
            _level(price, quantity)
            for price, quantity in (_default_bids() if bids is None else bids)
        ],
        asks=[
            _level(price, quantity)
            for price, quantity in (_default_asks() if asks is None else asks)
        ],
        timestamp=T0,
    )


def _event(
    *,
    symbol: str = "TESTUSDT",
    first_update_id: int = 101,
    final_update_id: int = 101,
    previous_final_update_id: int | None = 100,
    bids: list[tuple[float, float]] | None = None,
    asks: list[tuple[float, float]] | None = None,
) -> BinanceDiffDepthEvent:
    return BinanceDiffDepthEvent(
        symbol=symbol,
        first_update_id=first_update_id,
        final_update_id=final_update_id,
        previous_final_update_id=previous_final_update_id,
        bids=[_level(price, quantity) for price, quantity in (bids or [])],
        asks=[_level(price, quantity) for price, quantity in (asks or [])],
        transaction_time=T1,
    )


def _default_bids() -> list[tuple[float, float]]:
    return [(100.0, 1.0), (99.5, 2.0), (99.0, 3.0)]


def _default_asks() -> list[tuple[float, float]]:
    return [(100.5, 1.5), (101.0, 2.5), (101.5, 3.5)]


def _book(max_depth: int | None = None, *, allow_crossed: bool = False) -> LocalOrderBook:
    return LocalOrderBook(
        LocalOrderBookConfig(
            symbol="TESTUSDT",
            max_depth=max_depth,
            allow_crossed=allow_crossed,
        )
    )


def test_load_snapshot_replaces_state() -> None:
    book = _book()
    book.load_snapshot(_snapshot())
    book.load_snapshot(
        _snapshot(
            last_update_id=200,
            bids=[(98.0, 4.0)],
            asks=[(102.0, 5.0)],
        )
    )

    assert book.last_update_id == 200
    assert book.depth_counts() == {"bids": 1, "asks": 1}
    assert book.best_bid() is not None
    assert book.best_bid().price == 98.0
    assert book.best_ask() is not None
    assert book.best_ask().price == 102.0


def test_best_bid_and_best_ask() -> None:
    book = _book()
    book.load_snapshot(_snapshot())

    assert book.best_bid() is not None
    assert book.best_bid().price == 100.0
    assert book.best_ask() is not None
    assert book.best_ask().price == 100.5


def test_levels_are_sorted_correctly() -> None:
    book = _book()
    book.load_snapshot(
        _snapshot(
            bids=[(99.0, 3.0), (100.0, 1.0), (99.5, 2.0)],
            asks=[(101.0, 2.5), (100.5, 1.5), (101.5, 3.5)],
        )
    )

    assert [level.price for level in book.levels(Side.BID)] == [100.0, 99.5, 99.0]
    assert [level.price for level in book.levels(Side.ASK)] == [100.5, 101.0, 101.5]


def test_apply_diff_updates_quantity() -> None:
    book = _book()
    book.load_snapshot(_snapshot())

    book.apply_diff(_event(bids=[(100.0, 1.25)]))

    assert book.best_bid() is not None
    assert book.best_bid().quantity == 1.25
    assert book.last_update_id == 101
    assert book.timestamp == T1


def test_apply_diff_removes_zero_quantity_level() -> None:
    book = _book()
    book.load_snapshot(_snapshot())

    book.apply_diff(_event(bids=[(99.0, 0.0)]))

    assert [level.price for level in book.levels(Side.BID)] == [100.0, 99.5]


def test_apply_diff_adds_new_level() -> None:
    book = _book()
    book.load_snapshot(_snapshot())

    book.apply_diff(_event(bids=[(98.5, 4.0)], asks=[(101.75, 1.0)]))

    assert [level.price for level in book.levels(Side.BID)] == [
        100.0,
        99.5,
        99.0,
        98.5,
    ]
    assert [level.price for level in book.levels(Side.ASK)] == [
        100.5,
        101.0,
        101.5,
        101.75,
    ]


def test_max_depth_trimming() -> None:
    book = _book(max_depth=2)

    book.load_snapshot(_snapshot())
    book.apply_diff(_event(bids=[(100.25, 0.5)], asks=[(100.75, 0.5)]))

    assert book.depth_counts() == {"bids": 2, "asks": 2}
    assert [level.price for level in book.levels(Side.BID)] == [100.25, 100.0]
    assert [level.price for level in book.levels(Side.ASK)] == [100.5, 100.75]


def test_symbol_mismatch_raises() -> None:
    book = _book()
    book.load_snapshot(_snapshot())

    with pytest.raises(ValueError, match="symbol"):
        book.apply_diff(_event(symbol="OTHERUSDT"))


def test_apply_before_snapshot_raises() -> None:
    book = _book()

    with pytest.raises(RuntimeError, match="before load_snapshot"):
        book.apply_diff(_event())


def test_crossed_book_raises_when_not_allowed() -> None:
    book = _book()
    book.load_snapshot(_snapshot())

    with pytest.raises(ValueError, match="crossed"):
        book.apply_diff(_event(asks=[(99.5, 1.0)]))


def test_depth_counts() -> None:
    book = _book()
    book.load_snapshot(_snapshot())

    assert book.depth_counts() == {"bids": 3, "asks": 3}
