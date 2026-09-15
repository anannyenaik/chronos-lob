"""Tests for turnover and position-path utilities."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from chronoslob.backtest.execution import ExecutionFill, ExecutionMode, TradeSide
from chronoslob.backtest.turnover import (
    compute_average_holding_period,
    compute_position_path,
    compute_trade_count,
    compute_turnover,
)


def _fill(
    index: int,
    side: TradeSide,
    quantity: float,
    price: float,
    *,
    filled: bool = True,
) -> ExecutionFill:
    return ExecutionFill(
        timestamp=datetime(2024, 1, 1, 9, 30, tzinfo=UTC) + timedelta(seconds=index),
        symbol="TEST",
        mode=ExecutionMode.AGGRESSIVE,
        side=side,
        filled=filled,
        fill_price=price if filled else None,
        quantity=quantity if filled else 0.0,
        fees=0.0,
        spread_cost=0.0,
        slippage=0.0,
        latency_steps=0,
        adverse_selection=False,
    )


def test_compute_trade_count_counts_filled_trades() -> None:
    trades = [
        _fill(0, TradeSide.BUY, 1.0, 100.0),
        _fill(1, TradeSide.SELL, 1.0, 101.0, filled=False),
    ]

    assert compute_trade_count(trades) == 1


def test_compute_turnover_by_quantity() -> None:
    trades = [
        _fill(0, TradeSide.BUY, 2.0, 100.0),
        _fill(1, TradeSide.SELL, 3.0, 101.0),
    ]

    summary = compute_turnover(trades)

    assert summary.turnover_basis == "quantity"
    assert summary.turnover == pytest.approx(5.0)
    assert summary.total_quantity == pytest.approx(5.0)
    assert summary.n_trades == 2


def test_compute_turnover_by_notional() -> None:
    trades = [
        _fill(0, TradeSide.BUY, 2.0, 100.0),
        _fill(1, TradeSide.SELL, 3.0, 101.0),
    ]

    summary = compute_turnover(trades, use_notional=True)

    assert summary.turnover_basis == "notional"
    assert summary.turnover == pytest.approx(503.0)
    assert summary.total_notional == pytest.approx(503.0)


def test_compute_position_path_updates_deterministically() -> None:
    trades = [
        _fill(0, TradeSide.BUY, 2.0, 100.0),
        _fill(1, TradeSide.SELL, 1.0, 101.0),
        _fill(2, TradeSide.SELL, 1.0, 102.0),
    ]

    assert compute_position_path(trades) == [2.0, 1.0, 0.0]


def test_empty_trades_are_handled_clearly() -> None:
    summary = compute_turnover([])

    assert summary.n_trades == 0
    assert summary.turnover == 0.0
    assert summary.average_trade_quantity is None
    assert summary.average_holding_period is None
    assert summary.position_path == []


def test_average_holding_period_uses_closed_position_intervals() -> None:
    trades = [
        _fill(0, TradeSide.BUY, 1.0, 100.0),
        _fill(1, TradeSide.BUY, 1.0, 100.1),
        _fill(2, TradeSide.SELL, 2.0, 100.2),
    ]

    assert compute_average_holding_period(trades) == pytest.approx(2.0)
