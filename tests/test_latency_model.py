"""Tests for latency row-step utilities."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from chronoslob.backtest.execution import MarketState
from chronoslob.backtest.latency import (
    LatencyConfig,
    apply_latency,
    get_latency_state,
    latency_sensitivity_grid,
)


def _state(index: int) -> MarketState:
    mid = 100.0 + index
    spread = 0.02
    return MarketState(
        timestamp=datetime(2024, 1, 1, 9, 30, tzinfo=UTC) + timedelta(seconds=index),
        symbol="TEST",
        mid_price=mid,
        best_bid=mid - spread / 2.0,
        best_ask=mid + spread / 2.0,
        spread=spread,
        bid_size=10.0,
        ask_size=11.0,
    )


def test_zero_latency_returns_current_state() -> None:
    states = [_state(index) for index in range(3)]

    result = apply_latency(1, states, LatencyConfig(latency_steps=0))

    assert result.executable
    assert result.state == states[1]
    assert get_latency_state(1, states).mid_price == pytest.approx(101.0)


def test_positive_latency_returns_future_state() -> None:
    states = [_state(index) for index in range(4)]

    result = apply_latency(0, states, LatencyConfig(latency_steps=2))

    assert result.executable
    assert result.target_index == 2
    assert result.state == states[2]


def test_out_of_range_latency_marks_signal_unexecutable() -> None:
    states = [_state(index) for index in range(3)]

    result = apply_latency(2, states, LatencyConfig(latency_steps=1))

    assert not result.executable
    assert result.state is None
    assert result.reason == "latency_out_of_range"
    with pytest.raises(IndexError, match="latency_steps push beyond"):
        get_latency_state(2, states, LatencyConfig(latency_steps=1))


def test_latency_sensitivity_grid_generation() -> None:
    grid = latency_sensitivity_grid([0, 1, 2, 5, 10])

    assert [item.latency_steps for item in grid] == [0, 1, 2, 5, 10]


def test_latency_grid_is_deterministic() -> None:
    first = latency_sensitivity_grid([0, 2, 4])
    second = latency_sensitivity_grid([0, 2, 4])

    assert first == second


def test_negative_latency_raises_clearly() -> None:
    with pytest.raises(ValueError, match="latency_steps"):
        LatencyConfig(latency_steps=-1)
