"""Tests for future mid-price labels."""

from __future__ import annotations

import math

import pytest

from chronoslob.labels.midprice import (
    classify_direction,
    compute_direction_labels,
    compute_future_return,
    compute_future_returns,
    compute_return_quantile_labels,
)


def test_compute_future_log_return() -> None:
    prices = [100.0, 101.0, 102.0]
    expected = math.log(102.0 / 100.0)
    assert compute_future_return(prices, 0, 2) == pytest.approx(expected)


def test_compute_future_simple_return() -> None:
    prices = [100.0, 101.0, 105.0]
    assert compute_future_return(prices, 0, 2, log_return=False) == pytest.approx(0.05)


def test_compute_future_return_insufficient_horizon_raises() -> None:
    with pytest.raises(IndexError, match="insufficient future data"):
        compute_future_return([100.0, 101.0], 0, 2)


def test_compute_future_return_invalid_prices_raise() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        compute_future_return([100.0, 0.0, 101.0], 0, 1)
    with pytest.raises(ValueError, match="finite"):
        compute_future_return([100.0, math.inf], 0, 1)


def test_classify_direction_up_down_stationary() -> None:
    assert classify_direction(0.02, 0.01) == "up"
    assert classify_direction(-0.02, 0.01) == "down"
    assert classify_direction(0.005, 0.01) == "stationary"


def test_direction_threshold_validation() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        classify_direction(0.0, -0.1)
    with pytest.raises(ValueError, match="non-negative"):
        classify_direction(0.0, 0.1, down_threshold=-0.1)


def test_compute_direction_labels() -> None:
    labels = compute_direction_labels(
        [100.0, 101.0, 100.0],
        horizon=1,
        threshold=0.0,
        log_return=False,
    )
    assert labels == ["up", "down"]


def test_return_quantile_labels() -> None:
    labels = compute_return_quantile_labels([0.0, 1.0, 2.0, 3.0, 4.0])
    assert labels == [0, 1, 2, 3, 4]


def test_invalid_quantiles_raise() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_return_quantile_labels([0.0, 1.0], quantiles=(0.5, 0.5))
    with pytest.raises(ValueError, match="between 0 and 1"):
        compute_return_quantile_labels([0.0, 1.0], quantiles=(0.0,))


def test_future_returns_missing_drop() -> None:
    returns = compute_future_returns(
        [100.0, 101.0, 102.0, 103.0],
        horizon=2,
        log_return=False,
        missing="drop",
    )
    assert len(returns) == 2
    assert returns[0] == pytest.approx(0.02)


def test_future_returns_missing_none() -> None:
    returns = compute_future_returns(
        [100.0, 101.0, 102.0, 103.0],
        horizon=2,
        log_return=False,
        missing="none",
    )
    assert len(returns) == 4
    assert returns[:2] == pytest.approx([0.02, 2.0 / 101.0])
    assert returns[2:] == [None, None]
