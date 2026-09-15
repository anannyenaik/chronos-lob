"""Tests for future volatility labels."""

from __future__ import annotations

import math

import pytest

from chronoslob.labels.volatility import (
    classify_volatility_labels,
    compute_future_realised_volatility,
    compute_future_volatility_series,
)


def test_future_realised_volatility_formula() -> None:
    prices = [100.0, 101.0, 103.0]
    r1 = math.log(101.0 / 100.0)
    r2 = math.log(103.0 / 101.0)
    expected = math.sqrt(r1 * r1 + r2 * r2)
    assert compute_future_realised_volatility(prices, 0, 2) == pytest.approx(expected)


def test_future_volatility_series() -> None:
    prices = [100.0, 101.0, 102.0, 103.0]
    series = compute_future_volatility_series(prices, horizon=2)
    assert len(series) == 2
    assert series[0] == pytest.approx(
        compute_future_realised_volatility(prices, 0, 2)
    )


def test_future_volatility_series_missing_none() -> None:
    series = compute_future_volatility_series([100.0, 101.0, 102.0], 1, missing="none")
    assert len(series) == 3
    assert series[-1] is None


def test_future_realised_volatility_insufficient_horizon_raises() -> None:
    with pytest.raises(IndexError, match="insufficient future data"):
        compute_future_realised_volatility([100.0, 101.0], 0, 2)


def test_future_realised_volatility_invalid_prices_raise() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        compute_future_realised_volatility([100.0, -1.0], 0, 1)


def test_volatility_regime_classification() -> None:
    labels = classify_volatility_labels([0.01, 0.05, 0.1], 0.02, 0.08)
    assert labels == ["low_volatility", "medium_volatility", "high_volatility"]


def test_volatility_threshold_validation() -> None:
    with pytest.raises(ValueError, match="<="):
        classify_volatility_labels([0.01], 0.2, 0.1)
    with pytest.raises(ValueError, match="non-negative"):
        classify_volatility_labels([0.01], -0.1, 0.1)
