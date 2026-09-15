"""Tests for chronoslob.features.volatility."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import pytest

from chronoslob.features.volatility import (
    compute_event_intensity,
    compute_log_returns,
    compute_realised_volatility,
    compute_rolling_event_intensity,
    compute_rolling_realised_volatility,
)

T0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Log returns
# ---------------------------------------------------------------------------


def test_log_returns_length_is_n_minus_one() -> None:
    prices = [100.0, 101.0, 102.0]
    returns = compute_log_returns(prices)
    assert len(returns) == 2


def test_log_returns_values() -> None:
    prices = [100.0, 101.0]
    returns = compute_log_returns(prices)
    assert returns[0] == pytest.approx(math.log(101.0 / 100.0))


def test_log_returns_rejects_non_positive_prices() -> None:
    with pytest.raises(ValueError):
        compute_log_returns([100.0, 0.0])
    with pytest.raises(ValueError):
        compute_log_returns([100.0, -1.0])


def test_log_returns_handles_short_input() -> None:
    assert compute_log_returns([]) == []
    assert compute_log_returns([100.0]) == []


# ---------------------------------------------------------------------------
# Realised volatility
# ---------------------------------------------------------------------------


def test_realised_volatility_equals_rss_of_log_returns() -> None:
    prices = [100.0, 101.0, 99.0]
    returns = compute_log_returns(prices)
    expected = math.sqrt(sum(r * r for r in returns))
    assert compute_realised_volatility(prices) == pytest.approx(expected)


def test_realised_volatility_window_truncates_history() -> None:
    prices = [100.0, 200.0, 100.0, 101.0]
    full = compute_realised_volatility(prices)
    windowed = compute_realised_volatility(prices, window=2)
    expected_windowed = math.sqrt(math.log(101.0 / 100.0) ** 2)
    assert windowed == pytest.approx(expected_windowed)
    assert full != windowed


def test_realised_volatility_requires_two_prices() -> None:
    with pytest.raises(ValueError):
        compute_realised_volatility([100.0])


# ---------------------------------------------------------------------------
# Rolling realised volatility
# ---------------------------------------------------------------------------


def test_rolling_realised_volatility_is_past_only() -> None:
    prices = [100.0, 101.0, 102.0, 103.0]
    series = compute_rolling_realised_volatility(prices, window=3)
    assert len(series) == 4
    # At t=0 we have only one price, so result must be NaN by default.
    assert math.isnan(series[0])
    # Manually compute the expected value at the last position using only
    # past prices to confirm no future leakage.
    expected_last = math.sqrt(
        math.log(102.0 / 101.0) ** 2 + math.log(103.0 / 102.0) ** 2
    )
    assert series[-1] == pytest.approx(expected_last)


def test_rolling_realised_volatility_position_uses_only_history() -> None:
    prices = [100.0, 200.0, 100.0]  # noisy past
    series_a = compute_rolling_realised_volatility(prices, window=2)
    # At index 1 only prices[0..1] should be used; not prices[2].
    expected_at_1 = math.sqrt(math.log(200.0 / 100.0) ** 2)
    assert series_a[1] == pytest.approx(expected_at_1)


def test_rolling_realised_volatility_rejects_invalid_window() -> None:
    with pytest.raises(ValueError):
        compute_rolling_realised_volatility([100.0, 101.0], window=1)


def test_rolling_realised_volatility_rejects_non_positive_prices() -> None:
    with pytest.raises(ValueError):
        compute_rolling_realised_volatility([100.0, -1.0, 102.0], window=2)


# ---------------------------------------------------------------------------
# Event intensity
# ---------------------------------------------------------------------------


def test_event_intensity_trailing_window() -> None:
    timestamps = [
        T0,
        T0 + timedelta(seconds=10),
        T0 + timedelta(seconds=20),
        T0 + timedelta(seconds=55),
        T0 + timedelta(seconds=60),
    ]
    intensity = compute_event_intensity(timestamps, window_seconds=30.0)
    # The trailing 30 seconds ending at 60 includes timestamps at 55 and 60: 2 events.
    assert intensity == pytest.approx(2.0 / 30.0)


def test_event_intensity_rejects_naive_timestamps() -> None:
    naive = datetime(2024, 1, 1, 12, 0, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        compute_event_intensity([naive], window_seconds=30.0)


def test_event_intensity_rejects_non_monotonic_timestamps() -> None:
    timestamps = [T0 + timedelta(seconds=10), T0]
    with pytest.raises(ValueError, match="non-decreasing"):
        compute_event_intensity(timestamps, window_seconds=30.0)


def test_event_intensity_rejects_invalid_window() -> None:
    with pytest.raises(ValueError):
        compute_event_intensity([T0], window_seconds=0.0)
    with pytest.raises(ValueError):
        compute_event_intensity([T0], window_seconds=-1.0)


def test_event_intensity_returns_zero_for_empty_input() -> None:
    assert compute_event_intensity([], window_seconds=30.0) == 0.0


# ---------------------------------------------------------------------------
# Rolling event intensity
# ---------------------------------------------------------------------------


def test_rolling_event_intensity_grows_with_dense_history() -> None:
    timestamps = [T0 + timedelta(seconds=i) for i in range(5)]
    series = compute_rolling_event_intensity(timestamps, window_seconds=10.0)
    assert len(series) == 5
    # Strictly non-decreasing as more events fall inside the trailing window.
    for previous, current in pairwise(series):
        assert current >= previous


def test_rolling_event_intensity_past_only_one_point_intensity() -> None:
    timestamps = [T0]
    series = compute_rolling_event_intensity(timestamps, window_seconds=10.0)
    assert series == [pytest.approx(1.0 / 10.0)]
