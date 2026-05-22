"""Tests for future spread labels."""

from __future__ import annotations

import pytest

from chronoslob.labels.spread import (
    compute_future_spread_change,
    compute_spread_widening_label,
    compute_spread_widening_labels,
)


def test_future_spread_change() -> None:
    assert compute_future_spread_change([1.0, 1.2, 1.5], 0, 2) == pytest.approx(0.5)


def test_spread_widening_true_false() -> None:
    spreads = [1.0, 1.2, 0.8]
    assert compute_spread_widening_label(spreads, 0, 1, threshold=0.1) is True
    assert compute_spread_widening_label(spreads, 0, 2, threshold=0.0) is False


def test_negative_spread_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        compute_future_spread_change([1.0, -0.1], 0, 1)


def test_spread_widening_missing_drop() -> None:
    labels = compute_spread_widening_labels([1.0, 1.1, 0.9], horizon=1)
    assert labels == [True, False]


def test_spread_widening_missing_none() -> None:
    labels = compute_spread_widening_labels([1.0, 1.1, 0.9], horizon=1, missing="none")
    assert labels == [True, False, None]


def test_spread_widening_invalid_threshold_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        compute_spread_widening_label([1.0, 1.2], 0, 1, threshold=-0.1)
