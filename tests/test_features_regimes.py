"""Tests for chronoslob.features.regimes."""

from __future__ import annotations

import pandas as pd
import pytest

from chronoslob.features.regimes import (
    RegimeThresholds,
    classify_imbalance_regime,
    classify_liquidity_regime,
    classify_spread_regime,
    classify_volatility_regime,
    compute_regime_thresholds_from_frame,
)

# ---------------------------------------------------------------------------
# Thresholds dataclass validation
# ---------------------------------------------------------------------------


def test_regime_thresholds_default_values_pass() -> None:
    thresholds = RegimeThresholds()
    assert 0.0 < thresholds.wide_spread_quantile < 1.0
    assert 0.0 < thresholds.high_volatility_quantile < 1.0
    assert 0.0 < thresholds.low_liquidity_quantile < 1.0
    assert 0.0 < thresholds.imbalance_abs_threshold < 1.0


def test_regime_thresholds_reject_out_of_range_quantiles() -> None:
    with pytest.raises(ValueError):
        RegimeThresholds(wide_spread_quantile=1.5)
    with pytest.raises(ValueError):
        RegimeThresholds(high_volatility_quantile=0.0)
    with pytest.raises(ValueError):
        RegimeThresholds(low_liquidity_quantile=1.0)


def test_regime_thresholds_reject_out_of_range_imbalance_threshold() -> None:
    with pytest.raises(ValueError):
        RegimeThresholds(imbalance_abs_threshold=0.0)
    with pytest.raises(ValueError):
        RegimeThresholds(imbalance_abs_threshold=1.5)


# ---------------------------------------------------------------------------
# Classify spread / volatility / liquidity / imbalance regimes
# ---------------------------------------------------------------------------


def test_classify_spread_regime() -> None:
    assert classify_spread_regime(0.5, threshold=1.0) == "normal_spread"
    assert classify_spread_regime(1.0, threshold=1.0) == "wide_spread"
    assert classify_spread_regime(1.5, threshold=1.0) == "wide_spread"


def test_classify_volatility_regime() -> None:
    assert (
        classify_volatility_regime(0.0, low_threshold=0.1, high_threshold=0.5)
        == "low_volatility"
    )
    assert (
        classify_volatility_regime(0.2, low_threshold=0.1, high_threshold=0.5)
        == "medium_volatility"
    )
    assert (
        classify_volatility_regime(0.6, low_threshold=0.1, high_threshold=0.5)
        == "high_volatility"
    )


def test_classify_volatility_regime_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValueError):
        classify_volatility_regime(0.2, low_threshold=0.5, high_threshold=0.1)


def test_classify_liquidity_regime() -> None:
    assert classify_liquidity_regime(50.0, low_threshold=100.0) == "low_liquidity"
    assert classify_liquidity_regime(150.0, low_threshold=100.0) == "normal_liquidity"


def test_classify_imbalance_regime() -> None:
    assert classify_imbalance_regime(0.8) == "bid_heavy"
    assert classify_imbalance_regime(-0.8) == "ask_heavy"
    assert classify_imbalance_regime(0.0) == "balanced"
    assert classify_imbalance_regime(0.5) == "balanced"


def test_classify_imbalance_regime_rejects_threshold_out_of_range() -> None:
    with pytest.raises(ValueError):
        classify_imbalance_regime(0.5, threshold=1.5)
    with pytest.raises(ValueError):
        classify_imbalance_regime(0.5, threshold=0.0)


# ---------------------------------------------------------------------------
# compute_regime_thresholds_from_frame
# ---------------------------------------------------------------------------


def test_compute_regime_thresholds_with_all_columns() -> None:
    frame = pd.DataFrame(
        {
            "spread": [0.1, 0.2, 0.3, 0.4, 0.5],
            "realised_volatility": [0.01, 0.02, 0.03, 0.04, 0.05],
            "total_depth": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    thresholds = compute_regime_thresholds_from_frame(frame)
    assert "wide_spread_threshold" in thresholds
    assert "low_volatility_threshold" in thresholds
    assert "high_volatility_threshold" in thresholds
    assert "low_liquidity_threshold" in thresholds


def test_compute_regime_thresholds_with_partial_columns() -> None:
    frame = pd.DataFrame({"spread": [0.1, 0.2, 0.3]})
    thresholds = compute_regime_thresholds_from_frame(frame)
    assert "wide_spread_threshold" in thresholds
    assert "low_volatility_threshold" not in thresholds
    assert "low_liquidity_threshold" not in thresholds


def test_compute_regime_thresholds_falls_back_to_top_depth_sum() -> None:
    frame = pd.DataFrame(
        {
            "bid_depth_1": [1.0, 2.0, 3.0],
            "ask_depth_1": [1.0, 2.0, 3.0],
        }
    )
    thresholds = compute_regime_thresholds_from_frame(frame)
    assert "low_liquidity_threshold" in thresholds


def test_compute_regime_thresholds_empty_frame_skips_outputs() -> None:
    frame = pd.DataFrame({"spread": []})
    thresholds = compute_regime_thresholds_from_frame(frame)
    assert thresholds == {}


def test_compute_regime_thresholds_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError):
        compute_regime_thresholds_from_frame([1, 2, 3])  # type: ignore[arg-type]
