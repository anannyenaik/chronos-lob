"""Tests for ``chronoslob.analysis.regimes``."""

from __future__ import annotations

import math

import pytest

from chronoslob.analysis.regimes import (
    DEFAULT_CONFIDENCE_BUCKET_EDGES,
    DEFAULT_VOLATILITY_THRESHOLDS,
    SUPPORTED_REGIME_KINDS,
    UNKNOWN_REGIME_LABEL,
    RegimeDefinition,
    assign_confidence_bucket,
    assign_latency_regime,
    assign_liquidity_regime,
    assign_spread_regime,
    assign_volatility_regime,
    fit_regime_boundaries,
    summarise_by_regime,
)


def test_supported_regime_kinds_contains_required_kinds() -> None:
    assert set(SUPPORTED_REGIME_KINDS) >= {
        "volatility",
        "spread",
        "liquidity",
        "confidence",
        "latency",
    }


def test_explicit_regime_label_is_preferred() -> None:
    record = {"regime": "high", "volatility": 0.0001}
    assignment = assign_volatility_regime(record)
    assert assignment.label == "high"
    assert assignment.source == "explicit"


def test_volatility_threshold_assignment_low_medium_high() -> None:
    low = assign_volatility_regime({"volatility": 0.001})
    medium = assign_volatility_regime({"volatility": 0.01})
    high = assign_volatility_regime({"volatility": 0.05})
    assert low.label == "low"
    assert medium.label == "medium"
    assert high.label == "high"
    for assignment in (low, medium, high):
        assert assignment.source == "threshold"


def test_spread_threshold_assignment() -> None:
    tight = assign_spread_regime({"spread": 0.0001})
    normal = assign_spread_regime({"spread": 0.001})
    wide = assign_spread_regime({"spread": 0.01})
    assert tight.label == "tight"
    assert normal.label == "normal"
    assert wide.label == "wide"


def test_liquidity_threshold_assignment() -> None:
    thin = assign_liquidity_regime({"liquidity": 10.0})
    normal = assign_liquidity_regime({"liquidity": 100.0})
    deep = assign_liquidity_regime({"liquidity": 500.0})
    assert thin.label == "thin"
    assert normal.label == "normal"
    assert deep.label == "deep"


def test_confidence_bucket_assignment() -> None:
    assignments = [
        assign_confidence_bucket({"confidence": value})
        for value in (0.1, 0.6, 0.8, 0.95)
    ]
    labels = [assignment.label for assignment in assignments]
    assert labels == ["low", "medium", "high", "very_high"]


def test_latency_regime_assignment_zero_low_medium_high() -> None:
    assert assign_latency_regime({"latency_steps": 0}).label == "zero"
    assert assign_latency_regime({"latency_steps": 1}).label == "low"
    assert assign_latency_regime({"latency_steps": 2}).label == "medium"
    assert assign_latency_regime({"latency_steps": 5}).label == "high"


def test_missing_values_produce_unknown_regime() -> None:
    assignment = assign_volatility_regime({})
    assert assignment.label == UNKNOWN_REGIME_LABEL
    assert assignment.source == "default"
    nan_assignment = assign_volatility_regime({"volatility": float("nan")})
    assert nan_assignment.label == UNKNOWN_REGIME_LABEL


def test_summarise_by_regime_groups_correctly_and_is_deterministic() -> None:
    records = [
        {
            "regime": "low",
            "metric_name": "accuracy",
            "metric_value": 0.5,
            "is_synthetic": True,
        },
        {
            "regime": "low",
            "metric_name": "accuracy",
            "metric_value": 0.7,
            "is_synthetic": True,
        },
        {
            "regime": "high",
            "metric_name": "accuracy",
            "metric_value": 0.4,
            "is_synthetic": True,
        },
        {
            "regime": "low",
            "metric_name": "nll",
            "metric_value": 0.9,
            "is_synthetic": True,
        },
    ]
    summaries = summarise_by_regime(
        records,
        kind="volatility",
        metric_name="accuracy",
        assigner=lambda record: type(
            "Assignment",
            (),
            {
                "kind": "volatility",
                "label": record["regime"],
                "value": None,
                "source": "explicit",
            },
        )(),
    )
    labels = [summary.label for summary in summaries]
    assert labels == sorted(labels)
    means = {summary.label: summary.mean for summary in summaries}
    assert math.isclose(float(means["low"]), 0.6, rel_tol=1e-9)
    assert math.isclose(float(means["high"]), 0.4, rel_tol=1e-9)
    counts = {summary.label: summary.count for summary in summaries}
    assert counts["low"] == 2
    assert counts["high"] == 1


def test_summarise_by_regime_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        summarise_by_regime([], kind="unknown_kind", metric_name="accuracy")


def test_fit_regime_boundaries_requires_train_only_inputs() -> None:
    train_values = [float(i) for i in range(100)]
    boundaries = fit_regime_boundaries(train_values, n_bins=3, method="quantile")
    assert len(boundaries) == 2
    assert boundaries[0] < boundaries[1]


def test_fit_regime_boundaries_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        fit_regime_boundaries([1.0, 2.0, 3.0], method="kmeans")


def test_fit_regime_boundaries_rejects_no_finite_values() -> None:
    with pytest.raises(ValueError):
        fit_regime_boundaries([float("nan"), float("inf")])


def test_assignment_does_not_fit_thresholds_on_evaluation_data() -> None:
    record = {"volatility": 0.05}
    assignment = assign_volatility_regime(record)
    assert DEFAULT_VOLATILITY_THRESHOLDS[1] < 0.05
    assert assignment.label == "high"


def test_regime_definition_validates_thresholds_length() -> None:
    with pytest.raises(ValueError):
        RegimeDefinition(
            kind="volatility",
            labels=("low", "medium", "high"),
            thresholds=(0.001,),
        )


def test_confidence_bucket_edges_match_default() -> None:
    assert len(DEFAULT_CONFIDENCE_BUCKET_EDGES) == 3
