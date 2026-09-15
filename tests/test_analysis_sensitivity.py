"""Tests for ``chronoslob.analysis.sensitivity``."""

from __future__ import annotations

import math

import pytest

from chronoslob.analysis.sensitivity import (
    SensitivityPoint,
    build_sensitivity_curve,
    compare_sensitivity_curves,
    summarise_sensitivity_curve,
)


def _confidence_points() -> list[SensitivityPoint]:
    return [
        SensitivityPoint(
            parameter_name="confidence_threshold",
            parameter_value=0.5,
            metric_name="accuracy",
            metric_value=0.55,
            is_synthetic=True,
        ),
        SensitivityPoint(
            parameter_name="confidence_threshold",
            parameter_value=0.7,
            metric_name="accuracy",
            metric_value=0.62,
            is_synthetic=True,
        ),
        SensitivityPoint(
            parameter_name="confidence_threshold",
            parameter_value=0.6,
            metric_name="accuracy",
            metric_value=0.58,
            is_synthetic=True,
        ),
        SensitivityPoint(
            parameter_name="confidence_threshold",
            parameter_value=0.8,
            metric_name="accuracy",
            metric_value=None,
            is_synthetic=True,
        ),
    ]


def _latency_points() -> list[SensitivityPoint]:
    return [
        SensitivityPoint(
            parameter_name="latency_steps",
            parameter_value=0.0,
            metric_name="simulated_net_pnl",
            metric_value=0.4,
            is_synthetic=True,
        ),
        SensitivityPoint(
            parameter_name="latency_steps",
            parameter_value=2.0,
            metric_name="simulated_net_pnl",
            metric_value=0.2,
            is_synthetic=True,
        ),
        SensitivityPoint(
            parameter_name="latency_steps",
            parameter_value=5.0,
            metric_name="simulated_net_pnl",
            metric_value=-0.1,
            is_synthetic=True,
        ),
    ]


def test_sensitivity_curve_orders_by_parameter_value() -> None:
    curve = build_sensitivity_curve(
        _confidence_points(),
        parameter_name="confidence_threshold",
        metric_name="accuracy",
        metric_direction="higher_is_better",
    )
    values = curve.parameter_values()
    assert list(values) == sorted(values)


def test_threshold_curve_best_point_higher_is_better() -> None:
    curve = build_sensitivity_curve(
        _confidence_points(),
        parameter_name="confidence_threshold",
        metric_name="accuracy",
        metric_direction="higher_is_better",
    )
    summary = summarise_sensitivity_curve(curve)
    assert summary["n_valid_points"] == 3
    assert math.isclose(float(summary["best_metric_value"]), 0.62, rel_tol=1e-9)
    assert math.isclose(float(summary["best_parameter_value"]), 0.7, rel_tol=1e-9)


def test_latency_curve_best_point_lower_steps_for_pnl() -> None:
    curve = build_sensitivity_curve(
        _latency_points(),
        parameter_name="latency_steps",
        metric_name="simulated_net_pnl",
        metric_direction="higher_is_better",
    )
    summary = summarise_sensitivity_curve(curve)
    assert math.isclose(float(summary["best_metric_value"]), 0.4, rel_tol=1e-9)
    assert math.isclose(float(summary["best_parameter_value"]), 0.0, rel_tol=1e-9)


def test_lower_is_better_selects_minimum() -> None:
    records = [
        SensitivityPoint(
            parameter_name="fee_bps",
            parameter_value=1.0,
            metric_name="total_cost",
            metric_value=0.2,
            is_synthetic=True,
        ),
        SensitivityPoint(
            parameter_name="fee_bps",
            parameter_value=5.0,
            metric_name="total_cost",
            metric_value=0.4,
            is_synthetic=True,
        ),
    ]
    curve = build_sensitivity_curve(
        records,
        parameter_name="fee_bps",
        metric_name="total_cost",
        metric_direction="lower_is_better",
    )
    summary = summarise_sensitivity_curve(curve)
    assert math.isclose(float(summary["best_metric_value"]), 0.2, rel_tol=1e-9)


def test_missing_metric_values_produce_zero_valid_points() -> None:
    records = [
        SensitivityPoint(
            parameter_name="confidence_threshold",
            parameter_value=0.5,
            metric_name="accuracy",
            metric_value=None,
            is_synthetic=True,
        ),
    ]
    curve = build_sensitivity_curve(
        records,
        parameter_name="confidence_threshold",
        metric_name="accuracy",
        metric_direction="higher_is_better",
    )
    summary = summarise_sensitivity_curve(curve)
    assert summary["n_valid_points"] == 0
    assert summary["best_parameter_value"] is None
    assert summary["best_metric_value"] is None


def test_duplicate_parameter_value_raises() -> None:
    records = [
        SensitivityPoint(
            parameter_name="confidence_threshold",
            parameter_value=0.5,
            metric_name="accuracy",
            metric_value=0.6,
        ),
        SensitivityPoint(
            parameter_name="confidence_threshold",
            parameter_value=0.5,
            metric_name="accuracy",
            metric_value=0.7,
        ),
    ]
    with pytest.raises(ValueError):
        build_sensitivity_curve(
            records,
            parameter_name="confidence_threshold",
            metric_name="accuracy",
            metric_direction="higher_is_better",
        )


def test_unknown_parameter_name_raises() -> None:
    with pytest.raises(ValueError):
        build_sensitivity_curve(
            [],
            parameter_name="not_a_parameter",
            metric_name="accuracy",
            metric_direction="higher_is_better",
        )


def test_invalid_metric_direction_raises() -> None:
    with pytest.raises(ValueError):
        build_sensitivity_curve(
            [],
            parameter_name="confidence_threshold",
            metric_name="accuracy",
            metric_direction="middle_is_better",  # type: ignore[arg-type]
        )


def test_compare_sensitivity_curves_picks_best() -> None:
    points_one = [
        SensitivityPoint(
            parameter_name="confidence_threshold",
            parameter_value=0.5,
            metric_name="accuracy",
            metric_value=0.55,
            is_synthetic=True,
        )
    ]
    points_two = [
        SensitivityPoint(
            parameter_name="confidence_threshold",
            parameter_value=0.5,
            metric_name="accuracy",
            metric_value=0.65,
            is_synthetic=True,
        )
    ]
    curve_one = build_sensitivity_curve(
        points_one,
        parameter_name="confidence_threshold",
        metric_name="accuracy",
        metric_direction="higher_is_better",
    )
    curve_two = build_sensitivity_curve(
        points_two,
        parameter_name="confidence_threshold",
        metric_name="accuracy",
        metric_direction="higher_is_better",
    )
    comparison = compare_sensitivity_curves([curve_one, curve_two])
    assert comparison["best_curve_index"] == 1


def test_compare_sensitivity_curves_requires_shared_metric() -> None:
    curve_one = build_sensitivity_curve(
        [
            SensitivityPoint(
                parameter_name="confidence_threshold",
                parameter_value=0.5,
                metric_name="accuracy",
                metric_value=0.55,
            )
        ],
        parameter_name="confidence_threshold",
        metric_name="accuracy",
        metric_direction="higher_is_better",
    )
    curve_two = build_sensitivity_curve(
        [
            SensitivityPoint(
                parameter_name="confidence_threshold",
                parameter_value=0.5,
                metric_name="ece",
                metric_value=0.05,
            )
        ],
        parameter_name="confidence_threshold",
        metric_name="ece",
        metric_direction="lower_is_better",
    )
    with pytest.raises(ValueError):
        compare_sensitivity_curves([curve_one, curve_two])
