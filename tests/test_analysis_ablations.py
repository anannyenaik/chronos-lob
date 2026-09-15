"""Tests for ``chronoslob.analysis.ablations``."""

from __future__ import annotations

import math

import pytest

from chronoslob.analysis.ablations import (
    AblationResult,
    AblationSpec,
    compare_against_baseline,
    rank_ablations,
    summarise_ablation_table,
)


def _records_higher() -> list[AblationResult]:
    return [
        AblationResult(
            ablation_name="baseline",
            metric_name="accuracy",
            metric_value=0.50,
            metric_direction="higher_is_better",
            is_synthetic=True,
        ),
        AblationResult(
            ablation_name="no_calibration",
            metric_name="accuracy",
            metric_value=0.55,
            metric_direction="higher_is_better",
            is_synthetic=True,
        ),
        AblationResult(
            ablation_name="no_confidence_filtering",
            metric_name="accuracy",
            metric_value=0.45,
            metric_direction="higher_is_better",
            is_synthetic=True,
        ),
    ]


def _records_lower() -> list[AblationResult]:
    return [
        AblationResult(
            ablation_name="baseline",
            metric_name="nll",
            metric_value=1.0,
            metric_direction="lower_is_better",
            is_synthetic=False,
        ),
        AblationResult(
            ablation_name="no_calibration",
            metric_name="nll",
            metric_value=1.2,
            metric_direction="lower_is_better",
            is_synthetic=False,
        ),
        AblationResult(
            ablation_name="aggressive_only",
            metric_name="nll",
            metric_value=0.8,
            metric_direction="lower_is_better",
            is_synthetic=False,
        ),
    ]


def test_baseline_comparison_absolute_delta_higher_is_better() -> None:
    comparisons = compare_against_baseline(
        _records_higher(), baseline_name="baseline", metric_name="accuracy"
    )
    by_name = {comparison.ablation_name: comparison for comparison in comparisons}
    assert math.isclose(by_name["no_calibration"].absolute_delta, 0.05, abs_tol=1e-9)
    assert by_name["no_calibration"].is_improvement is True
    assert by_name["no_confidence_filtering"].is_improvement is False


def test_relative_delta_safe_when_baseline_is_zero() -> None:
    records = [
        AblationResult(
            ablation_name="baseline",
            metric_name="accuracy",
            metric_value=0.0,
            metric_direction="higher_is_better",
        ),
        AblationResult(
            ablation_name="variant",
            metric_name="accuracy",
            metric_value=0.1,
            metric_direction="higher_is_better",
        ),
    ]
    comparisons = compare_against_baseline(
        records, baseline_name="baseline", metric_name="accuracy"
    )
    assert comparisons[0].relative_delta is None


def test_lower_is_better_direction_handled() -> None:
    comparisons = compare_against_baseline(
        _records_lower(), baseline_name="baseline", metric_name="nll"
    )
    by_name = {comparison.ablation_name: comparison for comparison in comparisons}
    assert by_name["aggressive_only"].is_improvement is True
    assert by_name["no_calibration"].is_improvement is False


def test_rank_ablations_orders_by_improvement_first() -> None:
    comparisons = compare_against_baseline(
        _records_higher(), baseline_name="baseline", metric_name="accuracy"
    )
    ranked = rank_ablations(comparisons)
    assert ranked[0].ablation_name == "no_calibration"
    assert ranked[-1].ablation_name == "no_confidence_filtering"


def test_missing_baseline_raises_clearly() -> None:
    records = [
        AblationResult(
            ablation_name="variant",
            metric_name="accuracy",
            metric_value=0.5,
            metric_direction="higher_is_better",
        )
    ]
    with pytest.raises(KeyError):
        compare_against_baseline(
            records, baseline_name="baseline", metric_name="accuracy"
        )


def test_metric_direction_mismatch_raises() -> None:
    records = [
        AblationResult(
            ablation_name="baseline",
            metric_name="accuracy",
            metric_value=0.5,
            metric_direction="higher_is_better",
        ),
        AblationResult(
            ablation_name="variant",
            metric_name="accuracy",
            metric_value=0.4,
            metric_direction="lower_is_better",
        ),
    ]
    with pytest.raises(ValueError):
        compare_against_baseline(
            records, baseline_name="baseline", metric_name="accuracy"
        )


def test_synthetic_flag_preserved_in_comparisons() -> None:
    comparisons = compare_against_baseline(
        _records_higher(), baseline_name="baseline", metric_name="accuracy"
    )
    assert all(comparison.is_synthetic for comparison in comparisons)


def test_synthetic_flag_false_if_any_input_is_not_synthetic() -> None:
    comparisons = compare_against_baseline(
        _records_lower(), baseline_name="baseline", metric_name="nll"
    )
    assert all(comparison.is_synthetic is False for comparison in comparisons)


def test_summarise_ablation_table_counts_improvements() -> None:
    comparisons = compare_against_baseline(
        _records_higher(), baseline_name="baseline", metric_name="accuracy"
    )
    table = summarise_ablation_table(comparisons)
    assert table["metric_name"] == "accuracy"
    assert table["n_comparisons"] == 2
    assert table["n_improvements"] + table["n_regressions"] == 2


def test_ablation_spec_category_validation() -> None:
    with pytest.raises(ValueError):
        AblationSpec(name="bad", category="not_a_category")
    AblationSpec(name="no_ssl_pretraining", category="objective")
