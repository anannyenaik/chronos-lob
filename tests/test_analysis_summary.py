"""Tests for ``chronoslob.analysis.summary``."""

from __future__ import annotations

import math

import pytest

from chronoslob.analysis.summary import (
    EXECUTION_METRIC_NAMES,
    FORBIDDEN_COMBINED_FIELDS,
    METRIC_DIRECTIONS,
    PREDICTIVE_METRIC_NAMES,
    SUPPORTED_METRIC_NAMES,
    AnalysisRecord,
    aggregate_metric,
    aggregate_records,
    format_summary_table,
    run_robustness_analysis_smoke,
    summarise_records,
)


def _record(
    *,
    model: str = "model-A",
    symbol: str = "SYMBOL_A",
    metric_name: str = "accuracy",
    metric_value: float = 0.5,
    regime: str = "low",
    ablation: str = "baseline",
    is_synthetic: bool = True,
) -> AnalysisRecord:
    return AnalysisRecord(
        experiment_id="synthetic-id",
        model_name=model,
        dataset_name="synthetic-dataset",
        symbol=symbol,
        train_scope=symbol,
        eval_scope=symbol,
        regime=regime,
        ablation=ablation,
        sensitivity_parameter="",
        sensitivity_value=None,
        metric_name=metric_name,
        metric_value=metric_value,
        metric_direction=METRIC_DIRECTIONS[metric_name],
        is_synthetic=is_synthetic,
    )


def test_aggregate_metric_mean_count_min_max() -> None:
    aggregated = aggregate_metric([0.0, 1.0, 2.0])
    assert aggregated["count"] == 3
    assert math.isclose(float(aggregated["mean"]), 1.0, rel_tol=1e-9)
    assert math.isclose(float(aggregated["minimum"]), 0.0, rel_tol=1e-9)
    assert math.isclose(float(aggregated["maximum"]), 2.0, rel_tol=1e-9)


def test_aggregate_metric_handles_no_finite_values() -> None:
    aggregated = aggregate_metric([float("nan"), float("inf"), float("-inf")])
    assert aggregated["count"] == 0
    assert aggregated["mean"] is None


def test_aggregate_records_groups_by_model_and_symbol() -> None:
    records = [
        _record(model="A", symbol="X", metric_value=0.5),
        _record(model="A", symbol="X", metric_value=0.7),
        _record(model="A", symbol="Y", metric_value=0.4),
        _record(model="B", symbol="X", metric_value=0.6),
    ]
    summary = aggregate_records(
        records, metric_name="accuracy", group_by=("model_name", "symbol")
    )
    rows_by_key = {
        (row["model_name"], row["symbol"]): row for row in summary.rows
    }
    assert math.isclose(float(rows_by_key[("A", "X")]["mean"]), 0.6, rel_tol=1e-9)
    assert rows_by_key[("A", "X")]["count"] == 2
    assert rows_by_key[("A", "Y")]["count"] == 1
    assert rows_by_key[("B", "X")]["count"] == 1


def test_summarise_records_returns_one_summary_per_metric_name() -> None:
    records = [
        _record(metric_name="accuracy", metric_value=0.5),
        _record(metric_name="nll", metric_value=1.1),
    ]
    summaries = summarise_records(records, group_by=("model_name",))
    metric_names = sorted(summary.metric_name for summary in summaries)
    assert metric_names == ["accuracy", "nll"]


def test_format_summary_table_is_stable_and_deterministic() -> None:
    records = [
        _record(symbol="X", metric_value=0.5),
        _record(symbol="Y", metric_value=0.4),
    ]
    summary = aggregate_records(
        records, metric_name="accuracy", group_by=("symbol",)
    )
    text_one = format_summary_table(summary)
    text_two = format_summary_table(summary)
    assert text_one == text_two
    assert "metric=accuracy" in text_one
    assert "X" in text_one
    assert "Y" in text_one


def test_smoke_runner_returns_synthetic_only_payload() -> None:
    payload = run_robustness_analysis_smoke(n_records=12, seed=42)
    assert payload["is_synthetic"] is True
    assert payload["warning"].startswith("Synthetic analysis plumbing")
    assert payload["n_records"] == 12
    assert payload["regime_summary_counts"]
    assert payload["transfer_matrix"]["shape"][0] >= 1


def test_smoke_runner_deterministic_for_same_seed() -> None:
    first = run_robustness_analysis_smoke(n_records=10, seed=7)
    second = run_robustness_analysis_smoke(n_records=10, seed=7)
    assert first["transfer_matrix"]["values"] == second["transfer_matrix"]["values"]
    assert first["ablation_summary"] == second["ablation_summary"]


def test_smoke_payload_does_not_contain_combined_score_fields() -> None:
    payload = run_robustness_analysis_smoke(n_records=12, seed=42)
    for forbidden in FORBIDDEN_COMBINED_FIELDS:
        assert forbidden not in payload
    for example in payload["example_summary_rows"]:
        for forbidden in FORBIDDEN_COMBINED_FIELDS:
            assert forbidden not in example["row"]


def test_predictive_and_execution_metrics_are_separated() -> None:
    overlap = set(PREDICTIVE_METRIC_NAMES) & set(EXECUTION_METRIC_NAMES)
    assert overlap == set()
    assert set(PREDICTIVE_METRIC_NAMES) | set(EXECUTION_METRIC_NAMES) <= set(
        SUPPORTED_METRIC_NAMES
    )


def test_invalid_metric_name_fails_clearly() -> None:
    with pytest.raises(ValueError):
        aggregate_records([], metric_name="not_a_metric", group_by=("model_name",))


def test_analysis_record_rejects_forbidden_combined_score_mapping() -> None:
    with pytest.raises(ValueError):
        aggregate_records(
            [
                {
                    "experiment_id": "x",
                    "model_name": "m",
                    "dataset_name": "d",
                    "symbol": "S",
                    "train_scope": "S",
                    "eval_scope": "S",
                    "regime": "low",
                    "ablation": "baseline",
                    "sensitivity_parameter": "",
                    "sensitivity_value": None,
                    "metric_name": "accuracy",
                    "metric_value": 0.5,
                    "metric_direction": "higher_is_better",
                    "is_synthetic": True,
                    "combined_score": 0.9,
                }
            ],
            metric_name="accuracy",
            group_by=("model_name",),
        )


def test_metric_direction_must_match_metric_name() -> None:
    with pytest.raises(ValueError):
        AnalysisRecord(
            experiment_id="x",
            model_name="m",
            dataset_name="d",
            symbol="S",
            train_scope="S",
            eval_scope="S",
            regime="low",
            ablation="baseline",
            sensitivity_parameter="",
            sensitivity_value=None,
            metric_name="accuracy",
            metric_value=0.5,
            metric_direction="lower_is_better",
            is_synthetic=True,
        )
