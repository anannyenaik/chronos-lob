"""Tests for the synthetic calibration smoke experiment."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from chronoslob.training.calibration import (  # noqa: E402
    run_calibration_smoke,
    summarise_multitask_calibration,
)


def test_synthetic_calibration_smoke_runs() -> None:
    payload = run_calibration_smoke(
        n_examples=40,
        num_classes=3,
        seed=7,
        ece_bins=5,
    )

    assert payload["n_examples"] == 40
    assert payload["num_classes"] == 3
    assert payload["calibration_examples"] == 20
    assert payload["evaluation_examples"] == 20
    assert payload["write_outputs"] is False


def test_payload_marks_outputs_as_synthetic_plumbing_only() -> None:
    payload = run_calibration_smoke(n_examples=40, num_classes=3, seed=7)

    assert payload["synthetic_plumbing_only"] is True
    assert "synthetic" in str(payload["notes"]).lower()
    assert "plumbing" in str(payload["notes"]).lower()
    assert "split_discipline" in payload


def test_pre_and_post_calibration_metrics_are_present() -> None:
    payload = run_calibration_smoke(n_examples=40, num_classes=3, seed=7)

    for key in ("pre_calibration", "post_calibration"):
        metrics = payload[key]
        assert isinstance(metrics, dict)
        for metric_name in ("nll", "ece", "brier_score", "accuracy"):
            assert metric_name in metrics
            assert metrics[metric_name] is not None


def test_fitted_temperature_is_reported() -> None:
    payload = run_calibration_smoke(n_examples=40, num_classes=3, seed=7)

    assert "fitted_temperature" in payload
    assert payload["fitted_temperature"] > 0.0
    assert payload["temperature_state"]["temperature"] == pytest.approx(
        payload["fitted_temperature"]
    )


def test_confidence_filtering_results_are_present() -> None:
    payload = run_calibration_smoke(n_examples=40, num_classes=3, seed=7)

    filtering = payload["confidence_filtering"]
    assert isinstance(filtering, dict)
    assert filtering["n_total"] == payload["evaluation_examples"]
    assert filtering["buckets"]
    assert {"threshold", "coverage", "abstention_rate", "n_covered", "n_total"} <= (
        set(filtering["buckets"][0])
    )
    assert payload["abstention_curve"]


def test_no_market_performance_claim_fields_are_present() -> None:
    payload = run_calibration_smoke(n_examples=40, num_classes=3, seed=7)
    forbidden = {
        "pnl",
        "profit",
        "sharpe",
        "drawdown",
        "trade_count",
        "backtest",
        "execution_simulation",
        "live_metrics",
    }

    assert forbidden.isdisjoint(payload.keys())


def test_no_files_or_checkpoints_are_written(tmp_path: Path) -> None:
    before = sorted(tmp_path.iterdir())

    payload = run_calibration_smoke(n_examples=40, num_classes=3, seed=7)

    after = sorted(tmp_path.iterdir())
    assert before == after
    assert payload["write_outputs"] is False
    assert payload["checkpoints_written"] is False


def test_calibration_smoke_is_deterministic_under_fixed_seed() -> None:
    first = run_calibration_smoke(n_examples=40, num_classes=3, seed=7)
    second = run_calibration_smoke(n_examples=40, num_classes=3, seed=7)

    assert first == second


def test_multitask_calibration_summary_reports_per_task_metrics() -> None:
    logits = {
        "direction": torch.tensor([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0]]),
        "spread_widening": torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
    }
    targets = {
        "direction": torch.tensor([0, 1], dtype=torch.long),
        "spread_widening": torch.tensor([0, 1], dtype=torch.long),
    }

    summary = summarise_multitask_calibration(logits, targets)

    assert summary["task_count"] == 2
    assert "direction" in summary["task_summaries"]
    assert "spread_widening" in summary["task_summaries"]
    assert summary["averaging"] == "none; per-task summaries are reported separately"
