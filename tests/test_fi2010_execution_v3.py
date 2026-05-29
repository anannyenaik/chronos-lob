"""Tests for FI-2010 execution-aware proxy diagnostic v3."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from chronoslob.analysis.execution_v3 import build_fi2010_execution_v3
from chronoslob.analysis.fi2010_figures import build_fi2010_neural_figures
from chronoslob.experiments.final_report import build_final_empirical_report
from chronoslob.utils.paths import project_root


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prediction_rows(
    *,
    realised_returns: bool = False,
    market_context: bool = False,
    ambiguous: bool = False,
    regime: bool = False,
) -> pd.DataFrame:
    rows = pd.DataFrame(
        [
            {
                "row_id": 0,
                "y_true": 1,
                "y_pred": 1,
                "prob_up": 0.90,
                "prob_stationary": 0.05,
                "prob_down": 0.05,
                "confidence": 0.90,
            },
            {
                "row_id": 1,
                "y_true": 3,
                "y_pred": 1,
                "prob_up": 0.80,
                "prob_stationary": 0.10,
                "prob_down": 0.10,
                "confidence": 0.80,
            },
            {
                "row_id": 2,
                "y_true": 2,
                "y_pred": 2,
                "prob_up": 0.15,
                "prob_stationary": 0.70,
                "prob_down": 0.15,
                "confidence": 0.70,
            },
            {
                "row_id": 3,
                "y_true": 3,
                "y_pred": 3,
                "prob_up": 0.20,
                "prob_stationary": 0.20,
                "prob_down": 0.60,
                "confidence": 0.60,
            },
            {
                "row_id": 4,
                "y_true": 1,
                "y_pred": 3,
                "prob_up": 0.30,
                "prob_stationary": 0.25,
                "prob_down": 0.45,
                "confidence": 0.45,
            },
            {
                "row_id": 5,
                "y_true": 1,
                "y_pred": 1,
                "prob_up": 0.34,
                "prob_stationary": 0.33,
                "prob_down": 0.33,
                "confidence": 0.20,
            },
        ]
    )
    if ambiguous:
        rows = rows.drop(columns=["prob_up", "prob_stationary", "prob_down"]).assign(
            prob_0=[0.90, 0.80, 0.15, 0.20, 0.30, 0.34],
            prob_1=[0.05, 0.10, 0.70, 0.20, 0.25, 0.33],
            prob_2=[0.05, 0.10, 0.15, 0.60, 0.45, 0.33],
        )
    if realised_returns:
        rows["future_return"] = [0.01, -0.02, 0.0, -0.03, 0.01, 0.02]
    if market_context:
        rows["mid_price"] = [100.0] * len(rows)
        rows["spread"] = [0.02, 0.04, 0.01, 0.03, 0.05, 0.02]
        rows["bid_depth_1"] = [10, 4, 12, 8, 2, 10]
        rows["ask_depth_1"] = [9, 5, 11, 8, 3, 10]
        rows["imbalance"] = [0.1, 0.7, 0.0, -0.2, -0.8, 0.2]
    if regime:
        rows["volatility_regime"] = ["calm", "volatile", "calm", "volatile", "calm", "calm"]
    return rows


def _write_grid(
    base: Path,
    *,
    realised_returns: bool = False,
    market_context: bool = False,
    ambiguous: bool = False,
    smoke: bool = False,
    regime: bool = False,
) -> Path:
    grid = base / "grid"
    run_dir = grid / "runs" / "fold_1" / "horizon_10" / "seed_0" / "lookback_20" / "supervised"
    run_dir.mkdir(parents=True, exist_ok=True)
    _prediction_rows(
        realised_returns=realised_returns,
        market_context=market_context,
        ambiguous=ambiguous,
        regime=regime,
    ).to_csv(run_dir / "predictions.csv", index=False)
    _write_json(
        grid / "summary.json",
        {
            "execution_mode": "smoke" if smoke else "benchmark",
            "smoke_test": smoke,
            "completed_run_count": 1,
            "failed_run_count": 0,
        },
    )
    result_row = {
        "fold": 1,
        "horizon": 10,
        "seed": 0,
        "lookback": 20,
        "model_family": "matrix_transformer",
        "pretraining_objective": "none",
        "accuracy": 0.5,
        "macro_f1": 0.5,
        "mcc": 0.1,
        "ece": 0.2,
        "class_f1_up": 0.5,
        "class_f1_stationary": 0.4,
        "class_f1_down": 0.5,
        "prediction_file": run_dir.relative_to(grid).joinpath("predictions.csv").as_posix(),
        "status": "completed",
    }
    pd.DataFrame([result_row]).to_csv(grid / "results_summary.csv", index=False)
    pd.DataFrame([result_row]).to_csv(grid / "aggregate_summary.csv", index=False)
    _write_json(grid / "aggregate_summary.json", {"aggregate": []})
    pd.DataFrame(columns=["fold", "horizon", "seed", "objective", "reason"]).to_csv(
        grid / "failures.csv",
        index=False,
    )
    pd.DataFrame(columns=["fold", "horizon", "seed", "status"]).to_csv(
        grid / "ssl_comparison.csv",
        index=False,
    )
    return grid


def test_execution_v3_unit_payoff_thresholds_costs_and_stationary_no_trade(
    tmp_path: Path,
) -> None:
    grid = _write_grid(tmp_path)
    out = tmp_path / "execution_v3"

    summary = build_fi2010_execution_v3(
        neural_full_grid_dir=grid,
        out_dir=out,
        confidence_thresholds="0.33,0.5,0.95",
        fee_bps="0,100",
        spread_multipliers="0",
        latency_steps="0,1,10",
        overwrite=True,
    )

    assert summary.payoff_mode == "unit_payoff"
    assert summary.cost_mode == "unit_proxy"

    threshold = pd.read_csv(out / "confidence_threshold_summary.csv")
    row = threshold[threshold["threshold"] == 0.5].iloc[0]
    assert row["retained_sample_fraction"] == pytest.approx(4 / 6)
    assert row["active_trade_fraction"] == pytest.approx(3 / 6)
    assert row["directional_trade_hit_rate"] == pytest.approx(2 / 3)

    cost = pd.read_csv(out / "cost_sensitivity_summary.csv")
    cost_row = cost[cost["fee_bps"] == 100.0].iloc[0]
    assert cost_row["gross_proxy"] == pytest.approx(0.0)
    assert cost_row["net_proxy"] == pytest.approx(-0.04)
    assert cost_row["active_trade_count"] == 4

    manifest = json.loads((out / "execution_v3_manifest.json").read_text(encoding="utf-8"))
    assert manifest["payoff_mode"] == "unit_payoff"
    assert manifest["cost_mode"] == "unit_proxy"
    assert manifest["input_file_hashes"]
    assert any(
        item["diagnostic"] == "regime_execution"
        for item in manifest["skipped_diagnostics"]
    )


def test_execution_v3_uses_realised_return_and_spread_modes(tmp_path: Path) -> None:
    grid = _write_grid(tmp_path, realised_returns=True, market_context=True, regime=True)
    out = tmp_path / "execution_v3"

    summary = build_fi2010_execution_v3(
        neural_full_grid_dir=grid,
        out_dir=out,
        confidence_thresholds="0.33",
        fee_bps="0",
        spread_multipliers="1",
        latency_steps="0",
        overwrite=True,
    )

    assert summary.payoff_mode == "realised_return"
    assert summary.cost_mode == "spread_proxy"
    cost = pd.read_csv(out / "cost_sensitivity_summary.csv")
    assert cost.iloc[0]["payoff_mode"] == "realised_return"
    assert cost.iloc[0]["cost_mode"] == "spread_proxy"
    regime = pd.read_csv(out / "regime_execution_summary.csv")
    assert (regime["status"] == "ok").any()


def test_latency_fill_and_adverse_selection_label_proxy(tmp_path: Path) -> None:
    grid = _write_grid(tmp_path)
    out = tmp_path / "execution_v3"

    build_fi2010_execution_v3(
        neural_full_grid_dir=grid,
        out_dir=out,
        confidence_thresholds="0.33",
        fee_bps="0",
        spread_multipliers="0",
        latency_steps="0,1,10",
        overwrite=True,
    )

    latency = pd.read_csv(out / "latency_sensitivity_summary.csv")
    assert latency[latency["latency_step"] == 1].iloc[0]["retained_samples"] == 5
    assert latency[latency["latency_step"] == 10].iloc[0]["status"] == "skipped"

    fill = pd.read_csv(out / "fill_assumption_summary.csv")
    filled = dict(zip(fill["fill_mode"], fill["filled_count"], strict=False))
    assert filled["aggressive_crossing"] == 5
    assert filled["passive_optimistic"] == 4
    assert filled["passive_conservative"] == 2
    assert filled["abstain_only"] == 0

    adverse = pd.read_csv(out / "adverse_selection_summary.csv")
    aggressive = adverse[
        (adverse["fill_assumption"] == "aggressive_crossing")
        & (adverse["status"] == "ok")
    ]
    assert aggressive["adverse_selection_mode"].iloc[0] == "label_proxy"
    assert aggressive["adverse_count"].sum() == 2


def test_smoke_test_gating_and_strict_ambiguous_probability_failure(
    tmp_path: Path,
) -> None:
    smoke_grid = _write_grid(tmp_path / "smoke", smoke=True)
    with pytest.raises(ValueError, match="smoke-test artefacts"):
        build_fi2010_execution_v3(
            neural_full_grid_dir=smoke_grid,
            out_dir=tmp_path / "blocked",
        )
    summary = build_fi2010_execution_v3(
        neural_full_grid_dir=smoke_grid,
        out_dir=tmp_path / "smoke_out",
        allow_smoke_test=True,
        overwrite=True,
    )
    assert summary.smoke_test is True
    payload = json.loads((tmp_path / "smoke_out" / "summary.json").read_text())
    assert "not empirical evidence" in payload["smoke_test_status"]

    ambiguous_grid = _write_grid(tmp_path / "ambiguous", ambiguous=True)
    with pytest.raises(ValueError, match="label mapping audit failed"):
        build_fi2010_execution_v3(
            neural_full_grid_dir=ambiguous_grid,
            out_dir=tmp_path / "ambiguous_out",
            strict=True,
            overwrite=True,
        )


def test_execution_v3_cli_and_figure_pipeline(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    grid = _write_grid(tmp_path)
    out = tmp_path / "execution_v3"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chronoslob.cli",
            "build-fi2010-execution-v3",
            "--neural-full-grid",
            str(grid),
            "--out",
            str(out),
            "--confidence-thresholds",
            "0.33,0.5",
            "--latency-steps",
            "0,1",
            "--overwrite",
        ],
        cwd=project_root(),
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    figures = tmp_path / "figures"
    build_fi2010_neural_figures(
        neural_full_grid_dir=grid,
        execution_v3_dir=out,
        out_dir=figures,
        overwrite=True,
    )
    manifest = json.loads((figures / "figure_manifest.json").read_text(encoding="utf-8"))
    entries = {entry["figure_id"]: entry for entry in manifest["figures"]}
    assert entries["execution_v3_confidence_active_fraction"]["status"] == "completed"
    assert entries["execution_v3_cost_sensitivity"]["status"] == "completed"


def test_final_report_refuses_execution_v3_claim_when_missing(tmp_path: Path) -> None:
    from tests.test_fi2010_ssl_runner import _write_minimal_required_dirs

    dirs = _write_minimal_required_dirs(tmp_path)
    report_path = tmp_path / "report.md"
    build_final_empirical_report(
        classical_dir=dirs["classical"],
        neural_dir=dirs["neural"],
        uncertainty_dir=dirs["uncertainty"],
        out_path=report_path,
        overwrite=True,
    )

    text = report_path.read_text(encoding="utf-8")
    assert "no execution-v3 claim is made" in text
    assert "offline execution-aware proxy diagnostics only" in text
