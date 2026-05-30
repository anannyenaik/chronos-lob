from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from chronoslob.analysis.execution_centrepiece import build_execution_centrepiece
from chronoslob.utils.paths import project_root


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _make_execution_analysis(root: Path) -> Path:
    out = root / "execution_v3_analysis"
    _write_json(
        out / "summary.json",
        {
            "payoff_mode": "unit_payoff",
            "cost_mode": "unit_proxy",
            "run_group_count": 2,
            "smoke_test": False,
        },
    )
    confidence_rows = []
    for threshold, active, net in (
        (0.50, 0.40, -10.0),
        (0.70, 0.20, -3.0),
        (0.85, 0.05, 1.0),
    ):
        confidence_rows.append(
            {
                "pretraining_objective": "supervised",
                "horizon": 10,
                "threshold": threshold,
                "n_groups": 2,
                "mean_retained_fraction": active + 0.1,
                "mean_active_fraction": active,
                "mean_abstention_fraction": 1.0 - active,
                "mean_classification_accuracy": 0.45,
                "mean_macro_f1": 0.30 + threshold / 10.0,
                "mean_directional_hit_rate": 0.4,
                "mean_gross_directional_proxy": net + 2.0,
                "mean_cost_adjusted_proxy": net,
            }
        )
    _write_csv(out / "confidence_filtering_summary.csv", confidence_rows)
    _write_csv(
        out / "turnover_proxy_summary.csv",
        [
            {
                "pretraining_objective": "supervised",
                "horizon": 10,
                "threshold": row["threshold"],
                "n_groups": 2,
                "mean_signal_change_rate": row["mean_active_fraction"],
                "mean_turnover_adjusted_cost_proxy": -0.1,
            }
            for row in confidence_rows
        ],
    )
    _write_csv(
        out / "cost_sensitivity_summary.csv",
        [
            {
                "pretraining_objective": "supervised",
                "horizon": 10,
                "fee_bps": 0.0,
                "spread_multiplier": 0.0,
                "n_groups": 2,
                "mean_gross_proxy": 5.0,
                "mean_cost_adjusted_proxy": 5.0,
                "mean_degradation_percentage": 0.0,
            },
            {
                "pretraining_objective": "supervised",
                "horizon": 10,
                "fee_bps": 2.0,
                "spread_multiplier": 1.0,
                "n_groups": 2,
                "mean_gross_proxy": 5.0,
                "mean_cost_adjusted_proxy": 3.0,
                "mean_degradation_percentage": 40.0,
            },
        ],
    )
    _write_csv(
        out / "latency_sensitivity_summary.csv",
        [
            {
                "pretraining_objective": "supervised",
                "horizon": 10,
                "latency_step": 0,
                "n_groups": 2,
                "mean_directional_hit_rate": 0.50,
                "mean_net_degradation_vs_latency_0": 0.0,
            },
            {
                "pretraining_objective": "supervised",
                "horizon": 10,
                "latency_step": 10,
                "n_groups": 2,
                "mean_directional_hit_rate": 0.30,
                "mean_net_degradation_vs_latency_0": -2.0,
            },
        ],
    )
    _write_csv(
        out / "adverse_selection_proxy_summary.csv",
        [
            {
                "pretraining_objective": "supervised",
                "horizon": 10,
                "confidence_bucket": "0.85-1.00",
                "fill_assumption": "aggressive_crossing",
                "n_groups": 1,
                "total_filled": 20,
                "total_adverse": 4,
                "mean_adverse_fraction": 0.2,
                "weighted_adverse_fraction": 0.2,
                "adverse_selection_mode": "label_proxy",
            }
        ],
    )
    _write_csv(
        out / "fill_assumption_summary.csv",
        [
            {
                "pretraining_objective": "supervised",
                "horizon": 10,
                "fill_mode": "aggressive_crossing",
                "mean_fill_fraction": 1.0,
                "mean_directional_hit_rate": 0.4,
                "mean_cost_adjusted_proxy": 3.0,
            }
        ],
    )
    _write_json(
        out / "skipped_regime_diagnostics.json",
        {"status": "skipped", "reason": "regime context unavailable"},
    )
    _write_json(out / "execution_claim_assessment.json", {"claims": []})
    _write_json(out / "figure_manifest.json", {"figures": []})
    return out


def _make_neural_grid(root: Path) -> Path:
    out = root / "grid"
    _write_json(out / "summary.json", {"smoke_test": False})
    _write_csv(
        out / "aggregate_summary.csv",
        [
            {
                "model_family": "matrix_transformer",
                "pretraining_objective": "none",
                "horizon": 10,
                "lookback": 20,
                "completed_run_count": 2,
                "failed_run_count": 0,
                "mean_accuracy": 0.45,
                "mean_macro_f1": 0.33,
                "mean_mcc": 0.1,
                "mean_ece": 0.12,
                "mean_brier_score": 0.6,
                "mean_nll": 1.0,
            }
        ],
    )
    return out


def _make_execution_v3(root: Path) -> Path:
    out = root / "execution_v3"
    _write_json(out / "summary.json", {"smoke_test_status": "not smoke-test"})
    _write_json(out / "execution_v3_manifest.json", {"smoke_test": False})
    return out


def test_centrepiece_runs_from_retained_tables_without_raw_predictions(tmp_path: Path) -> None:
    analysis = _make_execution_analysis(tmp_path)
    neural = _make_neural_grid(tmp_path)
    execution_v3 = _make_execution_v3(tmp_path)
    out = tmp_path / "centrepiece"

    summary = build_execution_centrepiece(
        execution_analysis_dir=analysis,
        neural_full_grid_dir=neural,
        execution_v3_dir=execution_v3,
        out_dir=out,
        make_figures=False,
        overwrite=True,
    )

    assert summary.raw_predictions_required is False
    assert (out / "execution_centrepiece.md").is_file()
    assert (out / "forecasting_vs_signal_quality.csv").is_file()
    assert (out / "metric_to_proxy_gap.csv").is_file()
    assert (out / "figure_manifest.json").is_file()


def test_metric_gap_marks_unavailable_fields_explicitly(tmp_path: Path) -> None:
    out = tmp_path / "centrepiece"
    build_execution_centrepiece(
        execution_analysis_dir=_make_execution_analysis(tmp_path),
        neural_full_grid_dir=_make_neural_grid(tmp_path),
        execution_v3_dir=_make_execution_v3(tmp_path),
        out_dir=out,
        make_figures=False,
        overwrite=True,
    )

    frame = pd.read_csv(out / "metric_to_proxy_gap.csv")
    row = frame.iloc[0]
    assert row["pretraining_objective"] == "supervised"
    assert row["confidence_filtered_ece"].startswith("unavailable:")
    assert row["active_fraction_at_0_70"] == 0.20


def test_claim_assessment_blocks_high_risk_claims(tmp_path: Path) -> None:
    out = tmp_path / "centrepiece"
    summary = build_execution_centrepiece(
        execution_analysis_dir=_make_execution_analysis(tmp_path),
        neural_full_grid_dir=_make_neural_grid(tmp_path),
        execution_v3_dir=_make_execution_v3(tmp_path),
        out_dir=out,
        make_figures=False,
        overwrite=True,
    )

    assert summary.claim_statuses["forecasting_vs_signal_quality_gap_analysis"] == "supported"
    assert summary.claim_statuses["confidence_filtering_tradeoff_analysis"] == "supported"
    assert summary.claim_statuses["active_fraction_analysis"] == "supported"
    assert summary.claim_statuses["turnover_proxy_analysis"] == "supported"
    assert summary.claim_statuses["latency_cost_gap_analysis"] == "supported"
    assert summary.claim_statuses["adverse_selection_confidence_analysis"] == "supported"
    assert summary.claim_statuses["profitability_or_tradability"] == "forbidden"
    assert summary.claim_statuses["PnL"] == "forbidden"
    assert summary.claim_statuses["live_trading"] == "forbidden"


def test_cli_build_execution_centrepiece(tmp_path: Path) -> None:
    report = tmp_path / "centrepiece"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chronoslob.cli",
            "build-execution-centrepiece",
            "--execution-analysis",
            str(_make_execution_analysis(tmp_path)),
            "--execution-v3",
            str(_make_execution_v3(tmp_path)),
            "--neural-full-grid",
            str(_make_neural_grid(tmp_path)),
            "--out",
            str(report),
            "--no-figures",
            "--overwrite",
        ],
        cwd=project_root(),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ChronosLOB execution centrepiece builder" in completed.stdout
    assert (report / "centrepiece_summary.json").is_file()
