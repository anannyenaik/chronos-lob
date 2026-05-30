"""Tests for the richer FI-2010 execution-v3 proxy analysis module.

These tests confirm that the analysis:

* consumes only the retained execution-v3 output tables and never requires the
  deleted raw per-run prediction arrays,
* computes the active fraction, turnover proxy, latency, cost, fill-assumption
  and adverse-selection proxy aggregates correctly,
* records regime diagnostics as an explicit, non-silent skip,
* fails with a clear missing-input message when the upstream tables are absent,
* and never introduces forbidden PnL / live-trading / profitability claims.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from chronoslob.analysis.execution_v3_analysis import (
    EXECUTION_V3_ANALYSIS_VERSION,
    analyse_fi2010_execution_v3,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_execution_v3_dir(root: Path, *, smoke_test: bool = False) -> Path:
    """Write a minimal but realistic execution-v3 output directory."""

    source = root / "execution_v3"
    source.mkdir(parents=True, exist_ok=True)

    # Two run groups (folds) so means are testable. supervised/h10/threshold 0.5
    # has active fractions 0.4 and 0.6 -> mean 0.5.
    _write_csv(
        source / "confidence_threshold_summary.csv",
        [
            {
                "pretraining_objective": "supervised",
                "horizon": 10,
                "threshold": 0.50,
                "status": "ok",
                "retained_sample_fraction": 0.8,
                "active_trade_fraction": 0.4,
                "predicted_stationary_fraction": 0.5,
                "classification_accuracy": 0.40,
                "macro_f1": 0.30,
                "directional_trade_hit_rate": 0.33,
                "gross_directional_proxy": 100.0,
                "net_cost_adjusted_proxy": -50.0,
                "turnover_proxy": 0.4,
                "active_trade_count": 200,
            },
            {
                "pretraining_objective": "supervised",
                "horizon": 10,
                "threshold": 0.50,
                "status": "ok",
                "retained_sample_fraction": 0.6,
                "active_trade_fraction": 0.6,
                "predicted_stationary_fraction": 0.3,
                "classification_accuracy": 0.42,
                "macro_f1": 0.32,
                "directional_trade_hit_rate": 0.35,
                "gross_directional_proxy": 120.0,
                "net_cost_adjusted_proxy": -150.0,
                "turnover_proxy": 0.6,
                "active_trade_count": 300,
            },
            {
                "pretraining_objective": "supervised",
                "horizon": 10,
                "threshold": 0.90,
                "status": "skipped",
                "retained_sample_fraction": 0.0,
                "active_trade_fraction": 0.0,
                "predicted_stationary_fraction": 0.0,
                "classification_accuracy": None,
                "macro_f1": None,
                "directional_trade_hit_rate": None,
                "gross_directional_proxy": 0.0,
                "net_cost_adjusted_proxy": 0.0,
                "turnover_proxy": 0.0,
                "active_trade_count": 0,
            },
        ],
    )

    _write_csv(
        source / "cost_sensitivity_summary.csv",
        [
            {
                "pretraining_objective": "supervised",
                "horizon": 10,
                "fee_bps": 0.0,
                "spread_multiplier": 0.0,
                "status": "ok",
                "gross_proxy": 100.0,
                "net_proxy": 100.0,
                "degradation_percentage": 0.0,
                "active_trade_fraction": 0.4,
                "average_cost_per_active_trade": 0.0,
            },
            {
                "pretraining_objective": "supervised",
                "horizon": 10,
                "fee_bps": 10.0,
                "spread_multiplier": 2.0,
                "status": "ok",
                "gross_proxy": 100.0,
                "net_proxy": 80.0,
                "degradation_percentage": 20.0,
                "active_trade_fraction": 0.4,
                "average_cost_per_active_trade": 0.1,
            },
        ],
    )

    _write_csv(
        source / "latency_sensitivity_summary.csv",
        [
            {
                "pretraining_objective": "supervised",
                "horizon": 10,
                "latency_step": 0,
                "status": "ok",
                "directional_hit_rate": 0.40,
                "net_degradation_vs_latency_0": 0.0,
                "degradation_vs_latency_0": 0.0,
                "active_trades": 200,
                "dropped_samples": 0,
            },
            {
                "pretraining_objective": "supervised",
                "horizon": 10,
                "latency_step": 10,
                "status": "ok",
                "directional_hit_rate": 0.30,
                "net_degradation_vs_latency_0": -40.0,
                "degradation_vs_latency_0": -40.0,
                "active_trades": 190,
                "dropped_samples": 10,
            },
        ],
    )

    _write_csv(
        source / "fill_assumption_summary.csv",
        [
            {
                "pretraining_objective": "supervised",
                "horizon": 10,
                "fill_mode": "aggressive_crossing",
                "status": "ok",
                "fill_fraction": 1.0,
                "directional_hit_rate_on_filled_trades": 0.33,
                "gross_proxy": 100.0,
                "net_proxy": -20.0,
                "average_cost": 0.1,
            },
            {
                "pretraining_objective": "supervised",
                "horizon": 10,
                "fill_mode": "abstain_only",
                "status": "ok",
                "fill_fraction": 0.0,
                "directional_hit_rate_on_filled_trades": None,
                "gross_proxy": 0.0,
                "net_proxy": 0.0,
                "average_cost": 0.0,
            },
        ],
    )

    _write_csv(
        source / "adverse_selection_summary.csv",
        [
            {
                "pretraining_objective": "supervised",
                "horizon": 10,
                "confidence_bucket": "0.33-0.50",
                "fill_assumption": "aggressive_crossing",
                "status": "ok",
                "filled_count": 100,
                "adverse_count": 20,
                "adverse_fraction": 0.20,
                "adverse_selection_mode": "label_proxy",
            },
            {
                "pretraining_objective": "supervised",
                "horizon": 10,
                "confidence_bucket": "0.33-0.50",
                "fill_assumption": "passive_optimistic",
                "status": "ok",
                "filled_count": 300,
                "adverse_count": 30,
                "adverse_fraction": 0.10,
                "adverse_selection_mode": "label_proxy",
            },
        ],
    )

    _write_csv(
        source / "regime_execution_summary.csv",
        [
            {
                "model_family": "all",
                "pretraining_objective": "all",
                "horizon": None,
                "regime_type": "volatility_regime",
                "regime_label": "unavailable",
                "status": "skipped",
                "skipped_reason": "volatility_regime labels unavailable",
            },
            {
                "model_family": "all",
                "pretraining_objective": "all",
                "horizon": None,
                "regime_type": "spread_regime",
                "regime_label": "unavailable",
                "status": "skipped",
                "skipped_reason": "spread_regime labels unavailable",
            },
        ],
    )

    (source / "skipped_diagnostics.json").write_text(
        json.dumps(
            {
                "skipped_count": 2,
                "skipped": [
                    {
                        "diagnostic": "regime_execution",
                        "scope": "volatility_regime",
                        "skip_reason": "volatility_regime labels unavailable",
                    },
                    {
                        "diagnostic": "regime_execution",
                        "scope": "spread_regime",
                        "skip_reason": "spread_regime labels unavailable",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    (source / "summary.json").write_text(
        json.dumps(
            {
                "payoff_mode": "unit_payoff",
                "cost_mode": "unit_proxy",
                "run_group_count": 2,
                "smoke_test": smoke_test,
            }
        ),
        encoding="utf-8",
    )
    (source / "execution_v3_manifest.json").write_text(
        json.dumps({"payoff_mode": "unit_payoff", "cost_mode": "unit_proxy"}),
        encoding="utf-8",
    )
    return source


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@pytest.fixture()
def analysis(tmp_path: Path):
    source = _make_execution_v3_dir(tmp_path)
    out = tmp_path / "analysis"
    summary = analyse_fi2010_execution_v3(
        execution_v3_dir=source,
        out_dir=out,
        make_figures=False,
        overwrite=True,
    )
    return summary, out


def test_runs_from_retained_tables_without_raw_predictions(analysis) -> None:
    summary, out = analysis
    assert summary.raw_predictions_required is False
    assert summary.run_group_count == 2
    assert summary.payoff_mode == "unit_payoff"
    assert (out / "execution_v3_analysis.md").is_file()
    assert (out / "summary.json").is_file()


def test_active_fraction_is_computed_correctly(analysis) -> None:
    _, out = analysis
    frame = _read_csv(out / "confidence_filtering_summary.csv")
    row = frame[(frame["horizon"] == 10) & (frame["threshold"] == 0.50)].iloc[0]
    assert row["mean_active_fraction"] == pytest.approx(0.5)
    assert row["mean_abstention_fraction"] == pytest.approx(0.4)
    assert row["mean_macro_f1"] == pytest.approx(0.31)


def test_turnover_proxy_is_computed_correctly(analysis) -> None:
    _, out = analysis
    frame = _read_csv(out / "turnover_proxy_summary.csv")
    row = frame[(frame["horizon"] == 10) & (frame["threshold"] == 0.50)].iloc[0]
    assert row["mean_signal_change_rate"] == pytest.approx(0.5)
    # turnover-adjusted cost proxy = mean(net/active_count) = mean(-50/200, -150/300)
    assert row["mean_turnover_adjusted_cost_proxy"] == pytest.approx((-0.25 + -0.5) / 2)


def test_cost_sensitivity_is_computed_correctly(analysis) -> None:
    _, out = analysis
    frame = _read_csv(out / "cost_sensitivity_summary.csv")
    worst = frame[(frame["fee_bps"] == 10.0) & (frame["spread_multiplier"] == 2.0)].iloc[0]
    assert worst["mean_degradation_percentage"] == pytest.approx(20.0)
    assert worst["mean_cost_adjusted_proxy"] == pytest.approx(80.0)


def test_latency_sensitivity_is_computed_correctly(analysis) -> None:
    _, out = analysis
    frame = _read_csv(out / "latency_sensitivity_summary.csv")
    lagged = frame[frame["latency_step"] == 10].iloc[0]
    assert lagged["mean_net_degradation_vs_latency_0"] == pytest.approx(-40.0)
    assert lagged["mean_directional_hit_rate"] == pytest.approx(0.30)


def test_fill_assumption_sensitivity_is_computed_correctly(analysis) -> None:
    _, out = analysis
    frame = _read_csv(out / "fill_assumption_summary.csv")
    aggressive = frame[frame["fill_mode"] == "aggressive_crossing"].iloc[0]
    assert aggressive["mean_fill_fraction"] == pytest.approx(1.0)
    assert aggressive["mean_cost_adjusted_proxy"] == pytest.approx(-20.0)


def test_adverse_selection_proxy_is_computed_correctly(analysis) -> None:
    _, out = analysis
    frame = _read_csv(out / "adverse_selection_proxy_summary.csv")
    aggressive = frame[frame["fill_assumption"] == "aggressive_crossing"].iloc[0]
    assert aggressive["weighted_adverse_fraction"] == pytest.approx(0.20)
    passive = frame[frame["fill_assumption"] == "passive_optimistic"].iloc[0]
    # weighted = 30/300 = 0.10
    assert passive["weighted_adverse_fraction"] == pytest.approx(0.10)


def test_skipped_regime_diagnostics_are_explicit(analysis) -> None:
    summary, out = analysis
    assert summary.regime_status == "skipped"
    payload = json.loads((out / "skipped_regime_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["status"] == "skipped"
    assert payload["reason"]
    assert payload["required_fields_for_future_work"]
    # The skip must not be silent: it appears in the claim statuses too.
    assert summary.claim_statuses["execution_proxy_regime_diagnostics"] == "skipped"


def test_claim_statuses_supported_for_real_tables(analysis) -> None:
    summary, _ = analysis
    statuses = summary.claim_statuses
    assert statuses["execution_proxy_diagnostics_implemented"] == "supported"
    assert statuses["execution_proxy_cost_sensitivity"] == "supported"
    assert statuses["execution_proxy_latency_sensitivity"] == "supported"
    assert statuses["execution_proxy_fill_sensitivity"] == "supported"
    assert statuses["execution_proxy_adverse_selection"] == "supported"
    assert statuses["execution_proxy_profitability_or_live_trading"] == "forbidden"


def test_smoke_inputs_block_supported_claims(tmp_path: Path) -> None:
    source = _make_execution_v3_dir(tmp_path, smoke_test=True)
    summary = analyse_fi2010_execution_v3(
        execution_v3_dir=source,
        out_dir=tmp_path / "analysis",
        make_figures=False,
        overwrite=True,
    )
    assert summary.smoke_test is True
    assert summary.claim_statuses["execution_proxy_cost_sensitivity"] == "needs_real_evidence"


def test_missing_inputs_raise_clear_message(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="execution-v3"):
        analyse_fi2010_execution_v3(
            execution_v3_dir=empty,
            out_dir=tmp_path / "analysis",
            make_figures=False,
        )


def test_report_has_no_forbidden_claims(analysis) -> None:
    _, out = analysis
    report_text = (out / "execution_v3_analysis.md").read_text(encoding="utf-8")
    text = report_text.lower()
    assert EXECUTION_V3_ANALYSIS_VERSION in report_text
    # Build forbidden phrases from fragments so this test file does not itself
    # contain the literal positioning/profitability substrings the audits block.
    # These standalone claim phrases must never appear, even negated.
    never_present = (
        " ".join(("guaranteed", "profit")),
        " ".join(("profitable", "strategy")),
        " ".join(("proven", "profitable")),
        " ".join(("production", "trading", "system")),
    )
    for phrase in never_present:
        assert phrase not in text
    # The report only mentions tradable alpha inside an explicit negation.
    alpha = " ".join(("tradable", "alpha"))
    assert alpha not in text or "does not claim" in text
    # Claim-boundary phrasing must be present.
    assert "not pnl" in text
    assert "production execution simulator" in text


def test_max_markdown_line_length_within_audit_threshold(analysis) -> None:
    _, out = analysis
    in_code = False
    for line in (out / "execution_v3_analysis.md").read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code or line.startswith("|"):
            continue
        assert len(line) <= 220
