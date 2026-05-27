"""Tests for FI-2010 execution-aware evaluation v2.

The tests use tiny synthetic artefacts only. They never read FI-2010 data,
full prediction rows or model checkpoints.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from chronoslob.cli import _run_fi2010_execution_v2_impl
from chronoslob.experiments.execution_v2 import run_fi2010_execution_v2
from chronoslob.utils.paths import project_root

_REQUIRED_RESULT_COLUMNS = {
    "model_name",
    "fold_id",
    "split",
    "confidence_threshold",
    "cost_bps",
    "latency_steps",
    "eligible_predictions",
    "coverage",
    "trade_count_proxy",
    "turnover_proxy",
    "gross_signal_return_proxy",
    "cost_proxy",
    "net_signal_return_proxy",
    "hit_rate_proxy",
    "adverse_selection_proxy",
    "fill_assumption",
    "status",
    "skip_reason",
}

_ASSUMPTION_CAVEATS = (
    "not a backtest",
    "not a live-trading simulation",
    "no market impact model",
    "no queue-position ground truth",
    "fills are approximate or unavailable",
    "costs are scenario assumptions",
    "latency is row-step latency",
    "not exchange or network latency",
    "useful for stress-testing signal fragility, not for proving tradability",
)

_DOCS_TO_AUDIT = (
    "docs/FI2010_EXECUTION_V2.md",
    "docs/CLI_REFERENCE.md",
    "docs/EXPERIMENT_EVIDENCE_INDEX.md",
    "docs/REPRODUCIBILITY.md",
    "reports/10_10_research_protocol.md",
)


def _write_classical_dir(
    classical_dir: Path,
    *,
    include_trade_count: bool = True,
    latencies: tuple[int, ...] = (0, 1),
) -> None:
    classical_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "fold_id": fold,
                "model_name": "gradient_boosting",
                "split": "test",
                "macro_f1": 0.45 + 0.01 * fold,
                "status": "ok",
            }
            for fold in (1, 2)
        ]
    ).to_csv(classical_dir / "results_by_fold.csv", index=False)

    for fold in (1, 2):
        fold_dir = classical_dir / "folds" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        for threshold in (0.0, 0.6):
            for cost in (0.0, 5.0):
                for latency in latencies:
                    eligible = 100 - 20 * int(threshold > 0.0) - fold
                    gross = 12.0 - latency - 0.5 * fold + 2.0 * threshold
                    row: dict[str, object] = {
                        "model_name": "gradient_boosting",
                        "split": "test",
                        "confidence_threshold": threshold,
                        "cost_bps": cost,
                        "latency_steps": latency,
                        "eligible_predictions": eligible,
                        "turnover_proxy": float(eligible),
                        "gross_signal_return_proxy": gross,
                        "cost_proxy": cost,
                        "net_signal_return_proxy": gross - cost,
                        "hit_rate_proxy": 0.55 + 0.05 * threshold,
                    }
                    if include_trade_count:
                        row["trade_count_proxy"] = eligible - 1
                    rows.append(row)
        pd.DataFrame(rows).to_csv(fold_dir / "execution_sensitivity.csv", index=False)


def _write_neural_dir(neural_dir: Path) -> None:
    neural_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "fold_id": 1,
                "seed": 0,
                "model_name": "matrix_transformer",
                "split": "test",
                "macro_f1": 0.73,
                "status": "ok",
            }
        ]
    ).to_csv(neural_dir / "results_by_fold_seed.csv", index=False)


def test_cli_impl_runs_on_tiny_artefacts(tmp_path: Path) -> None:
    classical_dir = tmp_path / "classical"
    neural_dir = tmp_path / "neural"
    ablations_dir = tmp_path / "ablations"
    out_dir = tmp_path / "out"
    _write_classical_dir(classical_dir)
    _write_neural_dir(neural_dir)
    ablations_dir.mkdir()

    exit_code = _run_fi2010_execution_v2_impl(
        classical_dir=classical_dir,
        neural_dir=neural_dir,
        ablations_dir=ablations_dir,
        out=out_dir,
        models=None,
        cost_bps=None,
        latency_steps=None,
        confidence_thresholds=None,
        overwrite=True,
    )

    assert exit_code == 0
    for filename in (
        "summary.json",
        "execution_v2_results.csv",
        "cost_latency_surface.csv",
        "confidence_threshold_summary.csv",
        "turnover_summary.csv",
        "adverse_selection_summary.csv",
        "fill_assumption_summary.csv",
        "degradation_summary.csv",
        "skipped_diagnostics.json",
        "execution_assumptions.md",
        "execution_notes.md",
    ):
        assert (out_dir / filename).is_file()

    result = pd.read_csv(out_dir / "execution_v2_results.csv")
    assert set(result.columns) >= _REQUIRED_RESULT_COLUMNS
    assert {"ok", "skipped"} <= set(result["status"])
    surface = pd.read_csv(out_dir / "cost_latency_surface.csv")
    threshold = pd.read_csv(out_dir / "confidence_threshold_summary.csv")
    assert not surface.empty
    assert not threshold.empty


def test_missing_adverse_and_fill_inputs_are_recorded(tmp_path: Path) -> None:
    classical_dir = tmp_path / "classical"
    out_dir = tmp_path / "out"
    _write_classical_dir(
        classical_dir,
        include_trade_count=False,
        latencies=(0,),
    )

    summary = run_fi2010_execution_v2(
        classical_dir=classical_dir,
        neural_dir=None,
        ablations_dir=None,
        out_dir=out_dir,
        overwrite=True,
    )

    assert "adverse_selection" in summary.diagnostics_skipped
    assert "fill_assumption" in summary.diagnostics_skipped
    skipped = json.loads((out_dir / "skipped_diagnostics.json").read_text("utf-8"))
    reasons = {
        (entry["diagnostic"], entry["scope"]): entry["skip_reason"]
        for entry in skipped["skipped"]
    }
    assert ("adverse_selection", "all_models") in reasons
    assert "single latency step" in reasons[("adverse_selection", "all_models")]
    assert ("fill_assumption", "classical") in reasons
    assert "trade-count column" in reasons[("fill_assumption", "classical")]


def test_execution_assumptions_contain_required_caveats(tmp_path: Path) -> None:
    classical_dir = tmp_path / "classical"
    out_dir = tmp_path / "out"
    _write_classical_dir(classical_dir)

    run_fi2010_execution_v2(
        classical_dir=classical_dir,
        out_dir=out_dir,
        overwrite=True,
    )

    text = (out_dir / "execution_assumptions.md").read_text("utf-8").lower()
    normalised = re.sub(r"\s+", " ", text)
    for caveat in _ASSUMPTION_CAVEATS:
        assert caveat in normalised


def test_overwrite_protection(tmp_path: Path) -> None:
    classical_dir = tmp_path / "classical"
    out_dir = tmp_path / "out"
    _write_classical_dir(classical_dir)

    run_fi2010_execution_v2(
        classical_dir=classical_dir,
        out_dir=out_dir,
        overwrite=True,
    )

    with pytest.raises(FileExistsError):
        run_fi2010_execution_v2(
            classical_dir=classical_dir,
            out_dir=out_dir,
            overwrite=False,
        )


def test_no_full_predictions_or_checkpoints_are_required(tmp_path: Path) -> None:
    classical_dir = tmp_path / "classical"
    neural_dir = tmp_path / "neural"
    out_dir = tmp_path / "out"
    _write_classical_dir(classical_dir)
    _write_neural_dir(neural_dir)

    summary = run_fi2010_execution_v2(
        classical_dir=classical_dir,
        neural_dir=neural_dir,
        out_dir=out_dir,
        overwrite=True,
    )

    assert summary.full_predictions_required is False
    assert summary.checkpoints_required is False
    assert not list(out_dir.rglob("predictions*.csv"))
    assert not list(out_dir.rglob("*.pt"))
    assert not list(out_dir.rglob("*.pth"))
    assert not list(out_dir.rglob("*.ckpt"))


def test_docs_avoid_forbidden_public_claims() -> None:
    from chronoslob.utils.audit import (
        check_no_forbidden_claims,
        check_public_release_wording,
    )

    root = project_root()
    assert check_no_forbidden_claims(root, scan_paths=_DOCS_TO_AUDIT).ok
    assert check_public_release_wording(root, scan_paths=_DOCS_TO_AUDIT).ok
