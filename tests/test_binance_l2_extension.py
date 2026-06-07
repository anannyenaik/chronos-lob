"""Tests for the storage-light Binance L2 replay extension."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from chronoslob.binance_l2.pipeline import BinanceL2Config, run_binance_l2_pipeline
from chronoslob.experiments.evidence_pack import (
    EvidencePackConfig,
    audit_claims,
    discover_artefacts,
)
from chronoslob.experiments.final_report import build_final_empirical_report
from chronoslob.utils.paths import project_root

FIXTURES = Path(__file__).parent / "fixtures" / "binance"
SNAPSHOT = FIXTURES / "synthetic_snapshot.json"
UPDATES = FIXTURES / "synthetic_diff_updates.jsonl"
STALE_UPDATES = FIXTURES / "synthetic_stale_updates.jsonl"
GAP_UPDATES = FIXTURES / "synthetic_gap_updates.jsonl"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def test_binance_l2_pipeline_writes_storage_light_fixture_report(tmp_path: Path) -> None:
    out = tmp_path / "binance_l2"

    result = run_binance_l2_pipeline(
        out,
        BinanceL2Config(snapshot_path=SNAPSHOT, updates_path=UPDATES),
        overwrite=True,
    )

    assert result.replay_ok
    assert result.diff_event_count == 3
    assert result.applied_event_count == 3
    expected = {
        "summary.json",
        "replay_quality.json",
        "feature_summary.csv",
        "book_snapshot_summary.csv",
        "update_continuity_summary.csv",
        "binance_claim_assessment.json",
        "binance_l2_schema.md",
        "binance_l2_report.md",
        "figure_manifest.json",
    }
    assert expected <= {path.name for path in result.files_written}
    assert not list(out.glob("*_full.*"))
    assert not list(out.glob("raw_*.json*"))

    summary = _read_json(out / "summary.json")
    assert summary["fixture_data"] is True
    assert summary["evidence_level"] == "binance_l2_fixture_replay"
    assert summary["aggregated_level_updates"] is True
    assert summary["live_trading"] is False
    assert summary["network_calls"] == 0
    assert summary["input_file_hashes"]

    report = (out / "binance_l2_report.md").read_text(encoding="utf-8").lower()
    assert "synthetic fixtures" in report
    assert "not equity-market evidence" in report
    assert "not live trading" in report
    assert "does not establish profitability" in report
    assert "not individual order" in report

    claims = _read_json(out / "binance_claim_assessment.json")["claims"]
    assert claims["binance_l2_replay_pipeline"]["status"] == "supported"
    assert claims["real_captured_aggregated_l2_stream_path"]["status"] == "needs_real_evidence"
    assert claims["live_trading_or_profitability"]["status"] == "forbidden"


def test_binance_l2_pipeline_records_stale_and_gap_quality(tmp_path: Path) -> None:
    stale_out = tmp_path / "stale"
    stale = run_binance_l2_pipeline(
        stale_out,
        BinanceL2Config(snapshot_path=SNAPSHOT, updates_path=STALE_UPDATES),
        overwrite=True,
    )
    stale_quality = _read_json(stale_out / "replay_quality.json")

    assert stale.replay_ok
    assert stale_quality["skipped_stale_count"] == 2
    assert stale_quality["applied_event_count"] == 2

    gap_out = tmp_path / "gap"
    gap = run_binance_l2_pipeline(
        gap_out,
        BinanceL2Config(snapshot_path=SNAPSHOT, updates_path=GAP_UPDATES),
        overwrite=True,
    )
    gap_quality = _read_json(gap_out / "replay_quality.json")

    assert not gap.replay_ok
    assert gap_quality["gap_count"] == 1
    assert gap_quality["ok"] is False


def test_replay_binance_l2_sample_cli_writes_report(tmp_path: Path) -> None:
    out = tmp_path / "cli_binance_l2"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chronoslob.cli",
            "replay-binance-l2-sample",
            "--out",
            str(out),
            "--overwrite",
        ],
        cwd=project_root(),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "synthetic fixture" in completed.stdout
    assert "network calls:      none performed" in completed.stdout
    assert (out / "binance_l2_report.md").is_file()


def test_evidence_pack_audits_binance_l2_fixture_claims(tmp_path: Path) -> None:
    out = tmp_path / "binance_l2"
    run_binance_l2_pipeline(
        out,
        BinanceL2Config(snapshot_path=SNAPSHOT, updates_path=UPDATES),
        overwrite=True,
    )
    config = EvidencePackConfig(
        out_dir=tmp_path / "pack",
        classical_dir=tmp_path / "classical",
        ssl_dir=tmp_path / "ssl",
        proper_training_dir=tmp_path / "proper_training",
        ssl_analysis_dir=tmp_path / "ssl_analysis",
        neural_full_grid_dir=tmp_path / "grid",
        figures_dir=tmp_path / "figures",
        execution_v3_dir=tmp_path / "execution_v3",
        feature_audit_dir=None,
        feature_ablations_dir=tmp_path / "feature_ablations",
        feature_ablation_analysis_dir=tmp_path / "feature_ablation_analysis",
        ablation_figures_dir=tmp_path / "ablation_figures",
        final_report_path=tmp_path / "final_report.md",
        synthetic_lob_dir=tmp_path / "synthetic_lob",
        binance_l2_dir=out,
        project_audit_dir=None,
        strict=False,
        allow_smoke_test=True,
        overwrite=True,
    )

    records = discover_artefacts(config)
    record = next(item for item in records if item.artefact_name == "binance_l2_extension_report")
    assert record.status == "partial_real"
    assert record.input_hashes_present is True

    claims = {claim.claim_id: claim for claim in audit_claims(records)}
    assert claims["binance_l2.replay_pipeline"].status == "supported"
    assert (
        claims["binance_l2.real_captured_aggregated_l2_stream_path"].status
        == "needs_real_evidence"
    )
    assert claims["binance_l2.equity_market_generalisation"].status == "unsupported"
    assert claims["binance_l2.live_trading_or_profitability"].status == "forbidden"


def test_final_report_includes_binance_l2_section(tmp_path: Path) -> None:
    artefacts = _minimal_final_report_inputs(tmp_path / "final_inputs")
    binance_out = tmp_path / "binance_l2"
    run_binance_l2_pipeline(
        binance_out,
        BinanceL2Config(snapshot_path=SNAPSHOT, updates_path=UPDATES),
        overwrite=True,
    )
    report_path = tmp_path / "final_report.md"

    build_final_empirical_report(
        classical_dir=artefacts["classical"],
        neural_dir=artefacts["neural"],
        uncertainty_dir=artefacts["uncertainty"],
        binance_l2_dir=binance_out,
        out_path=report_path,
        overwrite=True,
    )

    text = report_path.read_text(encoding="utf-8")
    assert "## Binance L2 Replay" in text
    assert "binance_l2_fixture_replay" in text
    assert "aggregated depth-stream" in text
    assert "ingestion and replay" in text
    assert "not equity-market evidence" in text
    assert "not live-trading evidence" in text
    assert "profitability, tradability or predictive-success evidence" in text


def _minimal_final_report_inputs(base: Path) -> dict[str, Path]:
    classical = base / "classical"
    neural = base / "neural"
    uncertainty = base / "uncertainty"
    _write_json(
        classical / "summary.json",
        {
            "dataset_name": "FI-2010",
            "task_name": "midprice_direction",
            "target_horizon": 10,
            "fold_count": 1,
        },
    )
    _write_csv(
        classical / "results_summary.csv",
        ["model_name", "split", "fold_count", "macro_f1_mean", "mcc_mean"],
        [
            {
                "model_name": "logistic",
                "split": "test",
                "fold_count": 1,
                "macro_f1_mean": 0.4,
                "mcc_mean": 0.1,
            }
        ],
    )
    _write_json(
        neural / "summary.json",
        {"dataset_name": "FI-2010", "task_name": "midprice_direction", "fold_count": 1},
    )
    _write_csv(
        neural / "results_summary.csv",
        ["model_name", "lookback", "split", "fold_count", "seed_count", "macro_f1_mean"],
        [
            {
                "model_name": "matrix_transformer",
                "lookback": 20,
                "split": "test",
                "fold_count": 1,
                "seed_count": 1,
                "macro_f1_mean": 0.5,
            }
        ],
    )
    _write_json(uncertainty / "summary.json", {"warnings": []})
    _write_csv(
        uncertainty / "metric_confidence_intervals.csv",
        ["source", "model_name", "split", "metric", "n_folds", "mean", "ci_lower", "ci_upper"],
        [
            {
                "source": "classical",
                "model_name": "logistic",
                "split": "test",
                "metric": "macro_f1",
                "n_folds": 1,
                "mean": 0.4,
                "ci_lower": 0.4,
                "ci_upper": 0.4,
            }
        ],
    )
    _write_csv(
        uncertainty / "model_ranking.csv",
        ["source", "split", "metric", "rank", "model_name", "n_folds", "mean"],
        [
            {
                "source": "neural",
                "split": "test",
                "metric": "macro_f1",
                "rank": 1,
                "model_name": "matrix_transformer",
                "n_folds": 1,
                "mean": 0.5,
            }
        ],
    )
    return {"classical": classical, "neural": neural, "uncertainty": uncertainty}
