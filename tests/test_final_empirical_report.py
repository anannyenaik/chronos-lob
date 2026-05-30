"""Tests for the final FI-2010 empirical report builder."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from chronoslob.experiments.final_report import build_final_empirical_report
from chronoslob.utils.paths import project_root


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _phrase(*parts: str) -> str:
    return " ".join(parts)


@pytest.fixture()
def tiny_final_artefacts(tmp_path: Path) -> dict[str, Path]:
    base = tmp_path / "artefacts"
    classical = base / "fi2010_multifold_classical"
    neural = base / "fi2010_multifold_neural"
    uncertainty = base / "fi2010_uncertainty"
    ablations = base / "fi2010_brutal_ablations"
    execution = base / "fi2010_execution_v2"
    external = base / "fi2010_external_context"

    _write_json(
        classical / "summary.json",
        {
            "dataset_name": "FI-2010",
            "task_name": "midprice_direction",
            "target_horizon": 10,
            "fold_count": 2,
            "folds_completed": [1, 2],
            "models_requested": ["logistic", "gradient_boosting"],
            "seeds": [0],
            "full_predictions_written": False,
            "git_commit": "abc123",
            "warnings": [],
        },
    )
    _write_csv(
        classical / "results_summary.csv",
        [
            "model_name",
            "split",
            "fold_count",
            "run_count",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
            "mcc_mean",
            "mcc_std",
        ],
        [
            {
                "model_name": "logistic",
                "split": "test",
                "fold_count": 2,
                "run_count": 2,
                "accuracy_mean": 0.55,
                "accuracy_std": 0.01,
                "macro_f1_mean": 0.33,
                "macro_f1_std": 0.02,
                "mcc_mean": 0.12,
                "mcc_std": 0.01,
            },
            {
                "model_name": "gradient_boosting",
                "split": "test",
                "fold_count": 2,
                "run_count": 2,
                "accuracy_mean": 0.64,
                "accuracy_std": 0.01,
                "macro_f1_mean": 0.46,
                "macro_f1_std": 0.004,
                "mcc_mean": 0.27,
                "mcc_std": 0.01,
            },
        ],
    )

    _write_json(
        neural / "summary.json",
        {
            "dataset_name": "FI-2010",
            "task_name": "midprice_direction",
            "target_horizon": 10,
            "folds_completed": ["fold_1", "fold_2"],
            "models_requested": ["deeplob_style", "matrix_transformer"],
            "lookbacks": [20],
            "seeds": [0],
            "full_predictions_written": False,
            "checkpoints_written": False,
            "warnings": [],
        },
    )
    _write_csv(
        neural / "results_summary.csv",
        [
            "model_name",
            "lookback",
            "split",
            "fold_count",
            "seed_count",
            "run_count",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
            "mcc_mean",
            "mcc_std",
        ],
        [
            {
                "model_name": "deeplob_style",
                "lookback": 20,
                "split": "test",
                "fold_count": 2,
                "seed_count": 1,
                "run_count": 2,
                "accuracy_mean": 0.48,
                "accuracy_std": 0.02,
                "macro_f1_mean": 0.47,
                "macro_f1_std": 0.03,
                "mcc_mean": 0.29,
                "mcc_std": 0.02,
            },
            {
                "model_name": "matrix_transformer",
                "lookback": 20,
                "split": "test",
                "fold_count": 2,
                "seed_count": 1,
                "run_count": 2,
                "accuracy_mean": 0.80,
                "accuracy_std": 0.02,
                "macro_f1_mean": 0.73,
                "macro_f1_std": 0.03,
                "mcc_mean": 0.62,
                "mcc_std": 0.03,
            },
        ],
    )

    _write_json(
        uncertainty / "summary.json",
        {
            "classical": {"seed_variance_available": False},
            "neural": {"seed_variance_available": False, "seeds": [0]},
            "warnings": [],
        },
    )
    _write_csv(
        uncertainty / "metric_confidence_intervals.csv",
        [
            "source",
            "model_name",
            "lookback",
            "split",
            "metric",
            "n_folds",
            "n_seeds",
            "mean",
            "std",
            "ci_lower",
            "ci_upper",
            "bootstrap_lower",
            "bootstrap_upper",
        ],
        [
            {
                "source": "classical",
                "model_name": "gradient_boosting",
                "lookback": "",
                "split": "test",
                "metric": "macro_f1",
                "n_folds": 2,
                "n_seeds": 1,
                "mean": 0.46,
                "std": 0.004,
                "ci_lower": 0.45,
                "ci_upper": 0.47,
                "bootstrap_lower": 0.452,
                "bootstrap_upper": 0.468,
            },
            {
                "source": "neural",
                "model_name": "matrix_transformer",
                "lookback": 20,
                "split": "test",
                "metric": "macro_f1",
                "n_folds": 2,
                "n_seeds": 1,
                "mean": 0.73,
                "std": 0.03,
                "ci_lower": 0.70,
                "ci_upper": 0.76,
                "bootstrap_lower": 0.701,
                "bootstrap_upper": 0.759,
            },
        ],
    )
    _write_csv(
        uncertainty / "model_ranking.csv",
        ["source", "split", "metric", "rank", "model_name", "lookback", "n_folds", "mean"],
        [
            {
                "source": "neural",
                "split": "test",
                "metric": "macro_f1",
                "rank": 1,
                "model_name": "matrix_transformer",
                "lookback": 20,
                "n_folds": 2,
                "mean": 0.73,
            }
        ],
    )

    _write_json(
        ablations / "summary.json",
        {
            "families_run": ["feature_groups", "model_class", "horizon"],
            "families_skipped": ["lookback"],
            "checkpoints_written": False,
        },
    )
    ablation_headers = [
        "ablation_family",
        "ablation_name",
        "fold_id",
        "model_name",
        "split",
        "metric_name",
        "metric_value",
        "baseline_value",
        "delta_vs_baseline",
        "status",
        "skip_reason",
    ]
    for filename, family in (
        ("model_class_ablation.csv", "model_class"),
        ("feature_group_ablation.csv", "feature_groups"),
        ("horizon_ablation.csv", "horizon"),
    ):
        _write_csv(
            ablations / filename,
            ablation_headers,
            [
                {
                    "ablation_family": family,
                    "ablation_name": "baseline",
                    "fold_id": "fold_1",
                    "model_name": "logistic",
                    "split": "test",
                    "metric_name": "macro_f1",
                    "metric_value": 0.33,
                    "baseline_value": 0.33,
                    "delta_vs_baseline": 0.0,
                    "status": "ok",
                    "skip_reason": "",
                }
            ],
        )
    _write_csv(
        ablations / "calibration_threshold_ablation.csv",
        ablation_headers,
        [
            {
                "ablation_family": "calibration",
                "ablation_name": "reliability_ece",
                "fold_id": "fold_1",
                "model_name": "logistic",
                "split": "test",
                "metric_name": "ece",
                "metric_value": 0.02,
                "baseline_value": 0.0,
                "delta_vs_baseline": 0.02,
                "status": "ok",
                "skip_reason": "",
            }
        ],
    )
    _write_csv(
        ablations / "execution_cost_latency_ablation.csv",
        ablation_headers,
        [
            {
                "ablation_family": "execution",
                "ablation_name": "cost_0_latency_0",
                "fold_id": "fold_1",
                "model_name": "logistic",
                "split": "test",
                "metric_name": "net_signal_return_proxy",
                "metric_value": 1.0,
                "baseline_value": 1.0,
                "delta_vs_baseline": 0.0,
                "status": "ok",
                "skip_reason": "",
            }
        ],
    )
    _write_json(
        ablations / "skipped_ablations.json",
        {
            "skipped": [
                {
                    "ablation_name": "lookback_sweep",
                    "skip_reason": "not requested for this fixture",
                }
            ]
        },
    )

    _write_json(
        execution / "summary.json",
        {
            "proxy_diagnostics": True,
            "fill_assumption": "full_fill_at_mid_no_queue",
            "checkpoints_required": False,
            "full_predictions_required": False,
        },
    )
    _write_csv(
        execution / "degradation_summary.csv",
        [
            "model_name",
            "source",
            "statistical_value",
            "base_proxy_metric",
            "base_proxy_value",
            "exec_proxy_metric",
            "exec_proxy_value",
            "relative_degradation_proxy",
            "status",
            "skip_reason",
        ],
        [
            {
                "model_name": "gradient_boosting",
                "source": "classical",
                "statistical_value": 0.46,
                "base_proxy_metric": "gross_signal_return_proxy",
                "base_proxy_value": 2.0,
                "exec_proxy_metric": "net_signal_return_proxy",
                "exec_proxy_value": 1.2,
                "relative_degradation_proxy": 0.4,
                "status": "ok",
                "skip_reason": "",
            },
            {
                "model_name": "matrix_transformer",
                "source": "neural",
                "statistical_value": 0.73,
                "base_proxy_metric": "gross_signal_return_proxy",
                "base_proxy_value": "",
                "exec_proxy_metric": "net_signal_return_proxy",
                "exec_proxy_value": "",
                "relative_degradation_proxy": "",
                "status": "skipped",
                "skip_reason": "no stored execution proxy rows",
            },
        ],
    )
    for filename in (
        "confidence_threshold_summary.csv",
        "turnover_summary.csv",
        "fill_assumption_summary.csv",
        "adverse_selection_summary.csv",
    ):
        _write_csv(execution / filename, ["model_name", "status"], [])
    _write_json(
        execution / "skipped_diagnostics.json",
        {
            "skipped": [
                {
                    "diagnostic": "degradation",
                    "scope": "matrix_transformer",
                    "skip_reason": "no stored execution proxy rows",
                }
            ]
        },
    )

    _write_json(
        external / "benchmark_context.json",
        {
            "scope": "protocol comparison only; no external numeric metrics",
            "chronoslob_protocol": {
                "dataset": "FI-2010",
                "variant": "NoAuction ZScore",
                "folds": ["fold_1", "fold_2"],
                "horizon": "label_10",
                "split_protocol": "official split column; validation carved from train only",
            },
            "external_references": [
                {
                    "source_name": "Reference paper",
                    "source_type": "paper",
                    "metric_values_included": False,
                }
            ],
        },
    )
    _write_csv(
        external / "protocol_comparison.csv",
        ["source_name", "source_type", "caveat"],
        [{"source_name": "Reference paper", "source_type": "paper", "caveat": "protocol only"}],
    )

    return {
        "classical": classical,
        "neural": neural,
        "uncertainty": uncertainty,
        "ablations": ablations,
        "execution": execution,
        "external": external,
    }


def test_final_report_builder_writes_markdown_and_summary(
    tiny_final_artefacts: dict[str, Path],
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "final_report.md"
    summary = build_final_empirical_report(
        classical_dir=tiny_final_artefacts["classical"],
        neural_dir=tiny_final_artefacts["neural"],
        uncertainty_dir=tiny_final_artefacts["uncertainty"],
        ablation_dir=tiny_final_artefacts["ablations"],
        execution_dir=tiny_final_artefacts["execution"],
        external_dir=tiny_final_artefacts["external"],
        out_path=report_path,
        overwrite=True,
    )

    assert report_path.is_file()
    assert report_path.with_name("final_report_summary.json").is_file()
    assert summary.headline_metrics["best_classical"]["model_name"] == "gradient_boosting"
    assert summary.headline_metrics["best_neural"]["model_name"] == "matrix_transformer"
    assert summary.input_file_hashes


def test_summary_json_contains_traceability(
    tiny_final_artefacts: dict[str, Path],
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "trace_report.md"
    build_final_empirical_report(
        classical_dir=tiny_final_artefacts["classical"],
        neural_dir=tiny_final_artefacts["neural"],
        uncertainty_dir=tiny_final_artefacts["uncertainty"],
        ablation_dir=tiny_final_artefacts["ablations"],
        execution_dir=tiny_final_artefacts["execution"],
        external_dir=tiny_final_artefacts["external"],
        out_path=report_path,
        overwrite=True,
    )

    payload = json.loads(report_path.with_name("trace_report_summary.json").read_text())
    assert "generated_at" in payload
    assert payload["report_path"].endswith("trace_report.md")
    assert payload["input_artefact_paths"]["classical_summary"].endswith("summary.json")
    assert payload["input_file_hashes"]
    assert payload["headline_metrics"]["best_neural"]["seed_count"] == 1


def test_missing_optional_input_is_marked_skipped(
    tiny_final_artefacts: dict[str, Path],
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "missing_optional.md"
    summary = build_final_empirical_report(
        classical_dir=tiny_final_artefacts["classical"],
        neural_dir=tiny_final_artefacts["neural"],
        uncertainty_dir=tiny_final_artefacts["uncertainty"],
        ablation_dir=tiny_final_artefacts["ablations"],
        execution_dir=tiny_final_artefacts["execution"],
        external_dir=tmp_path / "missing_external",
        out_path=report_path,
        overwrite=True,
    )

    text = report_path.read_text(encoding="utf-8")
    assert "## External Benchmark Context" in text
    assert "Skipped: external context artefacts" in text
    assert summary.missing_sections
    assert summary.warnings


def test_missing_required_input_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="classical artefact directory"):
        build_final_empirical_report(
            classical_dir=tmp_path / "missing_classical",
            neural_dir=tmp_path / "missing_neural",
            uncertainty_dir=tmp_path / "missing_uncertainty",
            out_path=tmp_path / "report.md",
        )


def test_report_contains_key_sections_and_paths(
    tiny_final_artefacts: dict[str, Path],
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "sections.md"
    build_final_empirical_report(
        classical_dir=tiny_final_artefacts["classical"],
        neural_dir=tiny_final_artefacts["neural"],
        uncertainty_dir=tiny_final_artefacts["uncertainty"],
        ablation_dir=tiny_final_artefacts["ablations"],
        execution_dir=tiny_final_artefacts["execution"],
        external_dir=tiny_final_artefacts["external"],
        out_path=report_path,
        overwrite=True,
    )

    text = report_path.read_text(encoding="utf-8")
    for section in (
        "## Evidence Snapshot",
        "## Main Result Table",
        "## Uncertainty Summary",
        "## Artefact Traceability",
        "## Reproduction Commands",
    ):
        assert section in text
    assert "multi-fold" in text
    assert "reduced-scope, single-seed" in text
    assert "proxy diagnostics" in text
    assert "protocol context, not ranking claims" in text
    assert (tiny_final_artefacts["classical"] / "results_summary.csv").as_posix() in text


def test_report_does_not_add_unsupported_claims(
    tiny_final_artefacts: dict[str, Path],
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "claims.md"
    build_final_empirical_report(
        classical_dir=tiny_final_artefacts["classical"],
        neural_dir=tiny_final_artefacts["neural"],
        uncertainty_dir=tiny_final_artefacts["uncertainty"],
        ablation_dir=tiny_final_artefacts["ablations"],
        execution_dir=tiny_final_artefacts["execution"],
        external_dir=tiny_final_artefacts["external"],
        out_path=report_path,
        overwrite=True,
    )

    lowered = report_path.read_text(encoding="utf-8").lower()
    for phrase in (
        _phrase("profitable", "strategy"),
        _phrase("guaranteed", "alpha"),
        _phrase("market", "beating"),
        _phrase("high", "sharpe"),
        _phrase("production", "trading"),
        _phrase("live", "trading"),
        _phrase("state", "of", "the", "art"),
    ):
        assert phrase not in lowered


def test_report_includes_actual_feature_ablation_stability_scope(
    tiny_final_artefacts: dict[str, Path],
    tmp_path: Path,
) -> None:
    feature = tmp_path / "feature_ablations"
    analysis = tmp_path / "feature_ablation_analysis"
    _write_json(
        feature / "summary.json",
        {
            "runner_version": "test",
            "smoke_test": False,
            "completed_run_count": 2520,
            "failed_run_count": 0,
            "folds": ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"],
            "horizons": [10, 20, 50],
            "seeds": [0, 1, 2],
            "models": ["logistic", "ridge"],
            "feature_groups": ["snapshot_order_flow_proxy", "spread"],
            "proxy_groups": ["snapshot_order_flow_proxy"],
            "unsupported_groups": ["true_order_flow_imbalance", "queue_position"],
            "artefact_policy": {"raw_predictions_saved": False},
        },
    )
    _write_csv(feature / "results_summary.csv", ["status"], [{"status": "completed"}])
    _write_csv(
        feature / "aggregate_summary.csv",
        ["horizon", "model", "ablation_mode", "feature_group", "completed_runs"],
        [
            {
                "horizon": 20,
                "model": "logistic",
                "ablation_mode": "remove_one_group",
                "feature_group": "snapshot_order_flow_proxy",
                "completed_runs": 15,
            }
        ],
    )
    _write_csv(
        feature / "feature_delta_summary.csv",
        ["horizon", "model", "ablation_mode", "feature_group", "delta_macro_f1", "delta_mcc"],
        [
            {
                "horizon": 20,
                "model": "logistic",
                "ablation_mode": "remove_one_group",
                "feature_group": "snapshot_order_flow_proxy",
                "delta_macro_f1": -0.01,
                "delta_mcc": -0.02,
            }
        ],
    )
    _write_json(feature / "ablation_manifest.json", {"created_at": "2026-05-30T00:00:00Z"})
    _write_json(feature / "failures.json", {"failure_count": 0, "failures": []})

    _write_json(
        analysis / "summary.json",
        {"evidence_status": "partial_real", "raw_predictions_available": False},
    )
    _write_csv(
        analysis / "feature_group_stability.csv",
        [
            "feature_group",
            "mean_delta_macro_f1",
            "mean_delta_mcc",
            "macro_f1_degradation_fraction",
            "stability_score",
        ],
        [
            {
                "feature_group": "snapshot_order_flow_proxy",
                "mean_delta_macro_f1": -0.01,
                "mean_delta_mcc": -0.02,
                "macro_f1_degradation_fraction": 1.0,
                "stability_score": 1.0,
            }
        ],
    )
    for filename in (
        "feature_delta_by_horizon.csv",
        "feature_delta_by_model.csv",
        "feature_delta_by_fold.csv",
        "feature_delta_by_seed.csv",
    ):
        _write_csv(
            analysis / filename,
            ["feature_group", "mean_delta_macro_f1"],
            [{"feature_group": "snapshot_order_flow_proxy", "mean_delta_macro_f1": -0.01}],
        )
    _write_csv(
        analysis / "snapshot_order_flow_proxy_scope.csv",
        ["macro_f1_degraded_when_removed"],
        [{"macro_f1_degraded_when_removed": "true"}],
    )
    _write_json(
        analysis / "feature_claim_assessment.json",
        {
            "claims": {
                "broader_horizon_snapshot_proxy_importance": {
                    "status": "supported",
                    "reason": "horizon 20/50 rows support it",
                },
                "nonlinear_model_feature_stability": {
                    "status": "needs_real_evidence",
                    "reason": "non-linear slice absent",
                },
                "causal_feature_importance": {
                    "status": "forbidden",
                    "reason": "not causal",
                },
                "true_event_level_ofi": {
                    "status": "forbidden",
                    "reason": "not event-level",
                },
            }
        },
    )
    _write_json(analysis / "figure_manifest.json", {"figures": []})
    (analysis / "feature_ablation_analysis.md").write_text("analysis", encoding="utf-8")

    report_path = tmp_path / "feature_scope.md"
    build_final_empirical_report(
        classical_dir=tiny_final_artefacts["classical"],
        neural_dir=tiny_final_artefacts["neural"],
        uncertainty_dir=tiny_final_artefacts["uncertainty"],
        feature_ablation_dir=feature,
        feature_ablation_analysis_dir=analysis,
        out_path=report_path,
        overwrite=True,
    )

    text = report_path.read_text(encoding="utf-8")
    assert "## Feature Ablation and Stability Analysis" in text
    assert "partial_real" in text
    assert "10, 20, 50" in text
    assert "not causal feature importance" in text
    assert "labelled snapshot proxy" in text
    assert "Execution-aware ablation diagnostics require retained prediction-level outputs" in text


def test_overwrite_protection(
    tiny_final_artefacts: dict[str, Path],
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "overwrite.md"
    build_final_empirical_report(
        classical_dir=tiny_final_artefacts["classical"],
        neural_dir=tiny_final_artefacts["neural"],
        uncertainty_dir=tiny_final_artefacts["uncertainty"],
        out_path=report_path,
        overwrite=True,
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_final_empirical_report(
            classical_dir=tiny_final_artefacts["classical"],
            neural_dir=tiny_final_artefacts["neural"],
            uncertainty_dir=tiny_final_artefacts["uncertainty"],
            out_path=report_path,
            overwrite=False,
        )


def test_cli_build_final_empirical_report(
    tiny_final_artefacts: dict[str, Path],
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "cli_final.md"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chronoslob.cli",
            "build-final-empirical-report",
            "--classical",
            str(tiny_final_artefacts["classical"]),
            "--neural",
            str(tiny_final_artefacts["neural"]),
            "--uncertainty",
            str(tiny_final_artefacts["uncertainty"]),
            "--ablations",
            str(tiny_final_artefacts["ablations"]),
            "--execution",
            str(tiny_final_artefacts["execution"]),
            "--external",
            str(tiny_final_artefacts["external"]),
            "--out",
            str(report_path),
            "--overwrite",
        ],
        cwd=project_root(),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ChronosLOB final empirical report builder" in completed.stdout
    assert report_path.is_file()
    assert report_path.with_name("cli_final_summary.json").is_file()


def test_final_report_consumes_evidence_pack_summary(
    tiny_final_artefacts: dict[str, Path],
    tmp_path: Path,
) -> None:
    evidence_pack = tmp_path / "evidence_pack"
    _write_json(
        evidence_pack / "evidence_pack_manifest.json",
        {
            "artefact_status_counts": {"complete_real": 2},
            "claim_boundary": "stored artefact audit",
        },
    )
    _write_json(
        evidence_pack / "claim_audit.json",
        {
            "claim_status_counts": {"supported": 1, "forbidden": 1},
            "claims": [],
        },
    )
    (evidence_pack / "supported_claims.md").write_text(
        "# Supported Claims\n\n- ChronosLOB includes claim audit support.\n",
        encoding="utf-8",
    )
    (evidence_pack / "unsupported_claims.md").write_text(
        "# Unsupported Claims\n\n- Stronger result claims remain unsupported.\n",
        encoding="utf-8",
    )

    report_path = tmp_path / "with_evidence_pack.md"
    build_final_empirical_report(
        classical_dir=tiny_final_artefacts["classical"],
        neural_dir=tiny_final_artefacts["neural"],
        uncertainty_dir=tiny_final_artefacts["uncertainty"],
        ablation_dir=tiny_final_artefacts["ablations"],
        execution_dir=tiny_final_artefacts["execution"],
        external_dir=tiny_final_artefacts["external"],
        evidence_pack_dir=evidence_pack,
        out_path=report_path,
        overwrite=True,
    )

    text = report_path.read_text(encoding="utf-8")
    assert "## Evidence Pack Audit" in text
    assert "supported=1" in text
    assert "Release caveats from the evidence pack" in text
