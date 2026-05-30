from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from chronoslob.experiments.evidence_pack import (
    EvidencePackConfig,
    EvidencePackError,
    audit_claims,
    build_evidence_pack,
    discover_artefacts,
)
from chronoslob.experiments.manifests import sha256_file


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


def _touch_report(path: Path, *, forbidden: bool = False) -> None:
    text = "# Final Report\n\nStored artefact summary only.\n"
    if forbidden:
        text += "\nThis is a state-of-the-art system.\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    _write_json(
        path.with_name(f"{path.stem}_summary.json"),
        {"created_at": "2026-05-27T00:00:00Z", "input_file_hashes": {}},
    )


def _minimal_config(tmp_path: Path) -> EvidencePackConfig:
    return EvidencePackConfig(
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
        project_audit_dir=None,
        strict=False,
        allow_smoke_test=True,
        overwrite=True,
    )


def _write_complete_classical(path: Path) -> None:
    _write_json(
        path / "summary.json",
        {
            "created_at": "2026-05-27T00:00:00Z",
            "failure_count": 0,
            "result_rows": 2,
            "smoke_test": False,
        },
    )
    _write_csv(
        path / "results_summary.csv",
        ["model_name", "split", "macro_f1_mean"],
        [
            {"model_name": "logistic", "split": "test", "macro_f1_mean": 0.4},
            {"model_name": "gradient_boosting", "split": "test", "macro_f1_mean": 0.6},
        ],
    )


def _write_complete_grid(path: Path, *, smoke: bool = False, ssl_delta: float = 0.05) -> None:
    _write_json(
        path / "summary.json",
        {
            "created_at": "2026-05-27T00:00:00Z",
            "smoke_test": smoke,
            "completed_run_count": 3,
            "failed_run_count": 0,
            "core_grid_complete": not smoke,
            "evidence_level": "smoke_test_only" if smoke else "full_grid_complete",
            "objectives": ["supervised", "masked_reconstruction", "next_field"],
        },
    )
    _write_csv(
        path / "results_summary.csv",
        ["model_family", "split", "mean_macro_f1"],
        [{"model_family": "matrix_transformer", "split": "test", "mean_macro_f1": 0.7}],
    )
    _write_csv(
        path / "aggregate_summary.csv",
        ["model_family", "pretraining_objective", "mean_macro_f1"],
        [
            {
                "model_family": "matrix_transformer",
                "pretraining_objective": "none",
                "mean_macro_f1": 0.7,
            }
        ],
    )
    _write_json(
        path / "aggregate_summary.json",
        {"created_at": "2026-05-27T00:00:00Z", "completed_run_count": 3},
    )
    _write_csv(
        path / "ssl_comparison.csv",
        ["status", "delta_macro_f1", "delta_ece"],
        [{"status": "ok", "delta_macro_f1": ssl_delta, "delta_ece": -0.01}],
    )
    _write_csv(path / "failures.csv", ["status"], [])


def _write_complete_ssl(path: Path) -> None:
    _write_json(
        path / "summary.json",
        {
            "created_at": "2026-05-27T00:00:00Z",
            "execution_mode": "benchmark",
            "completed_run_count": 1,
            "failure_count": 0,
        },
    )
    _write_csv(path / "results_summary.csv", ["status"], [{"status": "completed"}])
    _write_csv(path / "comparison_summary.csv", ["status"], [{"status": "ok"}])


def _write_proper_training(
    path: Path,
    *,
    evidence_level: str,
    target_scope_complete: bool,
    completed_run_count: int,
) -> None:
    _write_json(
        path / "summary.json",
        {
            "created_at": "2026-05-29T00:00:00Z",
            "smoke_test": False,
            "execution_mode": "benchmark",
            "evidence_level": evidence_level,
            "target_scope_complete": target_scope_complete,
            "completed_run_count": completed_run_count,
            "failed_run_count": 0,
            "folds": [1, 2, 3],
            "horizons": [10, 50],
            "seeds": [0],
            "lookbacks": [50],
            "objectives": ["supervised", "masked_reconstruction", "next_field"],
            "max_epochs": 25,
            "early_stopping_patience": 5,
        },
    )
    _write_json(path / "config_snapshot.json", {"created_at": "2026-05-29T00:00:00Z"})
    _write_csv(path / "results_summary.csv", ["status"], [{"status": "completed"}])
    _write_csv(path / "aggregate_summary.csv", ["status"], [{"status": "completed"}])
    _write_json(
        path / "aggregate_summary.json",
        {"created_at": "2026-05-29T00:00:00Z", "completed_run_count": completed_run_count},
    )
    _write_csv(path / "training_curves_summary.csv", ["status"], [{"status": "completed"}])
    _write_csv(path / "ssl_comparison.csv", ["status"], [{"status": "matched"}])
    _write_csv(path / "failures.csv", ["status"], [])
    _write_json(path / "sha256_manifest.json", {"sha256": {}})


def _write_complete_figures(path: Path) -> None:
    _write_json(
        path / "figure_manifest.json",
        {
            "created_at": "2026-05-27T00:00:00Z",
            "smoke_test": False,
            "figures": [{"figure_id": "f1", "status": "completed", "smoke_test": False}],
        },
    )


def _write_complete_execution(path: Path) -> None:
    _write_json(
        path / "summary.json",
        {
            "created_at": "2026-05-27T00:00:00Z",
            "smoke_test": False,
            "diagnostics_produced": ["confidence_threshold"],
        },
    )
    _write_json(
        path / "execution_v3_manifest.json",
        {
            "created_at": "2026-05-27T00:00:00Z",
            "smoke_test": False,
            "output_file_hashes": {},
        },
    )
    _write_csv(
        path / "confidence_threshold_aggregate.csv",
        [
            "model_family",
            "pretraining_objective",
            "horizon",
            "threshold",
            "status",
            "mean_net_cost_adjusted_proxy",
        ],
        [
            {
                "model_family": "matrix_transformer",
                "pretraining_objective": "none",
                "horizon": 10,
                "threshold": 0.0,
                "status": "ok",
                "mean_net_cost_adjusted_proxy": 1.0,
            },
            {
                "model_family": "matrix_transformer",
                "pretraining_objective": "none",
                "horizon": 10,
                "threshold": 0.7,
                "status": "ok",
                "mean_net_cost_adjusted_proxy": 1.5,
            },
        ],
    )


def _write_complete_feature_ablations(path: Path) -> None:
    _write_json(
        path / "summary.json",
        {
            "created_at": "2026-05-27T00:00:00Z",
            "smoke_test": False,
            "completed_run_count": 5040,
            "failed_run_count": 0,
            "folds": ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"],
            "horizons": [10, 20, 50],
            "seeds": [0, 1, 2],
            "models": ["logistic", "ridge", "elastic_net", "gradient_boosting"],
        },
    )
    _write_csv(path / "results_summary.csv", ["status"], [{"status": "completed"}])
    _write_csv(path / "aggregate_summary.csv", ["status"], [{"status": "completed"}])
    _write_csv(
        path / "feature_delta_summary.csv",
        ["feature_group", "delta_macro_f1"],
        [{"feature_group": "spread", "delta_macro_f1": 0.02}],
    )
    _write_json(path / "ablation_manifest.json", {"created_at": "2026-05-27T00:00:00Z"})
    _write_json(path / "failures.json", {"failure_count": 0, "failures": []})


def _write_complete_feature_ablation_analysis(path: Path) -> None:
    _write_json(
        path / "summary.json",
        {
            "evidence_status": "partial_real",
            "completed_run_count": 2520,
            "failed_run_count": 0,
            "folds": ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"],
            "horizons": [10, 20, 50],
            "seeds": [0, 1, 2],
            "models": ["logistic", "ridge"],
            "raw_predictions_available": False,
        },
    )
    _write_csv(
        path / "feature_group_stability.csv",
        ["feature_group", "mean_delta_macro_f1", "stability_score"],
        [
            {
                "feature_group": "snapshot_order_flow_proxy",
                "mean_delta_macro_f1": -0.01,
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
            path / filename,
            ["feature_group", "mean_delta_macro_f1"],
            [{"feature_group": "snapshot_order_flow_proxy", "mean_delta_macro_f1": -0.01}],
        )
    _write_csv(
        path / "snapshot_order_flow_proxy_scope.csv",
        ["feature_group", "horizon", "model", "macro_f1_degraded_when_removed"],
        [
            {
                "feature_group": "snapshot_order_flow_proxy",
                "horizon": 20,
                "model": "logistic",
                "macro_f1_degraded_when_removed": "true",
            }
        ],
    )
    _write_json(
        path / "feature_claim_assessment.json",
        {
            "claims": {
                "horizon10_logistic_ridge_snapshot_proxy_importance": {
                    "status": "supported",
                    "reason": "retained h10 rows support the proxy finding",
                },
                "broader_horizon_snapshot_proxy_importance": {
                    "status": "supported",
                    "reason": "horizon 20/50 rows support the proxy finding",
                },
                "nonlinear_model_feature_stability": {
                    "status": "needs_real_evidence",
                    "reason": "non-linear slice absent",
                },
            }
        },
    )
    _write_json(path / "figure_manifest.json", {"figures": []})
    (path / "feature_ablation_analysis.md").parent.mkdir(parents=True, exist_ok=True)
    (path / "feature_ablation_analysis.md").write_text(
        "snapshot_order_flow_proxy is a labelled snapshot proxy.",
        encoding="utf-8",
    )


def _write_all_complete(config: EvidencePackConfig) -> None:
    _write_complete_classical(config.classical_dir)
    _write_complete_ssl(config.ssl_dir)
    _write_complete_grid(config.neural_full_grid_dir)
    _write_complete_figures(config.figures_dir)
    _write_complete_execution(config.execution_v3_dir)
    _write_complete_feature_ablations(config.feature_ablations_dir)
    _write_complete_feature_ablation_analysis(config.feature_ablation_analysis_dir)
    _write_complete_figures(config.ablation_figures_dir)
    _touch_report(config.final_report_path)


def _record(records: list[Any], name: str) -> Any:
    return next(record for record in records if record.artefact_name == name)


def test_artefact_discovery_classifies_missing(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    records = discover_artefacts(config)

    assert _record(records, "fi2010_neural_full_grid").status == "missing"


def test_smoke_test_only_classification(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    _write_complete_grid(config.neural_full_grid_dir, smoke=True)

    records = discover_artefacts(config)

    grid = _record(records, "fi2010_neural_full_grid")
    assert grid.status == "smoke_test_only"
    assert grid.smoke_test is True


def test_complete_real_classification(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    _write_complete_grid(config.neural_full_grid_dir)

    records = discover_artefacts(config)

    assert _record(records, "fi2010_neural_full_grid").status == "complete_real"


def test_partial_real_classification(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    _write_json(
        config.neural_full_grid_dir / "summary.json",
        {
            "created_at": "2026-05-27T00:00:00Z",
            "smoke_test": False,
            "completed_run_count": 1,
            "failed_run_count": 1,
        },
    )
    _write_csv(config.neural_full_grid_dir / "results_summary.csv", ["status"], [])

    records = discover_artefacts(config)

    assert _record(records, "fi2010_neural_full_grid").status == "partial_real"


def test_proper_training_status_depends_on_target_scope(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    _write_proper_training(
        config.proper_training_dir,
        evidence_level="partial_real",
        target_scope_complete=False,
        completed_run_count=18,
    )

    records = discover_artefacts(config)

    assert _record(records, "fi2010_neural_proper_training_subset").status == "partial_real"

    complete_config = _minimal_config(tmp_path / "complete")
    _write_proper_training(
        complete_config.proper_training_dir,
        evidence_level="complete_real",
        target_scope_complete=True,
        completed_run_count=30,
    )

    complete_records = discover_artefacts(complete_config)

    assert (
        _record(complete_records, "fi2010_neural_proper_training_subset").status
        == "complete_real"
    )


def test_invalid_artefact_classification(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    config.neural_full_grid_dir.mkdir(parents=True)
    (config.neural_full_grid_dir / "summary.json").write_text("{bad json", encoding="utf-8")

    records = discover_artefacts(config)

    assert _record(records, "fi2010_neural_full_grid").status == "invalid"


def test_stale_artefact_detection_from_hash(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    source = tmp_path / "source.csv"
    source.write_text("old\n", encoding="utf-8")
    source_hash = sha256_file(source)
    _write_complete_grid(config.neural_full_grid_dir)
    _write_json(
        config.neural_full_grid_dir / "summary.json",
        {
            "created_at": "2026-05-27T00:00:00Z",
            "smoke_test": False,
            "completed_run_count": 3,
            "failed_run_count": 0,
            "core_grid_complete": True,
            "input_file_hashes": {str(source): source_hash},
        },
    )
    source.write_text("new\n", encoding="utf-8")

    records = discover_artefacts(config)

    assert _record(records, "fi2010_neural_full_grid").status == "stale"


def test_claim_audit_supported_smoke_unsupported_and_forbidden(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    _write_all_complete(config)
    records = discover_artefacts(config)
    claims = audit_claims(records)
    by_id = {claim.claim_id: claim for claim in claims}

    assert by_id["empirical.gradient_boosting_best"].status == "supported"
    assert by_id["empirical.ssl_improved_macro_f1"].status == "supported"
    assert by_id["feature_ablation_infrastructure"].status == "supported"
    assert by_id["horizon10_logistic_ridge_snapshot_proxy_importance"].status == "supported"
    assert by_id["broader_horizon_snapshot_proxy_importance"].status == "supported"
    assert by_id["nonlinear_model_feature_stability"].status == "needs_real_evidence"
    assert by_id["causal_feature_importance"].status == "forbidden"
    assert by_id["true_event_level_ofi"].status == "forbidden"
    assert by_id["forbidden.1"].status == "forbidden"
    assert by_id["forbidden.1"].safe_rewording

    smoke_config = _minimal_config(tmp_path / "smoke")
    _write_complete_grid(smoke_config.neural_full_grid_dir, smoke=True)
    smoke_claims = {
        claim.claim_id: claim for claim in audit_claims(discover_artefacts(smoke_config))
    }
    assert smoke_claims["empirical.ssl_improved_macro_f1"].status == "smoke_only"

    missing_records = discover_artefacts(_minimal_config(tmp_path / "missing"))
    missing_claims = {claim.claim_id: claim for claim in audit_claims(missing_records)}
    assert missing_claims["empirical.ssl_improved_macro_f1"].status == "needs_real_evidence"


def test_build_outputs_are_conservative_and_complete(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    _write_all_complete(config)

    result = build_evidence_pack(config)

    output_names = {path.name for path in result.files_written}
    assert "artefact_inventory.csv" in output_names
    assert "claim_audit.json" in output_names
    assert "readme_result_snapshot.md" in output_names
    assert "reproduction_commands.md" in output_names
    assert "release_checklist.md" in output_names

    snapshot = (config.out_dir / "readme_result_snapshot.md").read_text(encoding="utf-8")
    assert "no broad SSL improvement claim" not in snapshot
    assert "offline proxy diagnostics" in snapshot

    conservative = (config.out_dir / "public_bullets_conservative.md").read_text(
        encoding="utf-8"
    )
    assert "profitable" not in conservative.lower()
    assert "state-of-the-art" not in conservative.lower()

    strong = (config.out_dir / "public_bullets_strong_if_supported.md").read_text(
        encoding="utf-8"
    )
    assert "supported only if" in strong
    assert "Safe fallback" in strong

    commands = (config.out_dir / "reproduction_commands.md").read_text(encoding="utf-8")
    assert "Smoke-test version" in commands
    assert "Real-run version" in commands

    checklist = (config.out_dir / "release_checklist.md").read_text(encoding="utf-8")
    assert "Git Hygiene" in checklist
    assert "Unsupported claims removed" in checklist
    assert "Do not automatically delete or stage" in checklist


def test_strict_mode_fails_on_invalid_artefact(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    config = EvidencePackConfig(**{**config.__dict__, "strict": True})
    config.neural_full_grid_dir.mkdir(parents=True)
    (config.neural_full_grid_dir / "summary.json").write_text("{bad", encoding="utf-8")

    with pytest.raises(EvidencePackError, match="invalid artefact"):
        build_evidence_pack(config)


def test_strict_and_non_strict_forbidden_public_claims(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    _write_all_complete(config)
    _touch_report(config.final_report_path, forbidden=True)
    strict_config = EvidencePackConfig(**{**config.__dict__, "strict": True})

    with pytest.raises(EvidencePackError, match="forbidden public claim"):
        build_evidence_pack(strict_config)

    non_strict = EvidencePackConfig(
        **{**config.__dict__, "strict": False, "out_dir": tmp_path / "pack2"}
    )
    result = build_evidence_pack(non_strict)

    assert any("forbidden public claim" in warning for warning in result.warnings)


def test_stale_detection_from_newer_input_timestamp(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    source = tmp_path / "input.txt"
    source.write_text("source\n", encoding="utf-8")
    source_hash = sha256_file(source)
    _write_complete_grid(config.neural_full_grid_dir)
    _write_json(
        config.neural_full_grid_dir / "summary.json",
        {
            "created_at": "2026-05-27T00:00:00Z",
            "smoke_test": False,
            "completed_run_count": 3,
            "failed_run_count": 0,
            "core_grid_complete": True,
            "input_file_hashes": {str(source): source_hash},
        },
    )
    old_time = time.time() - 100
    os.utime(config.neural_full_grid_dir / "summary.json", (old_time, old_time))
    new_time = time.time()
    os.utime(source, (new_time, new_time))

    records = discover_artefacts(config)

    assert _record(records, "fi2010_neural_full_grid").status == "stale"


_GIT_COMMIT_PATH = "chronoslob.experiments.evidence_pack._current_git_commit"


def _set_summary_field(path: Path, **fields: Any) -> None:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    summary.update(fields)
    _write_json(path / "summary.json", summary)


def test_older_commit_summary_is_archived_not_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_GIT_COMMIT_PATH, lambda: "f" * 40)
    config = _minimal_config(tmp_path)
    _write_complete_grid(config.neural_full_grid_dir)
    _set_summary_field(config.neural_full_grid_dir, git_commit="a" * 40)

    grid = _record(discover_artefacts(config), "fi2010_neural_full_grid")

    assert grid.status == "archived_valid"
    assert grid.freshness == "archived"
    assert "older" in grid.notes.lower()


def test_deleted_raw_prediction_hashes_are_archived_not_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_GIT_COMMIT_PATH, lambda: None)
    config = _minimal_config(tmp_path)
    _write_complete_grid(config.neural_full_grid_dir)
    _set_summary_field(
        config.neural_full_grid_dir,
        input_file_hashes={
            "experiments/fi2010_neural_full_grid/runs/fold_1/predictions.csv": "a" * 64,
        },
    )

    grid = _record(discover_artefacts(config), "fi2010_neural_full_grid")

    assert grid.status == "archived_valid"
    assert grid.freshness == "archived"
    assert "intentionally removed" in grid.notes.lower()


def test_missing_non_heavy_hashed_input_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_GIT_COMMIT_PATH, lambda: None)
    config = _minimal_config(tmp_path)
    _write_complete_grid(config.neural_full_grid_dir)
    _set_summary_field(
        config.neural_full_grid_dir,
        input_file_hashes={str(tmp_path / "source.csv"): "a" * 64},
    )

    grid = _record(discover_artefacts(config), "fi2010_neural_full_grid")

    assert grid.status == "stale"
    assert "hash path is missing" in grid.notes.lower()


def test_changed_retained_content_is_still_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_GIT_COMMIT_PATH, lambda: None)
    config = _minimal_config(tmp_path)
    retained = tmp_path / "retained.csv"
    retained.write_text("old\n", encoding="utf-8")
    _write_complete_grid(config.neural_full_grid_dir)
    _set_summary_field(
        config.neural_full_grid_dir,
        input_file_hashes={str(retained): sha256_file(retained)},
    )
    retained.write_text("changed\n", encoding="utf-8")

    grid = _record(discover_artefacts(config), "fi2010_neural_full_grid")

    assert grid.status == "stale"


def test_missing_legacy_ssl_runner_is_obsolete_superseded(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)

    record = _record(discover_artefacts(config), "fi2010_ssl_runner_outputs")

    assert record.status == "obsolete_superseded"
    assert record.freshness == "absent"


def test_missing_optional_feature_audit_is_optional_missing(tmp_path: Path) -> None:
    config = EvidencePackConfig(
        **{**_minimal_config(tmp_path).__dict__, "feature_audit_dir": tmp_path / "feature_audit"}
    )

    record = _record(discover_artefacts(config), "feature_registry_audit_outputs")

    assert record.status == "optional_missing"


def test_missing_required_artefact_still_missing(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)

    record = _record(discover_artefacts(config), "fi2010_classical_benchmarks")

    assert record.status == "missing"


def test_obsolete_ssl_runner_does_not_downgrade_train_only_ssl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_GIT_COMMIT_PATH, lambda: None)
    config = _minimal_config(tmp_path)
    _write_complete_grid(config.neural_full_grid_dir)

    claims = {claim.claim_id: claim for claim in audit_claims(discover_artefacts(config))}

    assert _record(discover_artefacts(config), "fi2010_ssl_runner_outputs").status == (
        "obsolete_superseded"
    )
    assert claims["general.train_only_ssl"].status == "supported"


def test_archived_grid_with_mixed_ssl_deltas_blocks_broad_improvement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_GIT_COMMIT_PATH, lambda: "f" * 40)
    config = _minimal_config(tmp_path)
    _write_complete_grid(config.neural_full_grid_dir)
    _set_summary_field(config.neural_full_grid_dir, git_commit="a" * 40)
    _write_csv(
        config.neural_full_grid_dir / "ssl_comparison.csv",
        ["status", "delta_macro_f1", "delta_ece"],
        [
            {"status": "ok", "delta_macro_f1": 0.02, "delta_ece": -0.01},
            {"status": "ok", "delta_macro_f1": -0.05, "delta_ece": 0.03},
        ],
    )

    records = discover_artefacts(config)
    by_id = {claim.claim_id: claim for claim in audit_claims(records)}

    assert _record(records, "fi2010_neural_full_grid").status == "archived_valid"
    assert by_id["empirical.ssl_improved_macro_f1"].status == "unsupported"
    assert by_id["empirical.ssl_improved_calibration"].status == "unsupported"


def test_forbidden_and_high_risk_claims_never_supported(tmp_path: Path) -> None:
    config = _minimal_config(tmp_path)
    _write_all_complete(config)

    claims = audit_claims(discover_artefacts(config))
    by_id = {claim.claim_id: claim for claim in claims}

    forbidden_ids = [cid for cid in by_id if cid.startswith("forbidden.")]
    assert forbidden_ids
    for cid in forbidden_ids:
        assert by_id[cid].status == "forbidden"
    for claim in claims:
        if claim.category == "forbidden or high-risk":
            assert claim.status == "forbidden"
        if claim.status == "supported":
            assert claim.category != "forbidden or high-risk"


def test_manifest_status_counts_match_inventory(tmp_path: Path) -> None:
    from collections import Counter

    config = _minimal_config(tmp_path)
    _write_all_complete(config)

    result = build_evidence_pack(config)
    manifest = json.loads(
        (config.out_dir / "evidence_pack_manifest.json").read_text(encoding="utf-8")
    )

    expected = dict(sorted(Counter(record.status for record in result.inventory).items()))
    assert manifest["artefact_status_counts"] == expected


def test_summary_has_archived_section_and_safe_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_GIT_COMMIT_PATH, lambda: "f" * 40)
    config = _minimal_config(tmp_path)
    _write_all_complete(config)
    _set_summary_field(config.neural_full_grid_dir, git_commit="a" * 40)

    build_evidence_pack(config)
    summary = (config.out_dir / "evidence_pack_summary.md").read_text(encoding="utf-8")

    assert "Status vocabulary" in summary
    assert "Archived or summary-valid artefacts" in summary
    assert "archived_valid" in summary
    for line in summary.splitlines():
        stripped = line.strip()
        if stripped.startswith(("|", "http")):
            continue
        assert len(line) <= 220


def test_real_repo_pack_has_no_spurious_stale_or_missing(tmp_path: Path) -> None:
    base = EvidencePackConfig()
    if not Path(base.classical_dir).exists():
        pytest.skip("real repository artefacts are not present")
    config = EvidencePackConfig(
        **{**base.__dict__, "out_dir": tmp_path / "pack", "strict": False, "overwrite": True}
    )

    records = discover_artefacts(config)
    by_name = {record.artefact_name: record for record in records}

    for name in (
        "fi2010_classical_benchmarks",
        "fi2010_neural_full_grid",
        "execution_v3_outputs",
    ):
        assert by_name[name].status in {"archived_valid", "complete_real"}, (
            name,
            by_name[name].status,
        )
    assert by_name["fi2010_ssl_runner_outputs"].status == "obsolete_superseded"
    assert by_name["feature_registry_audit_outputs"].status == "optional_missing"

    claims = {claim.claim_id: claim for claim in audit_claims(records)}
    for cid in (
        "general.reproducible_platform",
        "general.leakage_safe_fi2010",
        "general.supervised_vs_ssl_transformers",
        "general.train_only_ssl",
        "general.execution_proxy_diagnostics",
        "general.feature_ablations",
    ):
        assert claims[cid].status == "supported", (cid, claims[cid].status)
    assert claims["empirical.ssl_improved_macro_f1"].status == "unsupported"
    assert claims["empirical.ssl_improved_calibration"].status == "unsupported"
