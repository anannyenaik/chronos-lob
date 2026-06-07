"""Tests for the FI-2010 full supervised-vs-SSL neural evidence grid."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from chronoslob.cli import _run_fi2010_neural_full_grid_impl
from chronoslob.experiments.fi2010_neural_grid import (
    FI2010NeuralGridRunSpec,
    build_horizon_config_payload,
    build_ssl_comparison_rows,
    expand_fi2010_neural_grid,
    run_fi2010_neural_full_grid,
    write_horizon_config,
    write_neural_grid_aggregate_artifacts,
)
from chronoslob.experiments.neural_benchmarking import load_neural_benchmark_config
from chronoslob.utils.paths import project_root
from tests.test_fi2010_neural_runner import (
    _prepare_synthetic_processed_root,
    _write_tiny_neural_config,
)
from tests.test_fi2010_ssl_runner import _write_minimal_required_dirs

CONFIG_PATH = (
    project_root() / "configs" / "experiments" / "fi2010_neural_serious.yaml"
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _minimal_completed_metrics(spec: FI2010NeuralGridRunSpec) -> dict[str, Any]:
    return {
        "runner_version": "test",
        "run_id": spec.run_id,
        "status": "completed",
        "fold": spec.fold,
        "fold_id": spec.fold_id,
        "horizon": spec.horizon,
        "seed": spec.seed,
        "lookback": spec.lookback,
        "model_family": spec.model_family,
        "objective": spec.objective,
        "pretraining_objective": spec.pretraining_objective,
        "metrics": {
            "accuracy": 0.5,
            "macro_f1": 0.4,
            "mcc": 0.1,
            "ece": 0.2,
            "brier_score": 0.3,
            "nll": 1.2,
            "class_f1_down": 0.25,
            "class_f1_stationary": 0.5,
            "class_f1_up": 0.45,
        },
        "checkpoint_hash": "abc",
        "prediction_file": (
            "runs/fold_1/horizon_10/seed_0/lookback_20/supervised/predictions.csv"
        ),
        "architecture_hash": "arch",
        "preprocessing_hash": "prep",
    }


def _grid_row(
    *,
    objective: str,
    macro_f1: float,
    accuracy: float = 0.5,
    mcc: float = 0.1,
    ece: float = 0.2,
    arch: str = "arch",
    prep: str = "prep",
) -> dict[str, Any]:
    pretraining = "none" if objective == "supervised" else objective
    return {
        "fold": 1,
        "horizon": 10,
        "seed": 0,
        "lookback": 20,
        "model_family": "matrix_transformer",
        "pretraining_objective": pretraining,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "mcc": mcc,
        "ece": ece,
        "brier_score": 0.4,
        "nll": 1.0,
        "class_f1_down": 0.1,
        "class_f1_stationary": 0.2,
        "class_f1_up": 0.3,
        "checkpoint_hash": "hash",
        "prediction_file": f"runs/{objective}/predictions.csv",
        "status": "completed",
        "run_id": f"run_{objective}",
        "run_dir": f"runs/{objective}",
        "architecture_hash": arch,
        "preprocessing_hash": prep,
    }


def _write_full_grid_dir(base: Path, *, smoke: bool) -> Path:
    grid = base / "full_grid"
    grid.mkdir(parents=True, exist_ok=True)
    _write_json(
        grid / "summary.json",
        {
            "execution_mode": "smoke" if smoke else "benchmark",
            "smoke_test": smoke,
            "folds": [1],
            "horizons": [10],
            "seeds": [0],
            "lookbacks": [20],
            "objectives": ["supervised", "masked_reconstruction"],
            "completed_run_count": 2,
            "failed_run_count": 0,
            "core_grid_complete": False,
        },
    )
    pd.DataFrame([_grid_row(objective="supervised", macro_f1=0.4)]).to_csv(
        grid / "results_summary.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "horizon": 10,
                "lookback": 20,
                "model_family": "matrix_transformer",
                "pretraining_objective": "none",
                "completed_run_count": 1,
                "failed_run_count": 0,
                "mean_accuracy": 0.5,
                "std_accuracy": 0.0,
                "mean_macro_f1": 0.4,
                "std_macro_f1": 0.0,
                "mean_mcc": 0.1,
                "std_mcc": 0.0,
                "mean_ece": 0.2,
                "std_ece": 0.0,
                "mean_brier_score": 0.4,
                "mean_nll": 1.0,
            }
        ]
    ).to_csv(grid / "aggregate_summary.csv", index=False)
    _write_json(grid / "aggregate_summary.json", {"aggregate": []})
    pd.DataFrame(
        [
            {
                "fold": 1,
                "horizon": 10,
                "seed": 0,
                "lookback": 20,
                "model_family": "matrix_transformer",
                "ssl_objective": "masked_reconstruction",
                "comparison": "supervised_vs_masked_reconstruction",
                "delta_macro_f1": 0.01,
                "delta_mcc": 0.02,
                "delta_ece": -0.01,
                "macro_f1_outcome": "win",
                "mcc_outcome": "win",
                "ece_outcome": "win",
                "status": "matched",
                "reason": "",
            }
        ]
    ).to_csv(grid / "ssl_comparison.csv", index=False)
    pd.DataFrame(
        columns=[
            "fold",
            "horizon",
            "seed",
            "objective",
            "reason",
            "traceback",
            "invalidates_aggregate_claims",
            "status",
            "run_id",
        ]
    ).to_csv(grid / "failures.csv", index=False)
    return grid


def test_grid_expansion_produces_expected_number_of_specs() -> None:
    specs = expand_fi2010_neural_grid(
        folds=[1, 2],
        horizons=[10, 20],
        seeds=[0, 1],
        lookbacks=[20, 50],
        objectives=["supervised", "next_field"],
    )

    assert len(specs) == 2 * 2 * 2 * 2 * 2
    assert specs[0].run_id == "fold_1__h10__seed_0__lb20__supervised"
    assert specs[-1].run_id == "fold_2__h20__seed_1__lb50__next_field"


def test_per_horizon_config_generation_selects_label_column(tmp_path: Path) -> None:
    payload = build_horizon_config_payload(
        CONFIG_PATH,
        horizon=20,
        folds=[1],
        seeds=[3],
        lookbacks=[20],
        max_epochs=2,
        batch_size=8,
        device="cpu",
        smoke_test=True,
    )

    assert payload["target"] == {"horizon": 20, "label_column": "label_20"}
    config_path = write_horizon_config(
        CONFIG_PATH,
        tmp_path / "h20.yaml",
        horizon=20,
        folds=[1],
        seeds=[3],
        lookbacks=[20],
        max_epochs=2,
        batch_size=8,
        device="cpu",
        smoke_test=True,
    )
    loaded = load_neural_benchmark_config(config_path)
    assert loaded.target.horizon == 20
    assert loaded.target.label_column == "label_20"


def test_reuse_mode_does_not_rerun_completed_artefacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronoslob.experiments import fi2010_neural_grid

    out_dir = tmp_path / "grid"
    spec = FI2010NeuralGridRunSpec(
        fold=1,
        horizon=10,
        seed=0,
        lookback=20,
        objective="supervised",
    )
    run_dir = spec.run_dir(out_dir)
    run_dir.mkdir(parents=True)
    (run_dir / "status.txt").write_text("completed\n", encoding="utf-8")
    (run_dir / "predictions.csv").write_text("row_id,y_true,y_pred\n0,1,1\n", encoding="utf-8")
    _write_json(run_dir / "metrics.json", _minimal_completed_metrics(spec))

    def _boom(**_: Any) -> Any:
        raise AssertionError("runner should not be called when reusing artefacts")

    monkeypatch.setattr(fi2010_neural_grid, "_call_supervised_runner", _boom)

    summary = run_fi2010_neural_full_grid(
        CONFIG_PATH,
        processed_root=tmp_path / "missing_processed",
        out_dir=out_dir,
        folds=[1],
        horizons=[10],
        seeds=[0],
        lookbacks=[20],
        objectives=["supervised"],
        reuse_completed=True,
        smoke_test=True,
    )

    assert summary.completed_run_count == 1
    assert summary.skipped_existing_count == 1
    assert (run_dir / "status.txt").read_text(encoding="utf-8").strip() == "skipped_existing"
    failures = pd.read_csv(out_dir / "failures.csv")
    assert failures.loc[0, "status"] == "skipped_existing"


def test_failed_runs_are_recorded_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronoslob.experiments import fi2010_neural_grid

    def _fail(**_: Any) -> Any:
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(fi2010_neural_grid, "_call_supervised_runner", _fail)
    out_dir = tmp_path / "failed_grid"

    summary = run_fi2010_neural_full_grid(
        CONFIG_PATH,
        processed_root=tmp_path / "missing_processed",
        out_dir=out_dir,
        folds=[1],
        horizons=[10],
        seeds=[0],
        lookbacks=[20],
        objectives=["supervised"],
        reuse_completed=False,
        smoke_test=True,
    )

    assert summary.failed_run_count == 1
    failures = pd.read_csv(out_dir / "failures.csv")
    assert failures.loc[0, "status"] == "failed"
    assert failures.loc[0, "invalidates_aggregate_claims"] is True or str(
        failures.loc[0, "invalidates_aggregate_claims"]
    ) == "True"
    run_dir = FI2010NeuralGridRunSpec(1, 10, 0, 20, "supervised").run_dir(out_dir)
    assert (run_dir / "status.txt").read_text(encoding="utf-8").strip() == "failed"


def test_aggregate_summary_computes_means_and_standard_deviations(
    tmp_path: Path,
) -> None:
    rows = [
        _grid_row(objective="supervised", macro_f1=0.4, accuracy=0.5),
        {
            **_grid_row(objective="supervised", macro_f1=0.8, accuracy=0.7),
            "seed": 1,
            "run_id": "run_supervised_seed_1",
        },
    ]
    failures = [
        {
            "fold": 1,
            "horizon": 10,
            "seed": 2,
            "lookback": 20,
            "objective": "supervised",
            "reason": "failed",
            "traceback": "failed",
            "invalidates_aggregate_claims": True,
            "status": "failed",
            "run_id": "failed",
        }
    ]

    aggregate_rows, _, _ = write_neural_grid_aggregate_artifacts(
        tmp_path,
        result_rows=rows,
        failure_rows=failures,
    )

    assert len(aggregate_rows) == 1
    row = aggregate_rows[0]
    assert row["mean_accuracy"] == pytest.approx(0.6)
    assert row["std_accuracy"] == pytest.approx(0.1)
    assert row["mean_macro_f1"] == pytest.approx(0.6)
    assert row["std_macro_f1"] == pytest.approx(0.2)
    assert row["failed_run_count"] == 1


def test_matched_ssl_comparison_refuses_unmatched_pairs() -> None:
    rows = [
        _grid_row(objective="supervised", macro_f1=0.4, arch="arch_a"),
        _grid_row(
            objective="masked_reconstruction",
            macro_f1=0.5,
            arch="arch_b",
        ),
    ]

    comparison, missing = build_ssl_comparison_rows(rows)

    statuses = {row["ssl_objective"]: row["status"] for row in comparison}
    assert statuses["masked_reconstruction"] == "unmatched_metadata"
    assert statuses["next_field"] == "missing_pair"
    assert len(missing) == 2
    assert all(row["delta_macro_f1"] is None for row in missing)


def test_report_builder_marks_smoke_full_grid_as_unsupported(
    tmp_path: Path,
) -> None:
    from chronoslob.experiments.final_report import build_final_empirical_report

    dirs = _write_minimal_required_dirs(tmp_path)
    full_grid = _write_full_grid_dir(tmp_path, smoke=True)
    report_path = tmp_path / "report.md"

    summary = build_final_empirical_report(
        classical_dir=dirs["classical"],
        neural_dir=dirs["neural"],
        uncertainty_dir=dirs["uncertainty"],
        neural_full_grid_dir=full_grid,
        out_path=report_path,
        overwrite=True,
    )

    text = report_path.read_text(encoding="utf-8")
    assert "### Full Neural Grid" in text
    assert "smoke-test-only" in text
    assert "not empirical evidence" in text
    assert "Full Neural Grid" in summary.skipped_sections


def test_report_builder_distinguishes_missing_and_real_full_grid(
    tmp_path: Path,
) -> None:
    from chronoslob.experiments.final_report import build_final_empirical_report

    dirs = _write_minimal_required_dirs(tmp_path)
    missing_report = tmp_path / "missing_grid_report.md"
    missing_summary = build_final_empirical_report(
        classical_dir=dirs["classical"],
        neural_dir=dirs["neural"],
        uncertainty_dir=dirs["uncertainty"],
        out_path=missing_report,
        overwrite=True,
    )
    missing_text = missing_report.read_text(encoding="utf-8")
    assert "No full-grid neural result is claimed" in missing_text
    assert "Full Neural Grid" in missing_summary.skipped_sections

    real_grid = _write_full_grid_dir(tmp_path, smoke=False)
    real_report = tmp_path / "real_grid_report.md"
    real_summary = build_final_empirical_report(
        classical_dir=dirs["classical"],
        neural_dir=dirs["neural"],
        uncertainty_dir=dirs["uncertainty"],
        neural_full_grid_dir=real_grid,
        out_path=real_report,
        overwrite=True,
    )
    real_text = real_report.read_text(encoding="utf-8")
    assert "Status: loaded" in real_text
    assert "Matched SSL deltas" in real_text
    assert "Full Neural Grid" not in real_summary.skipped_sections
    assert "Self-Supervised Pretraining" not in real_summary.skipped_sections
    assert "SSL input not supplied" not in real_summary.missing_sections


def test_smoke_cli_writes_expected_directory_structure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("torch")
    config_path = _write_tiny_neural_config(tmp_path)
    processed_root = _prepare_synthetic_processed_root(tmp_path)
    out_dir = tmp_path / "cli_grid"

    exit_code = _run_fi2010_neural_full_grid_impl(
        config_path=config_path,
        processed_root=processed_root,
        out=out_dir,
        folds=["fold_1"],
        horizons=[10],
        seeds=[7],
        lookbacks=[2],
        objectives=["supervised"],
        pretrain_epochs=1,
        max_epochs=1,
        batch_size=4,
        device="cpu",
        reuse_completed=True,
        smoke_test=True,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "FI-2010 neural full grid runner" in captured.out
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "results_summary.csv").is_file()
    assert (out_dir / "aggregate_summary.csv").is_file()
    assert (out_dir / "aggregate_summary.json").is_file()
    assert (out_dir / "failures.csv").is_file()
    assert (out_dir / "ssl_comparison.csv").is_file()
    run_dir = FI2010NeuralGridRunSpec(1, 10, 7, 2, "supervised").run_dir(out_dir)
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "predictions.csv").is_file()
    assert (run_dir / "sha256_manifest.json").is_file()
