"""Tests for the FI-2010 proper-training neural subset runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from chronoslob.experiments import fi2010_neural_proper_training as proper
from chronoslob.experiments.fi2010_neural_proper_training import (
    FI2010_PROPER_TRAINING_VERSION,
    FI2010ProperTrainingRunSpec,
    expand_proper_training_specs,
    run_fi2010_neural_proper_training_subset,
)
from chronoslob.experiments.final_report import build_final_empirical_report
from chronoslob.experiments.manifests import sha256_file, stable_json_dumps
from tests.test_fi2010_neural_runner import _write_tiny_neural_config
from tests.test_fi2010_ssl_runner import _write_minimal_required_dirs


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _fake_execute_run_spec(
    spec: FI2010ProperTrainingRunSpec,
    *,
    config: Any,
    out_dir: Path,
    run_dir: Path,
    pretrain_epochs: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    device: str,
    monitored_metric: str,
    smoke_test: bool,
    mask_probability: float,
    bucket_count: int,
    git_commit: str | None,
    **_: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    macro_f1 = {
        "supervised": 0.50,
        "masked_reconstruction": 0.47,
        "next_field": 0.52,
    }[spec.objective]
    metrics = {
        "accuracy": 0.60,
        "macro_f1": macro_f1,
        "mcc": 0.10 if spec.objective != "masked_reconstruction" else 0.08,
        "ece": 0.20 if spec.objective != "next_field" else 0.18,
        "brier_score": 0.30,
        "nll": 1.20,
        "class_f1_down": 0.40,
        "class_f1_stationary": 0.55,
        "class_f1_up": 0.55,
    }
    curves = [
        {
            "epoch": 1,
            "train_loss": 1.0,
            "validation_loss": 0.9,
            "validation_accuracy": 0.55,
            "validation_macro_f1": 0.45,
            "validation_mcc": 0.05,
            "monitored_value": 0.45,
            "learning_rate": 0.001,
            "is_best": True,
            "early_stop": False,
        },
        {
            "epoch": 2,
            "train_loss": 0.8,
            "validation_loss": 0.95,
            "validation_accuracy": 0.54,
            "validation_macro_f1": 0.44,
            "validation_mcc": 0.04,
            "monitored_value": 0.44,
            "learning_rate": 0.001,
            "is_best": False,
            "early_stop": True,
        },
    ]
    pd.DataFrame(
        [
            {
                "row_id": 0,
                "sample_id": 0,
                "fold": spec.fold,
                "horizon": spec.horizon,
                "seed": spec.seed,
                "lookback": spec.lookback,
                "model_family": spec.model_family,
                "pretraining_objective": spec.pretraining_objective,
                "split": "test",
                "y_true": 1,
                "y_pred": 1,
                "prob_down": 0.1,
                "prob_stationary": 0.2,
                "prob_up": 0.7,
                "confidence": 0.7,
            }
        ]
    ).to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(curves).to_csv(run_dir / "curves.csv", index=False)
    (run_dir / "curves.json").write_text(
        stable_json_dumps({"run_id": spec.run_id, "curves": curves}),
        encoding="utf-8",
    )
    reuse_signature = proper._expected_reuse_signature(
        spec,
        config=config,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        pretrain_epochs=pretrain_epochs,
        device=device,
        smoke_test=smoke_test,
        mask_probability=mask_probability,
        bucket_count=bucket_count,
    )
    (run_dir / "config.json").write_text(
        stable_json_dumps(
            {
                "run_id": spec.run_id,
                "reuse_signature": reuse_signature,
                "max_epochs": max_epochs,
                "early_stopping_patience": patience,
                "early_stopping_metric": monitored_metric,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "git_commit.txt").write_text(git_commit or "", encoding="utf-8")
    payload = {
        "runner_version": FI2010_PROPER_TRAINING_VERSION,
        "subset_kind": "proper_training_subset",
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
        "metrics": metrics,
        "training": {
            "max_epochs": max_epochs,
            "epochs_ran": 2,
            "best_epoch": 1,
            "monitored_metric": monitored_metric,
            "best_validation_score": 0.45,
            "early_stopping_patience": patience,
            "early_stopped": True,
            "training_seconds": 0.01,
            "validation_only_model_selection": True,
            "best_checkpoint_restored_before_test": True,
        },
        "ssl_pretraining": None,
        "checkpoint_hash": None,
        "prediction_file": proper._relative_path(run_dir / "predictions.csv", out_dir),
        "curves_file": proper._relative_path(run_dir / "curves.csv", out_dir),
        "architecture_hash": "matched_architecture",
        "preprocessing_hash": "matched_preprocessing",
        "git_commit": git_commit,
    }
    (run_dir / "metrics.json").write_text(stable_json_dumps(payload), encoding="utf-8")
    proper._write_status(run_dir, "completed")
    proper._write_run_log(run_dir, {"run_id": spec.run_id, "status": "completed"})
    proper._write_run_manifest(run_dir)
    return (
        proper._result_row_from_payload(payload, out_dir=out_dir, run_dir=run_dir),
        proper._training_row_from_payload(payload),
    )


def test_expand_proper_training_specs_smoke_keeps_ssl_objectives() -> None:
    specs = expand_proper_training_specs(
        folds=[1, 2],
        horizons=[10, 20],
        seeds=[0, 1],
        lookbacks=[50],
        objectives=["supervised", "masked_reconstruction", "next_field"],
        smoke_test=True,
    )

    assert len(specs) == 3
    assert {spec.objective for spec in specs} == {
        "supervised",
        "masked_reconstruction",
        "next_field",
    }


def test_expand_proper_training_specs_supports_two_supervised_models(
    tmp_path: Path,
) -> None:
    specs = expand_proper_training_specs(
        folds=[1],
        horizons=[10],
        seeds=[0],
        lookbacks=[50],
        models=["matrix_transformer", "deeplob_style"],
        objectives=["supervised"],
    )

    assert len(specs) == 2
    assert {spec.model_family for spec in specs} == {
        "matrix_transformer",
        "deeplob_style",
    }
    run_dirs = {spec.model_family: spec.run_dir(tmp_path) for spec in specs}
    assert run_dirs["matrix_transformer"].parts[-1] == "supervised"
    assert run_dirs["deeplob_style"].parts[-2:] == ("deeplob_style", "supervised")
    assert len({spec.run_id for spec in specs}) == 2


def test_deeplob_proper_training_rejects_ssl_objectives() -> None:
    with pytest.raises(ValueError, match="supports the supervised objective only"):
        expand_proper_training_specs(
            folds=[1],
            horizons=[10],
            seeds=[0],
            lookbacks=[50],
            models=["deeplob_style"],
            objectives=["masked_reconstruction"],
        )


def test_proper_training_two_model_supervised_slice_is_collision_free(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(proper, "_execute_run_spec", _fake_execute_run_spec)
    out_dir = tmp_path / "proper_two_model"

    summary = run_fi2010_neural_proper_training_subset(
        config_path=_write_tiny_neural_config(tmp_path),
        processed_root=tmp_path / "processed",
        out_dir=out_dir,
        folds=[1],
        horizons=[10],
        seeds=[0],
        lookbacks=[2],
        models=["matrix_transformer", "deeplob_style"],
        objectives=["supervised"],
        pretrain_epochs=1,
        max_epochs=2,
        patience=1,
        batch_size=4,
    )

    results = pd.read_csv(out_dir / "results_summary.csv")
    assert summary.models == ["matrix_transformer", "deeplob_style"]
    assert summary.completed_run_count == 2
    assert summary.missing_pair_count == 0
    assert set(results["model_family"]) == {"matrix_transformer", "deeplob_style"}
    assert (
        out_dir
        / "runs"
        / "fold_1"
        / "horizon_10"
        / "seed_0"
        / "lookback_2"
        / "supervised"
        / "metrics.json"
    ).is_file()
    assert (
        out_dir
        / "runs"
        / "fold_1"
        / "horizon_10"
        / "seed_0"
        / "lookback_2"
        / "deeplob_style"
        / "supervised"
        / "metrics.json"
    ).is_file()


def test_broader_two_model_scope_is_complete_real() -> None:
    primary = proper._primary_target_complete(
        folds=[1, 2, 3, 4, 5],
        horizons=[10, 50],
        seeds=[0, 1, 2],
        lookbacks=[20, 50, 100],
        models=["matrix_transformer", "deeplob_style"],
        objectives=["supervised"],
        max_epochs=25,
        patience=5,
        planned_complete=True,
        smoke_test=False,
    )
    label = proper._scope_label(
        folds=[1, 2, 3, 4, 5],
        horizons=[10, 50],
        seeds=[0, 1, 2],
        lookbacks=[20, 50, 100],
        models=["matrix_transformer", "deeplob_style"],
        objectives=["supervised"],
        max_epochs=25,
        patience=5,
        planned_complete=True,
        primary_complete=primary,
        smoke_test=False,
    )

    assert primary is True
    assert label == "broader_proper_training_complete"


def test_proper_training_runner_writes_matched_partial_artefacts(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(proper, "_execute_run_spec", _fake_execute_run_spec)
    out_dir = tmp_path / "proper_subset"

    summary = run_fi2010_neural_proper_training_subset(
        config_path=_write_tiny_neural_config(tmp_path),
        processed_root=tmp_path / "processed",
        out_dir=out_dir,
        folds=[1],
        horizons=[10],
        seeds=[0],
        lookbacks=[2],
        objectives=["supervised", "masked_reconstruction", "next_field"],
        pretrain_epochs=1,
        max_epochs=2,
        patience=1,
        batch_size=4,
    )

    assert summary.completed_run_count == 3
    assert summary.failed_run_count == 0
    assert summary.missing_pair_count == 0
    assert summary.target_scope_complete is False

    expected = {
        "summary.json",
        "config_snapshot.json",
        "run_plan.csv",
        "results_summary.csv",
        "aggregate_summary.csv",
        "aggregate_summary.json",
        "training_curves_summary.csv",
        "ssl_comparison.csv",
        "missing_pairs.csv",
        "failures.csv",
        "README.md",
        "sha256_manifest.json",
    }
    assert expected.issubset({path.name for path in out_dir.iterdir()})

    summary_payload = _read_json(out_dir / "summary.json")
    assert summary_payload["evidence_level"] == "partial_real"
    assert summary_payload["planned_scope_complete"] is True
    assert summary_payload["scope_label"] == "tiny_or_limited_partial_slice"
    assert summary_payload["early_stopping"] == {
        "metric": "validation_macro_f1",
        "patience": 1,
        "validation_only_model_selection": True,
        "restore_best_checkpoint_before_test": True,
        "no_test_set_selection": True,
    }
    aggregate_payload = _read_json(out_dir / "aggregate_summary.json")
    assert aggregate_payload["runner_version"] == FI2010_PROPER_TRAINING_VERSION
    assert aggregate_payload["subset_kind"] == "proper_training_subset"

    comparison = pd.read_csv(out_dir / "ssl_comparison.csv")
    assert set(comparison["status"]) == {"matched"}
    assert set(comparison["ssl_objective"]) == {
        "masked_reconstruction",
        "next_field",
    }

    curves = pd.read_csv(
        out_dir
        / "runs"
        / "fold_1"
        / "horizon_10"
        / "seed_0"
        / "lookback_2"
        / "supervised"
        / "curves.csv"
    )
    assert {
        "validation_accuracy",
        "validation_macro_f1",
        "validation_mcc",
        "is_best",
        "early_stop",
    }.issubset(curves.columns)


def test_proper_training_reuse_refreshes_status_manifest(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(proper, "_execute_run_spec", _fake_execute_run_spec)
    out_dir = tmp_path / "proper_reuse"
    kwargs = {
        "config_path": _write_tiny_neural_config(tmp_path),
        "processed_root": tmp_path / "processed",
        "out_dir": out_dir,
        "folds": [1],
        "horizons": [10],
        "seeds": [0],
        "lookbacks": [2],
        "objectives": ["supervised"],
        "pretrain_epochs": 1,
        "max_epochs": 2,
        "patience": 1,
        "batch_size": 4,
    }
    run_fi2010_neural_proper_training_subset(**kwargs)

    def _boom(*_: Any, **__: Any) -> Any:
        raise AssertionError("completed run should be reused")

    monkeypatch.setattr(proper, "_execute_run_spec", _boom)
    summary = run_fi2010_neural_proper_training_subset(**kwargs)

    run_dir = out_dir / "runs" / "fold_1" / "horizon_10" / "seed_0" / "lookback_2" / "supervised"
    manifest = _read_json(run_dir / "sha256_manifest.json")
    assert summary.skipped_existing_count == 1
    assert (run_dir / "status.txt").read_text(encoding="utf-8").strip() == "skipped_existing"
    assert _read_json(run_dir / "run_log.json")["status"] == "skipped_existing"
    assert manifest["sha256"]["status.txt"] == sha256_file(run_dir / "status.txt")


def test_proper_training_reruns_completed_run_when_config_changes(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls = {"count": 0}

    def _counting_fake(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        calls["count"] += 1
        return _fake_execute_run_spec(*args, **kwargs)

    monkeypatch.setattr(proper, "_execute_run_spec", _counting_fake)
    out_dir = tmp_path / "proper_reuse_config_change"
    kwargs = {
        "config_path": _write_tiny_neural_config(tmp_path),
        "processed_root": tmp_path / "processed",
        "out_dir": out_dir,
        "folds": [1],
        "horizons": [10],
        "seeds": [0],
        "lookbacks": [2],
        "objectives": ["supervised"],
        "pretrain_epochs": 1,
        "max_epochs": 2,
        "patience": 1,
        "batch_size": 4,
    }
    run_fi2010_neural_proper_training_subset(**kwargs)
    changed = {**kwargs, "max_epochs": 3}
    summary = run_fi2010_neural_proper_training_subset(**changed)

    assert calls["count"] == 2
    assert summary.skipped_existing_count == 0
    assert summary.warnings == [
        "1 completed run(s) had a mismatched reuse signature and were rerun instead of reused."
    ]


def test_primary_scope_is_required_for_complete_real(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(proper, "_execute_run_spec", _fake_execute_run_spec)
    config_path = _write_tiny_neural_config(tmp_path)
    common = {
        "config_path": config_path,
        "processed_root": tmp_path / "processed",
        "seeds": [0],
        "lookbacks": [50],
        "objectives": ["supervised", "masked_reconstruction", "next_field"],
        "pretrain_epochs": 5,
        "max_epochs": 25,
        "patience": 5,
        "batch_size": 4,
    }

    fallback = run_fi2010_neural_proper_training_subset(
        **common,
        out_dir=tmp_path / "fallback",
        folds=[1, 2, 3],
        horizons=[10, 50],
    )
    primary = run_fi2010_neural_proper_training_subset(
        **common,
        out_dir=tmp_path / "primary",
        folds=[1, 2, 3, 4, 5],
        horizons=[10, 50],
    )

    assert fallback.planned_scope_complete is True
    assert fallback.target_scope_complete is False
    assert fallback.scope_label == "fallback_credible_slice"
    fallback_payload = _read_json(Path(fallback.output_dir) / "summary.json")
    assert fallback_payload["evidence_level"] == "partial_real"

    assert primary.planned_scope_complete is True
    assert primary.target_scope_complete is True
    assert primary.scope_label == "primary_credible_minimum"
    primary_payload = _read_json(Path(primary.output_dir) / "summary.json")
    assert primary_payload["evidence_level"] == "complete_real"


def test_final_report_loads_proper_training_subset(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(proper, "_execute_run_spec", _fake_execute_run_spec)
    dirs = _write_minimal_required_dirs(tmp_path)
    proper_dir = tmp_path / "proper_report"
    run_fi2010_neural_proper_training_subset(
        config_path=_write_tiny_neural_config(tmp_path),
        processed_root=tmp_path / "processed",
        out_dir=proper_dir,
        folds=[1],
        horizons=[10],
        seeds=[0],
        lookbacks=[2],
        objectives=["supervised", "masked_reconstruction", "next_field"],
        pretrain_epochs=1,
        max_epochs=2,
        patience=1,
        batch_size=4,
    )

    report_path = tmp_path / "report.md"
    summary = build_final_empirical_report(
        classical_dir=dirs["classical"],
        neural_dir=dirs["neural"],
        uncertainty_dir=dirs["uncertainty"],
        proper_training_dir=proper_dir,
        out_path=report_path,
        overwrite=True,
    )

    text = report_path.read_text(encoding="utf-8")
    assert "proper_training_neural_scope" in text
    assert "## Proper-Training Neural Subset" in text
    assert "partial_real" in text
    assert "validation-only early stopping; best checkpoint restored before test" in text
    assert "no broad SSL improvement is claimed" in text
    assert "Proper-Training Neural Subset" not in summary.skipped_sections

