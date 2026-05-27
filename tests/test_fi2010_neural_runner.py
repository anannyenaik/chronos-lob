"""Tests for the FI-2010 supervised neural benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]

from chronoslob.cli import _run_fi2010_neural_benchmark_impl
from chronoslob.experiments.fi2010_multifold import (
    load_multifold_config,
    prepare_multifold,
)
from chronoslob.experiments.fi2010_neural_runner import (
    run_fi2010_neural_benchmark,
)
from chronoslob.utils.audit import (
    AuditStatus,
    check_no_forbidden_claims,
    check_public_release_wording,
)
from chronoslob.utils.paths import project_root
from tests.test_fi2010_multifold import (
    _build_synthetic_extracted_root,
    _write_synthetic_multifold_config,
)

CONFIG_PATH = (
    project_root() / "configs" / "experiments" / "fi2010_neural_serious.yaml"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_tiny_neural_config(
    tmp_path: Path,
    *,
    folds: tuple[str, ...] = ("fold_1",),
    seeds: tuple[int, ...] = (11,),
    lookbacks: tuple[int, ...] = (2,),
) -> Path:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["study_name"] = "fi2010_neural_tiny_smoke"
    payload["folds"] = list(folds)
    payload["seeds"] = list(seeds)
    payload["lookbacks"] = list(lookbacks)
    payload["mode"] = "smoke"
    payload["benchmark_note"] = "Tiny supervised neural smoke run for tests."
    payload["device_selection"] = "cpu"
    payload["training"]["batch_size"] = 4
    payload["training"]["max_epochs"] = 1
    payload["training"]["early_stopping_patience"] = 1
    payload["training"]["learning_rate"] = 0.001
    payload["training"]["weight_decay"] = 0.0
    payload["training"]["dropout"] = 0.0
    payload["neural_models"]["deeplob_style"]["conv_channels"] = 2
    payload["neural_models"]["deeplob_style"]["lstm_hidden_size"] = 4
    payload["neural_models"]["deeplob_style"]["use_batch_norm"] = False
    payload["neural_models"]["deeplob_style"]["dropout"] = 0.0
    payload["neural_models"]["matrix_transformer"]["model_dim"] = 8
    payload["neural_models"]["matrix_transformer"]["num_heads"] = 2
    payload["neural_models"]["matrix_transformer"]["num_layers"] = 1
    payload["neural_models"]["matrix_transformer"]["feedforward_dim"] = 16
    payload["neural_models"]["matrix_transformer"]["dropout"] = 0.0
    config_path = tmp_path / "fi2010_neural_tiny.yaml"
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def _prepare_synthetic_processed_root(
    tmp_path: Path,
    *,
    folds: tuple[int, ...] = (1,),
    train_snapshots: int = 18,
    test_snapshots: int = 8,
) -> Path:
    config_path = _write_synthetic_multifold_config(tmp_path, folds=folds)
    config = load_multifold_config(config_path)
    extracted_root = _build_synthetic_extracted_root(
        tmp_path,
        folds=folds,
        train_snapshots=train_snapshots,
        test_snapshots=test_snapshots,
    )
    processed_root = tmp_path / "processed"
    prepare_multifold(
        config=config,
        config_source_path=config_path,
        extracted_root=extracted_root,
        processed_root=processed_root,
        output_dir=tmp_path / "prepare_out",
    )
    return processed_root


def test_cli_smoke_run_completes_on_tiny_prepared_fold(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("torch")
    config_path = _write_tiny_neural_config(tmp_path)
    processed_root = _prepare_synthetic_processed_root(tmp_path)
    out_dir = tmp_path / "neural_out"

    exit_code = _run_fi2010_neural_benchmark_impl(
        config_path=config_path,
        processed_root=processed_root,
        out=out_dir,
        folds=["fold_1"],
        models=["deeplob_style", "matrix_transformer"],
        seeds=[11],
        lookbacks=[2],
        max_epochs=1,
        overwrite=False,
        fail_fast=False,
        write_full_predictions=False,
        write_checkpoints=False,
        allow_full_benchmark=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "FI-2010 neural benchmark runner" in captured.out
    assert "execution mode:      smoke" in captured.out
    assert "full predictions:    not written" in captured.out
    assert "checkpoints:         not written" in captured.out
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "run_plan.csv").is_file()
    assert (out_dir / "results_by_fold_seed.csv").is_file()
    assert (out_dir / "results_summary.csv").is_file()
    assert (out_dir / "training_summary.csv").is_file()
    assert (out_dir / "model_capacity_summary.csv").is_file()
    assert (out_dir / "model_failures.json").is_file()
    assert not list(out_dir.rglob("predictions.csv"))
    assert not list(out_dir.rglob("*.pt"))

    summary = _read_json(out_dir / "summary.json")
    assert summary["execution_mode"] == "smoke"
    assert summary["full_benchmark_grid"] is False
    assert summary["full_predictions_written"] is False
    assert summary["checkpoints_written"] is False

    results = pd.read_csv(out_dir / "results_by_fold_seed.csv")
    assert set(results["status"]) == {"ok"}
    assert set(results["split"]) == {"test"}
    assert {"accuracy", "macro_f1", "mcc", "brier_score", "ece"}.issubset(
        results.columns,
    )

    run_payload = _read_json(
        out_dir / "runs" / "fold_1_seed_11_deeplob_style_lb2" / "result.json",
    )
    split = run_payload["split_summary"]
    assert split["split_method"] == "official_column"
    assert split["official_train_rows"] == 18
    assert split["official_test_rows"] == 8
    assert split["n_train"] + split["n_validation"] == 18
    assert split["n_test"] == 8


def test_run_subset_selection_uses_requested_fold_seed_model_and_lookback(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    config_path = _write_tiny_neural_config(
        tmp_path,
        folds=("fold_1", "fold_2"),
        seeds=(11, 12),
        lookbacks=(2, 3),
    )
    processed_root = _prepare_synthetic_processed_root(tmp_path, folds=(1, 2))
    out_dir = tmp_path / "subset_out"

    summary = run_fi2010_neural_benchmark(
        config_path,
        processed_root=processed_root,
        out_dir=out_dir,
        folds=["fold_2"],
        models=["matrix_transformer"],
        seeds=[12],
        lookbacks=[3],
        max_epochs=1,
    )

    assert summary.folds_requested == ["fold_2"]
    assert summary.models_requested == ["matrix_transformer"]
    assert summary.seeds == [12]
    assert summary.lookbacks == [3]
    plan = pd.read_csv(out_dir / "run_plan.csv")
    assert plan["run_id"].tolist() == [
        "fold_2__seed_12__matrix_transformer__lookback_3"
    ]


def test_unsupported_neural_model_fails_clearly(tmp_path: Path) -> None:
    config_path = _write_tiny_neural_config(tmp_path)
    processed_root = _prepare_synthetic_processed_root(tmp_path)

    with pytest.raises(ValueError, match="unsupported neural benchmark model"):
        run_fi2010_neural_benchmark(
            config_path,
            processed_root=processed_root,
            out_dir=tmp_path / "out",
            models=["ssl_transformer"],
            max_epochs=1,
        )


def test_overwrite_protection_blocks_existing_output(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    config_path = _write_tiny_neural_config(tmp_path)
    processed_root = _prepare_synthetic_processed_root(tmp_path)
    out_dir = tmp_path / "neural_out"

    run_fi2010_neural_benchmark(
        config_path,
        processed_root=processed_root,
        out_dir=out_dir,
        folds=["fold_1"],
        models=["deeplob_style"],
        seeds=[11],
        lookbacks=[2],
        max_epochs=1,
    )

    with pytest.raises(FileExistsError, match="non-empty output directory"):
        run_fi2010_neural_benchmark(
            config_path,
            processed_root=processed_root,
            out_dir=out_dir,
            folds=["fold_1"],
            models=["deeplob_style"],
            seeds=[11],
            lookbacks=[2],
            max_epochs=1,
        )


def test_failures_are_recorded_without_abort_when_fail_fast_is_false(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    config_path = _write_tiny_neural_config(tmp_path, lookbacks=(2, 99))
    processed_root = _prepare_synthetic_processed_root(tmp_path)
    out_dir = tmp_path / "failure_out"

    summary = run_fi2010_neural_benchmark(
        config_path,
        processed_root=processed_root,
        out_dir=out_dir,
        folds=["fold_1"],
        models=["deeplob_style"],
        seeds=[11],
        lookbacks=[2, 99],
        max_epochs=1,
        fail_fast=False,
    )

    assert summary.completed_run_count == 1
    assert summary.failure_count == 1
    failures = _read_json(out_dir / "model_failures.json")
    assert failures["failure_count"] == 1
    results = pd.read_csv(out_dir / "results_by_fold_seed.csv")
    assert set(results["status"]) == {"ok", "failed"}


def test_neural_benchmark_doc_avoids_forbidden_public_claims() -> None:
    scan_path = Path("docs") / "FI2010_NEURAL_BENCHMARKS.md"

    claims = check_no_forbidden_claims(project_root(), scan_paths=(scan_path,))
    wording = check_public_release_wording(project_root(), scan_paths=(scan_path,))

    assert claims.status == AuditStatus.PASS
    assert wording.status == AuditStatus.PASS


def test_validation_metric_value_picks_last_best_epoch() -> None:
    from chronoslob.experiments.fi2010_neural_runner import _validation_metric_value

    metadata = {
        "training_history": [
            {"epoch": 1, "is_best": True, "validation_macro_f1": 0.25},
            {"epoch": 2, "is_best": False, "validation_macro_f1": 0.24},
            {"epoch": 3, "is_best": True, "validation_macro_f1": 0.40},
            {"epoch": 4, "is_best": False, "validation_macro_f1": 0.39},
            {"epoch": 5, "is_best": True, "validation_macro_f1": 0.65},
        ],
    }

    value = _validation_metric_value(metadata, metric_name="validation_macro_f1")

    assert value == pytest.approx(0.65)
