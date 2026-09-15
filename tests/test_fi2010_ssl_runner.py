"""Tests for the FI-2010 SSL pretraining and fine-tuning benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]

from chronoslob.experiments.manifests import sha256_file
from chronoslob.utils.paths import project_root
from tests.test_fi2010_multifold import (
    _build_synthetic_extracted_root,
    _write_synthetic_multifold_config,
)

CONFIG_PATH = project_root() / "tests" / "fixtures" / "configs" / "fi2010_ssl_smoke.yaml"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_tiny_ssl_config(tmp_path: Path) -> Path:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["study_name"] = "fi2010_ssl_tiny_smoke"
    payload["folds"] = ["fold_1"]
    payload["seeds"] = [7]
    payload["lookbacks"] = [2]
    payload["training"]["batch_size"] = 4
    payload["training"]["max_epochs"] = 1
    payload["training"]["early_stopping_patience"] = 1
    payload["neural_models"]["matrix_transformer"]["model_dim"] = 8
    payload["neural_models"]["matrix_transformer"]["num_heads"] = 2
    payload["neural_models"]["matrix_transformer"]["num_layers"] = 1
    payload["neural_models"]["matrix_transformer"]["feedforward_dim"] = 16
    config_path = tmp_path / "fi2010_ssl_tiny.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def _prepare_synthetic_processed_root(
    tmp_path: Path,
    *,
    train_snapshots: int = 20,
    test_snapshots: int = 8,
) -> Path:
    from chronoslob.experiments.fi2010_multifold import (
        load_multifold_config,
        prepare_multifold,
    )

    config_path = _write_synthetic_multifold_config(tmp_path, folds=(1,))
    config = load_multifold_config(config_path)
    extracted_root = _build_synthetic_extracted_root(
        tmp_path,
        folds=(1,),
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


def test_ssl_runner_smoke_writes_artefacts(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from chronoslob.experiments.fi2010_ssl_runner import (
        run_fi2010_ssl_neural_benchmark,
    )

    config_path = _write_tiny_ssl_config(tmp_path)
    processed_root = _prepare_synthetic_processed_root(tmp_path)
    out_dir = tmp_path / "ssl_out"

    summary = run_fi2010_ssl_neural_benchmark(
        config_path,
        processed_root=processed_root,
        out_dir=out_dir,
        folds=["fold_1"],
        seeds=[7],
        lookbacks=[2],
        objective="both",
        pretrain_epochs=1,
        max_epochs=1,
        batch_size=4,
        device="cpu",
    )

    assert summary.completed_run_count == 1
    assert summary.failure_count == 0
    assert summary.ssl_artefacts_written is True

    # Top-level artefacts.
    for name in (
        "summary.json",
        "run_plan.csv",
        "results_by_fold_seed.csv",
        "results_summary.csv",
        "ssl_pretraining_summary.csv",
        "comparison_summary.csv",
        "model_failures.json",
    ):
        assert (out_dir / name).is_file(), name

    # Both the fine-tuned SSL model and the supervised baseline are recorded.
    results = pd.read_csv(out_dir / "results_by_fold_seed.csv")
    assert set(results["status"]) == {"ok"}
    assert set(results["model_name"]) == {"ssl_transformer", "supervised_transformer"}
    init = dict(zip(results["model_name"], results["init_source"], strict=True))
    assert init["ssl_transformer"] == "ssl_pretrained"
    assert init["supervised_transformer"] == "random_init"

    # Pretraining artefacts: checkpoint, config snapshot, metrics, manifest.
    run_dir = out_dir / "runs" / "fold_1_seed_7_lb2"
    pretrain = run_dir / "pretrain"
    for name in (
        "pretrained_encoder.pt",
        "pretrain_config.json",
        "pretrain_metrics.json",
        "artefact_manifest.json",
    ):
        assert (pretrain / name).is_file(), name

    manifest = _read_json(pretrain / "artefact_manifest.json")
    assert manifest["git_commit"] is not None or "git_commit" in manifest
    assert set(manifest["sha256"]) == {
        "pretrained_encoder.pt",
        "pretrain_config.json",
        "pretrain_metrics.json",
    }
    # The recorded checkpoint hash must match the checkpoint on disk.
    assert manifest["sha256"]["pretrained_encoder.pt"] == sha256_file(
        pretrain / "pretrained_encoder.pt"
    )

    # Predictions carry y_true, y_pred, probabilities and confidence.
    predictions = pd.read_csv(run_dir / "ssl_transformer" / "predictions.csv")
    assert {"label", "prediction", "confidence"}.issubset(predictions.columns)
    assert any(str(col).startswith("probability_") for col in predictions.columns)

    # Pretraining summary reports train and (train-carved) validation loss.
    pretraining = pd.read_csv(out_dir / "ssl_pretraining_summary.csv")
    assert pretraining.loc[0, "status"] == "ok"
    assert pd.notna(pretraining.loc[0, "final_pretrain_train_loss"])

    # Comparison row contains a macro_f1 delta.
    comparison = pd.read_csv(out_dir / "comparison_summary.csv")
    assert comparison.loc[0, "status"] == "ok"
    assert "macro_f1_delta" in comparison.columns


def test_ssl_runner_does_not_leak_validation_or_test_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    from chronoslob.experiments import fi2010_ssl_runner

    config_path = _write_tiny_ssl_config(tmp_path)
    processed_root = _prepare_synthetic_processed_root(tmp_path)

    # The official test rows must never enter the SSL data path.
    frame = pd.read_csv(processed_root / "fold1_combined.csv")
    test_rows = {
        index
        for index, value in enumerate(frame["split"].astype(str).str.lower())
        if value == "test"
    }
    train_rows = {
        index
        for index, value in enumerate(frame["split"].astype(str).str.lower())
        if value == "train"
    }
    assert test_rows and train_rows

    edge_calls: list[list[int]] = []
    window_calls: list[list[int]] = []
    original_edges = fi2010_ssl_runner.fit_feature_bucket_edges
    original_windows = fi2010_ssl_runner.build_contiguous_windows

    def _spy_edges(matrix: Any, *, train_indices: Any, bucket_count: int) -> Any:
        edge_calls.append([int(index) for index in train_indices])
        return original_edges(
            matrix, train_indices=train_indices, bucket_count=bucket_count
        )

    def _spy_windows(*, n_rows: int, window_length: int, allowed_indices: Any) -> Any:
        window_calls.append([int(index) for index in allowed_indices])
        return original_windows(
            n_rows=n_rows,
            window_length=window_length,
            allowed_indices=allowed_indices,
        )

    monkeypatch.setattr(fi2010_ssl_runner, "fit_feature_bucket_edges", _spy_edges)
    monkeypatch.setattr(fi2010_ssl_runner, "build_contiguous_windows", _spy_windows)

    fi2010_ssl_runner.run_fi2010_ssl_neural_benchmark(
        config_path,
        processed_root=processed_root,
        out_dir=tmp_path / "ssl_leak_out",
        folds=["fold_1"],
        seeds=[7],
        lookbacks=[2],
        objective="both",
        pretrain_epochs=1,
        max_epochs=1,
        batch_size=4,
        device="cpu",
    )

    assert edge_calls, "next-field objective should fit bucket edges"
    assert window_calls, "pretraining should build SSL windows"

    # Bucket edges are fit on the training sub-split only: no test rows and no
    # train-carved validation rows leak into the next-field statistics.
    for indices in edge_calls:
        assert set(indices).isdisjoint(test_rows)
        assert set(indices).issubset(train_rows)

    # Every pretraining window (train and train-carved validation) stays inside
    # the official training rows and never touches the held-out test rows.
    for indices in window_calls:
        assert set(indices).isdisjoint(test_rows)
        assert set(indices).issubset(train_rows)


def test_cli_ssl_impl_writes_artefacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("torch")
    from chronoslob.cli import _run_fi2010_ssl_neural_benchmark_impl

    config_path = _write_tiny_ssl_config(tmp_path)
    processed_root = _prepare_synthetic_processed_root(tmp_path)
    out_dir = tmp_path / "cli_ssl_out"

    exit_code = _run_fi2010_ssl_neural_benchmark_impl(
        config_path=config_path,
        processed_root=processed_root,
        out=out_dir,
        folds=["fold_1"],
        seeds=[7],
        lookbacks=[2],
        objective="masked_field",
        mask_probability=0.2,
        next_field_bucket_count=3,
        pretrain_epochs=1,
        max_epochs=1,
        batch_size=4,
        device="cpu",
        overwrite=False,
        fail_fast=False,
        write_full_predictions=True,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "FI-2010 SSL pretraining + fine-tuning runner" in captured.out
    assert "ssl artefacts:       written" in captured.out
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "comparison_summary.csv").is_file()
    assert (
        out_dir / "runs" / "fold_1_seed_7_lb2" / "pretrain" / "pretrained_encoder.pt"
    ).is_file()
