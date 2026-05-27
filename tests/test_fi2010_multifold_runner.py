"""Tests for the FI-2010 multi-fold classical runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from chronoslob.cli import _run_fi2010_multifold_classical_impl
from chronoslob.experiments.fi2010_multifold import load_multifold_config, prepare_multifold
from chronoslob.experiments.fi2010_multifold_runner import (
    CLASSICAL_MULTIFOLD_MODELS,
    load_multifold_classical_config,
    run_fi2010_multifold_classical,
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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _prepare_synthetic_folds(
    tmp_path: Path,
    *,
    folds: tuple[int, ...] = (1, 2),
    train_snapshots: int = 8,
    test_snapshots: int = 6,
) -> tuple[Path, Path, Path]:
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
    return config_path, processed_root, tmp_path / "classical_out"


def test_multifold_classical_config_parses_repo_config() -> None:
    config = load_multifold_classical_config(
        project_root() / "configs" / "experiments" / "fi2010_multifold.yaml",
    )

    assert config.folds == (1, 2, 3, 4, 5)
    assert config.classical_models == CLASSICAL_MULTIFOLD_MODELS
    assert config.label_name == "label_10"
    assert config.validation_fraction_within_train == 0.15
    assert config.seeds == (0,)


def test_runner_detects_missing_processed_fold_csv(tmp_path: Path) -> None:
    config_path = _write_synthetic_multifold_config(tmp_path, folds=(1,))

    with pytest.raises(FileNotFoundError, match="fold 1 processed CSV is missing"):
        run_fi2010_multifold_classical(
            config_path,
            processed_root=tmp_path / "processed",
            out_dir=tmp_path / "out",
            models=["majority"],
            folds=[1],
        )


def test_runner_handles_two_tiny_folds_and_writes_summaries(tmp_path: Path) -> None:
    config_path, processed_root, out_dir = _prepare_synthetic_folds(tmp_path)

    summary = run_fi2010_multifold_classical(
        config_path,
        processed_root=processed_root,
        out_dir=out_dir,
        models=["majority", "logistic"],
        folds=[1, 2],
    )

    assert summary.fold_count == 2
    assert summary.model_count == 2
    assert summary.failure_count == 0
    assert (out_dir / "results_by_fold.csv").is_file()
    assert (out_dir / "results_summary.csv").is_file()
    assert (out_dir / "calibration_summary.csv").is_file()
    assert (out_dir / "execution_summary.csv").is_file()
    assert (out_dir / "model_failures.json").is_file()
    assert (out_dir / "folds" / "fold_1" / "results.json").is_file()
    assert (out_dir / "folds" / "fold_1" / "confusion_matrix.json").is_file()
    assert (out_dir / "folds" / "fold_1" / "calibration_bins.csv").is_file()
    assert (out_dir / "folds" / "fold_1" / "execution_sensitivity.csv").is_file()
    assert (out_dir / "folds" / "fold_1" / "model_card.md").is_file()

    results = pd.read_csv(out_dir / "results_by_fold.csv")
    assert set(results["split"]) == {"validation", "test"}
    assert set(results["status"]) == {"ok"}
    assert {"accuracy", "macro_f1", "mcc", "brier_score", "ece"}.issubset(
        results.columns,
    )

    aggregate = pd.read_csv(out_dir / "results_summary.csv")
    assert {"accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"}.issubset(
        aggregate.columns,
    )


def test_runner_handles_one_fold_subset(tmp_path: Path) -> None:
    config_path, processed_root, out_dir = _prepare_synthetic_folds(tmp_path)

    summary = run_fi2010_multifold_classical(
        config_path,
        processed_root=processed_root,
        out_dir=out_dir,
        models=["majority"],
        folds=[1],
    )

    assert summary.folds_completed == [1]
    assert (out_dir / "folds" / "fold_1" / "results.json").is_file()
    assert not (out_dir / "folds" / "fold_2").exists()


def test_unsupported_model_names_fail_clearly(tmp_path: Path) -> None:
    config_path, processed_root, out_dir = _prepare_synthetic_folds(
        tmp_path,
        folds=(1,),
    )

    with pytest.raises(ValueError, match="not supported"):
        run_fi2010_multifold_classical(
            config_path,
            processed_root=processed_root,
            out_dir=out_dir,
            models=["deeplob_style"],
            folds=[1],
        )


def test_overwrite_protection_blocks_existing_output(tmp_path: Path) -> None:
    config_path, processed_root, out_dir = _prepare_synthetic_folds(
        tmp_path,
        folds=(1,),
    )
    run_fi2010_multifold_classical(
        config_path,
        processed_root=processed_root,
        out_dir=out_dir,
        models=["majority"],
        folds=[1],
    )

    with pytest.raises(FileExistsError, match="non-empty output directory"):
        run_fi2010_multifold_classical(
            config_path,
            processed_root=processed_root,
            out_dir=out_dir,
            models=["majority"],
            folds=[1],
        )


def test_summary_json_records_fold_and_model_counts(tmp_path: Path) -> None:
    config_path, processed_root, out_dir = _prepare_synthetic_folds(tmp_path)

    run_fi2010_multifold_classical(
        config_path,
        processed_root=processed_root,
        out_dir=out_dir,
        models=["majority"],
        folds=[1, 2],
    )

    payload = _read_json(out_dir / "summary.json")
    assert payload["fold_count"] == 2
    assert payload["model_count"] == 1
    assert payload["full_predictions_written"] is False


def test_official_split_semantics_are_preserved(tmp_path: Path) -> None:
    config_path, processed_root, out_dir = _prepare_synthetic_folds(
        tmp_path,
        folds=(1,),
        train_snapshots=8,
        test_snapshots=5,
    )

    run_fi2010_multifold_classical(
        config_path,
        processed_root=processed_root,
        out_dir=out_dir,
        models=["majority"],
        folds=[1],
    )

    fold_payload = _read_json(out_dir / "folds" / "fold_1" / "results.json")
    split = fold_payload["split_summary"]
    assert split["split_method"] == "official_column"
    assert split["official_train_rows"] == 8
    assert split["official_test_rows"] == 5
    assert split["n_train"] + split["n_validation"] == 8
    assert split["n_test"] == 5
    assert split["validation_start_index"] >= split["train_end_index"]
    assert split["test_start_index"] == 8


def test_invalid_split_labels_fail_clearly(tmp_path: Path) -> None:
    config_path, processed_root, out_dir = _prepare_synthetic_folds(
        tmp_path,
        folds=(1,),
    )
    fold_csv = processed_root / "fold1_combined.csv"
    frame = pd.read_csv(fold_csv)
    frame.loc[0, "split"] = "holdout"
    frame.to_csv(fold_csv, index=False)

    with pytest.raises(ValueError, match="invalid labels"):
        run_fi2010_multifold_classical(
            config_path,
            processed_root=processed_root,
            out_dir=out_dir,
            models=["majority"],
            folds=[1],
        )


def test_full_predictions_are_not_written_by_default(tmp_path: Path) -> None:
    config_path, processed_root, out_dir = _prepare_synthetic_folds(
        tmp_path,
        folds=(1,),
    )

    run_fi2010_multifold_classical(
        config_path,
        processed_root=processed_root,
        out_dir=out_dir,
        models=["majority"],
        folds=[1],
    )

    prediction_files = [
        path
        for path in out_dir.rglob("*")
        if path.name.startswith("predictions")
    ]
    assert prediction_files == []


def test_multifold_classical_cli_impl_runs_on_tiny_fold(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path, processed_root, out_dir = _prepare_synthetic_folds(
        tmp_path,
        folds=(1,),
    )

    exit_code = _run_fi2010_multifold_classical_impl(
        config_path=config_path,
        processed_root=processed_root,
        out=out_dir,
        models=["majority"],
        folds=[1],
        overwrite=False,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "FI-2010 multi-fold classical runner" in captured.out
    assert "full predictions:    not written" in captured.out
    assert (out_dir / "summary.json").is_file()


def test_multifold_classical_doc_avoids_forbidden_public_claims() -> None:
    scan_path = Path("docs") / "FI2010_MULTIFOLD_CLASSICAL.md"

    claims = check_no_forbidden_claims(project_root(), scan_paths=(scan_path,))
    wording = check_public_release_wording(project_root(), scan_paths=(scan_path,))

    assert claims.status == AuditStatus.PASS
    assert wording.status == AuditStatus.PASS
