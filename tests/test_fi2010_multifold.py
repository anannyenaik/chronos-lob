"""Tests for the FI-2010 multi-fold preparation layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from chronoslob.cli import (
    _inspect_fi2010_multifold_impl,
    _prepare_fi2010_multifold_impl,
)
from chronoslob.data.fi2010_official import (
    OFFICIAL_FI2010_HANDCRAFTED_ROW_COUNT,
    OFFICIAL_FI2010_LABEL_HORIZONS,
    OFFICIAL_FI2010_LEVEL_COUNT,
    OFFICIAL_FI2010_ROW_COUNT,
)
from chronoslob.experiments.fi2010_multifold import (
    discover_fold_files,
    inspect_multifold_files,
    load_multifold_config,
    prepare_multifold,
)
from chronoslob.utils.audit import (
    AuditStatus,
    check_no_forbidden_claims,
    check_public_release_wording,
)
from chronoslob.utils.paths import project_root

# ---------------------------------------------------------------------------
# Synthetic FI-2010 fixture helpers
# ---------------------------------------------------------------------------


def _write_synthetic_official_file(
    path: Path,
    *,
    n_snapshots: int = 4,
) -> Path:
    """Write a tiny synthetic FI-2010-shaped .txt matrix at ``path``."""
    rows: list[list[float | int | str]] = []
    for level_index in range(OFFICIAL_FI2010_LEVEL_COUNT):
        level = level_index + 1
        ask_price_row: list[float | int | str] = []
        ask_size_row: list[float | int | str] = []
        bid_price_row: list[float | int | str] = []
        bid_size_row: list[float | int | str] = []
        for sample in range(n_snapshots):
            ask_price = 100.0 + 0.10 * level + 0.001 * sample
            bid_price = 100.0 - 0.10 * level - 0.001 * sample
            ask_size_row.append(10 + level + sample)
            bid_size_row.append(11 + level + sample)
            ask_price_row.append(round(ask_price, 5))
            bid_price_row.append(round(bid_price, 5))
        rows.append(ask_price_row)
        rows.append(ask_size_row)
        rows.append(bid_price_row)
        rows.append(bid_size_row)
    for feature_index in range(OFFICIAL_FI2010_HANDCRAFTED_ROW_COUNT):
        rows.append(
            [
                round(0.001 * (feature_index + 1) + 0.0001 * sample, 6)
                for sample in range(n_snapshots)
            ],
        )
    label_cycle = ("1", "2", "3")
    for horizon_index in range(len(OFFICIAL_FI2010_LABEL_HORIZONS)):
        rows.append(
            [
                label_cycle[(sample + horizon_index) % len(label_cycle)]
                for sample in range(n_snapshots)
            ],
        )
    assert len(rows) == OFFICIAL_FI2010_ROW_COUNT
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(" ".join(str(value) for value in row))
            handle.write("\n")
    return path


def _build_synthetic_extracted_root(
    tmp_path: Path,
    *,
    folds: tuple[int, ...] = (1, 2),
    train_snapshots: int = 3,
    test_snapshots: int = 2,
) -> Path:
    """Build a tiny extracted FI-2010 dataset root containing two synthetic folds."""
    extracted_root = tmp_path / "extracted" / "BenchmarkDatasets"
    base = extracted_root / "NoAuction" / "1.NoAuction_Zscore"
    for fold in folds:
        fold_dir = base / f"NoAuction_Zscore_CF_{fold}"
        train_path = fold_dir / f"Train_Dst_NoAuction_ZScore_CF_{fold}.txt"
        test_path = fold_dir / f"Test_Dst_NoAuction_ZScore_CF_{fold}.txt"
        _write_synthetic_official_file(train_path, n_snapshots=train_snapshots)
        _write_synthetic_official_file(test_path, n_snapshots=test_snapshots)
    return extracted_root


def _write_synthetic_multifold_config(
    tmp_path: Path,
    *,
    folds: tuple[int, ...] = (1, 2),
    horizon: int = 10,
    train_value: str = "train",
    test_value: str = "test",
    processed_placeholder: str = "data/processed/fi2010",
) -> Path:
    """Write a tiny multi-fold preparation YAML config under ``tmp_path``."""
    payload = {
        "study_name": "fi2010_multifold_test_study",
        "dataset_name": "FI-2010",
        "task_name": "midprice_direction",
        "protocol_reference": "docs/RESEARCH_PROTOCOL.md",
        "implementation_note_reference": "reports/10_10_research_protocol.md",
        "executable": False,
        "notes": "synthetic test config",
        "local_data_root_path": processed_placeholder,
        "combined_csv_filename_template": "fold{fold}_combined.csv",
        "preparation": {
            "extracted_dataset_root_placeholder": (
                "data/raw/fi2010/extracted/BenchmarkDatasets"
            ),
            "processed_output_root_placeholder": processed_placeholder,
            "combined_csv_filename_template": "fold{fold}_combined.csv",
            "train_filename_template": (
                "NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_CF_{fold}/"
                "Train_Dst_NoAuction_ZScore_CF_{fold}.txt"
            ),
            "test_filename_template": (
                "NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_CF_{fold}/"
                "Test_Dst_NoAuction_ZScore_CF_{fold}.txt"
            ),
            "fold_overrides": {},
            "compute_source_hashes": True,
            "local_only_note": (
                "Raw and processed FI-2010 data are local-only and gitignored."
            ),
        },
        "folds": list(folds),
        "official_split": {
            "split_column": "split",
            "train_value": train_value,
            "test_value": test_value,
            "validation_fraction_within_train": 0.15,
        },
        "target": {
            "horizon": horizon,
            "label_column": f"label_{horizon}",
            "label_columns": [f"label_{h}" for h in (10, 50, 100)],
            "class_direction_map": {"1": 1, "2": -1, "3": 0},
        },
        "models": {"classical": ["majority"], "neural": [], "gated": []},
        "seeds": [0, 1, 2],
        "metrics": {"predictive": ["macro_f1"]},
        "calibration": {"enabled": True, "n_bins": 10, "fit_on": "train"},
        "execution_sensitivity": {
            "enabled": True,
            "confidence_thresholds": [0.0, 0.5],
            "cost_bps": [0.0, 1.0],
            "latency_steps": [0, 1],
            "return_proxy": {
                "kind": "mid_forward_return",
                "bid_price_column": "bid_price_1",
                "ask_price_column": "ask_price_1",
            },
        },
        "ablations": {"groups": []},
        "artefacts": {
            "experiment_root_path": "experiments/fi2010_multifold",
            "per_fold_subdir_template": "fold_{fold}",
            "per_seed_subdir_template": "seed_{seed}",
            "ablation_root_path": "experiments/fi2010_multifold_ablations",
            "systems_root_path": "experiments/fi2010_multifold_systems",
            "report_path": "reports/chronoslob_multifold_report.md",
        },
        "data_handling": {
            "download_fi2010": False,
            "commit_raw_data": False,
            "commit_processed_csv": False,
            "commit_predictions": False,
            "commit_intermediate_matrices": False,
            "expects_local_acquisition": True,
            "acquisition_reference": "docs/FI2010_DATA_ACQUISITION.md",
        },
        "claim_boundaries": {
            "no_profitable_strategy_claim": True,
            "no_market_beating_claim": True,
            "no_state_of_the_art_claim": True,
            "no_foundation_model_claim": True,
            "no_ssl_result_claim_until_gate_satisfied": True,
            "no_live_tradability_claim": True,
        },
    }
    config_path = tmp_path / "fi2010_multifold_test.yaml"
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return config_path


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_load_multifold_config_returns_validated_model(tmp_path: Path) -> None:
    config_path = _write_synthetic_multifold_config(tmp_path)
    config = load_multifold_config(config_path)

    assert config.study_name == "fi2010_multifold_test_study"
    assert config.dataset_name == "FI-2010"
    assert config.folds == (1, 2)
    assert config.target_horizon == 10
    assert config.split_column == "split"
    assert config.train_value == "train"
    assert config.test_value == "test"
    assert "label_10" in config.label_columns
    assert config.preparation.compute_source_hashes is True


def test_load_multifold_config_rejects_missing_preparation_block(
    tmp_path: Path,
) -> None:
    config_path = _write_synthetic_multifold_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    del payload["preparation"]
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="preparation"):
        load_multifold_config(config_path)


def test_load_multifold_config_rejects_invalid_target_horizon(
    tmp_path: Path,
) -> None:
    config_path = _write_synthetic_multifold_config(tmp_path, horizon=10)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["target"]["horizon"] = 7  # not in OFFICIAL_FI2010_LABEL_HORIZONS
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="official FI-2010"):
        load_multifold_config(config_path)


def test_repo_multifold_config_exposes_preparation_block() -> None:
    config = load_multifold_config(
        project_root() / "configs" / "experiments" / "fi2010_multifold.yaml",
    )
    assert config.folds == (1, 2, 3, 4, 5)
    assert config.preparation.compute_source_hashes is True
    assert "{fold}" in config.preparation.train_filename_template
    assert "{fold}" in config.preparation.test_filename_template


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discover_fold_files_finds_two_synthetic_folds(tmp_path: Path) -> None:
    config_path = _write_synthetic_multifold_config(tmp_path)
    config = load_multifold_config(config_path)
    extracted_root = _build_synthetic_extracted_root(tmp_path)

    plans = discover_fold_files(
        config,
        extracted_root=extracted_root,
        processed_root=tmp_path / "processed",
    )

    assert [plan.fold for plan in plans] == [1, 2]
    for plan in plans:
        assert plan.is_ready is True
        assert plan.train_path.is_file()
        assert plan.test_path.is_file()
        assert plan.combined_output_path.name == f"fold{plan.fold}_combined.csv"


def test_discover_fold_files_subset_only_returns_requested_folds(
    tmp_path: Path,
) -> None:
    config_path = _write_synthetic_multifold_config(tmp_path)
    config = load_multifold_config(config_path)
    extracted_root = _build_synthetic_extracted_root(tmp_path)

    plans = discover_fold_files(
        config,
        extracted_root=extracted_root,
        processed_root=tmp_path / "processed",
        folds=[2],
    )
    assert [plan.fold for plan in plans] == [2]


def test_discover_fold_files_marks_missing_files(tmp_path: Path) -> None:
    config_path = _write_synthetic_multifold_config(tmp_path)
    config = load_multifold_config(config_path)
    extracted_root = _build_synthetic_extracted_root(tmp_path)

    missing_train = (
        extracted_root
        / "NoAuction"
        / "1.NoAuction_Zscore"
        / "NoAuction_Zscore_CF_2"
        / "Train_Dst_NoAuction_ZScore_CF_2.txt"
    )
    missing_train.unlink()

    plans = inspect_multifold_files(
        config,
        extracted_root=extracted_root,
        processed_root=tmp_path / "processed",
    )
    fold2 = next(plan for plan in plans if plan.fold == 2)
    assert fold2.train_present is False
    assert fold2.test_present is True
    assert fold2.is_ready is False


def test_discover_fold_files_rejects_unknown_requested_fold(
    tmp_path: Path,
) -> None:
    config_path = _write_synthetic_multifold_config(tmp_path)
    config = load_multifold_config(config_path)
    extracted_root = _build_synthetic_extracted_root(tmp_path)

    with pytest.raises(ValueError, match="not configured"):
        discover_fold_files(
            config,
            extracted_root=extracted_root,
            processed_root=tmp_path / "processed",
            folds=[9],
        )


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def _run_preparation(
    tmp_path: Path, *, overwrite: bool = False, folds: list[int] | None = None
):
    config_path = _write_synthetic_multifold_config(tmp_path)
    config = load_multifold_config(config_path)
    extracted_root = _build_synthetic_extracted_root(tmp_path)
    processed_root = tmp_path / "processed"
    output_dir = tmp_path / "out"
    result = prepare_multifold(
        config=config,
        config_source_path=config_path,
        extracted_root=extracted_root,
        processed_root=processed_root,
        output_dir=output_dir,
        folds=folds,
        overwrite=overwrite,
    )
    return result, config_path, processed_root, output_dir


def test_prepare_multifold_writes_manifests_and_combined_csvs(
    tmp_path: Path,
) -> None:
    result, _config_path, processed_root, output_dir = _run_preparation(tmp_path)

    summary_path = output_dir / "summary.json"
    assert summary_path.is_file()
    folds_dir = output_dir / "folds"
    assert folds_dir.is_dir()

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["folds_prepared"] == [1, 2]
    assert summary_payload["folds_skipped"] == []
    assert summary_payload["preparation_version"].startswith(
        "phase-c/fi2010-multifold-preparation/"
    )
    assert "1" in summary_payload["per_fold_split_counts"]
    assert summary_payload["per_fold_split_counts"]["1"]["train"] == 3
    assert summary_payload["per_fold_split_counts"]["1"]["test"] == 2
    assert summary_payload["per_fold_row_counts"]["1"] == 5

    for manifest in result.fold_manifests:
        manifest_path = folds_dir / f"fold_{manifest.fold}_manifest.json"
        assert manifest_path.is_file()
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest_payload["fold"] == manifest.fold
        assert manifest_payload["combined_row_count"] == 5
        assert manifest_payload["split_counts"] == {"train": 3, "test": 2}
        assert manifest_payload["train_source"]["sha256"]
        assert manifest_payload["test_source"]["sha256"]

    combined_csv = processed_root / "fold1_combined.csv"
    assert combined_csv.is_file()
    text = combined_csv.read_text(encoding="utf-8")
    header_line, *data_lines = text.splitlines()
    header = header_line.split(",")
    assert "split" in header
    split_index = header.index("split")
    split_values = [line.split(",")[split_index] for line in data_lines]
    assert split_values.count("train") == 3
    assert split_values.count("test") == 2
    # train rows come before test rows in the combined file
    assert split_values[:3] == ["train", "train", "train"]
    assert split_values[3:] == ["test", "test"]


def test_prepare_multifold_subset_skips_unrequested_folds(tmp_path: Path) -> None:
    _result, _config_path, processed_root, output_dir = _run_preparation(
        tmp_path, folds=[1]
    )
    summary_payload = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary_payload["folds_requested"] == [1]
    assert summary_payload["folds_prepared"] == [1]
    assert summary_payload["folds_skipped"] == [2]
    assert (processed_root / "fold1_combined.csv").is_file()
    assert not (processed_root / "fold2_combined.csv").exists()
    assert (output_dir / "folds" / "fold_1_manifest.json").is_file()
    assert not (output_dir / "folds" / "fold_2_manifest.json").exists()


def test_prepare_multifold_refuses_overwrite_by_default(tmp_path: Path) -> None:
    _run_preparation(tmp_path)
    with pytest.raises(FileExistsError):
        _run_preparation(tmp_path, overwrite=False)


def test_prepare_multifold_overwrite_succeeds(tmp_path: Path) -> None:
    _run_preparation(tmp_path)
    _result, _config_path, processed_root, output_dir = _run_preparation(
        tmp_path, overwrite=True
    )
    summary_payload = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary_payload["folds_prepared"] == [1, 2]
    assert (processed_root / "fold1_combined.csv").is_file()
    assert (processed_root / "fold2_combined.csv").is_file()


def test_prepare_multifold_reports_missing_fold_file(tmp_path: Path) -> None:
    config_path = _write_synthetic_multifold_config(tmp_path)
    config = load_multifold_config(config_path)
    extracted_root = _build_synthetic_extracted_root(tmp_path)
    (
        extracted_root
        / "NoAuction"
        / "1.NoAuction_Zscore"
        / "NoAuction_Zscore_CF_2"
        / "Test_Dst_NoAuction_ZScore_CF_2.txt"
    ).unlink()
    with pytest.raises(FileNotFoundError, match="fold 2"):
        prepare_multifold(
            config=config,
            config_source_path=config_path,
            extracted_root=extracted_root,
            processed_root=tmp_path / "processed",
            output_dir=tmp_path / "out",
        )


# ---------------------------------------------------------------------------
# CLI impls
# ---------------------------------------------------------------------------


def test_inspect_fi2010_multifold_cli_reports_ready_folds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_synthetic_multifold_config(tmp_path)
    extracted_root = _build_synthetic_extracted_root(tmp_path)
    exit_code = _inspect_fi2010_multifold_impl(
        config_path=config_path,
        extracted_root=extracted_root,
        processed_root=tmp_path / "processed",
        folds=None,
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "FI-2010 multi-fold inspection" in captured.out
    assert "fold 1: ready" in captured.out
    assert "fold 2: ready" in captured.out
    assert "outputs:             not written" in captured.out


def test_inspect_fi2010_multifold_cli_reports_missing_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_synthetic_multifold_config(tmp_path)
    extracted_root = _build_synthetic_extracted_root(tmp_path)
    (
        extracted_root
        / "NoAuction"
        / "1.NoAuction_Zscore"
        / "NoAuction_Zscore_CF_1"
        / "Train_Dst_NoAuction_ZScore_CF_1.txt"
    ).unlink()
    exit_code = _inspect_fi2010_multifold_impl(
        config_path=config_path,
        extracted_root=extracted_root,
        processed_root=tmp_path / "processed",
        folds=None,
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "fold 1: missing" in captured.out
    assert "MISSING" in captured.out


def test_prepare_fi2010_multifold_cli_writes_artefacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_synthetic_multifold_config(tmp_path)
    extracted_root = _build_synthetic_extracted_root(tmp_path)
    processed_root = tmp_path / "processed"
    out_dir = tmp_path / "out"
    exit_code = _prepare_fi2010_multifold_impl(
        config_path=config_path,
        extracted_root=extracted_root,
        processed_root=processed_root,
        out=out_dir,
        folds=None,
        overwrite=False,
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "FI-2010 multi-fold preparation" in captured.out
    assert "prepared folds:      [1, 2]" in captured.out
    assert "predictions:         not written" in captured.out
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "folds" / "fold_1_manifest.json").is_file()
    assert (processed_root / "fold1_combined.csv").is_file()


def test_prepare_fi2010_multifold_cli_blocks_overwrite_without_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_synthetic_multifold_config(tmp_path)
    extracted_root = _build_synthetic_extracted_root(tmp_path)
    processed_root = tmp_path / "processed"
    out_dir = tmp_path / "out"
    first = _prepare_fi2010_multifold_impl(
        config_path=config_path,
        extracted_root=extracted_root,
        processed_root=processed_root,
        out=out_dir,
        folds=None,
        overwrite=False,
    )
    assert first == 0
    capsys.readouterr()
    second = _prepare_fi2010_multifold_impl(
        config_path=config_path,
        extracted_root=extracted_root,
        processed_root=processed_root,
        out=out_dir,
        folds=None,
        overwrite=False,
    )
    captured = capsys.readouterr()
    assert second == 1
    assert "Refusing to overwrite" in captured.err


def test_prepare_fi2010_multifold_cli_reports_missing_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = _prepare_fi2010_multifold_impl(
        config_path=tmp_path / "missing.yaml",
        extracted_root=tmp_path,
        processed_root=tmp_path / "processed",
        out=tmp_path / "out",
        folds=None,
        overwrite=False,
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "File not found" in captured.err


# ---------------------------------------------------------------------------
# Public claim boundaries for new docs
# ---------------------------------------------------------------------------


def test_new_multifold_doc_does_not_contain_forbidden_claims() -> None:
    result = check_no_forbidden_claims(
        project_root(),
        scan_paths=(Path("docs") / "FI2010_MULTIFOLD_PROTOCOL.md",),
    )
    assert result.status == AuditStatus.PASS


def test_new_multifold_doc_passes_public_release_wording_scan() -> None:
    result = check_public_release_wording(
        project_root(),
        scan_paths=(Path("docs") / "FI2010_MULTIFOLD_PROTOCOL.md",),
    )
    assert result.status == AuditStatus.PASS
