"""Tests for the FI-2010 local benchmark preparation layer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from chronoslob.experiments.artifacts import load_data_manifest
from chronoslob.experiments.fi2010_benchmark import (
    FI2010BenchmarkConfig,
    load_benchmark_config,
    prepare_fi2010_benchmark,
)
from chronoslob.utils.paths import project_root

CONFIG_PATH = project_root() / "configs" / "experiments" / "fi2010_midprice_h10.yaml"
TINY_FIXTURE_PATH = (
    project_root() / "tests" / "fixtures" / "fi2010" / "tiny_fi2010_like.csv"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _prepare_with_tiny_fixture(output_dir: Path) -> FI2010BenchmarkConfig:
    config = load_benchmark_config(CONFIG_PATH)
    prepare_fi2010_benchmark(
        config,
        data_path=TINY_FIXTURE_PATH,
        output_dir=output_dir,
        config_source_path=CONFIG_PATH,
    )
    return config


def test_placeholder_config_loads_with_safe_data_path() -> None:
    config = load_benchmark_config(CONFIG_PATH)

    assert config.experiment_name == "fi2010_midprice_h10"
    assert config.horizon == 10
    assert config.task_name == "midprice_direction"
    assert config.label_name == "label_10"
    assert config.data_path_is_placeholder


def test_placeholder_data_path_requires_explicit_data_path(tmp_path: Path) -> None:
    config = load_benchmark_config(CONFIG_PATH)
    output_dir = tmp_path / "prepare"

    with pytest.raises(FileNotFoundError, match="placeholder"):
        prepare_fi2010_benchmark(
            config,
            data_path=Path("<path-to-local-fi2010-file>"),
            output_dir=output_dir,
            config_source_path=CONFIG_PATH,
        )


def test_preparation_writes_expected_artefacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "prepare"

    _prepare_with_tiny_fixture(output_dir)

    expected_files = {
        "preparation_summary.json",
        "data_manifest.json",
        "label_summary.json",
        "split_summary.json",
        "validation_summary.json",
        "config.yaml",
    }
    actual_files = {path.name for path in output_dir.iterdir() if path.is_file()}
    assert expected_files.issubset(actual_files)


def test_results_and_predictions_are_not_written(tmp_path: Path) -> None:
    output_dir = tmp_path / "prepare"

    _prepare_with_tiny_fixture(output_dir)

    assert not (output_dir / "results.json").exists()
    assert not (output_dir / "predictions.csv").exists()
    assert not (output_dir / "predictions.parquet").exists()
    assert not (output_dir / "model_card.md").exists()


def test_data_manifest_validates_under_experiment_schema(tmp_path: Path) -> None:
    output_dir = tmp_path / "prepare"

    _prepare_with_tiny_fixture(output_dir)

    manifest = load_data_manifest(output_dir / "data_manifest.json")
    assert manifest.dataset_name == "FI-2010"
    assert manifest.label_name == "label_10"
    assert manifest.horizon == 10
    assert manifest.split_name == "temporal"
    assert manifest.source_kind == "local_file"
    assert manifest.row_count == 6
    assert manifest.source_sha256 is not None
    assert len(manifest.source_sha256) == 64


def test_split_summary_counts_are_consistent(tmp_path: Path) -> None:
    output_dir = tmp_path / "prepare"

    _prepare_with_tiny_fixture(output_dir)

    summary = _read_json(output_dir / "split_summary.json")
    n_rows = int(summary["n_rows"])
    assert summary["n_train"] + summary["n_validation"] + summary["n_test"] == n_rows
    assert n_rows == 6
    assert summary["split_name"] == "temporal"


def test_label_summary_has_finite_serialisable_distribution(tmp_path: Path) -> None:
    output_dir = tmp_path / "prepare"

    _prepare_with_tiny_fixture(output_dir)

    summary = _read_json(output_dir / "label_summary.json")
    assert summary["label_name"] == "label_10"
    assert summary["horizon"] == 10
    assert set(summary["class_counts"]) == set(summary["class_proportions"])
    total = sum(summary["class_counts"].values())
    assert total == summary["row_count"]
    for proportion in summary["class_proportions"].values():
        assert 0.0 <= float(proportion) <= 1.0


def test_preparation_summary_records_artefact_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "prepare"

    _prepare_with_tiny_fixture(output_dir)

    summary = _read_json(output_dir / "preparation_summary.json")
    assert summary["experiment_name"] == "fi2010_midprice_h10"
    assert summary["horizon"] == 10
    assert "data_manifest" in summary["artefacts"]
    assert "label_summary" in summary["artefacts"]
    assert "split_summary" in summary["artefacts"]
    assert "validation_summary" in summary["artefacts"]
    assert summary["artefacts"]["data_manifest"] == "data_manifest.json"


def test_missing_data_path_raises_clear_error(tmp_path: Path) -> None:
    config = load_benchmark_config(CONFIG_PATH)
    output_dir = tmp_path / "prepare"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        prepare_fi2010_benchmark(
            config,
            data_path=tmp_path / "missing.csv",
            output_dir=output_dir,
            config_source_path=CONFIG_PATH,
        )


def test_horizon_must_be_positive() -> None:
    with pytest.raises(ValueError, match="horizon"):
        FI2010BenchmarkConfig(
            experiment_name="fi2010_midprice_h10",
            horizon=0,
            output_dir="experiments/fi2010_midprice_h10",
            label_name="label_10",
        )


def test_label_name_must_be_in_label_columns() -> None:
    with pytest.raises(ValueError, match="label_name"):
        FI2010BenchmarkConfig(
            experiment_name="fi2010_midprice_h10",
            horizon=10,
            output_dir="experiments/fi2010_midprice_h10",
            label_name="not_in_list",
            label_columns=("label_10", "label_50"),
        )


def test_split_fractions_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        FI2010BenchmarkConfig(
            experiment_name="fi2010_midprice_h10",
            horizon=10,
            output_dir="experiments/fi2010_midprice_h10",
            label_name="label_10",
            train_fraction=0.5,
            validation_fraction=0.2,
            test_fraction=0.2,
        )


def test_cli_command_works_on_tiny_fixture(tmp_path: Path) -> None:
    output_dir = tmp_path / "prepare"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chronoslob.cli",
            "prepare-fi2010-benchmark",
            "--config",
            str(CONFIG_PATH),
            "--data-path",
            str(TINY_FIXTURE_PATH),
            "--out",
            str(output_dir),
        ],
        cwd=project_root(),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ChronosLOB FI-2010 benchmark preparation" in completed.stdout
    assert "results.json:        not written" in completed.stdout
    assert (output_dir / "data_manifest.json").is_file()
    assert (output_dir / "preparation_summary.json").is_file()
    assert not (output_dir / "results.json").exists()


def test_cli_command_fails_when_data_path_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "prepare"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chronoslob.cli",
            "prepare-fi2010-benchmark",
            "--config",
            str(CONFIG_PATH),
            "--data-path",
            str(tmp_path / "missing.csv"),
            "--out",
            str(output_dir),
        ],
        cwd=project_root(),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode != 0
    assert "File not found" in completed.stderr or "does not exist" in completed.stderr
