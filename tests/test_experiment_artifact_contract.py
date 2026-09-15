"""Tests for the experiment artefact contract."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from chronoslob.experiments.artifacts import (
    load_data_manifest,
    load_results,
    validate_experiment_directory,
)
from chronoslob.experiments.manifests import sha256_file
from chronoslob.experiments.schemas import (
    DataManifest,
    ExperimentConfigSummary,
    ModelResult,
)
from chronoslob.utils.paths import project_root

FIXTURE_DIR = (
    project_root()
    / "tests"
    / "fixtures"
    / "experiments"
    / "minimal_valid_experiment"
)


def _copy_fixture(tmp_path: Path) -> Path:
    destination = tmp_path / "experiment"
    shutil.copytree(FIXTURE_DIR, destination)
    return destination


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_json(path: Path, payload: dict[str, Any], *, allow_nan: bool = False) -> None:
    path.write_text(
        json.dumps(
            payload,
            allow_nan=allow_nan,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_valid_fixture_passes_contract_validation() -> None:
    report = validate_experiment_directory(FIXTURE_DIR, include_plots=False)

    assert report.is_valid
    assert report.missing_required == []
    assert load_data_manifest(FIXTURE_DIR).source_kind == "synthetic_fixture"
    assert load_results(FIXTURE_DIR).experiment_name == "synthetic_contract_fixture"


def test_missing_required_artefact_fails(tmp_path: Path) -> None:
    experiment = _copy_fixture(tmp_path)
    (experiment / "results.json").unlink()

    report = validate_experiment_directory(experiment, include_plots=False)

    assert not report.is_valid
    assert "results.json" in report.missing_required


@pytest.mark.parametrize("prediction_file", ["predictions.csv", "predictions.parquet"])
def test_predictions_csv_or_parquet_satisfies_optional_expectation(
    tmp_path: Path,
    prediction_file: str,
) -> None:
    experiment = _copy_fixture(tmp_path)
    prediction_path = experiment / prediction_file
    if prediction_file.endswith(".csv"):
        prediction_path.write_text("row_id,prediction\n1,0\n", encoding="utf-8")
    else:
        prediction_path.write_bytes(b"PAR1")

    report = validate_experiment_directory(experiment, include_plots=False)

    assert report.is_valid
    assert prediction_file in report.present_optional
    assert not any(
        "predictions.csv or predictions.parquet" in warning
        for warning in report.warnings
    )


def test_missing_optional_artefacts_warn_without_invalidating_fixture() -> None:
    report = validate_experiment_directory(FIXTURE_DIR, include_plots=False)

    assert report.is_valid
    assert any("predictions.csv or predictions.parquet" in warning for warning in report.warnings)
    assert any("calibration_bins.csv" in warning for warning in report.warnings)


@pytest.mark.parametrize("metric_value", [float("nan"), float("inf")])
def test_nan_and_infinite_metrics_are_rejected(
    tmp_path: Path,
    metric_value: float,
) -> None:
    experiment = _copy_fixture(tmp_path)
    results_path = experiment / "results.json"
    payload = _read_json(results_path)
    payload["model_results"][0]["metrics"]["macro_f1"] = metric_value
    _write_json(results_path, payload, allow_nan=True)

    report = validate_experiment_directory(experiment, include_plots=False)

    assert not report.is_valid
    result_status = next(
        status for status in report.artefact_statuses if status.path == "results.json"
    )
    assert result_status.exists
    assert result_status.message.startswith("invalid schema")


def test_naive_datetimes_are_rejected(tmp_path: Path) -> None:
    experiment = _copy_fixture(tmp_path)
    manifest_path = experiment / "data_manifest.json"
    payload = _read_json(manifest_path)
    payload["created_at"] = "2026-01-01T00:00:00"
    _write_json(manifest_path, payload)

    with pytest.raises(ValueError, match="timezone-aware"):
        load_data_manifest(experiment)


def test_negative_horizon_is_rejected() -> None:
    with pytest.raises(ValidationError, match="horizon"):
        DataManifest(
            dataset_name="synthetic_contract_fixture",
            dataset_variant="negative_horizon",
            source_kind="synthetic_fixture",
            source_path="tests/fixtures/experiments/minimal_valid_experiment",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            label_name="synthetic_midprice_direction",
            horizon=-1,
            split_name="synthetic_contract_split",
        )


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValidationError, match="seed"):
        ExperimentConfigSummary(
            experiment_name="synthetic_contract_fixture",
            task_name="midprice_direction",
            horizon=10,
            split_name="synthetic_contract_split",
            seed=-1,
            model_names=["synthetic_contract_model"],
            primary_metric="macro_f1",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_network_source_paths_are_rejected() -> None:
    with pytest.raises(ValidationError, match="local path"):
        DataManifest(
            dataset_name="synthetic_contract_fixture",
            dataset_variant="network_path",
            source_kind="local_file",
            source_path="https://example.invalid/data.csv",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            label_name="synthetic_midprice_direction",
            horizon=10,
            split_name="synthetic_contract_split",
        )


def test_result_artefact_paths_must_be_relative() -> None:
    with pytest.raises(ValidationError, match="relative"):
        ModelResult(
            model_name="synthetic_contract_model",
            split="test",
            horizon=10,
            metrics={"macro_f1": 0.5},
            artefacts={"predictions": "C:/absolute/predictions.csv"},
        )


def test_sha256_file_is_deterministic(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("chronoslob experiment contract\n", encoding="utf-8")

    assert sha256_file(sample) == sha256_file(sample)


def test_cli_inspection_command_works_on_minimal_fixture() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chronoslob.cli",
            "inspect-experiment-artifacts",
            "--experiment",
            str(FIXTURE_DIR),
        ],
        cwd=project_root(),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0
    assert "ChronosLOB experiment artefact inspection" in completed.stdout
    assert "valid:            yes" in completed.stdout
