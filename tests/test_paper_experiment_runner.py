"""Tests for the paper experiment runner."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from chronoslob.experiments.artifacts import (
    load_data_manifest,
    load_results,
    validate_experiment_directory,
)
from chronoslob.experiments.paper_runner import (
    PAPER_RUNNER_VERSION,
    SUPPORTED_PAPER_MODELS,
    run_paper_experiment,
)
from chronoslob.utils.paths import project_root

CONFIG_PATH = project_root() / "configs" / "experiments" / "fi2010_midprice_h10.yaml"
TINY_FIXTURE_PATH = (
    project_root() / "tests" / "fixtures" / "fi2010" / "tiny_fi2010_like.csv"
)

_REQUIRED_FILES: tuple[str, ...] = (
    "config.yaml",
    "data_manifest.json",
    "results.json",
    "predictions.csv",
    "model_card.md",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _run_on_tiny_fixture(
    output_dir: Path,
    *,
    models: tuple[str, ...] = ("majority",),
    overwrite: bool = False,
) -> Any:
    return run_paper_experiment(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        models=list(models),
        overwrite=overwrite,
    )


def test_runner_writes_required_artefacts_on_tiny_fixture(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_experiment"

    summary = _run_on_tiny_fixture(output_dir)

    assert summary.experiment_name == "fi2010_midprice_h10"
    assert summary.task_name == "midprice_direction"
    assert summary.horizon == 10
    assert summary.runner_version == PAPER_RUNNER_VERSION
    assert summary.is_fixture is True
    for required_file in _REQUIRED_FILES:
        assert (output_dir / required_file).is_file(), required_file


def test_runner_output_passes_validate_experiment_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_experiment"

    _run_on_tiny_fixture(output_dir)

    report = validate_experiment_directory(output_dir, include_plots=True)
    assert report.is_valid
    assert report.missing_required == []


def test_results_json_loads_as_experiment_results_schema(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_experiment"

    _run_on_tiny_fixture(output_dir)

    results = load_results(output_dir / "results.json")
    assert results.experiment_name == "fi2010_midprice_h10"
    assert results.task_name == "midprice_direction"
    assert len(results.model_results) == 1
    model_result = results.model_results[0]
    assert model_result.split == "test"
    assert "accuracy" in model_result.metrics
    assert "macro_f1" in model_result.metrics
    assert "predictive" not in model_result.metrics
    assert results.evidence_streams.predictive
    assert results.evidence_streams.calibration
    assert results.evidence_streams.execution


def test_data_manifest_records_local_file_provenance(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_experiment"

    _run_on_tiny_fixture(output_dir)

    manifest = load_data_manifest(output_dir / "data_manifest.json")
    assert manifest.dataset_name == "FI-2010"
    assert manifest.label_name == "label_10"
    assert manifest.horizon == 10
    assert manifest.source_kind == "local_file"
    assert manifest.source_sha256 is not None
    assert len(manifest.source_sha256) == 64


def test_predictions_csv_has_expected_columns_and_finite_probabilities(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "paper_experiment"

    _run_on_tiny_fixture(output_dir)

    frame = pd.read_csv(output_dir / "predictions.csv")
    for column in (
        "row_index",
        "split",
        "label",
        "prediction",
        "model_name",
        "confidence",
    ):
        assert column in frame.columns, column
    probability_columns = [
        column for column in frame.columns if column.startswith("probability_")
    ]
    assert probability_columns, "predictions.csv must include probability columns"
    for column in probability_columns:
        for value in frame[column].tolist():
            assert math.isfinite(float(value))
    assert (frame["split"] == "test").all()


def test_majority_model_predictions_have_normalised_probabilities(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "paper_experiment"

    _run_on_tiny_fixture(output_dir)

    frame = pd.read_csv(output_dir / "predictions.csv")
    probability_columns = [
        column for column in frame.columns if column.startswith("probability_")
    ]
    for _, row in frame.iterrows():
        total = sum(float(row[column]) for column in probability_columns)
        assert math.isclose(total, 1.0, abs_tol=1e-6)


def test_overwrite_protection_blocks_existing_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_experiment"
    output_dir.mkdir()
    (output_dir / "sentinel.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty output directory"):
        _run_on_tiny_fixture(output_dir, overwrite=False)
    assert (output_dir / "sentinel.txt").is_file()


def test_overwrite_flag_replaces_existing_artefacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_experiment"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("old", encoding="utf-8")

    _run_on_tiny_fixture(output_dir, overwrite=True)

    assert not (output_dir / "stale.txt").exists()
    for required_file in _REQUIRED_FILES:
        assert (output_dir / required_file).is_file(), required_file


def test_missing_data_path_raises_file_not_found(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_experiment"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        run_paper_experiment(
            config_path=CONFIG_PATH,
            data_path=tmp_path / "missing.csv",
            out_dir=output_dir,
            models=["majority"],
        )
    assert not output_dir.exists() or not any(output_dir.iterdir())


def test_unsupported_model_name_raises_clear_error(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_experiment"

    with pytest.raises(ValueError, match="unsupported model"):
        run_paper_experiment(
            config_path=CONFIG_PATH,
            data_path=TINY_FIXTURE_PATH,
            out_dir=output_dir,
            models=["transformer"],
        )
    assert not output_dir.exists() or not any(output_dir.iterdir())


def test_runner_supports_logistic_when_requested(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_experiment"

    summary = _run_on_tiny_fixture(
        output_dir,
        models=("majority", "logistic"),
    )

    assert summary.models_run == ["majority", "logistic"]
    frame = pd.read_csv(output_dir / "predictions.csv")
    assert set(frame["model_name"].unique()) == {
        "majority_class",
        "logistic_regression",
    }


def test_runner_refuses_models_without_majority(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_experiment"

    with pytest.raises(ValueError, match="majority"):
        run_paper_experiment(
            config_path=CONFIG_PATH,
            data_path=TINY_FIXTURE_PATH,
            out_dir=output_dir,
            models=["logistic"],
        )


def test_no_plot_files_are_required_for_validity(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_experiment"

    _run_on_tiny_fixture(output_dir)

    plots_dir = output_dir / "plots"
    if plots_dir.exists():
        assert not any(plots_dir.iterdir())
    report = validate_experiment_directory(output_dir, include_plots=True)
    assert report.is_valid


def test_tiny_fixture_model_card_does_not_claim_benchmark_evidence(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "paper_experiment"

    _run_on_tiny_fixture(output_dir)

    text = (output_dir / "model_card.md").read_text(encoding="utf-8")
    assert "synthetic fixture smoke" in text.lower()
    forbidden_phrases = [
        "profitable",
        " ".join(("production", "trading")),
        " ".join(("live", "trading")),
        "-".join(("market", "beating")),
        " ".join(("guaranteed", "alpha")),
    ]
    lowered = text.lower()
    for phrase in forbidden_phrases:
        assert phrase not in lowered, phrase


def test_supported_models_constant_reports_majority(tmp_path: Path) -> None:
    assert "majority" in SUPPORTED_PAPER_MODELS
    assert "logistic" in SUPPORTED_PAPER_MODELS
    assert "random_forest" not in SUPPORTED_PAPER_MODELS
    assert "deeplob" not in SUPPORTED_PAPER_MODELS
    assert "transformer" not in SUPPORTED_PAPER_MODELS


def test_cli_command_works_on_tiny_fixture(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_experiment"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chronoslob.cli",
            "run-paper-experiment",
            "--config",
            str(CONFIG_PATH),
            "--data-path",
            str(TINY_FIXTURE_PATH),
            "--out",
            str(output_dir),
            "--models",
            "majority",
            "--overwrite",
        ],
        cwd=project_root(),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ChronosLOB paper experiment runner" in completed.stdout
    assert "artefact validation: valid" in completed.stdout
    for required_file in _REQUIRED_FILES:
        assert (output_dir / required_file).is_file(), required_file


def test_cli_command_fails_when_data_path_missing(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_experiment"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chronoslob.cli",
            "run-paper-experiment",
            "--config",
            str(CONFIG_PATH),
            "--data-path",
            str(tmp_path / "missing.csv"),
            "--out",
            str(output_dir),
            "--models",
            "majority",
            "--overwrite",
        ],
        cwd=project_root(),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode != 0
    assert (
        "File not found" in completed.stderr
        or "does not exist" in completed.stderr
    )


def test_cli_command_rejects_unsupported_model(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_experiment"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chronoslob.cli",
            "run-paper-experiment",
            "--config",
            str(CONFIG_PATH),
            "--data-path",
            str(TINY_FIXTURE_PATH),
            "--out",
            str(output_dir),
            "--models",
            "transformer",
            "--overwrite",
        ],
        cwd=project_root(),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode != 0
    assert "unsupported model" in completed.stderr.lower()


def test_runner_summary_records_split_counts(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_experiment"

    _run_on_tiny_fixture(output_dir)

    payload = _read_json(output_dir / "runner_summary.json")
    counts = payload["split_counts"]
    total = counts["n_train"] + counts["n_validation"] + counts["n_test"]
    assert total == counts["n_rows"]
    assert counts["n_test"] >= 1
