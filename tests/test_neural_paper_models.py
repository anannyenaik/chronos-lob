"""Tests for neural models in the paper experiment runner."""

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
    load_results,
    validate_experiment_directory,
)
from chronoslob.experiments.model_registry import (
    SUPPORTED_PAPER_MODELS,
    get_paper_model_spec,
    normalise_paper_model_names,
)
from chronoslob.experiments.paper_runner import run_paper_experiment
from chronoslob.utils.paths import project_root

CONFIG_PATH = project_root() / "configs" / "experiments" / "fi2010_midprice_h10.yaml"
TINY_FIXTURE_PATH = (
    project_root() / "tests" / "fixtures" / "fi2010" / "tiny_fi2010_like.csv"
)
NORMALISED_SPLIT_FIXTURE_PATH = (
    project_root()
    / "tests"
    / "fixtures"
    / "fi2010"
    / "tiny_fi2010_normalised_split.csv"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _run(
    output_dir: Path,
    *,
    model_name: str,
) -> Any:
    return run_paper_experiment(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        models=["majority", model_name],
        overwrite=True,
    )


def _probability_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("probability_")]


def _assert_probability_rows_are_valid(frame: pd.DataFrame) -> None:
    probability_columns = _probability_columns(frame)
    assert probability_columns
    rows_with_probabilities = frame.dropna(subset=probability_columns)
    assert not rows_with_probabilities.empty
    for _, row in rows_with_probabilities.iterrows():
        total = 0.0
        for column in probability_columns:
            value = float(row[column])
            assert math.isfinite(value)
            total += value
        assert math.isclose(total, 1.0, abs_tol=1e-6)
        assert math.isfinite(float(row["confidence"]))


def _assert_successful_models_match_artefacts(
    output_dir: Path,
    models_run: list[str],
) -> None:
    results = load_results(output_dir / "results.json")
    assert [result.model_name for result in results.model_results] == models_run

    confusion = _read_json(output_dir / "confusion_matrix.json")
    assert [entry["model_name"] for entry in confusion["models"]] == models_run


def _assert_neural_model_ran_or_skipped_clearly(
    output_dir: Path,
    *,
    model_name: str,
    models_run: list[str],
    skipped: list[Any],
) -> None:
    frame = pd.read_csv(output_dir / "predictions.csv")
    if model_name in models_run:
        assert model_name in set(frame["model_name"].unique())
        model_rows = frame[frame["model_name"] == model_name]
        assert not model_rows.empty
        _assert_probability_rows_are_valid(model_rows)
    else:
        skip_payload = {skip.model_name: skip.reason for skip in skipped}
        assert model_name in skip_payload
        assert skip_payload[model_name].strip()


def test_registry_recognises_supported_neural_models() -> None:
    for name in ("deeplob_style", "transformer", "matrix_transformer"):
        assert name in SUPPORTED_PAPER_MODELS
        spec = get_paper_model_spec(name)
        assert spec.name == name
        assert spec.model_family == "neural"
        assert spec.emits_probabilities is True


def test_unsupported_neural_name_fails_clearly() -> None:
    with pytest.raises(ValueError, match="unsupported model"):
        get_paper_model_spec("ssl_transformer")
    with pytest.raises(ValueError, match="unsupported model"):
        normalise_paper_model_names(["majority", "ssl_transformer"])


@pytest.mark.parametrize("model_name", ["deeplob_style", "transformer"])
def test_neural_runner_request_on_tiny_fixture_runs_or_skips_clearly(
    tmp_path: Path,
    model_name: str,
) -> None:
    output_dir = tmp_path / f"paper_experiment_{model_name}"

    summary = _run(output_dir, model_name=model_name)

    assert summary.models_run[0] == "majority"
    _assert_neural_model_ran_or_skipped_clearly(
        output_dir,
        model_name=model_name,
        models_run=summary.models_run,
        skipped=list(summary.skipped_models),
    )
    _assert_successful_models_match_artefacts(output_dir, summary.models_run)
    report = validate_experiment_directory(output_dir, include_plots=True)
    assert report.is_valid


def test_combined_neural_smoke_outputs_model_card_and_metadata(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "paper_experiment_neural"

    summary = run_paper_experiment(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        models=["majority", "deeplob_style", "transformer"],
        overwrite=True,
    )

    _assert_successful_models_match_artefacts(output_dir, summary.models_run)
    predictions = pd.read_csv(output_dir / "predictions.csv")
    _assert_probability_rows_are_valid(predictions)
    results = load_results(output_dir / "results.json")
    for model_result in results.model_results:
        assert "brier_score" in model_result.metrics

    runner_summary = _read_json(output_dir / "runner_summary.json")
    assert "neural_settings" in runner_summary
    assert runner_summary["neural_settings"]["device"] == "cpu"
    assert runner_summary["neural_settings"]["max_epochs"] == 1
    assert runner_summary["neural_settings"]["supported_models"] == [
        "deeplob_style",
        "transformer",
        "matrix_transformer",
    ]
    assert "ssl_transformer" not in SUPPORTED_PAPER_MODELS

    model_metadata = runner_summary["model_metadata"]
    for model_name in ("deeplob_style", "transformer"):
        if model_name not in summary.models_run:
            continue
        assert model_name in model_metadata
        assert model_metadata[model_name]["sample_counts"]["test"] >= 1
        assert model_metadata[model_name]["class_mapping"]
        assert (
            model_metadata[model_name]["window_policy"]["windows_stay_inside_split"]
            is True
        )
    if "deeplob_style" in model_metadata:
        assert (
            model_metadata["deeplob_style"]["standardisation"]["fit_split"]
            == "train"
        )
    if "transformer" in model_metadata:
        assert model_metadata["transformer"]["standardisation"]["fit_split"] == "train"
        assert (
            model_metadata["transformer"]["matrix_path"]
            == "normalised FI-2010 matrix path"
        )
        assert model_metadata["transformer"]["raw_snapshot_construction"] is False

    model_card = (output_dir / "model_card.md").read_text(encoding="utf-8")
    lowered = model_card.lower()
    assert "## Neural Settings" in model_card
    assert "successfully run" in lowered
    assert "skipped" in lowered
    assert "deeplob-style" in lowered
    assert "not benchmark evidence" in lowered
    assert "synthetic fixture smoke" in lowered


def test_neural_cli_smoke_validates_under_inspector(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_experiment_neural_cli"

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
            "majority,deeplob_style,transformer",
            "--overwrite",
        ],
        cwd=project_root(),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "artefact validation: valid" in completed.stdout

    inspected = subprocess.run(
        [
            sys.executable,
            "-m",
            "chronoslob.cli",
            "inspect-experiment-artifacts",
            "--experiment",
            str(output_dir),
        ],
        cwd=project_root(),
        capture_output=True,
        check=False,
        text=True,
    )

    assert inspected.returncode == 0, inspected.stderr
    assert "valid:            yes" in inspected.stdout


def test_matrix_transformer_runs_on_normalised_split_fixture(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "paper_experiment_matrix_transformer"

    summary = run_paper_experiment(
        config_path=CONFIG_PATH,
        data_path=NORMALISED_SPLIT_FIXTURE_PATH,
        out_dir=output_dir,
        models=["majority", "matrix_transformer"],
        overwrite=True,
    )

    assert "matrix_transformer" in summary.models_run
    predictions = pd.read_csv(output_dir / "predictions.csv")
    model_rows = predictions[predictions["model_name"] == "matrix_transformer"]
    assert not model_rows.empty
    _assert_probability_rows_are_valid(model_rows)
    runner_summary = _read_json(output_dir / "runner_summary.json")
    metadata = runner_summary["model_metadata"]["matrix_transformer"]
    assert metadata["raw_snapshot_construction"] is False
    assert metadata["window_policy"]["windows_stay_inside_split"] is True
    assert metadata["standardisation"]["fit_split"] == "train"


def test_transformer_matrix_path_does_not_construct_raw_order_book_levels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronoslob.data import fi2010

    def _raise_if_constructed(*args: object, **kwargs: object) -> object:
        raise AssertionError("raw OrderBookLevel construction was called")

    monkeypatch.setattr(fi2010, "OrderBookLevel", _raise_if_constructed)
    output_dir = tmp_path / "paper_experiment_transformer_matrix"

    summary = run_paper_experiment(
        config_path=CONFIG_PATH,
        data_path=NORMALISED_SPLIT_FIXTURE_PATH,
        out_dir=output_dir,
        models=["majority", "transformer"],
        overwrite=True,
    )

    assert "transformer" in summary.models_run
    runner_summary = _read_json(output_dir / "runner_summary.json")
    assert (
        runner_summary["model_metadata"]["transformer"]["matrix_path"]
        == "normalised FI-2010 matrix path"
    )


def test_best_epoch_returns_last_is_best_marker() -> None:
    from chronoslob.experiments.neural_adapters import _best_epoch

    class _Item:
        def __init__(self, epoch: int, is_best: bool) -> None:
            self.epoch = epoch
            self.is_best = is_best

    history = [
        _Item(1, True),
        _Item(2, False),
        _Item(3, True),
        _Item(4, False),
        _Item(5, True),
        _Item(6, False),
    ]

    assert _best_epoch(history) == 5
    assert _best_epoch([]) is None
