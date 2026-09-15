"""Tests for the DeepLOB-style supervised neural smoke experiment."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from chronoslob.models.deeplob import DeepLOBConfig  # noqa: E402
from chronoslob.training.torch_experiment import (  # noqa: E402
    DeepLOBExperimentConfig,
    run_deeplob_experiment,
    run_deeplob_smoke_from_fi2010_fixture,
)
from chronoslob.training.torch_training import TorchTrainingConfig  # noqa: E402

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "fi2010" / "tiny_fi2010_like.csv"
)


def _timestamps(n_rows: int) -> list[datetime]:
    start = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
    return [start + timedelta(seconds=index) for index in range(n_rows)]


def _feature_frame(n_rows: int = 24, n_features: int = 4) -> pd.DataFrame:
    data: dict[str, list[float] | list[str] | list[datetime]] = {
        "timestamp": _timestamps(n_rows),
        "symbol": ["TEST"] * n_rows,
    }
    for feature_index in range(n_features):
        data[f"f{feature_index}"] = [
            float(row + feature_index) for row in range(n_rows)
        ]
    return pd.DataFrame(data)


def _label_frame(n_rows: int = 24) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": _timestamps(n_rows),
            "symbol": ["TEST"] * n_rows,
            "direction_1": [row % 2 for row in range(n_rows)],
        }
    )


def _config(**overrides: object) -> DeepLOBExperimentConfig:
    defaults: dict[str, object] = {
        "run_name": "deeplob-test",
        "seed": 42,
        "target_column": "direction_1",
        "lookback": 2,
        "batch_size": 4,
        "train_fraction": 0.5,
        "validation_fraction": 0.25,
        "test_fraction": 0.25,
        "training": TorchTrainingConfig(
            epochs=1,
            learning_rate=1e-2,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            device="cpu",
            seed=42,
        ),
    }
    defaults.update(overrides)
    return DeepLOBExperimentConfig(**defaults)  # type: ignore[arg-type]


def test_experiment_config_validates_run_name() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        DeepLOBExperimentConfig(
            run_name="  ",
            target_column="direction_1",
        )


def test_experiment_config_validates_target_column() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        DeepLOBExperimentConfig(
            run_name="deeplob-test",
            target_column="",
        )


def test_experiment_config_validates_lookback() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        DeepLOBExperimentConfig(
            run_name="deeplob-test",
            target_column="direction_1",
            lookback=0,
        )


def test_experiment_config_validates_fraction_sum() -> None:
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        DeepLOBExperimentConfig(
            run_name="deeplob-test",
            target_column="direction_1",
            train_fraction=0.5,
            validation_fraction=0.3,
            test_fraction=0.3,
        )


def test_run_deeplob_experiment_returns_in_memory_result() -> None:
    result = run_deeplob_experiment(_feature_frame(), _label_frame(), _config())

    assert result["run_name"] == "deeplob-test"
    assert result["target_column"] == "direction_1"
    assert result["feature_columns"] == ["f0", "f1", "f2", "f3"]
    assert result["feature_count"] == 4
    assert result["n_classes"] == 2
    assert result["model_parameter_count"] > 0
    assert isinstance(result["training_history"], list)
    assert len(result["training_history"]) == 1
    assert "train_loss" in result["training_history"][0]
    assert result["final_validation_metrics"] is not None
    assert result["notes"] == (
        "Synthetic fixture smoke test only; not benchmark performance."
    )


def test_no_files_written_when_write_outputs_false(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"

    run_deeplob_experiment(
        _feature_frame(),
        _label_frame(),
        _config(),
        output_root=output_root,
    )

    assert not output_root.exists()


def test_files_written_under_tmp_path_when_write_outputs_true(tmp_path: Path) -> None:
    result = run_deeplob_experiment(
        _feature_frame(),
        _label_frame(),
        _config(write_outputs=True),
        output_root=tmp_path,
    )

    run_path = Path(result["output_path"])
    assert run_path.is_dir()
    assert (run_path / "metadata.json").is_file()
    assert (run_path / "metrics.json").is_file()
    assert (run_path / "configs" / "deeplob_config.json").is_file()
    payload = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
    assert payload["run_name"] == "deeplob-test"


def test_no_checkpoint_files_are_written(tmp_path: Path) -> None:
    result = run_deeplob_experiment(
        _feature_frame(),
        _label_frame(),
        _config(write_outputs=True),
        output_root=tmp_path,
    )

    run_path = Path(result["output_path"])
    checkpoint_files = [
        candidate
        for candidate in run_path.rglob("*")
        if candidate.is_file()
        and candidate.suffix.lower() in {".pt", ".pth", ".ckpt", ".bin"}
    ]
    assert checkpoint_files == []


def test_train_only_standardisation_uses_training_rows_only() -> None:
    feature_frame = _feature_frame(n_rows=12, n_features=2)
    label_frame = _label_frame(n_rows=12)
    config = _config(
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
        lookback=2,
        batch_size=2,
    )

    result = run_deeplob_experiment(feature_frame, label_frame, config)

    standardisation = result["standardisation"]
    assert standardisation["standardise"] is True
    train_values_f0 = feature_frame["f0"].iloc[:6].to_numpy(dtype=float)
    train_values_f1 = feature_frame["f1"].iloc[:6].to_numpy(dtype=float)
    expected_mean_f0 = float(train_values_f0.mean())
    expected_mean_f1 = float(train_values_f1.mean())
    assert standardisation["mean"][0] == pytest.approx(expected_mean_f0)
    assert standardisation["mean"][1] == pytest.approx(expected_mean_f1)
    # The full-frame mean would be 5.5, verify the runner did not use it.
    full_mean = float(feature_frame["f0"].to_numpy(dtype=float).mean())
    assert standardisation["mean"][0] != pytest.approx(full_mean)


def test_result_contains_parameter_count_and_history() -> None:
    result = run_deeplob_experiment(_feature_frame(), _label_frame(), _config())

    assert "model_parameter_count" in result
    assert result["model_parameter_count"] > 0
    assert isinstance(result["training_history"], list)


def test_explicit_model_config_must_match_data_dimensions() -> None:
    feature_frame = _feature_frame()
    label_frame = _label_frame()

    bad_config = _config(
        model=DeepLOBConfig(input_features=99, n_classes=2),
    )
    with pytest.raises(ValueError, match="input_features"):
        run_deeplob_experiment(feature_frame, label_frame, bad_config)


def test_label_like_feature_columns_are_rejected() -> None:
    feature_frame = _feature_frame()
    feature_frame["label_10"] = [0] * len(feature_frame)

    with pytest.raises(ValueError, match="leakage check failed"):
        run_deeplob_experiment(feature_frame, _label_frame(), _config())


def test_run_deeplob_smoke_from_fixture_returns_result() -> None:
    result = run_deeplob_smoke_from_fi2010_fixture(
        FIXTURE_PATH,
        lookback=2,
        seed=42,
        epochs=1,
        batch_size=2,
    )

    assert result["run_name"] == "synthetic-fi2010-deeplob-smoke"
    assert result["lookback"] == 2
    assert result["sample_counts"]["train"] > 0
    assert result["model_parameter_count"] > 0
    assert (
        result["notes"]
        == "Synthetic fixture smoke test only; not benchmark performance."
    )


def test_no_checkpoint_files_committed_to_git() -> None:
    completed = subprocess.run(
        ["git", "ls-files", "runs"],
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )
    if completed.returncode != 0:
        pytest.skip("git is unavailable")

    assert completed.stdout.strip() == ""
