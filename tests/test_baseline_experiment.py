"""Tests for the classical baseline experiment runner."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from chronoslob.data.fi2010 import FI2010Config, load_fi2010
from chronoslob.features.pipeline import (
    FeaturePipelineConfig,
    build_feature_frame_from_fi2010,
)
from chronoslob.labels.pipeline import build_label_frame_from_fi2010
from chronoslob.models.baselines import BaselineModelConfig
from chronoslob.training.baseline_experiment import (
    BaselineExperimentConfig,
    BaselineSplitConfig,
    create_default_baseline_configs,
    run_baseline_experiment,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "fi2010" / "tiny_fi2010_like.csv"
)


def _timestamps(n_rows: int) -> list[datetime]:
    start = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
    return [start + timedelta(seconds=index) for index in range(n_rows)]


def _feature_frame(n_rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": _timestamps(n_rows),
            "symbol": ["TEST"] * n_rows,
            "f1": [float(index) for index in range(n_rows)],
            "f2": [float(index % 3) for index in range(n_rows)],
        }
    )


def _label_frame(n_rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": _timestamps(n_rows),
            "symbol": ["TEST"] * n_rows,
            "direction_1": [index % 2 for index in range(n_rows)],
        }
    )


def _config() -> BaselineExperimentConfig:
    return BaselineExperimentConfig(
        run_name="baseline-test",
        seed=42,
        target_column="direction_1",
        split=BaselineSplitConfig(
            train_fraction=0.5,
            validation_fraction=0.25,
            test_fraction=0.25,
        ),
        models=[
            BaselineModelConfig(
                name="majority",
                model_type="majority_class",
            )
        ],
    )


def test_default_baseline_configs_created() -> None:
    configs = create_default_baseline_configs(seed=7)

    assert [config.model_type for config in configs] == [
        "majority_class",
        "logistic_regression",
        "ridge_classifier",
        "random_forest",
    ]
    assert all(config.random_state == 7 for config in configs)


def test_run_baseline_experiment_returns_in_memory_results() -> None:
    result = run_baseline_experiment(_feature_frame(), _label_frame(), _config())

    assert result["run_name"] == "baseline-test"
    assert result["target_column"] == "direction_1"
    assert result["feature_columns"] == ["f1", "f2"]
    assert result["models"][0]["name"] == "majority"


def test_no_files_written_when_write_outputs_false(tmp_path: Path) -> None:
    output_root = tmp_path / "runs"

    run_baseline_experiment(
        _feature_frame(),
        _label_frame(),
        _config(),
        output_root=output_root,
        write_outputs=False,
    )

    assert not output_root.exists()


def test_files_written_under_tmp_path_when_write_outputs_true(tmp_path: Path) -> None:
    result = run_baseline_experiment(
        _feature_frame(),
        _label_frame(),
        _config(),
        output_root=tmp_path,
        write_outputs=True,
    )

    run_path = Path(result["output_path"])
    assert run_path.is_dir()
    assert (run_path / "metadata.json").is_file()
    assert (run_path / "metrics.json").is_file()
    assert (run_path / "configs" / "baseline_config.json").is_file()
    metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["run_name"] == "baseline-test"


def test_train_only_scaler_uses_training_rows_only() -> None:
    result = run_baseline_experiment(_feature_frame(), _label_frame(), _config())

    assert result["preprocessing"]["standardise"] is True
    assert result["preprocessing"]["scaler_mean"] == [2.5, 1.0]


def test_label_like_feature_columns_are_rejected() -> None:
    feature_frame = _feature_frame()
    feature_frame["label_10"] = [0] * len(feature_frame)

    with pytest.raises(ValueError, match="leakage check failed"):
        run_baseline_experiment(feature_frame, _label_frame(), _config())


def test_temporal_split_sizes_are_recorded() -> None:
    result = run_baseline_experiment(_feature_frame(), _label_frame(), _config())

    assert result["split_sizes"] == {"train": 6, "validation": 3, "test": 3}


def test_model_results_include_validation_metrics() -> None:
    result = run_baseline_experiment(_feature_frame(), _label_frame(), _config())

    validation = result["models"][0]["validation"]
    assert "metrics" in validation
    assert "confusion_matrix" in validation
    assert validation["metrics"]["n_samples"] == 3


def test_synthetic_fi2010_fixture_smoke_path_works() -> None:
    dataset = load_fi2010(
        FI2010Config(
            path=FIXTURE_PATH,
            timestamp_column="timestamp",
            split_column="split",
            label_columns=["label_10", "label_50", "label_100"],
            price_level_count=2,
        )
    )
    feature_frame = build_feature_frame_from_fi2010(
        dataset,
        FeaturePipelineConfig(
            include_order_flow=False,
            include_volatility=False,
        ),
    )
    labels = build_label_frame_from_fi2010(dataset, prefer_existing_labels=True)
    label_frame = labels.loc[:, ["timestamp", "symbol", "label_10"]]
    config = BaselineExperimentConfig(
        run_name="fixture-smoke",
        target_column="label_10",
        models=[
            BaselineModelConfig(
                name="majority",
                model_type="majority_class",
            )
        ],
    )

    result = run_baseline_experiment(feature_frame, label_frame, config)

    assert result["split_sizes"]["train"] > 0
    assert result["models"][0]["validation"]["metrics"]["n_samples"] > 0


def test_no_fake_run_results_are_tracked() -> None:
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
