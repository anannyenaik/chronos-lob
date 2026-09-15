"""Tests for the classical paper-runner model registry and preprocessing."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronoslob.experiments.model_registry import (
    DEFAULT_PAPER_MODELS,
    REQUIRED_PAPER_MODELS,
    SUPPORTED_PAPER_MODELS,
    build_paper_baseline_config,
    get_paper_model_spec,
    list_supported_paper_models,
    normalise_paper_model_names,
)
from chronoslob.experiments.paper_runner import run_paper_experiment
from chronoslob.models.baselines import (
    SUPPORTED_BASELINE_MODEL_TYPES,
    create_baseline_model,
)
from chronoslob.models.preprocessing import TrainOnlyStandardScaler
from chronoslob.utils.paths import project_root

CONFIG_PATH = project_root() / "configs" / "experiments" / "fi2010_midprice_h10.yaml"
TINY_FIXTURE_PATH = (
    project_root() / "tests" / "fixtures" / "fi2010" / "tiny_fi2010_like.csv"
)

_PHASE_D_MODEL_NAMES: tuple[str, ...] = (
    "majority",
    "logistic",
    "ridge",
    "elastic_net",
    "random_forest",
    "gradient_boosting",
)


def test_registry_lists_all_phase_d_model_names() -> None:
    for name in _PHASE_D_MODEL_NAMES:
        assert name in SUPPORTED_PAPER_MODELS, name
    assert list_supported_paper_models() == SUPPORTED_PAPER_MODELS


def test_registry_specs_reference_supported_baseline_model_types() -> None:
    for name in SUPPORTED_PAPER_MODELS:
        spec = get_paper_model_spec(name)
        if spec.model_family != "classical":
            continue
        assert spec.model_type in SUPPORTED_BASELINE_MODEL_TYPES, spec.model_type


def test_registry_default_and_required_lists_only_include_majority() -> None:
    assert DEFAULT_PAPER_MODELS == ("majority",)
    assert REQUIRED_PAPER_MODELS == ("majority",)


def test_get_paper_model_spec_is_case_insensitive() -> None:
    spec_lower = get_paper_model_spec("random_forest")
    spec_mixed = get_paper_model_spec("Random_Forest")
    assert spec_lower.name == spec_mixed.name == "random_forest"


def test_get_paper_model_spec_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unsupported model"):
        get_paper_model_spec("deeplob")


def test_normalise_paper_model_names_returns_default_when_none() -> None:
    assert normalise_paper_model_names(None) == DEFAULT_PAPER_MODELS


def test_normalise_paper_model_names_dedupes_and_orders() -> None:
    result = normalise_paper_model_names(
        ["majority", "MAJORITY", "logistic", "logistic"]
    )
    assert result == ("majority", "logistic")


def test_normalise_paper_model_names_requires_majority() -> None:
    with pytest.raises(ValueError, match="majority"):
        normalise_paper_model_names(["logistic", "random_forest"])


def test_normalise_paper_model_names_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unsupported model"):
        normalise_paper_model_names(["majority", "ssl_transformer"])


def test_normalise_paper_model_names_rejects_empty_after_strip() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        normalise_paper_model_names(["majority", " "])


def test_normalise_paper_model_names_rejects_bare_string_argument() -> None:
    with pytest.raises(TypeError, match="sequence"):
        normalise_paper_model_names("majority,logistic")  # type: ignore[arg-type]


def test_build_paper_baseline_config_uses_registry_short_name() -> None:
    for name in _PHASE_D_MODEL_NAMES:
        spec = get_paper_model_spec(name)
        config = build_paper_baseline_config(name, seed=0)
        assert config.name == spec.name
        assert config.model_type == spec.model_type
        assert config.random_state == 0


@pytest.mark.parametrize("name", list(_PHASE_D_MODEL_NAMES))
def test_registry_can_build_each_classical_model(name: str) -> None:
    config = build_paper_baseline_config(name, seed=1)
    model = create_baseline_model(config)
    assert model.model_name == name


def test_train_only_standardiser_does_not_fit_on_test_features() -> None:
    rng = np.random.default_rng(0)
    train_features = rng.normal(loc=0.0, scale=1.0, size=(10, 3))
    test_features = rng.normal(loc=100.0, scale=50.0, size=(5, 3))

    scaler = TrainOnlyStandardScaler()
    scaler.fit_transform(train_features)
    train_mean = scaler.mean_
    train_scale = scaler.scale_

    transformed_test = scaler.transform(test_features)
    assert transformed_test.shape == (5, 3)
    # The fitted statistics must come from the training matrix only;
    # transforming the test matrix must not update them.
    assert np.allclose(scaler.mean_, train_mean)
    assert np.allclose(scaler.scale_, train_scale)
    # Sanity: the train mean is far from the synthetic test centre, so
    # the train-only scaler would produce non-trivial test offsets.
    assert not np.allclose(train_mean, test_features.mean(axis=0), atol=1.0)


def test_cli_supports_comma_separated_model_list(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_experiment_classical_smoke"

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
            "majority,logistic",
            "--overwrite",
        ],
        cwd=project_root(),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "artefact validation: valid" in completed.stdout
    assert "models run:          majority, logistic" in completed.stdout


def test_multi_model_run_writes_predictions_for_each_model(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_experiment_classical"

    summary = run_paper_experiment(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        models=["majority", "logistic"],
        overwrite=True,
    )

    assert summary.models_run == ["majority", "logistic"]
    predictions = pd.read_csv(output_dir / "predictions.csv")
    counts = predictions["model_name"].value_counts().to_dict()
    assert counts.get("majority") == counts.get("logistic")
    assert counts.get("majority", 0) >= 1


def test_results_json_contains_one_record_per_successful_model(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "paper_experiment_classical"

    summary = run_paper_experiment(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        models=["majority", "logistic"],
        overwrite=True,
    )

    from chronoslob.experiments.artifacts import load_results

    results = load_results(output_dir / "results.json")
    assert {result.model_name for result in results.model_results} == {
        "majority",
        "logistic",
    }
    for result in results.model_results:
        # Each metric must be finite per the experiment results contract.
        for value in result.metrics.values():
            assert math.isfinite(value)
    assert summary.predictive_metric_names
    assert summary.calibration_metric_names


def test_probability_rows_sum_to_one_for_models_that_emit_them(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "paper_experiment_classical"

    run_paper_experiment(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        models=["majority", "logistic"],
        overwrite=True,
    )

    frame = pd.read_csv(output_dir / "predictions.csv")
    probability_columns = [
        column for column in frame.columns if column.startswith("probability_")
    ]
    assert probability_columns

    rows_with_probs = frame.dropna(subset=probability_columns)
    assert not rows_with_probs.empty
    for _, row in rows_with_probs.iterrows():
        total = sum(float(row[column]) for column in probability_columns)
        assert math.isclose(total, 1.0, abs_tol=1e-6)
        for column in probability_columns:
            assert math.isfinite(float(row[column]))
        assert math.isfinite(float(row["confidence"]))


def test_classical_smoke_output_validates_under_artefact_contract(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "paper_experiment_classical"

    run_paper_experiment(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        models=["majority", "logistic"],
        overwrite=True,
    )

    from chronoslob.experiments.artifacts import validate_experiment_directory

    report = validate_experiment_directory(output_dir, include_plots=True)
    assert report.is_valid
    # No plot files are written in this phase; the directory must remain
    # valid because plot artefacts are optional.
    plots_dir = output_dir / "plots"
    if plots_dir.exists():
        assert not any(plots_dir.iterdir())


def test_classical_smoke_model_card_says_not_benchmark_evidence(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "paper_experiment_classical"

    run_paper_experiment(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        models=["majority", "logistic"],
        overwrite=True,
    )

    text = (output_dir / "model_card.md").read_text(encoding="utf-8")
    lowered = text.lower()
    assert "not benchmark evidence" in lowered
    assert "synthetic fixture smoke" in lowered
