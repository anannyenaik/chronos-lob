"""Tests for serious FI-2010 neural benchmark planning."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from chronoslob.cli import _inspect_fi2010_neural_plan_impl
from chronoslob.experiments.neural_benchmarking import (
    NEURAL_BENCHMARK_ARTEFACTS,
    build_training_metadata,
    count_parameters,
    expected_lightweight_artefacts,
    generate_neural_run_plan,
    load_neural_benchmark_config,
    resolve_neural_device,
    training_metadata_schema_fields,
)
from chronoslob.utils.audit import (
    AuditStatus,
    check_no_forbidden_claims,
    check_public_release_wording,
)
from chronoslob.utils.paths import project_root

CONFIG_PATH = (
    project_root() / "configs" / "experiments" / "fi2010_neural_serious.yaml"
)


def test_neural_serious_yaml_parses() -> None:
    config = load_neural_benchmark_config(CONFIG_PATH)

    assert config.study_name == "fi2010_neural_benchmark_quality"
    assert config.dataset.name == "FI-2010"
    assert config.official_split.split_column == "split"
    assert config.folds == ("fold_1", "fold_2", "fold_3", "fold_4", "fold_5")
    assert config.seeds == (0, 1, 2)
    assert config.target.horizon == 10
    assert config.enabled_model_names == ("deeplob_style", "matrix_transformer")
    assert config.lookbacks == (20, 50, 100)
    assert config.device_selection == "auto"
    assert config.validation_metric == "macro_f1"
    assert config.checkpoint_policy.enabled is False
    assert config.artefacts.write_full_predictions_by_default is False


def test_neural_serious_yaml_has_required_top_level_keys() -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)

    required = {
        "study_name",
        "dataset",
        "official_split",
        "folds",
        "seeds",
        "target",
        "neural_models",
        "lookbacks",
        "training",
        "device_selection",
        "deterministic_seed_handling",
        "validation_metric",
        "checkpoint_policy",
        "artefacts",
        "mode",
        "benchmark_note",
    }
    assert required.issubset(payload)
    assert set(NEURAL_BENCHMARK_ARTEFACTS) == {
        "summary.json",
        "run_plan.csv",
        "results_by_fold_seed.csv",
        "results_summary.csv",
        "training_summary.csv",
        "model_capacity_summary.csv",
        "model_failures.json",
    }


def test_run_plan_generation_is_deterministic_grid() -> None:
    config = load_neural_benchmark_config(CONFIG_PATH)

    first = generate_neural_run_plan(config)
    second = generate_neural_run_plan(config)

    assert first == second
    assert len(first) == 5 * 3 * 2 * 3
    assert first[0].run_id == "fold_1__seed_0__deeplob_style__lookback_20"
    assert first[-1].run_id == "fold_5__seed_2__matrix_transformer__lookback_100"
    assert {item.checkpoint_path for item in first} == {None}


def test_run_plan_generation_accepts_subsets() -> None:
    config = load_neural_benchmark_config(CONFIG_PATH)

    plan = generate_neural_run_plan(
        config,
        folds="1,3",
        models="matrix_transformer",
        lookbacks=[50],
    )

    assert [item.fold_id for item in plan] == [
        "fold_1",
        "fold_1",
        "fold_1",
        "fold_3",
        "fold_3",
        "fold_3",
    ]
    assert {item.model_name for item in plan} == {"matrix_transformer"}
    assert {item.lookback for item in plan} == {50}


def test_unsupported_neural_model_fails_clearly() -> None:
    config = load_neural_benchmark_config(CONFIG_PATH)

    with pytest.raises(ValueError, match="unsupported neural benchmark model"):
        generate_neural_run_plan(config, models=["ssl_transformer"])


def test_device_resolver_handles_cpu_and_auto_safely() -> None:
    cpu = resolve_neural_device("cpu")
    auto = resolve_neural_device("auto")

    assert cpu.resolved == "cpu"
    assert cpu.requested == "cpu"
    assert auto.resolved in {"cpu", "cuda"}
    assert auto.requested == "auto"


def test_parameter_count_utility_works_on_tiny_model() -> None:
    torch = pytest.importorskip("torch")

    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 2),
    )

    assert count_parameters(model) == 26


def test_inspection_cli_runs_without_training(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = _inspect_fi2010_neural_plan_impl(
        config_path=CONFIG_PATH,
        folds=None,
        models=["deeplob_style", "matrix_transformer"],
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "ChronosLOB FI-2010 neural benchmark plan" in captured.out
    assert "planned runs:           90" in captured.out
    assert "training:               not run" in captured.out
    assert "outputs:                not written" in captured.out


def test_smoke_and_benchmark_mode_distinction_is_explicit() -> None:
    config = load_neural_benchmark_config(CONFIG_PATH)
    plan = generate_neural_run_plan(config, folds=[1], models=["deeplob_style"])

    assert config.is_benchmark_mode is True
    assert config.is_smoke_mode is False
    assert "not smoke" in config.benchmark_note.lower()
    assert {item.mode for item in plan} == {"benchmark"}


def test_no_predictions_or_checkpoints_are_default_outputs() -> None:
    config = load_neural_benchmark_config(CONFIG_PATH)
    plan = generate_neural_run_plan(config, folds=[1], models=["deeplob_style"])
    artefacts = expected_lightweight_artefacts(config)

    assert config.artefacts.write_full_predictions_by_default is False
    assert config.artefacts.write_checkpoints_by_default is False
    assert all(item.checkpoint_path is None for item in plan)
    assert all("prediction" not in path for path in artefacts.values())
    assert all("checkpoint" not in path for path in artefacts.values())


def test_training_metadata_schema_matches_expected_fields() -> None:
    config = load_neural_benchmark_config(CONFIG_PATH)
    plan = generate_neural_run_plan(config, folds=[1], models=["deeplob_style"])[0]

    metadata = build_training_metadata(
        plan=plan,
        device="cpu",
        parameter_count=10,
        best_epoch=1,
        early_stopped=False,
        training_seconds=0.5,
        validation_metric_value=0.25,
        test_metrics={"macro_f1": 0.2},
        status="ok",
    )

    expected = {
        "fold_id",
        "seed",
        "model_name",
        "lookback",
        "device",
        "parameter_count",
        "max_epochs",
        "best_epoch",
        "early_stopped",
        "training_seconds",
        "validation_metric",
        "validation_metric_value",
        "test_metrics",
        "status",
    }
    assert set(training_metadata_schema_fields()) == expected
    assert metadata.model_name == "deeplob_style"
    assert metadata.test_metrics == {"macro_f1": 0.2}


def test_neural_benchmark_doc_avoids_forbidden_public_claims() -> None:
    scan_path = Path("docs") / "NEURAL_BENCHMARK_PROTOCOL.md"

    claims = check_no_forbidden_claims(project_root(), scan_paths=(scan_path,))
    wording = check_public_release_wording(project_root(), scan_paths=(scan_path,))

    assert claims.status == AuditStatus.PASS
    assert wording.status == AuditStatus.PASS
