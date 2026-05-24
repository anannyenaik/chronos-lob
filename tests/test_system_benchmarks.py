"""Tests for the Phase I systems benchmark runner."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from chronoslob.experiments.artifacts import validate_experiment_directory
from chronoslob.experiments.system_benchmarks import (
    SYSTEM_BENCHMARK_RESULTS_COLUMNS,
    run_system_benchmarks,
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


def test_smoke_system_benchmark_runner_writes_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "system_benchmark_smoke"

    summary = run_system_benchmarks(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        benchmark_set="smoke",
        models=["majority", "logistic"],
        overwrite=True,
    )

    assert summary.benchmark_set == "smoke"
    assert summary.models_requested == ["majority", "logistic"]
    assert "loader_throughput" in summary.benchmarks_run
    assert "feature_generation_speed" in summary.benchmarks_run
    assert "experiment_runner_timing" in summary.benchmarks_run
    assert "memory_profile" in summary.benchmarks_run
    assert any("not benchmark evidence" in warning for warning in summary.warnings)

    for filename in (
        "system_benchmark_summary.json",
        "system_benchmark_results.csv",
        "environment.json",
    ):
        assert (output_dir / filename).is_file(), filename

    expected_reports = {
        "loader_throughput.md",
        "feature_generation_speed.md",
        "experiment_runner_timing.md",
        "inference_latency.md",
        "memory_profile.md",
    }
    actual_reports = {path.name for path in (output_dir / "reports").glob("*.md")}
    assert expected_reports.issubset(actual_reports)

    environment = _read_json(output_dir / "environment.json")
    assert environment["benchmark_set"] == "smoke"
    assert environment["data_source_kind"] == "synthetic_fixture"
    assert environment["data_size_bytes"] > 0
    assert environment["data_row_count"] == 6
    assert environment["models_requested"] == ["majority", "logistic"]
    assert environment["package_version"]
    assert "created_at" in environment

    summary_payload = _read_json(output_dir / "system_benchmark_summary.json")
    assert summary_payload["benchmark_set"] == "smoke"
    assert "not benchmark evidence" in " ".join(summary_payload["warnings"])


def test_system_benchmark_metrics_are_finite_and_child_experiment_valid(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "system_benchmark_smoke"
    summary = run_system_benchmarks(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        benchmark_set="smoke",
        models=["majority", "logistic"],
        overwrite=True,
    )

    results = pd.read_csv(output_dir / "system_benchmark_results.csv")
    assert list(results.columns) == list(SYSTEM_BENCHMARK_RESULTS_COLUMNS)
    assert int(results.isna().sum().sum()) == 0

    run_rows = results[results["status"] == "run"]
    assert not run_rows.empty
    for value in pd.to_numeric(run_rows["metric_value"]).tolist():
        assert math.isfinite(float(value))
        assert float(value) >= 0.0

    loader = run_rows[run_rows["benchmark_name"] == "loader_throughput"]
    assert "rows_per_second" in set(loader["metric_name"])
    feature = run_rows[run_rows["benchmark_name"] == "feature_generation_speed"]
    assert "rows_per_second" in set(feature["metric_name"])
    assert "features_per_second" in set(feature["metric_name"])

    child_relative = summary.child_experiments["paper_runner_timing"]
    child_dir = output_dir / child_relative
    report = validate_experiment_directory(child_dir, include_plots=True)
    assert report.is_valid

    runner_rows = run_rows[run_rows["benchmark_name"] == "experiment_runner_timing"]
    assert "prediction_rows" in set(runner_rows["metric_name"])
    assert float(
        runner_rows.loc[
            runner_rows["metric_name"] == "artefact_count",
            "metric_value",
        ].iloc[0]
    ) > 0.0


def test_skipped_system_benchmark_rows_have_warnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chronoslob.experiments import system_benchmarks

    monkeypatch.setattr(system_benchmarks.tracemalloc, "is_tracing", lambda: True)
    output_dir = tmp_path / "system_benchmark_memory_skip"

    summary = run_system_benchmarks(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        benchmark_set="smoke",
        models=["majority", "logistic"],
        overwrite=True,
    )

    assert "memory_profile" in summary.benchmarks_skipped
    results = pd.read_csv(output_dir / "system_benchmark_results.csv")
    skipped_rows = results[results["status"] == "skipped"]
    assert not skipped_rows.empty
    assert (skipped_rows["warning"].str.len() > 0).all()
    assert "tracemalloc is already active" in " ".join(skipped_rows["warning"])


def test_system_benchmark_rejects_unsupported_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported systems benchmark set"):
        run_system_benchmarks(
            config_path=CONFIG_PATH,
            data_path=TINY_FIXTURE_PATH,
            out_dir=tmp_path / "system_benchmark",
            benchmark_set="unknown",
            models=["majority", "logistic"],
        )


def test_system_benchmark_requires_existing_data_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        run_system_benchmarks(
            config_path=CONFIG_PATH,
            data_path=tmp_path / "missing.csv",
            out_dir=tmp_path / "system_benchmark",
            benchmark_set="smoke",
            models=["majority", "logistic"],
        )


def test_system_benchmark_overwrite_protection(tmp_path: Path) -> None:
    output_dir = tmp_path / "system_benchmark"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty output directory"):
        run_system_benchmarks(
            config_path=CONFIG_PATH,
            data_path=TINY_FIXTURE_PATH,
            out_dir=output_dir,
            benchmark_set="smoke",
            models=["majority", "logistic"],
            overwrite=False,
        )

    assert sentinel.is_file()


def test_system_benchmark_cli_command_works_on_tiny_fixture(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "system_benchmark_cli"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chronoslob.cli",
            "run-system-benchmarks",
            "--config",
            str(CONFIG_PATH),
            "--data-path",
            str(TINY_FIXTURE_PATH),
            "--out",
            str(output_dir),
            "--benchmark-set",
            "smoke",
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
    assert "ChronosLOB systems benchmark" in completed.stdout
    assert "benchmark set:       smoke" in completed.stdout
    assert "benchmarks run:" in completed.stdout
    assert "output directory:" in completed.stdout
    assert (output_dir / "system_benchmark_summary.json").is_file()
    assert (output_dir / "system_benchmark_results.csv").is_file()
    assert (output_dir / "environment.json").is_file()


def test_system_benchmark_fixture_wording_is_clear(tmp_path: Path) -> None:
    output_dir = tmp_path / "system_benchmark_wording"
    run_system_benchmarks(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        benchmark_set="smoke",
        models=["majority", "logistic"],
        overwrite=True,
    )

    report_text = (output_dir / "reports" / "loader_throughput.md").read_text(
        encoding="utf-8",
    )
    summary_text = (output_dir / "system_benchmark_summary.json").read_text(
        encoding="utf-8",
    )
    assert "smoke measurements" in report_text
    assert "not benchmark evidence" in report_text
    assert "not benchmark evidence" in summary_text


def test_system_benchmark_outputs_stay_under_requested_directory(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "requested_output"
    run_system_benchmarks(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        benchmark_set="smoke",
        models=["majority", "logistic"],
        overwrite=True,
    )

    assert {path.name for path in tmp_path.iterdir()} == {"requested_output"}
    assert (output_dir / "child_experiments" / "paper_runner_timing").is_dir()
