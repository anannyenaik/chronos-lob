"""Tests for the Phase H paper ablation runner."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml  # type: ignore[import-untyped]

from chronoslob.experiments.ablations import (
    ABLATION_RESULTS_COLUMNS,
    run_paper_ablations,
)
from chronoslob.experiments.artifacts import validate_experiment_directory
from chronoslob.experiments.paper_runner import run_paper_experiment
from chronoslob.utils.paths import project_root

CONFIG_PATH = project_root() / "configs" / "experiments" / "fi2010_midprice_h10.yaml"
TINY_FIXTURE_PATH = (
    project_root() / "tests" / "fixtures" / "fi2010" / "tiny_fi2010_like.csv"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_config_with_feature_patterns(
    tmp_path: Path,
    patterns: list[str],
) -> Path:
    payload = _read_yaml(CONFIG_PATH)
    payload["feature_patterns"] = patterns
    config_path = tmp_path / "config_with_feature_patterns.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return config_path


def test_smoke_ablation_runner_writes_traceable_outputs(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "paper_ablation_smoke"

    summary = run_paper_ablations(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        models=["majority", "logistic"],
        ablation_set="smoke",
        overwrite=True,
    )

    assert summary.ablation_set == "smoke"
    assert summary.is_fixture is True
    assert summary.data_source_kind == "synthetic_fixture"
    assert summary.ablations_run == [
        "baseline",
        "calibration_bins_5",
        "cost_0bps",
        "cost_1bps",
    ]
    assert summary.ablations_skipped == ["ssl_pretraining_ablation"]

    for filename in (
        "ablation_summary.json",
        "ablation_results.csv",
        "ablation_manifest.json",
    ):
        assert (output_dir / filename).is_file(), filename

    summary_payload = _read_json(output_dir / "ablation_summary.json")
    assert summary_payload["data_source_kind"] == "synthetic_fixture"
    assert "T" in summary_payload["created_at"]
    assert summary_payload["created_at"].endswith("Z")
    assert "not benchmark evidence" in " ".join(summary_payload["warnings"])

    child_dirs = {
        path.name
        for path in (output_dir / "experiments").iterdir()
        if path.is_dir()
    }
    assert child_dirs == set(summary.ablations_run)
    assert "ssl_pretraining_ablation" not in child_dirs

    for relative_path in summary.child_experiments.values():
        report = validate_experiment_directory(
            output_dir / relative_path,
            include_plots=True,
        )
        assert report.is_valid

    report_paths = {
        "reports/calibration_ablation.md",
        "reports/cost_sensitivity.md",
        "reports/ssl_pretraining_ablation.md",
    }
    assert report_paths.issubset(set(summary.reports_written))
    for relative_path in report_paths:
        assert (output_dir / relative_path).is_file()

    ssl_report = (
        output_dir / "reports" / "ssl_pretraining_ablation.md"
    ).read_text(encoding="utf-8")
    assert "skipped" in ssl_report.lower()
    assert (
        "reason: no traceable runner support for SSL "
        "pretraining/fine-tuning yet"
    ) in ssl_report
    assert "train-only pretraining" in ssl_report
    assert "held-out evaluation" in ssl_report
    assert "Key Metric Summary" not in ssl_report

    cost_report = (output_dir / "reports" / "cost_sensitivity.md").read_text(
        encoding="utf-8",
    )
    assert "synthetic fixture smoke run" in cost_report
    assert "not benchmark evidence" in cost_report

    results = pd.read_csv(output_dir / "ablation_results.csv")
    assert list(results.columns) == list(ABLATION_RESULTS_COLUMNS)
    assert int(results.isna().sum().sum()) == 0
    assert set(results["status"]) == {"run", "skipped"}

    run_rows = results[results["status"] == "run"]
    assert not run_rows.empty
    for value in pd.to_numeric(run_rows["metric_value"]).tolist():
        assert math.isfinite(float(value))

    skipped_rows = results[results["status"] == "skipped"]
    assert not skipped_rows.empty
    assert (skipped_rows["warning"].str.len() > 0).all()
    assert (skipped_rows["ablation_name"] == "ssl_pretraining_ablation").all()

    calibration_rows = results[results["ablation_name"] == "calibration_bins_5"]
    assert set(calibration_rows["calibration_bins"].astype(str)) == {"5"}
    cost_0_rows = results[results["ablation_name"] == "cost_0bps"]
    cost_1_rows = results[results["ablation_name"] == "cost_1bps"]
    assert "0.0" in set(cost_0_rows["cost_bps"].astype(str))
    assert "1.0" in set(cost_1_rows["cost_bps"].astype(str))

    calibration_config = _read_yaml(
        output_dir / "experiments" / "calibration_bins_5" / "config.yaml",
    )
    assert calibration_config["calibration"]["n_bins"] == 5
    cost_0_config = _read_yaml(
        output_dir / "experiments" / "cost_0bps" / "config.yaml",
    )
    assert cost_0_config["execution_sensitivity"]["cost_bps"] == [0.0]

    assert not list((output_dir / "experiments").glob("*ssl*"))
    assert not list(output_dir.glob("**/regime_breakdown.png"))


def test_feature_pattern_filtering_excludes_labels_and_tiny_matches(
    tmp_path: Path,
) -> None:
    label_config = _write_config_with_feature_patterns(tmp_path, ["label_*"])
    with pytest.raises(ValueError, match="no matching feature columns"):
        run_paper_experiment(
            config_path=label_config,
            data_path=TINY_FIXTURE_PATH,
            out_dir=tmp_path / "label_pattern",
            models=["majority"],
            overwrite=True,
        )

    one_column_config = _write_config_with_feature_patterns(
        tmp_path,
        ["bid_price_1"],
    )
    with pytest.raises(ValueError, match="too few matching feature columns"):
        run_paper_experiment(
            config_path=one_column_config,
            data_path=TINY_FIXTURE_PATH,
            out_dir=tmp_path / "one_column_pattern",
            models=["majority"],
            overwrite=True,
        )

    top_of_book_config = _write_config_with_feature_patterns(
        tmp_path,
        ["bid_price_1", "ask_price_1", "bid_quantity_1", "ask_quantity_1"],
    )
    summary = run_paper_experiment(
        config_path=top_of_book_config,
        data_path=TINY_FIXTURE_PATH,
        out_dir=tmp_path / "top_of_book_pattern",
        models=["majority"],
        overwrite=True,
    )
    assert summary.models_run == ["majority"]
    report = validate_experiment_directory(
        tmp_path / "top_of_book_pattern",
        include_plots=True,
    )
    assert report.is_valid


def test_smoke_ablation_rejects_unsupported_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported ablation set"):
        run_paper_ablations(
            config_path=CONFIG_PATH,
            data_path=TINY_FIXTURE_PATH,
            out_dir=tmp_path / "paper_ablation",
            models=["majority", "logistic"],
            ablation_set="unknown",
        )


def test_smoke_ablation_requires_existing_data_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        run_paper_ablations(
            config_path=CONFIG_PATH,
            data_path=tmp_path / "missing.csv",
            out_dir=tmp_path / "paper_ablation",
            models=["majority", "logistic"],
            ablation_set="smoke",
        )


def test_smoke_ablation_overwrite_protection(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_ablation"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty output directory"):
        run_paper_ablations(
            config_path=CONFIG_PATH,
            data_path=TINY_FIXTURE_PATH,
            out_dir=output_dir,
            models=["majority", "logistic"],
            ablation_set="smoke",
            overwrite=False,
        )

    assert sentinel.is_file()


def test_paper_ablation_cli_command_works_on_tiny_fixture(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "paper_ablation_cli"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chronoslob.cli",
            "run-paper-ablations",
            "--config",
            str(CONFIG_PATH),
            "--data-path",
            str(TINY_FIXTURE_PATH),
            "--out",
            str(output_dir),
            "--models",
            "majority,logistic",
            "--ablation-set",
            "smoke",
            "--overwrite",
        ],
        cwd=project_root(),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ChronosLOB paper ablation suite" in completed.stdout
    assert "ablation set:        smoke" in completed.stdout
    assert "ablations run:" in completed.stdout
    assert "ablations skipped:" in completed.stdout
    assert "ssl_pretraining_ablation" in completed.stdout
    assert (output_dir / "ablation_summary.json").is_file()
    assert (output_dir / "ablation_results.csv").is_file()
    assert (output_dir / "ablation_manifest.json").is_file()
