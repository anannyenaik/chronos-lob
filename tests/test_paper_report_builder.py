"""Tests for the Phase J empirical report builder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from chronoslob.experiments.ablations import run_paper_ablations
from chronoslob.experiments.paper_runner import run_paper_experiment
from chronoslob.experiments.reporting import build_paper_report, inspect_paper_report
from chronoslob.experiments.system_benchmarks import run_system_benchmarks
from chronoslob.utils.paths import project_root

CONFIG_PATH = project_root() / "configs" / "experiments" / "fi2010_midprice_h10.yaml"
TINY_FIXTURE_PATH = (
    project_root() / "tests" / "fixtures" / "fi2010" / "tiny_fi2010_like.csv"
)

REQUIRED_SECTIONS = (
    "## Abstract",
    "## 1. Dataset and provenance",
    "## 2. Label construction",
    "## 3. Leakage controls and temporal validation",
    "## 4. Models",
    "## 5. Predictive results",
    "## 6. Calibration results",
    "## 7. Execution-aware sensitivity",
    "## 8. Ablations and robustness",
    "## 9. Systems benchmarks",
    "## 10. Failure cases and warnings",
    "## 11. Limitations",
    "## 12. Reproducibility commands",
)


@pytest.fixture(scope="module")
def smoke_dirs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("paper_report")
    experiment_dir = base / "paper_experiment_smoke"
    ablation_dir = base / "paper_ablation_smoke"
    systems_dir = base / "system_benchmark_smoke"

    run_paper_experiment(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=experiment_dir,
        models=["majority", "logistic"],
        overwrite=True,
        build_plots=False,
    )
    run_paper_ablations(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=ablation_dir,
        models=["majority", "logistic"],
        ablation_set="smoke",
        overwrite=True,
    )
    run_system_benchmarks(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=systems_dir,
        benchmark_set="smoke",
        models=["majority", "logistic"],
        overwrite=True,
    )
    return {
        "base": base,
        "experiment": experiment_dir,
        "ablations": ablation_dir,
        "systems": systems_dir,
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _phrase(*parts: str) -> str:
    return " ".join(parts)


def test_report_builder_writes_markdown_and_summary(
    smoke_dirs: dict[str, Path],
) -> None:
    report_path = smoke_dirs["base"] / "chronoslob_empirical_report_smoke.md"

    summary = build_paper_report(
        experiment_dir=smoke_dirs["experiment"],
        ablation_dir=smoke_dirs["ablations"],
        systems_dir=smoke_dirs["systems"],
        out_path=report_path,
        overwrite=True,
    )

    summary_path = report_path.with_name("chronoslob_empirical_report_smoke_summary.json")
    assert report_path.is_file()
    assert summary_path.is_file()
    assert summary.report_path.endswith("chronoslob_empirical_report_smoke.md")
    assert summary.fixture_or_smoke_run is True
    assert len(summary.sections_written) == len(REQUIRED_SECTIONS)
    assert summary.artefacts_used

    payload = _read_json(summary_path)
    assert payload["fixture_or_smoke_run"] is True
    assert payload["sections_written"] == summary.sections_written
    assert payload["artefacts_used"]
    assert "T" in payload["created_at"]


def test_report_contains_required_sections_and_stored_evidence(
    smoke_dirs: dict[str, Path],
) -> None:
    report_path = smoke_dirs["base"] / "section_report.md"
    build_paper_report(
        experiment_dir=smoke_dirs["experiment"],
        ablation_dir=smoke_dirs["ablations"],
        systems_dir=smoke_dirs["systems"],
        out_path=report_path,
        overwrite=True,
    )

    text = report_path.read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in text

    assert "| majority | test |" in text
    assert "| logistic | test |" in text
    assert "`calibration_bins.csv` is present" in text
    assert "`execution_sensitivity.csv` is present" in text
    assert "ssl_pretraining_ablation" in text
    assert "skipped" in text
    assert "loader_throughput" in text
    assert "rows_per_second" in text
    assert "Failure cases and warnings" in text
    assert "smoke report" in text.lower()
    assert "not benchmark evidence" in text.lower()


def test_report_handles_missing_optional_inputs(
    smoke_dirs: dict[str, Path],
) -> None:
    report_path = smoke_dirs["base"] / "experiment_only_report.md"
    build_paper_report(
        experiment_dir=smoke_dirs["experiment"],
        out_path=report_path,
        overwrite=True,
    )

    text = report_path.read_text(encoding="utf-8")
    assert "Ablation directory: not supplied." in text
    assert "Systems benchmark directory: not supplied." in text


def test_report_limitations_do_not_overclaim(smoke_dirs: dict[str, Path]) -> None:
    report_path = smoke_dirs["base"] / "limitations_report.md"
    build_paper_report(
        experiment_dir=smoke_dirs["experiment"],
        ablation_dir=smoke_dirs["ablations"],
        systems_dir=smoke_dirs["systems"],
        out_path=report_path,
        overwrite=True,
    )

    lowered = report_path.read_text(encoding="utf-8").lower()
    assert "no live trading" in lowered
    assert "no production market impact model" in lowered
    assert "production backtest" in lowered
    for forbidden in (
        _phrase("profitable", "strategy"),
        _phrase("market", "beating"),
        _phrase("high", "sharpe"),
    ):
        assert forbidden not in lowered


def test_missing_required_experiment_dir_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="paper experiment directory"):
        build_paper_report(
            experiment_dir=tmp_path / "missing",
            out_path=tmp_path / "report.md",
        )


def test_report_overwrite_protection(smoke_dirs: dict[str, Path]) -> None:
    report_path = smoke_dirs["base"] / "overwrite_report.md"
    build_paper_report(
        experiment_dir=smoke_dirs["experiment"],
        out_path=report_path,
        overwrite=True,
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_paper_report(
            experiment_dir=smoke_dirs["experiment"],
            out_path=report_path,
            overwrite=False,
        )


def test_cli_build_and_inspect_paper_report(
    smoke_dirs: dict[str, Path],
) -> None:
    report_path = smoke_dirs["base"] / "cli_report.md"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chronoslob.cli",
            "build-paper-report",
            "--experiment",
            str(smoke_dirs["experiment"]),
            "--ablations",
            str(smoke_dirs["ablations"]),
            "--systems",
            str(smoke_dirs["systems"]),
            "--out",
            str(report_path),
            "--overwrite",
        ],
        cwd=project_root(),
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "ChronosLOB empirical report builder" in completed.stdout
    assert "fixture/smoke run:  yes" in completed.stdout
    assert report_path.is_file()
    assert report_path.with_name("cli_report_summary.json").is_file()

    inspection = inspect_paper_report(report_path)
    assert inspection.fixture_or_smoke_run is True
    assert inspection.artefacts_used_count > 0
    assert len(inspection.sections_detected) == len(REQUIRED_SECTIONS)

    inspected = subprocess.run(
        [
            sys.executable,
            "-m",
            "chronoslob.cli",
            "inspect-paper-report",
            "--report",
            str(report_path),
        ],
        cwd=project_root(),
        capture_output=True,
        check=False,
        text=True,
    )
    assert inspected.returncode == 0, inspected.stderr
    assert "ChronosLOB empirical report inspection" in inspected.stdout
    assert "summary JSON path:" in inspected.stdout


def test_fixture_report_is_not_written_under_public_reports(
    smoke_dirs: dict[str, Path],
) -> None:
    target = project_root() / "reports" / "chronoslob_empirical_report.md"
    if target.exists():
        pytest.skip("real empirical report already exists")

    with pytest.raises(ValueError, match="fixture or smoke empirical report"):
        build_paper_report(
            experiment_dir=smoke_dirs["experiment"],
            out_path=target,
            overwrite=True,
        )
    assert not target.exists()
