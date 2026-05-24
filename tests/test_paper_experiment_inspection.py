"""Tests for the inspect-paper-experiment CLI command (Phase G)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chronoslob.cli import (
    _build_paper_plots_impl,
    _inspect_paper_experiment_impl,
)
from chronoslob.experiments.paper_runner import run_paper_experiment
from chronoslob.utils.paths import project_root

CONFIG_PATH = project_root() / "configs" / "experiments" / "fi2010_midprice_h10.yaml"
TINY_FIXTURE_PATH = (
    project_root() / "tests" / "fixtures" / "fi2010" / "tiny_fi2010_like.csv"
)


pytest.importorskip("matplotlib")


def _prepare_experiment(output_dir: Path, *, build_plots: bool = True) -> Any:
    return run_paper_experiment(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        models=["majority", "logistic"],
        overwrite=False,
        build_plots=build_plots,
    )


def test_inspect_paper_experiment_prints_expected_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "exp"
    _prepare_experiment(output_dir)

    exit_code = _inspect_paper_experiment_impl(experiment=output_dir)
    assert exit_code == 0
    captured = capsys.readouterr().out
    for fragment in (
        "ChronosLOB paper experiment inspection",
        "experiment dir:",
        "artefact validation:",
        "requested models:",
        "models run:",
        "evidence streams:",
        "prediction rows:",
        "calibration rows:",
        "execution rows:",
        "plots present:",
        "fixture run:",
        "network calls:",
    ):
        assert fragment in captured, f"missing expected fragment {fragment!r}"


def test_inspect_paper_experiment_lists_generated_plots(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "exp"
    _prepare_experiment(output_dir)

    _inspect_paper_experiment_impl(experiment=output_dir)
    captured = capsys.readouterr().out
    assert "plots/reliability_curve.png" in captured
    assert "plots/cost_sensitivity.png" in captured
    assert "plots/confusion_matrix.png" in captured
    assert "plots/regime_breakdown.png" in captured


def test_inspect_paper_experiment_flags_fixture_smoke_runs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "exp"
    _prepare_experiment(output_dir)

    _inspect_paper_experiment_impl(experiment=output_dir)
    captured = capsys.readouterr().out.lower()
    assert "synthetic fixture smoke run" in captured
    assert "not benchmark evidence" in captured


def test_inspect_paper_experiment_reports_missing_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "does_not_exist"
    exit_code = _inspect_paper_experiment_impl(experiment=missing)
    assert exit_code != 0


def test_build_paper_plots_cli_impl_returns_success_on_smoke_run(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exp"
    _prepare_experiment(output_dir, build_plots=False)

    exit_code = _build_paper_plots_impl(
        experiment=output_dir,
        overwrite=False,
    )
    assert exit_code == 0
    assert (output_dir / "plots" / "reliability_curve.png").is_file()


def test_build_paper_plots_cli_impl_overwrite_replaces_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exp"
    _prepare_experiment(output_dir, build_plots=False)

    first = _build_paper_plots_impl(experiment=output_dir, overwrite=False)
    second = _build_paper_plots_impl(experiment=output_dir, overwrite=True)
    assert first == 0
    assert second == 0
    assert (output_dir / "plots" / "cost_sensitivity.png").is_file()


def test_build_paper_plots_cli_impl_reports_missing_directory(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    exit_code = _build_paper_plots_impl(experiment=missing, overwrite=False)
    assert exit_code == 2
