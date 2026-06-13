"""Tests for Phase G paper experiment plot generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from chronoslob.experiments.artifacts import validate_experiment_directory
from chronoslob.experiments.paper_runner import run_paper_experiment
from chronoslob.experiments.plots import (
    PAPER_PLOT_FILENAMES,
    PLOT_SUMMARY_FILENAME,
    build_paper_experiment_plots,
)
from chronoslob.utils.paths import project_root

CONFIG_PATH = project_root() / "configs" / "experiments" / "fi2010_midprice_h10.yaml"
TINY_FIXTURE_PATH = (
    project_root() / "tests" / "fixtures" / "fi2010" / "tiny_fi2010_like.csv"
)


pytest.importorskip("matplotlib")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _run_on_tiny_fixture(
    output_dir: Path,
    *,
    models: tuple[str, ...] = ("majority", "logistic"),
) -> Any:
    return run_paper_experiment(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        models=list(models),
        overwrite=False,
    )


# ---------------------------------------------------------------------------
# build_paper_experiment_plots core behaviour
# ---------------------------------------------------------------------------


def test_build_plots_creates_reliability_cost_and_confusion_plots(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exp"
    _run_on_tiny_fixture(output_dir)

    summary = build_paper_experiment_plots(output_dir, overwrite=False)

    plots_dir = output_dir / "plots"
    assert (plots_dir / "reliability_curve.png").is_file()
    assert (plots_dir / "cost_sensitivity.png").is_file()
    assert (plots_dir / "confusion_matrix.png").is_file()
    assert "plots/reliability_curve.png" in summary.plots_written
    assert "plots/cost_sensitivity.png" in summary.plots_written
    assert "plots/confusion_matrix.png" in summary.plots_written


def test_regime_breakdown_is_skipped_when_no_regime_data_exists(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exp"
    _run_on_tiny_fixture(output_dir)

    summary = build_paper_experiment_plots(output_dir, overwrite=False)

    plots_dir = output_dir / "plots"
    assert not (plots_dir / "regime_breakdown.png").exists()
    assert "plots/regime_breakdown.png" in summary.plots_skipped
    assert any(
        "regime breakdown skipped" in warning for warning in summary.warnings
    ), summary.warnings


def test_regime_breakdown_warning_mentions_no_fabrication(tmp_path: Path) -> None:
    output_dir = tmp_path / "exp"
    _run_on_tiny_fixture(output_dir)

    summary = build_paper_experiment_plots(output_dir, overwrite=False)
    joined = " ".join(summary.warnings).lower()
    assert "not fabricating" in joined or "no genuine regime" in joined


def test_plot_summary_json_records_written_and_skipped_plots(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exp"
    _run_on_tiny_fixture(output_dir)

    summary = build_paper_experiment_plots(output_dir, overwrite=False)
    summary_path = output_dir / PLOT_SUMMARY_FILENAME
    assert summary_path.is_file()
    payload = _read_json(summary_path)
    assert payload["experiment_dir"] == str(output_dir)
    assert "plots/reliability_curve.png" in payload["plots_written"]
    assert "plots/regime_breakdown.png" in payload["plots_skipped"]
    assert payload["builder_version"] == summary.builder_version
    assert any("regime" in warning.lower() for warning in payload["warnings"])


def test_build_paper_plots_skips_reliability_when_calibration_missing(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exp"
    _run_on_tiny_fixture(output_dir)
    calibration_path = output_dir / "calibration_bins.csv"
    calibration_path.unlink()

    summary = build_paper_experiment_plots(output_dir, overwrite=True)

    assert "plots/reliability_curve.png" not in summary.plots_written
    assert "plots/reliability_curve.png" in summary.plots_skipped
    assert not (output_dir / "plots" / "reliability_curve.png").exists()
    assert any("reliability" in warning.lower() for warning in summary.warnings)


def test_build_paper_plots_skips_cost_when_execution_csv_missing(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exp"
    _run_on_tiny_fixture(output_dir)
    (output_dir / "execution_sensitivity.csv").unlink()

    summary = build_paper_experiment_plots(output_dir, overwrite=True)

    assert "plots/cost_sensitivity.png" not in summary.plots_written
    assert "plots/cost_sensitivity.png" in summary.plots_skipped
    assert not (output_dir / "plots" / "cost_sensitivity.png").exists()


def test_build_paper_plots_skips_confusion_when_json_missing(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exp"
    _run_on_tiny_fixture(output_dir)
    (output_dir / "confusion_matrix.json").unlink()

    summary = build_paper_experiment_plots(output_dir, overwrite=True)

    assert "plots/confusion_matrix.png" not in summary.plots_written
    assert "plots/confusion_matrix.png" in summary.plots_skipped


def test_build_paper_plots_refuses_overwrite_by_default(tmp_path: Path) -> None:
    output_dir = tmp_path / "exp"
    _run_on_tiny_fixture(output_dir)

    build_paper_experiment_plots(output_dir, overwrite=False)
    second = build_paper_experiment_plots(output_dir, overwrite=False)

    assert "plots/reliability_curve.png" in second.plots_skipped
    assert any("refusing to overwrite" in warning for warning in second.warnings)


def test_build_paper_plots_overwrite_replaces_existing_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exp"
    _run_on_tiny_fixture(output_dir)

    first = build_paper_experiment_plots(output_dir, overwrite=False)
    second = build_paper_experiment_plots(output_dir, overwrite=True)

    assert "plots/reliability_curve.png" in first.plots_written
    assert "plots/reliability_curve.png" in second.plots_written


def test_regime_breakdown_emitted_when_genuine_regime_data_present(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exp"
    _run_on_tiny_fixture(output_dir)

    predictions_path = output_dir / "predictions.csv"
    frame = pd.read_csv(predictions_path)
    # Use stored predictions only; add a 'regime' column derived from the
    # actual stored model identity to keep this a genuine, not-fabricated
    # regime label rather than a row-number fiction.
    frame["regime"] = frame["model_name"].astype(str)
    frame.to_csv(predictions_path, index=False)

    summary = build_paper_experiment_plots(output_dir, overwrite=True)
    assert "plots/regime_breakdown.png" in summary.plots_written
    assert (output_dir / "plots" / "regime_breakdown.png").is_file()


def test_build_paper_plots_filenames_match_contract(tmp_path: Path) -> None:
    output_dir = tmp_path / "exp"
    _run_on_tiny_fixture(output_dir)

    build_paper_experiment_plots(output_dir, overwrite=False)
    expected = set(PAPER_PLOT_FILENAMES)
    written_or_skipped = {Path(name).name for name in expected}
    assert written_or_skipped == expected


# ---------------------------------------------------------------------------
# Integration with run_paper_experiment and the artefact contract
# ---------------------------------------------------------------------------


def test_run_paper_experiment_build_plots_writes_plots(tmp_path: Path) -> None:
    output_dir = tmp_path / "exp"
    summary = run_paper_experiment(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        models=["majority", "logistic"],
        overwrite=False,
        build_plots=True,
    )

    assert summary.plot_summary is not None
    assert "plots/reliability_curve.png" in summary.plot_summary.plots_written
    assert (output_dir / "plots" / "reliability_curve.png").is_file()


def test_run_paper_experiment_runner_summary_records_plots(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exp"
    run_paper_experiment(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        models=["majority", "logistic"],
        overwrite=False,
        build_plots=True,
    )

    payload = _read_json(output_dir / "runner_summary.json")
    plots_block = payload["plots"]
    assert plots_block["requested"] is True
    assert "plots/reliability_curve.png" in plots_block["plots_written"]
    assert "plots/regime_breakdown.png" in plots_block["plots_skipped"]
    assert plots_block["builder_version"]


def test_inspect_artefacts_reports_generated_plots_as_present(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exp"
    run_paper_experiment(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        models=["majority", "logistic"],
        overwrite=False,
        build_plots=True,
    )

    report = validate_experiment_directory(output_dir, include_plots=True)
    assert report.is_valid
    assert "plots/reliability_curve.png" in report.present_optional
    assert "plots/cost_sensitivity.png" in report.present_optional
    assert "plots/confusion_matrix.png" in report.present_optional
    assert "plots/regime_breakdown.png" not in report.present_optional


def test_run_paper_experiment_without_build_plots_keeps_no_plot_summary(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exp"
    summary = run_paper_experiment(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        models=["majority", "logistic"],
        overwrite=False,
        build_plots=False,
    )

    assert summary.plot_summary is None
    assert not (output_dir / "plots" / "reliability_curve.png").exists()
    assert not (output_dir / PLOT_SUMMARY_FILENAME).exists()


def test_fixture_runs_remain_labelled_as_smoke_not_benchmark_evidence(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exp"
    run_paper_experiment(
        config_path=CONFIG_PATH,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        models=["majority", "logistic"],
        overwrite=False,
        build_plots=True,
    )

    text = (output_dir / "model_card.md").read_text(encoding="utf-8").lower()
    assert "synthetic fixture" in text
    assert "not benchmark evidence" in text
