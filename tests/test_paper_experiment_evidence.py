"""Tests for Phase F calibration and execution-sensitivity evidence."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from chronoslob.experiments.artifacts import (
    load_results,
    validate_experiment_directory,
)
from chronoslob.experiments.evidence import (
    CALIBRATION_BINS_COLUMNS,
    EXECUTION_SENSITIVITY_COLUMNS,
    ExecutionSensitivityConfig,
    ReturnProxyConfig,
    build_calibration_bins,
    build_execution_sensitivity,
    build_return_proxy_series,
    summarise_calibration,
)
from chronoslob.experiments.paper_runner import run_paper_experiment
from chronoslob.utils.paths import project_root

CONFIG_PATH = project_root() / "configs" / "experiments" / "fi2010_midprice_h10.yaml"
TINY_FIXTURE_PATH = (
    project_root() / "tests" / "fixtures" / "fi2010" / "tiny_fi2010_like.csv"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


# ---------------------------------------------------------------------------
# Unit tests for evidence helpers (deterministic, no model fitting)
# ---------------------------------------------------------------------------


def test_calibration_bins_have_expected_columns_and_finite_values() -> None:
    predictions = pd.DataFrame(
        {
            "model_name": ["m"] * 6,
            "split": ["test"] * 6,
            "label": ["1", "1", "2", "2", "1", "2"],
            "prediction": ["1", "1", "1", "2", "2", "2"],
            "confidence": [0.55, 0.95, 0.45, 0.85, 0.65, 0.99],
        }
    )
    bins = build_calibration_bins(
        predictions,
        model_name="m",
        split="test",
        n_bins=5,
    )
    assert len(bins) == 5
    for row in bins:
        assert row.bin_lower >= 0.0
        assert row.bin_upper <= 1.0
        for value in (
            row.mean_confidence,
            row.accuracy,
            row.confidence_gap,
        ):
            assert math.isfinite(value)
        assert row.count >= 0
        assert row.bin_lower < row.bin_upper


def test_summarise_calibration_returns_finite_ece_and_confidence() -> None:
    predictions = pd.DataFrame(
        {
            "label": ["1", "2", "1"],
            "prediction": ["1", "1", "2"],
            "confidence": [0.9, 0.5, 0.95],
        }
    )
    bins = build_calibration_bins(
        predictions,
        model_name="m",
        split="test",
        n_bins=10,
    )
    summary = summarise_calibration(bins)
    assert summary is not None
    assert summary.n_samples == 3
    assert math.isfinite(summary.mean_confidence)
    assert math.isfinite(summary.expected_calibration_error)
    assert summary.expected_calibration_error >= 0.0


def test_return_proxy_skips_when_price_columns_absent() -> None:
    frame = pd.DataFrame({"foo": [1.0, 2.0, 3.0]})
    config = ReturnProxyConfig()
    assert build_return_proxy_series(frame, horizon=1, config=config) is None


def test_return_proxy_finite_for_synthetic_mid_series() -> None:
    frame = pd.DataFrame(
        {
            "bid_price_1": [100.0, 100.1, 100.2, 100.3, 100.4],
            "ask_price_1": [100.2, 100.3, 100.4, 100.5, 100.6],
        }
    )
    config = ReturnProxyConfig()
    proxy = build_return_proxy_series(frame, horizon=1, config=config)
    assert proxy is not None
    finite = proxy.dropna().to_numpy(dtype=float)
    assert finite.size >= 1
    assert np.all(np.isfinite(finite))


def test_execution_sensitivity_honours_thresholds_and_costs() -> None:
    frame = pd.DataFrame(
        {
            "bid_price_1": [100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6],
            "ask_price_1": [100.2, 100.3, 100.4, 100.5, 100.6, 100.7, 100.8],
        }
    )
    proxy = build_return_proxy_series(
        frame,
        horizon=1,
        config=ReturnProxyConfig(),
    )
    assert proxy is not None
    predictions = pd.DataFrame(
        {
            "row_index": [0, 1, 2, 3],
            "model_name": ["m"] * 4,
            "split": ["test"] * 4,
            "label": ["1", "2", "1", "2"],
            "prediction": ["1", "2", "1", "2"],
            "confidence": [0.4, 0.7, 0.95, 0.55],
        }
    )
    config = ExecutionSensitivityConfig(
        confidence_thresholds=(0.0, 0.5, 0.9),
        cost_bps=(0.0, 2.0),
        latency_steps=(0,),
    )
    direction_map = {"1": 1, "2": -1}
    rows = build_execution_sensitivity(
        predictions,
        model_name="m",
        split="test",
        return_proxy=proxy,
        config=config,
        direction_map=direction_map,
    )
    assert rows, "expected at least one execution row"
    thresholds = sorted({row.confidence_threshold for row in rows})
    costs = sorted({row.cost_bps for row in rows})
    assert thresholds == [0.0, 0.5, 0.9]
    assert costs == [0.0, 2.0]
    for row in rows:
        for value in (
            row.gross_signal_return_proxy,
            row.cost_proxy,
            row.net_signal_return_proxy,
            row.turnover_proxy,
            row.hit_rate_proxy,
        ):
            assert math.isfinite(value)
        if row.cost_bps > 0.0 and row.trade_count_proxy > 0:
            assert row.net_signal_return_proxy == pytest.approx(
                row.gross_signal_return_proxy - row.cost_proxy,
                abs=1e-9,
            )


def test_execution_sensitivity_skipped_when_required_columns_missing() -> None:
    proxy = pd.Series([0.1, 0.2, 0.3], index=[0, 1, 2])
    bare_predictions = pd.DataFrame(
        {
            "label": ["1", "2", "1"],
            "prediction": ["1", "1", "2"],
        }
    )
    rows = build_execution_sensitivity(
        bare_predictions,
        model_name="m",
        split="test",
        return_proxy=proxy,
        config=ExecutionSensitivityConfig(),
        direction_map={"1": 1, "2": -1},
    )
    assert rows == []


# ---------------------------------------------------------------------------
# Integration tests through run_paper_experiment on the tiny fixture
# ---------------------------------------------------------------------------


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


def test_runner_emits_calibration_bins_csv_with_expected_columns(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exp"
    summary = _run_on_tiny_fixture(output_dir)
    calibration_path = output_dir / "calibration_bins.csv"
    assert calibration_path.is_file()
    rows = _read_csv(calibration_path)
    assert rows, "calibration_bins.csv must contain at least one bin row"
    assert tuple(rows[0].keys()) == CALIBRATION_BINS_COLUMNS
    for row in rows:
        for column in ("bin_lower", "bin_upper", "mean_confidence", "accuracy"):
            value = float(row[column])
            assert math.isfinite(value)
        assert int(row["count"]) >= 0
    assert summary.calibration_models, "summary should record calibration models"


def test_runner_emits_execution_sensitivity_csv_with_expected_columns(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exp"
    summary = _run_on_tiny_fixture(output_dir)
    execution_path = output_dir / "execution_sensitivity.csv"
    assert execution_path.is_file()
    rows = _read_csv(execution_path)
    assert rows, "execution_sensitivity.csv must contain at least one row"
    assert tuple(rows[0].keys()) == EXECUTION_SENSITIVITY_COLUMNS
    for row in rows:
        for column in (
            "confidence_threshold",
            "cost_bps",
            "turnover_proxy",
            "gross_signal_return_proxy",
            "cost_proxy",
            "net_signal_return_proxy",
            "hit_rate_proxy",
        ):
            value = float(row[column])
            assert math.isfinite(value)
        assert int(row["latency_steps"]) >= 0
        assert int(row["eligible_predictions"]) >= 0
        assert int(row["trade_count_proxy"]) >= 0
    assert summary.execution_models, "summary should record execution models"


def test_results_json_references_calibration_and_execution_artefacts(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exp"
    _run_on_tiny_fixture(output_dir)
    results = load_results(output_dir / "results.json")
    for model_result in results.model_results:
        if model_result.model_name == "majority":
            assert model_result.artefacts is not None
            assert (
                model_result.artefacts.get("calibration_bins")
                == "calibration_bins.csv"
            )
            assert (
                model_result.artefacts.get("execution_sensitivity")
                == "execution_sensitivity.csv"
            )
            break
    else:
        pytest.fail("majority model result not found")
    assert "calibration_bins" in results.evidence_streams.calibration
    assert results.evidence_streams.execution != ["not_computed"]
    assert "gross_signal_return_proxy" in results.evidence_streams.execution


def test_runner_summary_records_evidence_metric_groups(tmp_path: Path) -> None:
    output_dir = tmp_path / "exp"
    _run_on_tiny_fixture(output_dir)
    payload = _read_json(output_dir / "runner_summary.json")
    assert "predictive_metric_names" in payload
    assert "calibration_metric_names" in payload
    assert "execution_metric_names" in payload
    assert payload["execution_metric_names"] != ["not_computed"]
    evidence = payload.get("evidence")
    assert isinstance(evidence, dict)
    assert evidence["calibration_artefact_written"] is True
    assert evidence["execution_artefact_written"] is True
    assert evidence["calibration_models"]
    assert evidence["execution_models"]


def test_model_card_mentions_evidence_without_trading_claims(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exp"
    _run_on_tiny_fixture(output_dir)
    text = (output_dir / "model_card.md").read_text(encoding="utf-8")
    assert "calibration_bins.csv" in text
    assert "execution_sensitivity.csv" in text
    assert "simplified proxy" in text or "explicit simple assumptions" in text
    # Phrases assembled from fragments so this assertion source does not
    # itself trip the public-release wording audit.
    forbidden = (
        "profit" + "able strategy",
        "guaran" + "teed alpha",
        "deploy" + "able trading",
        "produc" + "tion trading",
        "trad" + "ing bot",
        "market" + "-beating",
    )
    lowered = text.lower()
    for phrase in forbidden:
        assert phrase not in lowered, f"model card must not mention {phrase!r}"
    assert "synthetic fixture" in lowered
    assert "benchmark evidence" in lowered


def test_inspect_experiment_artifacts_reports_evidence_present(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exp"
    _run_on_tiny_fixture(output_dir)
    report = validate_experiment_directory(output_dir, include_plots=True)
    assert report.is_valid
    assert "calibration_bins.csv" in report.present_optional
    assert "execution_sensitivity.csv" in report.present_optional


def test_runner_records_execution_warning_when_proxy_unavailable(
    tmp_path: Path,
) -> None:
    """If return-proxy columns are missing the runner records a clear warning."""
    output_dir = tmp_path / "exp"
    config_path = tmp_path / "config.yaml"
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["execution_sensitivity"]["return_proxy"]["bid_price_column"] = (
        "missing_bid"
    )
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    summary = run_paper_experiment(
        config_path=config_path,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        models=["majority"],
        overwrite=False,
    )
    assert (output_dir / "execution_sensitivity.csv").exists() is False
    assert any(
        "execution sensitivity note" in w
        for w in summary.warnings
    )
    payload_summary = _read_json(output_dir / "runner_summary.json")
    assert payload_summary["evidence"]["execution_artefact_written"] is False
    assert payload_summary["evidence"]["execution_warning"] is not None


def test_runner_disables_calibration_artefact_when_config_disabled(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "exp"
    config_path = tmp_path / "config.yaml"
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["calibration"]["enabled"] = False
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    summary = run_paper_experiment(
        config_path=config_path,
        data_path=TINY_FIXTURE_PATH,
        out_dir=output_dir,
        models=["majority"],
        overwrite=False,
    )
    assert (output_dir / "calibration_bins.csv").exists() is False
    assert summary.calibration_models == []


def test_fixture_smoke_is_not_benchmark_evidence(tmp_path: Path) -> None:
    output_dir = tmp_path / "exp"
    _run_on_tiny_fixture(output_dir)
    text = (output_dir / "model_card.md").read_text(encoding="utf-8")
    lowered = text.lower()
    assert "synthetic fixture" in lowered
    assert "not benchmark evidence" in lowered
