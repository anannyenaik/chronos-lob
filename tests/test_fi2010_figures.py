"""Tests for FI-2010 neural full-grid figure generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from chronoslob.analysis.fi2010_figures import (
    build_fi2010_neural_figures,
    build_matched_ssl_delta_rows,
    select_best_models_by_horizon,
)
from chronoslob.analysis.fi2010_label_mapping import (
    FI2010_CANONICAL_CLASS_ORDER,
    canonical_class_name,
    class_name_to_raw_label,
    classwise_f1_from_row,
    labels_to_canonical_class_names,
    labels_to_raw_labels,
    probability_columns_for_order,
    validate_classwise_f1_columns,
    validate_confusion_matrix_axis_labels,
    validate_probability_columns,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prediction_rows(*, ambiguous: bool = False, regime: bool = False) -> pd.DataFrame:
    base = pd.DataFrame(
        [
            {
                "row_id": 1,
                "y_true": 1,
                "y_pred": 1,
                "prob_up": 0.72,
                "prob_stationary": 0.18,
                "prob_down": 0.10,
                "confidence": 0.72,
            },
            {
                "row_id": 2,
                "y_true": 2,
                "y_pred": 2,
                "prob_up": 0.15,
                "prob_stationary": 0.70,
                "prob_down": 0.15,
                "confidence": 0.70,
            },
            {
                "row_id": 3,
                "y_true": 3,
                "y_pred": 1,
                "prob_up": 0.55,
                "prob_stationary": 0.15,
                "prob_down": 0.30,
                "confidence": 0.55,
            },
            {
                "row_id": 4,
                "y_true": 1,
                "y_pred": 3,
                "prob_up": 0.34,
                "prob_stationary": 0.21,
                "prob_down": 0.45,
                "confidence": 0.45,
            },
        ]
    )
    if ambiguous:
        return base.drop(columns=["prob_up", "prob_stationary", "prob_down"]).assign(
            prob_0=[0.72, 0.15, 0.55, 0.34],
            prob_1=[0.18, 0.70, 0.15, 0.21],
            prob_2=[0.10, 0.15, 0.30, 0.45],
        )
    if regime:
        base["regime"] = ["calm", "calm", "volatile", "volatile"]
    return base


def _result_row(
    *,
    objective: str,
    prediction_file: str,
    macro_f1: float,
    ece: float,
    seed: int = 0,
    horizon: int = 10,
    lookback: int = 20,
) -> dict[str, Any]:
    pretraining = "none" if objective == "supervised" else objective
    return {
        "fold": 1,
        "horizon": horizon,
        "seed": seed,
        "lookback": lookback,
        "model_family": "matrix_transformer",
        "pretraining_objective": pretraining,
        "accuracy": 0.5,
        "macro_f1": macro_f1,
        "mcc": macro_f1 - 0.2,
        "ece": ece,
        "brier_score": 0.4,
        "nll": 1.0,
        "class_f1_down": 0.3,
        "class_f1_stationary": 0.4,
        "class_f1_up": 0.5,
        "checkpoint_hash": "hash",
        "prediction_file": prediction_file,
        "status": "completed",
        "run_id": f"run_{objective}_{seed}_{horizon}",
        "run_dir": str(Path(prediction_file).parent),
        "architecture_hash": "arch",
        "preprocessing_hash": "prep",
    }


def _write_synthetic_grid(
    base: Path,
    *,
    smoke: bool = False,
    ambiguous: bool = False,
    missing_predictions: bool = False,
    regime: bool = False,
) -> Path:
    grid = base / "grid"
    grid.mkdir(parents=True)
    _write_json(
        grid / "summary.json",
        {
            "execution_mode": "smoke" if smoke else "benchmark",
            "smoke_test": smoke,
            "folds": [1],
            "horizons": [10],
            "seeds": [0],
            "lookbacks": [20],
            "completed_run_count": 3,
            "failed_run_count": 0,
            "core_grid_complete": False,
        },
    )
    rows = []
    for objective, macro_f1, ece in (
        ("supervised", 0.50, 0.12),
        ("masked_reconstruction", 0.60, 0.08),
        ("next_field", 0.45, 0.14),
    ):
        prediction_file = f"runs/{objective}/predictions.csv"
        rows.append(
            _result_row(
                objective=objective,
                prediction_file=prediction_file,
                macro_f1=macro_f1,
                ece=ece,
            )
        )
        if not missing_predictions:
            path = grid / prediction_file
            path.parent.mkdir(parents=True, exist_ok=True)
            _prediction_rows(ambiguous=ambiguous, regime=regime).to_csv(path, index=False)
    pd.DataFrame(rows).to_csv(grid / "results_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(grid / "aggregate_summary.csv", index=False)
    _write_json(grid / "aggregate_summary.json", {"aggregate": []})
    pd.DataFrame(columns=["fold", "horizon", "seed", "objective", "reason"]).to_csv(
        grid / "failures.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "fold": 1,
                "horizon": 10,
                "seed": 0,
                "lookback": 20,
                "model_family": "matrix_transformer",
                "ssl_objective": "masked_reconstruction",
                "delta_macro_f1": 0.10,
                "delta_mcc": 0.10,
                "delta_ece": -0.04,
                "status": "matched",
            }
        ]
    ).to_csv(grid / "ssl_comparison.csv", index=False)
    return grid


def test_fi2010_label_mapping_conversion() -> None:
    assert canonical_class_name(1) == "up"
    assert canonical_class_name("2") == "stationary"
    assert canonical_class_name(3.0) == "down"
    assert class_name_to_raw_label("up") == 1
    assert labels_to_canonical_class_names([1, "stationary", 3]) == [
        "up",
        "stationary",
        "down",
    ]
    assert labels_to_raw_labels(["up", "stationary", "down"]) == [1, 2, 3]


def test_probability_column_validation_uses_named_fi2010_mapping() -> None:
    validation = validate_probability_columns(["prob_down", "prob_stationary", "prob_up"])
    assert validation.passed
    assert probability_columns_for_order(["prob_down", "prob_stationary", "prob_up"]) == (
        "prob_up",
        "prob_stationary",
        "prob_down",
    )
    ambiguous = validate_probability_columns(["prob_0", "prob_1", "prob_2"])
    assert not ambiguous.passed
    assert any("ambiguous" in error for error in ambiguous.errors)


def test_confusion_matrix_axis_and_classwise_f1_order() -> None:
    assert validate_confusion_matrix_axis_labels(FI2010_CANONICAL_CLASS_ORDER).passed
    assert not validate_confusion_matrix_axis_labels(["down", "stationary", "up"]).passed
    assert validate_classwise_f1_columns(
        ["class_f1_up", "class_f1_stationary", "class_f1_down"]
    ).passed
    extracted = classwise_f1_from_row(
        {"class_f1_up": 0.7, "class_f1_stationary": 0.5, "class_f1_down": 0.2}
    )
    assert list(extracted) == ["up", "stationary", "down"]
    assert extracted["down"] == pytest.approx(0.2)


def test_figure_manifest_creation_and_skipped_placeholders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    # Isolate from any real repo execution-v3 artefacts so the no-execution-v3
    # skip path is exercised regardless of local experiment outputs.
    monkeypatch.setattr(
        "chronoslob.analysis.fi2010_figures.project_root",
        lambda: tmp_path,
    )
    grid = _write_synthetic_grid(tmp_path)
    out = tmp_path / "figures"

    summary = build_fi2010_neural_figures(
        neural_full_grid_dir=grid,
        out_dir=out,
        overwrite=True,
        strict=True,
    )

    manifest = json.loads((out / "figure_manifest.json").read_text(encoding="utf-8"))
    entries = {entry["figure_id"]: entry for entry in manifest["figures"]}
    assert summary.completed_figures
    assert entries["reliability_curve"]["status"] == "completed"
    assert Path(entries["reliability_curve"]["file_path"]).name == "reliability_curve.png"
    assert (out / "source_data" / "reliability_curve.csv").is_file()
    assert entries["cost_adjusted_proxy"]["status"] == "skipped"
    assert entries["cost_adjusted_proxy"]["reason"] == "execution v3 artefacts not available"
    assert entries["regime_breakdown"]["status"] == "skipped"


def test_missing_prediction_artefacts_are_skipped_without_fabrication(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    grid = _write_synthetic_grid(tmp_path, missing_predictions=True)
    out = tmp_path / "figures"

    build_fi2010_neural_figures(
        neural_full_grid_dir=grid,
        out_dir=out,
        overwrite=True,
        strict=False,
    )

    manifest = json.loads((out / "figure_manifest.json").read_text(encoding="utf-8"))
    entries = {entry["figure_id"]: entry for entry in manifest["figures"]}
    assert entries["reliability_curve"]["status"] == "skipped"
    assert entries["ssl_matched_delta"]["status"] == "completed"


def test_smoke_test_gating_and_labelling(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    grid = _write_synthetic_grid(tmp_path, smoke=True)

    with pytest.raises(ValueError, match="smoke-test artefacts"):
        build_fi2010_neural_figures(
            neural_full_grid_dir=grid,
            out_dir=tmp_path / "blocked",
            overwrite=True,
        )

    build_fi2010_neural_figures(
        neural_full_grid_dir=grid,
        out_dir=tmp_path / "figures",
        overwrite=True,
        allow_smoke_test=True,
    )
    manifest = json.loads((tmp_path / "figures" / "figure_manifest.json").read_text())
    assert manifest["smoke_test"] is True
    assert all(entry["smoke_test"] is True for entry in manifest["figures"])


def test_best_model_selection_prefers_lower_ece_on_macro_f1_tie() -> None:
    frame = pd.DataFrame(
        [
            _result_row(
                objective="supervised",
                prediction_file="a.csv",
                macro_f1=0.6,
                ece=0.2,
            ),
            _result_row(
                objective="masked_reconstruction",
                prediction_file="b.csv",
                macro_f1=0.6,
                ece=0.1,
            ),
        ]
    )

    selection = select_best_models_by_horizon(frame)
    assert selection["selected"][0]["objective_label"] == "masked_reconstruction"


def test_matched_ssl_delta_uses_only_matched_pairs() -> None:
    frame = pd.DataFrame(
        [
            _result_row(
                objective="supervised",
                prediction_file="supervised.csv",
                macro_f1=0.5,
                ece=0.1,
                seed=0,
            ),
            _result_row(
                objective="masked_reconstruction",
                prediction_file="masked.csv",
                macro_f1=0.6,
                ece=0.08,
                seed=0,
            ),
            _result_row(
                objective="next_field",
                prediction_file="next.csv",
                macro_f1=0.7,
                ece=0.05,
                seed=1,
            ),
        ]
    )

    rows = build_matched_ssl_delta_rows(frame)
    assert len(rows) == 1
    assert rows[0]["ssl_objective"] == "masked_reconstruction"
    assert rows[0]["delta_macro_f1"] == pytest.approx(0.1)


def test_final_report_includes_figure_index(tmp_path: Path) -> None:
    from chronoslob.experiments.final_report import build_final_empirical_report
    from tests.test_fi2010_ssl_runner import _write_minimal_required_dirs

    dirs = _write_minimal_required_dirs(tmp_path)
    grid = _write_synthetic_grid(tmp_path / "real_grid")
    _write_json(
        grid / "figure_manifest.json",
        {
            "smoke_test": False,
            "figures": [
                {
                    "figure_id": "reliability_curve",
                    "title": "Reliability Curve",
                    "file_path": "reports/figures/fi2010_neural_full_grid/reliability_curve.png",
                    "source_data_path": (
                        "reports/figures/fi2010_neural_full_grid/"
                        "source_data/reliability_curve.csv"
                    ),
                    "status": "completed",
                    "reason": "",
                    "smoke_test": False,
                },
                {
                    "figure_id": "regime_breakdown",
                    "title": "Regime Breakdown",
                    "file_path": None,
                    "source_data_path": None,
                    "status": "skipped",
                    "reason": "regime labels not present in prediction artefacts",
                    "smoke_test": False,
                },
            ],
        },
    )
    report_path = tmp_path / "report.md"

    build_final_empirical_report(
        classical_dir=dirs["classical"],
        neural_dir=dirs["neural"],
        uncertainty_dir=dirs["uncertainty"],
        neural_full_grid_dir=grid,
        out_path=report_path,
        overwrite=True,
    )

    text = report_path.read_text(encoding="utf-8")
    assert "## Figure Index" in text
    assert "reliability_curve" in text
    assert "regime labels not present" in text


def test_strict_mode_fails_when_class_mapping_is_ambiguous(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    grid = _write_synthetic_grid(tmp_path, ambiguous=True)
    out = tmp_path / "figures"

    with pytest.raises(ValueError, match="label mapping audit failed"):
        build_fi2010_neural_figures(
            neural_full_grid_dir=grid,
            out_dir=out,
            overwrite=True,
            strict=True,
        )
    audit = json.loads((out / "label_mapping_audit.json").read_text(encoding="utf-8"))
    assert audit["status"] == "fail"
    assert any("ambiguous" in error for error in audit["errors"])
