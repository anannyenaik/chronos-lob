"""Tests for the FI-2010 SSL failure-analysis module and its integrations.

These tests confirm that the analysis consumes only retained lightweight
artefacts, never requires deleted raw prediction files, blocks broad SSL
improvement claims, labels the narrow fold-1/horizon-50 proper-training finding
as a partial_real / partially_supported result, and never claims a calibration
improvement when ECE worsens.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chronoslob.analysis.ssl_failure_analysis import (
    SSL_FAILURE_ANALYSIS_VERSION,
    analyse_fi2010_ssl_results,
)

_SSL_COMPARISON_COLUMNS = (
    "status",
    "ssl_objective",
    "fold",
    "horizon",
    "seed",
    "lookback",
    "delta_macro_f1",
    "delta_mcc",
    "delta_ece",
    "delta_brier_score",
    "delta_accuracy",
    "delta_nll",
    "macro_f1_outcome",
    "mcc_outcome",
    "ece_outcome",
)


def _outcome(delta: float, *, higher_better: bool) -> str:
    if delta == 0.0:
        return "tie"
    improved = delta > 0.0 if higher_better else delta < 0.0
    return "win" if improved else "loss"


def _row(
    *,
    objective: str,
    fold: int,
    horizon: int,
    seed: int,
    macro: float,
    mcc: float,
    ece: float,
) -> dict[str, Any]:
    return {
        "status": "matched",
        "ssl_objective": objective,
        "fold": fold,
        "horizon": horizon,
        "seed": seed,
        "lookback": 50,
        "delta_macro_f1": macro,
        "delta_mcc": mcc,
        "delta_ece": ece,
        "delta_brier_score": -macro,
        "delta_accuracy": macro,
        "delta_nll": -macro,
        "macro_f1_outcome": _outcome(macro, higher_better=True),
        "mcc_outcome": _outcome(mcc, higher_better=True),
        "ece_outcome": _outcome(ece, higher_better=False),
    }


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _aggregate_rows(
    horizon_metrics: dict[int, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon, objectives in horizon_metrics.items():
        for objective, metrics in objectives.items():
            rows.append(
                {
                    "completed_run_count": 1,
                    "failed_run_count": 0,
                    "horizon": horizon,
                    "lookback": 50,
                    "model_family": "matrix_transformer",
                    "pretraining_objective": objective,
                    "mean_accuracy": metrics["macro_f1"],
                    "mean_macro_f1": metrics["macro_f1"],
                    "mean_mcc": metrics["mcc"],
                    "mean_ece": metrics["ece"],
                    "mean_brier_score": 0.6,
                    "mean_nll": 1.0,
                }
            )
    return rows


def _write_full_grid(directory: Path) -> None:
    # Mixed matched deltas across folds/horizons/seeds: masked slightly negative,
    # next-field clearly negative, ECE not uniformly improved.
    rows: list[dict[str, Any]] = []
    for fold in (1, 2):
        for seed in (0, 1):
            rows.append(
                _row(
                    objective="masked_reconstruction",
                    fold=fold,
                    horizon=10,
                    seed=seed,
                    macro=-0.01,
                    mcc=-0.02,
                    ece=0.02,
                )
            )
            rows.append(
                _row(
                    objective="masked_reconstruction",
                    fold=fold,
                    horizon=50,
                    seed=seed,
                    macro=0.01,
                    mcc=0.01,
                    ece=-0.01,
                )
            )
            rows.append(
                _row(
                    objective="next_field",
                    fold=fold,
                    horizon=10,
                    seed=seed,
                    macro=-0.06,
                    mcc=-0.06,
                    ece=-0.01,
                )
            )
            rows.append(
                _row(
                    objective="next_field",
                    fold=fold,
                    horizon=50,
                    seed=seed,
                    macro=-0.04,
                    mcc=-0.05,
                    ece=0.03,
                )
            )
    _write_csv(directory / "ssl_comparison.csv", _SSL_COMPARISON_COLUMNS, rows)
    _write_csv(directory / "results_summary.csv", ("status",), [{"status": "completed"}])
    aggregate = _aggregate_rows(
        {
            10: {"none": {"macro_f1": 0.33, "mcc": 0.03, "ece": 0.11}},
            50: {"none": {"macro_f1": 0.41, "mcc": 0.16, "ece": 0.07}},
        }
    )
    _write_csv(
        directory / "aggregate_summary.csv",
        ("horizon", "pretraining_objective", "mean_macro_f1"),
        [
            {
                "horizon": row["horizon"],
                "pretraining_objective": row["pretraining_objective"],
                "mean_macro_f1": row["mean_macro_f1"],
            }
            for row in aggregate
        ],
    )
    _write_json(directory / "aggregate_summary.json", {"aggregate": aggregate})
    _write_csv(directory / "failures.csv", ("run_id", "reason"), [])
    _write_json(
        directory / "summary.json",
        {
            "created_at": "2026-05-29T00:00:00Z",
            "smoke_test": False,
            "completed_run_count": len(rows),
            "failed_run_count": 0,
            "core_grid_complete": True,
            "evidence_level": "full_grid_complete",
        },
    )


def _write_proper_training(directory: Path) -> None:
    # Horizon 10 ties on predictive metrics but worsens ECE; horizon 50 masked
    # improves macro-F1 and MCC while ECE still worsens.
    rows = [
        _row(
            objective="masked_reconstruction",
            fold=1,
            horizon=10,
            seed=0,
            macro=0.0,
            mcc=0.0,
            ece=0.0175,
        ),
        _row(
            objective="next_field",
            fold=1,
            horizon=10,
            seed=0,
            macro=0.0,
            mcc=0.0,
            ece=0.0746,
        ),
        _row(
            objective="masked_reconstruction",
            fold=1,
            horizon=50,
            seed=0,
            macro=0.0891,
            mcc=0.1238,
            ece=0.0245,
        ),
        _row(
            objective="next_field",
            fold=1,
            horizon=50,
            seed=0,
            macro=0.0065,
            mcc=0.0408,
            ece=0.0317,
        ),
    ]
    _write_csv(directory / "ssl_comparison.csv", _SSL_COMPARISON_COLUMNS, rows)
    _write_csv(directory / "results_summary.csv", ("status",), [{"status": "completed"}])
    aggregate = _aggregate_rows(
        {
            10: {"none": {"macro_f1": 0.2477, "mcc": 0.0, "ece": 0.0872}},
            50: {"none": {"macro_f1": 0.3883, "mcc": 0.0917, "ece": 0.0496}},
        }
    )
    _write_csv(
        directory / "aggregate_summary.csv",
        ("horizon", "pretraining_objective", "mean_macro_f1"),
        [
            {
                "horizon": row["horizon"],
                "pretraining_objective": row["pretraining_objective"],
                "mean_macro_f1": row["mean_macro_f1"],
            }
            for row in aggregate
        ],
    )
    _write_json(directory / "aggregate_summary.json", {"aggregate": aggregate})
    _write_csv(
        directory / "training_curves_summary.csv",
        ("run_id", "objective", "horizon", "epochs_ran", "best_epoch", "early_stopped",
         "best_validation_score", "test_macro_f1", "test_ece"),
        [
            {
                "run_id": "fold_1__h50__seed_0__masked_reconstruction",
                "objective": "masked_reconstruction",
                "horizon": 50,
                "epochs_ran": 25,
                "best_epoch": 25,
                "early_stopped": False,
                "best_validation_score": 0.4842,
                "test_macro_f1": 0.4774,
                "test_ece": 0.0741,
            }
        ],
    )
    _write_json(directory / "config_snapshot.json", {"folds": [1]})
    _write_csv(directory / "failures.csv", ("run_id", "reason"), [])
    _write_json(directory / "sha256_manifest.json", {"files": {}})
    _write_json(
        directory / "summary.json",
        {
            "created_at": "2026-05-29T23:00:00Z",
            "smoke_test": False,
            "completed_run_count": len(rows),
            "failed_run_count": 0,
            "evidence_level": "partial_real",
        },
    )


@pytest.fixture
def tiny_artefacts(tmp_path: Path) -> dict[str, Path]:
    full_grid = tmp_path / "grid"
    proper = tmp_path / "proper"
    _write_full_grid(full_grid)
    _write_proper_training(proper)
    return {"full_grid": full_grid, "proper_training": proper, "root": tmp_path}


def test_analysis_writes_all_required_outputs(tiny_artefacts: dict[str, Path]) -> None:
    out = tiny_artefacts["root"] / "ssl_analysis"
    summary = analyse_fi2010_ssl_results(
        full_grid_dir=tiny_artefacts["full_grid"],
        proper_training_dir=tiny_artefacts["proper_training"],
        out_dir=out,
        make_figures=False,
        overwrite=True,
    )

    for name in (
        "ssl_failure_analysis.md",
        "ssl_delta_by_horizon.csv",
        "ssl_delta_by_fold.csv",
        "ssl_delta_by_seed.csv",
        "ssl_delta_by_objective.csv",
        "ssl_metric_summary.csv",
        "ssl_claim_assessment.json",
        "figure_manifest.json",
        "summary.json",
    ):
        assert (out / name).is_file(), name
    assert summary.full_grid_matched_rows == 16
    assert summary.proper_training_matched_rows == 4
    assert SSL_FAILURE_ANALYSIS_VERSION in (out / "ssl_failure_analysis.md").read_text(
        encoding="utf-8"
    )


def test_analysis_does_not_require_raw_predictions(tiny_artefacts: dict[str, Path]) -> None:
    # Deliberately no runs/, predictions/ or *.pt files exist in the inputs.
    for directory in (tiny_artefacts["full_grid"], tiny_artefacts["proper_training"]):
        assert not (directory / "runs").exists()
        assert list(directory.glob("*.pt")) == []

    summary = analyse_fi2010_ssl_results(
        full_grid_dir=tiny_artefacts["full_grid"],
        proper_training_dir=tiny_artefacts["proper_training"],
        out_dir=tiny_artefacts["root"] / "ssl_analysis",
        make_figures=False,
        overwrite=True,
    )

    assert summary.raw_predictions_required is False
    assert summary.checkpoints_required is False


def test_broad_ssl_improvement_is_blocked(tiny_artefacts: dict[str, Path]) -> None:
    out = tiny_artefacts["root"] / "ssl_analysis"
    summary = analyse_fi2010_ssl_results(
        full_grid_dir=tiny_artefacts["full_grid"],
        proper_training_dir=tiny_artefacts["proper_training"],
        out_dir=out,
        make_figures=False,
        overwrite=True,
    )

    assert summary.claim_statuses["broad_ssl_improvement"] == "unsupported"
    payload = json.loads((out / "ssl_claim_assessment.json").read_text(encoding="utf-8"))
    broad = next(c for c in payload["claims"] if c["claim_id"] == "broad_ssl_improvement")
    assert broad["status"] == "unsupported"


def test_calibration_improvement_not_claimed_when_ece_worsens(
    tiny_artefacts: dict[str, Path],
) -> None:
    summary = analyse_fi2010_ssl_results(
        full_grid_dir=tiny_artefacts["full_grid"],
        proper_training_dir=tiny_artefacts["proper_training"],
        out_dir=tiny_artefacts["root"] / "ssl_analysis",
        make_figures=False,
        overwrite=True,
    )

    assert summary.claim_statuses["ssl_calibration_improvement"] == "unsupported"


def test_narrow_h50_finding_is_partial_real_and_partially_supported(
    tiny_artefacts: dict[str, Path],
) -> None:
    out = tiny_artefacts["root"] / "ssl_analysis"
    summary = analyse_fi2010_ssl_results(
        full_grid_dir=tiny_artefacts["full_grid"],
        proper_training_dir=tiny_artefacts["proper_training"],
        out_dir=out,
        make_figures=False,
        overwrite=True,
    )

    assert (
        summary.claim_statuses["proper_training_h50_predictive_improvement"]
        == "partially_supported"
    )
    payload = json.loads((out / "ssl_claim_assessment.json").read_text(encoding="utf-8"))
    claim = next(
        c
        for c in payload["claims"]
        if c["claim_id"] == "proper_training_h50_predictive_improvement"
    )
    assert "partial_real" in claim["scope"]
    assert "horizon 50" in claim["scope"]
    assert claim["calibration_worsened"] is True


def test_ssl_implementation_claim_supported(tiny_artefacts: dict[str, Path]) -> None:
    summary = analyse_fi2010_ssl_results(
        full_grid_dir=tiny_artefacts["full_grid"],
        proper_training_dir=tiny_artefacts["proper_training"],
        out_dir=tiny_artefacts["root"] / "ssl_analysis",
        make_figures=False,
        overwrite=True,
    )

    assert summary.claim_statuses["ssl_implemented_and_evaluated"] == "supported"


def test_report_contains_required_conclusion_and_no_forbidden_claims(
    tiny_artefacts: dict[str, Path],
) -> None:
    out = tiny_artefacts["root"] / "ssl_analysis"
    analyse_fi2010_ssl_results(
        full_grid_dir=tiny_artefacts["full_grid"],
        proper_training_dir=tiny_artefacts["proper_training"],
        out_dir=out,
        make_figures=False,
        overwrite=True,
    )
    text = (out / "ssl_failure_analysis.md").read_text(encoding="utf-8").lower()

    assert "fold-1/horizon-50" in text
    assert "calibration worsened" in text
    for forbidden in ("sota", "state-of-the-art", "foundation model", "pnl", "tradable alpha"):
        assert forbidden not in text
    # Every prose line stays well within the public 220-character threshold.
    for line in (out / "ssl_failure_analysis.md").read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            assert len(line) <= 220


def test_missing_required_file_raises(tiny_artefacts: dict[str, Path]) -> None:
    (tiny_artefacts["full_grid"] / "ssl_comparison.csv").unlink()
    with pytest.raises(FileNotFoundError, match="full-grid artefacts incomplete"):
        analyse_fi2010_ssl_results(
            full_grid_dir=tiny_artefacts["full_grid"],
            proper_training_dir=tiny_artefacts["proper_training"],
            out_dir=tiny_artefacts["root"] / "ssl_analysis",
            make_figures=False,
            overwrite=True,
        )


def test_requires_at_least_one_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        analyse_fi2010_ssl_results(
            full_grid_dir=None,
            proper_training_dir=None,
            out_dir=tmp_path / "out",
            make_figures=False,
            overwrite=True,
        )


def test_final_report_section_distinguishes_sources_and_states_conclusions() -> None:
    from chronoslob.experiments.final_report import (
        _SECTION_TITLES,
        _render_ssl_failure_analysis,
    )

    assert "SSL Failure Analysis" in _SECTION_TITLES

    data = SimpleNamespace(
        full_grid=SimpleNamespace(
            comparison_rows=[
                {
                    "status": "matched",
                    "ssl_objective": "masked_reconstruction",
                    "horizon": "10",
                    "delta_macro_f1": "-0.01",
                    "delta_mcc": "-0.02",
                    "delta_ece": "0.02",
                }
            ]
        ),
        proper_training=SimpleNamespace(
            comparison_rows=[
                {
                    "status": "matched",
                    "ssl_objective": "masked_reconstruction",
                    "horizon": "50",
                    "delta_macro_f1": "0.0891",
                    "delta_mcc": "0.1238",
                    "delta_ece": "0.0245",
                }
            ]
        ),
    )
    text = "\n".join(_render_ssl_failure_analysis(data))

    assert "ssl_failure_analysis.md" in text
    assert "Full-grid SSL does not improve overall" in text
    assert "fold-1/horizon-50" in text
    assert (
        "No broad SSL improvement or broad calibration improvement is claimed "
        "from the SSL-v1 and matched full-grid evidence"
    ) in text


def test_evidence_pack_records_ssl_analysis_and_claims(tiny_artefacts: dict[str, Path]) -> None:
    from chronoslob.experiments.evidence_pack import (
        EvidencePackConfig,
        audit_claims,
        discover_artefacts,
    )

    out = tiny_artefacts["root"] / "ssl_analysis"
    analyse_fi2010_ssl_results(
        full_grid_dir=tiny_artefacts["full_grid"],
        proper_training_dir=tiny_artefacts["proper_training"],
        out_dir=out,
        make_figures=False,
        overwrite=True,
    )

    root = tiny_artefacts["root"]
    config = EvidencePackConfig(
        out_dir=root / "pack",
        classical_dir=root / "missing_classical",
        ssl_dir=root / "missing_ssl",
        proper_training_dir=tiny_artefacts["proper_training"],
        ssl_analysis_dir=out,
        neural_full_grid_dir=tiny_artefacts["full_grid"],
        figures_dir=root / "missing_figures",
        execution_v3_dir=root / "missing_execution",
        feature_audit_dir=None,
        feature_ablations_dir=root / "missing_feature_ablations",
        ablation_figures_dir=root / "missing_ablation_figures",
        final_report_path=root / "missing_report.md",
        project_audit_dir=None,
        strict=False,
        allow_smoke_test=True,
        overwrite=True,
    )

    records = discover_artefacts(config)
    report_record = next(
        record for record in records if record.artefact_name == "ssl_failure_analysis_report"
    )
    assert report_record.status not in {"missing", "invalid", "unsupported"}

    by_id = {claim.claim_id: claim for claim in audit_claims(records)}
    assert by_id["empirical.ssl_implemented_evaluated"].status == "supported"
    assert by_id["empirical.ssl_proper_training_h50"].status == "partially_supported"
    assert "partial_real" in by_id["empirical.ssl_proper_training_h50"].safe_rewording
    assert by_id["empirical.ssl_improved_macro_f1"].status == "unsupported"
    assert by_id["empirical.ssl_improved_calibration"].status == "unsupported"
