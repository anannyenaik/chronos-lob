"""Tests for the FI-2010 statistical uncertainty layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from chronoslob.cli import _analyse_fi2010_uncertainty_impl
from chronoslob.experiments.statistics import (
    DEFAULT_UNCERTAINTY_METRICS,
    analyse_fi2010_uncertainty,
    bootstrap_mean_confidence_interval,
    compute_metric_confidence_intervals,
    compute_paired_model_comparisons,
    compute_rank_stability,
    load_classical_fold_results,
    load_neural_fold_results,
)
from chronoslob.utils.audit import (
    AuditStatus,
    check_no_forbidden_claims,
    check_public_release_wording,
)
from chronoslob.utils.paths import project_root


def _write_classical_fixture(path: Path) -> Path:
    rows = [
        # baseline gradient_boosting beats logistic on every fold
        {
            "fold_id": fold,
            "model_name": model,
            "split": split,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "mcc": mcc,
            "brier_score": brier,
            "ece": ece,
            "n_train": 100,
            "n_validation": 20,
            "n_test": 40,
            "seed": 0,
            "status": "ok",
            "error": "",
        }
        for (
            fold,
            model,
            split,
            accuracy,
            macro_f1,
            mcc,
            brier,
            ece,
        ) in (
            (1, "logistic", "test", 0.55, 0.32, 0.10, 0.55, 0.05),
            (1, "gradient_boosting", "test", 0.60, 0.40, 0.20, 0.50, 0.04),
            (1, "ridge", "test", 0.56, 0.33, 0.11, float("nan"), float("nan")),
            (2, "logistic", "test", 0.58, 0.34, 0.12, 0.53, 0.06),
            (2, "gradient_boosting", "test", 0.62, 0.42, 0.22, 0.49, 0.04),
            (2, "ridge", "test", 0.57, 0.34, 0.12, float("nan"), float("nan")),
            (3, "logistic", "test", 0.57, 0.33, 0.11, 0.54, 0.05),
            (3, "gradient_boosting", "test", 0.61, 0.41, 0.21, 0.50, 0.04),
            (3, "ridge", "test", 0.57, 0.33, 0.11, float("nan"), float("nan")),
        )
    ]
    frame = pd.DataFrame(rows)
    out_path = path / "results_by_fold.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return out_path


def _write_neural_fixture(path: Path, *, seeds: tuple[int, ...] = (0,)) -> Path:
    rows: list[dict[str, Any]] = []
    base_values = {
        ("fold_1", "deeplob_style"): (0.50, 0.48, 0.30, 0.65, 0.10),
        ("fold_1", "matrix_transformer"): (0.76, 0.68, 0.56, 0.34, 0.05),
        ("fold_2", "deeplob_style"): (0.45, 0.44, 0.27, 0.94, 0.41),
        ("fold_2", "matrix_transformer"): (0.82, 0.74, 0.64, 0.24, 0.02),
        ("fold_3", "deeplob_style"): (0.46, 0.45, 0.26, 0.75, 0.28),
        ("fold_3", "matrix_transformer"): (0.80, 0.73, 0.63, 0.27, 0.03),
    }
    for (fold, model), (accuracy, macro_f1, mcc, brier, ece) in base_values.items():
        for seed in seeds:
            jitter = 0.005 * seed
            rows.append(
                {
                    "fold_id": fold,
                    "seed": seed,
                    "model_name": model,
                    "lookback": 20,
                    "split": "test",
                    "accuracy": accuracy + jitter,
                    "macro_f1": macro_f1 + jitter,
                    "mcc": mcc + jitter,
                    "brier_score": brier - jitter,
                    "ece": ece - jitter,
                    "n_train": 200,
                    "n_validation": 40,
                    "n_test": 80,
                    "status": "ok",
                    "error": "",
                }
            )
    frame = pd.DataFrame(rows)
    out_path = path / "results_by_fold_seed.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Loader and statistic-level tests
# ---------------------------------------------------------------------------


def test_load_classical_handles_classical_only_artefacts(tmp_path: Path) -> None:
    classical_dir = tmp_path / "classical"
    _write_classical_fixture(classical_dir)

    frame = load_classical_fold_results(classical_dir / "results_by_fold.csv")

    assert set(frame["model_name"].unique()) == {
        "logistic",
        "gradient_boosting",
        "ridge",
    }
    assert (frame["fold_id"].astype(str).str.startswith("fold_")).all()
    assert "lookback" in frame.columns


def test_load_neural_preserves_seed_and_lookback_columns(tmp_path: Path) -> None:
    neural_dir = tmp_path / "neural"
    _write_neural_fixture(neural_dir, seeds=(0, 1))

    frame = load_neural_fold_results(neural_dir / "results_by_fold_seed.csv")

    assert set(frame["seed"].unique()) == {0, 1}
    assert {int(value) for value in frame["lookback"].dropna()} == {20}
    assert set(frame["fold_id"].unique()) == {"fold_1", "fold_2", "fold_3"}


def test_bootstrap_confidence_interval_is_deterministic() -> None:
    values = [0.40, 0.41, 0.39, 0.42, 0.40]

    lower_a, upper_a = bootstrap_mean_confidence_interval(
        values, iterations=200, ci_level=0.95, seed=42
    )
    lower_b, upper_b = bootstrap_mean_confidence_interval(
        values, iterations=200, ci_level=0.95, seed=42
    )

    assert lower_a == lower_b
    assert upper_a == upper_b
    assert lower_a < sum(values) / len(values) < upper_a


def test_confidence_interval_handles_missing_brier_and_ece(tmp_path: Path) -> None:
    classical_dir = tmp_path / "classical"
    _write_classical_fixture(classical_dir)
    frame = load_classical_fold_results(classical_dir / "results_by_fold.csv")

    intervals = compute_metric_confidence_intervals(
        frame,
        source="classical",
        metrics=DEFAULT_UNCERTAINTY_METRICS,
        ci_level=0.95,
        bootstrap_iterations=200,
        bootstrap_seed=0,
    )

    ridge_brier = intervals.loc[
        (intervals["model_name"] == "ridge")
        & (intervals["metric"] == "brier_score")
    ]
    assert len(ridge_brier) == 1
    assert int(ridge_brier.iloc[0]["n_folds"]) == 0
    assert int(ridge_brier.iloc[0]["n_missing"]) == 3
    assert pd.isna(ridge_brier.iloc[0]["mean"])

    gb_macro = intervals.loc[
        (intervals["model_name"] == "gradient_boosting")
        & (intervals["metric"] == "macro_f1")
    ].iloc[0]
    assert int(gb_macro["n_folds"]) == 3
    assert gb_macro["ci_lower"] < gb_macro["mean"] < gb_macro["ci_upper"]


def test_paired_comparison_records_wins_and_losses(tmp_path: Path) -> None:
    classical_dir = tmp_path / "classical"
    _write_classical_fixture(classical_dir)
    frame = load_classical_fold_results(classical_dir / "results_by_fold.csv")

    paired = compute_paired_model_comparisons(
        frame,
        source="classical",
        baseline_model="gradient_boosting",
        metrics=("macro_f1",),
        ci_level=0.95,
    )

    logistic_row = paired.loc[paired["candidate_model"] == "logistic"].iloc[0]
    assert int(logistic_row["wins"]) == 0
    assert int(logistic_row["losses"]) == 3
    assert logistic_row["mean_difference"] < 0


def test_rank_stability_marks_gradient_boosting_best(tmp_path: Path) -> None:
    classical_dir = tmp_path / "classical"
    _write_classical_fixture(classical_dir)
    frame = load_classical_fold_results(classical_dir / "results_by_fold.csv")

    ranks = compute_rank_stability(
        frame,
        source="classical",
        metrics=("macro_f1",),
    )

    gb_rank = ranks.loc[ranks["model_name"] == "gradient_boosting"].iloc[0]
    assert gb_rank["best_fraction"] == pytest.approx(1.0)
    assert int(gb_rank["best_count"]) == 3


# ---------------------------------------------------------------------------
# Module entry point and CLI
# ---------------------------------------------------------------------------


def test_analyse_writes_all_expected_artefacts(tmp_path: Path) -> None:
    classical_dir = tmp_path / "classical"
    neural_dir = tmp_path / "neural"
    _write_classical_fixture(classical_dir)
    _write_neural_fixture(neural_dir, seeds=(0,))
    out_dir = tmp_path / "uncertainty"

    summary = analyse_fi2010_uncertainty(
        classical_dir=classical_dir,
        neural_dir=neural_dir,
        out_dir=out_dir,
        baseline_model="gradient_boosting",
        bootstrap_iterations=200,
        bootstrap_seed=0,
        overwrite=False,
    )

    for name in (
        "summary.json",
        "metric_confidence_intervals.csv",
        "paired_model_comparisons.csv",
        "rank_stability.csv",
        "model_ranking.csv",
        "uncertainty_notes.md",
    ):
        assert (out_dir / name).is_file()
    payload = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["parameters"]["baseline_model"] == "gradient_boosting"
    assert summary.classical_seed_variance_available is False
    assert summary.neural_seed_variance_available is False


def test_analyse_reports_neural_seed_variance_when_present(tmp_path: Path) -> None:
    classical_dir = tmp_path / "classical"
    neural_dir = tmp_path / "neural"
    _write_classical_fixture(classical_dir)
    _write_neural_fixture(neural_dir, seeds=(0, 1))
    out_dir = tmp_path / "uncertainty"

    summary = analyse_fi2010_uncertainty(
        classical_dir=classical_dir,
        neural_dir=neural_dir,
        out_dir=out_dir,
        baseline_model="gradient_boosting",
        bootstrap_iterations=100,
        bootstrap_seed=0,
        overwrite=True,
    )

    assert summary.neural_seed_variance_available is True
    notes = (out_dir / "uncertainty_notes.md").read_text(encoding="utf-8")
    assert "Neural seed variance is available" in notes


def test_overwrite_protection_raises_on_existing_non_empty_dir(tmp_path: Path) -> None:
    classical_dir = tmp_path / "classical"
    _write_classical_fixture(classical_dir)
    out_dir = tmp_path / "uncertainty"
    out_dir.mkdir()
    (out_dir / "existing.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        analyse_fi2010_uncertainty(
            classical_dir=classical_dir,
            neural_dir=None,
            out_dir=out_dir,
            overwrite=False,
        )


def test_overwrite_true_replaces_existing_output(tmp_path: Path) -> None:
    classical_dir = tmp_path / "classical"
    _write_classical_fixture(classical_dir)
    out_dir = tmp_path / "uncertainty"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("stale", encoding="utf-8")

    analyse_fi2010_uncertainty(
        classical_dir=classical_dir,
        neural_dir=None,
        out_dir=out_dir,
        overwrite=True,
    )

    assert (out_dir / "summary.json").is_file()
    assert not (out_dir / "stale.txt").exists()


def test_cli_impl_runs_on_tiny_fake_artefacts(tmp_path: Path) -> None:
    classical_dir = tmp_path / "classical"
    neural_dir = tmp_path / "neural"
    _write_classical_fixture(classical_dir)
    _write_neural_fixture(neural_dir, seeds=(0,))
    out_dir = tmp_path / "uncertainty"

    exit_code = _analyse_fi2010_uncertainty_impl(
        classical_dir=classical_dir,
        neural_dir=neural_dir,
        out=out_dir,
        baseline_model="gradient_boosting",
        ci_level=0.95,
        bootstrap_iterations=200,
        bootstrap_seed=0,
        overwrite=True,
    )

    assert exit_code == 0
    assert (out_dir / "model_ranking.csv").is_file()


def test_cli_impl_reports_missing_directories_cleanly(tmp_path: Path) -> None:
    exit_code = _analyse_fi2010_uncertainty_impl(
        classical_dir=tmp_path / "missing_classical",
        neural_dir=tmp_path / "missing_neural",
        out=tmp_path / "uncertainty",
        baseline_model="gradient_boosting",
        ci_level=0.95,
        bootstrap_iterations=100,
        bootstrap_seed=0,
        overwrite=True,
    )

    assert exit_code == 2


# ---------------------------------------------------------------------------
# Docs and audit guardrails
# ---------------------------------------------------------------------------


def test_uncertainty_doc_avoids_forbidden_public_claims() -> None:
    doc_path = project_root() / "docs" / "STATISTICAL_UNCERTAINTY.md"
    result = check_no_forbidden_claims(scan_paths=[doc_path])

    assert result.status == AuditStatus.PASS, result.issues


def test_uncertainty_notes_artefact_avoids_forbidden_public_claims() -> None:
    notes_path = (
        project_root() / "experiments" / "fi2010_uncertainty" / "uncertainty_notes.md"
    )
    if not notes_path.is_file():
        pytest.skip("uncertainty_notes.md not generated in this checkout")
    result = check_no_forbidden_claims(scan_paths=[notes_path])

    assert result.status == AuditStatus.PASS, result.issues


def test_uncertainty_doc_passes_public_release_wording() -> None:
    doc_path = project_root() / "docs" / "STATISTICAL_UNCERTAINTY.md"
    result = check_public_release_wording(scan_paths=[doc_path])

    assert result.status == AuditStatus.PASS, result.issues
