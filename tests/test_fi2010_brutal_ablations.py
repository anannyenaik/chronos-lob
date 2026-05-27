"""Tests for the FI-2010 brutal ablation layer.

The tests use only tiny synthetic fixtures. They never read the real
FI-2010 data and never train a neural model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronoslob.cli import _run_fi2010_brutal_ablations_impl
from chronoslob.experiments.fi2010_brutal_ablations import (
    normalise_families,
    resolve_feature_groups,
    run_fi2010_brutal_ablations,
)
from chronoslob.utils.paths import project_root

_CONFIG_PATH = project_root() / "configs" / "experiments" / "fi2010_multifold.yaml"

_FEATURE_COLUMNS = (
    "bid_price_1",
    "bid_quantity_1",
    "ask_price_1",
    "ask_quantity_1",
    "bid_price_2",
    "ask_quantity_2",
    "f_001",
    "f_002",
)


def _write_fold_csv(path: Path, *, include_alt_horizons: bool = False) -> None:
    """Write a tiny split-aware synthetic fold CSV."""
    rng = np.random.default_rng(0)
    n_train = 48
    n_test = 24
    n_rows = n_train + n_test
    columns: dict[str, object] = {
        "split": ["train"] * n_train + ["test"] * n_test,
    }
    for name in _FEATURE_COLUMNS:
        columns[name] = rng.normal(size=n_rows)
    # Three balanced classes following the FI-2010 label convention.
    columns["label_10"] = [(index % 3) + 1 for index in range(n_rows)]
    if include_alt_horizons:
        columns["label_50"] = [((index + 1) % 3) + 1 for index in range(n_rows)]
        columns["label_100"] = [((index + 2) % 3) + 1 for index in range(n_rows)]
    pd.DataFrame(columns).to_csv(path, index=False)


def _write_classical_dir(classical_dir: Path) -> None:
    classical_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(
        [
            {
                "fold_id": fold,
                "model_name": model,
                "split": "test",
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "mcc": mcc,
                "ece": ece,
                "status": "ok",
            }
            for fold in (1, 2)
            for model, accuracy, macro_f1, mcc, ece in (
                ("gradient_boosting", 0.66, 0.46, 0.27, 0.02),
                ("logistic", 0.61, 0.33, 0.12, 0.03),
                ("majority", 0.60, 0.25, 0.0, 0.04),
            )
        ]
    )
    results.to_csv(classical_dir / "results_by_fold.csv", index=False)

    execution_rows: list[dict[str, object]] = []
    for model in ("gradient_boosting", "logistic"):
        for threshold in (0.0, 0.6):
            for cost in (0.0, 5.0):
                for latency in (0, 1):
                    execution_rows.append(
                        {
                            "model_name": model,
                            "split": "test",
                            "confidence_threshold": threshold,
                            "cost_bps": cost,
                            "latency_steps": latency,
                            "eligible_predictions": 100 if threshold == 0.0 else 60,
                            "hit_rate_proxy": 0.55 if threshold == 0.0 else 0.62,
                            "turnover_proxy": 100.0,
                            "gross_signal_return_proxy": 12.0,
                            "net_signal_return_proxy": 12.0 - cost - 2.0 * latency,
                        }
                    )
    pd.DataFrame(execution_rows).to_csv(
        classical_dir / "execution_summary.csv", index=False
    )


def test_resolve_feature_groups_partitions_columns() -> None:
    groups = resolve_feature_groups(_FEATURE_COLUMNS)

    assert groups["top_of_book_only"] == [
        "bid_price_1",
        "bid_quantity_1",
        "ask_price_1",
        "ask_quantity_1",
    ]
    assert groups["depth_features"] == ["bid_price_2", "ask_quantity_2"]
    assert set(groups["price_only"]) == {"bid_price_1", "ask_price_1", "bid_price_2"}
    assert groups["labels_excluded"] == list(_FEATURE_COLUMNS)


def test_normalise_families_rejects_unknown() -> None:
    assert normalise_families("feature_groups,model_class") == (
        "feature_groups",
        "model_class",
    )
    with pytest.raises(ValueError):
        normalise_families("not_a_family")


def test_feature_group_family_writes_results(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    _write_fold_csv(processed / "fold1_combined.csv")
    out_dir = tmp_path / "out"

    summary = run_fi2010_brutal_ablations(
        config_path=_CONFIG_PATH,
        processed_root=processed,
        out_dir=out_dir,
        families="feature_groups",
        folds="fold_1",
        overwrite=True,
    )

    assert "feature_groups" in summary.families_run
    frame = pd.read_csv(out_dir / "feature_group_ablation.csv")
    assert not frame.empty
    names = set(frame["ablation_name"])
    assert {"all_features", "top_of_book_only", "labels_excluded"} <= names
    # The leakage-control group equals all_features, so its delta is zero.
    labels_excluded = frame[
        (frame["ablation_name"] == "labels_excluded")
        & (frame["metric_name"] == "macro_f1")
    ]
    assert labels_excluded["delta_vs_baseline"].abs().max() == pytest.approx(0.0)


def test_unsupported_horizon_ablation_is_recorded(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    # Only label_10 is present, so label_50 and label_100 must be skipped.
    _write_fold_csv(processed / "fold1_combined.csv", include_alt_horizons=False)
    out_dir = tmp_path / "out"

    run_fi2010_brutal_ablations(
        config_path=_CONFIG_PATH,
        processed_root=processed,
        out_dir=out_dir,
        families="horizon",
        folds="fold_1",
        overwrite=True,
    )

    skipped = json.loads((out_dir / "skipped_ablations.json").read_text("utf-8"))
    reasons = {
        entry["ablation_name"]: entry["skip_reason"] for entry in skipped["skipped"]
    }
    assert "horizon_50" in reasons
    assert "label_50" in reasons["horizon_50"]
    horizon_frame = pd.read_csv(out_dir / "horizon_ablation.csv")
    skipped_rows = horizon_frame[horizon_frame["status"] == "skipped"]
    assert not skipped_rows.empty


def test_execution_and_calibration_aggregate_stored_summaries(tmp_path: Path) -> None:
    classical_dir = tmp_path / "classical"
    _write_classical_dir(classical_dir)
    out_dir = tmp_path / "out"

    summary = run_fi2010_brutal_ablations(
        config_path=_CONFIG_PATH,
        classical_dir=classical_dir,
        out_dir=out_dir,
        families="model_class,calibration,execution",
        overwrite=True,
    )

    assert {"model_class", "calibration", "execution"} <= set(summary.families_run)

    model_class = pd.read_csv(out_dir / "model_class_ablation.csv")
    gb = model_class[
        (model_class["model_name"] == "gradient_boosting")
        & (model_class["metric_name"] == "macro_f1")
    ]
    assert gb["delta_vs_baseline"].abs().max() == pytest.approx(0.0)

    execution = pd.read_csv(out_dir / "execution_cost_latency_ablation.csv")
    assert not execution.empty
    higher_cost = execution[
        (execution["metric_name"] == "net_signal_return_proxy")
        & (execution["cost_bps"] == 5.0)
        & (execution["latency_steps"] == 0)
    ]
    assert (higher_cost["delta_vs_baseline"] < 0).all()

    calibration = pd.read_csv(out_dir / "calibration_threshold_ablation.csv")
    assert "reliability_ece" in set(calibration["ablation_name"])


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    _write_fold_csv(processed / "fold1_combined.csv")
    out_dir = tmp_path / "out"

    summary = run_fi2010_brutal_ablations(
        config_path=_CONFIG_PATH,
        processed_root=processed,
        out_dir=out_dir,
        families="feature_groups",
        folds="fold_1",
        dry_run=True,
    )

    assert summary.dry_run is True
    assert not out_dir.exists()


def test_overwrite_protection(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    _write_fold_csv(processed / "fold1_combined.csv")
    out_dir = tmp_path / "out"

    run_fi2010_brutal_ablations(
        config_path=_CONFIG_PATH,
        processed_root=processed,
        out_dir=out_dir,
        families="feature_groups",
        folds="fold_1",
        overwrite=True,
    )

    with pytest.raises(FileExistsError):
        run_fi2010_brutal_ablations(
            config_path=_CONFIG_PATH,
            processed_root=processed,
            out_dir=out_dir,
            families="feature_groups",
            folds="fold_1",
            overwrite=False,
        )


def test_no_full_predictions_or_checkpoints_written(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    _write_fold_csv(processed / "fold1_combined.csv")
    classical_dir = tmp_path / "classical"
    _write_classical_dir(classical_dir)
    out_dir = tmp_path / "out"

    summary = run_fi2010_brutal_ablations(
        config_path=_CONFIG_PATH,
        processed_root=processed,
        classical_dir=classical_dir,
        out_dir=out_dir,
        folds="fold_1",
        overwrite=True,
    )

    assert summary.full_predictions_written is False
    assert summary.checkpoints_written is False
    assert not list(out_dir.rglob("predictions*.csv"))
    assert not list(out_dir.rglob("*.pt"))
    assert not (out_dir / "_neural_lookback_runs").exists()


def test_cli_impl_runs_on_tiny_folds(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    _write_fold_csv(processed / "fold1_combined.csv")
    classical_dir = tmp_path / "classical"
    _write_classical_dir(classical_dir)
    out_dir = tmp_path / "out"

    exit_code = _run_fi2010_brutal_ablations_impl(
        config_path=_CONFIG_PATH,
        neural_config_path=None,
        processed_root=processed,
        classical_dir=classical_dir,
        neural_dir=None,
        out=out_dir,
        families="feature_groups,model_class,calibration,execution,lookback",
        folds="fold_1",
        models=None,
        neural_lookbacks=None,
        max_epochs=5,
        overwrite=True,
        dry_run=False,
    )

    assert exit_code == 0
    for filename in (
        "summary.json",
        "ablation_results.csv",
        "ablation_summary.csv",
        "skipped_ablations.json",
        "feature_group_ablation.csv",
        "ablation_notes.md",
    ):
        assert (out_dir / filename).is_file()
    # The neural lookback sweep is skipped by default and recorded.
    skipped = json.loads((out_dir / "skipped_ablations.json").read_text("utf-8"))
    families = {entry["ablation_family"] for entry in skipped["skipped"]}
    assert "lookback" in families


def test_docs_avoid_forbidden_public_claims() -> None:
    # Reuse the canonical audit scanners rather than embedding forbidden
    # literals here, so this test file itself stays release-clean.
    from chronoslob.utils.audit import (
        check_no_forbidden_claims,
        check_public_release_wording,
    )

    root = project_root()
    relative = "docs/FI2010_BRUTAL_ABLATIONS.md"
    assert check_no_forbidden_claims(root, scan_paths=[relative]).ok
    assert check_public_release_wording(root, scan_paths=[relative]).ok
