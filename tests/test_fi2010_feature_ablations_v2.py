from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from chronoslob.analysis.execution_v3 import load_feature_ablation_predictions
from chronoslob.analysis.fi2010_ablation_figures import build_fi2010_ablation_figures
from chronoslob.analysis.fi2010_feature_ablation_analysis import (
    SNAPSHOT_PROXY_SCOPE_NOTE,
    analyse_fi2010_feature_ablations,
)
from chronoslob.experiments.fi2010_feature_ablations import (
    build_aggregate_summary,
    build_feature_delta_summary,
    expand_ablation_specs,
    run_fi2010_feature_ablations,
)


def test_ablation_spec_expansion_modes() -> None:
    group_columns = {
        "price_levels": ["bid_price_1"],
        "spread": ["spread"],
        "snapshot_order_flow_proxy": ["snapshot_delta_bid_price_1"],
    }
    specs = expand_ablation_specs(
        group_columns,
        feature_groups=tuple(group_columns),
        ablation_modes=[
            "all_features",
            "remove_one_group",
            "only_one_group",
            "no_proxy_features",
        ],
    )
    modes = [spec.mode for spec in specs]
    assert "all_features" in modes
    assert modes.count("remove_one_group") == 3
    assert modes.count("only_one_group") == 3
    assert any(
        spec.mode == "no_proxy_features" and "snapshot_order_flow_proxy" not in spec.groups
        for spec in specs
    )


def test_aggregate_and_delta_summary_math() -> None:
    rows = [
        {
            "fold": "fold_1",
            "horizon": 10,
            "seed": 0,
            "model": "logistic",
            "ablation_mode": "all_features",
            "feature_group": "all",
            "accuracy": 0.5,
            "macro_f1": 0.4,
            "mcc": 0.1,
            "ece": 0.2,
            "status": "completed",
        },
        {
            "fold": "fold_1",
            "horizon": 10,
            "seed": 0,
            "model": "logistic",
            "ablation_mode": "only_one_group",
            "feature_group": "spread",
            "accuracy": 0.6,
            "macro_f1": 0.45,
            "mcc": 0.2,
            "ece": 0.15,
            "status": "completed",
        },
    ]
    aggregate = build_aggregate_summary(rows)
    assert any(row["mean_macro_f1"] == 0.45 for row in aggregate)
    delta = build_feature_delta_summary(rows)
    spread = next(row for row in delta if row["feature_group"] == "spread")
    assert spread["delta_macro_f1"] == 0.04999999999999999
    assert spread["interpretation"] == "helped"


def test_runner_smoke_outputs_and_execution_schema(tmp_path: Path) -> None:
    out = tmp_path / "ablations"
    summary = run_fi2010_feature_ablations(
        out_dir=out,
        data_path=Path("tests/fixtures/fi2010/tiny_fi2010_like.csv"),
        folds="1",
        horizons="10",
        seeds="0",
        models="logistic",
        feature_groups="spread,midprice,snapshot_order_flow_proxy",
        ablation_modes="all_features,no_proxy_features",
        reuse_completed=False,
        strict=True,
        smoke_test=True,
        save_predictions=True,
        save_heavy_artefacts=False,
        summary_only=False,
    )
    assert summary.completed_run_count == 2
    assert (out / "results_summary.csv").is_file()

    second_run = run_fi2010_feature_ablations(
        out_dir=out,
        data_path=Path("tests/fixtures/fi2010/tiny_fi2010_like.csv"),
        folds="1",
        horizons="10",
        seeds="0",
        models="logistic",
        feature_groups="spread,midprice,snapshot_order_flow_proxy",
        ablation_modes="all_features,no_proxy_features",
        reuse_completed=True,
        strict=True,
        smoke_test=True,
        save_predictions=True,
        save_heavy_artefacts=False,
        summary_only=False,
    )
    assert second_run.completed_run_count == 2
    assert second_run.skipped_existing_count == 2
    manifest = json.loads((out / "ablation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status_counts"]["skipped_existing"] == 2

    predictions, paths, warnings = load_feature_ablation_predictions(out)
    assert paths
    assert not warnings
    assert {"ablation_mode", "feature_group", "prob_up", "prob_stationary", "prob_down"} <= set(
        predictions.columns
    )


def test_runner_summary_only_skips_predictions_and_feature_matrices(tmp_path: Path) -> None:
    out = tmp_path / "ablations"
    summary = run_fi2010_feature_ablations(
        out_dir=out,
        data_path=Path("tests/fixtures/fi2010/tiny_fi2010_like.csv"),
        folds="1",
        horizons="10",
        seeds="0",
        models="logistic",
        feature_groups="spread,midprice,snapshot_order_flow_proxy",
        ablation_modes="all_features,no_proxy_features",
        reuse_completed=False,
        strict=True,
        smoke_test=True,
        save_predictions=False,
        save_heavy_artefacts=False,
        summary_only=True,
    )

    assert summary.summary_only is True
    assert summary.prediction_files_written == 0
    assert summary.feature_matrix_files_written == 0
    assert not list(out.glob("runs/*/predictions.csv"))
    assert not list(out.glob("features/*/artefacts/features.csv"))

    metrics = json.loads(next(out.glob("runs/*/metrics.json")).read_text(encoding="utf-8"))
    assert metrics["prediction_file"] is None
    assert metrics["predictions_saved"] is False

    predictions, paths, warnings = load_feature_ablation_predictions(out)
    assert predictions.empty
    assert paths == []
    assert warnings == ["no feature-ablation prediction files were available"]


def test_feature_ablation_analysis_consumes_lightweight_artefacts(tmp_path: Path) -> None:
    ablations = tmp_path / "ablations"
    run_fi2010_feature_ablations(
        out_dir=ablations,
        data_path=Path("tests/fixtures/fi2010/tiny_fi2010_like.csv"),
        folds="1",
        horizons="10",
        seeds="0",
        models="logistic",
        feature_groups="spread,snapshot_order_flow_proxy",
        ablation_modes="all_features,remove_one_group",
        reuse_completed=False,
        strict=True,
        smoke_test=True,
        summary_only=True,
    )

    out = tmp_path / "analysis"
    summary = analyse_fi2010_feature_ablations(
        ablation_dir=ablations,
        out_dir=out,
        figures=False,
        allow_smoke_test=True,
    )

    assert summary.raw_predictions_available is False
    assert (out / "feature_delta_by_horizon.csv").is_file()
    assert (out / "feature_group_stability.csv").is_file()
    assert (out / "snapshot_order_flow_proxy_scope.csv").is_file()
    assert SNAPSHOT_PROXY_SCOPE_NOTE in (out / "feature_ablation_analysis.md").read_text(
        encoding="utf-8"
    )
    claims = json.loads((out / "feature_claim_assessment.json").read_text(encoding="utf-8"))
    assert claims["claims"]["causal_feature_importance"]["status"] == "forbidden"
    assert claims["claims"]["true_event_level_ofi"]["status"] == "forbidden"
    assert (
        claims["claims"]["execution_aware_ablation_diagnostics"]["status"]
        == "needs_prediction_outputs"
    )


def test_ablation_figure_manifest(tmp_path: Path) -> None:
    ablations = tmp_path / "ablations"
    ablations.mkdir()
    (ablations / "summary.json").write_text(
        json.dumps({"smoke_test": True}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "fold": "fold_1",
                "horizon": 10,
                "seed": 0,
                "model": "logistic",
                "ablation_mode": "all_features",
                "feature_group": "all",
                "features_used": 3,
                "proxy_features_used": 1,
                "unsupported_groups": "queue_position",
                "accuracy": 0.5,
                "macro_f1": 0.4,
                "mcc": 0.1,
                "ece": 0.2,
                "brier_score": 0.6,
                "status": "completed",
            },
            {
                "fold": "fold_1",
                "horizon": 10,
                "seed": 0,
                "model": "logistic",
                "ablation_mode": "only_one_group",
                "feature_group": "spread",
                "features_used": 1,
                "proxy_features_used": 0,
                "unsupported_groups": "queue_position",
                "accuracy": 0.6,
                "macro_f1": 0.45,
                "mcc": 0.2,
                "ece": 0.15,
                "brier_score": 0.5,
                "status": "completed",
            },
        ]
    ).to_csv(ablations / "results_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "horizon": 10,
                "model": "logistic",
                "ablation_mode": "only_one_group",
                "feature_group": "spread",
                "completed_runs": 1,
                "failed_runs": 0,
                "mean_accuracy": 0.6,
                "std_accuracy": 0.0,
                "mean_macro_f1": 0.45,
                "std_macro_f1": 0.0,
                "mean_mcc": 0.2,
                "std_mcc": 0.0,
                "mean_ece": 0.15,
            },
        ]
    ).to_csv(ablations / "aggregate_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "fold": "fold_1",
                "horizon": 10,
                "seed": 0,
                "model": "logistic",
                "ablation_mode": "only_one_group",
                "feature_group": "spread",
                "delta_macro_f1": 0.05,
                "delta_mcc": 0.1,
                "interpretation": "helped",
            },
        ]
    ).to_csv(ablations / "feature_delta_summary.csv", index=False)
    summary = build_fi2010_ablation_figures(
        ablation_dir=ablations,
        out_dir=tmp_path / "figures",
        allow_smoke_test=True,
    )
    manifest = json.loads(Path(summary.manifest_path).read_text(encoding="utf-8"))
    assert "figures" in manifest
    assert any(
        entry["figure_id"] == "feature_group_delta_macro_f1" for entry in manifest["figures"]
    )
