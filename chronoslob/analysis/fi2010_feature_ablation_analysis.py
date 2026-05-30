"""Stability analysis for lightweight FI-2010 feature-ablation artefacts."""

from __future__ import annotations

import math
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from chronoslob.experiments.manifests import sha256_file, stable_json_dumps
from chronoslob.features.registry import unsupported_group_names
from chronoslob.utils.paths import project_root

__all__ = [
    "FI2010_FEATURE_ABLATION_ANALYSIS_VERSION",
    "FeatureAblationAnalysisSummary",
    "analyse_fi2010_feature_ablations",
]

FI2010_FEATURE_ABLATION_ANALYSIS_VERSION = "fi2010-feature-ablation-stability/v1"
DEFAULT_FEATURE_ABLATION_ANALYSIS_DIR = Path("reports/feature_ablation_analysis")
SNAPSHOT_PROXY_SCOPE_NOTE = (
    "snapshot_order_flow_proxy is a labelled snapshot proxy derived from FI-2010 matrices. "
    "It should not be interpreted as true event-level order-flow imbalance."
)

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)
_FULL_SCOPE_FOLDS = {"fold_1", "fold_2", "fold_3", "fold_4", "fold_5"}
_FULL_SCOPE_HORIZONS = {10, 20, 50}
_FULL_SCOPE_SEEDS = {0, 1, 2}
_FULL_SCOPE_MODELS = {"logistic", "ridge", "elastic_net", "gradient_boosting"}
_FULL_SCOPE_RUNS = 5040
_PLOT_DPI = 160


class FeatureAblationAnalysisSummary(BaseModel):
    """Summary returned by the feature-ablation stability analysis."""

    model_config = _MODEL_CONFIG

    output_dir: str
    ablation_dir: str
    extra_ablation_dirs: list[str] = Field(default_factory=list)
    evidence_status: str
    completed_run_count: int
    failed_run_count: int
    folds: list[str]
    horizons: list[int]
    seeds: list[int]
    models: list[str]
    raw_predictions_available: bool
    files_written: dict[str, str]
    figures_completed: list[str] = Field(default_factory=list)
    figures_skipped: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime
    analysis_version: str

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value


def analyse_fi2010_feature_ablations(
    *,
    ablation_dir: str | Path,
    extra_ablation_dirs: Sequence[str | Path] | str | None = None,
    out_dir: str | Path = DEFAULT_FEATURE_ABLATION_ANALYSIS_DIR,
    figures: bool = True,
    overwrite: bool = False,
    allow_smoke_test: bool = False,
) -> FeatureAblationAnalysisSummary:
    """Analyse feature-ablation stability from retained lightweight CSV artefacts."""
    root = Path(ablation_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"feature-ablation directory missing: {root}")
    summary_path = root / "summary.json"
    results_path = root / "results_summary.csv"
    delta_path = root / "feature_delta_summary.csv"
    for path in (summary_path, results_path, delta_path):
        if not path.is_file():
            raise FileNotFoundError(f"required feature-ablation artefact missing: {path}")
    summary = _read_json(summary_path)
    input_dirs = [root]
    extra_dirs = _normalise_extra_dirs(extra_ablation_dirs)
    extra_summaries: list[dict[str, Any]] = []
    extra_results: list[pd.DataFrame] = []
    extra_deltas: list[pd.DataFrame] = []
    for extra_dir in extra_dirs:
        extra_summary, extra_result, extra_delta = _load_ablation_tables(extra_dir)
        extra_summaries.append(extra_summary)
        extra_results.append(extra_result)
        extra_deltas.append(extra_delta)
        input_dirs.append(extra_dir)
    smoke_test = bool(summary.get("smoke_test"))
    if any(bool(payload.get("smoke_test")) for payload in extra_summaries):
        smoke_test = True
    if smoke_test and not allow_smoke_test:
        raise ValueError(
            "feature-ablation stability analysis refuses smoke-test artefacts unless "
            "allow_smoke_test=True"
        )

    output_dir = Path(out_dir)
    _ensure_output_dir(output_dir, input_dir=root, overwrite=overwrite)
    source_dir = output_dir / "source_data"
    metadata_dir = output_dir / "metadata"
    source_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    results = pd.concat([pd.read_csv(results_path), *extra_results], ignore_index=True, sort=False)
    delta = pd.concat([pd.read_csv(delta_path), *extra_deltas], ignore_index=True, sort=False)
    summary = _combined_summary([summary, *extra_summaries], results)
    completed_delta = _completed_delta(delta)
    remove_delta = completed_delta[
        completed_delta["ablation_mode"].astype(str) == "remove_one_group"
    ].copy()
    warnings: list[str] = []
    if remove_delta.empty:
        warnings.append("no completed remove_one_group delta rows were available")

    by_horizon = _aggregate_dimension(remove_delta, "horizon")
    by_model = _aggregate_dimension(remove_delta, "model")
    by_fold = _aggregate_dimension(remove_delta, "fold")
    by_seed = _aggregate_dimension(remove_delta, "seed")
    stability = _feature_group_stability(remove_delta)
    snapshot_scope = _snapshot_scope(remove_delta)
    unsupported = _unsupported_event_level_groups()
    claim_assessment = _claim_assessment(
        summary=summary,
        results=results,
        remove_delta=remove_delta,
        snapshot_scope=snapshot_scope,
        raw_predictions_available=_raw_predictions_available(input_dirs),
    )

    files_written: dict[str, str] = {}
    _write_frame(by_horizon, output_dir / "feature_delta_by_horizon.csv", files_written)
    _write_frame(by_model, output_dir / "feature_delta_by_model.csv", files_written)
    _write_frame(by_fold, output_dir / "feature_delta_by_fold.csv", files_written)
    _write_frame(by_seed, output_dir / "feature_delta_by_seed.csv", files_written)
    _write_frame(stability, output_dir / "feature_group_stability.csv", files_written)
    _write_frame(
        snapshot_scope,
        output_dir / "snapshot_order_flow_proxy_scope.csv",
        files_written,
    )
    _write_frame(unsupported, output_dir / "unsupported_event_level_groups.csv", files_written)

    claim_path = output_dir / "feature_claim_assessment.json"
    claim_path.write_text(stable_json_dumps(claim_assessment), encoding="utf-8")
    files_written["feature_claim_assessment"] = _display_path(claim_path)

    figure_entries = (
        _build_figures(
            output_dir=output_dir,
            source_dir=source_dir,
            metadata_dir=metadata_dir,
            by_horizon=by_horizon,
            by_model=by_model,
            stability=stability,
            snapshot_scope=snapshot_scope,
            unsupported=unsupported,
        )
        if figures
        else [
            {
                "figure_id": "all_figures",
                "title": "All figures",
                "status": "skipped",
                "reason": "figure generation disabled",
                "file_path": None,
                "source_data_path": None,
                "metadata_path": None,
            }
        ]
    )
    manifest = {
        "analysis_version": FI2010_FEATURE_ABLATION_ANALYSIS_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "ablation_dir": _display_path(root),
        "extra_ablation_dirs": [_display_path(path) for path in extra_dirs],
        "output_dir": _display_path(output_dir),
        "figures": figure_entries,
    }
    figure_manifest_path = output_dir / "figure_manifest.json"
    figure_manifest_path.write_text(stable_json_dumps(manifest), encoding="utf-8")
    files_written["figure_manifest"] = _display_path(figure_manifest_path)

    markdown = _render_markdown(
        summary=summary,
        evidence_status=_evidence_status(summary),
        stability=stability,
        snapshot_scope=snapshot_scope,
        claim_assessment=claim_assessment,
        raw_predictions_available=_raw_predictions_available(input_dirs),
    )
    report_path = output_dir / "feature_ablation_analysis.md"
    report_path.write_text(markdown, encoding="utf-8")
    files_written["feature_ablation_analysis"] = _display_path(report_path)

    completed_run_count = _int_payload(summary.get("completed_run_count")) or int(
        (results.get("status", pd.Series(dtype=str)).astype(str) == "completed").sum()
    )
    failed_run_count = _int_payload(summary.get("failed_run_count")) or int(
        (results.get("status", pd.Series(dtype=str)).astype(str) != "completed").sum()
    )
    raw_predictions_available = _raw_predictions_available(input_dirs)
    summary_out = output_dir / "summary.json"
    files_written["summary"] = _display_path(summary_out)
    summary_model = FeatureAblationAnalysisSummary(
        output_dir=str(output_dir),
        ablation_dir=str(root),
        extra_ablation_dirs=[str(path) for path in extra_dirs],
        evidence_status=_evidence_status(summary),
        completed_run_count=completed_run_count,
        failed_run_count=failed_run_count,
        folds=sorted(_string_set(summary.get("folds"))),
        horizons=sorted(_int_set(summary.get("horizons"))),
        seeds=sorted(_int_set(summary.get("seeds"))),
        models=sorted(_string_set(summary.get("models"))),
        raw_predictions_available=raw_predictions_available,
        files_written=dict(sorted(files_written.items())),
        figures_completed=[
            str(entry["figure_id"])
            for entry in figure_entries
            if entry.get("status") == "completed"
        ],
        figures_skipped=[
            str(entry["figure_id"])
            for entry in figure_entries
            if entry.get("status") != "completed"
        ],
        warnings=warnings,
        created_at=datetime.now(UTC),
        analysis_version=FI2010_FEATURE_ABLATION_ANALYSIS_VERSION,
    )
    payload = summary_model.model_dump(mode="json")
    payload["claim_boundary"] = SNAPSHOT_PROXY_SCOPE_NOTE
    payload["feature_claim_assessment_path"] = _display_path(claim_path)
    summary_out.write_text(stable_json_dumps(payload), encoding="utf-8")
    _write_sha256_manifest(output_dir)
    return summary_model


def _completed_delta(delta: pd.DataFrame) -> pd.DataFrame:
    if delta.empty:
        return delta.copy()
    frame = delta.copy()
    for column in ("delta_macro_f1", "delta_mcc", "baseline_macro_f1", "macro_f1"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "delta_macro_f1" in frame.columns:
        frame = frame.dropna(subset=["delta_macro_f1"])
    return frame


def _normalise_extra_dirs(value: Sequence[str | Path] | str | None) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, str):
        raw: Sequence[str | Path] = [item.strip() for item in value.split(",")]
    else:
        raw = value
    dirs: list[Path] = []
    for item in raw:
        text = str(item).strip()
        if not text:
            continue
        path = Path(text)
        if not path.is_dir():
            raise FileNotFoundError(f"extra feature-ablation directory missing: {path}")
        dirs.append(path)
    return dirs


def _load_ablation_tables(path: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    summary_path = path / "summary.json"
    results_path = path / "results_summary.csv"
    delta_path = path / "feature_delta_summary.csv"
    for candidate in (summary_path, results_path, delta_path):
        if not candidate.is_file():
            raise FileNotFoundError(
                f"required extra feature-ablation artefact missing: {candidate}"
            )
    return _read_json(summary_path), pd.read_csv(results_path), pd.read_csv(delta_path)


def _combined_summary(
    summaries: Sequence[Mapping[str, Any]],
    results: pd.DataFrame,
) -> dict[str, Any]:
    completed = int((results.get("status", pd.Series(dtype=str)).astype(str) == "completed").sum())
    failed = int((results.get("status", pd.Series(dtype=str)).astype(str) != "completed").sum())
    folds: set[str] = set()
    horizons: set[int] = set()
    seeds: set[int] = set()
    models: set[str] = set()
    feature_groups: set[str] = set()
    ablation_modes: set[str] = set()
    unsupported: set[str] = set()
    proxy_groups: set[str] = set()
    smoke = False
    for summary in summaries:
        folds.update(_string_set(summary.get("folds")))
        horizons.update(_int_set(summary.get("horizons")))
        seeds.update(_int_set(summary.get("seeds")))
        models.update(_string_set(summary.get("models")))
        feature_groups.update(_string_set(summary.get("feature_groups")))
        ablation_modes.update(_string_set(summary.get("ablation_modes")))
        unsupported.update(_string_set(summary.get("unsupported_groups")))
        proxy_groups.update(_string_set(summary.get("proxy_groups")))
        smoke = smoke or bool(summary.get("smoke_test"))
    return {
        "smoke_test": smoke,
        "completed_run_count": completed,
        "failed_run_count": failed,
        "folds": sorted(folds),
        "horizons": sorted(horizons),
        "seeds": sorted(seeds),
        "models": sorted(models),
        "feature_groups": sorted(feature_groups),
        "ablation_modes": sorted(ablation_modes),
        "unsupported_groups": sorted(unsupported),
        "proxy_groups": sorted(proxy_groups),
    }


def _raw_predictions_available(input_dirs: Sequence[Path]) -> bool:
    return any(any(path.glob("runs/*/predictions.csv")) for path in input_dirs)


def _aggregate_dimension(frame: pd.DataFrame, dimension: str) -> pd.DataFrame:
    columns = [
        "feature_group",
        dimension,
        "run_count",
        "mean_delta_macro_f1",
        "mean_delta_mcc",
        "std_delta_macro_f1",
        "std_delta_mcc",
        "macro_f1_degradation_count",
        "macro_f1_improvement_count",
        "macro_f1_neutral_count",
        "mcc_degradation_count",
        "mcc_improvement_count",
        "mcc_neutral_count",
    ]
    if frame.empty or dimension not in frame.columns:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(["feature_group", dimension], dropna=False, sort=True):
        feature_group, value = keys
        macro = pd.to_numeric(group["delta_macro_f1"], errors="coerce").dropna()
        mcc = pd.to_numeric(group.get("delta_mcc"), errors="coerce").dropna()
        rows.append(
            {
                "feature_group": feature_group,
                dimension: value,
                "run_count": len(group),
                "mean_delta_macro_f1": _mean(macro),
                "mean_delta_mcc": _mean(mcc),
                "std_delta_macro_f1": _std(macro),
                "std_delta_mcc": _std(mcc),
                "macro_f1_degradation_count": int((macro < 0.0).sum()),
                "macro_f1_improvement_count": int((macro > 0.0).sum()),
                "macro_f1_neutral_count": int((macro == 0.0).sum()),
                "mcc_degradation_count": int((mcc < 0.0).sum()),
                "mcc_improvement_count": int((mcc > 0.0).sum()),
                "mcc_neutral_count": int((mcc == 0.0).sum()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _feature_group_stability(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "feature_group",
        "run_count",
        "mean_delta_macro_f1",
        "mean_delta_mcc",
        "abs_mean_delta_macro_f1",
        "macro_f1_degradation_count",
        "macro_f1_improvement_count",
        "macro_f1_degradation_fraction",
        "mcc_degradation_count",
        "mcc_improvement_count",
        "mcc_degradation_fraction",
        "fold_consistency",
        "seed_consistency",
        "horizon_consistency",
        "model_consistency",
        "stability_score",
        "folds_covered",
        "seeds_covered",
        "horizons_covered",
        "models_covered",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for feature_group, group in frame.groupby("feature_group", sort=True):
        macro = pd.to_numeric(group["delta_macro_f1"], errors="coerce").dropna()
        mcc = pd.to_numeric(group.get("delta_mcc"), errors="coerce").dropna()
        consistencies = {
            "fold_consistency": _negative_mean_fraction(group, "fold"),
            "seed_consistency": _negative_mean_fraction(group, "seed"),
            "horizon_consistency": _negative_mean_fraction(group, "horizon"),
            "model_consistency": _negative_mean_fraction(group, "model"),
        }
        valid_consistencies = [
            value for value in consistencies.values() if value is not None and math.isfinite(value)
        ]
        rows.append(
            {
                "feature_group": feature_group,
                "run_count": len(group),
                "mean_delta_macro_f1": _mean(macro),
                "mean_delta_mcc": _mean(mcc),
                "abs_mean_delta_macro_f1": abs(_mean(macro) or 0.0),
                "macro_f1_degradation_count": int((macro < 0.0).sum()),
                "macro_f1_improvement_count": int((macro > 0.0).sum()),
                "macro_f1_degradation_fraction": _fraction(int((macro < 0.0).sum()), len(macro)),
                "mcc_degradation_count": int((mcc < 0.0).sum()),
                "mcc_improvement_count": int((mcc > 0.0).sum()),
                "mcc_degradation_fraction": _fraction(int((mcc < 0.0).sum()), len(mcc)),
                **consistencies,
                "stability_score": _mean(pd.Series(valid_consistencies)),
                "folds_covered": _join_sorted(group.get("fold")),
                "seeds_covered": _join_sorted(group.get("seed")),
                "horizons_covered": _join_sorted(group.get("horizon")),
                "models_covered": _join_sorted(group.get("model")),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["abs_mean_delta_macro_f1", "feature_group"],
        ascending=[False, True],
    )


def _snapshot_scope(frame: pd.DataFrame) -> pd.DataFrame:
    source = frame[frame["feature_group"].astype(str) == "snapshot_order_flow_proxy"].copy()
    columns = [
        "fold",
        "horizon",
        "seed",
        "model",
        "ablation_mode",
        "feature_group",
        "delta_macro_f1",
        "delta_mcc",
        "macro_f1_degraded_when_removed",
        "mcc_degraded_when_removed",
        "horizon_beyond_10",
        "proxy_scope_note",
    ]
    if source.empty:
        return pd.DataFrame(columns=columns)
    source["macro_f1_degraded_when_removed"] = (
        pd.to_numeric(source["delta_macro_f1"], errors="coerce") < 0.0
    )
    source["mcc_degraded_when_removed"] = pd.to_numeric(source["delta_mcc"], errors="coerce") < 0.0
    source["horizon_beyond_10"] = pd.to_numeric(source["horizon"], errors="coerce") != 10
    source["proxy_scope_note"] = SNAPSHOT_PROXY_SCOPE_NOTE
    return source[columns].sort_values(["horizon", "model", "fold", "seed"]).reset_index(drop=True)


def _unsupported_event_level_groups() -> pd.DataFrame:
    rows = [
        {
            "feature_group": group,
            "status": "unsupported",
            "reason": "FI-2010 snapshot matrices do not contain event messages or queue position.",
        }
        for group in unsupported_group_names()
        if group
        in {
            "time_context",
            "true_order_flow_imbalance",
            "cancellation_imbalance",
            "trade_imbalance",
            "queue_position",
        }
    ]
    return pd.DataFrame(rows, columns=["feature_group", "status", "reason"])


def _claim_assessment(
    *,
    summary: Mapping[str, Any],
    results: pd.DataFrame,
    remove_delta: pd.DataFrame,
    snapshot_scope: pd.DataFrame,
    raw_predictions_available: bool,
) -> dict[str, Any]:
    h10 = snapshot_scope[
        (pd.to_numeric(snapshot_scope.get("horizon"), errors="coerce") == 10)
        & snapshot_scope.get("model", pd.Series(dtype=str)).astype(str).isin(["logistic", "ridge"])
    ]
    broader = snapshot_scope[
        pd.to_numeric(snapshot_scope.get("horizon"), errors="coerce").isin([20, 50])
    ]
    nonlinear = snapshot_scope[
        snapshot_scope.get("model", pd.Series(dtype=str)).astype(str).isin(["gradient_boosting"])
    ]
    return {
        "analysis_version": FI2010_FEATURE_ABLATION_ANALYSIS_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "evidence_status": _evidence_status(summary),
        "actual_scope": {
            "folds": sorted(_string_set(summary.get("folds"))),
            "horizons": sorted(_int_set(summary.get("horizons"))),
            "seeds": sorted(_int_set(summary.get("seeds"))),
            "models": sorted(_string_set(summary.get("models"))),
            "completed_run_count": _int_payload(summary.get("completed_run_count")),
            "failed_run_count": _int_payload(summary.get("failed_run_count")),
            "completed_result_rows": int(
                (results.get("status", pd.Series(dtype=str)).astype(str) == "completed").sum()
            ),
        },
        "claims": {
            "feature_ablation_infrastructure": {
                "status": "supported",
                "reason": "required feature-ablation summary and delta tables were loaded",
            },
            "horizon10_logistic_ridge_snapshot_proxy_importance": _snapshot_status(h10),
            "broader_horizon_snapshot_proxy_importance": _snapshot_status(broader),
            "nonlinear_model_feature_stability": _snapshot_status(
                nonlinear,
                absent_status="needs_real_evidence",
            ),
            "execution_aware_ablation_diagnostics": {
                "status": "supported" if raw_predictions_available else "needs_prediction_outputs",
                "reason": (
                    "retained ablation prediction files are available"
                    if raw_predictions_available
                    else (
                        "Execution-aware ablation diagnostics require retained "
                        "prediction-level outputs or a targeted rerun."
                    )
                ),
            },
            "causal_feature_importance": {
                "status": "forbidden",
                "reason": "ablation deltas are associational diagnostics, not causal evidence",
            },
            "true_event_level_ofi": {
                "status": "forbidden",
                "reason": SNAPSHOT_PROXY_SCOPE_NOTE,
            },
        },
        "strongest_negative_remove_effects": _top_effects(remove_delta, ascending=True),
        "strongest_positive_remove_effects": _top_effects(remove_delta, ascending=False),
        "proxy_scope_note": SNAPSHOT_PROXY_SCOPE_NOTE,
    }


def _snapshot_status(
    frame: pd.DataFrame,
    *,
    absent_status: str = "needs_real_evidence",
) -> dict[str, Any]:
    if frame.empty:
        return {
            "status": absent_status,
            "run_count": 0,
            "degraded_count": 0,
            "reason": "no matching snapshot_order_flow_proxy rows were available",
        }
    degraded = int(frame["macro_f1_degraded_when_removed"].sum())
    total = len(frame)
    if degraded == total:
        status = "supported"
        reason = "removing snapshot_order_flow_proxy degraded macro-F1 in every matched row"
    elif degraded > total / 2:
        status = "partially_supported"
        reason = "removing snapshot_order_flow_proxy degraded macro-F1 in a majority of rows"
    else:
        status = "unsupported"
        reason = "removing snapshot_order_flow_proxy did not consistently degrade macro-F1"
    return {
        "status": status,
        "run_count": total,
        "degraded_count": degraded,
        "degraded_fraction": _fraction(degraded, total),
        "mean_delta_macro_f1": _mean(pd.to_numeric(frame["delta_macro_f1"], errors="coerce")),
        "mean_delta_mcc": _mean(pd.to_numeric(frame["delta_mcc"], errors="coerce")),
        "reason": reason,
    }


def _top_effects(frame: pd.DataFrame, *, ascending: bool) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    grouped = (
        frame.assign(delta_macro_f1=pd.to_numeric(frame["delta_macro_f1"], errors="coerce"))
        .dropna(subset=["delta_macro_f1"])
        .groupby("feature_group", sort=True)["delta_macro_f1"]
        .mean()
        .reset_index()
        .sort_values("delta_macro_f1", ascending=ascending)
        .head(5)
    )
    return [
        {
            "feature_group": str(row["feature_group"]),
            "mean_delta_macro_f1": float(row["delta_macro_f1"]),
        }
        for _, row in grouped.iterrows()
    ]


def _render_markdown(
    *,
    summary: Mapping[str, Any],
    evidence_status: str,
    stability: pd.DataFrame,
    snapshot_scope: pd.DataFrame,
    claim_assessment: Mapping[str, Any],
    raw_predictions_available: bool,
) -> str:
    claims = claim_assessment.get("claims", {})
    broader = claims.get("broader_horizon_snapshot_proxy_importance", {})
    h10 = claims.get("horizon10_logistic_ridge_snapshot_proxy_importance", {})
    nonlinear = claims.get("nonlinear_model_feature_stability", {})
    lines = [
        "# FI-2010 Feature Ablation And Stability Analysis",
        "",
        "This report strengthens the retained feature-ablation evidence into a "
        "scoped feature-stability analysis.",
        "",
        SNAPSHOT_PROXY_SCOPE_NOTE,
        "",
        "These diagnostics are not causal feature importance and should not be "
        "read as universal feature importance across all models or horizons.",
        "",
        "## Completed Scope",
        "",
        *_markdown_table(
            ("field", "value"),
            [
                ("evidence status", evidence_status),
                ("completed fits", str(summary.get("completed_run_count", "not available"))),
                ("failed fits", str(summary.get("failed_run_count", "not available"))),
                ("folds", _join_values(summary.get("folds"))),
                ("horizons", _join_values(summary.get("horizons"))),
                ("seeds", _join_values(summary.get("seeds"))),
                ("models", _join_values(summary.get("models"))),
                ("raw predictions retained", "yes" if raw_predictions_available else "no"),
            ],
        ),
        "",
        "## Snapshot Proxy Finding",
        "",
        (
            "- Horizon-10 logistic/ridge status: "
            f"{h10.get('status', 'not available')} "
            f"({h10.get('degraded_count', 0)}/{h10.get('run_count', 0)} "
            "matched rows degraded when removed)."
        ),
        (
            "- Horizon-20/50 status: "
            f"{broader.get('status', 'not available')} "
            f"({broader.get('degraded_count', 0)}/{broader.get('run_count', 0)} "
            "matched rows degraded when removed)."
        ),
        f"- Non-linear slice status: {nonlinear.get('status', 'not available')}.",
        "",
    ]
    if not raw_predictions_available:
        lines.extend(
            [
                "Execution-aware ablation diagnostics require retained "
                "prediction-level outputs or a targeted rerun.",
                "",
            ]
        )
    lines.extend(["## Strongest Mean Remove-One-Group Effects", ""])
    if stability.empty:
        lines.append("No completed remove-one-group rows were available.")
    else:
        top = stability.head(10)
        rows = [
            (
                str(row["feature_group"]),
                _format_float(_any_float(row["mean_delta_macro_f1"])),
                _format_float(_any_float(row["mean_delta_mcc"])),
                _format_float(_any_float(row["macro_f1_degradation_fraction"])),
                _format_float(_any_float(row["stability_score"])),
            )
            for _, row in top.iterrows()
        ]
        lines.extend(
            _markdown_table(
                (
                    "feature group",
                    "mean delta macro-F1",
                    "mean delta MCC",
                    "degradation fraction",
                    "stability score",
                ),
                rows,
            )
        )
    lines.extend(
        [
            "",
            "## Claim Assessment",
            "",
            *_markdown_table(
                ("claim", "status", "reason"),
                [
                    (
                        str(claim_id),
                        str(payload.get("status", "")),
                        str(payload.get("reason", "")),
                    )
                    for claim_id, payload in claims.items()
                    if isinstance(payload, Mapping)
                ],
            ),
        ]
    )
    if not snapshot_scope.empty:
        horizons = _join_values(sorted(set(snapshot_scope["horizon"].astype(str))))
        lines.extend(
            [
                "",
                "## Snapshot Proxy Scope Rows",
                "",
                (
                    "`snapshot_order_flow_proxy_scope.csv` contains "
                    f"{len(snapshot_scope)} matched remove-one-group rows "
                    f"across horizons {horizons}."
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def _build_figures(
    *,
    output_dir: Path,
    source_dir: Path,
    metadata_dir: Path,
    by_horizon: pd.DataFrame,
    by_model: pd.DataFrame,
    stability: pd.DataFrame,
    snapshot_scope: pd.DataFrame,
    unsupported: pd.DataFrame,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        plt = _import_matplotlib()
    except RuntimeError as exc:
        for figure_id in (
            "feature_group_delta_by_horizon",
            "feature_group_delta_by_model",
            "snapshot_order_flow_proxy_delta_by_horizon_fold",
            "stability_heatmap",
            "top_feature_groups_by_absolute_delta",
        ):
            _skip(entries, figure_id, str(exc))
        _csv_only(
            entries,
            figure_id="unsupported_event_level_groups",
            title="Unsupported Event-Level Groups",
            frame=unsupported,
            source_dir=source_dir,
        )
        return entries

    _bar_figure(
        plt,
        entries,
        by_horizon,
        output_dir,
        source_dir,
        metadata_dir,
        figure_id="feature_group_delta_by_horizon",
        title="Feature Group Delta By Horizon",
        index="feature_group",
        columns="horizon",
        values="mean_delta_macro_f1",
        ylabel="mean remove-group delta macro-F1",
    )
    _bar_figure(
        plt,
        entries,
        by_model,
        output_dir,
        source_dir,
        metadata_dir,
        figure_id="feature_group_delta_by_model",
        title="Feature Group Delta By Model",
        index="feature_group",
        columns="model",
        values="mean_delta_macro_f1",
        ylabel="mean remove-group delta macro-F1",
    )
    snapshot_grouped = (
        snapshot_scope.assign(
            delta_macro_f1=pd.to_numeric(snapshot_scope.get("delta_macro_f1"), errors="coerce")
        )
        .groupby(["horizon", "fold"], sort=True)["delta_macro_f1"]
        .mean()
        .reset_index()
        if not snapshot_scope.empty
        else pd.DataFrame()
    )
    _bar_figure(
        plt,
        entries,
        snapshot_grouped,
        output_dir,
        source_dir,
        metadata_dir,
        figure_id="snapshot_order_flow_proxy_delta_by_horizon_fold",
        title="snapshot_order_flow_proxy Delta By Horizon/Fold",
        index="fold",
        columns="horizon",
        values="delta_macro_f1",
        ylabel="mean remove-group delta macro-F1",
    )
    _heatmap_figure(plt, entries, stability, output_dir, source_dir, metadata_dir)
    top = stability.sort_values("abs_mean_delta_macro_f1", ascending=False).head(10)
    _bar_figure(
        plt,
        entries,
        top,
        output_dir,
        source_dir,
        metadata_dir,
        figure_id="top_feature_groups_by_absolute_delta",
        title="Top Feature Groups By Absolute Delta",
        index="feature_group",
        columns=None,
        values="abs_mean_delta_macro_f1",
        ylabel="absolute mean remove-group delta macro-F1",
    )
    _csv_only(
        entries,
        figure_id="unsupported_event_level_groups",
        title="Unsupported Event-Level Groups",
        frame=unsupported,
        source_dir=source_dir,
    )
    return entries


def _bar_figure(
    plt: Any,
    entries: list[dict[str, Any]],
    frame: pd.DataFrame,
    output_dir: Path,
    source_dir: Path,
    metadata_dir: Path,
    *,
    figure_id: str,
    title: str,
    index: str,
    columns: str | None,
    values: str,
    ylabel: str,
) -> None:
    if frame.empty or values not in frame.columns or index not in frame.columns:
        _skip(entries, figure_id, f"{values} rows unavailable")
        return
    source = frame.copy()
    source[values] = pd.to_numeric(source[values], errors="coerce")
    source = source.dropna(subset=[values])
    if source.empty:
        _skip(entries, figure_id, f"no finite {values} values")
        return
    source_path = source_dir / f"{figure_id}.csv"
    source.to_csv(source_path, index=False)
    target = output_dir / f"{figure_id}.png"
    metadata_path = _metadata(metadata_dir, figure_id, title, source_path)
    figure, ax = plt.subplots(figsize=(8.0, 4.8), dpi=_PLOT_DPI)
    if columns is not None and columns in source.columns:
        pivot = source.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
        pivot.plot(kind="bar", ax=ax, width=0.78)
        ax.legend(loc="best", fontsize="x-small", frameon=False)
    else:
        source.plot(kind="bar", x=index, y=values, ax=ax, legend=False, width=0.78)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(index.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    figure.autofmt_xdate(rotation=30, ha="right")
    figure.tight_layout()
    figure.savefig(target, dpi=_PLOT_DPI)
    plt.close(figure)
    _complete(entries, figure_id, title, target, source_path, metadata_path, len(source))


def _heatmap_figure(
    plt: Any,
    entries: list[dict[str, Any]],
    stability: pd.DataFrame,
    output_dir: Path,
    source_dir: Path,
    metadata_dir: Path,
) -> None:
    figure_id = "stability_heatmap"
    title = "Feature Stability Heatmap"
    columns = ["fold_consistency", "seed_consistency", "horizon_consistency", "model_consistency"]
    if stability.empty or not set(columns) <= set(stability.columns):
        _skip(entries, figure_id, "stability rows unavailable")
        return
    source = stability[["feature_group", *columns]].copy()
    for column in columns:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source = source.dropna(subset=columns, how="all").head(12)
    if source.empty:
        _skip(entries, figure_id, "no finite stability values")
        return
    source_path = source_dir / f"{figure_id}.csv"
    source.to_csv(source_path, index=False)
    target = output_dir / f"{figure_id}.png"
    metadata_path = _metadata(metadata_dir, figure_id, title, source_path)
    matrix = source.set_index("feature_group")[columns]
    figure, ax = plt.subplots(figsize=(6.8, 5.2), dpi=_PLOT_DPI)
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([column.replace("_", " ") for column in columns], rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index.tolist())
    ax.set_title(title)
    figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(target, dpi=_PLOT_DPI)
    plt.close(figure)
    _complete(entries, figure_id, title, target, source_path, metadata_path, len(source))


def _csv_only(
    entries: list[dict[str, Any]],
    *,
    figure_id: str,
    title: str,
    frame: pd.DataFrame,
    source_dir: Path,
) -> None:
    source_path = source_dir / f"{figure_id}.csv"
    frame.to_csv(source_path, index=False)
    entries.append(
        {
            "figure_id": figure_id,
            "title": title,
            "status": "csv_only",
            "reason": "unsupported event-level groups are marked in source CSV",
            "file_path": None,
            "source_data_path": _display_path(source_path),
            "metadata_path": None,
            "row_count": len(frame),
        }
    )


def _complete(
    entries: list[dict[str, Any]],
    figure_id: str,
    title: str,
    target: Path,
    source_path: Path,
    metadata_path: Path,
    row_count: int,
) -> None:
    entries.append(
        {
            "figure_id": figure_id,
            "title": title,
            "status": "completed",
            "reason": "",
            "file_path": _display_path(target),
            "source_data_path": _display_path(source_path),
            "metadata_path": _display_path(metadata_path),
            "row_count": row_count,
        }
    )


def _skip(entries: list[dict[str, Any]], figure_id: str, reason: str) -> None:
    entries.append(
        {
            "figure_id": figure_id,
            "title": figure_id.replace("_", " ").title(),
            "status": "skipped",
            "reason": reason,
            "file_path": None,
            "source_data_path": None,
            "metadata_path": None,
            "row_count": 0,
        }
    )


def _metadata(metadata_dir: Path, figure_id: str, title: str, source_path: Path) -> Path:
    path = metadata_dir / f"{figure_id}.json"
    path.write_text(
        stable_json_dumps(
            {
                "figure_id": figure_id,
                "title": title,
                "source_data_path": _display_path(source_path),
                "source_sha256": sha256_file(source_path),
                "created_at": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    return path


def _import_matplotlib() -> Any:
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for figure generation") from exc
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _write_frame(frame: pd.DataFrame, path: Path, files_written: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    files_written[path.stem] = _display_path(path)


def _write_sha256_manifest(output_dir: Path) -> None:
    files = {
        str(path.relative_to(output_dir)).replace("\\", "/"): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "sha256_manifest.json"
    }
    (output_dir / "sha256_manifest.json").write_text(
        stable_json_dumps({"files": files}),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return dict(payload)


def _ensure_output_dir(out_dir: Path, *, input_dir: Path, overwrite: bool) -> None:
    resolved_out = out_dir.resolve(strict=False)
    resolved_input = input_dir.resolve(strict=False)
    if resolved_out == resolved_input or resolved_input.is_relative_to(resolved_out):
        raise ValueError("analysis output directory must not contain the input directory")
    if out_dir.exists():
        if not out_dir.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {out_dir}")
        if any(out_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"refusing to overwrite non-empty output directory: {out_dir}"
                )
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _evidence_status(summary: Mapping[str, Any]) -> str:
    if bool(summary.get("smoke_test")):
        return "smoke_test_only"
    folds = _string_set(summary.get("folds"))
    horizons = _int_set(summary.get("horizons"))
    seeds = _int_set(summary.get("seeds"))
    models = _string_set(summary.get("models"))
    completed = _int_payload(summary.get("completed_run_count"))
    failed = _int_payload(summary.get("failed_run_count")) or 0
    if (
        folds >= _FULL_SCOPE_FOLDS
        and horizons >= _FULL_SCOPE_HORIZONS
        and seeds >= _FULL_SCOPE_SEEDS
        and models >= _FULL_SCOPE_MODELS
        and completed is not None
        and completed >= _FULL_SCOPE_RUNS
        and failed == 0
    ):
        return "complete_real"
    return "partial_real"


def _negative_mean_fraction(frame: pd.DataFrame, dimension: str) -> float | None:
    if dimension not in frame.columns or frame.empty:
        return None
    grouped = (
        frame.assign(delta_macro_f1=pd.to_numeric(frame["delta_macro_f1"], errors="coerce"))
        .dropna(subset=["delta_macro_f1"])
        .groupby(dimension, dropna=False)["delta_macro_f1"]
        .mean()
    )
    if grouped.empty:
        return None
    return float((grouped < 0.0).sum() / len(grouped))


def _mean(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    value = float(clean.mean())
    return value if math.isfinite(value) else None


def _std(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return 0.0
    value = float(clean.std(ddof=0))
    return value if math.isfinite(value) else 0.0


def _fraction(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _string_set(value: Any) -> set[str]:
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if str(item)}
    return set()


def _int_set(value: Any) -> set[int]:
    result: set[int] = set()
    if not isinstance(value, (list, tuple, set)):
        return result
    for item in value:
        number = _int_payload(item)
        if number is not None:
            result.add(number)
    return result


def _int_payload(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _any_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _join_sorted(values: Any) -> str:
    if values is None:
        return ""
    try:
        return ",".join(sorted({str(value) for value in values if str(value)}))
    except TypeError:
        return ""


def _join_values(values: Any) -> str:
    if not isinstance(values, (list, tuple, set)):
        return "not available"
    return ", ".join(str(value) for value in values) if values else "not available"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape(str(value)) for value in row) + " |")
    return lines


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def _display_path(path: Path) -> str:
    candidate = Path(path)
    try:
        resolved = candidate.resolve(strict=False)
        root = project_root().resolve(strict=False)
        return resolved.relative_to(root).as_posix()
    except (OSError, ValueError):
        return candidate.as_posix()
