"""Reviewer-facing forecasting-versus-signal-quality centrepiece.

This module consumes retained lightweight execution-v3 analysis outputs and
retained neural full-grid aggregate summaries. It never opens deleted raw
prediction arrays. The output is an offline execution-aware proxy diagnostic:
it connects predictive metrics, calibration, confidence filtering, active
fraction, turnover proxy, cost-adjusted proxy, latency sensitivity and
adverse-selection proxy diagnostics without making a PnL, live-trading or
tradability claim.
"""

from __future__ import annotations

import json
import math
import shutil
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from chronoslob import __version__
from chronoslob.experiments.manifests import sha256_file, stable_json_dumps
from chronoslob.training.experiment import get_git_commit
from chronoslob.utils.paths import project_root

__all__ = [
    "EXECUTION_CENTREPIECE_VERSION",
    "ExecutionCentrepieceSummary",
    "build_execution_centrepiece",
]

EXECUTION_CENTREPIECE_VERSION = "execution-centrepiece/v1"

DEFAULT_SELECTED_THRESHOLDS: tuple[float, ...] = (0.50, 0.70, 0.85)
DEFAULT_REPRESENTATIVE_FEE_BPS = 2.0
DEFAULT_REPRESENTATIVE_SPREAD_MULTIPLIER = 1.0
DEFAULT_REPRESENTATIVE_LATENCY_STEP = 10

_REQUIRED_EXECUTION_ANALYSIS_FILES: tuple[str, ...] = (
    "summary.json",
    "confidence_filtering_summary.csv",
    "turnover_proxy_summary.csv",
    "cost_sensitivity_summary.csv",
    "latency_sensitivity_summary.csv",
    "adverse_selection_proxy_summary.csv",
)
_OPTIONAL_EXECUTION_ANALYSIS_FILES: tuple[str, ...] = (
    "fill_assumption_summary.csv",
    "skipped_regime_diagnostics.json",
    "execution_claim_assessment.json",
    "figure_manifest.json",
)
_OPTIONAL_NEURAL_FILES: tuple[str, ...] = ("summary.json", "aggregate_summary.csv")
_OPTIONAL_EXECUTION_V3_FILES: tuple[str, ...] = ("summary.json", "execution_v3_manifest.json")


@dataclass(frozen=True)
class ExecutionCentrepieceSummary:
    """Compact return value for the generated centrepiece."""

    output_dir: Path
    execution_analysis_dir: Path
    execution_v3_dir: Path | None = None
    neural_full_grid_dir: Path | None = None
    artefacts: Mapping[str, str] = field(default_factory=dict)
    figures_generated: tuple[str, ...] = ()
    claim_statuses: Mapping[str, str] = field(default_factory=dict)
    unavailable_fields: Mapping[str, str] = field(default_factory=dict)
    raw_predictions_required: bool = False


def build_execution_centrepiece(
    *,
    execution_analysis_dir: str | Path = Path("reports/execution_v3_analysis"),
    out_dir: str | Path = Path("reports/execution_centrepiece"),
    execution_v3_dir: str | Path | None = Path("experiments/fi2010_execution_v3"),
    neural_full_grid_dir: str | Path | None = Path("experiments/fi2010_neural_full_grid"),
    selected_thresholds: Sequence[float] = DEFAULT_SELECTED_THRESHOLDS,
    representative_fee_bps: float = DEFAULT_REPRESENTATIVE_FEE_BPS,
    representative_spread_multiplier: float = DEFAULT_REPRESENTATIVE_SPREAD_MULTIPLIER,
    representative_latency_step: int = DEFAULT_REPRESENTATIVE_LATENCY_STEP,
    make_figures: bool = True,
    overwrite: bool = False,
) -> ExecutionCentrepieceSummary:
    """Build the reviewer-facing execution centrepiece from retained tables."""

    analysis_dir = Path(execution_analysis_dir)
    _require_analysis_inputs(analysis_dir)
    output_dir = Path(out_dir)
    _ensure_output_dir(output_dir, overwrite=overwrite)

    execution_v3_path = Path(execution_v3_dir) if execution_v3_dir is not None else None
    neural_path = Path(neural_full_grid_dir) if neural_full_grid_dir is not None else None

    analysis_summary = _read_json(analysis_dir / "summary.json")
    confidence = _read_csv(analysis_dir / "confidence_filtering_summary.csv")
    turnover = _read_csv(analysis_dir / "turnover_proxy_summary.csv")
    cost = _read_csv(analysis_dir / "cost_sensitivity_summary.csv")
    latency = _read_csv(analysis_dir / "latency_sensitivity_summary.csv")
    adverse = _read_csv(analysis_dir / "adverse_selection_proxy_summary.csv")
    fill = _read_csv(analysis_dir / "fill_assumption_summary.csv")
    skipped_regime = _read_json(analysis_dir / "skipped_regime_diagnostics.json")
    execution_claims = _read_json(analysis_dir / "execution_claim_assessment.json")

    predictive = _load_predictive_summary(neural_path)
    execution_v3_summary = (
        _read_json(execution_v3_path / "summary.json") if execution_v3_path else {}
    )
    execution_v3_manifest = (
        _read_json(execution_v3_path / "execution_v3_manifest.json") if execution_v3_path else {}
    )

    thresholds = tuple(_validate_thresholds(selected_thresholds))
    tradeoff = _confidence_threshold_tradeoff(confidence, turnover)
    forecasting_vs_signal = _forecasting_vs_signal_quality(predictive, tradeoff, adverse, latency)
    metric_gap, unavailable_fields = _metric_to_proxy_gap(
        predictive=predictive,
        tradeoff=tradeoff,
        cost=cost,
        latency=latency,
        adverse=adverse,
        selected_thresholds=thresholds,
        representative_fee_bps=representative_fee_bps,
        representative_spread_multiplier=representative_spread_multiplier,
        representative_latency_step=representative_latency_step,
    )
    latency_cost = _latency_cost_gap(
        cost=cost,
        latency=latency,
        representative_fee_bps=representative_fee_bps,
        representative_spread_multiplier=representative_spread_multiplier,
        representative_latency_step=representative_latency_step,
    )
    adverse_by_confidence = _adverse_selection_by_confidence(adverse)
    claims = _claim_assessment(
        smoke_test=bool(analysis_summary.get("smoke_test")),
        predictive=predictive,
        tradeoff=tradeoff,
        cost=cost,
        latency=latency,
        adverse=adverse,
    )
    claim_statuses = {str(claim["claim_id"]): str(claim["status"]) for claim in claims}

    figure_entries = _build_figures(
        out_dir=output_dir,
        tradeoff=tradeoff,
        adverse=adverse_by_confidence,
        predictive=predictive,
        make_figures=make_figures,
    )
    figures_generated = tuple(
        str(entry["figure_id"]) for entry in figure_entries if entry.get("status") == "completed"
    )

    artefacts = {
        "report": "execution_centrepiece.md",
        "summary": "centrepiece_summary.json",
        "forecasting_vs_signal_quality": "forecasting_vs_signal_quality.csv",
        "confidence_threshold_tradeoff": "confidence_threshold_tradeoff.csv",
        "metric_to_proxy_gap": "metric_to_proxy_gap.csv",
        "latency_cost_gap": "latency_cost_gap.csv",
        "adverse_selection_by_confidence": "adverse_selection_by_confidence.csv",
        "claim_assessment": "execution_centrepiece_claim_assessment.json",
        "figure_manifest": "figure_manifest.json",
    }

    _frame_to_csv(
        forecasting_vs_signal,
        output_dir / "forecasting_vs_signal_quality.csv",
    )
    _frame_to_csv(tradeoff, output_dir / "confidence_threshold_tradeoff.csv")
    _frame_to_csv(metric_gap, output_dir / "metric_to_proxy_gap.csv")
    _frame_to_csv(latency_cost, output_dir / "latency_cost_gap.csv")
    _frame_to_csv(
        adverse_by_confidence,
        output_dir / "adverse_selection_by_confidence.csv",
    )

    claim_payload = {
        "builder_version": EXECUTION_CENTREPIECE_VERSION,
        "package_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "offline_diagnostic": True,
        "execution_aware_proxy_diagnostic": True,
        "raw_predictions_required": False,
        "claim_boundary": (
            "The execution centrepiece is an offline diagnostic. It is not PnL, "
            "not live-trading evidence and not a production execution simulator."
        ),
        "claims": claims,
    }
    (output_dir / "execution_centrepiece_claim_assessment.json").write_text(
        stable_json_dumps(claim_payload),
        encoding="utf-8",
    )

    figure_manifest = {
        "builder_version": EXECUTION_CENTREPIECE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "figures": figure_entries,
    }
    (output_dir / "figure_manifest.json").write_text(
        stable_json_dumps(figure_manifest),
        encoding="utf-8",
    )

    summary = ExecutionCentrepieceSummary(
        output_dir=output_dir,
        execution_analysis_dir=analysis_dir,
        execution_v3_dir=execution_v3_path,
        neural_full_grid_dir=neural_path,
        artefacts=artefacts,
        figures_generated=figures_generated,
        claim_statuses=claim_statuses,
        unavailable_fields=unavailable_fields,
        raw_predictions_required=False,
    )
    report_text = _render_report(
        summary=summary,
        analysis_summary=analysis_summary,
        execution_v3_summary=execution_v3_summary,
        execution_v3_manifest=execution_v3_manifest,
        execution_claims=execution_claims,
        skipped_regime=skipped_regime,
        predictive=predictive,
        tradeoff=tradeoff,
        metric_gap=metric_gap,
        latency_cost=latency_cost,
        adverse_by_confidence=adverse_by_confidence,
        fill=fill,
        unavailable_fields=unavailable_fields,
        figure_entries=figure_entries,
    )
    (output_dir / "execution_centrepiece.md").write_text(report_text, encoding="utf-8")

    summary_payload = _summary_payload(
        output_dir=output_dir,
        analysis_dir=analysis_dir,
        execution_v3_path=execution_v3_path,
        neural_path=neural_path,
        analysis_summary=analysis_summary,
        artefacts=artefacts,
        figures_generated=figures_generated,
        claim_statuses=claim_statuses,
        unavailable_fields=unavailable_fields,
        thresholds=thresholds,
        representative_fee_bps=representative_fee_bps,
        representative_spread_multiplier=representative_spread_multiplier,
        representative_latency_step=representative_latency_step,
    )
    (output_dir / "centrepiece_summary.json").write_text(
        stable_json_dumps(summary_payload),
        encoding="utf-8",
    )
    return summary


def _require_analysis_inputs(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"execution-v3 analysis directory missing: {path}")
    for filename in _REQUIRED_EXECUTION_ANALYSIS_FILES:
        candidate = path / filename
        if not candidate.is_file():
            raise FileNotFoundError(
                "execution centrepiece requires retained execution-v3 analysis "
                f"tables; missing {candidate}"
            )


def _ensure_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {path}")
        if any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    "refusing to write into a non-empty output directory; "
                    f"pass overwrite=True to replace it: {path}"
                )
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _load_predictive_summary(neural_full_grid_dir: Path | None) -> pd.DataFrame:
    if neural_full_grid_dir is None:
        return pd.DataFrame()
    path = neural_full_grid_dir / "aggregate_summary.csv"
    frame = _read_csv(path)
    if frame.empty:
        return frame
    keep = [
        column
        for column in (
            "model_family",
            "pretraining_objective",
            "horizon",
            "lookback",
            "completed_run_count",
            "failed_run_count",
            "mean_accuracy",
            "mean_macro_f1",
            "mean_mcc",
            "mean_ece",
            "mean_brier_score",
            "mean_nll",
        )
        if column in frame.columns
    ]
    result = frame[keep].copy()
    if "pretraining_objective" in result.columns:
        result["pretraining_objective"] = result["pretraining_objective"].map(
            lambda value: "supervised" if str(value) == "none" else str(value)
        )
    return result


def _validate_thresholds(values: Sequence[float]) -> list[float]:
    thresholds: list[float] = []
    for value in values:
        number = float(value)
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"confidence threshold must be within [0, 1], got {value}")
        if number not in thresholds:
            thresholds.append(number)
    if not thresholds:
        raise ValueError("at least one selected threshold is required")
    return thresholds


def _confidence_threshold_tradeoff(
    confidence: pd.DataFrame,
    turnover: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["pretraining_objective", "horizon", "threshold"]
    if confidence.empty:
        return pd.DataFrame(columns=keys)
    left = confidence.copy()
    right = turnover.copy()
    if not right.empty:
        right = right[
            [
                column
                for column in (
                    *keys,
                    "mean_signal_change_rate",
                    "mean_turnover_adjusted_cost_proxy",
                )
                if column in right.columns
            ]
        ]
        merged = left.merge(right, on=keys, how="left")
    else:
        merged = left
        merged["mean_signal_change_rate"] = "unavailable: turnover proxy summary missing"
        merged["mean_turnover_adjusted_cost_proxy"] = (
            "unavailable: turnover proxy summary missing"
        )
    ordered = [
        column
        for column in (
            "pretraining_objective",
            "horizon",
            "threshold",
            "n_groups",
            "mean_retained_fraction",
            "mean_active_fraction",
            "mean_abstention_fraction",
            "mean_classification_accuracy",
            "mean_macro_f1",
            "mean_directional_hit_rate",
            "mean_gross_directional_proxy",
            "mean_cost_adjusted_proxy",
            "mean_signal_change_rate",
            "mean_turnover_adjusted_cost_proxy",
        )
        if column in merged.columns
    ]
    return merged[ordered].sort_values(keys, kind="stable").reset_index(drop=True)


def _forecasting_vs_signal_quality(
    predictive: pd.DataFrame,
    tradeoff: pd.DataFrame,
    adverse: pd.DataFrame,
    latency: pd.DataFrame,
) -> pd.DataFrame:
    frame = tradeoff.copy()
    if frame.empty:
        return frame
    frame = _merge_predictive(frame, predictive)
    adverse_summary = _adverse_high_confidence_summary(adverse)
    if not adverse_summary.empty:
        frame = frame.merge(
            adverse_summary,
            on=["pretraining_objective", "horizon"],
            how="left",
        )
    else:
        frame["high_confidence_adverse_selection_proxy"] = (
            "unavailable: adverse-selection proxy summary missing"
        )
    latency_summary = _latency_representative_summary(
        latency,
        representative_latency_step=DEFAULT_REPRESENTATIVE_LATENCY_STEP,
    )
    if not latency_summary.empty:
        frame = frame.merge(
            latency_summary,
            on=["pretraining_objective", "horizon"],
            how="left",
        )
    else:
        frame["representative_latency_degradation"] = (
            "unavailable: latency sensitivity summary missing"
        )
    frame["confidence_filtered_ece"] = (
        "unavailable: retained threshold tables do not include ECE"
    )
    return frame


def _merge_predictive(frame: pd.DataFrame, predictive: pd.DataFrame) -> pd.DataFrame:
    if predictive.empty:
        result = frame.copy()
        for column in ("mean_accuracy_raw", "mean_macro_f1_raw", "mean_ece"):
            result[column] = "unavailable: retained full-grid aggregate missing"
        return result
    keep = [
        column
        for column in (
            "pretraining_objective",
            "horizon",
            "mean_accuracy",
            "mean_macro_f1",
            "mean_mcc",
            "mean_ece",
            "mean_brier_score",
            "mean_nll",
        )
        if column in predictive.columns
    ]
    renamed = predictive[keep].rename(
        columns={
            "mean_accuracy": "mean_accuracy_raw",
            "mean_macro_f1": "mean_macro_f1_raw",
            "mean_mcc": "mean_mcc_raw",
            "mean_brier_score": "mean_brier_score_raw",
            "mean_nll": "mean_nll_raw",
        }
    )
    return frame.merge(renamed, on=["pretraining_objective", "horizon"], how="left")


def _metric_to_proxy_gap(
    *,
    predictive: pd.DataFrame,
    tradeoff: pd.DataFrame,
    cost: pd.DataFrame,
    latency: pd.DataFrame,
    adverse: pd.DataFrame,
    selected_thresholds: Sequence[float],
    representative_fee_bps: float,
    representative_spread_multiplier: float,
    representative_latency_step: int,
) -> tuple[pd.DataFrame, dict[str, str]]:
    keys = _union_keys(predictive, tradeoff, cost, latency, adverse)
    unavailable: dict[str, str] = {
        "confidence_filtered_ece": (
            "unavailable: retained confidence-threshold tables do not include ECE"
        ),
        "raw_predictions": "not required and not read; deleted raw predictions are unavailable",
        "supported_regime_diagnostics": (
            "unavailable: retained tables lack regime labels or snapshot context"
        ),
        "realised_execution": "unavailable: offline diagnostic has no broker or venue fills",
    }
    records: list[dict[str, Any]] = []
    for objective, horizon in keys:
        record: dict[str, Any] = {
            "pretraining_objective": objective,
            "horizon": horizon,
        }
        pred_row = _first_row(predictive, objective, horizon)
        record["predictive_macro_f1"] = _cell(pred_row, "mean_macro_f1")
        record["predictive_ece"] = _cell(pred_row, "mean_ece")
        record["predictive_accuracy"] = _cell(pred_row, "mean_accuracy")
        record["confidence_filtered_ece"] = unavailable["confidence_filtered_ece"]
        for threshold in selected_thresholds:
            suffix = _threshold_suffix(threshold)
            row = _threshold_row(tradeoff, objective, horizon, threshold)
            record[f"active_fraction_at_{suffix}"] = _cell(row, "mean_active_fraction")
            record[f"turnover_proxy_at_{suffix}"] = _cell(row, "mean_signal_change_rate")
            record[f"threshold_macro_f1_at_{suffix}"] = _cell(row, "mean_macro_f1")
            record[f"cost_adjusted_proxy_at_{suffix}"] = _cell(
                row,
                "mean_cost_adjusted_proxy",
            )
        cost_row = _representative_cost_row(
            cost,
            objective,
            horizon,
            representative_fee_bps,
            representative_spread_multiplier,
        )
        record["representative_fee_bps"] = _cell(cost_row, "fee_bps")
        record["representative_spread_multiplier"] = _cell(cost_row, "spread_multiplier")
        record["representative_cost_adjusted_proxy"] = _cell(
            cost_row,
            "mean_cost_adjusted_proxy",
        )
        record["representative_cost_degradation_pct"] = _cell(
            cost_row,
            "mean_degradation_percentage",
        )
        latency_row = _representative_latency_row(
            latency,
            objective,
            horizon,
            representative_latency_step,
        )
        record["representative_latency_step"] = _cell(latency_row, "latency_step")
        record["latency_degradation_vs_lag0"] = _cell(
            latency_row,
            "mean_net_degradation_vs_latency_0",
        )
        record["latency_directional_hit_rate"] = _cell(
            latency_row,
            "mean_directional_hit_rate",
        )
        adverse_row = _high_confidence_adverse_row(adverse, objective, horizon)
        record["high_confidence_bucket"] = _cell(adverse_row, "confidence_bucket")
        record["high_confidence_fill_assumption"] = _cell(adverse_row, "fill_assumption")
        record["high_confidence_adverse_selection_proxy"] = _cell(
            adverse_row,
            "weighted_adverse_fraction",
        )
        records.append(record)
    return pd.DataFrame(records), unavailable


def _latency_cost_gap(
    *,
    cost: pd.DataFrame,
    latency: pd.DataFrame,
    representative_fee_bps: float,
    representative_spread_multiplier: float,
    representative_latency_step: int,
) -> pd.DataFrame:
    keys = _union_keys(pd.DataFrame(), pd.DataFrame(), cost, latency, pd.DataFrame())
    records: list[dict[str, Any]] = []
    for objective, horizon in keys:
        reference = _representative_cost_row(cost, objective, horizon, 0.0, 0.0)
        representative = _representative_cost_row(
            cost,
            objective,
            horizon,
            representative_fee_bps,
            representative_spread_multiplier,
        )
        worst = _worst_cost_row(cost, objective, horizon)
        lagged = _representative_latency_row(
            latency,
            objective,
            horizon,
            representative_latency_step,
        )
        records.append(
            {
                "pretraining_objective": objective,
                "horizon": horizon,
                "reference_cost_adjusted_proxy": _cell(
                    reference,
                    "mean_cost_adjusted_proxy",
                ),
                "representative_fee_bps": _cell(representative, "fee_bps"),
                "representative_spread_multiplier": _cell(
                    representative,
                    "spread_multiplier",
                ),
                "representative_cost_adjusted_proxy": _cell(
                    representative,
                    "mean_cost_adjusted_proxy",
                ),
                "representative_cost_degradation_pct": _cell(
                    representative,
                    "mean_degradation_percentage",
                ),
                "max_cost_fee_bps": _cell(worst, "fee_bps"),
                "max_cost_spread_multiplier": _cell(worst, "spread_multiplier"),
                "max_cost_adjusted_proxy": _cell(worst, "mean_cost_adjusted_proxy"),
                "max_cost_degradation_pct": _cell(worst, "mean_degradation_percentage"),
                "representative_latency_step": _cell(lagged, "latency_step"),
                "latency_degradation_vs_lag0": _cell(
                    lagged,
                    "mean_net_degradation_vs_latency_0",
                ),
                "latency_directional_hit_rate": _cell(
                    lagged,
                    "mean_directional_hit_rate",
                ),
            }
        )
    return pd.DataFrame(records)


def _adverse_selection_by_confidence(adverse: pd.DataFrame) -> pd.DataFrame:
    if adverse.empty:
        return pd.DataFrame(
            columns=[
                "pretraining_objective",
                "horizon",
                "confidence_bucket",
                "fill_assumption",
                "filled_count",
                "adverse_count",
                "weighted_adverse_fraction",
                "adverse_selection_mode",
            ]
        )
    frame = adverse.copy()
    ordered = [
        column
        for column in (
            "pretraining_objective",
            "horizon",
            "confidence_bucket",
            "fill_assumption",
            "total_filled",
            "total_adverse",
            "mean_adverse_fraction",
            "weighted_adverse_fraction",
            "adverse_selection_mode",
        )
        if column in frame.columns
    ]
    return frame[ordered].sort_values(ordered[:4], kind="stable").reset_index(drop=True)


def _union_keys(*frames: pd.DataFrame) -> list[tuple[str, Any]]:
    keys: set[tuple[str, Any]] = set()
    for frame in frames:
        if frame.empty or not {"pretraining_objective", "horizon"} <= set(frame.columns):
            continue
        for _, row in frame[["pretraining_objective", "horizon"]].drop_duplicates().iterrows():
            keys.add((str(row["pretraining_objective"]), row["horizon"]))
    return sorted(keys, key=lambda item: (item[0], _sort_horizon(item[1])))


def _sort_horizon(value: Any) -> float:
    parsed = _safe_float(value)
    return parsed if parsed is not None else math.inf


def _first_row(frame: pd.DataFrame, objective: str, horizon: Any) -> pd.Series | None:
    if frame.empty or not {"pretraining_objective", "horizon"} <= set(frame.columns):
        return None
    scoped = frame[
        (frame["pretraining_objective"].astype(str) == objective)
        & (_numeric_equal(frame["horizon"], horizon))
    ]
    if scoped.empty:
        return None
    return scoped.iloc[0]


def _threshold_row(
    frame: pd.DataFrame,
    objective: str,
    horizon: Any,
    threshold: float,
) -> pd.Series | None:
    if frame.empty or "threshold" not in frame.columns:
        return None
    scoped = frame[
        (frame["pretraining_objective"].astype(str) == objective)
        & (_numeric_equal(frame["horizon"], horizon))
    ].copy()
    if scoped.empty:
        return None
    scoped["_distance"] = (
        pd.to_numeric(scoped["threshold"], errors="coerce") - float(threshold)
    ).abs()
    candidate = scoped.sort_values("_distance", kind="stable").iloc[0]
    distance = _safe_float(candidate.get("_distance"))
    if distance is None or distance > 1e-8:
        return None
    return candidate.drop(labels=["_distance"])


def _representative_cost_row(
    frame: pd.DataFrame,
    objective: str,
    horizon: Any,
    fee_bps: float,
    spread_multiplier: float,
) -> pd.Series | None:
    if frame.empty or not {"fee_bps", "spread_multiplier"} <= set(frame.columns):
        return None
    scoped = frame[
        (frame["pretraining_objective"].astype(str) == objective)
        & (_numeric_equal(frame["horizon"], horizon))
    ].copy()
    if scoped.empty:
        return None
    scoped["_distance"] = (
        (pd.to_numeric(scoped["fee_bps"], errors="coerce") - fee_bps).abs()
        + (pd.to_numeric(scoped["spread_multiplier"], errors="coerce") - spread_multiplier).abs()
    )
    candidate = scoped.sort_values("_distance", kind="stable").iloc[0]
    return candidate.drop(labels=["_distance"])


def _worst_cost_row(frame: pd.DataFrame, objective: str, horizon: Any) -> pd.Series | None:
    if frame.empty or "mean_degradation_percentage" not in frame.columns:
        return None
    scoped = frame[
        (frame["pretraining_objective"].astype(str) == objective)
        & (_numeric_equal(frame["horizon"], horizon))
    ].copy()
    if scoped.empty:
        return None
    scoped["_degradation"] = pd.to_numeric(
        scoped["mean_degradation_percentage"],
        errors="coerce",
    )
    sorted_rows = scoped.sort_values("_degradation", ascending=False, kind="stable")
    return sorted_rows.iloc[0].drop(labels=["_degradation"])


def _representative_latency_row(
    frame: pd.DataFrame,
    objective: str,
    horizon: Any,
    latency_step: int,
) -> pd.Series | None:
    if frame.empty or "latency_step" not in frame.columns:
        return None
    scoped = frame[
        (frame["pretraining_objective"].astype(str) == objective)
        & (_numeric_equal(frame["horizon"], horizon))
    ].copy()
    if scoped.empty:
        return None
    scoped["_distance"] = (
        pd.to_numeric(scoped["latency_step"], errors="coerce") - int(latency_step)
    ).abs()
    candidate = scoped.sort_values("_distance", kind="stable").iloc[0]
    return candidate.drop(labels=["_distance"])


def _high_confidence_adverse_row(
    frame: pd.DataFrame,
    objective: str,
    horizon: Any,
) -> pd.Series | None:
    if frame.empty or "confidence_bucket" not in frame.columns:
        return None
    scoped = frame[
        (frame["pretraining_objective"].astype(str) == objective)
        & (_numeric_equal(frame["horizon"], horizon))
    ].copy()
    if scoped.empty:
        return None
    scoped["_bucket_low"] = scoped["confidence_bucket"].map(_bucket_low)
    high = scoped[scoped["_bucket_low"] >= 0.85]
    if high.empty:
        return None
    high["_fill_rank"] = high["fill_assumption"].map(
        lambda value: 0 if str(value) == "aggressive_crossing" else 1
    )
    candidate = high.sort_values(
        ["_bucket_low", "_fill_rank"],
        ascending=[False, True],
        kind="stable",
    ).iloc[0]
    return candidate.drop(labels=["_bucket_low", "_fill_rank"])


def _adverse_high_confidence_summary(adverse: pd.DataFrame) -> pd.DataFrame:
    if adverse.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for objective, horizon in _union_keys(adverse):
        row = _high_confidence_adverse_row(adverse, objective, horizon)
        rows.append(
            {
                "pretraining_objective": objective,
                "horizon": horizon,
                "high_confidence_adverse_selection_proxy": _cell(
                    row,
                    "weighted_adverse_fraction",
                ),
                "high_confidence_bucket": _cell(row, "confidence_bucket"),
            }
        )
    return pd.DataFrame(rows)


def _latency_representative_summary(
    latency: pd.DataFrame,
    *,
    representative_latency_step: int,
) -> pd.DataFrame:
    if latency.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for objective, horizon in _union_keys(latency):
        row = _representative_latency_row(
            latency,
            objective,
            horizon,
            representative_latency_step,
        )
        rows.append(
            {
                "pretraining_objective": objective,
                "horizon": horizon,
                "representative_latency_step": _cell(row, "latency_step"),
                "representative_latency_degradation": _cell(
                    row,
                    "mean_net_degradation_vs_latency_0",
                ),
            }
        )
    return pd.DataFrame(rows)


def _numeric_equal(series: pd.Series, value: Any) -> pd.Series:
    left = pd.to_numeric(series, errors="coerce")
    right = _safe_float(value)
    if right is None:
        return series.astype(str) == str(value)
    return (left - right).abs() <= 1e-8


def _threshold_suffix(threshold: float) -> str:
    return f"{threshold:.2f}".replace(".", "_")


def _cell(row: pd.Series | None, column: str) -> Any:
    if row is None or column not in row.index:
        return "unavailable"
    value = row.get(column)
    if value is None:
        return "unavailable"
    try:
        if pd.isna(value):
            return "unavailable"
    except TypeError:
        return value
    return value


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bucket_low(value: Any) -> float:
    text = str(value)
    if "-" not in text:
        return -math.inf
    first = text.split("-", maxsplit=1)[0]
    parsed = _safe_float(first)
    return parsed if parsed is not None else -math.inf


def _claim_assessment(
    *,
    smoke_test: bool,
    predictive: pd.DataFrame,
    tradeoff: pd.DataFrame,
    cost: pd.DataFrame,
    latency: pd.DataFrame,
    adverse: pd.DataFrame,
) -> list[dict[str, Any]]:
    def status(present: bool) -> str:
        if smoke_test:
            return "needs_real_evidence"
        return "supported" if present else "needs_real_evidence"

    return [
        {
            "claim_id": "forecasting_vs_signal_quality_gap_analysis",
            "status": status(not predictive.empty and not tradeoff.empty),
            "scope": "retained predictive/calibration summaries joined to proxy diagnostics",
            "safe_rewording": (
                "The centrepiece shows a forecasting-versus-signal-quality gap "
                "using retained offline diagnostic tables."
            ),
        },
        {
            "claim_id": "confidence_filtering_tradeoff_analysis",
            "status": status(not tradeoff.empty),
            "scope": "confidence filtering, retained fraction and active fraction by threshold",
            "safe_rewording": (
                "Report confidence filtering as an offline signal-quality proxy diagnostic."
            ),
        },
        {
            "claim_id": "active_fraction_analysis",
            "status": status("mean_active_fraction" in tradeoff.columns and not tradeoff.empty),
            "scope": "active fraction by objective, horizon and confidence threshold",
            "safe_rewording": "Report active fraction shrinkage under confidence filtering.",
        },
        {
            "claim_id": "turnover_proxy_analysis",
            "status": status("mean_signal_change_rate" in tradeoff.columns and not tradeoff.empty),
            "scope": "turnover proxy by objective, horizon and confidence threshold",
            "safe_rewording": "Report turnover proxy as signal churn, not real order turnover.",
        },
        {
            "claim_id": "latency_cost_gap_analysis",
            "status": status(not cost.empty and not latency.empty),
            "scope": "cost sensitivity and row-step latency sensitivity retained tables",
            "safe_rewording": (
                "Report cost-adjusted proxy and latency sensitivity as offline diagnostics."
            ),
        },
        {
            "claim_id": "adverse_selection_confidence_analysis",
            "status": status(not adverse.empty),
            "scope": "adverse-selection proxy by confidence bucket and fill assumption",
            "safe_rewording": (
                "Report adverse-selection proxy by confidence bucket; it is a proxy, "
                "not measured adverse selection."
            ),
        },
        {
            "claim_id": "profitability_or_tradability",
            "status": "forbidden",
            "scope": "blocked by claim boundary",
            "safe_rewording": (
                "State that the centrepiece does not establish profitability or tradability."
            ),
        },
        {
            "claim_id": "PnL",
            "status": "forbidden",
            "scope": "blocked by claim boundary",
            "safe_rewording": "Use cost-adjusted proxy wording, not PnL wording.",
        },
        {
            "claim_id": "live_trading",
            "status": "forbidden",
            "scope": "blocked by claim boundary",
            "safe_rewording": "Describe the output as an offline diagnostic only.",
        },
    ]


def _build_figures(
    *,
    out_dir: Path,
    tradeoff: pd.DataFrame,
    adverse: pd.DataFrame,
    predictive: pd.DataFrame,
    make_figures: bool,
) -> list[dict[str, Any]]:
    figure_id = "forecasting_vs_signal_quality"
    if not make_figures:
        return [_skipped_figure(figure_id, "figure generation disabled")]
    if tradeoff.empty:
        return [_skipped_figure(figure_id, "confidence-threshold rows are unavailable")]
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return [_skipped_figure(figure_id, "matplotlib is not installed")]

    grouped = _mean_by_threshold(tradeoff)
    if grouped.empty:
        return [_skipped_figure(figure_id, "no numeric confidence-threshold rows")]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    axis_active, axis_cost, axis_turnover, axis_adverse = axes.flatten()

    thresholds = [float(value) for value in grouped["threshold"]]
    axis_active.plot(
        thresholds,
        grouped["mean_active_fraction"],
        color="#1f77b4",
        marker="o",
    )
    axis_active.set_title("Active fraction shrinks")
    axis_active.set_xlabel("confidence threshold")
    axis_active.set_ylabel("active fraction")

    axis_cost.plot(
        thresholds,
        grouped["mean_cost_adjusted_proxy"],
        color="#2ca02c",
        marker="o",
    )
    axis_cost.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
    axis_cost.set_title("Cost-adjusted proxy changes")
    axis_cost.set_xlabel("confidence threshold")
    axis_cost.set_ylabel("cost-adjusted proxy")

    axis_turnover.plot(
        thresholds,
        grouped["mean_signal_change_rate"],
        color="#9467bd",
        marker="o",
    )
    axis_turnover.set_title("Turnover proxy falls with filtering")
    axis_turnover.set_xlabel("confidence threshold")
    axis_turnover.set_ylabel("turnover proxy")

    if adverse.empty:
        axis_adverse.text(0.5, 0.5, "adverse-selection proxy unavailable", ha="center")
        axis_adverse.set_axis_off()
    else:
        adverse_grouped = _mean_adverse_by_bucket(adverse)
        labels = [str(value) for value in adverse_grouped["confidence_bucket"]]
        values = [
            _safe_float(value) or 0.0
            for value in adverse_grouped["weighted_adverse_fraction"]
        ]
        axis_adverse.bar(range(len(values)), values, color="#d62728")
        axis_adverse.set_xticks(range(len(values)))
        axis_adverse.set_xticklabels(labels, rotation=20, ha="right")
        axis_adverse.set_title("Adverse-selection proxy by confidence")
        axis_adverse.set_ylabel("proxy rate")

    fig.suptitle("Forecasting Metrics Versus Execution-Aware Signal-Quality Proxies")
    annotation = _predictive_annotation(predictive)
    if annotation:
        fig.text(0.5, 0.01, annotation, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path = out_dir / f"{figure_id}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return [
        {
            "figure_id": figure_id,
            "title": "Forecasting metrics versus execution-aware signal-quality proxies",
            "status": "completed",
            "reason": "",
            "file_path": path.as_posix(),
        }
    ]


def _skipped_figure(figure_id: str, reason: str) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "title": figure_id.replace("_", " "),
        "status": "skipped",
        "reason": reason,
        "file_path": None,
    }


def _mean_by_threshold(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        column
        for column in (
            "mean_active_fraction",
            "mean_cost_adjusted_proxy",
            "mean_signal_change_rate",
        )
        if column in frame.columns
    ]
    if not numeric_cols or "threshold" not in frame.columns:
        return pd.DataFrame()
    work = frame[["threshold", *numeric_cols]].copy()
    for column in numeric_cols:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    return work.groupby("threshold", sort=True, dropna=False)[numeric_cols].mean().reset_index()


def _mean_adverse_by_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "confidence_bucket" not in frame.columns:
        return pd.DataFrame()
    work = frame[["confidence_bucket", "weighted_adverse_fraction"]].copy()
    work["weighted_adverse_fraction"] = pd.to_numeric(
        work["weighted_adverse_fraction"],
        errors="coerce",
    )
    work["_bucket_low"] = work["confidence_bucket"].map(_bucket_low)
    grouped = (
        work.groupby(["confidence_bucket", "_bucket_low"], sort=True, dropna=False)[
            "weighted_adverse_fraction"
        ]
        .mean()
        .reset_index()
        .sort_values("_bucket_low", kind="stable")
    )
    return grouped.drop(columns=["_bucket_low"])


def _predictive_annotation(predictive: pd.DataFrame) -> str:
    if predictive.empty:
        return "Predictive macro-F1/ECE unavailable from retained aggregate summaries."
    macro = pd.to_numeric(predictive.get("mean_macro_f1"), errors="coerce").dropna()
    ece = pd.to_numeric(predictive.get("mean_ece"), errors="coerce").dropna()
    if macro.empty or ece.empty:
        return ""
    return (
        f"Retained raw predictive range: macro-F1 {macro.min():.3f}-{macro.max():.3f}; "
        f"ECE {ece.min():.3f}-{ece.max():.3f}. Proxy panels use retained execution-v3 summaries."
    )


def _render_report(
    *,
    summary: ExecutionCentrepieceSummary,
    analysis_summary: Mapping[str, Any],
    execution_v3_summary: Mapping[str, Any],
    execution_v3_manifest: Mapping[str, Any],
    execution_claims: Mapping[str, Any],
    skipped_regime: Mapping[str, Any],
    predictive: pd.DataFrame,
    tradeoff: pd.DataFrame,
    metric_gap: pd.DataFrame,
    latency_cost: pd.DataFrame,
    adverse_by_confidence: pd.DataFrame,
    fill: pd.DataFrame,
    unavailable_fields: Mapping[str, str],
    figure_entries: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Execution Centrepiece",
        "",
        "This report is an execution-aware proxy diagnostic built from retained "
        "execution-v3 analysis tables and retained full-grid aggregate summaries.",
        "Deleted raw prediction arrays are not required and are not read.",
        "",
        "It is an offline diagnostic, not PnL, not live-trading evidence and not a "
        "production execution simulator.",
        "",
        "## Inputs",
        "",
    ]
    lines.extend(
        _markdown_table(
            ("input", "path or status"),
            [
                ("execution_v3_analysis", _display_path(summary.execution_analysis_dir)),
                (
                    "execution_v3",
                    _display_path(summary.execution_v3_dir)
                    if summary.execution_v3_dir
                    else "not supplied",
                ),
                (
                    "neural_full_grid",
                    _display_path(summary.neural_full_grid_dir)
                    if summary.neural_full_grid_dir
                    else "not supplied",
                ),
                ("raw_predictions_required", "false"),
                ("payoff_mode", str(analysis_summary.get("payoff_mode", "unknown"))),
                ("cost_mode", str(analysis_summary.get("cost_mode", "unknown"))),
                ("run_group_count", str(analysis_summary.get("run_group_count", "unknown"))),
                ("execution_v3_status", str(execution_v3_summary.get("smoke_test_status", ""))),
                ("execution_manifest_loaded", "true" if execution_v3_manifest else "false"),
            ],
        )
    )
    lines.extend(
        [
            "",
            "## What Predictive Metrics Show",
            "",
        ]
    )
    lines.extend(_predictive_summary_block(predictive))
    lines.extend(
        [
            "",
            "Calibration is represented by retained ECE in the full-grid aggregate "
            "summary. Threshold-level ECE is unavailable because retained "
            "confidence-filtering tables do not include thresholded calibration bins.",
            "",
            "## What Confidence Filtering Changes",
            "",
        ]
    )
    lines.extend(_tradeoff_summary_block(tradeoff))
    lines.extend(
        [
            "",
            "## Metric-To-Proxy Gap",
            "",
            "Selected objective/horizon rows show the gap between retained forecast "
            "metrics and execution-aware signal-quality proxy diagnostics.",
            "",
        ]
    )
    lines.extend(_table_sample(metric_gap, max_rows=9))
    lines.extend(
        [
            "",
            "## Cost And Latency",
            "",
            "Cost-adjusted proxy and latency sensitivity are retained proxy "
            "diagnostics. They are not realised execution outcomes.",
            "",
        ]
    )
    lines.extend(_table_sample(latency_cost, max_rows=9))
    lines.extend(
        [
            "",
            "## Adverse-Selection Proxy",
            "",
            "The adverse-selection proxy is reported by confidence bucket and fill "
            "assumption. It is a label or future-move proxy, not measured adverse "
            "selection from exchange-confirmed fills.",
            "",
        ]
    )
    lines.extend(_table_sample(adverse_by_confidence, max_rows=12))
    lines.extend(
        [
            "",
            "## Fill Assumptions",
            "",
        ]
    )
    lines.extend(_fill_summary_block(fill))
    lines.extend(
        [
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ("figure", "status", "path"),
            [
                (
                    str(entry.get("figure_id", "")),
                    str(entry.get("status", "")),
                    str(entry.get("file_path", "")),
                )
                for entry in figure_entries
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Unavailable Fields",
            "",
        ]
    )
    lines.extend(_markdown_table(("field", "reason"), sorted(unavailable_fields.items())))
    lines.extend(["", "Regime diagnostics remain skipped:", ""])
    lines.extend(
        _wrap_text(str(skipped_regime.get("reason", "no skipped-regime payload available")))
    )
    lines.extend(["", "## Claim Assessment", ""])
    lines.extend(
        _markdown_table(
            ("claim", "status"),
            sorted(summary.claim_statuses.items()),
        )
    )
    if execution_claims:
        lines.extend(
            [
                "",
                "The upstream execution-v3 claim assessment is retained and remains "
                "consistent with this centrepiece.",
            ]
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            (
                "The execution centrepiece does not establish profitability or "
                "tradability."
            ),
            (
                "It shows why forecast metrics must be interpreted alongside "
                "calibration, confidence filtering, active fraction, turnover, "
                "latency, cost and adverse-selection proxy diagnostics."
            ),
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _predictive_summary_block(predictive: pd.DataFrame) -> list[str]:
    if predictive.empty:
        return ["Retained full-grid predictive aggregate rows were unavailable."]
    rows = []
    for _, row in predictive.sort_values(
        ["horizon", "pretraining_objective"],
        kind="stable",
    ).iterrows():
        rows.append(
            (
                str(row.get("pretraining_objective", "")),
                str(row.get("horizon", "")),
                _format_float(row.get("mean_macro_f1")),
                _format_float(row.get("mean_ece")),
                _format_float(row.get("mean_accuracy")),
            )
        )
    return _markdown_table(("objective", "horizon", "macro-F1", "ECE", "accuracy"), rows)


def _tradeoff_summary_block(tradeoff: pd.DataFrame) -> list[str]:
    if tradeoff.empty:
        return ["Confidence-threshold tradeoff rows were unavailable."]
    grouped = _mean_by_threshold(tradeoff)
    rows = []
    for _, row in grouped.iterrows():
        rows.append(
            (
                _format_float(row.get("threshold"), places=2),
                _format_float(row.get("mean_active_fraction")),
                _format_float(row.get("mean_signal_change_rate")),
                _format_float(row.get("mean_cost_adjusted_proxy"), places=1),
            )
        )
    return _markdown_table(
        ("threshold", "active fraction", "turnover proxy", "cost-adjusted proxy"),
        rows,
    )


def _fill_summary_block(fill: pd.DataFrame) -> list[str]:
    if fill.empty:
        return ["Fill-assumption summary rows were unavailable."]
    cols = [
        column
        for column in (
            "pretraining_objective",
            "horizon",
            "fill_mode",
            "mean_fill_fraction",
            "mean_directional_hit_rate",
            "mean_cost_adjusted_proxy",
        )
        if column in fill.columns
    ]
    return _table_sample(fill[cols], max_rows=10)


def _table_sample(frame: pd.DataFrame, *, max_rows: int) -> list[str]:
    if frame.empty:
        return ["No retained rows were available."]
    sample = frame.head(max_rows).copy()
    headers = [str(column) for column in sample.columns]
    rows = []
    for _, row in sample.iterrows():
        rows.append(tuple(_format_cell(row[column]) for column in sample.columns))
    return _markdown_table(headers, rows)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(str(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_cell(str(cell)) for cell in row) + " |")
    return lines


def _wrap_text(text: str, *, width: int = 100) -> list[str]:
    return textwrap.wrap(text, width=width) or [""]


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def _format_cell(value: Any) -> str:
    try:
        if pd.isna(value):
            return "unavailable"
    except TypeError:
        pass
    number = _safe_float(value)
    if number is not None:
        if abs(number) >= 100:
            return f"{number:.1f}"
        return f"{number:.4f}"
    return str(value)


def _format_float(value: Any, *, places: int = 4) -> str:
    number = _safe_float(value)
    return "unavailable" if number is None else f"{number:.{places}f}"


def _summary_payload(
    *,
    output_dir: Path,
    analysis_dir: Path,
    execution_v3_path: Path | None,
    neural_path: Path | None,
    analysis_summary: Mapping[str, Any],
    artefacts: Mapping[str, str],
    figures_generated: Sequence[str],
    claim_statuses: Mapping[str, str],
    unavailable_fields: Mapping[str, str],
    thresholds: Sequence[float],
    representative_fee_bps: float,
    representative_spread_multiplier: float,
    representative_latency_step: int,
) -> dict[str, Any]:
    input_paths = _input_paths(
        analysis_dir=analysis_dir,
        execution_v3_path=execution_v3_path,
        neural_path=neural_path,
    )
    input_hashes = {
        key: sha256_file(Path(path))
        for key, path in input_paths.items()
        if Path(path).is_file()
    }
    output_files = {
        key: str(output_dir / relative)
        for key, relative in artefacts.items()
        if key != "summary"
    }
    output_hashes = {
        key: sha256_file(Path(path))
        for key, path in output_files.items()
        if Path(path).is_file()
    }
    return {
        "builder_version": EXECUTION_CENTREPIECE_VERSION,
        "package_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "offline_diagnostic": True,
        "execution_aware_proxy_diagnostic": True,
        "raw_predictions_required": False,
        "input_artefact_paths": input_paths,
        "input_file_hashes": input_hashes,
        "output_files": output_files,
        "output_file_hashes": output_hashes,
        "artefacts": dict(artefacts),
        "claim_statuses": dict(claim_statuses),
        "figures_generated": list(figures_generated),
        "unavailable_fields": dict(unavailable_fields),
        "selected_thresholds": list(thresholds),
        "representative_fee_bps": representative_fee_bps,
        "representative_spread_multiplier": representative_spread_multiplier,
        "representative_latency_step": representative_latency_step,
        "payoff_mode": analysis_summary.get("payoff_mode"),
        "cost_mode": analysis_summary.get("cost_mode"),
        "run_group_count": analysis_summary.get("run_group_count"),
        "smoke_test": bool(analysis_summary.get("smoke_test")),
    }


def _input_paths(
    *,
    analysis_dir: Path,
    execution_v3_path: Path | None,
    neural_path: Path | None,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    for filename in (*_REQUIRED_EXECUTION_ANALYSIS_FILES, *_OPTIONAL_EXECUTION_ANALYSIS_FILES):
        candidate = analysis_dir / filename
        if candidate.is_file():
            paths[f"execution_analysis_{Path(filename).stem}"] = str(candidate)
    if neural_path is not None:
        for filename in _OPTIONAL_NEURAL_FILES:
            candidate = neural_path / filename
            if candidate.is_file():
                paths[f"neural_full_grid_{Path(filename).stem}"] = str(candidate)
    if execution_v3_path is not None:
        for filename in _OPTIONAL_EXECUTION_V3_FILES:
            candidate = execution_v3_path / filename
            if candidate.is_file():
                paths[f"execution_v3_{Path(filename).stem}"] = str(candidate)
    return paths


def _frame_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _display_path(path: Path | None) -> str:
    if path is None:
        return "not supplied"
    candidate = Path(path)
    try:
        root = project_root().resolve(strict=False)
        return candidate.resolve(strict=False).relative_to(root).as_posix()
    except (OSError, ValueError):
        return candidate.as_posix()
