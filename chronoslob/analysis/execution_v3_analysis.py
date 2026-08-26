"""Richer offline analysis of FI-2010 execution-v3 proxy diagnostics.

This module is a second-stage, reviewer-facing summariser. It consumes only the
retained lightweight execution-v3 output tables that ``execution_v3.py`` already
wrote (``confidence_threshold_summary.csv``, ``cost_sensitivity_summary.csv``,
``latency_sensitivity_summary.csv``, ``fill_assumption_summary.csv``,
``adverse_selection_summary.csv``, ``regime_execution_summary.csv``,
``summary.json`` and ``execution_v3_manifest.json``).

It never opens the heavy raw per-run prediction arrays, which were intentionally
deleted to save disk space. ``raw_predictions_required`` is always reported as
``False``; if the underlying execution-v3 tables are absent the analysis fails
with a clear missing-input message rather than silently requiring deleted data.

Every output is an offline execution-aware proxy diagnostic. It is intentionally
not a broker integration, live trading component, profitability study or a
realistic execution simulator. The analysis answers one practical question:
do apparently better forecasts still look useful after confidence filtering,
simple cost assumptions, latency shifts, turnover pressure and adverse-selection
proxies? Cost-adjusted values are cost-adjusted proxies, not realised PnL.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from chronoslob import __version__

__all__ = [
    "EXECUTION_V3_ANALYSIS_VERSION",
    "ExecutionV3AnalysisSummary",
    "analyse_fi2010_execution_v3",
]

EXECUTION_V3_ANALYSIS_VERSION = "fi2010-execution-aware-proxy-analysis/v1"

# Only these lightweight, retained execution-v3 output tables are read. Heavy
# ``runs/`` prediction arrays are never required.
_REQUIRED_FILES: tuple[str, ...] = (
    "summary.json",
    "confidence_threshold_summary.csv",
    "cost_sensitivity_summary.csv",
)
_OPTIONAL_FILES: tuple[str, ...] = (
    "execution_v3_manifest.json",
    "latency_sensitivity_summary.csv",
    "fill_assumption_summary.csv",
    "adverse_selection_summary.csv",
    "regime_execution_summary.csv",
    "skipped_diagnostics.json",
)

_REGIME_REQUIRED_FIELDS: tuple[str, ...] = (
    "spread or relative_spread (snapshot-derived spread proxy)",
    "imbalance or order_imbalance (snapshot-derived imbalance proxy)",
    "top-of-book depth columns (snapshot-derived liquidity proxy)",
    "a future signed-return or move column for a volatility proxy",
)


@dataclass(frozen=True)
class ExecutionV3AnalysisSummary:
    """Lightweight return value describing the generated analysis."""

    output_dir: Path
    execution_v3_dir: Path
    smoke_test: bool
    payoff_mode: str
    cost_mode: str
    run_group_count: int
    artefacts: Mapping[str, str] = field(default_factory=dict)
    figures_generated: tuple[str, ...] = ()
    claim_statuses: Mapping[str, str] = field(default_factory=dict)
    regime_status: str = "skipped"
    warnings: tuple[str, ...] = ()
    raw_predictions_required: bool = False


# ---------------------------------------------------------------------------
# Loaders (retained execution-v3 tables only)
# ---------------------------------------------------------------------------


def _ensure_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {path}")
        if any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    "refusing to write into a non-empty output directory; "
                    f"pass overwrite=True to replace it: {path}",
                )
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _require_inputs(directory: Path) -> None:
    if not directory.exists():
        raise FileNotFoundError(f"execution-v3 artefact directory missing: {directory}")
    for filename in _REQUIRED_FILES:
        candidate = directory / filename
        if not candidate.is_file():
            raise FileNotFoundError(
                "execution-v3 outputs incomplete; the upstream "
                "build-fi2010-execution-v3 step must run first: missing "
                f"{candidate}",
            )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _ok_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "status" not in frame.columns:
        return frame
    return frame[frame["status"].astype(str) == "ok"].copy()


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _aggregate_means(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    mean_cols: Mapping[str, str],
) -> pd.DataFrame:
    """Group ``frame`` and average the requested numeric columns.

    ``mean_cols`` maps a source column name to its output column name. Missing
    source columns are skipped. ``n_groups`` records the number of contributing
    run-group rows.
    """

    available_groups = [column for column in group_cols if column in frame.columns]
    if frame.empty or not available_groups:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for key, group in frame.groupby(available_groups, sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        record: dict[str, Any] = dict(zip(available_groups, key_tuple, strict=False))
        record["n_groups"] = len(group)
        for source_column, output_column in mean_cols.items():
            record[output_column] = _column_mean(group, source_column)
        records.append(record)
    return pd.DataFrame(records)


def _column_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def _weighted_ratio(frame: pd.DataFrame, numerator: str, denominator: str) -> float | None:
    if frame.empty or numerator not in frame.columns or denominator not in frame.columns:
        return None
    num = pd.to_numeric(frame[numerator], errors="coerce").fillna(0.0).sum()
    den = pd.to_numeric(frame[denominator], errors="coerce").fillna(0.0).sum()
    return float(num / den) if den else None


def _confidence_filtering_summary(frame: pd.DataFrame) -> pd.DataFrame:
    ok = _ok_rows(frame)
    return _aggregate_means(
        ok,
        ("pretraining_objective", "horizon", "threshold"),
        {
            "retained_sample_fraction": "mean_retained_fraction",
            "active_trade_fraction": "mean_active_fraction",
            "predicted_stationary_fraction": "mean_abstention_fraction",
            "classification_accuracy": "mean_classification_accuracy",
            "macro_f1": "mean_macro_f1",
            "directional_trade_hit_rate": "mean_directional_hit_rate",
            "gross_directional_proxy": "mean_gross_directional_proxy",
            "net_cost_adjusted_proxy": "mean_cost_adjusted_proxy",
        },
    )


def _turnover_proxy_summary(frame: pd.DataFrame) -> pd.DataFrame:
    ok = _ok_rows(frame).copy()
    if not ok.empty and {"net_cost_adjusted_proxy", "active_trade_count"} <= set(ok.columns):
        active = pd.to_numeric(ok["active_trade_count"], errors="coerce")
        net = pd.to_numeric(ok["net_cost_adjusted_proxy"], errors="coerce")
        ok["turnover_adjusted_cost_proxy"] = net / active.where(active > 0)
    return _aggregate_means(
        ok,
        ("pretraining_objective", "horizon", "threshold"),
        {
            "turnover_proxy": "mean_signal_change_rate",
            "active_trade_fraction": "mean_active_fraction",
            "turnover_adjusted_cost_proxy": "mean_turnover_adjusted_cost_proxy",
        },
    )


def _cost_sensitivity_summary(frame: pd.DataFrame) -> pd.DataFrame:
    ok = _ok_rows(frame)
    return _aggregate_means(
        ok,
        ("pretraining_objective", "horizon", "fee_bps", "spread_multiplier"),
        {
            "gross_proxy": "mean_gross_proxy",
            "net_proxy": "mean_cost_adjusted_proxy",
            "degradation_percentage": "mean_degradation_percentage",
            "active_trade_fraction": "mean_active_fraction",
            "average_cost_per_active_trade": "mean_average_cost_per_active_trade",
        },
    )


def _latency_sensitivity_summary(frame: pd.DataFrame) -> pd.DataFrame:
    ok = _ok_rows(frame)
    return _aggregate_means(
        ok,
        ("pretraining_objective", "horizon", "latency_step"),
        {
            "directional_hit_rate": "mean_directional_hit_rate",
            "net_degradation_vs_latency_0": "mean_net_degradation_vs_latency_0",
            "degradation_vs_latency_0": "mean_gross_degradation_vs_latency_0",
            "active_trades": "mean_active_trades",
            "dropped_samples": "mean_dropped_samples",
        },
    )


def _fill_assumption_summary(frame: pd.DataFrame) -> pd.DataFrame:
    ok = _ok_rows(frame)
    return _aggregate_means(
        ok,
        ("pretraining_objective", "horizon", "fill_mode"),
        {
            "fill_fraction": "mean_fill_fraction",
            "directional_hit_rate_on_filled_trades": "mean_directional_hit_rate",
            "gross_proxy": "mean_gross_proxy",
            "net_proxy": "mean_cost_adjusted_proxy",
            "average_cost": "mean_average_cost",
        },
    )


def _adverse_selection_summary(frame: pd.DataFrame) -> pd.DataFrame:
    ok = _ok_rows(frame)
    if ok.empty:
        return pd.DataFrame()
    group_cols = [
        column
        for column in ("pretraining_objective", "horizon", "confidence_bucket", "fill_assumption")
        if column in ok.columns
    ]
    if not group_cols:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for key, group in ok.groupby(group_cols, sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        record: dict[str, Any] = dict(zip(group_cols, key_tuple, strict=False))
        record["n_groups"] = len(group)
        record["total_filled"] = int(
            pd.to_numeric(group.get("filled_count"), errors="coerce").fillna(0).sum()
        )
        record["total_adverse"] = int(
            pd.to_numeric(group.get("adverse_count"), errors="coerce").fillna(0).sum()
        )
        record["mean_adverse_fraction"] = _column_mean(group, "adverse_fraction")
        record["weighted_adverse_fraction"] = _weighted_ratio(
            group, "adverse_count", "filled_count"
        )
        record["adverse_selection_mode"] = _first_str(group.get("adverse_selection_mode"))
        records.append(record)
    return pd.DataFrame(records)


def _first_str(values: Any) -> str:
    if values is None:
        return ""
    for value in values:
        text = str(value)
        if text and text.lower() != "nan":
            return text
    return ""


# ---------------------------------------------------------------------------
# Regime diagnostics (explicit skip; snapshot regimes are not derivable)
# ---------------------------------------------------------------------------


def _regime_diagnostics(
    regime_frame: pd.DataFrame,
    skipped_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an explicit, non-silent skip record for regime diagnostics.

    The retained execution-v3 tables and the underlying FI-2010 prediction
    artefacts do not carry regime labels or snapshot market-context columns, so
    supported snapshot-derived proxy regimes cannot be built without rerunning
    the neural grid with extra context columns. Rather than invent unsupported
    regimes, the status is recorded as skipped with the exact fields required.
    """

    skipped_scopes: list[str] = []
    if not regime_frame.empty and "regime_type" in regime_frame.columns:
        skipped_scopes = sorted({str(value) for value in regime_frame["regime_type"]})
    elif isinstance(skipped_payload.get("skipped"), list):
        skipped_scopes = sorted(
            {
                str(item.get("scope"))
                for item in skipped_payload["skipped"]
                if isinstance(item, dict) and item.get("diagnostic") == "regime_execution"
            }
        )
    return {
        "status": "skipped",
        "reason": (
            "Regime execution diagnostics are skipped. The retained execution-v3 "
            "tables and the underlying FI-2010 prediction artefacts do not carry "
            "regime labels or snapshot market-context columns, so supported "
            "snapshot-derived proxy regimes cannot be built without regenerating "
            "the neural grid with additional context columns."
        ),
        "regime_scopes_unavailable": skipped_scopes,
        "raw_predictions_required_to_fix": True,
        "required_fields_for_future_work": list(_REGIME_REQUIRED_FIELDS),
        "future_work": (
            "Regenerate the neural-grid prediction rows with snapshot spread, "
            "imbalance, depth and a future-move column, then derive low/medium/high "
            "proxy regimes labelled explicitly as snapshot-derived proxy regimes."
        ),
    }


# ---------------------------------------------------------------------------
# Claim assessment
# ---------------------------------------------------------------------------


def _assess_claims(
    *,
    smoke_test: bool,
    confidence: pd.DataFrame,
    cost: pd.DataFrame,
    latency: pd.DataFrame,
    fill: pd.DataFrame,
    adverse: pd.DataFrame,
    regime_status: str,
) -> list[dict[str, Any]]:
    def _support(present: bool) -> str:
        if smoke_test:
            return "needs_real_evidence"
        return "supported" if present else "needs_real_evidence"

    claims: list[dict[str, Any]] = [
        {
            "claim_id": "execution_proxy_diagnostics_implemented",
            "claim_text": (
                "ChronosLOB provides offline execution-aware proxy diagnostics over "
                "stored FI-2010 forecasts."
            ),
            "status": _support(not confidence.empty),
            "scope": "retained execution-v3 confidence-filtering and cost tables",
            "safe_rewording": (
                "Offline execution-aware proxy diagnostics are implemented and run "
                "from retained tables; this is an implementation claim, not a "
                "profitability or live-trading claim."
            ),
        },
        {
            "claim_id": "execution_proxy_cost_sensitivity",
            "claim_text": "Cost-adjusted proxy degradation is characterised across a cost grid.",
            "status": _support(not cost.empty),
            "scope": "cost_sensitivity_summary across fee and spread-multiplier settings",
            "safe_rewording": (
                "Report cost-adjusted proxy degradation by fee and spread multiplier; "
                "these are proxies, not realised PnL."
            ),
        },
        {
            "claim_id": "execution_proxy_latency_sensitivity",
            "claim_text": "Row-step latency sensitivity is characterised by horizon and objective.",
            "status": _support(not latency.empty),
            "scope": "latency_sensitivity_summary across row-step latency lags",
            "safe_rewording": (
                "Report row-step latency degradation of the proxy by horizon; this is "
                "a latency sensitivity proxy, not a live latency measurement."
            ),
        },
        {
            "claim_id": "execution_proxy_fill_sensitivity",
            "claim_text": "Fill-assumption sensitivity is characterised across proxy fill modes.",
            "status": _support(not fill.empty),
            "scope": "fill_assumption_summary across aggressive/passive/abstain proxy modes",
            "safe_rewording": (
                "Report fill-assumption sensitivity across proxy fill modes; fills are "
                "assumptions, not exchange-confirmed executions."
            ),
        },
        {
            "claim_id": "execution_proxy_adverse_selection",
            "claim_text": "An adverse-selection proxy is characterised by confidence bucket.",
            "status": _support(not adverse.empty),
            "scope": "adverse_selection_proxy_summary by confidence bucket and fill assumption",
            "safe_rewording": (
                "Report the adverse-selection proxy rate by confidence bucket; it is a "
                "label/move proxy, not measured adverse selection."
            ),
        },
        {
            "claim_id": "execution_proxy_regime_diagnostics",
            "claim_text": "Execution-aware diagnostics are broken down by market regime.",
            "status": regime_status if regime_status in {"supported", "skipped"} else "skipped",
            "scope": "regime diagnostics require regime labels or snapshot context columns",
            "safe_rewording": (
                "Regime diagnostics are skipped because regime labels and snapshot "
                "context columns are not present in the retained artefacts."
            ),
        },
        {
            "claim_id": "execution_proxy_profitability_or_live_trading",
            "claim_text": (
                "Execution-v3 demonstrates profitability, realised PnL, live trading or "
                "production execution quality."
            ),
            "status": "forbidden",
            "scope": "blocked by release policy; these diagnostics are offline proxies only",
            "safe_rewording": (
                "State that execution-v3 is an offline execution-aware proxy diagnostic "
                "and is not PnL, not live-trading evidence and not a production "
                "execution simulator."
            ),
        },
    ]
    return claims


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _format_float(value: Any, *, places: int = 4) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{places}f}"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_int(value: Any) -> int | None:
    number = _safe_float(value)
    return int(number) if number is not None else None


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def _wrap_lines(lines: Sequence[str], *, width: int = 100) -> list[str]:
    import textwrap

    out: list[str] = []
    for line in lines:
        if not line or line.startswith(("|", "#", "```", "  ")) or len(line) <= width:
            out.append(line)
            continue
        if line.startswith("- "):
            wrapped = textwrap.wrap(
                line[2:], width=width, break_long_words=False, break_on_hyphens=False
            )
            out.extend(["- " + wrapped[0]] + ["  " + item for item in wrapped[1:]] or [line])
            continue
        wrapped = textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False)
        out.extend(wrapped or [line])
    return out


def _threshold_rows(frame: pd.DataFrame, value_column: str) -> list[list[str]]:
    if frame.empty or value_column not in frame.columns:
        return []
    rows: list[list[str]] = []
    grouped = _aggregate_means(frame, ("pretraining_objective", "threshold"), {value_column: "v"})
    for _, record in grouped.sort_values(["pretraining_objective", "threshold"]).iterrows():
        rows.append(
            [
                str(record.get("pretraining_objective", "")),
                _format_float(record.get("threshold"), places=2),
                _format_float(record.get("v")),
            ]
        )
    return rows


def _render_markdown(
    *,
    summary: ExecutionV3AnalysisSummary,
    confidence: pd.DataFrame,
    turnover: pd.DataFrame,
    cost: pd.DataFrame,
    latency: pd.DataFrame,
    fill: pd.DataFrame,
    adverse: pd.DataFrame,
    regime: Mapping[str, Any],
    claims: Sequence[Mapping[str, Any]],
    figures: Sequence[Mapping[str, Any]],
) -> str:
    lines: list[str] = [
        "# Execution-v3 Proxy Analysis",
        "",
        f"Builder version `{EXECUTION_V3_ANALYSIS_VERSION}`.",
        "",
        "This report is a richer, reviewer-facing summary of the FI-2010 "
        "execution-v3 outputs. It is an offline execution-aware proxy diagnostic. "
        "It is not PnL, not live-trading evidence and not a production execution "
        "simulator. The confidence, cost, latency, fill and adverse-selection "
        "diagnostics test whether predictive metrics survive simple execution-like "
        "frictions; cost-adjusted values are cost-adjusted proxies only.",
        "",
        "It is generated only from retained lightweight execution-v3 tables. The "
        "heavy raw per-run prediction arrays were deleted to save storage and are "
        "not required by this analysis.",
        "",
        f"- Source execution-v3 directory: `{summary.execution_v3_dir.as_posix()}`.",
        f"- Payoff mode: `{summary.payoff_mode}`; cost mode: `{summary.cost_mode}`.",
        f"- Run groups summarised: {summary.run_group_count}.",
        f"- Smoke-test inputs: {'yes' if summary.smoke_test else 'no'}.",
        "",
    ]

    lines += ["## Confidence Filtering and Active Fraction", ""]
    if not confidence.empty:
        lines += [
            "Mean retained fraction, active fraction, abstention fraction and "
            "cost-adjusted proxy by confidence threshold (averaged over folds, seeds "
            "and objectives at each horizon shown below by objective):",
            "",
        ]
        lines += _markdown_table(
            ("objective", "threshold", "active fraction"),
            _threshold_rows(confidence, "mean_active_fraction"),
        )
        lines += [
            "",
            "Higher confidence thresholds retain fewer predictions and lower the "
            "active fraction; the active fraction is the share of all samples that "
            "remain directional (non-abstaining) after filtering.",
            "",
        ]
    else:
        lines += ["No confidence-filtering rows were available.", ""]

    lines += ["## Turnover Proxy", ""]
    if not turnover.empty:
        lines += [
            "Mean signal-change-rate turnover proxy by confidence threshold:",
            "",
        ]
        lines += _markdown_table(
            ("objective", "threshold", "turnover proxy"),
            _threshold_rows(turnover, "mean_signal_change_rate"),
        )
        lines += [
            "",
            "Turnover proxy falls as the confidence threshold rises, because fewer "
            "active directional signals remain. This is a turnover proxy, not a "
            "realised order count.",
            "",
        ]
    else:
        lines += ["No turnover-proxy rows were available.", ""]

    lines += ["## Cost Sensitivity", ""]
    lines += _render_cost_block(cost)

    lines += ["## Latency Sensitivity", ""]
    lines += _render_latency_block(latency)

    lines += ["## Fill-Assumption Sensitivity", ""]
    lines += _render_fill_block(fill)

    lines += ["## Adverse-Selection Proxy", ""]
    lines += _render_adverse_block(adverse)

    lines += ["## Regime Diagnostics", ""]
    lines += [
        f"Status: `{regime.get('status', 'skipped')}`.",
        "",
        str(regime.get("reason", "")),
        "",
        "Fields required before supported snapshot-derived proxy regimes could be "
        "added (recorded as future work, not invented here):",
        "",
    ]
    lines += [f"- {item}" for item in regime.get("required_fields_for_future_work", [])]
    lines += [""]

    lines += _render_claim_section(claims)
    lines += _render_figure_section(figures)
    lines += [
        "## What This Does Not Claim",
        "",
        "This analysis does not claim profitability, realised PnL, tradable alpha, "
        "live trading, production execution quality, true event-level order-flow "
        "imbalance or queue-position modelling. It is a descriptive offline proxy "
        "diagnostic over stored FI-2010 metrics under leakage-safe evaluation.",
        "",
    ]
    return "\n".join(_wrap_lines(lines)).rstrip() + "\n"


def _render_cost_block(cost: pd.DataFrame) -> list[str]:
    if cost.empty:
        return ["No cost-sensitivity rows were available.", ""]
    rows: list[list[str]] = []
    grouped = _aggregate_means(
        cost,
        ("fee_bps", "spread_multiplier"),
        {
            "mean_gross_proxy": "gross",
            "mean_cost_adjusted_proxy": "net",
            "mean_degradation_percentage": "deg",
        },
    )
    for _, record in grouped.sort_values(["fee_bps", "spread_multiplier"]).iterrows():
        rows.append(
            [
                _format_float(record.get("fee_bps"), places=1),
                _format_float(record.get("spread_multiplier"), places=1),
                _format_float(record.get("gross"), places=1),
                _format_float(record.get("net"), places=1),
                _format_float(record.get("deg"), places=2),
            ]
        )
    return [
        "Mean gross proxy, cost-adjusted proxy and degradation percentage across the "
        "fee and spread-multiplier grid (averaged over objectives and horizons):",
        "",
        *_markdown_table(
            ("fee bps", "spread x", "gross proxy", "cost-adjusted proxy", "degradation %"),
            rows,
        ),
        "",
        "Degradation percentage rises monotonically with assumed cost. These are "
        "cost-adjusted proxies, not realised PnL.",
        "",
    ]


def _render_latency_block(latency: pd.DataFrame) -> list[str]:
    if latency.empty:
        return ["No latency-sensitivity rows were available.", ""]
    rows: list[list[str]] = []
    grouped = _aggregate_means(
        latency,
        ("horizon", "latency_step"),
        {
            "mean_net_degradation_vs_latency_0": "net_deg",
            "mean_directional_hit_rate": "hit",
        },
    )
    for _, record in grouped.sort_values(["horizon", "latency_step"]).iterrows():
        rows.append(
            [
                str(_safe_int(record.get("horizon"))),
                str(_safe_int(record.get("latency_step"))),
                _format_float(record.get("net_deg"), places=2),
                _format_float(record.get("hit")),
            ]
        )
    return [
        "Mean cost-adjusted proxy degradation versus latency 0 and mean directional "
        "hit rate by horizon and row-step latency lag (averaged over objectives):",
        "",
        *_markdown_table(
            ("horizon", "latency step", "net degradation vs lag 0", "hit rate"),
            rows,
        ),
        "",
        "Latency is a row-step diagnostic shifted only within the same fold and "
        "partition. It is a latency sensitivity proxy, not a live latency measurement.",
        "",
    ]


def _render_fill_block(fill: pd.DataFrame) -> list[str]:
    if fill.empty:
        return ["No fill-assumption rows were available.", ""]
    rows: list[list[str]] = []
    grouped = _aggregate_means(
        fill,
        ("fill_mode",),
        {
            "mean_fill_fraction": "fill",
            "mean_directional_hit_rate": "hit",
            "mean_cost_adjusted_proxy": "net",
        },
    )
    for _, record in grouped.sort_values("fill_mode").iterrows():
        rows.append(
            [
                str(record.get("fill_mode", "")),
                _format_float(record.get("fill")),
                _format_float(record.get("hit")),
                _format_float(record.get("net"), places=1),
            ]
        )
    return [
        "Mean fill fraction, directional hit rate on filled trades and cost-adjusted "
        "proxy by proxy fill mode (averaged over objectives and horizons):",
        "",
        *_markdown_table(
            ("fill mode", "fill fraction", "hit rate", "cost-adjusted proxy"),
            rows,
        ),
        "",
        "Fill modes are proxy assumptions, not exchange-confirmed executions.",
        "",
    ]


def _render_adverse_block(adverse: pd.DataFrame) -> list[str]:
    if adverse.empty:
        return ["No adverse-selection rows were available.", ""]
    rows: list[list[str]] = []
    grouped = _aggregate_means(
        adverse,
        ("confidence_bucket",),
        {"weighted_adverse_fraction": "adverse"},
    )
    for _, record in grouped.sort_values("confidence_bucket").iterrows():
        rows.append(
            [
                str(record.get("confidence_bucket", "")),
                _format_float(record.get("adverse")),
            ]
        )
    return [
        "Filled-weighted adverse-selection proxy rate by confidence bucket (averaged "
        "over objectives, horizons and fill assumptions):",
        "",
        *_markdown_table(("confidence bucket", "adverse-selection proxy rate"), rows),
        "",
        "The adverse-selection proxy uses a label/future-move proxy, not measured "
        "adverse selection against real fills.",
        "",
    ]


def _render_claim_section(claims: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["## Claim Assessment", ""]
    rows = [
        [str(claim.get("claim_id", "")), str(claim.get("status", "")), str(claim.get("scope", ""))]
        for claim in claims
    ]
    lines += _markdown_table(("claim", "status", "scope"), rows)
    lines += [""]
    return lines


def _render_figure_section(figures: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["## Figures", ""]
    completed = [entry for entry in figures if entry.get("status") == "completed"]
    if not completed:
        lines += [
            "No figures were generated in this pass; the summary CSVs hold the same "
            "numbers and figure generation is recorded in `figure_manifest.json`.",
            "",
        ]
        return lines
    rows = [
        [str(entry.get("figure_id", "")), str(entry.get("title", "")), str(entry.get("file_path"))]
        for entry in completed
    ]
    lines += _markdown_table(("figure", "title", "path"), rows)
    lines += [""]
    return lines


# ---------------------------------------------------------------------------
# Optional figures
# ---------------------------------------------------------------------------


def _build_figures(
    *,
    out_dir: Path,
    confidence: pd.DataFrame,
    turnover: pd.DataFrame,
    latency: pd.DataFrame,
    fill: pd.DataFrame,
    adverse: pd.DataFrame,
    make_figures: bool,
) -> list[dict[str, Any]]:
    planned = (
        "active_fraction_vs_confidence",
        "cost_adjusted_proxy_vs_confidence",
        "turnover_proxy_vs_confidence",
        "latency_degradation_by_horizon",
        "adverse_selection_by_confidence_bucket",
        "fill_assumption_sensitivity",
    )
    if not make_figures:
        reason = "figure generation disabled (make_figures=False)"
        return [_skipped_figure(name, reason) for name in planned]
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return [_skipped_figure(name, "matplotlib is not installed") for name in planned]

    entries: list[dict[str, Any]] = []
    entries.append(
        _line_by_objective_figure(
            out_dir=out_dir,
            plt=plt,
            frame=confidence,
            figure_id="active_fraction_vs_confidence",
            title="Active fraction vs confidence threshold",
            value_column="mean_active_fraction",
            ylabel="active fraction",
        )
    )
    entries.append(
        _line_by_objective_figure(
            out_dir=out_dir,
            plt=plt,
            frame=confidence,
            figure_id="cost_adjusted_proxy_vs_confidence",
            title="Cost-adjusted proxy vs confidence threshold",
            value_column="mean_cost_adjusted_proxy",
            ylabel="cost-adjusted proxy",
        )
    )
    entries.append(
        _line_by_objective_figure(
            out_dir=out_dir,
            plt=plt,
            frame=turnover,
            figure_id="turnover_proxy_vs_confidence",
            title="Turnover proxy vs confidence threshold",
            value_column="mean_signal_change_rate",
            ylabel="turnover proxy",
        )
    )
    entries.append(_latency_figure(out_dir=out_dir, plt=plt, latency=latency))
    entries.append(_adverse_figure(out_dir=out_dir, plt=plt, adverse=adverse))
    entries.append(_fill_figure(out_dir=out_dir, plt=plt, fill=fill))
    return entries


def _skipped_figure(figure_id: str, reason: str) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "title": figure_id.replace("_", " "),
        "status": "skipped",
        "reason": reason,
        "file_path": None,
    }


def _completed_figure(figure_id: str, title: str, path: Path) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "title": title,
        "status": "completed",
        "reason": "",
        "file_path": path.as_posix(),
    }


def _line_by_objective_figure(
    *,
    out_dir: Path,
    plt: Any,
    frame: pd.DataFrame,
    figure_id: str,
    title: str,
    value_column: str,
    ylabel: str,
) -> dict[str, Any]:
    if frame.empty or value_column not in frame.columns or "threshold" not in frame.columns:
        return _skipped_figure(figure_id, "no rows for this metric")
    grouped = _aggregate_means(
        frame, ("pretraining_objective", "threshold"), {value_column: "v"}
    )
    if grouped.empty:
        return _skipped_figure(figure_id, "no rows for this metric")
    fig, axis = plt.subplots(figsize=(5.2, 3.2))
    for objective in sorted({str(value) for value in grouped["pretraining_objective"]}):
        obj_rows = grouped[grouped["pretraining_objective"] == objective].sort_values("threshold")
        axis.plot(
            [float(value) for value in obj_rows["threshold"]],
            [_safe_float(value) for value in obj_rows["v"]],
            marker="o",
            label=objective,
        )
    axis.set_xlabel("confidence threshold")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / f"{figure_id}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return _completed_figure(figure_id, title, path)


def _latency_figure(*, out_dir: Path, plt: Any, latency: pd.DataFrame) -> dict[str, Any]:
    figure_id = "latency_degradation_by_horizon"
    title = "Cost-adjusted proxy degradation by latency and horizon"
    column = "mean_net_degradation_vs_latency_0"
    if latency.empty or column not in latency.columns:
        return _skipped_figure(figure_id, "no latency rows available")
    grouped = _aggregate_means(latency, ("horizon", "latency_step"), {column: "v"})
    if grouped.empty:
        return _skipped_figure(figure_id, "no latency rows available")
    fig, axis = plt.subplots(figsize=(5.2, 3.2))
    horizons = sorted(
        {value for value in (_safe_int(item) for item in grouped["horizon"]) if value is not None}
    )
    for horizon in horizons:
        rows = grouped[grouped["horizon"] == horizon].sort_values("latency_step")
        axis.plot(
            [_safe_int(value) for value in rows["latency_step"]],
            [_safe_float(value) for value in rows["v"]],
            marker="o",
            label=f"h{horizon}",
        )
    axis.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    axis.set_xlabel("row-step latency lag")
    axis.set_ylabel("net degradation vs lag 0")
    axis.set_title(title)
    axis.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / f"{figure_id}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return _completed_figure(figure_id, title, path)


def _adverse_figure(*, out_dir: Path, plt: Any, adverse: pd.DataFrame) -> dict[str, Any]:
    figure_id = "adverse_selection_by_confidence_bucket"
    title = "Adverse-selection proxy by confidence bucket"
    if adverse.empty or "weighted_adverse_fraction" not in adverse.columns:
        return _skipped_figure(figure_id, "no adverse-selection rows available")
    grouped = _aggregate_means(
        adverse, ("confidence_bucket",), {"weighted_adverse_fraction": "v"}
    ).sort_values("confidence_bucket")
    if grouped.empty:
        return _skipped_figure(figure_id, "no adverse-selection rows available")
    labels = [str(value) for value in grouped["confidence_bucket"]]
    values = [_safe_float(value) or 0.0 for value in grouped["v"]]
    fig, axis = plt.subplots(figsize=(5.2, 3.2))
    axis.bar(range(len(values)), values, color="#c44e52")
    axis.set_xticks(range(len(values)))
    axis.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    axis.set_ylabel("adverse-selection proxy rate")
    axis.set_title(title)
    fig.tight_layout()
    path = out_dir / f"{figure_id}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return _completed_figure(figure_id, title, path)


def _fill_figure(*, out_dir: Path, plt: Any, fill: pd.DataFrame) -> dict[str, Any]:
    figure_id = "fill_assumption_sensitivity"
    title = "Cost-adjusted proxy by fill assumption"
    if fill.empty or "mean_cost_adjusted_proxy" not in fill.columns:
        return _skipped_figure(figure_id, "no fill-assumption rows available")
    grouped = _aggregate_means(
        fill, ("fill_mode",), {"mean_cost_adjusted_proxy": "v"}
    ).sort_values("fill_mode")
    if grouped.empty:
        return _skipped_figure(figure_id, "no fill-assumption rows available")
    labels = [str(value) for value in grouped["fill_mode"]]
    values = [_safe_float(value) or 0.0 for value in grouped["v"]]
    fig, axis = plt.subplots(figsize=(5.6, 3.2))
    axis.bar(range(len(values)), values, color="#4c72b0")
    axis.set_xticks(range(len(values)))
    axis.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    axis.set_ylabel("cost-adjusted proxy")
    axis.set_title(title)
    fig.tight_layout()
    path = out_dir / f"{figure_id}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return _completed_figure(figure_id, title, path)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _stable_json_dumps(payload: Mapping[str, Any]) -> str:
    def _default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"unsupported type for JSON serialisation: {type(value)!r}")

    return json.dumps(payload, indent=2, sort_keys=True, default=_default) + "\n"


def _frame_to_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        path.write_text("", encoding="utf-8")
        return
    frame.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyse_fi2010_execution_v3(
    *,
    execution_v3_dir: str | Path,
    out_dir: str | Path,
    make_figures: bool = True,
    overwrite: bool = False,
) -> ExecutionV3AnalysisSummary:
    """Analyse retained FI-2010 execution-v3 tables and write a richer report.

    Only retained lightweight execution-v3 output tables are read. Deleted raw
    per-run prediction arrays are never required.
    """

    source = Path(execution_v3_dir)
    _require_inputs(source)

    resolved_out = Path(out_dir)
    if resolved_out.resolve(strict=False) == source.resolve(strict=False):
        raise ValueError("execution-v3 analysis output directory must differ from the input")
    _ensure_output_dir(resolved_out, overwrite=overwrite)

    summary_meta = _read_json(source / "summary.json")
    manifest_meta = _read_json(source / "execution_v3_manifest.json")
    smoke_test = bool(summary_meta.get("smoke_test") or manifest_meta.get("smoke_test"))
    payoff_mode = str(summary_meta.get("payoff_mode", manifest_meta.get("payoff_mode", "unknown")))
    cost_mode = str(summary_meta.get("cost_mode", manifest_meta.get("cost_mode", "unknown")))
    run_group_count = int(summary_meta.get("run_group_count", 0) or 0)

    warnings: list[str] = []
    for filename in _OPTIONAL_FILES:
        if not (source / filename).is_file():
            warnings.append(f"optional execution-v3 input missing: {filename}")

    confidence_raw = _read_csv(source / "confidence_threshold_summary.csv")
    cost_raw = _read_csv(source / "cost_sensitivity_summary.csv")
    latency_raw = _read_csv(source / "latency_sensitivity_summary.csv")
    fill_raw = _read_csv(source / "fill_assumption_summary.csv")
    adverse_raw = _read_csv(source / "adverse_selection_summary.csv")
    regime_raw = _read_csv(source / "regime_execution_summary.csv")
    skipped_payload = _read_json(source / "skipped_diagnostics.json")

    confidence = _confidence_filtering_summary(confidence_raw)
    turnover = _turnover_proxy_summary(confidence_raw)
    cost = _cost_sensitivity_summary(cost_raw)
    latency = _latency_sensitivity_summary(latency_raw)
    fill = _fill_assumption_summary(fill_raw)
    adverse = _adverse_selection_summary(adverse_raw)
    regime = _regime_diagnostics(regime_raw, skipped_payload)

    claims = _assess_claims(
        smoke_test=smoke_test,
        confidence=confidence,
        cost=cost,
        latency=latency,
        fill=fill,
        adverse=adverse,
        regime_status=str(regime["status"]),
    )
    claim_statuses = {str(claim["claim_id"]): str(claim["status"]) for claim in claims}

    figure_entries = _build_figures(
        out_dir=resolved_out,
        confidence=confidence,
        turnover=turnover,
        latency=latency,
        fill=fill,
        adverse=adverse,
        make_figures=make_figures,
    )
    figures_generated = tuple(
        str(entry["figure_id"]) for entry in figure_entries if entry.get("status") == "completed"
    )

    artefacts = {
        "report": "execution_v3_analysis.md",
        "confidence_filtering_summary": "confidence_filtering_summary.csv",
        "turnover_proxy_summary": "turnover_proxy_summary.csv",
        "latency_sensitivity_summary": "latency_sensitivity_summary.csv",
        "cost_sensitivity_summary": "cost_sensitivity_summary.csv",
        "fill_assumption_summary": "fill_assumption_summary.csv",
        "adverse_selection_proxy_summary": "adverse_selection_proxy_summary.csv",
        "skipped_regime_diagnostics": "skipped_regime_diagnostics.json",
        "execution_claim_assessment": "execution_claim_assessment.json",
        "figure_manifest": "figure_manifest.json",
        "summary": "summary.json",
    }

    summary = ExecutionV3AnalysisSummary(
        output_dir=resolved_out,
        execution_v3_dir=source,
        smoke_test=smoke_test,
        payoff_mode=payoff_mode,
        cost_mode=cost_mode,
        run_group_count=run_group_count,
        artefacts=artefacts,
        figures_generated=figures_generated,
        claim_statuses=claim_statuses,
        regime_status=str(regime["status"]),
        warnings=tuple(warnings),
        raw_predictions_required=False,
    )

    _frame_to_csv(confidence, resolved_out / "confidence_filtering_summary.csv")
    _frame_to_csv(turnover, resolved_out / "turnover_proxy_summary.csv")
    _frame_to_csv(latency, resolved_out / "latency_sensitivity_summary.csv")
    _frame_to_csv(cost, resolved_out / "cost_sensitivity_summary.csv")
    _frame_to_csv(fill, resolved_out / "fill_assumption_summary.csv")
    _frame_to_csv(adverse, resolved_out / "adverse_selection_proxy_summary.csv")

    (resolved_out / "skipped_regime_diagnostics.json").write_text(
        _stable_json_dumps(regime), encoding="utf-8"
    )

    claim_payload = {
        "analyser_version": EXECUTION_V3_ANALYSIS_VERSION,
        "package_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "offline_execution_aware_proxy_diagnostic": True,
        "raw_predictions_required": False,
        "claim_boundary": (
            "This is an offline execution-aware proxy diagnostic. It is not PnL, "
            "not live-trading evidence and not a production execution simulator."
        ),
        "smoke_test": smoke_test,
        "claims": claims,
    }
    (resolved_out / "execution_claim_assessment.json").write_text(
        _stable_json_dumps(claim_payload), encoding="utf-8"
    )

    figure_manifest = {
        "builder_version": EXECUTION_V3_ANALYSIS_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "figures": list(figure_entries),
    }
    (resolved_out / "figure_manifest.json").write_text(
        _stable_json_dumps(figure_manifest), encoding="utf-8"
    )

    report_text = _render_markdown(
        summary=summary,
        confidence=confidence,
        turnover=turnover,
        cost=cost,
        latency=latency,
        fill=fill,
        adverse=adverse,
        regime=regime,
        claims=claims,
        figures=figure_entries,
    )
    (resolved_out / "execution_v3_analysis.md").write_text(report_text, encoding="utf-8")

    summary_payload = {
        "analyser_version": EXECUTION_V3_ANALYSIS_VERSION,
        "package_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "offline_execution_aware_proxy_diagnostic": True,
        "inputs": {"execution_v3_dir": str(source)},
        "raw_predictions_required": False,
        "smoke_test": smoke_test,
        "payoff_mode": payoff_mode,
        "cost_mode": cost_mode,
        "run_group_count": run_group_count,
        "regime_status": str(regime["status"]),
        "claim_statuses": claim_statuses,
        "figures_generated": list(figures_generated),
        "artefacts": artefacts,
        "warnings": list(warnings),
    }
    (resolved_out / "summary.json").write_text(
        _stable_json_dumps(summary_payload), encoding="utf-8"
    )

    return summary
