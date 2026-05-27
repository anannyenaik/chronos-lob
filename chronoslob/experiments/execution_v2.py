"""Execution-Aware Evaluation v2 for FI-2010.

This layer makes the forecasting-versus-tradability gap explicit. It
consumes the lightweight artefacts already produced by the multi-fold
classical runner, the multi-fold neural runner and the brutal ablation
layer, and re-frames them as a focused set of execution-aware proxy
diagnostics:

* cost and latency sensitivity surfaces;
* confidence-threshold coverage and hit-rate proxy curves;
* a turnover proxy view;
* an adverse-selection proxy derived from latency-induced signal decay;
* a fill-assumption proxy derived from the eligible-to-trade share;
* a degradation summary that contrasts a statistical metric with an
  execution-aware proxy metric under a stressed cost and latency
  scenario.

It does not retrain any model, does not require full prediction rows or
checkpoints, and writes only small aggregate artefacts. Every execution
number is a simplified proxy under stated assumptions. It is not a
backtest, not a live-trading simulation, and it carries no profitability
or tradability claim. Neural runs ship no stored execution proxy rows, so
their execution-aware diagnostics are recorded as explicit skips rather
than fabricated.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from chronoslob import __version__
from chronoslob.experiments.manifests import sha256_file, stable_json_dumps
from chronoslob.training.experiment import get_git_commit

__all__ = [
    "ADVERSE_SELECTION_SUMMARY_COLUMNS",
    "CONFIDENCE_THRESHOLD_SUMMARY_COLUMNS",
    "COST_LATENCY_SURFACE_COLUMNS",
    "DEGRADATION_SUMMARY_COLUMNS",
    "EXECUTION_V2_RESULT_COLUMNS",
    "FI2010_EXECUTION_V2_VERSION",
    "FILL_ASSUMPTION_LABEL",
    "FILL_ASSUMPTION_SUMMARY_COLUMNS",
    "TURNOVER_SUMMARY_COLUMNS",
    "ExecutionV2Summary",
    "run_fi2010_execution_v2",
]

FI2010_EXECUTION_V2_VERSION = "phase-h/fi2010-execution-v2/v1"

# The fill model assumed by the stored proxy: every eligible directional
# signal is assumed filled at the mid price with no queue position.
FILL_ASSUMPTION_LABEL = "full_fill_at_mid_no_queue"

STATISTICAL_METRIC_NAME = "test_macro_f1"
BASE_PROXY_METRIC_NAME = "gross_signal_return_proxy"
EXEC_PROXY_METRIC_NAME = "net_signal_return_proxy"

# Canonical numeric columns carried by a stored execution-sensitivity row.
_CANONICAL_METRIC_COLUMNS: tuple[str, ...] = (
    "eligible_predictions",
    "trade_count_proxy",
    "turnover_proxy",
    "gross_signal_return_proxy",
    "cost_proxy",
    "net_signal_return_proxy",
    "hit_rate_proxy",
)
_SCENARIO_COLUMNS: tuple[str, ...] = (
    "confidence_threshold",
    "cost_bps",
    "latency_steps",
)

EXECUTION_V2_RESULT_COLUMNS: tuple[str, ...] = (
    "model_name",
    "source",
    "fold_id",
    "split",
    "confidence_threshold",
    "cost_bps",
    "latency_steps",
    "eligible_predictions",
    "coverage",
    "trade_count_proxy",
    "turnover_proxy",
    "gross_signal_return_proxy",
    "cost_proxy",
    "net_signal_return_proxy",
    "hit_rate_proxy",
    "adverse_selection_proxy",
    "fill_assumption",
    "fill_assumption_proxy",
    "status",
    "skip_reason",
)

COST_LATENCY_SURFACE_COLUMNS: tuple[str, ...] = (
    "model_name",
    "split",
    "confidence_threshold",
    "cost_bps",
    "latency_steps",
    "fold_count",
    "coverage_mean",
    "eligible_predictions_mean",
    "turnover_proxy_mean",
    "gross_signal_return_proxy_mean",
    "cost_proxy_mean",
    "net_signal_return_proxy_mean",
    "net_signal_return_proxy_std",
    "hit_rate_proxy_mean",
    "status",
)

CONFIDENCE_THRESHOLD_SUMMARY_COLUMNS: tuple[str, ...] = (
    "model_name",
    "split",
    "confidence_threshold",
    "cost_bps",
    "latency_steps",
    "fold_count",
    "coverage_mean",
    "eligible_predictions_mean",
    "trade_count_proxy_mean",
    "hit_rate_proxy_mean",
    "coverage_delta_vs_base",
    "hit_rate_proxy_delta_vs_base",
    "status",
)

TURNOVER_SUMMARY_COLUMNS: tuple[str, ...] = (
    "model_name",
    "split",
    "confidence_threshold",
    "cost_bps",
    "latency_steps",
    "fold_count",
    "turnover_proxy_mean",
    "turnover_proxy_std",
    "trade_count_proxy_mean",
    "coverage_mean",
    "status",
)

ADVERSE_SELECTION_SUMMARY_COLUMNS: tuple[str, ...] = (
    "model_name",
    "source",
    "split",
    "confidence_threshold",
    "cost_bps",
    "latency_steps",
    "base_latency_steps",
    "fold_count",
    "gross_signal_return_proxy_mean",
    "adverse_selection_proxy_mean",
    "adverse_selection_proxy_std",
    "status",
    "skip_reason",
)

FILL_ASSUMPTION_SUMMARY_COLUMNS: tuple[str, ...] = (
    "model_name",
    "source",
    "split",
    "confidence_threshold",
    "cost_bps",
    "latency_steps",
    "fold_count",
    "fill_assumption",
    "eligible_predictions_mean",
    "trade_count_proxy_mean",
    "fill_assumption_proxy_mean",
    "fill_assumption_proxy_std",
    "status",
    "skip_reason",
)

DEGRADATION_SUMMARY_COLUMNS: tuple[str, ...] = (
    "model_name",
    "source",
    "fold_id",
    "statistical_metric",
    "statistical_value",
    "base_proxy_metric",
    "base_proxy_value",
    "exec_proxy_metric",
    "exec_proxy_value",
    "absolute_degradation_proxy",
    "relative_degradation_proxy",
    "status",
    "skip_reason",
)

_CLAIM_BOUNDARIES: tuple[str, ...] = (
    "These are simplified proxy diagnostics, not a backtest.",
    "This is not a live-trading simulation and models no market impact.",
    "No profitability or live tradability claim is made.",
    "No foundation-model, leading-benchmark or self-supervised result is claimed.",
    "Neural superiority over the classical baseline is not asserted.",
)

_FOLD_RE = re.compile(r"fold_(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Summary container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionV2Summary:
    """Lightweight return value from the execution v2 entry point."""

    output_dir: Path
    classical_dir: Path | None
    neural_dir: Path | None
    ablations_dir: Path | None
    classical_models: tuple[str, ...]
    neural_models: tuple[str, ...]
    folds: tuple[str, ...]
    result_row_count: int
    ok_row_count: int
    skipped_row_count: int
    diagnostics_produced: tuple[str, ...]
    diagnostics_skipped: tuple[str, ...]
    full_predictions_required: bool
    checkpoints_required: bool
    artefacts: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _ensure_output_dir(path: Path, *, overwrite: bool) -> None:
    import shutil

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


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _mean(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    value = float(numeric.mean())
    return value if math.isfinite(value) else None


def _std(series: pd.Series, *, ddof: int = 0) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    if len(numeric) <= ddof:
        return 0.0
    value = float(numeric.std(ddof=ddof))
    return value if math.isfinite(value) else None


def _normalise_fold_token(value: Any) -> str:
    text = str(value).strip().casefold()
    if not text:
        return "aggregate"
    if text == "aggregate":
        return "aggregate"
    match = _FOLD_RE.fullmatch(text)
    if match is not None:
        return f"fold_{int(match.group(1))}"
    if text.isdigit():
        return f"fold_{int(text)}"
    return text


def _parse_float_filter(value: Sequence[float] | str | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.casefold() == "all":
            return None
        tokens: Sequence[Any] = [token.strip() for token in text.split(",")]
    else:
        tokens = list(value)
    cleaned: list[float] = []
    for token in tokens:
        if str(token).strip() == "":
            continue
        numeric = float(token)
        if numeric not in cleaned:
            cleaned.append(numeric)
    return tuple(cleaned) if cleaned else None


def _parse_int_filter(value: Sequence[int] | str | None) -> tuple[int, ...] | None:
    floats = _parse_float_filter(value)
    if floats is None:
        return None
    return tuple(int(item) for item in floats)


def _parse_model_filter(value: Sequence[str] | str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.casefold() == "all":
            return None
        tokens: Sequence[str] = [token.strip() for token in text.split(",")]
    else:
        tokens = list(value)
    cleaned: list[str] = []
    for token in tokens:
        name = str(token).strip()
        if name and name not in cleaned:
            cleaned.append(name)
    return tuple(cleaned) if cleaned else None


def _write_csv(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    columns: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    projected = [{column: row.get(column) for column in columns} for row in rows]
    if projected:
        frame = pd.DataFrame(projected, columns=list(columns))
    else:
        frame = pd.DataFrame(columns=list(columns))
    frame.to_csv(path, index=False)


def _build_environment_payload() -> dict[str, str]:
    import platform

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_version": __version__,
    }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_classical_execution_rows(classical_dir: Path | None) -> tuple[pd.DataFrame, list[str]]:
    """Load per-fold classical execution proxy rows in canonical form.

    Per-fold ``execution_sensitivity.csv`` files are preferred. When they
    are absent the aggregate ``execution_summary.csv`` is used with a
    synthetic ``aggregate`` fold id. Missing optional metric columns are
    tolerated and filled with ``NaN`` (or derived where unambiguous).
    """
    warnings: list[str] = []
    if classical_dir is None or not classical_dir.is_dir():
        return _empty_execution_frame(), [
            "classical artefact directory was not provided or does not exist"
        ]

    frames: list[pd.DataFrame] = []
    folds_dir = classical_dir / "folds"
    if folds_dir.is_dir():
        for fold_path in sorted(folds_dir.glob("fold_*/execution_sensitivity.csv")):
            sub = pd.read_csv(fold_path)
            if sub.empty:
                continue
            sub = sub.assign(fold_id=_normalise_fold_token(fold_path.parent.name))
            frames.append(sub)

    if not frames:
        summary_path = classical_dir / "execution_summary.csv"
        if summary_path.is_file():
            summary = pd.read_csv(summary_path)
            if not summary.empty:
                summary = summary.assign(fold_id="aggregate")
                frames.append(summary)
                warnings.append(
                    "per-fold execution_sensitivity.csv files were absent; used "
                    "the aggregate execution_summary.csv with fold_id=aggregate"
                )

    if not frames:
        return _empty_execution_frame(), [
            "no classical execution proxy rows were found under the classical directory"
        ]

    combined = pd.concat(frames, ignore_index=True)
    return _canonicalise_execution_frame(combined), warnings


def _empty_execution_frame() -> pd.DataFrame:
    columns = ["model_name", "source", "fold_id", "split", *_SCENARIO_COLUMNS]
    columns.extend(_CANONICAL_METRIC_COLUMNS)
    return pd.DataFrame(columns=columns)


def _canonicalise_execution_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["model_name"] = frame.get("model_name", pd.Series(dtype=str)).astype(str)
    out["source"] = "classical"
    if "fold_id" in frame.columns:
        out["fold_id"] = frame["fold_id"].apply(_normalise_fold_token)
    else:
        out["fold_id"] = "aggregate"
    out["split"] = frame.get("split", pd.Series(["test"] * len(frame))).astype(str)
    for column in _SCENARIO_COLUMNS:
        out[column] = pd.to_numeric(frame.get(column), errors="coerce")
    for column in _CANONICAL_METRIC_COLUMNS:
        if column in frame.columns:
            out[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            out[column] = pd.NA
    # Derive a missing cost proxy from gross minus net where both exist.
    needs_cost = out["cost_proxy"].isna()
    derived = out["gross_signal_return_proxy"] - out["net_signal_return_proxy"]
    out.loc[needs_cost, "cost_proxy"] = derived[needs_cost]
    out = out.dropna(subset=["confidence_threshold", "cost_bps", "latency_steps"])
    out["latency_steps"] = out["latency_steps"].astype(int)
    return out.reset_index(drop=True)


def _load_statistical_metric(
    *,
    path: Path | None,
    fold_column_candidates: Sequence[str],
) -> dict[str, float]:
    """Return the mean held-out test macro-F1 per model from a metric table."""
    if path is None or not path.is_file():
        return {}
    frame = pd.read_csv(path)
    if frame.empty or "model_name" not in frame.columns:
        return {}
    if "macro_f1" not in frame.columns:
        return {}
    if "status" in frame.columns:
        frame = frame.loc[frame["status"].astype(str).str.lower() == "ok"]
    if "split" in frame.columns:
        frame = frame.loc[frame["split"].astype(str).str.lower() == "test"]
    if frame.empty:
        return {}
    metrics: dict[str, float] = {}
    for model_name, group in frame.groupby("model_name", sort=True):
        value = _mean(group["macro_f1"])
        if value is not None:
            metrics[str(model_name)] = value
    _ = fold_column_candidates  # accepted for symmetry; not needed for the mean
    return metrics


# ---------------------------------------------------------------------------
# Per-row derived diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Scenario:
    thresholds: tuple[float, ...]
    costs: tuple[float, ...]
    latencies: tuple[int, ...]

    @property
    def ref_threshold(self) -> float:
        return self.thresholds[0]

    @property
    def ref_cost(self) -> float:
        return self.costs[0]

    @property
    def ref_latency(self) -> int:
        return self.latencies[0]

    @property
    def stress_cost(self) -> float:
        return self.costs[-1]

    @property
    def stress_latency(self) -> int:
        return self.latencies[-1]

    @property
    def has_latency_contrast(self) -> bool:
        return len(self.latencies) >= 2

    @property
    def has_cost_contrast(self) -> bool:
        return len(self.costs) >= 2


def _resolve_scenario(frame: pd.DataFrame) -> _Scenario | None:
    if frame.empty:
        return None
    thresholds = tuple(sorted({float(v) for v in frame["confidence_threshold"].dropna()}))
    costs = tuple(sorted({float(v) for v in frame["cost_bps"].dropna()}))
    latencies = tuple(sorted({int(v) for v in frame["latency_steps"].dropna()}))
    if not thresholds or not costs or not latencies:
        return None
    return _Scenario(thresholds=thresholds, costs=costs, latencies=latencies)


def _augment_with_derived_columns(
    frame: pd.DataFrame,
    *,
    scenario: _Scenario,
) -> pd.DataFrame:
    out = frame.copy()
    # Coverage: eligible relative to the most permissive threshold for the
    # same model, fold and latency (cost does not change eligibility).
    denom = out.groupby(["model_name", "fold_id", "latency_steps"])[
        "eligible_predictions"
    ].transform("max")
    out["coverage"] = _ratio(out["eligible_predictions"], denom)

    # Adverse selection: gross signal lost relative to the reference latency
    # for the same model, fold, threshold and cost.
    if scenario.has_latency_contrast:
        base = out.loc[out["latency_steps"] == scenario.ref_latency]
        base = base[
            [
                "model_name",
                "fold_id",
                "confidence_threshold",
                "cost_bps",
                "gross_signal_return_proxy",
            ]
        ].rename(columns={"gross_signal_return_proxy": "_base_gross"})
        out = out.merge(
            base,
            on=["model_name", "fold_id", "confidence_threshold", "cost_bps"],
            how="left",
        )
        out["adverse_selection_proxy"] = (
            out["_base_gross"] - out["gross_signal_return_proxy"]
        )
        out = out.drop(columns=["_base_gross"])
    else:
        out["adverse_selection_proxy"] = pd.NA

    out["fill_assumption"] = FILL_ASSUMPTION_LABEL
    out["fill_assumption_proxy"] = _ratio(
        out["trade_count_proxy"], out["eligible_predictions"]
    )
    out["status"] = "ok"
    out["skip_reason"] = ""
    return out


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    safe = den.where(den > 0.0)
    return num / safe


# ---------------------------------------------------------------------------
# Surface builders
# ---------------------------------------------------------------------------


def _cost_latency_surface(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    keys = ["model_name", "split", "confidence_threshold", "cost_bps", "latency_steps"]
    for key_values, group in frame.groupby(keys, sort=True):
        model_name, split, threshold, cost, latency = key_values
        rows.append(
            {
                "model_name": str(model_name),
                "split": str(split),
                "confidence_threshold": float(threshold),
                "cost_bps": float(cost),
                "latency_steps": int(latency),
                "fold_count": int(group["fold_id"].nunique()),
                "coverage_mean": _mean(group["coverage"]),
                "eligible_predictions_mean": _mean(group["eligible_predictions"]),
                "turnover_proxy_mean": _mean(group["turnover_proxy"]),
                "gross_signal_return_proxy_mean": _mean(group["gross_signal_return_proxy"]),
                "cost_proxy_mean": _mean(group["cost_proxy"]),
                "net_signal_return_proxy_mean": _mean(group["net_signal_return_proxy"]),
                "net_signal_return_proxy_std": _std(group["net_signal_return_proxy"]),
                "hit_rate_proxy_mean": _mean(group["hit_rate_proxy"]),
                "status": "ok",
            }
        )
    return rows


def _confidence_threshold_summary(
    frame: pd.DataFrame,
    *,
    scenario: _Scenario,
) -> list[dict[str, Any]]:
    reference = frame.loc[
        (frame["cost_bps"] == scenario.ref_cost)
        & (frame["latency_steps"] == scenario.ref_latency)
    ]
    if reference.empty:
        return []
    rows: list[dict[str, Any]] = []
    for model_name, model_group in reference.groupby("model_name", sort=True):
        per_threshold: dict[float, dict[str, float | None]] = {}
        for threshold, group in model_group.groupby("confidence_threshold", sort=True):
            per_threshold[float(threshold)] = {
                "coverage_mean": _mean(group["coverage"]),
                "hit_rate_proxy_mean": _mean(group["hit_rate_proxy"]),
            }
        base = per_threshold.get(scenario.ref_threshold, {})
        base_coverage = base.get("coverage_mean")
        base_hit_rate = base.get("hit_rate_proxy_mean")
        for threshold, group in model_group.groupby("confidence_threshold", sort=True):
            coverage_mean = _mean(group["coverage"])
            hit_rate_mean = _mean(group["hit_rate_proxy"])
            rows.append(
                {
                    "model_name": str(model_name),
                    "split": "test",
                    "confidence_threshold": float(threshold),
                    "cost_bps": float(scenario.ref_cost),
                    "latency_steps": int(scenario.ref_latency),
                    "fold_count": int(group["fold_id"].nunique()),
                    "coverage_mean": coverage_mean,
                    "eligible_predictions_mean": _mean(group["eligible_predictions"]),
                    "trade_count_proxy_mean": _mean(group["trade_count_proxy"]),
                    "hit_rate_proxy_mean": hit_rate_mean,
                    "coverage_delta_vs_base": _delta(coverage_mean, base_coverage),
                    "hit_rate_proxy_delta_vs_base": _delta(hit_rate_mean, base_hit_rate),
                    "status": "ok",
                }
            )
    return rows


def _turnover_summary(frame: pd.DataFrame, *, scenario: _Scenario) -> list[dict[str, Any]]:
    reference = frame.loc[
        (frame["cost_bps"] == scenario.ref_cost)
        & (frame["latency_steps"] == scenario.ref_latency)
    ]
    if reference.empty:
        return []
    rows: list[dict[str, Any]] = []
    keys = ["model_name", "confidence_threshold"]
    for key_values, group in reference.groupby(keys, sort=True):
        model_name, threshold = key_values
        rows.append(
            {
                "model_name": str(model_name),
                "split": "test",
                "confidence_threshold": float(threshold),
                "cost_bps": float(scenario.ref_cost),
                "latency_steps": int(scenario.ref_latency),
                "fold_count": int(group["fold_id"].nunique()),
                "turnover_proxy_mean": _mean(group["turnover_proxy"]),
                "turnover_proxy_std": _std(group["turnover_proxy"]),
                "trade_count_proxy_mean": _mean(group["trade_count_proxy"]),
                "coverage_mean": _mean(group["coverage"]),
                "status": "ok",
            }
        )
    return rows


def _adverse_selection_summary(
    frame: pd.DataFrame,
    *,
    scenario: _Scenario,
    neural_models: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    skips: list[dict[str, str]] = []
    if not scenario.has_latency_contrast:
        skips.append(
            {
                "diagnostic": "adverse_selection",
                "scope": "all_models",
                "skip_reason": (
                    "stored execution rows contain a single latency step, so the "
                    "latency-induced signal decay cannot be measured"
                ),
            }
        )
    else:
        reference = frame.loc[
            (frame["confidence_threshold"] == scenario.ref_threshold)
            & (frame["cost_bps"] == scenario.ref_cost)
            & (frame["latency_steps"] != scenario.ref_latency)
        ]
        keys = ["model_name", "latency_steps"]
        for key_values, group in reference.groupby(keys, sort=True):
            model_name, latency = key_values
            rows.append(
                {
                    "model_name": str(model_name),
                    "source": "classical",
                    "split": "test",
                    "confidence_threshold": float(scenario.ref_threshold),
                    "cost_bps": float(scenario.ref_cost),
                    "latency_steps": int(latency),
                    "base_latency_steps": int(scenario.ref_latency),
                    "fold_count": int(group["fold_id"].nunique()),
                    "gross_signal_return_proxy_mean": _mean(
                        group["gross_signal_return_proxy"]
                    ),
                    "adverse_selection_proxy_mean": _mean(
                        group["adverse_selection_proxy"]
                    ),
                    "adverse_selection_proxy_std": _std(group["adverse_selection_proxy"]),
                    "status": "ok",
                    "skip_reason": "",
                }
            )
    for model_name in neural_models:
        reason = (
            "neural runs ship no stored execution proxy rows, so an "
            "adverse-selection proxy cannot be computed"
        )
        rows.append(
            {
                "model_name": model_name,
                "source": "neural",
                "split": "test",
                "confidence_threshold": None,
                "cost_bps": None,
                "latency_steps": None,
                "base_latency_steps": None,
                "fold_count": 0,
                "gross_signal_return_proxy_mean": None,
                "adverse_selection_proxy_mean": None,
                "adverse_selection_proxy_std": None,
                "status": "skipped",
                "skip_reason": reason,
            }
        )
        skips.append(
            {"diagnostic": "adverse_selection", "scope": model_name, "skip_reason": reason}
        )
    return rows, skips


def _fill_assumption_summary(
    frame: pd.DataFrame,
    *,
    scenario: _Scenario,
    neural_models: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    skips: list[dict[str, str]] = []
    reference = frame.loc[
        (frame["cost_bps"] == scenario.ref_cost)
        & (frame["latency_steps"] == scenario.ref_latency)
    ]
    has_fill = (
        not reference.empty
        and pd.to_numeric(reference["fill_assumption_proxy"], errors="coerce").notna().any()
    )
    if has_fill:
        keys = ["model_name", "confidence_threshold"]
        for key_values, group in reference.groupby(keys, sort=True):
            model_name, threshold = key_values
            rows.append(
                {
                    "model_name": str(model_name),
                    "source": "classical",
                    "split": "test",
                    "confidence_threshold": float(threshold),
                    "cost_bps": float(scenario.ref_cost),
                    "latency_steps": int(scenario.ref_latency),
                    "fold_count": int(group["fold_id"].nunique()),
                    "fill_assumption": FILL_ASSUMPTION_LABEL,
                    "eligible_predictions_mean": _mean(group["eligible_predictions"]),
                    "trade_count_proxy_mean": _mean(group["trade_count_proxy"]),
                    "fill_assumption_proxy_mean": _mean(group["fill_assumption_proxy"]),
                    "fill_assumption_proxy_std": _std(group["fill_assumption_proxy"]),
                    "status": "ok",
                    "skip_reason": "",
                }
            )
    else:
        skips.append(
            {
                "diagnostic": "fill_assumption",
                "scope": "classical",
                "skip_reason": (
                    "stored execution rows lack a trade-count column, so the "
                    "eligible-to-fill share cannot be computed"
                ),
            }
        )
    for model_name in neural_models:
        reason = (
            "neural runs ship no stored execution proxy rows, so a "
            "fill-assumption proxy cannot be computed"
        )
        rows.append(
            {
                "model_name": model_name,
                "source": "neural",
                "split": "test",
                "confidence_threshold": None,
                "cost_bps": None,
                "latency_steps": None,
                "fold_count": 0,
                "fill_assumption": "unavailable",
                "eligible_predictions_mean": None,
                "trade_count_proxy_mean": None,
                "fill_assumption_proxy_mean": None,
                "fill_assumption_proxy_std": None,
                "status": "skipped",
                "skip_reason": reason,
            }
        )
        skips.append(
            {"diagnostic": "fill_assumption", "scope": model_name, "skip_reason": reason}
        )
    return rows, skips


def _degradation_summary(
    frame: pd.DataFrame,
    *,
    scenario: _Scenario | None,
    classical_metric: Mapping[str, float],
    neural_metric: Mapping[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    skips: list[dict[str, str]] = []

    if scenario is not None and not frame.empty:
        for model_name, group in frame.groupby("model_name", sort=True):
            base = group.loc[
                (group["confidence_threshold"] == scenario.ref_threshold)
                & (group["cost_bps"] == scenario.ref_cost)
                & (group["latency_steps"] == scenario.ref_latency)
            ]
            stressed = group.loc[
                (group["confidence_threshold"] == scenario.ref_threshold)
                & (group["cost_bps"] == scenario.stress_cost)
                & (group["latency_steps"] == scenario.stress_latency)
            ]
            base_value = _mean(base["gross_signal_return_proxy"]) if not base.empty else None
            exec_value = (
                _mean(stressed["net_signal_return_proxy"]) if not stressed.empty else None
            )
            rows.append(
                {
                    "model_name": str(model_name),
                    "source": "classical",
                    "fold_id": "aggregate",
                    "statistical_metric": STATISTICAL_METRIC_NAME,
                    "statistical_value": classical_metric.get(str(model_name)),
                    "base_proxy_metric": (
                        f"{BASE_PROXY_METRIC_NAME}@thr{scenario.ref_threshold:g}"
                        f"_cost{scenario.ref_cost:g}_lat{scenario.ref_latency}"
                    ),
                    "base_proxy_value": base_value,
                    "exec_proxy_metric": (
                        f"{EXEC_PROXY_METRIC_NAME}@thr{scenario.ref_threshold:g}"
                        f"_cost{scenario.stress_cost:g}_lat{scenario.stress_latency}"
                    ),
                    "exec_proxy_value": exec_value,
                    "absolute_degradation_proxy": _delta(base_value, exec_value),
                    "relative_degradation_proxy": _relative_degradation(base_value, exec_value),
                    "status": "ok",
                    "skip_reason": "",
                }
            )

    for model_name, value in sorted(neural_metric.items()):
        reason = (
            "neural runs ship no stored execution proxy rows, so the execution "
            "side of the degradation cannot be computed"
        )
        rows.append(
            {
                "model_name": model_name,
                "source": "neural",
                "fold_id": "aggregate",
                "statistical_metric": STATISTICAL_METRIC_NAME,
                "statistical_value": value,
                "base_proxy_metric": BASE_PROXY_METRIC_NAME,
                "base_proxy_value": None,
                "exec_proxy_metric": EXEC_PROXY_METRIC_NAME,
                "exec_proxy_value": None,
                "absolute_degradation_proxy": None,
                "relative_degradation_proxy": None,
                "status": "skipped",
                "skip_reason": reason,
            }
        )
        skips.append({"diagnostic": "degradation", "scope": model_name, "skip_reason": reason})
    return rows, skips


def _delta(value: float | None, base: float | None) -> float | None:
    if value is None or base is None:
        return None
    return float(value) - float(base)


def _relative_degradation(base: float | None, exec_value: float | None) -> float | None:
    if base is None or exec_value is None:
        return None
    if abs(base) < 1e-12:
        return None
    return (float(base) - float(exec_value)) / abs(float(base))


# ---------------------------------------------------------------------------
# Notes and assumptions
# ---------------------------------------------------------------------------


def _most_cost_fragile(
    surface_rows: Sequence[Mapping[str, Any]],
    *,
    scenario: _Scenario,
) -> tuple[str, float] | None:
    candidates = [
        row
        for row in surface_rows
        if _safe_float(row.get("confidence_threshold")) == scenario.ref_threshold
        and _safe_float(row.get("cost_bps")) == scenario.stress_cost
        and int(row.get("latency_steps") or 0) == scenario.ref_latency
        and _safe_float(row.get("net_signal_return_proxy_mean")) is not None
    ]
    if not candidates:
        return None
    worst = min(candidates, key=lambda r: _safe_float(r.get("net_signal_return_proxy_mean")) or 0.0)
    return str(worst["model_name"]), float(worst["net_signal_return_proxy_mean"])


def _most_latency_sensitive(
    adverse_rows: Sequence[Mapping[str, Any]],
    *,
    scenario: _Scenario,
) -> tuple[str, float] | None:
    candidates = [
        row
        for row in adverse_rows
        if row.get("status") == "ok"
        and int(row.get("latency_steps") or 0) == scenario.stress_latency
        and _safe_float(row.get("adverse_selection_proxy_mean")) is not None
    ]
    if not candidates:
        return None
    worst = max(
        candidates,
        key=lambda r: _safe_float(r.get("adverse_selection_proxy_mean")) or 0.0,
    )
    return str(worst["model_name"]), float(worst["adverse_selection_proxy_mean"])


def _largest_degradation(
    degradation_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, float] | None:
    candidates = [
        row
        for row in degradation_rows
        if row.get("status") == "ok"
        and _safe_float(row.get("absolute_degradation_proxy")) is not None
    ]
    if not candidates:
        return None
    worst = max(
        candidates,
        key=lambda r: _safe_float(r.get("absolute_degradation_proxy")) or 0.0,
    )
    return str(worst["model_name"]), float(worst["absolute_degradation_proxy"])


def _format_assumptions() -> str:
    lines = [
        "# FI-2010 Execution-Aware Evaluation v2 - Assumptions",
        "",
        "Every number produced by this layer is a simplified proxy diagnostic.",
        "The assumptions below are deliberately explicit so the gap between a",
        "forecasting metric and a tradability proxy is never read as a result.",
        "",
        "## What this is not",
        "",
        "- This is not a backtest.",
        "- This is not a live-trading simulation.",
        "- There is no market impact model.",
        "- There is no queue-position ground truth.",
        "- Fills are approximate or unavailable depending on the input artefacts.",
        "",
        "## What the proxies assume",
        "",
        "- Costs are scenario assumptions expressed in basis points, applied as a",
        "  fixed per-trade deduction; they are not measured exchange fees.",
        "- Latency is row-step latency (the realised forward return is read a fixed",
        "  number of rows later); it is not exchange or network latency.",
        f"- The fill model is `{FILL_ASSUMPTION_LABEL}`: every eligible directional",
        "  signal is assumed filled at the mid price with no queue position.",
        "- The return proxy is a forward mid-price change in basis points, inherited",
        "  from the stored execution-sensitivity rows.",
        "",
        "## How to read the output",
        "",
        "- The metrics are useful for stress-testing signal fragility, not for",
        "  proving tradability.",
        "- A model can hold a respectable forecasting metric while its net proxy",
        "  signal shrinks or turns negative once cost and latency are applied.",
        "",
        "## Boundaries",
        "",
    ]
    lines.extend(f"- {boundary}" for boundary in _CLAIM_BOUNDARIES)
    lines.append("")
    return "\n".join(lines)


def _format_notes(
    *,
    scenario: _Scenario | None,
    surface_rows: Sequence[Mapping[str, Any]],
    threshold_rows: Sequence[Mapping[str, Any]],
    adverse_rows: Sequence[Mapping[str, Any]],
    degradation_rows: Sequence[Mapping[str, Any]],
    neural_models: Sequence[str],
) -> str:
    lines = [
        "# FI-2010 Execution-Aware Evaluation v2 - Notes",
        "",
        "These notes summarise the stored proxy diagnostics. They describe",
        "fragility, not tradability, and make no profitability claim.",
        "",
        "## Most sensitive to cost",
        "",
    ]
    if scenario is None:
        lines.append("- No classical execution proxy rows were available.")
    else:
        cost_finding = _most_cost_fragile(surface_rows, scenario=scenario)
        lines.append(
            "- The cost proxy is a fixed per-trade deduction, so the absolute net "
            "reduction from cost is the same across models; the model whose gross "
            "proxy return is smallest crosses into the weakest net proxy first."
        )
        if cost_finding is not None:
            model_name, net_value = cost_finding
            lines.append(
                f"- At the highest stored cost ({scenario.stress_cost:g} bps, "
                f"reference threshold and latency) `{model_name}` has the lowest "
                f"mean net signal return proxy ({net_value:+.4f} bps)."
            )
        else:
            lines.append("- No stressed-cost surface row was available.")

    lines.extend(["", "## Most sensitive to latency", ""])
    if scenario is None or not scenario.has_latency_contrast:
        lines.append(
            "- Stored rows contain a single latency step, so latency sensitivity "
            "cannot be measured and the adverse-selection proxy is skipped."
        )
    else:
        latency_finding = _most_latency_sensitive(adverse_rows, scenario=scenario)
        if latency_finding is not None:
            model_name, decay = latency_finding
            lines.append(
                f"- At the highest stored latency ({scenario.stress_latency} steps) "
                f"`{model_name}` shows the largest adverse-selection proxy "
                f"({decay:+.4f} bps of gross signal lost versus latency "
                f"{scenario.ref_latency})."
            )
        else:
            lines.append("- No adverse-selection proxy row was available.")

    lines.extend(["", "## Confidence thresholding", ""])
    threshold_line = _threshold_finding(threshold_rows)
    if threshold_line is not None:
        lines.append(threshold_line)
    else:
        lines.append("- No confidence-threshold rows were available.")
    lines.append(
        "- Raising the confidence threshold lowers coverage (fewer eligible "
        "predictions) and generally raises the hit-rate proxy; "
        "`confidence_threshold_summary.csv` records both deltas per model."
    )

    lines.extend(["", "## Where the net proxy signal degrades most", ""])
    degradation_finding = _largest_degradation(degradation_rows)
    if degradation_finding is not None:
        model_name, gap = degradation_finding
        lines.append(
            f"- `{model_name}` shows the largest gap between its base gross proxy "
            f"return and its stressed net proxy return ({gap:+.4f} bps); see "
            "`degradation_summary.csv`."
        )
    else:
        lines.append("- No classical degradation rows were available.")
    if neural_models:
        lines.append(
            "- Neural runs report a forecasting metric but ship no execution proxy "
            "rows, so their execution-aware side is skipped, not assumed; the gap "
            "between their forecasting metric and tradability is therefore "
            "unquantified here."
        )

    lines.extend(["", "## What cannot be concluded", ""])
    lines.extend(
        [
            "- Nothing here demonstrates profitability or live tradability.",
            "- Without a market-impact model, queue ground truth and a realistic fill",
            "  model, the net proxy return cannot be read as an achievable return.",
            "- Cross-model net comparisons are conditioned on identical scenario",
            "  assumptions and a shared, simplified return proxy.",
            "",
        ]
    )
    return "\n".join(lines)


def _threshold_finding(threshold_rows: Sequence[Mapping[str, Any]]) -> str | None:
    candidates = [
        row
        for row in threshold_rows
        if row.get("status") == "ok"
        and _safe_float(row.get("coverage_delta_vs_base")) is not None
        and _safe_float(row.get("hit_rate_proxy_delta_vs_base")) is not None
        and (_safe_float(row.get("coverage_delta_vs_base")) or 0.0) < 0.0
    ]
    if not candidates:
        return None
    sharpest = min(
        candidates,
        key=lambda r: _safe_float(r.get("coverage_delta_vs_base")) or 0.0,
    )
    model_name = str(sharpest["model_name"])
    threshold = _safe_float(sharpest.get("confidence_threshold")) or 0.0
    coverage_delta = float(sharpest["coverage_delta_vs_base"])
    hit_delta = float(sharpest["hit_rate_proxy_delta_vs_base"])
    return (
        f"- The sharpest stored coverage trade-off is `{model_name}` at threshold "
        f"{threshold:g}: coverage changes by {coverage_delta:+.4f} and the hit-rate "
        f"proxy by {hit_delta:+.4f} versus the most permissive threshold."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_fi2010_execution_v2(
    *,
    classical_dir: str | Path | None,
    neural_dir: str | Path | None = None,
    ablations_dir: str | Path | None = None,
    out_dir: str | Path,
    models: Sequence[str] | str | None = None,
    cost_bps: Sequence[float] | str | None = None,
    latency_steps: Sequence[int] | str | None = None,
    confidence_thresholds: Sequence[float] | str | None = None,
    overwrite: bool = False,
) -> ExecutionV2Summary:
    """Build FI-2010 execution-aware v2 proxy diagnostics from stored artefacts."""
    resolved_classical = Path(classical_dir) if classical_dir is not None else None
    resolved_neural = Path(neural_dir) if neural_dir is not None else None
    resolved_ablations = Path(ablations_dir) if ablations_dir is not None else None
    resolved_out = Path(out_dir)

    model_filter = _parse_model_filter(models)
    cost_filter = _parse_float_filter(cost_bps)
    latency_filter = _parse_int_filter(latency_steps)
    threshold_filter = _parse_float_filter(confidence_thresholds)

    warnings: list[str] = []
    execution_frame, load_warnings = _load_classical_execution_rows(resolved_classical)
    warnings.extend(load_warnings)

    classical_metric = _load_statistical_metric(
        path=(
            resolved_classical / "results_by_fold.csv"
            if resolved_classical is not None
            else None
        ),
        fold_column_candidates=("fold_id",),
    )
    neural_metric = _load_statistical_metric(
        path=(
            resolved_neural / "results_by_fold_seed.csv"
            if resolved_neural is not None
            else None
        ),
        fold_column_candidates=("fold_id",),
    )

    # Apply optional filters before any aggregation.
    if not execution_frame.empty:
        if model_filter is not None:
            execution_frame = execution_frame.loc[
                execution_frame["model_name"].isin(model_filter)
            ]
        if threshold_filter is not None:
            execution_frame = execution_frame.loc[
                execution_frame["confidence_threshold"].isin(threshold_filter)
            ]
        if cost_filter is not None:
            execution_frame = execution_frame.loc[
                execution_frame["cost_bps"].isin(cost_filter)
            ]
        if latency_filter is not None:
            execution_frame = execution_frame.loc[
                execution_frame["latency_steps"].isin(latency_filter)
            ]
        execution_frame = execution_frame.reset_index(drop=True)

    if model_filter is not None:
        neural_metric = {
            name: value for name, value in neural_metric.items() if name in model_filter
        }

    neural_models = tuple(sorted(neural_metric))
    scenario = _resolve_scenario(execution_frame)

    if scenario is not None:
        enriched = _augment_with_derived_columns(execution_frame, scenario=scenario)
    else:
        enriched = execution_frame
        if resolved_classical is not None and execution_frame.empty:
            warnings.append(
                "no classical execution rows remained after loading and filtering; "
                "execution surfaces are empty and neural diagnostics are skipped"
            )

    _ensure_output_dir(resolved_out, overwrite=overwrite)

    diagnostics_produced: list[str] = []
    diagnostics_skipped: list[str] = []
    skipped_records: list[dict[str, str]] = []

    # --- main per-scenario result rows -----------------------------------
    result_rows = _result_rows(enriched, scenario=scenario, neural_models=neural_models)
    _write_csv(
        result_rows,
        resolved_out / "execution_v2_results.csv",
        EXECUTION_V2_RESULT_COLUMNS,
    )

    # --- surfaces ---------------------------------------------------------
    surface_rows = _cost_latency_surface(enriched) if scenario is not None else []
    _write_csv(
        surface_rows, resolved_out / "cost_latency_surface.csv", COST_LATENCY_SURFACE_COLUMNS
    )
    _mark(diagnostics_produced, diagnostics_skipped, "cost_latency_surface", bool(surface_rows))

    threshold_rows = (
        _confidence_threshold_summary(enriched, scenario=scenario)
        if scenario is not None
        else []
    )
    _write_csv(
        threshold_rows,
        resolved_out / "confidence_threshold_summary.csv",
        CONFIDENCE_THRESHOLD_SUMMARY_COLUMNS,
    )
    _mark(
        diagnostics_produced, diagnostics_skipped, "confidence_threshold", bool(threshold_rows)
    )

    turnover_rows = _turnover_summary(enriched, scenario=scenario) if scenario is not None else []
    _write_csv(turnover_rows, resolved_out / "turnover_summary.csv", TURNOVER_SUMMARY_COLUMNS)
    _mark(diagnostics_produced, diagnostics_skipped, "turnover", bool(turnover_rows))

    if scenario is not None:
        adverse_rows, adverse_skips = _adverse_selection_summary(
            enriched, scenario=scenario, neural_models=neural_models
        )
        fill_rows, fill_skips = _fill_assumption_summary(
            enriched, scenario=scenario, neural_models=neural_models
        )
    else:
        adverse_rows, adverse_skips = [], [
            {
                "diagnostic": "adverse_selection",
                "scope": "all_models",
                "skip_reason": "no classical execution rows were available",
            }
        ]
        fill_rows, fill_skips = [], [
            {
                "diagnostic": "fill_assumption",
                "scope": "all_models",
                "skip_reason": "no classical execution rows were available",
            }
        ]
    skipped_records.extend(adverse_skips)
    skipped_records.extend(fill_skips)
    _write_csv(
        adverse_rows,
        resolved_out / "adverse_selection_summary.csv",
        ADVERSE_SELECTION_SUMMARY_COLUMNS,
    )
    _write_csv(
        fill_rows, resolved_out / "fill_assumption_summary.csv", FILL_ASSUMPTION_SUMMARY_COLUMNS
    )
    _mark(
        diagnostics_produced,
        diagnostics_skipped,
        "adverse_selection",
        any(row.get("status") == "ok" for row in adverse_rows),
    )
    _mark(
        diagnostics_produced,
        diagnostics_skipped,
        "fill_assumption",
        any(row.get("status") == "ok" for row in fill_rows),
    )

    degradation_rows, degradation_skips = _degradation_summary(
        enriched,
        scenario=scenario,
        classical_metric=classical_metric,
        neural_metric=neural_metric,
    )
    skipped_records.extend(degradation_skips)
    _write_csv(
        degradation_rows, resolved_out / "degradation_summary.csv", DEGRADATION_SUMMARY_COLUMNS
    )
    _mark(
        diagnostics_produced,
        diagnostics_skipped,
        "degradation",
        any(row.get("status") == "ok" for row in degradation_rows),
    )

    if resolved_ablations is not None and not resolved_ablations.is_dir():
        skipped_records.append(
            {
                "diagnostic": "ablation_cross_reference",
                "scope": "ablations_dir",
                "skip_reason": (
                    f"ablation artefact directory {resolved_ablations} does not exist; "
                    "the cross-reference input was not consumed"
                ),
            }
        )

    # --- notes, assumptions, skipped record, summary ----------------------
    (resolved_out / "execution_assumptions.md").write_text(
        _format_assumptions(), encoding="utf-8"
    )
    (resolved_out / "execution_notes.md").write_text(
        _format_notes(
            scenario=scenario,
            surface_rows=surface_rows,
            threshold_rows=threshold_rows,
            adverse_rows=adverse_rows,
            degradation_rows=degradation_rows,
            neural_models=neural_models,
        ),
        encoding="utf-8",
    )
    (resolved_out / "skipped_diagnostics.json").write_text(
        stable_json_dumps(
            {"skipped_count": len(skipped_records), "skipped": skipped_records}
        ),
        encoding="utf-8",
    )

    classical_models = (
        tuple(sorted(enriched["model_name"].astype(str).unique()))
        if not enriched.empty
        else ()
    )
    folds = (
        tuple(sorted(enriched["fold_id"].astype(str).unique())) if not enriched.empty else ()
    )
    ok_count = sum(1 for row in result_rows if row.get("status") == "ok")
    skipped_count = len(result_rows) - ok_count

    artefacts = {
        "summary": "summary.json",
        "execution_v2_results": "execution_v2_results.csv",
        "cost_latency_surface": "cost_latency_surface.csv",
        "confidence_threshold_summary": "confidence_threshold_summary.csv",
        "turnover_summary": "turnover_summary.csv",
        "adverse_selection_summary": "adverse_selection_summary.csv",
        "fill_assumption_summary": "fill_assumption_summary.csv",
        "degradation_summary": "degradation_summary.csv",
        "skipped_diagnostics": "skipped_diagnostics.json",
        "execution_assumptions": "execution_assumptions.md",
        "execution_notes": "execution_notes.md",
    }

    summary = ExecutionV2Summary(
        output_dir=resolved_out,
        classical_dir=resolved_classical,
        neural_dir=resolved_neural,
        ablations_dir=resolved_ablations,
        classical_models=classical_models,
        neural_models=neural_models,
        folds=folds,
        result_row_count=len(result_rows),
        ok_row_count=ok_count,
        skipped_row_count=skipped_count,
        diagnostics_produced=tuple(diagnostics_produced),
        diagnostics_skipped=tuple(diagnostics_skipped),
        full_predictions_required=False,
        checkpoints_required=False,
        artefacts=artefacts,
        warnings=tuple(warnings),
    )

    _write_summary_json(
        path=resolved_out / "summary.json",
        summary=summary,
        scenario=scenario,
        classical_dir=resolved_classical,
        neural_dir=resolved_neural,
        ablations_dir=resolved_ablations,
        filters={
            "models": list(model_filter) if model_filter is not None else None,
            "cost_bps": list(cost_filter) if cost_filter is not None else None,
            "latency_steps": list(latency_filter) if latency_filter is not None else None,
            "confidence_thresholds": (
                list(threshold_filter) if threshold_filter is not None else None
            ),
        },
        skipped_records=skipped_records,
    )
    return summary


def _result_rows(
    frame: pd.DataFrame,
    *,
    scenario: _Scenario | None,
    neural_models: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if scenario is not None and not frame.empty:
        projected = frame.loc[:, [c for c in EXECUTION_V2_RESULT_COLUMNS if c in frame.columns]]
        for record in projected.to_dict("records"):
            rows.append(
                {
                    column: _clean_value(record.get(column))
                    for column in EXECUTION_V2_RESULT_COLUMNS
                }
            )
    for model_name in neural_models:
        rows.append(
            {
                "model_name": model_name,
                "source": "neural",
                "fold_id": "aggregate",
                "split": "test",
                "confidence_threshold": None,
                "cost_bps": None,
                "latency_steps": None,
                "eligible_predictions": None,
                "coverage": None,
                "trade_count_proxy": None,
                "turnover_proxy": None,
                "gross_signal_return_proxy": None,
                "cost_proxy": None,
                "net_signal_return_proxy": None,
                "hit_rate_proxy": None,
                "adverse_selection_proxy": None,
                "fill_assumption": "unavailable",
                "fill_assumption_proxy": None,
                "status": "skipped",
                "skip_reason": (
                    "neural runs ship no stored execution proxy rows; the "
                    "execution-aware diagnostics are skipped, not fabricated"
                ),
            }
        )
    return rows


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def _mark(
    produced: list[str],
    skipped: list[str],
    name: str,
    ok: bool,
) -> None:
    if ok:
        produced.append(name)
    else:
        skipped.append(name)


def _write_summary_json(
    *,
    path: Path,
    summary: ExecutionV2Summary,
    scenario: _Scenario | None,
    classical_dir: Path | None,
    neural_dir: Path | None,
    ablations_dir: Path | None,
    filters: Mapping[str, Any],
    skipped_records: Sequence[Mapping[str, str]],
) -> None:
    payload: dict[str, Any] = {
        "runner_version": FI2010_EXECUTION_V2_VERSION,
        "package_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "proxy_diagnostics": True,
        "disclaimer": (
            "All execution outputs are simplified proxy diagnostics. This is not a "
            "backtest or a live-trading simulation and carries no profitability or "
            "tradability claim."
        ),
        "inputs": {
            "classical_dir": str(classical_dir) if classical_dir is not None else None,
            "neural_dir": str(neural_dir) if neural_dir is not None else None,
            "ablations_dir": str(ablations_dir) if ablations_dir is not None else None,
        },
        "input_artefacts": _input_artefact_manifest(
            classical_dir=classical_dir,
            neural_dir=neural_dir,
            ablations_dir=ablations_dir,
        ),
        "filters": dict(filters),
        "scenario": (
            {
                "confidence_thresholds": list(scenario.thresholds),
                "cost_bps": list(scenario.costs),
                "latency_steps": list(scenario.latencies),
                "reference_threshold": scenario.ref_threshold,
                "reference_cost_bps": scenario.ref_cost,
                "reference_latency_steps": scenario.ref_latency,
                "stress_cost_bps": scenario.stress_cost,
                "stress_latency_steps": scenario.stress_latency,
            }
            if scenario is not None
            else None
        ),
        "classical_models": list(summary.classical_models),
        "neural_models": list(summary.neural_models),
        "folds": list(summary.folds),
        "counts": {
            "result_rows": summary.result_row_count,
            "ok_rows": summary.ok_row_count,
            "skipped_rows": summary.skipped_row_count,
            "skipped_diagnostics": len(skipped_records),
        },
        "diagnostics_produced": list(summary.diagnostics_produced),
        "diagnostics_skipped": list(summary.diagnostics_skipped),
        "fill_assumption": FILL_ASSUMPTION_LABEL,
        "artefacts": dict(summary.artefacts),
        "full_predictions_required": summary.full_predictions_required,
        "checkpoints_required": summary.checkpoints_required,
        "warnings": list(summary.warnings),
        "environment": _build_environment_payload(),
        "claim_boundaries": list(_CLAIM_BOUNDARIES),
    }
    path.write_text(stable_json_dumps(payload), encoding="utf-8")


def _input_artefact_manifest(
    *,
    classical_dir: Path | None,
    neural_dir: Path | None,
    ablations_dir: Path | None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    candidates = {
        "classical_execution_summary": (
            classical_dir / "execution_summary.csv" if classical_dir is not None else None
        ),
        "classical_results_by_fold": (
            classical_dir / "results_by_fold.csv" if classical_dir is not None else None
        ),
        "neural_results_by_fold_seed": (
            neural_dir / "results_by_fold_seed.csv" if neural_dir is not None else None
        ),
        "ablations_execution_cost_latency": (
            ablations_dir / "execution_cost_latency_ablation.csv"
            if ablations_dir is not None
            else None
        ),
    }
    for key, candidate in candidates.items():
        if candidate is not None and candidate.is_file():
            manifest[key] = {"path": str(candidate), "sha256": sha256_file(candidate)}
        else:
            manifest[key] = {
                "path": str(candidate) if candidate is not None else None,
                "sha256": None,
            }
    return manifest
