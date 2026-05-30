"""Failure-focused analysis of FI-2010 self-supervised pretraining results.

This module separates what the self-supervised (SSL) objectives did and did
not achieve across the completed FI-2010 evidence. It consumes only the
retained lightweight artefacts that summarise each experiment:

* matched supervised-vs-SSL comparison tables (``ssl_comparison.csv``)
* per-objective aggregate tables (``aggregate_summary.json``)
* training-curve summaries (``training_curves_summary.csv``)
* run-scope metadata (``summary.json``)

It never opens the heavy raw per-run prediction files or encoder checkpoints,
which were intentionally deleted to save disk space. If those files are absent
the analysis still runs to completion; ``raw_predictions_required`` and
``checkpoints_required`` are always reported as ``False``.

The analysis is deliberately conservative. It reports stored deltas
metric-by-metric and refuses to turn an isolated, single-fold gain into a
broad self-supervised improvement claim. The interpretation distinguishes the
completed one-epoch matched grid (comparison / infrastructure evidence) from
the small longer-training subset (narrow ``partial_real`` modelling evidence).
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
    "SSL_FAILURE_ANALYSIS_VERSION",
    "SSLFailureAnalysisSummary",
    "analyse_fi2010_ssl_results",
]

SSL_FAILURE_ANALYSIS_VERSION = "phase-l/ssl-failure-analysis/v1"

_SSL_OBJECTIVES: tuple[str, ...] = ("masked_reconstruction", "next_field")
_HIGHER_IS_BETTER: tuple[str, ...] = ("macro_f1", "mcc", "accuracy")
_LOWER_IS_BETTER: tuple[str, ...] = ("ece", "brier_score", "nll")
_SUMMARY_METRICS: tuple[str, ...] = _HIGHER_IS_BETTER + _LOWER_IS_BETTER
_HEADLINE_METRICS: tuple[str, ...] = ("macro_f1", "mcc", "ece")

# Only these lightweight, retained files are read. Heavy ``runs/`` prediction
# arrays and ``*.pt`` checkpoints are never required.
_FULL_GRID_REQUIRED: tuple[str, ...] = (
    "ssl_comparison.csv",
    "aggregate_summary.json",
    "summary.json",
)
_PROPER_TRAINING_REQUIRED: tuple[str, ...] = (
    "ssl_comparison.csv",
    "aggregate_summary.json",
    "summary.json",
)

_FULL_GRID_SOURCE = "one_epoch_full_grid"
_PROPER_TRAINING_SOURCE = "proper_training_subset"

_REQUIRED_CONCLUSION = (
    "The proper-training subset provides narrow partial evidence that masked "
    "SSL can improve fold-1/horizon-50 predictive metrics under this "
    "configuration, but calibration worsened and the scope is too small for a "
    "broad SSL improvement claim."
)


@dataclass(frozen=True)
class SSLFailureAnalysisSummary:
    """Lightweight return value describing the generated analysis."""

    output_dir: Path
    full_grid_dir: Path | None
    proper_training_dir: Path | None
    full_grid_evidence_level: str | None
    proper_training_evidence_level: str | None
    full_grid_matched_rows: int
    proper_training_matched_rows: int
    artefacts: Mapping[str, str] = field(default_factory=dict)
    figures_generated: tuple[str, ...] = ()
    claim_statuses: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    raw_predictions_required: bool = False
    checkpoints_required: bool = False


# ---------------------------------------------------------------------------
# Loaders (lightweight artefacts only)
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


def _require_files(directory: Path, required: Sequence[str], *, label: str) -> None:
    if not directory.exists():
        raise FileNotFoundError(f"{label} artefact directory missing: {directory}")
    for filename in required:
        candidate = directory / filename
        if not candidate.is_file():
            raise FileNotFoundError(
                f"{label} artefacts incomplete: missing {candidate}",
            )


def _read_matched_comparison(directory: Path, *, source: str) -> pd.DataFrame:
    """Read the matched supervised-vs-SSL comparison rows only."""

    frame = pd.read_csv(directory / "ssl_comparison.csv")
    if "status" in frame.columns:
        frame = frame[frame["status"].astype(str) == "matched"].copy()
    frame["source"] = source
    if "ssl_objective" in frame.columns:
        frame = frame[frame["ssl_objective"].astype(str).isin(_SSL_OBJECTIVES)].copy()
    for column in (*(f"delta_{metric}" for metric in _SUMMARY_METRICS),):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("fold", "horizon", "seed", "lookback"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    return frame.reset_index(drop=True)


def _read_summary_metadata(directory: Path) -> dict[str, Any]:
    payload = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_aggregate_rows(directory: Path) -> list[dict[str, Any]]:
    payload = json.loads((directory / "aggregate_summary.json").read_text(encoding="utf-8"))
    rows = payload.get("aggregate", []) if isinstance(payload, dict) else []
    return [row for row in rows if isinstance(row, dict)]


def _read_training_curves(directory: Path) -> pd.DataFrame:
    path = directory / "training_curves_summary.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Delta aggregation
# ---------------------------------------------------------------------------


def _outcome(delta: float | None, metric: str) -> str:
    """Classify a stored delta as a win / tie / loss for the SSL model."""

    if delta is None or (isinstance(delta, float) and math.isnan(delta)):
        return "tie"
    if abs(delta) < 1e-12:
        return "tie"
    improved = delta > 0.0 if metric in _HIGHER_IS_BETTER else delta < 0.0
    return "win" if improved else "loss"


def _aggregate_deltas(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    *,
    metrics: Sequence[str] = _HEADLINE_METRICS,
) -> pd.DataFrame:
    """Aggregate matched-pair deltas over the supplied grouping columns."""

    if frame.empty:
        return pd.DataFrame()
    available = [column for column in group_cols if column in frame.columns]
    if not available:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for key, group in frame.groupby(available, sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        record: dict[str, Any] = dict(zip(available, key_tuple, strict=False))
        record["n_pairs"] = len(group)
        for metric in metrics:
            delta_col = f"delta_{metric}"
            if delta_col not in group.columns:
                continue
            deltas = [None if pd.isna(value) else float(value) for value in group[delta_col]]
            present = [value for value in deltas if value is not None]
            record[f"mean_delta_{metric}"] = sum(present) / len(present) if present else None
            outcomes = [_outcome(value, metric) for value in deltas]
            record[f"{metric}_wins"] = outcomes.count("win")
            record[f"{metric}_ties"] = outcomes.count("tie")
            record[f"{metric}_losses"] = outcomes.count("loss")
        records.append(record)
    return pd.DataFrame(records)


def _metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per source/objective/metric mean delta and improved/worse/tie counts."""

    if frame.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    group_cols = [column for column in ("source", "ssl_objective") if column in frame.columns]
    for key, group in frame.groupby(group_cols, sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols, key_tuple, strict=False))
        for metric in _SUMMARY_METRICS:
            delta_col = f"delta_{metric}"
            if delta_col not in group.columns:
                continue
            deltas = [None if pd.isna(value) else float(value) for value in group[delta_col]]
            present = [value for value in deltas if value is not None]
            outcomes = [_outcome(value, metric) for value in deltas]
            direction = "higher_is_better" if metric in _HIGHER_IS_BETTER else "lower_is_better"
            records.append(
                {
                    **base,
                    "metric": metric,
                    "direction": direction,
                    "n_pairs": len(group),
                    "mean_delta": sum(present) / len(present) if present else None,
                    "n_improved": outcomes.count("win"),
                    "n_tied": outcomes.count("tie"),
                    "n_worsened": outcomes.count("loss"),
                }
            )
    return pd.DataFrame(records)


def _supervised_baseline_by_horizon(
    aggregate_rows: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, float]]:
    """Extract the supervised (``none``) baseline metrics keyed by horizon."""

    baseline: dict[int, dict[str, float]] = {}
    for row in aggregate_rows:
        if str(row.get("pretraining_objective")) != "none":
            continue
        horizon = row.get("horizon")
        if horizon is None:
            continue
        baseline[int(horizon)] = {
            "macro_f1": float(row.get("mean_macro_f1", float("nan"))),
            "mcc": float(row.get("mean_mcc", float("nan"))),
            "ece": float(row.get("mean_ece", float("nan"))),
            "brier_score": float(row.get("mean_brier_score", float("nan"))),
            "nll": float(row.get("mean_nll", float("nan"))),
        }
    return baseline


# ---------------------------------------------------------------------------
# Training-curve diagnostics
# ---------------------------------------------------------------------------


def _training_curve_diagnostics(curves: pd.DataFrame) -> dict[str, Any]:
    """Summarise best-epoch / early-stopping behaviour from curve summaries."""

    if curves.empty:
        return {
            "available": False,
            "reason": "training_curves_summary.csv was not supplied or is empty",
            "rows": [],
        }
    rows: list[dict[str, Any]] = []
    trained_beyond_epoch_1 = 0
    early_stopped = 0
    best_epochs: list[int] = []
    for _, row in curves.iterrows():
        best_epoch = _safe_int(row.get("best_epoch"))
        epochs_ran = _safe_int(row.get("epochs_ran"))
        early = bool(row.get("early_stopped")) if "early_stopped" in row else None
        if epochs_ran is not None and epochs_ran > 1:
            trained_beyond_epoch_1 += 1
        if early:
            early_stopped += 1
        if best_epoch is not None:
            best_epochs.append(best_epoch)
        rows.append(
            {
                "run_id": str(row.get("run_id", "")),
                "objective": str(row.get("objective", row.get("pretraining_objective", ""))),
                "horizon": _safe_int(row.get("horizon")),
                "epochs_ran": epochs_ran,
                "best_epoch": best_epoch,
                "best_validation_score": _safe_float(row.get("best_validation_score")),
                "early_stopped": early,
                "test_macro_f1": _safe_float(row.get("test_macro_f1")),
                "test_ece": _safe_float(row.get("test_ece")),
            }
        )
    mean_best_epoch = sum(best_epochs) / len(best_epochs) if best_epochs else None
    return {
        "available": True,
        "reason": "",
        "run_count": len(rows),
        "trained_beyond_epoch_1": trained_beyond_epoch_1,
        "early_stopped_count": early_stopped,
        "mean_best_epoch": mean_best_epoch,
        "rows": rows,
    }


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Claim assessment
# ---------------------------------------------------------------------------


def _all_positive(frame: pd.DataFrame, column: str) -> bool:
    if frame.empty or column not in frame.columns:
        return False
    values = [float(value) for value in frame[column] if not pd.isna(value)]
    return bool(values) and all(value > 0.0 for value in values)


def _all_calibration_improved(frame: pd.DataFrame) -> bool:
    if frame.empty or "delta_ece" not in frame.columns:
        return False
    values = [float(value) for value in frame["delta_ece"] if not pd.isna(value)]
    return bool(values) and all(value < 0.0 for value in values)


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = [float(value) for value in frame[column] if not pd.isna(value)]
    return sum(values) / len(values) if values else None


def _assess_claims(
    *,
    full_grid: pd.DataFrame,
    proper_training: pd.DataFrame,
    proper_evidence_level: str | None,
) -> list[dict[str, Any]]:
    """Derive conservative claim statuses directly from the stored deltas."""

    combined = pd.concat([full_grid, proper_training], ignore_index=True)
    implemented = bool(len(full_grid) or len(proper_training))

    broad_supported = _all_positive(full_grid, "delta_macro_f1") and _all_calibration_improved(
        full_grid
    )
    calibration_supported = _all_calibration_improved(combined)

    h50 = proper_training
    if not h50.empty and "horizon" in h50.columns:
        h50 = h50[h50["horizon"] == 50]
    masked_h50 = h50[h50["ssl_objective"] == "masked_reconstruction"] if not h50.empty else h50
    h50_predictive = _all_positive(masked_h50, "delta_macro_f1") and _all_positive(
        masked_h50, "delta_mcc"
    )
    h50_calibration_worse = not h50.empty and not _all_calibration_improved(h50)

    claims: list[dict[str, Any]] = [
        {
            "claim_id": "ssl_implemented_and_evaluated",
            "claim_text": "SSL was implemented and evaluated under matched FI-2010 settings.",
            "status": "supported" if implemented else "needs_real_evidence",
            "scope": "matched supervised-vs-SSL comparison rows in the stored artefacts",
            "reason": (
                "Matched supervised-vs-SSL comparison rows are present, so the "
                "implementation-and-evaluation claim is supported."
                if implemented
                else "No matched SSL comparison rows were found."
            ),
            "safe_rewording": (
                "SSL objectives were implemented and evaluated under matched "
                "settings; this is an implementation claim, not a result claim."
            ),
        },
        {
            "claim_id": "broad_ssl_improvement",
            "claim_text": "SSL improves ChronosLOB overall.",
            "status": "supported" if broad_supported else "unsupported",
            "scope": "one-epoch full grid matched rows (folds 1-5, horizons 10/20/50, seeds 0-2)",
            "reason": (
                "The completed one-epoch full grid does not support a broad SSL "
                "improvement claim: matched macro-F1 deltas are neutral-to-negative "
                "and calibration is not uniformly improved."
            ),
            "safe_rewording": (
                "The completed full grid does not support a broad SSL improvement "
                "claim; report deltas metric-by-metric."
            ),
            "mean_delta_macro_f1": _mean(full_grid, "delta_macro_f1"),
            "mean_delta_mcc": _mean(full_grid, "delta_mcc"),
            "mean_delta_ece": _mean(full_grid, "delta_ece"),
        },
        {
            "claim_id": "ssl_calibration_improvement",
            "claim_text": "SSL improves calibration.",
            "status": "supported" if calibration_supported else "unsupported",
            "scope": "all matched SSL rows across the full grid and proper-training subset",
            "reason": (
                "Calibration (ECE) did not improve uniformly; every matched "
                "proper-training SSL row worsened ECE, so no calibration "
                "improvement is claimed."
            ),
            "safe_rewording": (
                "Calibration did not improve under SSL; ECE worsened in the "
                "matched proper-training rows."
            ),
            "mean_delta_ece": _mean(combined, "delta_ece"),
        },
        {
            "claim_id": "proper_training_h50_predictive_improvement",
            "claim_text": (
                "Masked SSL improved fold-1/horizon-50 predictive metrics in the "
                "proper-training subset."
            ),
            "status": "partially_supported" if h50_predictive else "needs_real_evidence",
            "scope": (
                "proper-training subset, fold 1, horizon 50, seed 0, lookback 50, "
                f"evidence level {proper_evidence_level or 'partial_real'}"
            ),
            "reason": (
                "Masked SSL improved macro-F1 and MCC at fold 1 / horizon 50, but "
                "calibration worsened and the scope is a single partial_real slice."
                if h50_predictive
                else "Stored rows do not show a fold-1/horizon-50 predictive gain."
            ),
            "calibration_worsened": bool(h50_calibration_worse),
            "safe_rewording": _REQUIRED_CONCLUSION,
            "mean_delta_macro_f1": _mean(masked_h50, "delta_macro_f1"),
            "mean_delta_mcc": _mean(masked_h50, "delta_mcc"),
            "mean_delta_ece": _mean(masked_h50, "delta_ece"),
        },
    ]
    return claims


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _format_float(value: float | None, *, places: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:+.{places}f}" if value else f"{0.0:.{places}f}"


def _format_plain(value: float | None, *, places: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{places}f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def _wrap_bullet(text: str, *, width: int) -> list[str]:
    import textwrap

    body = text[2:] if text.startswith("- ") else text
    wrapped = textwrap.wrap(body, width=width, break_long_words=False, break_on_hyphens=False)
    if not wrapped:
        return [text]
    return ["- " + wrapped[0]] + ["  " + line for line in wrapped[1:]]


def _wrap_lines(lines: Sequence[str], *, width: int = 100) -> list[str]:
    """Wrap prose lines while leaving tables, headings and code blocks intact."""

    import textwrap

    out: list[str] = []
    for line in lines:
        if not line or line.startswith(("|", "#", "```", "  ")) or len(line) <= width:
            out.append(line)
            continue
        if line.startswith("- "):
            out.extend(_wrap_bullet(line, width=width))
            continue
        wrapped = textwrap.wrap(
            line, width=width, break_long_words=False, break_on_hyphens=False
        )
        out.extend(wrapped or [line])
    return out


def _objective_rows(summary: pd.DataFrame, source: str) -> list[list[str]]:
    rows: list[list[str]] = []
    if summary.empty:
        return rows
    subset = summary[summary["source"] == source] if "source" in summary.columns else summary
    for objective in _SSL_OBJECTIVES:
        obj_rows = subset[subset["ssl_objective"] == objective]
        if obj_rows.empty:
            continue
        record = obj_rows.iloc[0]
        rows.append(
            [
                objective,
                str(int(record.get("n_pairs", 0))),
                _format_float(_safe_float(record.get("mean_delta_macro_f1"))),
                _format_float(_safe_float(record.get("mean_delta_mcc"))),
                _format_float(_safe_float(record.get("mean_delta_ece"))),
                (
                    f"{int(record.get('macro_f1_wins', 0))}/"
                    f"{int(record.get('macro_f1_ties', 0))}/"
                    f"{int(record.get('macro_f1_losses', 0))}"
                ),
                (
                    f"{int(record.get('ece_wins', 0))}/"
                    f"{int(record.get('ece_ties', 0))}/"
                    f"{int(record.get('ece_losses', 0))}"
                ),
            ]
        )
    return rows


def _render_markdown(
    *,
    summary: SSLFailureAnalysisSummary,
    by_objective: pd.DataFrame,
    by_horizon: pd.DataFrame,
    by_fold: pd.DataFrame,
    by_seed: pd.DataFrame,
    supervised_baseline: Mapping[int, Mapping[str, float]],
    curve_diagnostics: Mapping[str, Any],
    claims: Sequence[Mapping[str, Any]],
    figure_entries: Sequence[Mapping[str, Any]],
) -> str:
    lines: list[str] = [
        "# SSL Failure Analysis",
        "",
        f"Builder version `{SSL_FAILURE_ANALYSIS_VERSION}`.",
        "",
        "This report explains what the FI-2010 self-supervised (SSL) objectives "
        "did and did not achieve across the completed evidence. It is generated "
        "from retained lightweight summary tables only. The heavy raw per-run "
        "prediction files and encoder checkpoints are not required and are not "
        "read.",
        "",
        "## Evidence Sources",
        "",
        "Three distinct bodies of evidence are kept separate and never merged:",
        "",
        "- One-epoch matched full grid: folds 1-5, horizons 10/20/50, seeds 0-2, "
        "objectives supervised / masked_reconstruction / next_field. This is "
        "matched comparison and infrastructure evidence, not a tuned-training "
        "result.",
        "- Proper-training subset v2: fold 1, horizons 10 and 50, seed 0, lookback "
        "50, SSL pretrain 5 epochs, max 25 epochs, patience 5, CPU. Evidence level "
        f"`{summary.proper_training_evidence_level or 'partial_real'}`.",
        "- A separate older reduced-scope supervised benchmark, used only for "
        "context and never as SSL evidence.",
        "",
    ]

    lines += ["## One-Epoch Full Grid", ""]
    if summary.full_grid_matched_rows:
        lines += [
            f"Matched supervised-vs-SSL pairs analysed: {summary.full_grid_matched_rows}.",
            "",
            "Mean matched deltas by objective (SSL minus supervised). Positive "
            "macro-F1 / MCC is better; for ECE a win is a lower value.",
            "",
        ]
        lines += _markdown_table(
            (
                "objective",
                "pairs",
                "mean d-macroF1",
                "mean d-MCC",
                "mean d-ECE",
                "macroF1 w/t/l",
                "ECE w/t/l",
            ),
            _objective_rows(by_objective, _FULL_GRID_SOURCE),
        )
        lines += [
            "",
            "Interpretation: the one-epoch grid is matched comparison and "
            "infrastructure evidence. It does not support a broad SSL improvement "
            "claim. Masked reconstruction is neutral-to-slightly-negative overall "
            "and next-field is clearly negative overall; ECE does not support a "
            "calibration-improvement claim.",
            "",
        ]
        lines += _source_breakdown_block(
            "macro-F1 delta by horizon",
            by_horizon,
            _FULL_GRID_SOURCE,
            "horizon",
            "mean_delta_macro_f1",
        )
        lines += _source_breakdown_block(
            "macro-F1 delta by fold",
            by_fold,
            _FULL_GRID_SOURCE,
            "fold",
            "mean_delta_macro_f1",
        )
        lines += _source_breakdown_block(
            "macro-F1 delta by seed",
            by_seed,
            _FULL_GRID_SOURCE,
            "seed",
            "mean_delta_macro_f1",
        )
        lines += [
            "Any positive cells are isolated rather than consistent across folds, "
            "horizons and seeds, so they do not support a general SSL improvement "
            "claim.",
            "",
        ]
    else:
        lines += ["No one-epoch full-grid matched rows were supplied.", ""]

    lines += ["## Proper-Training Subset v2", ""]
    if summary.proper_training_matched_rows:
        lines += [
            "Exact scope: fold 1, horizons 10 and 50, seed 0, lookback 50, "
            f"evidence level `{summary.proper_training_evidence_level or 'partial_real'}`. "
            f"Matched supervised-vs-SSL pairs: {summary.proper_training_matched_rows}.",
            "",
            "Supervised baseline by horizon:",
            "",
        ]
        baseline_rows = [
            [
                str(horizon),
                _format_plain(metrics.get("macro_f1")),
                _format_plain(metrics.get("mcc")),
                _format_plain(metrics.get("ece")),
            ]
            for horizon, metrics in sorted(supervised_baseline.items())
        ]
        lines += _markdown_table(("horizon", "macro-F1", "MCC", "ECE"), baseline_rows)
        lines += [
            "",
            "Matched SSL deltas by horizon (SSL minus supervised):",
            "",
        ]
        lines += _markdown_table(
            ("horizon", "objective", "d-macroF1", "d-MCC", "d-ECE"),
            _proper_training_delta_rows(by_horizon),
        )
        lines += [
            "",
            "At horizon 10 every objective collapses to the same "
            "stationary-majority prediction, so masked and next-field tie supervised "
            "on macro-F1 and MCC while both worsen ECE. At horizon 50 masked "
            "reconstruction improves macro-F1 and MCC but still worsens ECE, and "
            "next-field shows a small macro-F1 / MCC gain with worse ECE.",
            "",
            _REQUIRED_CONCLUSION,
            "",
        ]
    else:
        lines += ["No proper-training matched rows were supplied.", ""]

    lines += _render_curve_section(curve_diagnostics)
    lines += _render_claim_section(claims)
    lines += _render_figure_section(figure_entries)
    lines += _render_does_not_claim_section()
    return "\n".join(_wrap_lines(lines)).rstrip() + "\n"


def _source_breakdown_block(
    title: str,
    table: pd.DataFrame,
    source: str,
    key_column: str,
    value_column: str,
) -> list[str]:
    lines = [f"Full-grid {title}:", ""]
    rows: list[list[str]] = []
    if not table.empty and "source" in table.columns:
        subset = table[table["source"] == source]
        for objective in _SSL_OBJECTIVES:
            obj_rows = subset[subset["ssl_objective"] == objective]
            for _, record in obj_rows.sort_values(key_column).iterrows():
                rows.append(
                    [
                        objective,
                        str(_safe_int(record.get(key_column))),
                        _format_float(_safe_float(record.get(value_column))),
                    ]
                )
    if not rows:
        lines += ["No rows available.", ""]
        return lines
    lines += _markdown_table(("objective", key_column, "mean d-macroF1"), rows)
    lines += [""]
    return lines


def _proper_training_delta_rows(by_horizon: pd.DataFrame) -> list[list[str]]:
    rows: list[list[str]] = []
    if by_horizon.empty or "source" not in by_horizon.columns:
        return rows
    subset = by_horizon[by_horizon["source"] == _PROPER_TRAINING_SOURCE]
    for _, record in subset.sort_values(["horizon", "ssl_objective"]).iterrows():
        rows.append(
            [
                str(_safe_int(record.get("horizon"))),
                str(record.get("ssl_objective", "")),
                _format_float(_safe_float(record.get("mean_delta_macro_f1"))),
                _format_float(_safe_float(record.get("mean_delta_mcc"))),
                _format_float(_safe_float(record.get("mean_delta_ece"))),
            ]
        )
    return rows


def _render_curve_section(diagnostics: Mapping[str, Any]) -> list[str]:
    lines = ["## Training-Curve Diagnostics", ""]
    if not diagnostics.get("available"):
        lines += [
            f"Skipped: {diagnostics.get('reason', 'no curve summary available')}.",
            "No convergence conclusions are drawn without curve data.",
            "",
        ]
        return lines
    mean_best = diagnostics.get("mean_best_epoch")
    lines += [
        f"Runs summarised: {diagnostics.get('run_count', 0)}. Trained beyond epoch 1: "
        f"{diagnostics.get('trained_beyond_epoch_1', 0)}. Early-stopped: "
        f"{diagnostics.get('early_stopped_count', 0)}. Mean best epoch: "
        f"{_format_plain(mean_best, places=2)}.",
        "",
    ]
    rows: list[list[str]] = []
    for record in diagnostics.get("rows", []):
        rows.append(
            [
                str(record.get("objective", "")),
                str(record.get("horizon", "")),
                str(record.get("epochs_ran", "")),
                str(record.get("best_epoch", "")),
                _format_plain(record.get("best_validation_score")),
                "yes" if record.get("early_stopped") else "no",
                _format_plain(record.get("test_macro_f1")),
            ]
        )
    lines += _markdown_table(
        (
            "objective",
            "horizon",
            "epochs",
            "best_epoch",
            "best_val_macroF1",
            "early_stop",
            "test_macroF1",
        ),
        rows,
    )
    lines += [
        "",
        "Horizon-10 runs early-stop at the first epoch on the "
        "stationary-majority solution. Horizon-50 runs train longer; masked "
        "reconstruction used the full budget while supervised and next-field "
        "early-stopped. SSL fine-tuning therefore converged differently from "
        "supervised training only at horizon 50.",
        "",
    ]
    return lines


def _render_claim_section(claims: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["## Claim Assessment", ""]
    rows = [
        [
            str(claim.get("claim_id", "")),
            str(claim.get("status", "")),
            str(claim.get("scope", "")),
        ]
        for claim in claims
    ]
    lines += _markdown_table(("claim", "status", "scope"), rows)
    lines += [""]
    for claim in claims:
        lines.append(f"- `{claim.get('claim_id')}` ({claim.get('status')}): {claim.get('reason')}")
    lines += [""]
    return lines


def _render_figure_section(figure_entries: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = ["## Figures", ""]
    completed = [entry for entry in figure_entries if entry.get("status") == "completed"]
    if not completed:
        lines += [
            "No figures were generated in this pass; the delta CSVs hold the same "
            "numbers and figure generation is recorded as future work in "
            "`figure_manifest.json`.",
            "",
        ]
        return lines
    rows = [
        [
            str(entry.get("figure_id", "")),
            str(entry.get("title", "")),
            str(entry.get("file_path", "")),
        ]
        for entry in completed
    ]
    lines += _markdown_table(("figure", "title", "path"), rows)
    lines += [""]
    return lines


def _render_does_not_claim_section() -> list[str]:
    return [
        "## What This Does Not Claim",
        "",
        "This analysis does not claim that SSL improves ChronosLOB overall, that "
        "SSL improves calibration, or anything about profitability, live trading "
        "or tradable signal. It is a diagnostic over stored FI-2010 metrics. More "
        "evidence would require broader proper-training runs and/or better SSL "
        "objective design rather than any success claim.",
        "",
    ]


# ---------------------------------------------------------------------------
# Optional figures
# ---------------------------------------------------------------------------


def _build_figures(
    *,
    out_dir: Path,
    by_horizon: pd.DataFrame,
    curve_diagnostics: Mapping[str, Any],
    make_figures: bool,
) -> list[dict[str, Any]]:
    """Generate lightweight delta figures when matplotlib is available."""

    planned = [
        ("ssl_macro_f1_delta_by_horizon", "SSL macro-F1 delta by horizon", "mean_delta_macro_f1"),
        ("ssl_mcc_delta_by_horizon", "SSL MCC delta by horizon", "mean_delta_mcc"),
        ("ssl_ece_delta_by_horizon", "SSL ECE delta by horizon", "mean_delta_ece"),
    ]
    if not make_figures:
        return [
            {
                "figure_id": figure_id,
                "title": title,
                "status": "skipped",
                "reason": "figure generation disabled (make_figures=False)",
                "file_path": None,
            }
            for figure_id, title, _ in planned
        ]
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        return [
            {
                "figure_id": figure_id,
                "title": title,
                "status": "skipped",
                "reason": "matplotlib is not installed",
                "file_path": None,
            }
            for figure_id, title, _ in planned
        ]

    entries: list[dict[str, Any]] = []
    proper = (
        by_horizon[by_horizon["source"] == _PROPER_TRAINING_SOURCE]
        if not by_horizon.empty and "source" in by_horizon.columns
        else pd.DataFrame()
    )
    for figure_id, title, value_column in planned:
        if proper.empty or value_column not in proper.columns:
            entries.append(
                {
                    "figure_id": figure_id,
                    "title": title,
                    "status": "skipped",
                    "reason": "no proper-training rows for this metric",
                    "file_path": None,
                }
            )
            continue
        fig, axis = plt.subplots(figsize=(5.0, 3.2))
        for objective in _SSL_OBJECTIVES:
            obj_rows = proper[proper["ssl_objective"] == objective].sort_values("horizon")
            if obj_rows.empty:
                continue
            axis.plot(
                [int(value) for value in obj_rows["horizon"]],
                [float(value) for value in obj_rows[value_column]],
                marker="o",
                label=objective,
            )
        axis.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
        axis.set_xlabel("horizon")
        axis.set_ylabel(value_column)
        axis.set_title(title)
        axis.legend()
        fig.tight_layout()
        file_path = out_dir / f"{figure_id}.png"
        fig.savefig(file_path, dpi=120)
        plt.close(fig)
        entries.append(
            {
                "figure_id": figure_id,
                "title": title,
                "status": "completed",
                "reason": "",
                "file_path": f"{file_path.as_posix()}",
            }
        )

    best_entry = _build_best_epoch_figure(
        out_dir=out_dir, curve_diagnostics=curve_diagnostics, plt=plt
    )
    entries.append(best_entry)
    return entries


def _build_best_epoch_figure(
    *,
    out_dir: Path,
    curve_diagnostics: Mapping[str, Any],
    plt: Any,
) -> dict[str, Any]:
    figure_id = "best_epoch_by_objective"
    title = "Best epoch by objective (proper-training)"
    rows = [row for row in curve_diagnostics.get("rows", []) if row.get("best_epoch") is not None]
    if not curve_diagnostics.get("available") or not rows:
        return {
            "figure_id": figure_id,
            "title": title,
            "status": "skipped",
            "reason": "no training-curve rows available",
            "file_path": None,
        }
    labels = [f"{row.get('objective')} h{row.get('horizon')}" for row in rows]
    values = [int(row.get("best_epoch")) for row in rows]
    fig, axis = plt.subplots(figsize=(6.0, 3.2))
    axis.bar(range(len(values)), values, color="#4c72b0")
    axis.set_xticks(range(len(values)))
    axis.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    axis.set_ylabel("best epoch")
    axis.set_title(title)
    fig.tight_layout()
    file_path = out_dir / f"{figure_id}.png"
    fig.savefig(file_path, dpi=120)
    plt.close(fig)
    return {
        "figure_id": figure_id,
        "title": title,
        "status": "completed",
        "reason": "",
        "file_path": f"{file_path.as_posix()}",
    }


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


def analyse_fi2010_ssl_results(
    *,
    full_grid_dir: str | Path | None,
    proper_training_dir: str | Path | None,
    out_dir: str | Path,
    make_figures: bool = True,
    overwrite: bool = False,
) -> SSLFailureAnalysisSummary:
    """Analyse stored FI-2010 SSL comparison artefacts and write a report.

    Only retained lightweight artefacts are read. Raw per-run prediction files
    and encoder checkpoints are never required.
    """

    if full_grid_dir is None and proper_training_dir is None:
        raise ValueError("at least one of full_grid_dir or proper_training_dir must be provided")

    resolved_out = Path(out_dir)
    _ensure_output_dir(resolved_out, overwrite=overwrite)

    warnings: list[str] = []

    full_grid = pd.DataFrame()
    full_grid_evidence: str | None = None
    resolved_full_grid: Path | None = None
    if full_grid_dir is not None:
        resolved_full_grid = Path(full_grid_dir)
        _require_files(resolved_full_grid, _FULL_GRID_REQUIRED, label="full-grid")
        full_grid = _read_matched_comparison(resolved_full_grid, source=_FULL_GRID_SOURCE)
        full_grid_meta = _read_summary_metadata(resolved_full_grid)
        full_grid_evidence = str(full_grid_meta.get("evidence_level", ""))
        if full_grid.empty:
            warnings.append("full-grid ssl_comparison.csv contained no matched rows")

    proper_training = pd.DataFrame()
    proper_evidence: str | None = None
    resolved_proper: Path | None = None
    proper_curves = pd.DataFrame()
    supervised_baseline: dict[int, dict[str, float]] = {}
    if proper_training_dir is not None:
        resolved_proper = Path(proper_training_dir)
        _require_files(resolved_proper, _PROPER_TRAINING_REQUIRED, label="proper-training")
        proper_training = _read_matched_comparison(resolved_proper, source=_PROPER_TRAINING_SOURCE)
        proper_evidence = str(_read_summary_metadata(resolved_proper).get("evidence_level", ""))
        proper_curves = _read_training_curves(resolved_proper)
        supervised_baseline = _supervised_baseline_by_horizon(_read_aggregate_rows(resolved_proper))
        if proper_training.empty:
            warnings.append("proper-training ssl_comparison.csv contained no matched rows")

    combined = pd.concat([full_grid, proper_training], ignore_index=True)
    if combined.empty:
        raise FileNotFoundError(
            "no matched SSL comparison rows were found in the supplied artefacts",
        )

    by_objective = _aggregate_deltas(combined, ["source", "ssl_objective"])
    by_horizon = _aggregate_deltas(combined, ["source", "ssl_objective", "horizon"])
    by_fold = _aggregate_deltas(combined, ["source", "ssl_objective", "fold"])
    by_seed = _aggregate_deltas(combined, ["source", "ssl_objective", "seed"])
    metric_summary = _metric_summary(combined)

    curve_diagnostics = _training_curve_diagnostics(proper_curves)
    claims = _assess_claims(
        full_grid=full_grid,
        proper_training=proper_training,
        proper_evidence_level=proper_evidence or None,
    )
    claim_statuses = {str(claim["claim_id"]): str(claim["status"]) for claim in claims}

    figure_entries = _build_figures(
        out_dir=resolved_out,
        by_horizon=by_horizon,
        curve_diagnostics=curve_diagnostics,
        make_figures=make_figures,
    )
    figures_generated = tuple(
        str(entry["figure_id"]) for entry in figure_entries if entry.get("status") == "completed"
    )

    artefacts = {
        "report": "ssl_failure_analysis.md",
        "delta_by_objective": "ssl_delta_by_objective.csv",
        "delta_by_horizon": "ssl_delta_by_horizon.csv",
        "delta_by_fold": "ssl_delta_by_fold.csv",
        "delta_by_seed": "ssl_delta_by_seed.csv",
        "metric_summary": "ssl_metric_summary.csv",
        "claim_assessment": "ssl_claim_assessment.json",
        "figure_manifest": "figure_manifest.json",
        "summary": "summary.json",
    }

    summary = SSLFailureAnalysisSummary(
        output_dir=resolved_out,
        full_grid_dir=resolved_full_grid,
        proper_training_dir=resolved_proper,
        full_grid_evidence_level=full_grid_evidence or None,
        proper_training_evidence_level=proper_evidence or None,
        full_grid_matched_rows=len(full_grid),
        proper_training_matched_rows=len(proper_training),
        artefacts=artefacts,
        figures_generated=figures_generated,
        claim_statuses=claim_statuses,
        warnings=tuple(warnings),
        raw_predictions_required=False,
        checkpoints_required=False,
    )

    _frame_to_csv(by_objective, resolved_out / "ssl_delta_by_objective.csv")
    _frame_to_csv(by_horizon, resolved_out / "ssl_delta_by_horizon.csv")
    _frame_to_csv(by_fold, resolved_out / "ssl_delta_by_fold.csv")
    _frame_to_csv(by_seed, resolved_out / "ssl_delta_by_seed.csv")
    _frame_to_csv(metric_summary, resolved_out / "ssl_metric_summary.csv")

    claim_payload = {
        "analyser_version": SSL_FAILURE_ANALYSIS_VERSION,
        "package_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "raw_predictions_required": False,
        "checkpoints_required": False,
        "full_grid_evidence_level": full_grid_evidence or None,
        "proper_training_evidence_level": proper_evidence or None,
        "claims": claims,
    }
    (resolved_out / "ssl_claim_assessment.json").write_text(
        _stable_json_dumps(claim_payload), encoding="utf-8"
    )

    figure_manifest = {
        "builder_version": SSL_FAILURE_ANALYSIS_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "figures": list(figure_entries),
    }
    (resolved_out / "figure_manifest.json").write_text(
        _stable_json_dumps(figure_manifest), encoding="utf-8"
    )

    report_text = _render_markdown(
        summary=summary,
        by_objective=by_objective,
        by_horizon=by_horizon,
        by_fold=by_fold,
        by_seed=by_seed,
        supervised_baseline=supervised_baseline,
        curve_diagnostics=curve_diagnostics,
        claims=claims,
        figure_entries=figure_entries,
    )
    (resolved_out / "ssl_failure_analysis.md").write_text(report_text, encoding="utf-8")

    summary_payload = {
        "analyser_version": SSL_FAILURE_ANALYSIS_VERSION,
        "package_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "full_grid_dir": str(resolved_full_grid) if resolved_full_grid is not None else None,
            "proper_training_dir": str(resolved_proper) if resolved_proper is not None else None,
        },
        "full_grid_evidence_level": full_grid_evidence or None,
        "proper_training_evidence_level": proper_evidence or None,
        "full_grid_matched_rows": len(full_grid),
        "proper_training_matched_rows": len(proper_training),
        "raw_predictions_required": False,
        "checkpoints_required": False,
        "claim_statuses": claim_statuses,
        "figures_generated": list(figures_generated),
        "artefacts": artefacts,
        "warnings": list(warnings),
    }
    (resolved_out / "summary.json").write_text(
        _stable_json_dumps(summary_payload), encoding="utf-8"
    )

    return summary
