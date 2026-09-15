"""Statistical uncertainty analysis for FI-2010 multi-fold artefacts.

This module loads classical and supervised neural per-fold metric tables
that the multi-fold runners write, then quantifies fold-level variance
without re-running any model. It produces aggregate metrics with
confidence intervals, paired fold-level comparisons against a baseline,
bootstrap intervals over folds and rank stability counts.

The module is intentionally read-only with respect to upstream
artefacts. Probability-based metrics that the upstream runner could not
compute (for example ``brier_score`` and ``ece`` for ``ridge``) are
treated as missing and dropped from per-metric statistics.

It is a diagnostic layer. It does not promote any model to a tradable
or production-deployment status, and it does not claim
state-of-the-art performance.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chronoslob import __version__

__all__ = [
    "DEFAULT_UNCERTAINTY_METRICS",
    "FI2010_UNCERTAINTY_VERSION",
    "UncertaintyAnalysisSummary",
    "analyse_fi2010_uncertainty",
    "bootstrap_mean_confidence_interval",
    "compute_metric_confidence_intervals",
    "compute_paired_model_comparisons",
    "compute_rank_stability",
    "load_classical_fold_results",
    "load_neural_fold_results",
]

FI2010_UNCERTAINTY_VERSION = "phase-f/fi2010-uncertainty/v1"

DEFAULT_UNCERTAINTY_METRICS: tuple[str, ...] = (
    "accuracy",
    "macro_f1",
    "mcc",
    "brier_score",
    "ece",
)

_CLASSICAL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "fold_id",
    "model_name",
    "split",
    "accuracy",
    "macro_f1",
    "mcc",
)

_NEURAL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "fold_id",
    "seed",
    "model_name",
    "lookback",
    "split",
    "accuracy",
    "macro_f1",
    "mcc",
)

_METRIC_CI_COLUMNS: tuple[str, ...] = (
    "source",
    "model_name",
    "lookback",
    "split",
    "metric",
    "n_folds",
    "n_seeds",
    "mean",
    "std",
    "standard_error",
    "ci_level",
    "ci_lower",
    "ci_upper",
    "bootstrap_lower",
    "bootstrap_upper",
    "bootstrap_iterations",
    "n_missing",
)

_PAIRED_COMPARISON_COLUMNS: tuple[str, ...] = (
    "source",
    "split",
    "metric",
    "baseline_model",
    "candidate_model",
    "lookback",
    "n_folds",
    "mean_difference",
    "std_difference",
    "standard_error",
    "ci_level",
    "ci_lower",
    "ci_upper",
    "wins",
    "losses",
    "ties",
)

_RANK_STABILITY_COLUMNS: tuple[str, ...] = (
    "source",
    "split",
    "metric",
    "model_name",
    "lookback",
    "n_folds",
    "best_count",
    "best_fraction",
    "mean_rank",
    "rank_std",
)

_RANKING_COLUMNS: tuple[str, ...] = (
    "source",
    "split",
    "metric",
    "rank",
    "model_name",
    "lookback",
    "n_folds",
    "mean",
    "standard_error",
    "ci_lower",
    "ci_upper",
)


@dataclass(frozen=True)
class UncertaintyAnalysisSummary:
    """Lightweight return type for the analysis entry point."""

    output_dir: Path
    classical_input: Path | None
    neural_input: Path | None
    baseline_model: str
    ci_level: float
    bootstrap_iterations: int
    bootstrap_seed: int
    metrics: tuple[str, ...]
    classical_models: tuple[str, ...] = ()
    neural_models: tuple[str, ...] = ()
    classical_folds: tuple[str, ...] = ()
    neural_folds: tuple[str, ...] = ()
    neural_seeds: tuple[int, ...] = ()
    neural_lookbacks: tuple[int, ...] = ()
    artefacts: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    classical_seed_variance_available: bool = False
    neural_seed_variance_available: bool = False


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"results file not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"results file is empty: {path}")
    return frame


def _check_required_columns(
    frame: pd.DataFrame, required: Sequence[str], *, source: str
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{source} results frame is missing required columns: {missing}",
        )


def _coerce_status(frame: pd.DataFrame) -> pd.DataFrame:
    if "status" not in frame.columns:
        return frame
    status_lower = frame["status"].astype(str).str.lower()
    return frame.loc[status_lower == "ok"].copy()


def load_classical_fold_results(path: str | Path) -> pd.DataFrame:
    """Load the classical multi-fold per-fold metrics table."""
    frame = _read_csv(Path(path))
    _check_required_columns(frame, _CLASSICAL_REQUIRED_COLUMNS, source="classical")
    frame = _coerce_status(frame)
    if "seed" not in frame.columns:
        frame = frame.assign(seed=0)
    frame["fold_id"] = frame["fold_id"].apply(_normalise_fold_id)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce").fillna(0).astype(int)
    frame["lookback"] = pd.NA
    return frame


def load_neural_fold_results(path: str | Path) -> pd.DataFrame:
    """Load the neural multi-fold per-fold/seed/lookback metrics table."""
    frame = _read_csv(Path(path))
    _check_required_columns(frame, _NEURAL_REQUIRED_COLUMNS, source="neural")
    frame = _coerce_status(frame)
    frame["fold_id"] = frame["fold_id"].apply(_normalise_fold_id)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce").fillna(0).astype(int)
    frame["lookback"] = pd.to_numeric(frame["lookback"], errors="coerce").astype("Int64")
    return frame


def _normalise_fold_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.isdigit():
        return f"fold_{int(text)}"
    return text


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------


def _student_t_critical(confidence: float, degrees_of_freedom: int) -> float:
    """Approximate two-sided Student-t critical value.

    We avoid a SciPy dependency. For small samples we use a small lookup
    that covers the common ``df = 1..10`` cases and fall back to the
    normal approximation otherwise. Confidence is the central area
    (e.g. ``0.95`` -> upper-tail alpha/2 = 0.025).
    """
    if degrees_of_freedom <= 0:
        return float("nan")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence!r}")
    table: Mapping[float, Mapping[int, float]] = {
        0.80: {
            1: 3.078,
            2: 1.886,
            3: 1.638,
            4: 1.533,
            5: 1.476,
            6: 1.440,
            7: 1.415,
            8: 1.397,
            9: 1.383,
            10: 1.372,
        },
        0.90: {
            1: 6.314,
            2: 2.920,
            3: 2.353,
            4: 2.132,
            5: 2.015,
            6: 1.943,
            7: 1.895,
            8: 1.860,
            9: 1.833,
            10: 1.812,
        },
        0.95: {
            1: 12.706,
            2: 4.303,
            3: 3.182,
            4: 2.776,
            5: 2.571,
            6: 2.447,
            7: 2.365,
            8: 2.306,
            9: 2.262,
            10: 2.228,
        },
        0.99: {
            1: 63.657,
            2: 9.925,
            3: 5.841,
            4: 4.604,
            5: 4.032,
            6: 3.707,
            7: 3.499,
            8: 3.355,
            9: 3.250,
            10: 3.169,
        },
    }
    normal_quantile = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    if confidence in table and degrees_of_freedom in table[confidence]:
        return table[confidence][degrees_of_freedom]
    if confidence in normal_quantile:
        return normal_quantile[confidence]
    raise ValueError(
        f"confidence {confidence!r} is not supported; "
        "use 0.80, 0.90, 0.95 or 0.99",
    )


def _normal_critical(confidence: float) -> float:
    quantiles = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    if confidence not in quantiles:
        raise ValueError(
            f"confidence {confidence!r} is not supported; "
            "use 0.80, 0.90, 0.95 or 0.99",
        )
    return quantiles[confidence]


def _finite_values(values: Iterable[Any]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return array
    return array[np.isfinite(array)]


def bootstrap_mean_confidence_interval(
    values: Sequence[float] | Iterable[float],
    *,
    iterations: int,
    ci_level: float,
    seed: int,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean using a fixed random seed."""
    array = _finite_values(values)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if array.size == 0:
        return (float("nan"), float("nan"))
    if array.size == 1:
        return (float(array[0]), float(array[0]))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(iterations, array.size))
    means = array[indices].mean(axis=1)
    lower_pct = (1.0 - ci_level) / 2.0 * 100.0
    upper_pct = 100.0 - lower_pct
    lower = float(np.percentile(means, lower_pct))
    upper = float(np.percentile(means, upper_pct))
    return lower, upper


def _summarise_fold_metric(
    fold_values: pd.DataFrame,
    *,
    metric: str,
    ci_level: float,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> Mapping[str, Any]:
    raw_values = fold_values[metric].to_numpy(dtype=float, copy=True, na_value=np.nan)
    finite = raw_values[np.isfinite(raw_values)]
    n_missing = int(np.sum(~np.isfinite(raw_values)))
    n = int(finite.size)
    if n == 0:
        return {
            "n_folds": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "standard_error": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "bootstrap_lower": float("nan"),
            "bootstrap_upper": float("nan"),
            "bootstrap_iterations": bootstrap_iterations,
            "n_missing": n_missing,
        }
    mean_value = float(finite.mean())
    if n >= 2:
        std_value = float(finite.std(ddof=1))
        standard_error = std_value / math.sqrt(n)
        critical = _student_t_critical(ci_level, n - 1)
        half_width = critical * standard_error
        ci_lower = mean_value - half_width
        ci_upper = mean_value + half_width
    else:
        std_value = 0.0
        standard_error = 0.0
        ci_lower = mean_value
        ci_upper = mean_value
    bootstrap_lower, bootstrap_upper = bootstrap_mean_confidence_interval(
        finite,
        iterations=bootstrap_iterations,
        ci_level=ci_level,
        seed=bootstrap_seed,
    )
    return {
        "n_folds": n,
        "mean": mean_value,
        "std": std_value,
        "standard_error": standard_error,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_lower": bootstrap_lower,
        "bootstrap_upper": bootstrap_upper,
        "bootstrap_iterations": bootstrap_iterations,
        "n_missing": n_missing,
    }


def _fold_aggregated_frame(frame: pd.DataFrame, *, metrics: Sequence[str]) -> pd.DataFrame:
    """Collapse multiple seeds per (model, fold, lookback, split) to a per-fold mean.

    Folds become the unit of variance. If only one seed is present the
    mean is the original value.
    """
    if frame.empty:
        return frame
    has_lookback = "lookback" in frame.columns
    group_cols: list[str] = ["model_name", "split", "fold_id"]
    if has_lookback:
        group_cols.append("lookback")
    aggregated = (
        frame.groupby(group_cols, dropna=False, as_index=False)[list(metrics)]
        .mean(numeric_only=False)
    )
    seed_counts = (
        frame.groupby(group_cols, dropna=False, as_index=False)["seed"]
        .nunique()
        .rename(columns={"seed": "n_seeds"})
    )
    return aggregated.merge(seed_counts, on=group_cols, how="left")


def compute_metric_confidence_intervals(
    frame: pd.DataFrame,
    *,
    source: str,
    metrics: Sequence[str] = DEFAULT_UNCERTAINTY_METRICS,
    ci_level: float = 0.95,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """Aggregate per-fold metrics into mean / std / SE / CI rows."""
    if frame.empty:
        return pd.DataFrame(columns=list(_METRIC_CI_COLUMNS))
    fold_means = _fold_aggregated_frame(frame, metrics=metrics)
    has_lookback = "lookback" in fold_means.columns

    rows: list[Mapping[str, Any]] = []
    grouping_cols: list[str] = ["model_name", "split"]
    if has_lookback:
        grouping_cols.append("lookback")
    for keys, subset in fold_means.groupby(grouping_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(grouping_cols, keys, strict=False))
        for metric in metrics:
            if metric not in subset.columns:
                continue
            summary = _summarise_fold_metric(
                subset,
                metric=metric,
                ci_level=ci_level,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed,
            )
            row: dict[str, Any] = {
                "source": source,
                "model_name": key_map.get("model_name"),
                "lookback": key_map.get("lookback") if has_lookback else None,
                "split": key_map.get("split"),
                "metric": metric,
                "n_seeds": int(subset["n_seeds"].max())
                if "n_seeds" in subset.columns and not subset["n_seeds"].empty
                else 1,
                "ci_level": ci_level,
            }
            row.update(summary)
            rows.append(row)
    out = pd.DataFrame(rows, columns=list(_METRIC_CI_COLUMNS))
    return out


def compute_paired_model_comparisons(
    frame: pd.DataFrame,
    *,
    source: str,
    baseline_model: str,
    metrics: Sequence[str] = DEFAULT_UNCERTAINTY_METRICS,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """Per-metric paired fold differences between every model and the baseline.

    A positive ``mean_difference`` means the candidate scored higher
    than the baseline on average across paired folds. Ties count toward
    neither wins nor losses.
    """
    if frame.empty:
        return pd.DataFrame(columns=list(_PAIRED_COMPARISON_COLUMNS))
    fold_means = _fold_aggregated_frame(frame, metrics=metrics)
    has_lookback = "lookback" in fold_means.columns

    rows: list[Mapping[str, Any]] = []
    splits = sorted(fold_means["split"].dropna().unique())
    for split in splits:
        split_frame = fold_means.loc[fold_means["split"] == split]
        if has_lookback:
            lookbacks = list(split_frame["lookback"].dropna().unique())
            if not lookbacks:
                lookbacks = [None]
        else:
            lookbacks = [None]
        for lookback in lookbacks:
            if lookback is None:
                lookback_frame = split_frame
            else:
                lookback_frame = split_frame.loc[split_frame["lookback"] == lookback]
            baseline_rows = lookback_frame.loc[
                lookback_frame["model_name"] == baseline_model
            ]
            if baseline_rows.empty:
                continue
            baseline_by_fold = baseline_rows.set_index("fold_id")
            for metric in metrics:
                if metric not in lookback_frame.columns:
                    continue
                for candidate_model, candidate_rows in lookback_frame.groupby(
                    "model_name", dropna=False, sort=True
                ):
                    if candidate_model == baseline_model:
                        continue
                    candidate_by_fold = candidate_rows.set_index("fold_id")
                    common_folds = sorted(
                        set(candidate_by_fold.index).intersection(baseline_by_fold.index)
                    )
                    if not common_folds:
                        continue
                    pairs: list[tuple[float, float]] = []
                    for fold in common_folds:
                        baseline_value = baseline_by_fold.at[fold, metric]
                        candidate_value = candidate_by_fold.at[fold, metric]
                        if isinstance(baseline_value, pd.Series):
                            baseline_value = baseline_value.iloc[0]
                        if isinstance(candidate_value, pd.Series):
                            candidate_value = candidate_value.iloc[0]
                        baseline_float = (
                            float(baseline_value)
                            if pd.notna(baseline_value)
                            else float("nan")
                        )
                        candidate_float = (
                            float(candidate_value)
                            if pd.notna(candidate_value)
                            else float("nan")
                        )
                        if math.isfinite(baseline_float) and math.isfinite(candidate_float):
                            pairs.append((candidate_float, baseline_float))
                    if not pairs:
                        continue
                    diffs = np.array(
                        [candidate - baseline for candidate, baseline in pairs],
                        dtype=float,
                    )
                    n_pairs = int(diffs.size)
                    mean_diff = float(diffs.mean())
                    if n_pairs >= 2:
                        std_diff = float(diffs.std(ddof=1))
                        standard_error = std_diff / math.sqrt(n_pairs)
                        critical = _student_t_critical(ci_level, n_pairs - 1)
                        half_width = critical * standard_error
                        ci_lower = mean_diff - half_width
                        ci_upper = mean_diff + half_width
                    else:
                        std_diff = 0.0
                        standard_error = 0.0
                        ci_lower = mean_diff
                        ci_upper = mean_diff
                    wins = int(np.sum(diffs > 0))
                    losses = int(np.sum(diffs < 0))
                    ties = int(np.sum(diffs == 0))
                    rows.append(
                        {
                            "source": source,
                            "split": split,
                            "metric": metric,
                            "baseline_model": baseline_model,
                            "candidate_model": candidate_model,
                            "lookback": lookback,
                            "n_folds": n_pairs,
                            "mean_difference": mean_diff,
                            "std_difference": std_diff,
                            "standard_error": standard_error,
                            "ci_level": ci_level,
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper,
                            "wins": wins,
                            "losses": losses,
                            "ties": ties,
                        }
                    )
    return pd.DataFrame(rows, columns=list(_PAIRED_COMPARISON_COLUMNS))


def compute_rank_stability(
    frame: pd.DataFrame,
    *,
    source: str,
    metrics: Sequence[str] = DEFAULT_UNCERTAINTY_METRICS,
) -> pd.DataFrame:
    """How often each model is best per fold, and rank statistics per model.

    Higher is better for accuracy, macro_f1 and mcc. Lower is better for
    brier_score and ece. The ranker accounts for that direction.
    """
    if frame.empty:
        return pd.DataFrame(columns=list(_RANK_STABILITY_COLUMNS))
    fold_means = _fold_aggregated_frame(frame, metrics=metrics)
    has_lookback = "lookback" in fold_means.columns
    lower_is_better = {"brier_score", "ece"}

    rows: list[Mapping[str, Any]] = []
    splits = sorted(fold_means["split"].dropna().unique())
    for split in splits:
        split_frame = fold_means.loc[fold_means["split"] == split]
        if has_lookback:
            lookbacks = list(split_frame["lookback"].dropna().unique())
            if not lookbacks:
                lookbacks = [None]
        else:
            lookbacks = [None]
        for lookback in lookbacks:
            if lookback is None:
                lookback_frame = split_frame
            else:
                lookback_frame = split_frame.loc[split_frame["lookback"] == lookback]
            for metric in metrics:
                if metric not in lookback_frame.columns:
                    continue
                metric_frame = lookback_frame.dropna(subset=[metric])
                if metric_frame.empty:
                    continue
                ascending = metric in lower_is_better
                folds = sorted(metric_frame["fold_id"].dropna().unique())
                fold_ranks: dict[str, dict[str, float]] = {}
                best_counts: dict[str, int] = {}
                model_names = sorted(metric_frame["model_name"].dropna().unique())
                for model_name in model_names:
                    fold_ranks[model_name] = {}
                    best_counts[model_name] = 0
                fold_count = 0
                for fold in folds:
                    fold_subset = metric_frame.loc[metric_frame["fold_id"] == fold]
                    if fold_subset.empty:
                        continue
                    ranks = (
                        fold_subset[metric]
                        .rank(method="min", ascending=ascending)
                    )
                    fold_subset = fold_subset.assign(_rank=ranks.values)
                    if fold_subset["_rank"].isna().all():
                        continue
                    fold_count += 1
                    best_row = fold_subset.loc[fold_subset["_rank"].idxmin()]
                    best_model = str(best_row["model_name"])
                    best_counts[best_model] = best_counts.get(best_model, 0) + 1
                    for _, row in fold_subset.iterrows():
                        model_name = str(row["model_name"])
                        fold_ranks.setdefault(model_name, {})[str(fold)] = float(
                            row["_rank"]
                        )
                if fold_count == 0:
                    continue
                for model_name in model_names:
                    rank_values = list(fold_ranks.get(model_name, {}).values())
                    if not rank_values:
                        continue
                    rank_array = np.asarray(rank_values, dtype=float)
                    best_count = best_counts.get(model_name, 0)
                    mean_rank = float(rank_array.mean())
                    rank_std = (
                        float(rank_array.std(ddof=1))
                        if rank_array.size >= 2
                        else 0.0
                    )
                    rows.append(
                        {
                            "source": source,
                            "split": split,
                            "metric": metric,
                            "model_name": model_name,
                            "lookback": lookback,
                            "n_folds": int(rank_array.size),
                            "best_count": best_count,
                            "best_fraction": (
                                best_count / fold_count if fold_count else 0.0
                            ),
                            "mean_rank": mean_rank,
                            "rank_std": rank_std,
                        }
                    )
    return pd.DataFrame(rows, columns=list(_RANK_STABILITY_COLUMNS))


def _build_model_ranking(
    confidence_intervals: pd.DataFrame,
    *,
    metric: str = "macro_f1",
    split: str = "test",
) -> pd.DataFrame:
    if confidence_intervals.empty:
        return pd.DataFrame(columns=list(_RANKING_COLUMNS))
    subset = confidence_intervals.loc[
        (confidence_intervals["metric"] == metric)
        & (confidence_intervals["split"] == split)
    ].copy()
    if subset.empty:
        return pd.DataFrame(columns=list(_RANKING_COLUMNS))
    subset = subset.sort_values("mean", ascending=False).reset_index(drop=True)
    subset["rank"] = np.arange(1, len(subset) + 1, dtype=int)
    return subset[list(_RANKING_COLUMNS)].copy()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _ensure_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(
                f"output path exists and is not a directory: {path}",
            )
        if any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    "refusing to write into a non-empty output directory; "
                    f"pass overwrite=True to replace it: {path}",
                )
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _unique_sorted(values: Iterable[Any]) -> list[Any]:
    seen: list[Any] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if pd.isna(value):
            continue
        if value not in seen:
            seen.append(value)
    try:
        return sorted(seen)
    except TypeError:
        return seen


def _format_uncertainty_notes(
    *,
    summary: UncertaintyAnalysisSummary,
    classical_ci: pd.DataFrame,
    neural_ci: pd.DataFrame,
    paired: pd.DataFrame,
    ranking: pd.DataFrame,
) -> str:
    def _format_ci_row(row: pd.Series) -> str:
        mean = row["mean"]
        ci_lower = row["ci_lower"]
        ci_upper = row["ci_upper"]
        lookback = row.get("lookback")
        tag = f" (lookback={int(lookback)})" if pd.notna(lookback) else ""
        return (
            f"  - {row['model_name']}{tag}: mean={mean:.4f} "
            f"[{ci_lower:.4f}, {ci_upper:.4f}] across {int(row['n_folds'])} folds"
        )

    lines: list[str] = []
    lines.append("# FI-2010 Uncertainty Notes")
    lines.append("")
    lines.append(
        "Fold variance is available because the classical and neural runners "
        "store per-fold metric tables.",
    )
    if summary.neural_seed_variance_available:
        lines.append(
            "Neural seed variance is available because the neural artefacts "
            "include more than one seed per (model, fold, lookback)."
        )
    else:
        lines.append(
            "Neural seed variance is not fully measured. The stored neural "
            "evidence covers a single seed per (model, fold, lookback). "
            "Cross-seed variance therefore remains future work unless an "
            "additional seed run is recorded."
        )
    if summary.classical_seed_variance_available:
        lines.append(
            "Classical seed variance is available because more than one seed "
            "is stored per (model, fold).",
        )
    else:
        lines.append(
            "Classical seed variance is not available; the classical "
            "runner records a single seed per (model, fold).",
        )
    lines.append(
        f"Confidence intervals use a Student-t two-sided interval at "
        f"{summary.ci_level:.2f} together with a percentile bootstrap "
        f"using {summary.bootstrap_iterations} iterations and "
        f"seed={summary.bootstrap_seed}. Fold is the unit of variance."
    )
    lines.append(
        f"Comparisons against the baseline `{summary.baseline_model}` are "
        f"paired per fold."
    )
    n_classical_folds = len(summary.classical_folds)
    n_neural_folds = len(summary.neural_folds)
    if n_classical_folds:
        lines.append(
            f"Classical comparisons are based on {n_classical_folds} FI-2010 folds.",
        )
    if n_neural_folds:
        lines.append(
            f"Neural comparisons are based on {n_neural_folds} FI-2010 folds.",
        )
    lines.append(
        "The execution proxy summary and calibration summary in the "
        "upstream multi-fold directories remain diagnostic and are not "
        "live tradability claims."
    )
    lines.append("")

    test_classical_macro = classical_ci.loc[
        (classical_ci["metric"] == "macro_f1") & (classical_ci["split"] == "test")
    ].sort_values("mean", ascending=False)
    if not test_classical_macro.empty:
        lines.append("## Classical test macro-F1 with confidence intervals")
        lines.append("")
        for _, row in test_classical_macro.iterrows():
            lines.append(_format_ci_row(row))
        lines.append("")

    test_neural_macro = neural_ci.loc[
        (neural_ci["metric"] == "macro_f1") & (neural_ci["split"] == "test")
    ].sort_values("mean", ascending=False)
    if not test_neural_macro.empty:
        lines.append("## Neural test macro-F1 with confidence intervals")
        lines.append("")
        for _, row in test_neural_macro.iterrows():
            lines.append(_format_ci_row(row))
        lines.append("")
        lines.append(
            "The neural numbers above remain reduced-scope, single-seed "
            "evidence unless additional seed runs are recorded; do not "
            "interpret them as cross-seed validated."
        )
        lines.append("")

    macro_paired = paired.loc[
        (paired["metric"] == "macro_f1") & (paired["split"] == "test")
    ].sort_values("mean_difference", ascending=False)
    if not macro_paired.empty:
        lines.append(
            f"## Paired fold differences vs `{summary.baseline_model}` "
            "(test macro-F1)"
        )
        lines.append("")
        for _, row in macro_paired.iterrows():
            lookback = row.get("lookback")
            tag = (
                f" (lookback={int(lookback)})"
                if pd.notna(lookback)
                else ""
            )
            lines.append(
                f"  - {row['candidate_model']}{tag}: mean diff="
                f"{row['mean_difference']:.4f} "
                f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}], "
                f"wins={int(row['wins'])}/losses={int(row['losses'])}/"
                f"ties={int(row['ties'])} across {int(row['n_folds'])} folds"
            )
        lines.append("")

    if not ranking.empty:
        lines.append("## Combined ranking (test macro-F1)")
        lines.append("")
        for _, row in ranking.iterrows():
            lookback = row.get("lookback")
            tag = (
                f" (lookback={int(lookback)})"
                if pd.notna(lookback)
                else ""
            )
            lines.append(
                f"  {int(row['rank'])}. {row['model_name']}{tag} "
                f"({row['source']}): "
                f"mean={row['mean']:.4f} "
                f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] over "
                f"{int(row['n_folds'])} folds"
            )
        lines.append("")

    lines.append("## Reading the artefacts")
    lines.append("")
    lines.append(
        "- `metric_confidence_intervals.csv`: per-model, per-split, "
        "per-metric mean, std, standard error and Student-t plus "
        "percentile-bootstrap confidence intervals."
    )
    lines.append(
        "  Missing probability metrics (for example `ridge` Brier and ECE) "
        "are dropped and tracked via `n_missing`."
    )
    lines.append(
        "- `paired_model_comparisons.csv`: paired fold-level mean "
        "differences between each candidate model and the baseline."
    )
    lines.append(
        "- `rank_stability.csv`: how often each model is best per fold and "
        "the per-model mean rank across folds."
    )
    lines.append(
        "- `model_ranking.csv`: the combined classical+neural ranking on "
        "test macro-F1, ordered by mean, with the same confidence interval "
        "as the per-metric table."
    )
    lines.append(
        "- `summary.json`: the inputs, parameters, models, folds and "
        "artefact paths used by this run."
    )
    lines.append("")
    lines.append(
        "## What this analysis does not claim"
    )
    lines.append("")
    lines.append(
        "- It does not establish profitability, market-beating "
        "performance or live tradability."
    )
    lines.append(
        "- It does not promote any model to foundation-model status or "
        "state-of-the-art status."
    )
    lines.append(
        "- It does not report self-supervised pretraining results; that "
        "ablation remains gated upstream."
    )
    lines.append(
        "- Neural superiority over the classical baseline must not be "
        "asserted without the caveats above."
    )
    lines.append("")
    return "\n".join(lines)


def _seed_variance_available(frame: pd.DataFrame) -> bool:
    if frame.empty or "seed" not in frame.columns:
        return False
    group_cols = [
        column
        for column in ("model_name", "fold_id", "lookback", "split")
        if column in frame.columns
    ]
    counts = frame.groupby(group_cols, dropna=False)["seed"].nunique()
    return bool((counts >= 2).any())


def analyse_fi2010_uncertainty(
    *,
    classical_dir: str | Path | None,
    neural_dir: str | Path | None,
    out_dir: str | Path,
    baseline_model: str = "gradient_boosting",
    metrics: Sequence[str] = DEFAULT_UNCERTAINTY_METRICS,
    ci_level: float = 0.95,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 0,
    overwrite: bool = False,
) -> UncertaintyAnalysisSummary:
    """Compute uncertainty artefacts from stored multi-fold tables."""
    if classical_dir is None and neural_dir is None:
        raise ValueError(
            "at least one of classical_dir or neural_dir must be provided",
        )
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1); got {ci_level!r}")
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    metrics_tuple = tuple(metrics)
    if not metrics_tuple:
        raise ValueError("metrics must contain at least one entry")

    resolved_out_dir = Path(out_dir)
    _ensure_output_dir(resolved_out_dir, overwrite=overwrite)

    warnings: list[str] = []

    classical_frame = pd.DataFrame()
    classical_input: Path | None = None
    if classical_dir is not None:
        classical_input = Path(classical_dir) / "results_by_fold.csv"
        if classical_input.is_file():
            classical_frame = load_classical_fold_results(classical_input)
        else:
            warnings.append(
                f"classical results_by_fold.csv not found at {classical_input}",
            )
            classical_input = None

    neural_frame = pd.DataFrame()
    neural_input: Path | None = None
    if neural_dir is not None:
        neural_input = Path(neural_dir) / "results_by_fold_seed.csv"
        if neural_input.is_file():
            neural_frame = load_neural_fold_results(neural_input)
        else:
            warnings.append(
                f"neural results_by_fold_seed.csv not found at {neural_input}",
            )
            neural_input = None

    if classical_frame.empty and neural_frame.empty:
        raise FileNotFoundError(
            "no multi-fold artefacts were found at the supplied paths",
        )

    classical_ci = compute_metric_confidence_intervals(
        classical_frame,
        source="classical",
        metrics=metrics_tuple,
        ci_level=ci_level,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    neural_ci = compute_metric_confidence_intervals(
        neural_frame,
        source="neural",
        metrics=metrics_tuple,
        ci_level=ci_level,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )

    classical_paired = compute_paired_model_comparisons(
        classical_frame,
        source="classical",
        baseline_model=baseline_model,
        metrics=metrics_tuple,
        ci_level=ci_level,
    )
    neural_paired = compute_paired_model_comparisons(
        neural_frame,
        source="neural",
        baseline_model=baseline_model,
        metrics=metrics_tuple,
        ci_level=ci_level,
    )

    classical_rank = compute_rank_stability(
        classical_frame,
        source="classical",
        metrics=metrics_tuple,
    )
    neural_rank = compute_rank_stability(
        neural_frame,
        source="neural",
        metrics=metrics_tuple,
    )

    confidence_intervals = pd.concat([classical_ci, neural_ci], ignore_index=True)
    paired_comparisons = pd.concat([classical_paired, neural_paired], ignore_index=True)
    rank_stability = pd.concat([classical_rank, neural_rank], ignore_index=True)

    ranking = _build_model_ranking(
        confidence_intervals,
        metric="macro_f1",
        split="test",
    )

    metric_ci_path = resolved_out_dir / "metric_confidence_intervals.csv"
    paired_path = resolved_out_dir / "paired_model_comparisons.csv"
    rank_path = resolved_out_dir / "rank_stability.csv"
    ranking_path = resolved_out_dir / "model_ranking.csv"
    notes_path = resolved_out_dir / "uncertainty_notes.md"
    summary_path = resolved_out_dir / "summary.json"

    confidence_intervals.to_csv(metric_ci_path, index=False)
    paired_comparisons.to_csv(paired_path, index=False)
    rank_stability.to_csv(rank_path, index=False)
    ranking.to_csv(ranking_path, index=False)

    classical_models = tuple(
        _unique_sorted(classical_frame.get("model_name", pd.Series(dtype=str)))
    )
    neural_models = tuple(
        _unique_sorted(neural_frame.get("model_name", pd.Series(dtype=str)))
    )
    classical_folds = tuple(
        _unique_sorted(classical_frame.get("fold_id", pd.Series(dtype=str)))
    )
    neural_folds = tuple(
        _unique_sorted(neural_frame.get("fold_id", pd.Series(dtype=str)))
    )
    neural_seeds = tuple(
        _unique_sorted(neural_frame.get("seed", pd.Series(dtype=int)))
    )
    neural_lookbacks = tuple(
        _unique_sorted(
            [
                int(value)
                for value in neural_frame.get("lookback", pd.Series(dtype=int))
                if pd.notna(value)
            ]
        )
    )

    classical_seed_variance = _seed_variance_available(classical_frame)
    neural_seed_variance = _seed_variance_available(neural_frame)

    summary = UncertaintyAnalysisSummary(
        output_dir=resolved_out_dir,
        classical_input=classical_input,
        neural_input=neural_input,
        baseline_model=baseline_model,
        ci_level=ci_level,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
        metrics=metrics_tuple,
        classical_models=classical_models,
        neural_models=neural_models,
        classical_folds=classical_folds,
        neural_folds=neural_folds,
        neural_seeds=tuple(int(value) for value in neural_seeds),
        neural_lookbacks=neural_lookbacks,
        artefacts={
            "summary": "summary.json",
            "metric_confidence_intervals": "metric_confidence_intervals.csv",
            "paired_model_comparisons": "paired_model_comparisons.csv",
            "rank_stability": "rank_stability.csv",
            "model_ranking": "model_ranking.csv",
            "uncertainty_notes": "uncertainty_notes.md",
        },
        warnings=tuple(warnings),
        classical_seed_variance_available=classical_seed_variance,
        neural_seed_variance_available=neural_seed_variance,
    )

    notes_text = _format_uncertainty_notes(
        summary=summary,
        classical_ci=classical_ci,
        neural_ci=neural_ci,
        paired=paired_comparisons,
        ranking=ranking,
    )
    notes_path.write_text(notes_text, encoding="utf-8")

    summary_payload: dict[str, Any] = {
        "analyser_version": FI2010_UNCERTAINTY_VERSION,
        "package_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "classical_results_by_fold": (
                str(classical_input) if classical_input is not None else None
            ),
            "neural_results_by_fold_seed": (
                str(neural_input) if neural_input is not None else None
            ),
        },
        "parameters": {
            "baseline_model": baseline_model,
            "ci_level": ci_level,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_seed": bootstrap_seed,
            "metrics": list(metrics_tuple),
        },
        "classical": {
            "models": list(classical_models),
            "folds": list(classical_folds),
            "seed_variance_available": classical_seed_variance,
            "row_count": len(classical_frame),
        },
        "neural": {
            "models": list(neural_models),
            "folds": list(neural_folds),
            "seeds": [int(value) for value in neural_seeds],
            "lookbacks": list(neural_lookbacks),
            "seed_variance_available": neural_seed_variance,
            "row_count": len(neural_frame),
        },
        "artefacts": dict(summary.artefacts),
        "warnings": list(warnings),
        "claim_boundaries": [
            "Diagnostic only; no profitability or live tradability claim.",
            "No state-of-the-art claim.",
            "No foundation-model claim.",
            "No self-supervised result.",
            (
                "Neural numbers remain reduced-scope, single-seed evidence "
                "unless additional seed runs are recorded."
            ),
        ],
    }
    summary_path.write_text(
        _stable_json_dumps(summary_payload),
        encoding="utf-8",
    )
    return summary


def _stable_json_dumps(payload: Mapping[str, Any]) -> str:
    import json

    def _default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        raise TypeError(f"unsupported type for JSON serialisation: {type(value)!r}")

    return json.dumps(payload, indent=2, sort_keys=True, default=_default) + "\n"
