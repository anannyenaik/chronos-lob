"""FI-2010 brutal ablation layer.

This module quantifies where the supervised FI-2010 signal survives and
where it breaks across six practical ablation families:

* ``feature_groups`` - refit a fast linear baseline on each feature
  subset (top-of-book, depth, liquidity depth, price levels, all
  features and a label-leakage control group).
* ``model_class`` - reuse the stored classical per-fold metric table and
  measure each model class against the strongest stored baseline.
* ``lookback`` - vary the supervised neural lookback window. This is
  CPU-expensive, so it is skipped by default and recorded with a reason
  unless an explicit lookback subset is requested.
* ``horizon`` - retrain the linear baseline on alternative configured
  prediction horizons when their label columns are present.
* ``calibration`` - reuse stored reliability and confidence-threshold
  diagnostics.
* ``execution`` - reuse stored cost and latency proxy diagnostics.

Cheap families refit only a fast linear baseline. The model-class,
calibration and execution families reuse stored evidence and do not
retrain anything. The neural lookback family is skipped by default.

The execution numbers are simplified proxy diagnostics under stated
assumptions. They are not a backtest. This layer does not assert
profitability, live tradability, production execution quality,
foundation-model status, leading-benchmark performance or any
self-supervised pretraining result, and it does not claim neural
superiority over the classical baseline.
"""

from __future__ import annotations

import math
import re
import shutil
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chronoslob import __version__
from chronoslob.experiments.fi2010_benchmark import (
    BenchmarkSplitConfig,
    FI2010BenchmarkConfig,
    build_benchmark_split,
)
from chronoslob.experiments.fi2010_multifold_runner import (
    MultifoldClassicalConfig,
    load_multifold_classical_config,
)
from chronoslob.experiments.manifests import sha256_file, stable_json_dumps
from chronoslob.experiments.model_registry import (
    build_paper_baseline_config,
    get_paper_model_spec,
)
from chronoslob.models.baselines import create_baseline_model
from chronoslob.models.preprocessing import (
    TrainOnlyStandardScaler,
    build_feature_matrix,
    build_target_vector,
    select_feature_columns,
)
from chronoslob.training.experiment import get_git_commit
from chronoslob.training.metrics import compute_classification_metrics
from chronoslob.training.splitters import SplitIndices

__all__ = [
    "ABLATION_FAMILIES",
    "ABLATION_RESULT_COLUMNS",
    "DEFAULT_FEATURE_GROUP_MODELS",
    "DEFAULT_MODEL_CLASS_BASELINE",
    "FI2010_BRUTAL_ABLATIONS_VERSION",
    "BrutalAblationSummary",
    "normalise_families",
    "resolve_feature_groups",
    "run_fi2010_brutal_ablations",
]

FI2010_BRUTAL_ABLATIONS_VERSION = "phase-g/fi2010-brutal-ablations/v1"

ABLATION_FAMILIES: tuple[str, ...] = (
    "feature_groups",
    "model_class",
    "lookback",
    "horizon",
    "calibration",
    "execution",
)

# Families that refit a model from prepared fold CSVs.
_FIT_FAMILIES: frozenset[str] = frozenset({"feature_groups", "horizon"})
# Families that reuse the stored classical artefact directory.
_CLASSICAL_REUSE_FAMILIES: frozenset[str] = frozenset(
    {"model_class", "calibration", "execution"}
)

DEFAULT_FEATURE_GROUP_MODELS: tuple[str, ...] = ("logistic",)
DEFAULT_NEURAL_LOOKBACK_MODELS: tuple[str, ...] = ("matrix_transformer",)
DEFAULT_MODEL_CLASS_BASELINE = "gradient_boosting"
DEFAULT_PREDICTIVE_METRICS: tuple[str, ...] = ("accuracy", "macro_f1", "mcc")

# Confidence threshold and cost/latency reference values used as baselines
# when reading the stored execution proxy tables.
_BASELINE_THRESHOLD = 0.0
_BASELINE_COST_BPS = 0.0
_BASELINE_LATENCY = 0

ABLATION_RESULT_COLUMNS: tuple[str, ...] = (
    "ablation_family",
    "ablation_name",
    "fold_id",
    "model_name",
    "split",
    "metric_name",
    "metric_value",
    "baseline_value",
    "delta_vs_baseline",
    "status",
    "skip_reason",
)

_FEATURE_GROUP_COLUMNS = (*ABLATION_RESULT_COLUMNS, "n_features")
_MODEL_CLASS_COLUMNS = (*ABLATION_RESULT_COLUMNS, "baseline_model")
_LOOKBACK_COLUMNS = (*ABLATION_RESULT_COLUMNS, "lookback")
_HORIZON_COLUMNS = (*ABLATION_RESULT_COLUMNS, "horizon", "label_column")
_CALIBRATION_COLUMNS = (*ABLATION_RESULT_COLUMNS, "confidence_threshold")
_EXECUTION_COLUMNS = (
    *ABLATION_RESULT_COLUMNS,
    "confidence_threshold",
    "cost_bps",
    "latency_steps",
)

_ABLATION_SUMMARY_COLUMNS = (
    "ablation_family",
    "ablation_name",
    "model_name",
    "split",
    "metric_name",
    "fold_count",
    "metric_value_mean",
    "baseline_value_mean",
    "delta_vs_baseline_mean",
    "delta_vs_baseline_std",
    "status",
)

_LEVEL_RE = re.compile(r"^(?:bid|ask)_(?:price|quantity)_(\d+)$", re.IGNORECASE)
_PRICE_RE = re.compile(r"^(?:bid|ask)_price_\d+$", re.IGNORECASE)
_QUANTITY_RE = re.compile(r"^(?:bid|ask)_quantity_\d+$", re.IGNORECASE)

_CLAIM_BOUNDARIES: tuple[str, ...] = (
    "Diagnostic only; no profitability or live tradability claim is made.",
    "Execution numbers are simplified proxy diagnostics, not a backtest.",
    "No foundation-model or leading-benchmark claim.",
    "No self-supervised pretraining result is reported.",
    "Neural superiority over the classical baseline is not asserted.",
)


# ---------------------------------------------------------------------------
# Summary container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrutalAblationSummary:
    """Lightweight return value from the brutal ablation entry point."""

    output_dir: Path
    config_path: Path
    neural_config_path: Path | None
    processed_root: Path | None
    classical_dir: Path | None
    neural_dir: Path | None
    families_requested: tuple[str, ...]
    families_run: tuple[str, ...]
    families_skipped: tuple[str, ...]
    folds: tuple[str, ...]
    fit_models: tuple[str, ...]
    feature_groups: tuple[str, ...]
    horizons: tuple[int, ...]
    result_row_count: int
    ok_row_count: int
    skipped_count: int
    full_predictions_written: bool
    checkpoints_written: bool
    dry_run: bool
    artefacts: Mapping[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def normalise_families(families: Sequence[str] | str | None) -> tuple[str, ...]:
    """Resolve a requested family subset against the supported families."""
    if families is None:
        return ABLATION_FAMILIES
    if isinstance(families, str):
        text = families.strip()
        if not text or text.casefold() == "all":
            return ABLATION_FAMILIES
        tokens: Sequence[str] = [token.strip() for token in text.split(",")]
    else:
        tokens = list(families)
    cleaned: list[str] = []
    allowed = set(ABLATION_FAMILIES)
    for token in tokens:
        name = str(token).strip().casefold()
        if not name:
            continue
        if name not in allowed:
            raise ValueError(
                f"unsupported ablation family {token!r}; "
                f"supported families: {list(ABLATION_FAMILIES)}",
            )
        if name not in cleaned:
            cleaned.append(name)
    if not cleaned:
        raise ValueError("at least one ablation family must be selected")
    return tuple(cleaned)


def _normalise_fold_token(value: str | int) -> str:
    text = str(value).strip().casefold()
    if not text:
        raise ValueError("fold identifiers must be non-empty")
    if text.isdigit():
        number = int(text)
    elif text.startswith("fold_") and text.removeprefix("fold_").isdigit():
        number = int(text.removeprefix("fold_"))
    else:
        raise ValueError(f"invalid fold identifier {value!r}; expected fold_N or N")
    if number <= 0:
        raise ValueError("fold identifiers must be positive")
    return f"fold_{number}"


def _fold_number(fold_id: str) -> int:
    return int(fold_id.removeprefix("fold_"))


def _normalise_fold_selection(
    folds: Sequence[str | int] | str | None,
) -> tuple[str, ...] | None:
    if folds is None:
        return None
    if isinstance(folds, str):
        text = folds.strip()
        if not text or text.casefold() == "all":
            return None
        raw: Sequence[str | int] = [token.strip() for token in text.split(",")]
    else:
        raw = list(folds)
    selected = [
        _normalise_fold_token(item) for item in raw if str(item).strip()
    ]
    if not selected:
        raise ValueError("fold selection must not be empty")
    return tuple(dict.fromkeys(selected))


def _split_requested_models(
    models: Sequence[str] | str | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split a requested model list into classical and neural buckets."""
    if models is None:
        return (), ()
    if isinstance(models, str):
        tokens: Sequence[str] = [token.strip() for token in models.split(",")]
    else:
        tokens = list(models)
    classical: list[str] = []
    neural: list[str] = []
    for token in tokens:
        name = str(token).strip().casefold()
        if not name:
            continue
        try:
            spec = get_paper_model_spec(name)
        except (KeyError, ValueError):
            continue
        if spec.model_family == "classical":
            if name not in classical:
                classical.append(spec.name)
        elif spec.name not in neural:
            neural.append(spec.name)
    return tuple(classical), tuple(neural)


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


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _record(
    *,
    family: str,
    name: str,
    fold_id: str,
    model_name: str,
    split: str,
    metric_name: str,
    metric_value: float | None,
    baseline_value: float | None,
    status: str = "ok",
    skip_reason: str = "",
    **extra: Any,
) -> dict[str, Any]:
    delta: float | None = None
    if metric_value is not None and baseline_value is not None:
        delta = float(metric_value) - float(baseline_value)
    row: dict[str, Any] = {
        "ablation_family": family,
        "ablation_name": name,
        "fold_id": fold_id,
        "model_name": model_name,
        "split": split,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "baseline_value": baseline_value,
        "delta_vs_baseline": delta,
        "status": status,
        "skip_reason": skip_reason,
    }
    row.update(extra)
    return row


def _skip_record(
    *,
    family: str,
    name: str,
    skip_reason: str,
    fold_id: str = "-",
    model_name: str = "-",
    split: str = "-",
    metric_name: str = "-",
    **extra: Any,
) -> dict[str, Any]:
    return _record(
        family=family,
        name=name,
        fold_id=fold_id,
        model_name=model_name,
        split=split,
        metric_name=metric_name,
        metric_value=None,
        baseline_value=None,
        status="skipped",
        skip_reason=skip_reason,
        **extra,
    )


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


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------


def _book_level(column: str) -> int | None:
    match = _LEVEL_RE.match(column)
    return int(match.group(1)) if match is not None else None


def resolve_feature_groups(
    feature_columns: Sequence[str],
) -> OrderedDict[str, list[str]]:
    """Return the named feature-group column subsets for the ablation.

    Groups whose definition matches no column are returned empty so the
    caller can record a documented skip rather than crashing.
    """
    columns = list(feature_columns)
    top = [c for c in columns if _book_level(c) == 1]
    depth = [c for c in columns if (_book_level(c) or 0) >= 2]
    liquidity = [c for c in columns if _QUANTITY_RE.match(c)]
    price = [c for c in columns if _PRICE_RE.match(c)]
    groups: OrderedDict[str, list[str]] = OrderedDict()
    groups["all_features"] = list(columns)
    groups["top_of_book_only"] = top
    groups["depth_features"] = depth
    groups["liquidity_depth_features"] = liquidity
    groups["price_only"] = price
    # The leakage-control group equals all_features by construction because
    # label columns are excluded from the feature matrix upstream.
    groups["labels_excluded"] = list(columns)
    return groups


def _feature_columns(frame: pd.DataFrame, config: MultifoldClassicalConfig) -> list[str]:
    excluded = {config.split_column, config.label_name, *config.label_columns}
    excluded.update(
        column for column in frame.columns if str(column).lower().startswith("label")
    )
    candidate = frame.drop(
        columns=[column for column in excluded if column in frame.columns]
    )
    columns = select_feature_columns(candidate, reject_label_like=True)
    if not columns:
        raise ValueError("brutal ablation runner requires at least one feature column")
    return columns


@dataclass(frozen=True)
class _SplitScores:
    """Predictive metrics for one fit on validation and test splits."""

    by_split: dict[str, dict[str, float]]


def _score_classical(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    split: SplitIndices,
    model_name: str,
    seed: int,
) -> _SplitScores:
    spec = get_paper_model_spec(model_name)
    columns = list(feature_columns)
    train_x_raw = build_feature_matrix(
        frame, feature_columns=columns, row_indices=split.train
    ).x
    train_y = build_target_vector(
        frame, target_column=target_column, row_indices=split.train
    ).y
    all_labels = list(
        build_target_vector(frame, target_column=target_column).classes or []
    )
    model = create_baseline_model(build_paper_baseline_config(model_name, seed=seed))
    scaler: TrainOnlyStandardScaler | None = None
    if spec.requires_standardisation:
        scaler = TrainOnlyStandardScaler()
        train_x = scaler.fit_transform(train_x_raw)
    else:
        train_x = train_x_raw
    model.fit(train_x, train_y)

    by_split: dict[str, dict[str, float]] = {}
    for split_name, indices in (("validation", split.validation), ("test", split.test)):
        if not indices:
            continue
        x_raw = build_feature_matrix(
            frame, feature_columns=columns, row_indices=indices
        ).x
        x = scaler.transform(x_raw) if scaler is not None else x_raw
        y = build_target_vector(
            frame, target_column=target_column, row_indices=indices
        ).y
        predictions = np.asarray(model.predict(x))
        metrics = compute_classification_metrics(
            y.tolist(), predictions.tolist(), labels=all_labels
        )
        by_split[split_name] = {
            "accuracy": float(metrics.accuracy),
            "macro_f1": float(metrics.macro_f1),
            "mcc": float(metrics.matthews_corrcoef),
        }
    return _SplitScores(by_split=by_split)


def _benchmark_split(
    frame: pd.DataFrame,
    *,
    config: MultifoldClassicalConfig,
    seed: int,
) -> SplitIndices:
    benchmark_config = FI2010BenchmarkConfig(
        experiment_name=f"{config.study_name}_brutal_ablation",
        dataset_name=config.dataset_name,
        task_name=config.task_name,
        horizon=config.target_horizon,
        split_name="official_column",
        seed=seed,
        local_data_path="<brutal-ablation-processed-root>",
        output_dir="experiments/fi2010_brutal_ablations",
        label_name=config.label_name,
        label_columns=config.label_columns,
        timestamp_column=None,
        split_column=config.split_column,
        price_level_count=10,
        split=BenchmarkSplitConfig(
            method="official_column",
            split_column=config.split_column,
            validation_fraction_within_train=config.validation_fraction_within_train,
            train_value=config.train_value,
            test_value=config.test_value,
        ),
    )
    return build_benchmark_split(benchmark_config, frame)


# ---------------------------------------------------------------------------
# Stored-evidence loaders (reuse families)
# ---------------------------------------------------------------------------


def _load_classical_fold_results(classical_dir: Path) -> pd.DataFrame:
    path = classical_dir / "results_by_fold.csv"
    if not path.is_file():
        raise FileNotFoundError(f"classical results_by_fold.csv not found: {path}")
    frame = pd.read_csv(path)
    if "status" in frame.columns:
        frame = frame.loc[frame["status"].astype(str).str.lower() == "ok"].copy()
    frame["fold_id"] = frame["fold_id"].apply(_normalise_fold_token)
    return frame


def _load_execution_rows(classical_dir: Path) -> pd.DataFrame:
    """Load per-fold execution proxy rows, falling back to the summary."""
    columns = [
        "fold_id",
        "model_name",
        "split",
        "confidence_threshold",
        "cost_bps",
        "latency_steps",
        "eligible_predictions",
        "hit_rate_proxy",
        "turnover_proxy",
        "gross_signal_return_proxy",
        "net_signal_return_proxy",
    ]
    folds_dir = classical_dir / "folds"
    frames: list[pd.DataFrame] = []
    if folds_dir.is_dir():
        for fold_path in sorted(folds_dir.glob("fold_*/execution_sensitivity.csv")):
            fold_id = fold_path.parent.name
            sub = pd.read_csv(fold_path)
            if sub.empty:
                continue
            sub = sub.assign(fold_id=_normalise_fold_token(fold_id))
            frames.append(sub)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        present = [column for column in columns if column in combined.columns]
        return combined.loc[:, present].copy()
    summary_path = classical_dir / "execution_summary.csv"
    if not summary_path.is_file():
        return pd.DataFrame(columns=columns)
    summary = pd.read_csv(summary_path)
    summary = summary.assign(fold_id="aggregate")
    present = [column for column in columns if column in summary.columns]
    return summary.loc[:, present].copy()


# ---------------------------------------------------------------------------
# Family builders
# ---------------------------------------------------------------------------


def _feature_group_rows(
    *,
    frame: pd.DataFrame,
    fold_id: str,
    split: SplitIndices,
    feature_columns: Sequence[str],
    target_column: str,
    models: Sequence[str],
    seed: int,
    baseline_by_model: Mapping[str, _SplitScores],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = resolve_feature_groups(feature_columns)
    for model_name in models:
        baseline = baseline_by_model[model_name]
        for group_name, group_columns in groups.items():
            if not group_columns:
                rows.append(
                    _skip_record(
                        family="feature_groups",
                        name=group_name,
                        fold_id=fold_id,
                        model_name=model_name,
                        skip_reason=(
                            "no column matched the feature-group definition "
                            "in this fold CSV"
                        ),
                        n_features=0,
                    )
                )
                continue
            if group_name in {"all_features", "labels_excluded"}:
                scores = baseline
            else:
                scores = _score_classical(
                    frame,
                    feature_columns=group_columns,
                    target_column=target_column,
                    split=split,
                    model_name=model_name,
                    seed=seed,
                )
            for split_name, metrics in scores.by_split.items():
                base_metrics = baseline.by_split.get(split_name, {})
                for metric_name in DEFAULT_PREDICTIVE_METRICS:
                    rows.append(
                        _record(
                            family="feature_groups",
                            name=group_name,
                            fold_id=fold_id,
                            model_name=model_name,
                            split=split_name,
                            metric_name=metric_name,
                            metric_value=_safe_float(metrics.get(metric_name)),
                            baseline_value=_safe_float(base_metrics.get(metric_name)),
                            n_features=len(group_columns),
                        )
                    )
    return rows


def _horizon_rows(
    *,
    frame: pd.DataFrame,
    fold_id: str,
    split: SplitIndices,
    feature_columns: Sequence[str],
    target_column: str,
    target_horizon: int,
    configured_label_columns: Sequence[str],
    models: Sequence[str],
    seed: int,
    baseline_by_model: Mapping[str, _SplitScores],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered: list[str] = [target_column]
    for column in configured_label_columns:
        if column != target_column and column not in ordered:
            ordered.append(column)
    for label_column in ordered:
        horizon = _horizon_from_label(label_column, default=target_horizon)
        if label_column not in frame.columns:
            rows.append(
                _skip_record(
                    family="horizon",
                    name=f"horizon_{horizon}",
                    fold_id=fold_id,
                    skip_reason=(
                        f"configured label column {label_column!r} is not "
                        "present in this fold CSV"
                    ),
                    horizon=horizon,
                    label_column=label_column,
                )
            )
            continue
        for model_name in models:
            if label_column == target_column:
                scores = baseline_by_model[model_name]
            else:
                scores = _score_classical(
                    frame,
                    feature_columns=feature_columns,
                    target_column=label_column,
                    split=split,
                    model_name=model_name,
                    seed=seed,
                )
            base_scores = baseline_by_model[model_name]
            for split_name, metrics in scores.by_split.items():
                base_metrics = base_scores.by_split.get(split_name, {})
                for metric_name in DEFAULT_PREDICTIVE_METRICS:
                    rows.append(
                        _record(
                            family="horizon",
                            name=f"horizon_{horizon}",
                            fold_id=fold_id,
                            model_name=model_name,
                            split=split_name,
                            metric_name=metric_name,
                            metric_value=_safe_float(metrics.get(metric_name)),
                            baseline_value=_safe_float(base_metrics.get(metric_name)),
                            horizon=horizon,
                            label_column=label_column,
                        )
                    )
    return rows


def _horizon_from_label(label_column: str, *, default: int) -> int:
    suffix = label_column.rsplit("_", maxsplit=1)[-1]
    return int(suffix) if suffix.isdigit() else default


def _model_class_rows(
    *,
    results: pd.DataFrame,
    baseline_model: str,
    requested_models: Sequence[str],
    fold_filter: tuple[str, ...] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    frame = results
    if fold_filter is not None:
        frame = frame.loc[frame["fold_id"].isin(fold_filter)]
    if frame.empty:
        return rows, ["model_class: no stored classical rows after fold filtering"]
    candidate_models = sorted(frame["model_name"].astype(str).unique())
    if requested_models:
        allowed = set(requested_models) | {baseline_model}
        candidate_models = [m for m in candidate_models if m in allowed]
    if baseline_model not in set(frame["model_name"].astype(str)):
        warnings.append(
            f"model_class: baseline model {baseline_model!r} absent from stored "
            "classical results; deltas use a per-fold maximum instead"
        )
    for split_name in sorted(frame["split"].astype(str).unique()):
        split_frame = frame.loc[frame["split"].astype(str) == split_name]
        for metric_name in DEFAULT_PREDICTIVE_METRICS:
            if metric_name not in split_frame.columns:
                continue
            for fold_id in sorted(split_frame["fold_id"].astype(str).unique()):
                fold_frame = split_frame.loc[
                    split_frame["fold_id"].astype(str) == fold_id
                ]
                baseline_value = _baseline_metric_value(
                    fold_frame, baseline_model=baseline_model, metric=metric_name
                )
                for model_name in candidate_models:
                    model_rows = fold_frame.loc[
                        fold_frame["model_name"].astype(str) == model_name
                    ]
                    if model_rows.empty:
                        continue
                    value = _safe_float(model_rows[metric_name].iloc[0])
                    rows.append(
                        _record(
                            family="model_class",
                            name=model_name,
                            fold_id=fold_id,
                            model_name=model_name,
                            split=split_name,
                            metric_name=metric_name,
                            metric_value=value,
                            baseline_value=baseline_value,
                            baseline_model=baseline_model,
                        )
                    )
    return rows, warnings


def _baseline_metric_value(
    fold_frame: pd.DataFrame,
    *,
    baseline_model: str,
    metric: str,
) -> float | None:
    baseline_rows = fold_frame.loc[
        fold_frame["model_name"].astype(str) == baseline_model
    ]
    if not baseline_rows.empty:
        return _safe_float(baseline_rows[metric].iloc[0])
    numeric = pd.to_numeric(fold_frame[metric], errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.max())


def _calibration_rows(
    *,
    results: pd.DataFrame,
    execution_rows: pd.DataFrame,
    requested_models: Sequence[str],
    fold_filter: tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frame = results
    if fold_filter is not None:
        frame = frame.loc[frame["fold_id"].isin(fold_filter)]
    if "ece" in frame.columns:
        reliability = frame.loc[frame["split"].astype(str) == "test"]
        for _, row in reliability.iterrows():
            model_name = str(row["model_name"])
            if requested_models and model_name not in requested_models:
                continue
            ece = _safe_float(row.get("ece"))
            if ece is None:
                continue
            rows.append(
                _record(
                    family="calibration",
                    name="reliability_ece",
                    fold_id=str(row["fold_id"]),
                    model_name=model_name,
                    split="test",
                    metric_name="ece",
                    metric_value=ece,
                    baseline_value=0.0,
                    confidence_threshold=_BASELINE_THRESHOLD,
                )
            )
    rows.extend(
        _confidence_threshold_rows(
            execution_rows=execution_rows,
            requested_models=requested_models,
            fold_filter=fold_filter,
        )
    )
    return rows


def _confidence_threshold_rows(
    *,
    execution_rows: pd.DataFrame,
    requested_models: Sequence[str],
    fold_filter: tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if execution_rows.empty:
        return rows
    frame = execution_rows.copy()
    for column in ("confidence_threshold", "cost_bps", "latency_steps"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        (frame.get("cost_bps") == _BASELINE_COST_BPS)
        & (frame.get("latency_steps") == _BASELINE_LATENCY)
    ]
    if fold_filter is not None and "fold_id" in frame.columns:
        keep = set(fold_filter) | {"aggregate"}
        frame = frame.loc[frame["fold_id"].astype(str).isin(keep)]
    metric_names = ("hit_rate_proxy", "eligible_predictions")
    for model_name in sorted(frame["model_name"].astype(str).unique()):
        if requested_models and model_name not in requested_models:
            continue
        model_frame = frame.loc[frame["model_name"].astype(str) == model_name]
        for fold_id in sorted(model_frame["fold_id"].astype(str).unique()):
            fold_frame = model_frame.loc[
                model_frame["fold_id"].astype(str) == fold_id
            ]
            base = fold_frame.loc[
                fold_frame["confidence_threshold"] == _BASELINE_THRESHOLD
            ]
            for metric_name in metric_names:
                if metric_name not in fold_frame.columns:
                    continue
                base_value = (
                    _safe_float(base[metric_name].iloc[0]) if not base.empty else None
                )
                for threshold in sorted(
                    fold_frame["confidence_threshold"].dropna().unique()
                ):
                    selected = fold_frame.loc[
                        fold_frame["confidence_threshold"] == threshold
                    ]
                    value = _safe_float(selected[metric_name].iloc[0])
                    rows.append(
                        _record(
                            family="calibration",
                            name=f"confidence_threshold_{threshold:g}",
                            fold_id=fold_id,
                            model_name=model_name,
                            split="test",
                            metric_name=metric_name,
                            metric_value=value,
                            baseline_value=base_value,
                            confidence_threshold=float(threshold),
                        )
                    )
    return rows


def _execution_rows(
    *,
    execution_rows: pd.DataFrame,
    requested_models: Sequence[str],
    fold_filter: tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if execution_rows.empty:
        return rows
    frame = execution_rows.copy()
    for column in ("confidence_threshold", "cost_bps", "latency_steps"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame.get("confidence_threshold") == _BASELINE_THRESHOLD]
    if fold_filter is not None and "fold_id" in frame.columns:
        keep = set(fold_filter) | {"aggregate"}
        frame = frame.loc[frame["fold_id"].astype(str).isin(keep)]
    metric_names = (
        "net_signal_return_proxy",
        "gross_signal_return_proxy",
        "turnover_proxy",
    )
    for model_name in sorted(frame["model_name"].astype(str).unique()):
        if requested_models and model_name not in requested_models:
            continue
        model_frame = frame.loc[frame["model_name"].astype(str) == model_name]
        for fold_id in sorted(model_frame["fold_id"].astype(str).unique()):
            fold_frame = model_frame.loc[
                model_frame["fold_id"].astype(str) == fold_id
            ]
            base = fold_frame.loc[
                (fold_frame["cost_bps"] == _BASELINE_COST_BPS)
                & (fold_frame["latency_steps"] == _BASELINE_LATENCY)
            ]
            for _, row in fold_frame.iterrows():
                cost = _safe_float(row.get("cost_bps")) or 0.0
                latency = int(row.get("latency_steps") or 0)
                name = f"cost_{cost:g}_latency_{latency}"
                for metric_name in metric_names:
                    if metric_name not in fold_frame.columns:
                        continue
                    base_value = (
                        _safe_float(base[metric_name].iloc[0])
                        if not base.empty
                        else None
                    )
                    rows.append(
                        _record(
                            family="execution",
                            name=name,
                            fold_id=fold_id,
                            model_name=model_name,
                            split="test",
                            metric_name=metric_name,
                            metric_value=_safe_float(row.get(metric_name)),
                            baseline_value=base_value,
                            confidence_threshold=_BASELINE_THRESHOLD,
                            cost_bps=cost,
                            latency_steps=latency,
                        )
                    )
    return rows


def _lookback_reference_rows(
    *,
    neural_dir: Path | None,
    requested_models: Sequence[str],
    fold_filter: tuple[str, ...] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if neural_dir is None:
        return rows
    path = neural_dir / "results_by_fold_seed.csv"
    if not path.is_file():
        return rows
    frame = pd.read_csv(path)
    if "status" in frame.columns:
        frame = frame.loc[frame["status"].astype(str).str.lower() == "ok"]
    if frame.empty:
        return rows
    frame = frame.assign(fold_id=frame["fold_id"].apply(_normalise_fold_token))
    if fold_filter is not None:
        frame = frame.loc[frame["fold_id"].isin(fold_filter)]
    models = requested_models or DEFAULT_NEURAL_LOOKBACK_MODELS
    frame = frame.loc[frame["model_name"].astype(str).isin(set(models))]
    for _, row in frame.iterrows():
        lookback = int(row["lookback"])
        for metric_name in DEFAULT_PREDICTIVE_METRICS:
            mapped = "mcc" if metric_name == "mcc" else metric_name
            value = _safe_float(row.get(mapped))
            rows.append(
                _record(
                    family="lookback",
                    name=f"lookback_{lookback}_reference",
                    fold_id=str(row["fold_id"]),
                    model_name=str(row["model_name"]),
                    split="test",
                    metric_name=metric_name,
                    metric_value=value,
                    baseline_value=value,
                    lookback=lookback,
                )
            )
    return rows


def _run_neural_lookback(
    *,
    neural_config_path: Path,
    processed_root: Path,
    out_dir: Path,
    folds: tuple[str, ...] | None,
    lookbacks: Sequence[int],
    models: Sequence[str],
    seed: int,
    max_epochs: int,
) -> list[dict[str, Any]]:
    """Train matrix-transformer lookbacks and return lookback delta rows.

    Heavy and CPU-bound; only invoked when an explicit lookback subset is
    requested. Any failure is surfaced to the caller as an exception so it
    can be recorded as a skip.
    """
    from chronoslob.experiments.fi2010_neural_runner import (
        run_fi2010_neural_benchmark,
    )

    scratch = out_dir / "_neural_lookback_runs"
    fold_argument = "all" if folds is None else ",".join(folds)
    run_fi2010_neural_benchmark(
        config_path=neural_config_path,
        processed_root=processed_root,
        out_dir=scratch,
        folds=fold_argument,
        models=list(models),
        seeds=[seed],
        lookbacks=list(lookbacks),
        max_epochs=max_epochs,
        overwrite=True,
        write_full_predictions=False,
        write_checkpoints=False,
    )
    results_path = scratch / "results_by_fold_seed.csv"
    frame = pd.read_csv(results_path)
    if "status" in frame.columns:
        frame = frame.loc[frame["status"].astype(str).str.lower() == "ok"]
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return rows
    frame = frame.assign(fold_id=frame["fold_id"].apply(_normalise_fold_token))
    smallest = int(min(lookbacks))
    for fold_id in sorted(frame["fold_id"].astype(str).unique()):
        fold_frame = frame.loc[frame["fold_id"].astype(str) == fold_id]
        for model_name in sorted(fold_frame["model_name"].astype(str).unique()):
            model_frame = fold_frame.loc[
                fold_frame["model_name"].astype(str) == model_name
            ]
            base = model_frame.loc[model_frame["lookback"] == smallest]
            for _, row in model_frame.iterrows():
                lookback = int(row["lookback"])
                for metric_name in DEFAULT_PREDICTIVE_METRICS:
                    base_value = (
                        _safe_float(base[metric_name].iloc[0])
                        if not base.empty
                        else None
                    )
                    rows.append(
                        _record(
                            family="lookback",
                            name=f"lookback_{lookback}",
                            fold_id=fold_id,
                            model_name=model_name,
                            split="test",
                            metric_name=metric_name,
                            metric_value=_safe_float(row.get(metric_name)),
                            baseline_value=base_value,
                            lookback=lookback,
                        )
                    )
    return rows


# ---------------------------------------------------------------------------
# Aggregation, notes and summary writing
# ---------------------------------------------------------------------------


def _build_ablation_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ok_rows = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("metric_value") is not None
    ]
    if not ok_rows:
        return []
    frame = pd.DataFrame(ok_rows)
    summary: list[dict[str, Any]] = []
    group_columns = [
        "ablation_family",
        "ablation_name",
        "model_name",
        "split",
        "metric_name",
    ]
    for keys, group in frame.groupby(group_columns, sort=True):
        values = pd.to_numeric(group["metric_value"], errors="coerce").dropna()
        deltas = pd.to_numeric(group["delta_vs_baseline"], errors="coerce").dropna()
        baselines = pd.to_numeric(group["baseline_value"], errors="coerce").dropna()
        summary.append(
            {
                "ablation_family": keys[0],
                "ablation_name": keys[1],
                "model_name": keys[2],
                "split": keys[3],
                "metric_name": keys[4],
                "fold_count": int(group["fold_id"].nunique()),
                "metric_value_mean": float(values.mean()) if not values.empty else None,
                "baseline_value_mean": (
                    float(baselines.mean()) if not baselines.empty else None
                ),
                "delta_vs_baseline_mean": (
                    float(deltas.mean()) if not deltas.empty else None
                ),
                "delta_vs_baseline_std": (
                    float(deltas.std(ddof=0)) if len(deltas) >= 1 else None
                ),
                "status": "ok",
            }
        )
    return summary


def _format_notes(
    *,
    summary: BrutalAblationSummary,
    rows: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, Any]],
) -> str:
    lines: list[str] = [
        "# FI-2010 Brutal Ablation Notes",
        "",
        "## Purpose",
        "",
        (
            "These ablations stress where the supervised FI-2010 signal "
            "survives and where it breaks across feature groups, model class, "
            "lookback, horizon, calibration threshold and execution cost or "
            "latency assumptions."
        ),
        "",
        "## Families run",
        "",
    ]
    for family in summary.families_run:
        lines.append(f"- `{family}`")
    if summary.families_skipped:
        lines.append("")
        lines.append("## Families skipped")
        lines.append("")
        for family in summary.families_skipped:
            lines.append(f"- `{family}`")
    lines.extend(["", "## Reading the deltas", "", ])
    lines.append(
        "- `delta_vs_baseline` is `metric_value - baseline_value` for the "
        "family baseline."
    )
    lines.append(
        "- feature_groups baseline is `all_features`; a large negative delta "
        "means that feature subset loses signal."
    )
    lines.append(
        f"- model_class baseline is `{DEFAULT_MODEL_CLASS_BASELINE}`; a "
        "negative delta means the simpler model class is weaker."
    )
    lines.append(
        "- horizon baseline is the configured target horizon; positive deltas "
        "mark easier horizons."
    )
    lines.append(
        "- calibration and execution deltas are measured against the zero "
        "threshold, zero cost and zero latency reference."
    )
    lines.append("")
    lines.append("## Strongest and weakest findings")
    lines.append("")
    strongest, weakest = _headline_feature_group_findings(rows)
    if strongest is not None:
        lines.append(f"- Most robust feature group: {strongest}.")
    if weakest is not None:
        lines.append(f"- Weakest feature group: {weakest}.")
    if strongest is None and weakest is None:
        lines.append("- No feature-group rows were produced in this run.")
    lines.append("")
    lines.append("## Skipped ablations")
    lines.append("")
    if skipped:
        for entry in skipped[:40]:
            skip_family = entry.get("ablation_family")
            skip_name = entry.get("ablation_name")
            skip_reason = entry.get("skip_reason")
            lines.append(f"- `{skip_family}/{skip_name}`: {skip_reason}")
    else:
        lines.append("- None.")
    lines.append("")
    lines.append("## Execution metrics are proxies")
    lines.append("")
    lines.append(
        "Execution cost and latency numbers are simplified proxy "
        "diagnostics under stated assumptions, not a backtest."
    )
    lines.append("")
    lines.append("## Boundaries")
    lines.append("")
    for boundary in _CLAIM_BOUNDARIES:
        lines.append(f"- {boundary}")
    lines.append("")
    return "\n".join(lines)


def _headline_feature_group_findings(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str | None, str | None]:
    candidates = [
        row
        for row in rows
        if row.get("ablation_family") == "feature_groups"
        and row.get("split") == "test"
        and row.get("metric_name") == "macro_f1"
        and row.get("status") == "ok"
        and row.get("ablation_name") not in {"all_features", "labels_excluded"}
        and row.get("delta_vs_baseline") is not None
    ]
    if not candidates:
        return None, None
    frame = pd.DataFrame(candidates)
    grouped = (
        frame.groupby("ablation_name")["delta_vs_baseline"].mean().sort_values()
    )
    weakest_name = str(grouped.index[0])
    strongest_name = str(grouped.index[-1])
    strongest = (
        f"`{strongest_name}` (mean test macro-F1 delta "
        f"{grouped.iloc[-1]:+.4f} vs all_features)"
    )
    weakest = (
        f"`{weakest_name}` (mean test macro-F1 delta "
        f"{grouped.iloc[0]:+.4f} vs all_features)"
    )
    return strongest, weakest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_fi2010_brutal_ablations(
    config_path: str | Path,
    *,
    neural_config_path: str | Path | None = None,
    processed_root: str | Path | None = None,
    classical_dir: str | Path | None = None,
    neural_dir: str | Path | None = None,
    out_dir: str | Path,
    families: Sequence[str] | str | None = None,
    folds: Sequence[str | int] | str | None = None,
    models: Sequence[str] | str | None = None,
    neural_lookbacks: Sequence[int] | str | None = None,
    max_epochs: int = 5,
    seed: int = 0,
    overwrite: bool = False,
    dry_run: bool = False,
) -> BrutalAblationSummary:
    """Run the requested FI-2010 brutal ablation families."""
    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_file():
        raise FileNotFoundError(
            f"brutal ablation config not found: {resolved_config_path}",
        )
    config = load_multifold_classical_config(resolved_config_path)

    selected_families = normalise_families(families)
    fold_filter = _normalise_fold_selection(folds)
    requested_classical, requested_neural = _split_requested_models(models)
    fit_models = requested_classical or DEFAULT_FEATURE_GROUP_MODELS
    neural_models = requested_neural or DEFAULT_NEURAL_LOOKBACK_MODELS
    lookback_subset = _normalise_lookbacks(neural_lookbacks)

    resolved_neural_config = (
        Path(neural_config_path) if neural_config_path is not None else None
    )
    resolved_processed = (
        Path(processed_root) if processed_root is not None else None
    )
    resolved_classical = Path(classical_dir) if classical_dir is not None else None
    resolved_neural = Path(neural_dir) if neural_dir is not None else None
    resolved_out_dir = Path(out_dir)

    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    families_run: list[str] = []
    families_skipped: list[str] = []
    fold_ids_seen: list[str] = []
    feature_groups_seen: tuple[str, ...] = ()
    horizons_seen: list[int] = []

    # Resolve the fold inputs for fit-based families.
    fit_fold_inputs: list[tuple[str, Path]] = []
    if _FIT_FAMILIES & set(selected_families):
        fit_fold_inputs, fit_skips = _resolve_fit_fold_inputs(
            config=config,
            processed_root=resolved_processed,
            fold_filter=fold_filter,
        )
        for skip in fit_skips:
            warnings.append(skip)

    if dry_run:
        return _dry_run_summary(
            config_path=resolved_config_path,
            neural_config_path=resolved_neural_config,
            processed_root=resolved_processed,
            classical_dir=resolved_classical,
            neural_dir=resolved_neural,
            out_dir=resolved_out_dir,
            selected_families=selected_families,
            fold_inputs=fit_fold_inputs,
            fold_filter=fold_filter,
            fit_models=fit_models,
            warnings=warnings,
        )

    _ensure_output_dir(resolved_out_dir, overwrite=overwrite)

    # --- Fit-based families (feature_groups, horizon) ---------------------
    run_feature_groups = "feature_groups" in selected_families
    run_horizon = "horizon" in selected_families
    if (run_feature_groups or run_horizon) and not fit_fold_inputs:
        reason = (
            "no prepared fold CSV was available; pass --processed-root with "
            "prepared combined CSVs"
        )
        for family in ("feature_groups", "horizon"):
            if family in selected_families:
                rows.append(_skip_record(family=family, name=family, skip_reason=reason))
                families_skipped.append(family)
        run_feature_groups = run_horizon = False

    for fold_id, fold_path in fit_fold_inputs:
        frame = pd.read_csv(fold_path)
        split = _benchmark_split(frame, config=config, seed=seed)
        feature_columns = _feature_columns(frame, config)
        feature_groups_seen = tuple(resolve_feature_groups(feature_columns).keys())
        if fold_id not in fold_ids_seen:
            fold_ids_seen.append(fold_id)
        baseline_by_model = {
            model_name: _score_classical(
                frame,
                feature_columns=feature_columns,
                target_column=config.label_name,
                split=split,
                model_name=model_name,
                seed=seed,
            )
            for model_name in fit_models
        }
        if run_feature_groups:
            rows.extend(
                _feature_group_rows(
                    frame=frame,
                    fold_id=fold_id,
                    split=split,
                    feature_columns=feature_columns,
                    target_column=config.label_name,
                    models=fit_models,
                    seed=seed,
                    baseline_by_model=baseline_by_model,
                )
            )
        if run_horizon:
            rows.extend(
                _horizon_rows(
                    frame=frame,
                    fold_id=fold_id,
                    split=split,
                    feature_columns=feature_columns,
                    target_column=config.label_name,
                    target_horizon=config.target_horizon,
                    configured_label_columns=config.label_columns,
                    models=fit_models,
                    seed=seed,
                    baseline_by_model=baseline_by_model,
                )
            )
            horizons_seen = sorted(
                {
                    _horizon_from_label(column, default=config.target_horizon)
                    for column in config.label_columns
                }
            )
    if run_feature_groups:
        families_run.append("feature_groups")
    if run_horizon:
        families_run.append("horizon")

    # --- Reuse-based classical families -----------------------------------
    classical_results: pd.DataFrame | None = None
    execution_table: pd.DataFrame | None = None
    if _CLASSICAL_REUSE_FAMILIES & set(selected_families):
        if resolved_classical is None or not resolved_classical.is_dir():
            reason = (
                "no stored classical artefact directory was provided; pass "
                "--classical with results_by_fold.csv"
            )
            for family in ("model_class", "calibration", "execution"):
                if family in selected_families:
                    rows.append(
                        _skip_record(family=family, name=family, skip_reason=reason)
                    )
                    families_skipped.append(family)
        else:
            try:
                classical_results = _load_classical_fold_results(resolved_classical)
            except (OSError, ValueError) as exc:
                warnings.append(f"classical results unavailable: {exc}")
            execution_table = _load_execution_rows(resolved_classical)

    if "model_class" in selected_families and classical_results is not None:
        model_class_rows, model_class_warnings = _model_class_rows(
            results=classical_results,
            baseline_model=DEFAULT_MODEL_CLASS_BASELINE,
            requested_models=requested_classical,
            fold_filter=fold_filter,
        )
        rows.extend(model_class_rows)
        warnings.extend(model_class_warnings)
        families_run.append("model_class")
    elif "model_class" in selected_families and classical_results is None:
        if "model_class" not in families_skipped:
            rows.append(
                _skip_record(
                    family="model_class",
                    name="model_class",
                    skip_reason="stored classical results could not be loaded",
                )
            )
            families_skipped.append("model_class")

    if "calibration" in selected_families and classical_results is not None:
        rows.extend(
            _calibration_rows(
                results=classical_results,
                execution_rows=(
                    execution_table
                    if execution_table is not None
                    else pd.DataFrame()
                ),
                requested_models=requested_classical,
                fold_filter=fold_filter,
            )
        )
        families_run.append("calibration")
    elif "calibration" in selected_families and classical_results is None:
        if "calibration" not in families_skipped:
            rows.append(
                _skip_record(
                    family="calibration",
                    name="calibration",
                    skip_reason="stored classical results could not be loaded",
                )
            )
            families_skipped.append("calibration")

    if "execution" in selected_families:
        if execution_table is not None and not execution_table.empty:
            rows.extend(
                _execution_rows(
                    execution_rows=execution_table,
                    requested_models=requested_classical,
                    fold_filter=fold_filter,
                )
            )
            families_run.append("execution")
        elif "execution" not in families_skipped:
            rows.append(
                _skip_record(
                    family="execution",
                    name="execution",
                    skip_reason="no stored execution proxy table was available",
                )
            )
            families_skipped.append("execution")

    # --- Neural lookback family ------------------------------------------
    if "lookback" in selected_families:
        lookback_rows, lookback_ran, lookback_warning = _resolve_lookback_family(
            neural_config_path=resolved_neural_config,
            processed_root=resolved_processed,
            neural_dir=resolved_neural,
            out_dir=resolved_out_dir,
            fold_filter=fold_filter,
            neural_models=neural_models,
            lookback_subset=lookback_subset,
            max_epochs=max_epochs,
            seed=seed,
        )
        rows.extend(lookback_rows)
        if lookback_warning:
            warnings.append(lookback_warning)
        if lookback_ran:
            families_run.append("lookback")
        else:
            families_skipped.append("lookback")

    # --- Write artefacts --------------------------------------------------
    artefacts = _write_artefacts(
        out_dir=resolved_out_dir,
        rows=rows,
        config_path=resolved_config_path,
        neural_config_path=resolved_neural_config,
        processed_root=resolved_processed,
        classical_dir=resolved_classical,
        neural_dir=resolved_neural,
        selected_families=selected_families,
        families_run=tuple(dict.fromkeys(families_run)),
        families_skipped=tuple(dict.fromkeys(families_skipped)),
        folds=tuple(fold_ids_seen),
        fit_models=fit_models,
        feature_groups=feature_groups_seen,
        horizons=tuple(horizons_seen),
        warnings=tuple(warnings),
    )

    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    skipped_count = sum(1 for row in rows if row.get("status") == "skipped")
    return BrutalAblationSummary(
        output_dir=resolved_out_dir,
        config_path=resolved_config_path,
        neural_config_path=resolved_neural_config,
        processed_root=resolved_processed,
        classical_dir=resolved_classical,
        neural_dir=resolved_neural,
        families_requested=selected_families,
        families_run=tuple(dict.fromkeys(families_run)),
        families_skipped=tuple(dict.fromkeys(families_skipped)),
        folds=tuple(fold_ids_seen),
        fit_models=tuple(fit_models),
        feature_groups=feature_groups_seen,
        horizons=tuple(horizons_seen),
        result_row_count=len(rows),
        ok_row_count=ok_count,
        skipped_count=skipped_count,
        full_predictions_written=False,
        checkpoints_written=False,
        dry_run=False,
        artefacts=artefacts,
        warnings=tuple(warnings),
    )


def _normalise_lookbacks(
    neural_lookbacks: Sequence[int] | str | None,
) -> tuple[int, ...]:
    if neural_lookbacks is None:
        return ()
    if isinstance(neural_lookbacks, str):
        text = neural_lookbacks.strip()
        if not text:
            return ()
        tokens: Sequence[str | int] = [token.strip() for token in text.split(",")]
    else:
        tokens = list(neural_lookbacks)
    cleaned: list[int] = []
    for token in tokens:
        if str(token).strip() == "":
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("neural lookbacks must be positive")
        if value not in cleaned:
            cleaned.append(value)
    return tuple(cleaned)


def _resolve_fit_fold_inputs(
    *,
    config: MultifoldClassicalConfig,
    processed_root: Path | None,
    fold_filter: tuple[str, ...] | None,
) -> tuple[list[tuple[str, Path]], list[str]]:
    skips: list[str] = []
    if processed_root is None:
        return [], ["fit families: no --processed-root was provided"]
    requested_numbers = (
        {_fold_number(fold) for fold in fold_filter}
        if fold_filter is not None
        else set(config.folds)
    )
    inputs: list[tuple[str, Path]] = []
    for fold_number in sorted(requested_numbers):
        filename = config.combined_csv_filename_template.replace(
            "{fold}", str(fold_number)
        )
        path = processed_root / filename
        if path.is_file():
            inputs.append((f"fold_{fold_number}", path))
        else:
            skips.append(
                f"fit families: prepared CSV for fold_{fold_number} is missing "
                f"at {path}"
            )
    return inputs, skips


def _resolve_lookback_family(
    *,
    neural_config_path: Path | None,
    processed_root: Path | None,
    neural_dir: Path | None,
    out_dir: Path,
    fold_filter: tuple[str, ...] | None,
    neural_models: Sequence[str],
    lookback_subset: tuple[int, ...],
    max_epochs: int,
    seed: int,
) -> tuple[list[dict[str, Any]], bool, str]:
    reference_rows = _lookback_reference_rows(
        neural_dir=neural_dir,
        requested_models=neural_models,
        fold_filter=fold_filter,
    )
    if not lookback_subset:
        skip_reason = (
            "neural lookback sweep not requested; this is CPU-expensive, so "
            "pass --neural-lookbacks (and --max-epochs) to execute it"
        )
        rows = list(reference_rows)
        rows.append(
            _skip_record(
                family="lookback",
                name="lookback_sweep",
                skip_reason=skip_reason,
            )
        )
        return rows, False, ""
    if neural_config_path is None or not neural_config_path.is_file():
        rows = list(reference_rows)
        rows.append(
            _skip_record(
                family="lookback",
                name="lookback_sweep",
                skip_reason="neural lookback sweep requested but --neural-config is missing",
            )
        )
        return rows, False, ""
    if processed_root is None:
        rows = list(reference_rows)
        rows.append(
            _skip_record(
                family="lookback",
                name="lookback_sweep",
                skip_reason="neural lookback sweep requested but --processed-root is missing",
            )
        )
        return rows, False, ""
    try:
        executed = _run_neural_lookback(
            neural_config_path=neural_config_path,
            processed_root=processed_root,
            out_dir=out_dir,
            folds=fold_filter,
            lookbacks=lookback_subset,
            models=neural_models,
            seed=seed,
            max_epochs=max_epochs,
        )
    except (
        FileNotFoundError,
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        rows = list(reference_rows)
        rows.append(
            _skip_record(
                family="lookback",
                name="lookback_sweep",
                skip_reason=f"neural lookback sweep failed: {type(exc).__name__}: {exc}",
            )
        )
        return rows, False, f"lookback sweep failed: {type(exc).__name__}: {exc}"
    rows = list(reference_rows)
    rows.extend(executed)
    return rows, bool(executed), ""


def _write_artefacts(
    *,
    out_dir: Path,
    rows: Sequence[dict[str, Any]],
    config_path: Path,
    neural_config_path: Path | None,
    processed_root: Path | None,
    classical_dir: Path | None,
    neural_dir: Path | None,
    selected_families: tuple[str, ...],
    families_run: tuple[str, ...],
    families_skipped: tuple[str, ...],
    folds: tuple[str, ...],
    fit_models: Sequence[str],
    feature_groups: tuple[str, ...],
    horizons: tuple[int, ...],
    warnings: tuple[str, ...],
) -> dict[str, str]:
    _write_csv(rows, out_dir / "ablation_results.csv", ABLATION_RESULT_COLUMNS)
    _write_csv(
        _build_ablation_summary(rows),
        out_dir / "ablation_summary.csv",
        _ABLATION_SUMMARY_COLUMNS,
    )

    family_files = {
        "feature_groups": ("feature_group_ablation.csv", _FEATURE_GROUP_COLUMNS),
        "model_class": ("model_class_ablation.csv", _MODEL_CLASS_COLUMNS),
        "lookback": ("lookback_ablation.csv", _LOOKBACK_COLUMNS),
        "horizon": ("horizon_ablation.csv", _HORIZON_COLUMNS),
        "calibration": ("calibration_threshold_ablation.csv", _CALIBRATION_COLUMNS),
        "execution": ("execution_cost_latency_ablation.csv", _EXECUTION_COLUMNS),
    }
    for family, (filename, columns) in family_files.items():
        family_rows = [row for row in rows if row.get("ablation_family") == family]
        _write_csv(family_rows, out_dir / filename, columns)

    skipped = [row for row in rows if row.get("status") == "skipped"]
    skipped_payload = {
        "skipped_count": len(skipped),
        "skipped": [
            {
                "ablation_family": row.get("ablation_family"),
                "ablation_name": row.get("ablation_name"),
                "fold_id": row.get("fold_id"),
                "model_name": row.get("model_name"),
                "skip_reason": row.get("skip_reason"),
            }
            for row in skipped
        ],
    }
    (out_dir / "skipped_ablations.json").write_text(
        stable_json_dumps(skipped_payload), encoding="utf-8"
    )

    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    summary_for_notes = BrutalAblationSummary(
        output_dir=out_dir,
        config_path=config_path,
        neural_config_path=neural_config_path,
        processed_root=processed_root,
        classical_dir=classical_dir,
        neural_dir=neural_dir,
        families_requested=selected_families,
        families_run=families_run,
        families_skipped=families_skipped,
        folds=folds,
        fit_models=tuple(fit_models),
        feature_groups=feature_groups,
        horizons=horizons,
        result_row_count=len(rows),
        ok_row_count=ok_count,
        skipped_count=len(skipped),
        full_predictions_written=False,
        checkpoints_written=False,
        dry_run=False,
    )
    (out_dir / "ablation_notes.md").write_text(
        _format_notes(summary=summary_for_notes, rows=rows, skipped=skipped),
        encoding="utf-8",
    )

    artefacts = {
        "summary": "summary.json",
        "ablation_results": "ablation_results.csv",
        "ablation_summary": "ablation_summary.csv",
        "skipped_ablations": "skipped_ablations.json",
        "feature_group_ablation": "feature_group_ablation.csv",
        "model_class_ablation": "model_class_ablation.csv",
        "lookback_ablation": "lookback_ablation.csv",
        "horizon_ablation": "horizon_ablation.csv",
        "calibration_threshold_ablation": "calibration_threshold_ablation.csv",
        "execution_cost_latency_ablation": "execution_cost_latency_ablation.csv",
        "ablation_notes": "ablation_notes.md",
    }

    summary_payload: dict[str, Any] = {
        "runner_version": FI2010_BRUTAL_ABLATIONS_VERSION,
        "package_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "inputs": {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "neural_config": (
                str(neural_config_path) if neural_config_path is not None else None
            ),
            "processed_root": (
                str(processed_root) if processed_root is not None else None
            ),
            "classical_dir": str(classical_dir) if classical_dir is not None else None,
            "neural_dir": str(neural_dir) if neural_dir is not None else None,
        },
        "families_requested": list(selected_families),
        "families_run": list(families_run),
        "families_skipped": list(families_skipped),
        "folds": list(folds),
        "fit_models": list(fit_models),
        "feature_groups": list(feature_groups),
        "horizons": list(horizons),
        "counts": {
            "result_rows": len(rows),
            "ok_rows": ok_count,
            "skipped_rows": len(skipped),
        },
        "artefacts": artefacts,
        "full_predictions_written": False,
        "checkpoints_written": False,
        "warnings": list(warnings),
        "environment": _build_environment_payload(),
        "claim_boundaries": list(_CLAIM_BOUNDARIES),
    }
    (out_dir / "summary.json").write_text(
        stable_json_dumps(summary_payload), encoding="utf-8"
    )
    return artefacts


def _build_environment_payload() -> dict[str, str]:
    import platform

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_version": __version__,
    }


def _dry_run_summary(
    *,
    config_path: Path,
    neural_config_path: Path | None,
    processed_root: Path | None,
    classical_dir: Path | None,
    neural_dir: Path | None,
    out_dir: Path,
    selected_families: tuple[str, ...],
    fold_inputs: Sequence[tuple[str, Path]],
    fold_filter: tuple[str, ...] | None,
    fit_models: Sequence[str],
    warnings: Sequence[str],
) -> BrutalAblationSummary:
    folds = (
        tuple(fold_id for fold_id, _ in fold_inputs)
        if fold_inputs
        else (fold_filter or ())
    )
    return BrutalAblationSummary(
        output_dir=out_dir,
        config_path=config_path,
        neural_config_path=neural_config_path,
        processed_root=processed_root,
        classical_dir=classical_dir,
        neural_dir=neural_dir,
        families_requested=selected_families,
        families_run=(),
        families_skipped=(),
        folds=folds,
        fit_models=tuple(fit_models),
        feature_groups=(),
        horizons=(),
        result_row_count=0,
        ok_row_count=0,
        skipped_count=0,
        full_predictions_written=False,
        checkpoints_written=False,
        dry_run=True,
        artefacts={},
        warnings=tuple(warnings),
    )
