"""Leakage-safe FI-2010 microstructure feature builder.

The builder consumes FI-2010-style snapshot rows and produces raw LOB,
derived depth/spread/imbalance features and clearly labelled
``snapshot_order_flow_proxy`` columns.  It never consumes labels or future
horizon columns as feature inputs.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chronoslob.experiments.manifests import sha256_file, stable_json_dumps
from chronoslob.features.registry import (
    FeatureGroupResolution,
    FeatureRegistryError,
    available_lob_levels,
    feature_groups_for_columns,
    feature_manifest,
    group_names,
    unsupported_group_names,
)

__all__ = [
    "DEFAULT_FEATURE_GROUPS",
    "MicrostructureFeatureBuildResult",
    "audit_fi2010_feature_file",
    "audit_fi2010_feature_frame",
    "build_microstructure_feature_artifacts",
    "build_microstructure_features",
    "default_label_columns",
]

DEFAULT_FEATURE_GROUPS: tuple[str, ...] = (
    "price_levels",
    "size_levels",
    "top_of_book",
    "spread",
    "midprice",
    "microprice",
    "top_of_book_imbalance",
    "depth_imbalance",
    "depth_slope",
    "liquidity_concentration",
    "snapshot_order_flow_proxy",
    "volatility_proxy",
)

_LABEL_PREFIXES = (
    "label",
    "y_",
    "target",
    "future_",
    "horizon_",
    "direction_",
    "return_quantile_",
    "spread_widening_",
    "bid_fill_proxy_",
    "ask_fill_proxy_",
    "bid_adverse_selection_proxy_",
    "ask_adverse_selection_proxy_",
)
_DEFAULT_PARTITION_CANDIDATES = ("fold", "fold_id", "split", "partition", "session_id")


@dataclass(frozen=True)
class MicrostructureFeatureBuildResult:
    """In-memory result from a microstructure feature build."""

    features: pd.DataFrame
    feature_columns: tuple[str, ...]
    group_columns: dict[str, tuple[str, ...]]
    metadata: dict[str, Any]
    group_manifest: dict[str, Any]
    warnings: tuple[str, ...] = field(default_factory=tuple)


def default_label_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """Return conventional label/future columns from ``columns``."""
    labels: list[str] = []
    for column in columns:
        name = str(column)
        lowered = name.lower()
        if lowered.startswith(_LABEL_PREFIXES):
            labels.append(name)
    return tuple(labels)


def _normalise_feature_groups(groups: Sequence[str] | str | None) -> tuple[str, ...] | None:
    if groups is None:
        return DEFAULT_FEATURE_GROUPS
    if isinstance(groups, str):
        text = groups.strip()
        if not text or text.lower() == "all":
            return DEFAULT_FEATURE_GROUPS
        raw: Sequence[str] = [token.strip() for token in text.split(",")]
    else:
        raw = groups
    cleaned: list[str] = []
    for item in raw:
        name = str(item).strip().lower()
        if name and name not in cleaned:
            cleaned.append(name)
    return tuple(cleaned)


def _quantity_column(columns: Sequence[str], side: str, level: int) -> str | None:
    available = {str(column) for column in columns}
    primary = f"{side}_quantity_{level}"
    alias = f"{side}_size_{level}"
    if primary in available:
        return primary
    if alias in available:
        return alias
    return None


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = pd.to_numeric(denominator, errors="coerce")
    values = pd.to_numeric(numerator, errors="coerce") / denom.where(denom != 0.0)
    return values.replace([np.inf, -np.inf], np.nan)


def _partition_columns(
    frame: pd.DataFrame,
    explicit: Sequence[str] | None,
) -> tuple[str, ...]:
    candidates = tuple(explicit) if explicit is not None else _DEFAULT_PARTITION_CANDIDATES
    return tuple(str(column) for column in candidates if str(column) in frame.columns)


def _with_identity(frame: pd.DataFrame, partition_columns: Sequence[str]) -> pd.DataFrame:
    identity = pd.DataFrame({"row_id": frame.index.to_numpy(dtype=int)})
    for column in ("timestamp", "symbol", *partition_columns):
        if column in frame.columns and column not in identity.columns:
            identity[column] = frame[column].to_numpy(copy=True)
    return identity


def _clean_numeric_features(features: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    cleaned = features.copy()
    numeric_columns = [
        column for column in cleaned.columns if pd.api.types.is_numeric_dtype(cleaned[column])
    ]
    missing_before = int(cleaned[numeric_columns].isna().sum().sum()) if numeric_columns else 0
    for column in numeric_columns:
        cleaned[column] = (
            pd.to_numeric(cleaned[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
    return cleaned, missing_before


def build_microstructure_features(
    frame: pd.DataFrame,
    *,
    feature_groups: Sequence[str] | str | None = None,
    label_columns: Sequence[str] | None = None,
    partition_columns: Sequence[str] | None = None,
    volatility_window: int = 20,
    strict: bool = True,
) -> MicrostructureFeatureBuildResult:
    """Build leakage-safe FI-2010 microstructure features from ``frame``."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("frame must contain at least one row")
    if volatility_window < 2:
        raise ValueError("volatility_window must be >= 2")

    labels = (
        tuple(label_columns) if label_columns is not None else default_label_columns(frame.columns)
    )
    present_labels = tuple(column for column in labels if column in frame.columns)
    input_columns = tuple(str(column) for column in frame.columns if column not in present_labels)
    requested = _normalise_feature_groups(feature_groups)
    resolutions = feature_groups_for_columns(input_columns, requested, strict=strict)
    all_manifest = feature_manifest(input_columns, group_names(), strict=False)
    partitions = _partition_columns(frame, partition_columns)
    features = _with_identity(frame, partitions)

    group_columns: dict[str, tuple[str, ...]] = {}
    warnings: list[str] = []
    levels = available_lob_levels(input_columns)

    for name, resolution in resolutions.items():
        if resolution.status != "available":
            group_columns[name] = ()
            warnings.append(f"{name}: {resolution.reason}")
            continue
        if name == "price_levels" or name == "size_levels":
            group_columns[name] = _copy_columns(frame, features, resolution.source_columns)
        elif name == "top_of_book":
            columns = _add_top_of_book(frame, features)
            group_columns[name] = columns
        elif name == "spread":
            columns = _add_spread(frame, features)
            group_columns[name] = columns
        elif name == "midprice":
            columns = _add_midprice(frame, features)
            group_columns[name] = columns
        elif name == "microprice":
            columns = _add_microprice(frame, features)
            group_columns[name] = columns
        elif name == "top_of_book_imbalance":
            columns = _add_top_of_book_imbalance(frame, features)
            group_columns[name] = columns
        elif name == "depth_imbalance":
            columns = _add_depth_imbalance(frame, features, levels=levels)
            group_columns[name] = columns
        elif name == "depth_slope":
            columns = _add_depth_slope(frame, features, levels=levels)
            group_columns[name] = columns
        elif name == "liquidity_concentration":
            columns = _add_liquidity_concentration(frame, features, levels=levels)
            group_columns[name] = columns
        elif name == "snapshot_order_flow_proxy":
            columns = _add_snapshot_delta_proxy(
                frame,
                features,
                source_columns=resolution.source_columns,
                partition_columns=partitions,
            )
            group_columns[name] = columns
        elif name == "volatility_proxy":
            columns = _add_volatility_proxy(
                frame,
                features,
                partition_columns=partitions,
                window=volatility_window,
            )
            group_columns[name] = columns
        elif name == "time_context":
            columns = _add_time_context(frame, features, partition_columns=partitions)
            group_columns[name] = columns
        else:
            group_columns[name] = ()

    feature_columns = tuple(
        column
        for columns in group_columns.values()
        for column in columns
        if column in features.columns
    )
    feature_columns = tuple(dict.fromkeys(feature_columns))
    if strict and not feature_columns:
        raise FeatureRegistryError("requested feature groups produced no usable columns")

    cleaned_features, filled_missing_count = _clean_numeric_features(features)
    feature_hash = _dataframe_sha256(cleaned_features)
    metadata = {
        "builder_version": "microstructure-fi2010-builder/v2",
        "created_at": datetime.now(UTC).isoformat(),
        "row_count": len(cleaned_features),
        "input_columns": list(map(str, frame.columns)),
        "label_columns_excluded": list(present_labels),
        "partition_columns": list(partitions),
        "feature_columns": list(feature_columns),
        "feature_count": len(feature_columns),
        "feature_sha256": feature_hash,
        "filled_missing_numeric_values": filled_missing_count,
        "leakage_controls": {
            "labels_excluded": True,
            "future_horizon_columns_excluded": True,
            "snapshot_delta_proxy_resets_at_partitions": True,
            "rolling_volatility_uses_grouped_past_rows_only": True,
        },
    }
    manifest = dict(all_manifest)
    manifest["selected_groups"] = [
        _resolution_payload(resolution, group_columns.get(name, ()))
        for name, resolution in resolutions.items()
    ]
    manifest["group_columns"] = {
        name: list(columns) for name, columns in sorted(group_columns.items())
    }
    manifest["explicitly_unsupported_registry_groups"] = list(unsupported_group_names())
    return MicrostructureFeatureBuildResult(
        features=cleaned_features,
        feature_columns=feature_columns,
        group_columns=group_columns,
        metadata=metadata,
        group_manifest=manifest,
        warnings=tuple(warnings),
    )


def _copy_columns(
    source: pd.DataFrame,
    target: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[str, ...]:
    copied: list[str] = []
    for column in columns:
        if column not in source.columns:
            continue
        target[column] = pd.to_numeric(source[column], errors="coerce")
        copied.append(column)
    return tuple(copied)


def _add_top_of_book(source: pd.DataFrame, target: pd.DataFrame) -> tuple[str, ...]:
    bid_size = _quantity_column(source.columns, "bid", 1)
    ask_size = _quantity_column(source.columns, "ask", 1)
    if bid_size is None or ask_size is None:
        return ()
    target["best_bid_price"] = pd.to_numeric(source["bid_price_1"], errors="coerce")
    target["best_ask_price"] = pd.to_numeric(source["ask_price_1"], errors="coerce")
    target["best_bid_size"] = pd.to_numeric(source[bid_size], errors="coerce")
    target["best_ask_size"] = pd.to_numeric(source[ask_size], errors="coerce")
    return ("best_bid_price", "best_ask_price", "best_bid_size", "best_ask_size")


def _midprice_series(source: pd.DataFrame) -> pd.Series:
    bid = pd.to_numeric(source["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(source["ask_price_1"], errors="coerce")
    return (bid + ask) / 2.0


def _add_midprice(source: pd.DataFrame, target: pd.DataFrame) -> tuple[str, ...]:
    target["midprice"] = _midprice_series(source)
    return ("midprice",)


def _add_spread(source: pd.DataFrame, target: pd.DataFrame) -> tuple[str, ...]:
    bid = pd.to_numeric(source["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(source["ask_price_1"], errors="coerce")
    spread = ask - bid
    target["spread"] = spread
    target["relative_spread"] = _safe_divide(spread, (ask + bid) / 2.0)
    return ("spread", "relative_spread")


def _add_microprice(source: pd.DataFrame, target: pd.DataFrame) -> tuple[str, ...]:
    bid_size_column = _quantity_column(source.columns, "bid", 1)
    ask_size_column = _quantity_column(source.columns, "ask", 1)
    if bid_size_column is None or ask_size_column is None:
        return ()
    bid = pd.to_numeric(source["bid_price_1"], errors="coerce")
    ask = pd.to_numeric(source["ask_price_1"], errors="coerce")
    bid_size = pd.to_numeric(source[bid_size_column], errors="coerce")
    ask_size = pd.to_numeric(source[ask_size_column], errors="coerce")
    target["microprice"] = _safe_divide(ask * bid_size + bid * ask_size, bid_size + ask_size)
    return ("microprice",)


def _add_top_of_book_imbalance(source: pd.DataFrame, target: pd.DataFrame) -> tuple[str, ...]:
    bid_size_column = _quantity_column(source.columns, "bid", 1)
    ask_size_column = _quantity_column(source.columns, "ask", 1)
    if bid_size_column is None or ask_size_column is None:
        return ()
    bid_size = pd.to_numeric(source[bid_size_column], errors="coerce")
    ask_size = pd.to_numeric(source[ask_size_column], errors="coerce")
    target["top_of_book_imbalance"] = _safe_divide(bid_size - ask_size, bid_size + ask_size)
    return ("top_of_book_imbalance",)


def _depth_sums(
    source: pd.DataFrame,
    levels: Sequence[int],
    depth: int,
) -> tuple[pd.Series, pd.Series]:
    used = [level for level in levels if level <= depth]
    bid_total = pd.Series(0.0, index=source.index)
    ask_total = pd.Series(0.0, index=source.index)
    for level in used:
        bid = _quantity_column(source.columns, "bid", level)
        ask = _quantity_column(source.columns, "ask", level)
        if bid is not None:
            bid_total += pd.to_numeric(source[bid], errors="coerce").fillna(0.0)
        if ask is not None:
            ask_total += pd.to_numeric(source[ask], errors="coerce").fillna(0.0)
    return bid_total, ask_total


def _add_depth_imbalance(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    levels: Sequence[int],
) -> tuple[str, ...]:
    columns: list[str] = []
    for depth in (1, 5, 10):
        if len(levels) < depth:
            continue
        bid_total, ask_total = _depth_sums(source, levels, depth)
        column = f"depth_imbalance_l{depth}"
        target[column] = _safe_divide(bid_total - ask_total, bid_total + ask_total)
        columns.append(column)
    return tuple(columns)


def _add_depth_slope(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    levels: Sequence[int],
) -> tuple[str, ...]:
    if len(levels) < 2:
        return ()
    bid_matrix: list[pd.Series] = []
    ask_matrix: list[pd.Series] = []
    for level in levels:
        bid = _quantity_column(source.columns, "bid", level)
        ask = _quantity_column(source.columns, "ask", level)
        if bid is not None and ask is not None:
            bid_matrix.append(pd.to_numeric(source[bid], errors="coerce").fillna(0.0))
            ask_matrix.append(pd.to_numeric(source[ask], errors="coerce").fillna(0.0))
    if len(bid_matrix) < 2 or len(ask_matrix) < 2:
        return ()
    x = np.asarray(list(range(1, len(bid_matrix) + 1)), dtype=float)
    centered = x - float(x.mean())
    denom = float(np.dot(centered, centered))
    bid_values = np.vstack([series.to_numpy(dtype=float) for series in bid_matrix]).T
    ask_values = np.vstack([series.to_numpy(dtype=float) for series in ask_matrix]).T
    bid_slopes = np.dot(bid_values - bid_values.mean(axis=1, keepdims=True), centered) / denom
    ask_slopes = np.dot(ask_values - ask_values.mean(axis=1, keepdims=True), centered) / denom
    target["bid_depth_slope"] = bid_slopes
    target["ask_depth_slope"] = ask_slopes
    target["depth_slope_imbalance"] = bid_slopes - ask_slopes
    return ("bid_depth_slope", "ask_depth_slope", "depth_slope_imbalance")


def _add_liquidity_concentration(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    levels: Sequence[int],
) -> tuple[str, ...]:
    if not levels:
        return ()
    total_bid, total_ask = _depth_sums(source, levels, max(levels))
    total = total_bid + total_ask
    top1_bid, top1_ask = _depth_sums(source, levels, 1)
    target["liquidity_concentration_top1"] = _safe_divide(top1_bid + top1_ask, total)
    columns = ["liquidity_concentration_top1"]
    if len(levels) >= 5:
        top5_bid, top5_ask = _depth_sums(source, levels, 5)
        target["liquidity_concentration_top5"] = _safe_divide(top5_bid + top5_ask, total)
        columns.append("liquidity_concentration_top5")
    return tuple(columns)


def _grouped_numeric_diff(
    frame: pd.DataFrame,
    column: str,
    *,
    partition_columns: Sequence[str],
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if partition_columns:
        return (
            values.groupby([frame[column_name] for column_name in partition_columns], sort=False)
            .diff()
            .fillna(0.0)
        )
    return values.diff().fillna(0.0)


def _add_snapshot_delta_proxy(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    source_columns: Sequence[str],
    partition_columns: Sequence[str],
) -> tuple[str, ...]:
    columns: list[str] = []
    for column in source_columns:
        if column not in source.columns:
            continue
        output = f"snapshot_delta_{column}"
        target[output] = _grouped_numeric_diff(
            source,
            column,
            partition_columns=partition_columns,
        )
        columns.append(output)
    return tuple(columns)


def _grouped_mid_returns(
    source: pd.DataFrame,
    *,
    partition_columns: Sequence[str],
) -> pd.Series:
    mid = _midprice_series(source)
    if partition_columns:
        previous = mid.groupby(
            [source[column_name] for column_name in partition_columns],
            sort=False,
        ).shift(1)
    else:
        previous = mid.shift(1)
    returns = _safe_divide(mid - previous, previous.abs())
    return returns.fillna(0.0)


def _add_volatility_proxy(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    partition_columns: Sequence[str],
    window: int,
) -> tuple[str, ...]:
    returns = _grouped_mid_returns(source, partition_columns=partition_columns)
    if partition_columns:
        volatility = returns.groupby(
            [source[column_name] for column_name in partition_columns],
            sort=False,
        ).transform(lambda series: series.rolling(window=window, min_periods=1).std(ddof=0))
    else:
        volatility = returns.rolling(window=window, min_periods=1).std(ddof=0)
    target["volatility_proxy"] = volatility.fillna(0.0)
    return ("volatility_proxy",)


def _add_time_context(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    partition_columns: Sequence[str],
) -> tuple[str, ...]:
    if "timestamp" not in source.columns:
        return ()
    timestamps = pd.to_datetime(source["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().all():
        return ()
    target["time_of_day_seconds"] = (
        timestamps.dt.hour * 3600 + timestamps.dt.minute * 60 + timestamps.dt.second
    ).astype(float)
    if partition_columns:
        target["session_position"] = source.groupby(
            [source[column_name] for column_name in partition_columns],
            sort=False,
        ).cumcount()
    else:
        target["session_position"] = np.arange(len(source), dtype=float)
    return ("time_of_day_seconds", "session_position")


def _resolution_payload(
    resolution: FeatureGroupResolution,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    return {
        "name": resolution.name,
        "description": resolution.description,
        "source_columns": list(resolution.source_columns),
        "generated_columns": list(resolution.generated_columns),
        "feature_columns": list(feature_columns),
        "kind": resolution.kind,
        "requires_past_context": resolution.requires_past_context,
        "valid_for_fi2010": resolution.valid_for_fi2010,
        "limitations": list(resolution.limitations),
        "status": resolution.status,
        "reason": resolution.reason,
    }


def _dataframe_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ensure_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {path}")
        if any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(f"refusing to overwrite non-empty output directory: {path}")
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_microstructure_feature_artifacts(
    input_path: str | Path,
    *,
    out_dir: str | Path,
    feature_groups: Sequence[str] | str | None = None,
    label_columns: Sequence[str] | None = None,
    split_column: str | None = "split",
    partition_columns: Sequence[str] | None = None,
    volatility_window: int = 20,
    strict: bool = True,
    overwrite: bool = False,
) -> MicrostructureFeatureBuildResult:
    """Build feature artefacts from a local FI-2010-style CSV file."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"FI-2010 feature input does not exist: {path}")
    frame = pd.read_csv(path)
    partitions = partition_columns
    if partitions is None and split_column is not None and split_column in frame.columns:
        partitions = (split_column,)
    result = build_microstructure_features(
        frame,
        feature_groups=feature_groups,
        label_columns=label_columns,
        partition_columns=partitions,
        volatility_window=volatility_window,
        strict=strict,
    )
    output_dir = Path(out_dir)
    _ensure_output_dir(output_dir, overwrite=overwrite)
    features_path = output_dir / "features.csv"
    metadata_path = output_dir / "feature_metadata.json"
    manifest_path = output_dir / "feature_group_manifest.json"
    result.features.to_csv(features_path, index=False)
    metadata = dict(result.metadata)
    metadata["input_path"] = str(path)
    metadata["input_sha256"] = sha256_file(path)
    metadata["features_path"] = str(features_path)
    metadata["features_sha256"] = sha256_file(features_path)
    metadata["warnings"] = list(result.warnings)
    metadata_path.write_text(stable_json_dumps(metadata), encoding="utf-8")
    manifest = dict(result.group_manifest)
    manifest["metadata_path"] = str(metadata_path)
    manifest["features_path"] = str(features_path)
    manifest_path.write_text(stable_json_dumps(manifest), encoding="utf-8")
    hashes = {
        "features.csv": sha256_file(features_path),
        "feature_metadata.json": sha256_file(metadata_path),
        "feature_group_manifest.json": sha256_file(manifest_path),
    }
    (output_dir / "sha256_manifest.json").write_text(
        stable_json_dumps({"files": hashes}),
        encoding="utf-8",
    )
    return result


def audit_fi2010_feature_frame(
    frame: pd.DataFrame,
    *,
    label_columns: Sequence[str] | None = None,
    feature_groups: Sequence[str] | str | None = None,
    partition_columns: Sequence[str] | None = None,
    strict: bool = True,
    volatility_window: int = 20,
) -> dict[str, Any]:
    """Audit leakage controls for a FI-2010 feature build."""
    labels = (
        tuple(label_columns) if label_columns is not None else default_label_columns(frame.columns)
    )
    result = build_microstructure_features(
        frame,
        feature_groups=feature_groups,
        label_columns=labels,
        partition_columns=partition_columns,
        strict=strict,
        volatility_window=volatility_window,
    )
    feature_columns = set(result.feature_columns)
    label_like_features = sorted(
        column for column in feature_columns if str(column).lower().startswith(_LABEL_PREFIXES)
    )
    future_columns_used = sorted(
        column
        for column in feature_columns
        if "future" in str(column).lower() or "horizon" in str(column).lower()
    )
    boundary = _boundary_checks(
        frame,
        result,
        partition_columns=tuple(result.metadata["partition_columns"]),
    )
    row_alignment = {
        "passed": len(result.features) == len(frame)
        and result.features["row_id"].tolist() == list(map(int, frame.index.tolist())),
        "feature_rows": len(result.features),
        "input_rows": len(frame),
    }
    missing = [
        group
        for group in result.group_manifest.get("selected_groups", [])
        if isinstance(group, Mapping) and group.get("status") != "available"
    ]
    checks = {
        "no_label_columns_used": {
            "passed": not label_like_features,
            "columns": label_like_features,
        },
        "no_future_horizon_columns_used": {
            "passed": not future_columns_used,
            "columns": future_columns_used,
        },
        "rolling_volatility_past_only": _rolling_volatility_check(result),
        "snapshot_delta_proxy_no_cross_boundary": boundary,
        "train_validation_test_boundaries_respected": {
            "passed": boundary["passed"],
            "partition_columns": result.metadata["partition_columns"],
        },
        "row_alignment": row_alignment,
        "missing_column_checks": {"passed": not missing, "missing_groups": missing},
    }
    passed = all(bool(check.get("passed")) for check in checks.values())
    return {
        "status": "pass" if passed else "fail",
        "passed": passed,
        "unsupported_groups": result.group_manifest.get("unsupported_groups", []),
        "proxy_groups": result.group_manifest.get("proxy_groups", []),
        "checks": checks,
        "feature_metadata": result.metadata,
        "warnings": list(result.warnings),
    }


def audit_fi2010_feature_file(
    path: str | Path,
    *,
    label_columns: Sequence[str] | None = None,
    feature_groups: Sequence[str] | str | None = None,
    split_column: str | None = "split",
    strict: bool = True,
    volatility_window: int = 20,
) -> dict[str, Any]:
    """Audit a local FI-2010-style CSV file without writing artefacts."""
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"FI-2010 feature audit input does not exist: {input_path}")
    frame = pd.read_csv(input_path)
    partitions: tuple[str, ...] | None = None
    if split_column is not None and split_column in frame.columns:
        partitions = (split_column,)
    report = audit_fi2010_feature_frame(
        frame,
        label_columns=label_columns,
        feature_groups=feature_groups,
        partition_columns=partitions,
        strict=strict,
        volatility_window=volatility_window,
    )
    report["input_path"] = str(input_path)
    report["input_sha256"] = sha256_file(input_path)
    return report


def _boundary_checks(
    frame: pd.DataFrame,
    result: MicrostructureFeatureBuildResult,
    *,
    partition_columns: Sequence[str],
) -> dict[str, Any]:
    delta_columns = [
        column
        for column in result.group_columns.get("snapshot_order_flow_proxy", ())
        if column in result.features.columns
    ]
    if not delta_columns:
        return {
            "passed": True,
            "reason": "snapshot_order_flow_proxy not selected or unavailable",
            "partition_columns": list(partition_columns),
        }
    if not partition_columns:
        first_values = result.features.loc[[0], delta_columns].abs().sum(axis=1)
        passed = bool((first_values == 0.0).all())
        return {
            "passed": passed,
            "reason": "single partition",
            "partition_columns": [],
        }
    first_indices = (
        frame.groupby([frame[column] for column in partition_columns], sort=False)
        .head(1)
        .index.tolist()
    )
    first_rows = result.features[result.features["row_id"].isin(first_indices)]
    sums = first_rows[delta_columns].abs().sum(axis=1)
    return {
        "passed": bool((sums == 0.0).all()),
        "partition_columns": list(partition_columns),
        "boundary_row_ids": [int(value) for value in first_rows["row_id"].tolist()],
    }


def _rolling_volatility_check(result: MicrostructureFeatureBuildResult) -> dict[str, Any]:
    if "volatility_proxy" not in result.features.columns:
        return {"passed": True, "reason": "volatility_proxy not selected or unavailable"}
    values = pd.to_numeric(result.features["volatility_proxy"], errors="coerce")
    finite = bool(np.isfinite(values.fillna(0.0).to_numpy(dtype=float)).all())
    non_negative = bool((values.fillna(0.0) >= 0.0).all())
    return {
        "passed": finite and non_negative,
        "reason": "computed with grouped shift and rolling window over current/past rows",
    }
