"""Label pipeline for future-window market-state labels."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from chronoslob.data.schemas import (
    DataQualityIssue,
    LabelRow,
    LabelValue,
    OrderBookSnapshot,
    Side,
)
from chronoslob.data.validation import DataValidationResult
from chronoslob.labels.adverse_selection import (
    compute_adverse_selection_after_fill_proxy,
)
from chronoslob.labels.fill_probability import compute_passive_fill_proxy
from chronoslob.labels.midprice import (
    classify_direction,
    compute_future_return,
    compute_return_quantile_labels,
)
from chronoslob.labels.spread import compute_spread_widening_label
from chronoslob.labels.volatility import compute_future_realised_volatility

if TYPE_CHECKING:
    from chronoslob.data.fi2010 import FI2010Dataset

__all__ = [
    "LabelPipelineConfig",
    "build_label_frame_from_fi2010",
    "build_label_frame_from_snapshots",
    "build_label_rows_from_snapshots",
    "validate_label_frame",
]


class LabelPipelineConfig(BaseModel):
    """Configuration for deterministic future-window label generation."""

    model_config = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)

    horizons: tuple[int, ...] = (10, 50, 100)
    return_threshold: float = 0.0
    log_returns: bool = True
    include_direction: bool = True
    include_return: bool = True
    include_return_quantiles: bool = True
    include_volatility: bool = True
    include_spread_widening: bool = True
    include_fill_probability: bool = True
    include_adverse_selection: bool = True
    fill_horizon: int = 10
    adverse_evaluation_horizon: int = 50
    missing: str = "drop"

    @field_validator("horizons")
    @classmethod
    def _validate_horizons(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("horizons must contain at least one positive integer")
        for horizon in value:
            if isinstance(horizon, bool) or not isinstance(horizon, int):
                raise TypeError("horizons must contain integers")
            if horizon <= 0:
                raise ValueError(f"horizons must be positive; got {horizon!r}")
        return tuple(value)

    @field_validator("return_threshold")
    @classmethod
    def _validate_return_threshold(cls, value: float) -> float:
        return _validate_non_negative_finite(value, name="return_threshold")

    @field_validator("fill_horizon")
    @classmethod
    def _validate_fill_horizon(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("fill_horizon must be an integer")
        if value <= 0:
            raise ValueError("fill_horizon must be positive")
        return value

    @field_validator("missing")
    @classmethod
    def _validate_missing(cls, value: str) -> str:
        if value not in {"drop", "none"}:
            raise ValueError("missing must be one of {'drop', 'none'}")
        return value

    @model_validator(mode="after")
    def _validate_adverse_horizon(self) -> LabelPipelineConfig:
        if (
            isinstance(self.adverse_evaluation_horizon, bool)
            or not isinstance(self.adverse_evaluation_horizon, int)
        ):
            raise TypeError("adverse_evaluation_horizon must be an integer")
        if self.adverse_evaluation_horizon < self.fill_horizon:
            raise ValueError("adverse_evaluation_horizon must be >= fill_horizon")
        return self


def _validate_non_negative_finite(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _is_timezone_aware(timestamp: datetime) -> bool:
    return (
        timestamp.tzinfo is not None
        and timestamp.tzinfo.utcoffset(timestamp) is not None
    )


def _check_snapshots(snapshots: Sequence[OrderBookSnapshot]) -> list[OrderBookSnapshot]:
    if not snapshots:
        raise ValueError("snapshots must be non-empty")
    seq = list(snapshots)
    previous: datetime | None = None
    for position, snapshot in enumerate(seq):
        if not isinstance(snapshot, OrderBookSnapshot):
            raise TypeError(
                "snapshots must contain OrderBookSnapshot objects; "
                f"got {type(snapshot).__name__} at index {position}"
            )
        if not _is_timezone_aware(snapshot.timestamp):
            raise ValueError(f"snapshots[{position}].timestamp must be timezone-aware")
        if previous is not None and snapshot.timestamp < previous:
            raise ValueError("snapshots must be ordered by non-decreasing timestamp")
        if snapshot.mid_price is None:
            raise ValueError("snapshots require top-of-book bid and ask levels")
        if snapshot.spread is None:
            raise ValueError("snapshots require top-of-book bid and ask levels")
        snapshot.assert_not_crossed()
        previous = snapshot.timestamp
    return seq


def _mid_prices(snapshots: Sequence[OrderBookSnapshot]) -> list[float]:
    values: list[float] = []
    for position, snapshot in enumerate(snapshots):
        mid_price = snapshot.mid_price
        if mid_price is None:
            raise ValueError(f"snapshots[{position}] has no mid-price")
        values.append(mid_price)
    return values


def _spreads(snapshots: Sequence[OrderBookSnapshot]) -> list[float]:
    values: list[float] = []
    for position, snapshot in enumerate(snapshots):
        spread = snapshot.spread
        if spread is None:
            raise ValueError(f"snapshots[{position}] has no spread")
        values.append(spread)
    return values


def _max_required_horizon(config: LabelPipelineConfig) -> int:
    required = max(config.horizons)
    if config.include_fill_probability:
        required = max(required, config.fill_horizon)
    if config.include_adverse_selection:
        required = max(required, config.adverse_evaluation_horizon)
    return required


def _eligible_indices(length: int, required_horizon: int) -> range:
    if length <= required_horizon:
        return range(0)
    return range(length - required_horizon)


def _precompute_returns(
    mid_prices: Sequence[float],
    indices: range,
    config: LabelPipelineConfig,
) -> dict[int, dict[int, float]]:
    by_horizon: dict[int, dict[int, float]] = {}
    for horizon in config.horizons:
        by_index: dict[int, float] = {}
        for idx in indices:
            by_index[idx] = compute_future_return(
                mid_prices,
                idx,
                horizon,
                log_return=config.log_returns,
            )
        by_horizon[horizon] = by_index
    return by_horizon


def _precompute_quantiles(
    returns_by_horizon: dict[int, dict[int, float]],
) -> dict[int, dict[int, int]]:
    quantiles_by_horizon: dict[int, dict[int, int]] = {}
    for horizon, by_index in returns_by_horizon.items():
        ordered_indices = sorted(by_index)
        labels = compute_return_quantile_labels(
            [by_index[idx] for idx in ordered_indices]
        )
        quantiles_by_horizon[horizon] = dict(zip(ordered_indices, labels, strict=True))
    return quantiles_by_horizon


def _build_labels_for_index(
    snapshots: Sequence[OrderBookSnapshot],
    mid_prices: Sequence[float],
    spreads: Sequence[float],
    index: int,
    config: LabelPipelineConfig,
    returns_by_horizon: dict[int, dict[int, float]],
    quantiles_by_horizon: dict[int, dict[int, int]],
) -> dict[str, LabelValue]:
    labels: dict[str, LabelValue] = {}
    for horizon in config.horizons:
        future_return = returns_by_horizon[horizon][index]
        if config.include_return:
            labels[f"future_return_{horizon}"] = future_return
        if config.include_direction:
            labels[f"direction_{horizon}"] = classify_direction(
                future_return,
                config.return_threshold,
            )
        if config.include_return_quantiles:
            labels[f"return_quantile_{horizon}"] = quantiles_by_horizon[horizon][index]
        if config.include_volatility:
            labels[f"future_volatility_{horizon}"] = (
                compute_future_realised_volatility(mid_prices, index, horizon)
            )
        if config.include_spread_widening:
            labels[f"spread_widening_{horizon}"] = compute_spread_widening_label(
                spreads,
                index,
                horizon,
            )

    if config.include_fill_probability:
        labels[f"bid_fill_proxy_{config.fill_horizon}"] = compute_passive_fill_proxy(
            snapshots,
            index,
            config.fill_horizon,
            Side.BID,
        )
        labels[f"ask_fill_proxy_{config.fill_horizon}"] = compute_passive_fill_proxy(
            snapshots,
            index,
            config.fill_horizon,
            Side.ASK,
        )

    if config.include_adverse_selection:
        horizon = config.adverse_evaluation_horizon
        labels[f"bid_adverse_selection_proxy_{horizon}"] = (
            compute_adverse_selection_after_fill_proxy(
                snapshots,
                index,
                config.fill_horizon,
                horizon,
                Side.BID,
            )
        )
        labels[f"ask_adverse_selection_proxy_{horizon}"] = (
            compute_adverse_selection_after_fill_proxy(
                snapshots,
                index,
                config.fill_horizon,
                horizon,
                Side.ASK,
            )
        )
    return labels


def build_label_rows_from_snapshots(
    snapshots: Sequence[OrderBookSnapshot],
    config: LabelPipelineConfig | None = None,
) -> list[LabelRow]:
    """Build future-window :class:`LabelRow` objects from ordered snapshots."""
    cfg = LabelPipelineConfig() if config is None else config
    seq = _check_snapshots(snapshots)
    mids = _mid_prices(seq)
    spread_values = _spreads(seq)
    required_horizon = _max_required_horizon(cfg)
    indices = _eligible_indices(len(seq), required_horizon)
    returns_by_horizon = _precompute_returns(mids, indices, cfg)
    quantiles_by_horizon = (
        _precompute_quantiles(returns_by_horizon)
        if cfg.include_return_quantiles
        else {}
    )

    rows: list[LabelRow] = []
    for idx in indices:
        labels = _build_labels_for_index(
            seq,
            mids,
            spread_values,
            idx,
            cfg,
            returns_by_horizon,
            quantiles_by_horizon,
        )
        horizon_start = (
            seq[idx].timestamp
            if required_horizon == 1
            else seq[idx + 1].timestamp
        )
        row = LabelRow(
            timestamp=seq[idx].timestamp,
            symbol=seq[idx].symbol,
            labels=labels,
            horizon_start=horizon_start,
            horizon_end=seq[idx + required_horizon].timestamp,
            metadata={
                "source": "chronoslob_generated",
                "missing_policy": cfg.missing,
                "max_horizon": required_horizon,
            },
        )
        rows.append(row)
    return rows


def _label_row_to_record(row: LabelRow) -> dict[str, object]:
    record: dict[str, object] = {
        "timestamp": row.timestamp,
        "symbol": row.symbol,
        "horizon_start": row.horizon_start,
        "horizon_end": row.horizon_end,
    }
    record.update(row.labels)
    return record


def build_label_frame_from_snapshots(
    snapshots: Sequence[OrderBookSnapshot],
    config: LabelPipelineConfig | None = None,
) -> pd.DataFrame:
    """Build a pandas label frame from ordered snapshots."""
    rows = build_label_rows_from_snapshots(snapshots, config=config)
    return pd.DataFrame.from_records([_label_row_to_record(row) for row in rows])


def _timestamps_from_dataset(dataset: FI2010Dataset) -> pd.Series:
    cfg = dataset.config
    if cfg.timestamp_column is not None and cfg.timestamp_column in dataset.frame.columns:
        return pd.to_datetime(
            dataset.frame[cfg.timestamp_column],
            utc=True,
            errors="raise",
        )
    base = datetime(2000, 1, 1, tzinfo=UTC)
    return pd.Series(
        [base + timedelta(seconds=idx) for idx in range(dataset.n_rows)],
        index=dataset.frame.index,
    )


def build_label_frame_from_fi2010(
    dataset: FI2010Dataset,
    config: LabelPipelineConfig | None = None,
    *,
    prefer_existing_labels: bool = True,
) -> pd.DataFrame:
    """Build or extract a label frame from an FI-2010-style dataset.

    Existing FI-2010 labels are preserved as benchmark labels with
    ``label_source='fi2010_existing_labels'``. They are not presented as
    ChronosLOB-generated labels.
    """
    cfg = LabelPipelineConfig() if config is None else config
    if prefer_existing_labels and dataset.has_labels:
        timestamps = _timestamps_from_dataset(dataset).reset_index(drop=True)
        labels = dataset.get_labels().reset_index(drop=True).copy()
        frame = pd.DataFrame(
            {
                "timestamp": timestamps,
                "symbol": dataset.config.symbol,
            }
        )
        split_values = dataset.split_values()
        if split_values is not None:
            frame["split"] = split_values.reset_index(drop=True)
        for column in labels.columns:
            frame[column] = labels[column]
        frame["label_source"] = "fi2010_existing_labels"
        return frame

    snapshots = dataset.to_snapshot_rows()
    if not snapshots:
        raise ValueError(
            "FI-2010 dataset has no snapshots available for generated labels"
        )
    return build_label_frame_from_snapshots(snapshots, config=cfg)


_NON_LABEL_COLUMNS = {
    "timestamp",
    "symbol",
    "horizon_start",
    "horizon_end",
    "split",
    "label_source",
}
_FEATURE_ONLY_NAMES = (
    "mid_price",
    "spread",
    "microprice",
    "bid_depth",
    "ask_depth",
    "imbalance",
)


def _append_issue(
    issues: list[DataQualityIssue],
    *,
    severity: str,
    code: str,
    message: str,
    metadata: dict[str, str | int | float | bool] | None = None,
) -> None:
    issues.append(
        DataQualityIssue(
            severity=severity,
            code=code,
            message=message,
            metadata=metadata or {},
        )
    )


def _requires_horizon_columns(frame: pd.DataFrame) -> bool:
    if "label_source" not in frame.columns:
        return True
    sources = {str(value) for value in frame["label_source"].dropna().unique()}
    return sources != {"fi2010_existing_labels"}


def _label_columns(frame: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in frame.columns
        if str(column) not in _NON_LABEL_COLUMNS
    ]


def _series_has_inf(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    values = series.to_numpy(copy=False)
    try:
        return bool(np.isinf(values).any())
    except TypeError:
        return False


def validate_label_frame(
    frame: pd.DataFrame,
    *,
    allow_nan: bool = False,
) -> DataValidationResult:
    """Validate a pandas label frame without mutating it."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    issues: list[DataQualityIssue] = []
    n_rows, n_columns = frame.shape
    if n_rows == 0:
        _append_issue(
            issues,
            severity="error",
            code="empty_frame",
            message="label frame contains no rows",
        )

    for required in ("timestamp", "symbol"):
        if required not in frame.columns:
            _append_issue(
                issues,
                severity="error",
                code="missing_required_column",
                message=f"label frame is missing required column {required!r}",
                metadata={"column": required},
            )

    if _requires_horizon_columns(frame):
        for required in ("horizon_start", "horizon_end"):
            if required not in frame.columns:
                _append_issue(
                    issues,
                    severity="error",
                    code="missing_horizon_column",
                    message=f"label frame is missing horizon column {required!r}",
                    metadata={"column": required},
                )

    label_columns = _label_columns(frame)
    if not label_columns:
        _append_issue(
            issues,
            severity="error",
            code="no_label_columns",
            message="label frame contains no label value columns",
        )

    for column in label_columns:
        lowered = column.lower()
        if any(
            lowered == name or lowered.startswith(f"{name}_")
            for name in _FEATURE_ONLY_NAMES
        ):
            _append_issue(
                issues,
                severity="error",
                code="feature_like_column_in_labels",
                message=f"label frame contains suspicious feature column {column!r}",
                metadata={"column": column},
            )
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            if _series_has_inf(series):
                _append_issue(
                    issues,
                    severity="error",
                    code="inf_in_label",
                    message=f"label column {column!r} contains infinite values",
                    metadata={"column": column},
                )
            if not allow_nan and series.isna().any():
                _append_issue(
                    issues,
                    severity="error",
                    code="nan_in_label",
                    message=f"label column {column!r} contains NaN values",
                    metadata={"column": column},
                )
        elif not allow_nan and series.isna().any():
            _append_issue(
                issues,
                severity="error",
                code="nan_in_label",
                message=f"label column {column!r} contains missing values",
                metadata={"column": column},
            )

    return DataValidationResult(
        ok=not any(issue.severity == "error" for issue in issues),
        issues=issues,
        n_rows=n_rows,
        n_columns=n_columns,
    )
