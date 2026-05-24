"""Phase F paper evidence helpers.

This module computes calibration and execution-aware sensitivity
artefacts from stored paper experiment predictions. It is deliberately
small, typed and deterministic, and it works directly on
``predictions.csv``-shaped rows so it does not require model objects or
any retraining.

The execution-sensitivity calculation is an explicit simplified proxy
analysis under stated cost assumptions. It is not a production
backtest, does not simulate live order placement and does not imply
tradable profitability.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "CALIBRATION_BINS_COLUMNS",
    "EXECUTION_SENSITIVITY_COLUMNS",
    "CalibrationBinRow",
    "CalibrationConfig",
    "CalibrationSummary",
    "ExecutionRow",
    "ExecutionSensitivityConfig",
    "ExecutionSummary",
    "ReturnProxyConfig",
    "build_calibration_bins",
    "build_execution_sensitivity",
    "build_return_proxy_series",
    "summarise_calibration",
    "summarise_execution",
    "write_calibration_bins_csv",
    "write_execution_sensitivity_csv",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)

CALIBRATION_BINS_COLUMNS: tuple[str, ...] = (
    "model_name",
    "split",
    "bin_index",
    "bin_lower",
    "bin_upper",
    "count",
    "mean_confidence",
    "accuracy",
    "confidence_gap",
)

EXECUTION_SENSITIVITY_COLUMNS: tuple[str, ...] = (
    "model_name",
    "split",
    "confidence_threshold",
    "cost_bps",
    "latency_steps",
    "eligible_predictions",
    "trade_count_proxy",
    "turnover_proxy",
    "gross_signal_return_proxy",
    "cost_proxy",
    "net_signal_return_proxy",
    "hit_rate_proxy",
)


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------


class CalibrationConfig(BaseModel):
    """Configuration for calibration-bin evidence."""

    model_config = _MODEL_CONFIG

    enabled: bool = True
    n_bins: int = 10

    @field_validator("n_bins")
    @classmethod
    def _validate_n_bins(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("n_bins must be an integer")
        if value <= 0:
            raise ValueError("n_bins must be positive")
        if value > 1000:
            raise ValueError("n_bins must be at most 1000")
        return value


class ReturnProxyConfig(BaseModel):
    """Configuration for the simplified forward-return proxy series."""

    model_config = _MODEL_CONFIG

    kind: str = "mid_forward_return"
    bid_price_column: str = "bid_price_1"
    ask_price_column: str = "ask_price_1"
    horizon_offset: int | None = None

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("return proxy kind must be a non-empty string")
        cleaned = value.strip().lower()
        if cleaned != "mid_forward_return":
            raise ValueError(
                "only 'mid_forward_return' is supported as a return proxy"
            )
        return cleaned

    @field_validator("bid_price_column", "ask_price_column")
    @classmethod
    def _validate_columns(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("return proxy column names must be non-empty strings")
        return value.strip()

    @field_validator("horizon_offset")
    @classmethod
    def _validate_horizon_offset(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("horizon_offset must be an integer when provided")
        if value <= 0:
            raise ValueError("horizon_offset must be positive when provided")
        return value


class ExecutionSensitivityConfig(BaseModel):
    """Configuration for cost-aware signal sensitivity evidence."""

    model_config = _MODEL_CONFIG

    enabled: bool = True
    confidence_thresholds: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7)
    cost_bps: tuple[float, ...] = (0.0, 1.0, 5.0)
    latency_steps: tuple[int, ...] = (0,)
    class_direction_map: dict[str, int] | None = None
    return_proxy: ReturnProxyConfig = Field(default_factory=ReturnProxyConfig)

    @field_validator("confidence_thresholds")
    @classmethod
    def _validate_thresholds(cls, value: Sequence[float]) -> tuple[float, ...]:
        cleaned: list[float] = []
        for raw in value:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError("confidence_thresholds must be finite numbers")
            numeric = float(raw)
            if not math.isfinite(numeric):
                raise ValueError("confidence_thresholds must be finite")
            if numeric < 0.0 or numeric > 1.0:
                raise ValueError("confidence_thresholds must lie in [0, 1]")
            cleaned.append(numeric)
        if not cleaned:
            raise ValueError("confidence_thresholds must not be empty")
        return tuple(sorted(set(cleaned)))

    @field_validator("cost_bps")
    @classmethod
    def _validate_cost_bps(cls, value: Sequence[float]) -> tuple[float, ...]:
        cleaned: list[float] = []
        for raw in value:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError("cost_bps must be finite numbers")
            numeric = float(raw)
            if not math.isfinite(numeric):
                raise ValueError("cost_bps must be finite")
            if numeric < 0.0:
                raise ValueError("cost_bps must be non-negative")
            cleaned.append(numeric)
        if not cleaned:
            raise ValueError("cost_bps must not be empty")
        return tuple(sorted(set(cleaned)))

    @field_validator("latency_steps")
    @classmethod
    def _validate_latency_steps(cls, value: Sequence[int]) -> tuple[int, ...]:
        cleaned: list[int] = []
        for raw in value:
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise ValueError("latency_steps must be integers")
            if raw < 0:
                raise ValueError("latency_steps must be non-negative")
            cleaned.append(int(raw))
        if not cleaned:
            raise ValueError("latency_steps must not be empty")
        return tuple(sorted(set(cleaned)))

    @field_validator("class_direction_map")
    @classmethod
    def _validate_class_direction_map(
        cls,
        value: Mapping[str, int] | None,
    ) -> dict[str, int] | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("class_direction_map must be a mapping when provided")
        cleaned: dict[str, int] = {}
        for label_key, sign in value.items():
            label_str = str(label_key).strip()
            if not label_str:
                raise ValueError("class_direction_map keys must be non-empty strings")
            if isinstance(sign, bool) or not isinstance(sign, int):
                raise ValueError("class_direction_map values must be integers")
            if sign not in (-1, 0, 1):
                raise ValueError("class_direction_map values must be -1, 0 or 1")
            cleaned[label_str] = int(sign)
        if not cleaned:
            raise ValueError("class_direction_map must contain at least one entry")
        return cleaned

    @model_validator(mode="after")
    def _validate_non_trivial(self) -> ExecutionSensitivityConfig:
        if not self.confidence_thresholds:
            raise ValueError("at least one confidence threshold is required")
        if not self.cost_bps:
            raise ValueError("at least one cost_bps value is required")
        if not self.latency_steps:
            raise ValueError("at least one latency step is required")
        return self


# ---------------------------------------------------------------------------
# Output row containers
# ---------------------------------------------------------------------------


class CalibrationBinRow(BaseModel):
    """One reliability bin record for one model and split."""

    model_config = _MODEL_CONFIG

    model_name: str
    split: str
    bin_index: int
    bin_lower: float
    bin_upper: float
    count: int
    mean_confidence: float
    accuracy: float
    confidence_gap: float

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "split": self.split,
            "bin_index": int(self.bin_index),
            "bin_lower": float(self.bin_lower),
            "bin_upper": float(self.bin_upper),
            "count": int(self.count),
            "mean_confidence": float(self.mean_confidence),
            "accuracy": float(self.accuracy),
            "confidence_gap": float(self.confidence_gap),
        }


class CalibrationSummary(BaseModel):
    """Per-model calibration summary across all bins on one split."""

    model_config = _MODEL_CONFIG

    model_name: str
    split: str
    n_bins: int
    n_samples: int
    mean_confidence: float
    expected_calibration_error: float


class ExecutionRow(BaseModel):
    """One execution-sensitivity record for one model and parameter set."""

    model_config = _MODEL_CONFIG

    model_name: str
    split: str
    confidence_threshold: float
    cost_bps: float
    latency_steps: int
    eligible_predictions: int
    trade_count_proxy: int
    turnover_proxy: float
    gross_signal_return_proxy: float
    cost_proxy: float
    net_signal_return_proxy: float
    hit_rate_proxy: float

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "split": self.split,
            "confidence_threshold": float(self.confidence_threshold),
            "cost_bps": float(self.cost_bps),
            "latency_steps": int(self.latency_steps),
            "eligible_predictions": int(self.eligible_predictions),
            "trade_count_proxy": int(self.trade_count_proxy),
            "turnover_proxy": float(self.turnover_proxy),
            "gross_signal_return_proxy": float(self.gross_signal_return_proxy),
            "cost_proxy": float(self.cost_proxy),
            "net_signal_return_proxy": float(self.net_signal_return_proxy),
            "hit_rate_proxy": float(self.hit_rate_proxy),
        }


class ExecutionSummary(BaseModel):
    """Per-model execution-sensitivity summary recorded in runner outputs."""

    model_config = _MODEL_CONFIG

    model_name: str
    split: str
    n_rows: int
    confidence_thresholds: list[float]
    cost_bps: list[float]
    latency_steps: list[int]


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def _bin_edges(n_bins: int) -> np.ndarray:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    return np.linspace(0.0, 1.0, int(n_bins) + 1)


def _bin_assignment(
    confidences: np.ndarray,
    *,
    n_bins: int,
) -> np.ndarray:
    raw = np.floor(confidences * float(n_bins)).astype(int)
    return np.clip(raw, 0, n_bins - 1)


def build_calibration_bins(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    split: str,
    n_bins: int,
) -> list[CalibrationBinRow]:
    """Compute reliability-bin rows for one model on one prediction split.

    ``predictions`` must contain at least the ``confidence``, ``label``
    and ``prediction`` columns. Rows with non-finite confidence are
    skipped. Empty bins are recorded with ``count`` 0 and finite
    placeholder values so that the bin edges remain deterministic in
    the resulting artefact.
    """
    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame")
    if predictions.empty:
        return []
    if "confidence" not in predictions.columns:
        return []
    if "label" not in predictions.columns or "prediction" not in predictions.columns:
        return []

    confidences = pd.to_numeric(
        predictions["confidence"],
        errors="coerce",
    ).to_numpy(dtype=float)
    correct = (
        predictions["prediction"].astype(str).to_numpy()
        == predictions["label"].astype(str).to_numpy()
    )
    finite_mask = np.isfinite(confidences)
    confidences = confidences[finite_mask]
    correct = correct[finite_mask]

    edges = _bin_edges(n_bins)
    rows: list[CalibrationBinRow] = []
    if confidences.size == 0:
        for bin_index in range(n_bins):
            lower = float(edges[bin_index])
            upper = float(edges[bin_index + 1])
            rows.append(
                CalibrationBinRow(
                    model_name=model_name,
                    split=split,
                    bin_index=bin_index,
                    bin_lower=lower,
                    bin_upper=upper,
                    count=0,
                    mean_confidence=(lower + upper) / 2.0,
                    accuracy=0.0,
                    confidence_gap=0.0,
                )
            )
        return rows

    assignments = _bin_assignment(confidences, n_bins=n_bins)

    for bin_index in range(n_bins):
        lower = float(edges[bin_index])
        upper = float(edges[bin_index + 1])
        in_bin = assignments == bin_index
        count = int(in_bin.sum())
        if count == 0:
            mean_confidence = (lower + upper) / 2.0
            accuracy = 0.0
            gap = 0.0
        else:
            mean_confidence = float(confidences[in_bin].mean())
            accuracy = float(correct[in_bin].mean())
            gap = abs(mean_confidence - accuracy)
        rows.append(
            CalibrationBinRow(
                model_name=model_name,
                split=split,
                bin_index=bin_index,
                bin_lower=lower,
                bin_upper=upper,
                count=count,
                mean_confidence=mean_confidence,
                accuracy=accuracy,
                confidence_gap=gap,
            )
        )
    return rows


def summarise_calibration(
    rows: Sequence[CalibrationBinRow],
) -> CalibrationSummary | None:
    """Aggregate per-bin rows into a per-model calibration summary."""
    if not rows:
        return None
    total = sum(int(row.count) for row in rows)
    if total <= 0:
        return CalibrationSummary(
            model_name=rows[0].model_name,
            split=rows[0].split,
            n_bins=len(rows),
            n_samples=0,
            mean_confidence=0.0,
            expected_calibration_error=0.0,
        )
    weighted_confidence = (
        sum(row.mean_confidence * int(row.count) for row in rows) / float(total)
    )
    ece = sum(row.confidence_gap * (int(row.count) / float(total)) for row in rows)
    return CalibrationSummary(
        model_name=rows[0].model_name,
        split=rows[0].split,
        n_bins=len(rows),
        n_samples=total,
        mean_confidence=float(weighted_confidence),
        expected_calibration_error=float(ece),
    )


def write_calibration_bins_csv(
    rows: Iterable[CalibrationBinRow],
    path: Path,
) -> int:
    """Write calibration bins to a CSV at ``path``; return the row count."""
    materialised = [row.to_csv_row() for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialised:
        frame = pd.DataFrame(columns=list(CALIBRATION_BINS_COLUMNS))
    else:
        frame = pd.DataFrame(materialised)
        frame = frame.loc[:, list(CALIBRATION_BINS_COLUMNS)]
    frame.to_csv(path, index=False)
    return len(materialised)


# ---------------------------------------------------------------------------
# Execution-sensitivity helpers
# ---------------------------------------------------------------------------


def build_return_proxy_series(
    frame: pd.DataFrame,
    *,
    horizon: int,
    config: ReturnProxyConfig,
) -> pd.Series | None:
    """Return forward mid-price return in basis points, indexed by row.

    Returns ``None`` if the required price columns are missing or do not
    contain finite numeric data. Rows for which the forward horizon
    falls outside the frame are filled with ``NaN``.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if config.bid_price_column not in frame.columns:
        return None
    if config.ask_price_column not in frame.columns:
        return None
    bid = pd.to_numeric(frame[config.bid_price_column], errors="coerce")
    ask = pd.to_numeric(frame[config.ask_price_column], errors="coerce")
    if bid.isna().all() or ask.isna().all():
        return None
    mid = (bid + ask) / 2.0
    if not np.isfinite(mid.to_numpy(dtype=float)).any():
        return None
    offset = int(config.horizon_offset) if config.horizon_offset is not None else int(horizon)
    if offset <= 0:
        return None
    mid_forward = mid.shift(-offset)
    safe_mid = mid.where(mid > 0.0, other=np.nan)
    ratio = (mid_forward - mid) / safe_mid
    proxy_bps = ratio * 10_000.0
    proxy_bps.index = frame.index
    return proxy_bps


def _default_direction_map(distinct_labels: Sequence[str]) -> dict[str, int]:
    """Return a sensible default sign mapping for integer-coded labels.

    The mapping follows the FI-2010 convention where ``1`` is an upward
    move, ``2`` is a downward move and ``3`` is a stationary class. For
    label sets that do not match this convention the returned mapping
    leaves all labels at sign ``0`` so a config-supplied
    ``class_direction_map`` is required for a meaningful run.
    """
    convention: dict[str, int] = {"1": 1, "2": -1, "3": 0}
    return {label: convention.get(str(label), 0) for label in distinct_labels}


def _resolve_direction_map(
    *,
    config: ExecutionSensitivityConfig,
    distinct_labels: Sequence[str],
) -> dict[str, int]:
    if config.class_direction_map is not None:
        return {str(key): int(value) for key, value in config.class_direction_map.items()}
    return _default_direction_map(distinct_labels)


def _safe_float(value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        return 0.0
    return numeric


def build_execution_sensitivity(
    predictions: pd.DataFrame,
    *,
    model_name: str,
    split: str,
    return_proxy: pd.Series,
    config: ExecutionSensitivityConfig,
    direction_map: Mapping[str, int],
) -> list[ExecutionRow]:
    """Compute execution-sensitivity rows for one model and split.

    The calculation is a simplified proxy:

    * ``direction_sign`` for each row comes from ``direction_map``
      applied to the predicted class label.
    * ``gross_signal_return_proxy`` is the mean of
      ``direction_sign * forward_mid_return_bps`` over eligible rows.
    * ``trade_count_proxy`` counts rows with a non-zero direction sign
      and confidence above the threshold.
    * ``cost_proxy`` is ``cost_bps`` multiplied by the per-row trade
      indicator.
    * ``net_signal_return_proxy`` is ``gross`` minus ``cost`` on the
      same per-row basis.

    Rows without a valid forward return (out of frame) are excluded.
    The result is empty when no row meets the threshold; in that case
    the runner still records the parameter combination with zero counts
    so the artefact remains traceable.
    """
    if predictions.empty:
        return []
    if "confidence" not in predictions.columns:
        return []
    if "prediction" not in predictions.columns or "label" not in predictions.columns:
        return []
    if "row_index" not in predictions.columns:
        return []

    confidences = pd.to_numeric(
        predictions["confidence"],
        errors="coerce",
    ).to_numpy(dtype=float)
    row_indices = pd.to_numeric(
        predictions["row_index"],
        errors="coerce",
    ).to_numpy(dtype=float)
    predicted_labels = predictions["prediction"].astype(str).to_numpy()
    true_labels = predictions["label"].astype(str).to_numpy()
    if confidences.size == 0:
        return []

    proxy_values = np.full(confidences.shape[0], np.nan, dtype=float)
    proxy_index = return_proxy.index
    for position, raw_row_index in enumerate(row_indices):
        if not math.isfinite(raw_row_index):
            continue
        lookup = int(raw_row_index)
        if lookup not in proxy_index:
            continue
        value = float(return_proxy.loc[lookup])
        if math.isfinite(value):
            proxy_values[position] = value

    signs = np.array(
        [int(direction_map.get(label, 0)) for label in predicted_labels],
        dtype=int,
    )
    correct = (predicted_labels == true_labels).astype(int)

    rows: list[ExecutionRow] = []
    for threshold in config.confidence_thresholds:
        for latency in config.latency_steps:
            shifted_proxy = _shift_proxy(
                proxy_values=proxy_values,
                latency=latency,
            )
            eligible_mask = (
                np.isfinite(confidences)
                & (confidences >= threshold)
                & np.isfinite(shifted_proxy)
            )
            trade_mask = eligible_mask & (signs != 0)
            eligible_count = int(eligible_mask.sum())
            trade_count = int(trade_mask.sum())
            hit_rate = (
                0.0
                if eligible_count == 0
                else float(correct[eligible_mask].mean())
            )
            turnover_proxy = float(np.abs(signs[trade_mask]).sum()) if trade_count else 0.0
            for cost in config.cost_bps:
                if trade_count == 0:
                    gross = 0.0
                    cost_proxy = 0.0
                    net = 0.0
                else:
                    contributions = signs[trade_mask] * shifted_proxy[trade_mask]
                    gross = float(np.nanmean(contributions))
                    cost_proxy = float(cost)
                    net = gross - cost_proxy
                rows.append(
                    ExecutionRow(
                        model_name=model_name,
                        split=split,
                        confidence_threshold=float(threshold),
                        cost_bps=float(cost),
                        latency_steps=int(latency),
                        eligible_predictions=eligible_count,
                        trade_count_proxy=trade_count,
                        turnover_proxy=_safe_float(turnover_proxy),
                        gross_signal_return_proxy=_safe_float(gross),
                        cost_proxy=_safe_float(cost_proxy),
                        net_signal_return_proxy=_safe_float(net),
                        hit_rate_proxy=_safe_float(hit_rate),
                    )
                )
    return rows


def _shift_proxy(
    *,
    proxy_values: np.ndarray,
    latency: int,
) -> np.ndarray:
    """Apply a non-negative latency shift to the per-row proxy values.

    ``latency`` is interpreted as "the realised forward return arrives
    ``latency`` rows later than the prediction row". This is a
    simplified pessimistic assumption: latency rows at the end of the
    prediction window are dropped (NaN) because their realised proxy
    is not available within the held-out predictions.
    """
    if latency <= 0:
        return proxy_values.copy()
    shifted = np.full_like(proxy_values, np.nan, dtype=float)
    if latency >= proxy_values.size:
        return shifted
    shifted[: -latency] = proxy_values[latency:]
    return shifted


def summarise_execution(
    rows: Sequence[ExecutionRow],
    *,
    model_name: str,
    split: str,
) -> ExecutionSummary:
    """Aggregate per-row execution records into a per-model summary."""
    thresholds = sorted({float(row.confidence_threshold) for row in rows})
    costs = sorted({float(row.cost_bps) for row in rows})
    latencies = sorted({int(row.latency_steps) for row in rows})
    return ExecutionSummary(
        model_name=model_name,
        split=split,
        n_rows=len(rows),
        confidence_thresholds=thresholds,
        cost_bps=costs,
        latency_steps=latencies,
    )


def write_execution_sensitivity_csv(
    rows: Iterable[ExecutionRow],
    path: Path,
) -> int:
    """Write execution-sensitivity rows to a CSV at ``path``."""
    materialised = [row.to_csv_row() for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialised:
        frame = pd.DataFrame(columns=list(EXECUTION_SENSITIVITY_COLUMNS))
    else:
        frame = pd.DataFrame(materialised)
        frame = frame.loc[:, list(EXECUTION_SENSITIVITY_COLUMNS)]
    frame.to_csv(path, index=False)
    return len(materialised)
