"""Regime analysis utilities for organising metrics across market regimes.

The module assigns regime labels to analysis records by either using an
explicit label already present in the record, applying threshold-based rules
on numeric fields or, when explicitly requested, applying boundaries that
were fitted on an upstream training/calibration split.

Regime fitting is deliberately separated from regime assignment to prevent
accidental data-snooping on evaluation data.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DEFAULT_CONFIDENCE_BUCKET_EDGES",
    "DEFAULT_LATENCY_LABELS",
    "DEFAULT_LIQUIDITY_THRESHOLDS",
    "DEFAULT_SPREAD_THRESHOLDS",
    "DEFAULT_VOLATILITY_THRESHOLDS",
    "SUPPORTED_REGIME_KINDS",
    "UNKNOWN_REGIME_LABEL",
    "RegimeAssignment",
    "RegimeDefinition",
    "RegimeMetricSummary",
    "assign_confidence_bucket",
    "assign_latency_regime",
    "assign_liquidity_regime",
    "assign_spread_regime",
    "assign_volatility_regime",
    "fit_regime_boundaries",
    "summarise_by_regime",
]


UNKNOWN_REGIME_LABEL = "unknown"

SUPPORTED_REGIME_KINDS: tuple[str, ...] = (
    "volatility",
    "spread",
    "liquidity",
    "confidence",
    "latency",
)

DEFAULT_VOLATILITY_THRESHOLDS: tuple[float, float] = (0.005, 0.02)
DEFAULT_SPREAD_THRESHOLDS: tuple[float, float] = (0.0005, 0.002)
DEFAULT_LIQUIDITY_THRESHOLDS: tuple[float, float] = (50.0, 250.0)
DEFAULT_CONFIDENCE_BUCKET_EDGES: tuple[float, float, float] = (0.5, 0.7, 0.85)
DEFAULT_LATENCY_LABELS: Mapping[int, str] = {
    0: "zero",
    1: "low",
    2: "medium",
    5: "high",
}


@dataclass(frozen=True)
class RegimeDefinition:
    """A descriptor for an explicit set of regime labels."""

    kind: str
    labels: tuple[str, ...]
    thresholds: tuple[float, ...] | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("RegimeDefinition.kind must be a non-empty string")
        if not isinstance(self.labels, tuple) or not all(
            isinstance(label, str) and label for label in self.labels
        ):
            raise ValueError("RegimeDefinition.labels must be a tuple of strings")
        if self.thresholds is not None:
            if not isinstance(self.thresholds, tuple) or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in self.thresholds
            ):
                raise ValueError(
                    "RegimeDefinition.thresholds must be a tuple of numbers"
                )
            if len(self.thresholds) != max(0, len(self.labels) - 1):
                raise ValueError(
                    "RegimeDefinition.thresholds must have len(labels) - 1 entries"
                )


@dataclass(frozen=True)
class RegimeAssignment:
    """The result of assigning a regime label to a single record."""

    kind: str
    label: str
    value: float | None
    source: str

    def __post_init__(self) -> None:
        if self.source not in {"explicit", "threshold", "fitted", "default"}:
            raise ValueError(
                "RegimeAssignment.source must be one of "
                "{'explicit', 'threshold', 'fitted', 'default'}"
            )


@dataclass(frozen=True)
class RegimeMetricSummary:
    """Aggregated metric values for a single regime label and metric name."""

    kind: str
    label: str
    metric_name: str
    count: int
    mean: float | None
    minimum: float | None
    maximum: float | None
    is_synthetic: bool

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("RegimeMetricSummary.count must be non-negative")


def _coerce_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _label_from_thresholds(
    value: float | None,
    thresholds: Sequence[float],
    labels: Sequence[str],
) -> str:
    if value is None:
        return UNKNOWN_REGIME_LABEL
    if len(thresholds) + 1 != len(labels):
        raise ValueError("thresholds and labels are incompatible")
    for boundary, label in zip(thresholds, labels[:-1], strict=False):
        if value < boundary:
            return label
    return labels[-1]


def _record_value(record: Mapping[str, Any], field_name: str) -> Any:
    if not isinstance(record, Mapping):
        raise TypeError("record must be a Mapping")
    return record.get(field_name)


def _explicit_label(
    record: Mapping[str, Any], explicit_field: str | None
) -> str | None:
    if explicit_field is None:
        return None
    raw = _record_value(record, explicit_field)
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        return None
    return raw


def assign_volatility_regime(
    record: Mapping[str, Any],
    *,
    field_name: str = "volatility",
    explicit_field: str | None = "regime",
    thresholds: Sequence[float] = DEFAULT_VOLATILITY_THRESHOLDS,
    labels: Sequence[str] = ("low", "medium", "high"),
) -> RegimeAssignment:
    """Assign a volatility regime label using explicit value or thresholds."""
    explicit = _explicit_label(record, explicit_field)
    if explicit is not None and explicit in labels:
        value = _coerce_finite_float(_record_value(record, field_name))
        return RegimeAssignment(
            kind="volatility", label=explicit, value=value, source="explicit"
        )
    value = _coerce_finite_float(_record_value(record, field_name))
    label = _label_from_thresholds(value, list(thresholds), list(labels))
    source = "threshold" if value is not None else "default"
    return RegimeAssignment(
        kind="volatility", label=label, value=value, source=source
    )


def assign_spread_regime(
    record: Mapping[str, Any],
    *,
    field_name: str = "spread",
    explicit_field: str | None = "spread_regime",
    thresholds: Sequence[float] = DEFAULT_SPREAD_THRESHOLDS,
    labels: Sequence[str] = ("tight", "normal", "wide"),
) -> RegimeAssignment:
    """Assign a spread regime label using explicit value or thresholds."""
    explicit = _explicit_label(record, explicit_field)
    if explicit is not None and explicit in labels:
        value = _coerce_finite_float(_record_value(record, field_name))
        return RegimeAssignment(
            kind="spread", label=explicit, value=value, source="explicit"
        )
    value = _coerce_finite_float(_record_value(record, field_name))
    label = _label_from_thresholds(value, list(thresholds), list(labels))
    source = "threshold" if value is not None else "default"
    return RegimeAssignment(kind="spread", label=label, value=value, source=source)


def assign_liquidity_regime(
    record: Mapping[str, Any],
    *,
    field_name: str = "liquidity",
    explicit_field: str | None = "liquidity_regime",
    thresholds: Sequence[float] = DEFAULT_LIQUIDITY_THRESHOLDS,
    labels: Sequence[str] = ("thin", "normal", "deep"),
) -> RegimeAssignment:
    """Assign a liquidity regime label using explicit value or thresholds."""
    explicit = _explicit_label(record, explicit_field)
    if explicit is not None and explicit in labels:
        value = _coerce_finite_float(_record_value(record, field_name))
        return RegimeAssignment(
            kind="liquidity", label=explicit, value=value, source="explicit"
        )
    value = _coerce_finite_float(_record_value(record, field_name))
    label = _label_from_thresholds(value, list(thresholds), list(labels))
    source = "threshold" if value is not None else "default"
    return RegimeAssignment(
        kind="liquidity", label=label, value=value, source=source
    )


def assign_confidence_bucket(
    record: Mapping[str, Any],
    *,
    field_name: str = "confidence",
    explicit_field: str | None = "confidence_bucket",
    edges: Sequence[float] = DEFAULT_CONFIDENCE_BUCKET_EDGES,
    labels: Sequence[str] = ("low", "medium", "high", "very_high"),
) -> RegimeAssignment:
    """Assign a confidence bucket label using explicit value or edges."""
    if len(edges) + 1 != len(labels):
        raise ValueError("edges must have len(labels) - 1 entries")
    explicit = _explicit_label(record, explicit_field)
    if explicit is not None and explicit in labels:
        value = _coerce_finite_float(_record_value(record, field_name))
        return RegimeAssignment(
            kind="confidence", label=explicit, value=value, source="explicit"
        )
    value = _coerce_finite_float(_record_value(record, field_name))
    label = _label_from_thresholds(value, list(edges), list(labels))
    source = "threshold" if value is not None else "default"
    return RegimeAssignment(
        kind="confidence", label=label, value=value, source=source
    )


def assign_latency_regime(
    record: Mapping[str, Any],
    *,
    field_name: str = "latency_steps",
    explicit_field: str | None = "latency_regime",
    label_map: Mapping[int, str] = DEFAULT_LATENCY_LABELS,
) -> RegimeAssignment:
    """Assign a latency regime label using an integer-step lookup."""
    explicit = _explicit_label(record, explicit_field)
    if explicit is not None:
        value = _coerce_finite_float(_record_value(record, field_name))
        return RegimeAssignment(
            kind="latency", label=explicit, value=value, source="explicit"
        )
    raw = _record_value(record, field_name)
    if raw is None or isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return RegimeAssignment(
            kind="latency", label=UNKNOWN_REGIME_LABEL, value=None, source="default"
        )
    numeric = float(raw)
    if not math.isfinite(numeric):
        return RegimeAssignment(
            kind="latency", label=UNKNOWN_REGIME_LABEL, value=None, source="default"
        )
    int_value = int(numeric)
    label = label_map.get(int_value)
    if label is None:
        sorted_keys = sorted(label_map.keys())
        label = label_map[sorted_keys[-1]] if sorted_keys else UNKNOWN_REGIME_LABEL
        for key in sorted_keys:
            if int_value <= key:
                label = label_map[key]
                break
    return RegimeAssignment(
        kind="latency", label=label, value=numeric, source="threshold"
    )


@dataclass(frozen=True)
class _GroupAccumulator:
    values: list[float] = field(default_factory=list)
    synthetic_flags: set[bool] = field(default_factory=set)


def summarise_by_regime(
    records: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    metric_name: str,
    metric_field: str = "metric_value",
    metric_name_field: str = "metric_name",
    assigner: Any = None,
    synthetic_field: str = "is_synthetic",
) -> list[RegimeMetricSummary]:
    """Group records by regime label and compute count/mean/min/max."""
    if kind not in SUPPORTED_REGIME_KINDS:
        raise ValueError(
            f"unsupported regime kind {kind!r}; "
            f"supported: {SUPPORTED_REGIME_KINDS}"
        )
    if assigner is None:
        assigner = {
            "volatility": assign_volatility_regime,
            "spread": assign_spread_regime,
            "liquidity": assign_liquidity_regime,
            "confidence": assign_confidence_bucket,
            "latency": assign_latency_regime,
        }[kind]

    groups: dict[str, _GroupAccumulator] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("each record must be a Mapping")
        if record.get(metric_name_field) != metric_name:
            continue
        assignment = assigner(record)
        if assignment.kind != kind:
            raise ValueError(
                "regime assigner produced kind "
                f"{assignment.kind!r} but expected {kind!r}"
            )
        bucket = groups.setdefault(assignment.label, _GroupAccumulator())
        value = _coerce_finite_float(record.get(metric_field))
        if value is not None:
            bucket.values.append(value)
        synthetic = record.get(synthetic_field, False)
        bucket.synthetic_flags.add(bool(synthetic))

    summaries: list[RegimeMetricSummary] = []
    for label in sorted(groups.keys()):
        bucket = groups[label]
        count = len(bucket.values)
        if count == 0:
            mean: float | None = None
            minimum: float | None = None
            maximum: float | None = None
        else:
            mean = sum(bucket.values) / count
            minimum = min(bucket.values)
            maximum = max(bucket.values)
        # Synthetic flag is True only if every record in the bucket is synthetic.
        is_synthetic = (
            bool(bucket.synthetic_flags) and all(bucket.synthetic_flags)
        )
        summaries.append(
            RegimeMetricSummary(
                kind=kind,
                label=label,
                metric_name=metric_name,
                count=count,
                mean=mean,
                minimum=minimum,
                maximum=maximum,
                is_synthetic=is_synthetic,
            )
        )
    return summaries


def fit_regime_boundaries(
    train_values: Sequence[float],
    *,
    n_bins: int = 3,
    method: str = "quantile",
) -> tuple[float, ...]:
    """Fit boundary values for regime assignment from a training sample.

    This function is explicitly intended to be called with values from a
    training or calibration split only. Callers must not pass evaluation
    or test data here.
    """
    if method != "quantile":
        raise ValueError("only method='quantile' is currently supported")
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins < 2:
        raise ValueError("n_bins must be an integer >= 2")
    finite_values: list[float] = []
    for value in train_values:
        coerced = _coerce_finite_float(value)
        if coerced is not None:
            finite_values.append(coerced)
    if not finite_values:
        raise ValueError("train_values must contain at least one finite number")
    finite_values.sort()
    n = len(finite_values)
    boundaries: list[float] = []
    for k in range(1, n_bins):
        fraction = k / n_bins
        index = int(fraction * (n - 1))
        boundaries.append(finite_values[index])
    return tuple(boundaries)
