"""Sensitivity-curve utilities for organising sweep results.

The module organises supplied parameter-sweep records (e.g. confidence
threshold, latency steps, fee bps) into ordered curves and selects the
best point under a stated metric direction.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "SENSITIVITY_PARAMETERS",
    "MetricDirection",
    "SensitivityCurve",
    "SensitivityParameter",
    "SensitivityPoint",
    "build_sensitivity_curve",
    "compare_sensitivity_curves",
    "summarise_sensitivity_curve",
]


MetricDirection = Literal["higher_is_better", "lower_is_better"]

SENSITIVITY_PARAMETERS: tuple[str, ...] = (
    "confidence_threshold",
    "latency_steps",
    "fee_bps",
    "spread_multiplier",
    "turnover_cap",
    "inventory_cap",
    "mask_probability",
    "temperature",
)


@dataclass(frozen=True)
class SensitivityParameter:
    """A description of a sensitivity parameter and its semantics."""

    name: str
    description: str = ""
    units: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SensitivityParameter.name must be non-empty")
        if self.name not in SENSITIVITY_PARAMETERS:
            raise ValueError(
                f"SensitivityParameter.name must be one of {SENSITIVITY_PARAMETERS}"
            )


@dataclass(frozen=True)
class SensitivityPoint:
    """A single (parameter_value, metric_value) sample for one metric."""

    parameter_name: str
    parameter_value: float
    metric_name: str
    metric_value: float | None
    is_synthetic: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.parameter_name:
            raise ValueError("SensitivityPoint.parameter_name must be non-empty")
        if self.parameter_name not in SENSITIVITY_PARAMETERS:
            raise ValueError(
                f"parameter_name must be one of {SENSITIVITY_PARAMETERS}; "
                f"got {self.parameter_name!r}"
            )
        if isinstance(self.parameter_value, bool) or not isinstance(
            self.parameter_value, (int, float)
        ):
            raise TypeError("SensitivityPoint.parameter_value must be a real number")
        if not math.isfinite(float(self.parameter_value)):
            raise ValueError("SensitivityPoint.parameter_value must be finite")
        if self.metric_value is not None:
            if isinstance(self.metric_value, bool) or not isinstance(
                self.metric_value, (int, float)
            ):
                raise TypeError(
                    "SensitivityPoint.metric_value must be a real number or None"
                )
            if not math.isfinite(float(self.metric_value)):
                raise ValueError("SensitivityPoint.metric_value must be finite or None")


@dataclass(frozen=True)
class SensitivityCurve:
    """An ordered sensitivity curve for one parameter and metric."""

    parameter_name: str
    metric_name: str
    metric_direction: MetricDirection
    points: tuple[SensitivityPoint, ...]
    is_synthetic: bool

    def parameter_values(self) -> tuple[float, ...]:
        return tuple(point.parameter_value for point in self.points)

    def metric_values(self) -> tuple[float | None, ...]:
        return tuple(point.metric_value for point in self.points)


def _coerce_points(
    records: Iterable[SensitivityPoint | Mapping[str, Any]],
) -> list[SensitivityPoint]:
    coerced: list[SensitivityPoint] = []
    for record in records:
        if isinstance(record, SensitivityPoint):
            coerced.append(record)
            continue
        if not isinstance(record, Mapping):
            raise TypeError(
                "each sensitivity record must be a SensitivityPoint or Mapping"
            )
        metric_value_raw = record.get("metric_value")
        metric_value: float | None
        metric_value = None if metric_value_raw is None else float(metric_value_raw)
        coerced.append(
            SensitivityPoint(
                parameter_name=str(record.get("parameter_name", "")),
                parameter_value=float(record["parameter_value"]),
                metric_name=str(record.get("metric_name", "")),
                metric_value=metric_value,
                is_synthetic=bool(record.get("is_synthetic", False)),
                notes=str(record.get("notes", "")),
            )
        )
    return coerced


def build_sensitivity_curve(
    records: Iterable[SensitivityPoint | Mapping[str, Any]],
    *,
    parameter_name: str,
    metric_name: str,
    metric_direction: MetricDirection,
) -> SensitivityCurve:
    """Build an ordered sensitivity curve for ``parameter_name``/``metric_name``."""
    if parameter_name not in SENSITIVITY_PARAMETERS:
        raise ValueError(
            f"parameter_name must be one of {SENSITIVITY_PARAMETERS}; "
            f"got {parameter_name!r}"
        )
    if metric_direction not in ("higher_is_better", "lower_is_better"):
        raise ValueError(
            "metric_direction must be 'higher_is_better' or 'lower_is_better'"
        )
    coerced = _coerce_points(records)
    relevant = [
        point
        for point in coerced
        if point.parameter_name == parameter_name
        and point.metric_name == metric_name
    ]
    seen_parameter_values: set[float] = set()
    for point in relevant:
        if point.parameter_value in seen_parameter_values:
            raise ValueError(
                "duplicate parameter_value "
                f"{point.parameter_value!r} for "
                f"{parameter_name!r}/{metric_name!r}"
            )
        seen_parameter_values.add(point.parameter_value)
    relevant.sort(key=lambda point: point.parameter_value)
    is_synthetic = (
        bool(relevant) and all(point.is_synthetic for point in relevant)
    )
    return SensitivityCurve(
        parameter_name=parameter_name,
        metric_name=metric_name,
        metric_direction=metric_direction,
        points=tuple(relevant),
        is_synthetic=is_synthetic,
    )


def summarise_sensitivity_curve(curve: SensitivityCurve) -> dict[str, Any]:
    """Summarise a sensitivity curve and select its best point if any."""
    valid_points = [point for point in curve.points if point.metric_value is not None]
    if not valid_points:
        return {
            "parameter_name": curve.parameter_name,
            "metric_name": curve.metric_name,
            "metric_direction": curve.metric_direction,
            "n_points": len(curve.points),
            "n_valid_points": 0,
            "best_parameter_value": None,
            "best_metric_value": None,
            "is_synthetic": curve.is_synthetic,
        }
    if curve.metric_direction == "higher_is_better":
        best = max(valid_points, key=lambda point: float(point.metric_value or 0.0))
    else:
        best = min(valid_points, key=lambda point: float(point.metric_value or 0.0))
    return {
        "parameter_name": curve.parameter_name,
        "metric_name": curve.metric_name,
        "metric_direction": curve.metric_direction,
        "n_points": len(curve.points),
        "n_valid_points": len(valid_points),
        "best_parameter_value": best.parameter_value,
        "best_metric_value": best.metric_value,
        "is_synthetic": curve.is_synthetic,
    }


def compare_sensitivity_curves(
    curves: Sequence[SensitivityCurve],
) -> dict[str, Any]:
    """Compare multiple sensitivity curves that share metric semantics."""
    if not curves:
        return {
            "metric_name": None,
            "metric_direction": None,
            "n_curves": 0,
            "best_curve_index": None,
            "best_parameter_value": None,
            "best_metric_value": None,
            "summaries": [],
        }
    metric_names = {curve.metric_name for curve in curves}
    metric_directions = {curve.metric_direction for curve in curves}
    if len(metric_names) != 1 or len(metric_directions) != 1:
        raise ValueError(
            "all curves must share the same metric_name and metric_direction"
        )
    summaries = [summarise_sensitivity_curve(curve) for curve in curves]
    best_index: int | None = None
    best_value: float | None = None
    direction = curves[0].metric_direction
    for index, summary in enumerate(summaries):
        candidate = summary["best_metric_value"]
        if candidate is None:
            continue
        if best_value is None:
            best_index = index
            best_value = float(candidate)
            continue
        if (
            direction == "higher_is_better" and float(candidate) > best_value
        ) or (
            direction == "lower_is_better" and float(candidate) < best_value
        ):
            best_index = index
            best_value = float(candidate)
    if best_index is None:
        best_parameter_value: float | None = None
    else:
        best_parameter_value = summaries[best_index]["best_parameter_value"]
    return {
        "metric_name": curves[0].metric_name,
        "metric_direction": direction,
        "n_curves": len(curves),
        "best_curve_index": best_index,
        "best_parameter_value": best_parameter_value,
        "best_metric_value": best_value,
        "summaries": summaries,
    }
