"""Ablation analysis utilities for comparing variants against a baseline.

The module organises supplied ablation result records into structured
comparisons against a baseline configuration. It does not train models
and does not invent results.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "ABLATION_CATEGORIES",
    "AblationComparison",
    "AblationResult",
    "AblationSpec",
    "MetricDirection",
    "compare_against_baseline",
    "rank_ablations",
    "summarise_ablation_table",
]


MetricDirection = Literal["higher_is_better", "lower_is_better"]

ABLATION_CATEGORIES: tuple[str, ...] = (
    "feature",
    "token_field",
    "model_component",
    "objective",
    "task_head",
    "execution_setting",
)


@dataclass(frozen=True)
class AblationSpec:
    """A descriptor of a single ablation configuration."""

    name: str
    category: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("AblationSpec.name must be a non-empty string")
        if self.category not in ABLATION_CATEGORIES:
            raise ValueError(
                f"AblationSpec.category must be one of {ABLATION_CATEGORIES}; "
                f"got {self.category!r}"
            )


@dataclass(frozen=True)
class AblationResult:
    """A metric record attributed to one ablation configuration."""

    ablation_name: str
    metric_name: str
    metric_value: float
    metric_direction: MetricDirection
    is_synthetic: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.ablation_name:
            raise ValueError("AblationResult.ablation_name must be non-empty")
        if not self.metric_name:
            raise ValueError("AblationResult.metric_name must be non-empty")
        if isinstance(self.metric_value, bool) or not isinstance(
            self.metric_value, (int, float)
        ):
            raise TypeError("AblationResult.metric_value must be a real number")
        if not math.isfinite(float(self.metric_value)):
            raise ValueError("AblationResult.metric_value must be finite")
        if self.metric_direction not in ("higher_is_better", "lower_is_better"):
            raise ValueError(
                "AblationResult.metric_direction must be "
                "'higher_is_better' or 'lower_is_better'"
            )


@dataclass(frozen=True)
class AblationComparison:
    """A baseline-relative comparison for a single ablation and metric."""

    baseline_name: str
    ablation_name: str
    metric_name: str
    metric_direction: MetricDirection
    baseline_value: float
    ablation_value: float
    absolute_delta: float
    relative_delta: float | None
    is_improvement: bool
    is_synthetic: bool


def _coerce_results(
    records: Iterable[AblationResult | Mapping[str, Any]],
) -> list[AblationResult]:
    coerced: list[AblationResult] = []
    for record in records:
        if isinstance(record, AblationResult):
            coerced.append(record)
            continue
        if not isinstance(record, Mapping):
            raise TypeError(
                "each ablation record must be an AblationResult or Mapping"
            )
        coerced.append(
            AblationResult(
                ablation_name=str(record.get("ablation_name", "")),
                metric_name=str(record.get("metric_name", "")),
                metric_value=float(record["metric_value"]),
                metric_direction=str(
                    record.get("metric_direction", "higher_is_better")
                ),  # type: ignore[arg-type]
                is_synthetic=bool(record.get("is_synthetic", False)),
                notes=str(record.get("notes", "")),
            )
        )
    return coerced


def _safe_relative_delta(
    ablation: float, baseline: float
) -> float | None:
    if baseline == 0.0 or not math.isfinite(baseline):
        return None
    return (ablation - baseline) / baseline


def compare_against_baseline(
    records: Iterable[AblationResult | Mapping[str, Any]],
    *,
    baseline_name: str,
    metric_name: str,
) -> list[AblationComparison]:
    """Compute per-ablation comparisons against ``baseline_name``."""
    if not baseline_name:
        raise ValueError("baseline_name must be a non-empty string")
    if not metric_name:
        raise ValueError("metric_name must be a non-empty string")
    coerced = _coerce_results(records)
    relevant = [result for result in coerced if result.metric_name == metric_name]

    baseline_candidates = [
        result for result in relevant if result.ablation_name == baseline_name
    ]
    if not baseline_candidates:
        raise KeyError(
            f"baseline {baseline_name!r} not found in records for metric "
            f"{metric_name!r}"
        )
    if len(baseline_candidates) > 1:
        raise ValueError(
            f"multiple baseline entries for {baseline_name!r} on metric "
            f"{metric_name!r}; expected exactly one"
        )
    baseline = baseline_candidates[0]

    comparisons: list[AblationComparison] = []
    for result in relevant:
        if result.ablation_name == baseline_name:
            continue
        if result.metric_direction != baseline.metric_direction:
            raise ValueError(
                "metric_direction mismatch between baseline and "
                f"{result.ablation_name!r} on metric {metric_name!r}"
            )
        absolute_delta = float(result.metric_value) - float(baseline.metric_value)
        relative_delta = _safe_relative_delta(
            float(result.metric_value), float(baseline.metric_value)
        )
        if result.metric_direction == "higher_is_better":
            is_improvement = absolute_delta > 0.0
        else:
            is_improvement = absolute_delta < 0.0
        comparisons.append(
            AblationComparison(
                baseline_name=baseline_name,
                ablation_name=result.ablation_name,
                metric_name=metric_name,
                metric_direction=result.metric_direction,
                baseline_value=float(baseline.metric_value),
                ablation_value=float(result.metric_value),
                absolute_delta=absolute_delta,
                relative_delta=relative_delta,
                is_improvement=is_improvement,
                is_synthetic=bool(baseline.is_synthetic and result.is_synthetic),
            )
        )
    comparisons.sort(key=lambda comp: comp.ablation_name)
    return comparisons


def rank_ablations(
    comparisons: Sequence[AblationComparison],
) -> list[AblationComparison]:
    """Rank ablations from largest improvement to largest regression.

    Sorting uses the absolute delta with the metric direction applied so
    that higher improvement appears first regardless of direction.
    """
    if not comparisons:
        return []
    direction = comparisons[0].metric_direction
    if any(comparison.metric_direction != direction for comparison in comparisons):
        raise ValueError("all comparisons must share the same metric_direction")
    sign = 1.0 if direction == "higher_is_better" else -1.0
    return sorted(
        comparisons,
        key=lambda comparison: (
            -sign * comparison.absolute_delta,
            comparison.ablation_name,
        ),
    )


def summarise_ablation_table(
    comparisons: Sequence[AblationComparison],
) -> dict[str, Any]:
    """Produce a structured summary of a ranked ablation table."""
    if not comparisons:
        return {
            "metric_name": None,
            "metric_direction": None,
            "n_comparisons": 0,
            "n_improvements": 0,
            "n_regressions": 0,
            "is_synthetic": False,
            "ranked": [],
        }
    metric_names = {comparison.metric_name for comparison in comparisons}
    if len(metric_names) != 1:
        raise ValueError("comparisons must share a single metric_name")
    metric_name = next(iter(metric_names))
    ranked = rank_ablations(comparisons)
    n_improvements = sum(1 for comparison in comparisons if comparison.is_improvement)
    n_regressions = len(comparisons) - n_improvements
    is_synthetic = all(comparison.is_synthetic for comparison in comparisons)
    return {
        "metric_name": metric_name,
        "metric_direction": comparisons[0].metric_direction,
        "n_comparisons": len(comparisons),
        "n_improvements": n_improvements,
        "n_regressions": n_regressions,
        "is_synthetic": is_synthetic,
        "ranked": [
            {
                "ablation_name": comparison.ablation_name,
                "baseline_value": comparison.baseline_value,
                "ablation_value": comparison.ablation_value,
                "absolute_delta": comparison.absolute_delta,
                "relative_delta": comparison.relative_delta,
                "is_improvement": comparison.is_improvement,
            }
            for comparison in ranked
        ],
    }
