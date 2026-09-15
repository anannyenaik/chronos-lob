"""Transfer analysis utilities for in-domain versus out-of-domain comparisons.

The module organises supplied transfer-style result records into a matrix
keyed by (train_scope, eval_scope, metric_name) and supports simple
in-domain versus out-of-domain comparison. It does not train models and
does not generate evidence by itself.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "TransferMatrix",
    "TransferResult",
    "TransferSplit",
    "build_transfer_matrix",
    "compare_in_domain_vs_out_of_domain",
    "summarise_transfer_results",
]


@dataclass(frozen=True)
class TransferSplit:
    """A description of one source/target transfer split."""

    source_scope: str
    target_scope: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.source_scope:
            raise ValueError("TransferSplit.source_scope must be a non-empty string")
        if not self.target_scope:
            raise ValueError("TransferSplit.target_scope must be a non-empty string")


@dataclass(frozen=True)
class TransferResult:
    """A single transfer-evaluation record for one model and metric."""

    train_scope: str
    eval_scope: str
    metric_name: str
    metric_value: float | None
    model_name: str = ""
    is_synthetic: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.train_scope:
            raise ValueError("TransferResult.train_scope must be a non-empty string")
        if not self.eval_scope:
            raise ValueError("TransferResult.eval_scope must be a non-empty string")
        if not self.metric_name:
            raise ValueError("TransferResult.metric_name must be a non-empty string")
        if self.metric_value is not None:
            if isinstance(self.metric_value, bool) or not isinstance(
                self.metric_value, (int, float)
            ):
                raise TypeError(
                    "TransferResult.metric_value must be a float, int or None"
                )
            if not math.isfinite(float(self.metric_value)):
                raise ValueError("TransferResult.metric_value must be finite or None")


@dataclass(frozen=True)
class TransferMatrix:
    """A transfer matrix for a single metric across train/eval scopes."""

    metric_name: str
    train_scopes: tuple[str, ...]
    eval_scopes: tuple[str, ...]
    values: tuple[tuple[float | None, ...], ...]
    is_synthetic: bool

    def shape(self) -> tuple[int, int]:
        return (len(self.train_scopes), len(self.eval_scopes))

    def get(self, train_scope: str, eval_scope: str) -> float | None:
        try:
            row_index = self.train_scopes.index(train_scope)
            column_index = self.eval_scopes.index(eval_scope)
        except ValueError as exc:
            raise KeyError(
                f"transfer matrix does not contain ({train_scope!r}, {eval_scope!r})"
            ) from exc
        return self.values[row_index][column_index]


def _coerce_results(
    records: Iterable[TransferResult | Mapping[str, Any]],
) -> list[TransferResult]:
    coerced: list[TransferResult] = []
    for record in records:
        if isinstance(record, TransferResult):
            coerced.append(record)
            continue
        if not isinstance(record, Mapping):
            raise TypeError(
                "each transfer record must be a TransferResult or Mapping"
            )
        coerced.append(
            TransferResult(
                train_scope=str(record.get("train_scope", "")),
                eval_scope=str(record.get("eval_scope", "")),
                metric_name=str(record.get("metric_name", "")),
                metric_value=(
                    None
                    if record.get("metric_value") is None
                    else float(record["metric_value"])
                ),
                model_name=str(record.get("model_name", "")),
                is_synthetic=bool(record.get("is_synthetic", False)),
                notes=str(record.get("notes", "")),
            )
        )
    return coerced


def build_transfer_matrix(
    records: Iterable[TransferResult | Mapping[str, Any]],
    *,
    metric_name: str,
) -> TransferMatrix:
    """Build a transfer matrix for ``metric_name`` from records.

    Missing (train_scope, eval_scope) cells are filled with ``None`` rather
    than fabricated values. Rows and columns are ordered by first appearance
    to keep the output deterministic.
    """
    if not metric_name:
        raise ValueError("metric_name must be a non-empty string")
    coerced = _coerce_results(records)
    relevant = [result for result in coerced if result.metric_name == metric_name]

    train_scopes: list[str] = []
    eval_scopes: list[str] = []
    for result in relevant:
        if result.train_scope not in train_scopes:
            train_scopes.append(result.train_scope)
        if result.eval_scope not in eval_scopes:
            eval_scopes.append(result.eval_scope)
    train_scopes.sort()
    eval_scopes.sort()

    cell_index: dict[tuple[str, str], TransferResult] = {}
    for result in relevant:
        key = (result.train_scope, result.eval_scope)
        if key in cell_index:
            raise ValueError(
                "duplicate transfer record for "
                f"train={result.train_scope!r}, eval={result.eval_scope!r}, "
                f"metric={metric_name!r}"
            )
        cell_index[key] = result

    rows: list[tuple[float | None, ...]] = []
    synthetic_flags: list[bool] = []
    for train_scope in train_scopes:
        row: list[float | None] = []
        for eval_scope in eval_scopes:
            cell_result = cell_index.get((train_scope, eval_scope))
            if cell_result is None:
                row.append(None)
            else:
                row.append(cell_result.metric_value)
                synthetic_flags.append(cell_result.is_synthetic)
        rows.append(tuple(row))

    is_synthetic = bool(synthetic_flags) and all(synthetic_flags)
    return TransferMatrix(
        metric_name=metric_name,
        train_scopes=tuple(train_scopes),
        eval_scopes=tuple(eval_scopes),
        values=tuple(rows),
        is_synthetic=is_synthetic,
    )


def summarise_transfer_results(
    records: Iterable[TransferResult | Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Summarise transfer records grouped by metric name.

    For each metric the output contains: count of present cells, count of
    missing cells across the implied grid, distinct train scopes and
    distinct eval scopes, and whether the metric is entirely synthetic.
    """
    coerced = _coerce_results(records)
    grouped: dict[str, list[TransferResult]] = {}
    for result in coerced:
        grouped.setdefault(result.metric_name, []).append(result)

    summary: dict[str, dict[str, Any]] = {}
    for metric_name in sorted(grouped.keys()):
        results = grouped[metric_name]
        train_scopes = sorted({result.train_scope for result in results})
        eval_scopes = sorted({result.eval_scope for result in results})
        cell_count = len(train_scopes) * len(eval_scopes)
        present_cells = sum(1 for result in results if result.metric_value is not None)
        missing_cells = cell_count - present_cells
        is_synthetic = (
            bool(results) and all(result.is_synthetic for result in results)
        )
        summary[metric_name] = {
            "metric_name": metric_name,
            "train_scopes": train_scopes,
            "eval_scopes": eval_scopes,
            "n_records": len(results),
            "n_present_cells": present_cells,
            "n_missing_cells": max(0, missing_cells),
            "is_synthetic": is_synthetic,
        }
    return summary


def compare_in_domain_vs_out_of_domain(
    records: Iterable[TransferResult | Mapping[str, Any]],
    *,
    metric_name: str,
) -> dict[str, Any]:
    """Compare in-domain (train_scope == eval_scope) and out-of-domain cells."""
    if not metric_name:
        raise ValueError("metric_name must be a non-empty string")
    coerced = _coerce_results(records)
    in_values: list[float] = []
    out_values: list[float] = []
    synthetic_flags: list[bool] = []
    for result in coerced:
        if result.metric_name != metric_name:
            continue
        if result.metric_value is None:
            continue
        value = float(result.metric_value)
        synthetic_flags.append(result.is_synthetic)
        if result.train_scope == result.eval_scope:
            in_values.append(value)
        else:
            out_values.append(value)

    def _agg(values: Sequence[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "mean": None, "minimum": None, "maximum": None}
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "minimum": min(values),
            "maximum": max(values),
        }

    in_summary = _agg(in_values)
    out_summary = _agg(out_values)
    if in_summary["mean"] is None or out_summary["mean"] is None:
        absolute_gap: float | None = None
    else:
        absolute_gap = float(in_summary["mean"]) - float(out_summary["mean"])
    is_synthetic = bool(synthetic_flags) and all(synthetic_flags)
    return {
        "metric_name": metric_name,
        "in_domain": in_summary,
        "out_of_domain": out_summary,
        "absolute_gap_in_minus_out": absolute_gap,
        "is_synthetic": is_synthetic,
    }
