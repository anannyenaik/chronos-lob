"""Unified analysis summaries for predictive, calibration and execution metrics.

The summary layer keeps predictive, calibration and execution metrics
clearly separated. It does not combine them into a single magic score
because predictive and execution metrics are not interchangeable.

The synthetic smoke runner constructs deterministic synthetic records and
exercises the analysis layer end to end. It does not measure any real
market signal and emits only plumbing diagnostics.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from chronoslob.analysis.ablations import (
    AblationResult,
    compare_against_baseline,
    summarise_ablation_table,
)
from chronoslob.analysis.regimes import (
    SUPPORTED_REGIME_KINDS,
    summarise_by_regime,
)
from chronoslob.analysis.sensitivity import (
    SensitivityPoint,
    build_sensitivity_curve,
    summarise_sensitivity_curve,
)
from chronoslob.analysis.transfer import (
    TransferResult,
    build_transfer_matrix,
)

__all__ = [
    "ANALYSIS_TYPES",
    "EXECUTION_METRIC_NAMES",
    "FORBIDDEN_COMBINED_FIELDS",
    "METRIC_DIRECTIONS",
    "PREDICTIVE_METRIC_NAMES",
    "SUPPORTED_METRIC_NAMES",
    "SYNTHETIC_ANALYSIS_WARNING",
    "AnalysisMetric",
    "AnalysisRecord",
    "AnalysisSummary",
    "MetricDirection",
    "aggregate_metric",
    "aggregate_records",
    "format_summary_table",
    "run_robustness_analysis_smoke",
    "summarise_records",
]


MetricDirection = Literal["higher_is_better", "lower_is_better"]


METRIC_DIRECTIONS: Mapping[str, MetricDirection] = {
    "accuracy": "higher_is_better",
    "macro_f1": "higher_is_better",
    "mcc": "higher_is_better",
    "nll": "lower_is_better",
    "brier_score": "lower_is_better",
    "ece": "lower_is_better",
    "coverage": "higher_is_better",
    "fill_rate": "higher_is_better",
    "simulated_net_pnl": "higher_is_better",
    "total_cost": "lower_is_better",
    "turnover": "higher_is_better",
    "adverse_selection_rate": "lower_is_better",
    "max_drawdown": "lower_is_better",
    "latency_steps": "lower_is_better",
}

PREDICTIVE_METRIC_NAMES: tuple[str, ...] = (
    "accuracy",
    "macro_f1",
    "mcc",
    "nll",
    "brier_score",
    "ece",
)

EXECUTION_METRIC_NAMES: tuple[str, ...] = (
    "coverage",
    "fill_rate",
    "simulated_net_pnl",
    "total_cost",
    "turnover",
    "adverse_selection_rate",
    "max_drawdown",
    "latency_steps",
)

SUPPORTED_METRIC_NAMES: tuple[str, ...] = tuple(METRIC_DIRECTIONS.keys())

ANALYSIS_TYPES: tuple[str, ...] = (
    "regime",
    "transfer",
    "ablation",
    "sensitivity",
    "summary",
)

SYNTHETIC_ANALYSIS_WARNING = (
    "Synthetic analysis plumbing only; records are not market evidence, alpha "
    "evidence, tradability evidence or live performance."
)

FORBIDDEN_COMBINED_FIELDS: tuple[str, ...] = (
    "combined_score",
    "magic_score",
    "alpha_score",
    "tradability_score",
    "sharpe",
)


@dataclass(frozen=True)
class AnalysisMetric:
    """Metadata describing a single supported metric name."""

    name: str
    direction: MetricDirection
    family: Literal["predictive", "execution"]

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_METRIC_NAMES:
            raise ValueError(
                f"unsupported metric name {self.name!r}; "
                f"supported: {SUPPORTED_METRIC_NAMES}"
            )
        expected = METRIC_DIRECTIONS[self.name]
        if self.direction != expected:
            raise ValueError(
                f"metric {self.name!r} expects direction {expected!r}; "
                f"got {self.direction!r}"
            )
        if self.family == "predictive" and self.name not in PREDICTIVE_METRIC_NAMES:
            raise ValueError(
                f"metric {self.name!r} is not predictive"
            )
        if self.family == "execution" and self.name not in EXECUTION_METRIC_NAMES:
            raise ValueError(
                f"metric {self.name!r} is not execution-aware"
            )


@dataclass(frozen=True)
class AnalysisRecord:
    """A structured analysis record consumed by the summary layer."""

    experiment_id: str
    model_name: str
    dataset_name: str
    symbol: str
    train_scope: str
    eval_scope: str
    regime: str
    ablation: str
    sensitivity_parameter: str
    sensitivity_value: float | None
    metric_name: str
    metric_value: float
    metric_direction: MetricDirection
    is_synthetic: bool
    notes: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "experiment_id",
            "model_name",
            "dataset_name",
            "symbol",
            "train_scope",
            "eval_scope",
            "regime",
            "ablation",
            "sensitivity_parameter",
            "metric_name",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
        if self.metric_name not in SUPPORTED_METRIC_NAMES:
            raise ValueError(
                f"unsupported metric_name {self.metric_name!r}; "
                f"supported: {SUPPORTED_METRIC_NAMES}"
            )
        if self.metric_name in FORBIDDEN_COMBINED_FIELDS:
            raise ValueError(
                "combined or magic scores are not supported in this layer"
            )
        if isinstance(self.metric_value, bool) or not isinstance(
            self.metric_value, (int, float)
        ):
            raise TypeError("metric_value must be a real number")
        if not math.isfinite(float(self.metric_value)):
            raise ValueError("metric_value must be finite")
        expected_direction = METRIC_DIRECTIONS[self.metric_name]
        if self.metric_direction != expected_direction:
            raise ValueError(
                f"metric {self.metric_name!r} requires direction "
                f"{expected_direction!r}; got {self.metric_direction!r}"
            )
        if self.sensitivity_value is not None:
            if isinstance(self.sensitivity_value, bool) or not isinstance(
                self.sensitivity_value, (int, float)
            ):
                raise TypeError("sensitivity_value must be a real number or None")
            if not math.isfinite(float(self.sensitivity_value)):
                raise ValueError("sensitivity_value must be finite or None")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "symbol": self.symbol,
            "train_scope": self.train_scope,
            "eval_scope": self.eval_scope,
            "regime": self.regime,
            "ablation": self.ablation,
            "sensitivity_parameter": self.sensitivity_parameter,
            "sensitivity_value": self.sensitivity_value,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "metric_direction": self.metric_direction,
            "is_synthetic": self.is_synthetic,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class AnalysisSummary:
    """A grouped summary of analysis records for one metric."""

    metric_name: str
    metric_direction: MetricDirection
    group_key: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    is_synthetic: bool


def _validate_metric_name(metric_name: str) -> None:
    if metric_name not in SUPPORTED_METRIC_NAMES:
        raise ValueError(
            f"unsupported metric_name {metric_name!r}; "
            f"supported: {SUPPORTED_METRIC_NAMES}"
        )
    if metric_name in FORBIDDEN_COMBINED_FIELDS:
        raise ValueError(
            "combined scores are not supported by this analysis layer"
        )


def aggregate_metric(values: Iterable[float]) -> dict[str, Any]:
    """Aggregate finite metric values into count/mean/min/max."""
    finite_values: list[float] = []
    for value in values:
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        finite_values.append(numeric)
    if not finite_values:
        return {"count": 0, "mean": None, "minimum": None, "maximum": None}
    return {
        "count": len(finite_values),
        "mean": sum(finite_values) / len(finite_values),
        "minimum": min(finite_values),
        "maximum": max(finite_values),
    }


def aggregate_records(
    records: Iterable[AnalysisRecord | Mapping[str, Any]],
    *,
    metric_name: str,
    group_by: Sequence[str],
) -> AnalysisSummary:
    """Group records by ``group_by`` and aggregate ``metric_name``."""
    _validate_metric_name(metric_name)
    coerced = _coerce_records(records)
    grouped: dict[tuple[str, ...], list[AnalysisRecord]] = {}
    for record in coerced:
        if record.metric_name != metric_name:
            continue
        key = tuple(getattr(record, field_name) for field_name in group_by)
        grouped.setdefault(key, []).append(record)

    rows: list[dict[str, Any]] = []
    synthetic_flags: list[bool] = []
    for key in sorted(grouped.keys()):
        bucket = grouped[key]
        synthetic_flags.append(all(record.is_synthetic for record in bucket))
        aggregated = aggregate_metric(record.metric_value for record in bucket)
        row: dict[str, Any] = dict(zip(group_by, key, strict=False))
        row.update(aggregated)
        row["is_synthetic"] = all(record.is_synthetic for record in bucket)
        rows.append(row)

    is_synthetic = bool(synthetic_flags) and all(synthetic_flags)
    return AnalysisSummary(
        metric_name=metric_name,
        metric_direction=METRIC_DIRECTIONS[metric_name],
        group_key=tuple(group_by),
        rows=tuple(rows),
        is_synthetic=is_synthetic,
    )


def summarise_records(
    records: Iterable[AnalysisRecord | Mapping[str, Any]],
    *,
    group_by: Sequence[str] = ("model_name",),
) -> list[AnalysisSummary]:
    """Summarise records grouped by ``group_by`` for every metric present."""
    coerced = _coerce_records(records)
    metric_names = sorted({record.metric_name for record in coerced})
    return [
        aggregate_records(coerced, metric_name=metric_name, group_by=group_by)
        for metric_name in metric_names
    ]


def format_summary_table(summary: AnalysisSummary) -> str:
    """Format an ``AnalysisSummary`` as a deterministic plain-text table."""
    headers = [
        *summary.group_key,
        "count",
        "mean",
        "minimum",
        "maximum",
        "is_synthetic",
    ]
    table_rows: list[list[str]] = [headers]
    for row in summary.rows:
        formatted: list[str] = []
        for column in headers:
            value = row.get(column)
            if value is None:
                formatted.append("n/a")
            elif isinstance(value, bool):
                formatted.append("True" if value else "False")
            elif isinstance(value, float):
                formatted.append(f"{value:.6f}")
            else:
                formatted.append(str(value))
        table_rows.append(formatted)

    widths = [
        max(len(table_row[index]) for table_row in table_rows)
        for index in range(len(headers))
    ]
    lines = [
        f"# metric={summary.metric_name} direction={summary.metric_direction}",
    ]
    for table_row in table_rows:
        line = "  ".join(
            cell.ljust(widths[index]) for index, cell in enumerate(table_row)
        )
        lines.append(line)
    lines.append(f"# is_synthetic={summary.is_synthetic}")
    return "\n".join(lines)


def _coerce_records(
    records: Iterable[AnalysisRecord | Mapping[str, Any]],
) -> list[AnalysisRecord]:
    coerced: list[AnalysisRecord] = []
    for record in records:
        if isinstance(record, AnalysisRecord):
            coerced.append(record)
            continue
        if not isinstance(record, Mapping):
            raise TypeError(
                "each analysis record must be an AnalysisRecord or Mapping"
            )
        for forbidden in FORBIDDEN_COMBINED_FIELDS:
            if forbidden in record:
                raise ValueError(
                    f"record contains forbidden combined-score field {forbidden!r}"
                )
        coerced.append(
            AnalysisRecord(
                experiment_id=str(record.get("experiment_id", "")),
                model_name=str(record.get("model_name", "")),
                dataset_name=str(record.get("dataset_name", "")),
                symbol=str(record.get("symbol", "")),
                train_scope=str(record.get("train_scope", "")),
                eval_scope=str(record.get("eval_scope", "")),
                regime=str(record.get("regime", "")),
                ablation=str(record.get("ablation", "")),
                sensitivity_parameter=str(record.get("sensitivity_parameter", "")),
                sensitivity_value=(
                    None
                    if record.get("sensitivity_value") is None
                    else float(record["sensitivity_value"])
                ),
                metric_name=str(record["metric_name"]),
                metric_value=float(record["metric_value"]),
                metric_direction=str(
                    record.get(
                        "metric_direction",
                        METRIC_DIRECTIONS.get(
                            str(record.get("metric_name", "")), "higher_is_better"
                        ),
                    )
                ),  # type: ignore[arg-type]
                is_synthetic=bool(record.get("is_synthetic", False)),
                notes=str(record.get("notes", "")),
            )
        )
    return coerced


def _direction_for(metric_name: str) -> MetricDirection:
    return METRIC_DIRECTIONS[metric_name]


@dataclass(frozen=True)
class _SmokeConfig:
    n_records: int
    seed: int
    symbols: tuple[str, ...] = ("SYMBOL_A", "SYMBOL_B")
    ablation_names: tuple[str, ...] = (
        "baseline",
        "no_order_flow_features",
        "no_depth_imbalance",
        "no_ssl_pretraining",
        "no_calibration",
        "no_confidence_filtering",
        "no_latency",
        "aggressive_only",
        "passive_only",
    )
    regime_labels: tuple[str, ...] = ("low", "medium", "high")
    confidence_thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8)
    latency_steps: tuple[int, ...] = (0, 1, 2, 5)


def _build_synthetic_records(
    config: _SmokeConfig,
) -> list[AnalysisRecord]:
    rng = random.Random(config.seed)
    records: list[AnalysisRecord] = []

    metric_pool: list[tuple[str, tuple[float, float]]] = [
        ("accuracy", (0.30, 0.60)),
        ("macro_f1", (0.20, 0.55)),
        ("mcc", (-0.05, 0.30)),
        ("nll", (0.50, 1.40)),
        ("brier_score", (0.10, 0.55)),
        ("ece", (0.02, 0.20)),
        ("coverage", (0.20, 0.95)),
        ("fill_rate", (0.30, 0.95)),
        ("simulated_net_pnl", (-0.50, 0.80)),
        ("total_cost", (0.05, 0.40)),
        ("turnover", (0.20, 1.50)),
        ("adverse_selection_rate", (0.05, 0.40)),
        ("max_drawdown", (0.05, 0.45)),
    ]

    index = 0
    while len(records) < config.n_records:
        symbol = config.symbols[index % len(config.symbols)]
        ablation = config.ablation_names[index % len(config.ablation_names)]
        regime_label = config.regime_labels[index % len(config.regime_labels)]
        metric_name, (lower, upper) = metric_pool[index % len(metric_pool)]
        train_scope = symbol
        eval_scope = config.symbols[(index + 1) % len(config.symbols)]
        value = lower + rng.random() * (upper - lower)
        records.append(
            AnalysisRecord(
                experiment_id=f"synthetic-record-{index:04d}",
                model_name="synthetic-model",
                dataset_name="synthetic-dataset",
                symbol=symbol,
                train_scope=train_scope,
                eval_scope=eval_scope,
                regime=regime_label,
                ablation=ablation,
                sensitivity_parameter="",
                sensitivity_value=None,
                metric_name=metric_name,
                metric_value=value,
                metric_direction=METRIC_DIRECTIONS[metric_name],
                is_synthetic=True,
                notes="synthetic smoke record",
            )
        )
        index += 1
    return records


def _build_synthetic_sensitivity_points(
    config: _SmokeConfig,
) -> list[SensitivityPoint]:
    rng = random.Random(config.seed + 1)
    points: list[SensitivityPoint] = []
    for threshold in config.confidence_thresholds:
        points.append(
            SensitivityPoint(
                parameter_name="confidence_threshold",
                parameter_value=float(threshold),
                metric_name="accuracy",
                metric_value=0.45 + 0.1 * threshold + rng.random() * 0.05,
                is_synthetic=True,
                notes="synthetic smoke sensitivity",
            )
        )
        points.append(
            SensitivityPoint(
                parameter_name="confidence_threshold",
                parameter_value=float(threshold),
                metric_name="coverage",
                metric_value=max(0.0, 1.0 - threshold + rng.random() * 0.05),
                is_synthetic=True,
                notes="synthetic smoke sensitivity",
            )
        )
    for steps in config.latency_steps:
        points.append(
            SensitivityPoint(
                parameter_name="latency_steps",
                parameter_value=float(steps),
                metric_name="simulated_net_pnl",
                metric_value=0.4 - 0.05 * steps + rng.random() * 0.05,
                is_synthetic=True,
                notes="synthetic smoke sensitivity",
            )
        )
    return points


def _build_synthetic_transfer_records(
    config: _SmokeConfig,
) -> list[TransferResult]:
    rng = random.Random(config.seed + 2)
    transfer_records: list[TransferResult] = []
    for train_symbol in config.symbols:
        for eval_symbol in config.symbols:
            transfer_records.append(
                TransferResult(
                    train_scope=train_symbol,
                    eval_scope=eval_symbol,
                    metric_name="accuracy",
                    metric_value=0.4 + rng.random() * 0.2,
                    model_name="synthetic-model",
                    is_synthetic=True,
                    notes="synthetic smoke transfer",
                )
            )
    return transfer_records


def _build_synthetic_ablation_records(
    config: _SmokeConfig,
) -> list[AblationResult]:
    rng = random.Random(config.seed + 3)
    ablation_records: list[AblationResult] = []
    for name in config.ablation_names:
        base = 0.5 if name == "baseline" else 0.5 - rng.random() * 0.15
        ablation_records.append(
            AblationResult(
                ablation_name=name,
                metric_name="accuracy",
                metric_value=base,
                metric_direction="higher_is_better",
                is_synthetic=True,
                notes="synthetic smoke ablation",
            )
        )
    return ablation_records


def run_robustness_analysis_smoke(
    *,
    n_records: int = 36,
    seed: int = 42,
) -> dict[str, Any]:
    """Run the deterministic synthetic robustness-analysis smoke pipeline.

    The function produces synthetic records only. Outputs are not market
    evidence, alpha evidence, tradability evidence or live performance.
    """
    if not isinstance(n_records, int) or isinstance(n_records, bool):
        raise TypeError("n_records must be an integer")
    if n_records <= 0:
        raise ValueError("n_records must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")

    config = _SmokeConfig(n_records=n_records, seed=seed)
    analysis_records = _build_synthetic_records(config)
    transfer_records = _build_synthetic_transfer_records(config)
    sensitivity_points = _build_synthetic_sensitivity_points(config)
    ablation_records = _build_synthetic_ablation_records(config)

    # Regime summaries for each supported kind, using the regime field.
    regime_summaries: dict[str, list[dict[str, Any]]] = {}
    for kind in SUPPORTED_REGIME_KINDS:
        assigner: Any
        if kind == "volatility":
            assigner = _explicit_assigner_factory("volatility", config.regime_labels)
        elif kind == "spread":
            assigner = _explicit_assigner_factory("spread", config.regime_labels)
        elif kind == "liquidity":
            assigner = _explicit_assigner_factory("liquidity", config.regime_labels)
        elif kind == "confidence":
            assigner = _explicit_assigner_factory(
                "confidence", config.regime_labels
            )
        else:
            assigner = _explicit_assigner_factory("latency", config.regime_labels)
        summaries = summarise_by_regime(
            [record.to_mapping() for record in analysis_records],
            kind=kind,
            metric_name="accuracy",
            assigner=assigner,
        )
        regime_summaries[kind] = [
            {
                "kind": summary.kind,
                "label": summary.label,
                "count": summary.count,
                "mean": summary.mean,
                "minimum": summary.minimum,
                "maximum": summary.maximum,
                "is_synthetic": summary.is_synthetic,
            }
            for summary in summaries
        ]

    transfer_matrix = build_transfer_matrix(
        transfer_records, metric_name="accuracy"
    )

    ablation_comparisons = compare_against_baseline(
        ablation_records, baseline_name="baseline", metric_name="accuracy"
    )
    ablation_summary = summarise_ablation_table(ablation_comparisons)

    confidence_curve = build_sensitivity_curve(
        sensitivity_points,
        parameter_name="confidence_threshold",
        metric_name="accuracy",
        metric_direction="higher_is_better",
    )
    coverage_curve = build_sensitivity_curve(
        sensitivity_points,
        parameter_name="confidence_threshold",
        metric_name="coverage",
        metric_direction="higher_is_better",
    )
    latency_curve = build_sensitivity_curve(
        sensitivity_points,
        parameter_name="latency_steps",
        metric_name="simulated_net_pnl",
        metric_direction="higher_is_better",
    )
    sensitivity_curve_summaries = {
        "confidence_accuracy": summarise_sensitivity_curve(confidence_curve),
        "confidence_coverage": summarise_sensitivity_curve(coverage_curve),
        "latency_simulated_net_pnl": summarise_sensitivity_curve(latency_curve),
    }

    grouped_summaries = summarise_records(
        analysis_records, group_by=("model_name", "symbol")
    )
    example_summary_rows: list[dict[str, Any]] = []
    for summary in grouped_summaries[:3]:
        for row in summary.rows[:2]:
            example_summary_rows.append(
                {
                    "metric_name": summary.metric_name,
                    "metric_direction": summary.metric_direction,
                    "row": row,
                }
            )

    return {
        "warning": SYNTHETIC_ANALYSIS_WARNING,
        "is_synthetic": True,
        "n_records": len(analysis_records),
        "n_transfer_records": len(transfer_records),
        "n_sensitivity_points": len(sensitivity_points),
        "n_ablation_records": len(ablation_records),
        "regime_summary_counts": {
            kind: len(summary_list) for kind, summary_list in regime_summaries.items()
        },
        "regime_summaries": regime_summaries,
        "transfer_matrix": {
            "metric_name": transfer_matrix.metric_name,
            "train_scopes": list(transfer_matrix.train_scopes),
            "eval_scopes": list(transfer_matrix.eval_scopes),
            "shape": list(transfer_matrix.shape()),
            "values": [list(row) for row in transfer_matrix.values],
            "is_synthetic": transfer_matrix.is_synthetic,
        },
        "ablation_comparisons_count": len(ablation_comparisons),
        "ablation_summary": ablation_summary,
        "sensitivity_curves_produced": len(sensitivity_curve_summaries),
        "sensitivity_curve_summaries": sensitivity_curve_summaries,
        "example_summary_rows": example_summary_rows,
        "supported_metric_names": list(SUPPORTED_METRIC_NAMES),
        "predictive_metric_names": list(PREDICTIVE_METRIC_NAMES),
        "execution_metric_names": list(EXECUTION_METRIC_NAMES),
    }


def _explicit_assigner_factory(kind: str, allowed_labels: Sequence[str]) -> Any:
    """Create an assigner that maps the record's ``regime`` field for ``kind``."""
    from chronoslob.analysis.regimes import (
        UNKNOWN_REGIME_LABEL,
        RegimeAssignment,
    )

    allowed = tuple(allowed_labels)

    def _assigner(record: Mapping[str, Any]) -> RegimeAssignment:
        raw = record.get("regime")
        if isinstance(raw, str) and raw in allowed:
            return RegimeAssignment(
                kind=kind, label=raw, value=None, source="explicit"
            )
        return RegimeAssignment(
            kind=kind, label=UNKNOWN_REGIME_LABEL, value=None, source="default"
        )

    return _assigner


