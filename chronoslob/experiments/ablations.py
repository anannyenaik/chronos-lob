"""Phase H paper-experiment ablation suite.

This module orchestrates controlled paper-experiment runs that vary a
single dimension at a time (calibration bin count, cost level, latency,
horizon, lookback or feature group) while keeping everything else fixed.
It composes the paper experiment runner rather than duplicating training
logic and writes a small aggregate summary plus concise markdown reports.

The runner is local-only, deterministic and never downloads data, makes
a network call or invents ablation conclusions. SSL-pretraining
ablations are recorded as skipped because the paper runner does not yet
support a traceable SSL pretraining and fine-tuning path.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator

from chronoslob.experiments.fi2010_benchmark import (
    FI2010BenchmarkConfig,
    load_benchmark_config,
)
from chronoslob.experiments.manifests import stable_json_dumps
from chronoslob.experiments.model_registry import (
    get_paper_model_spec,
    normalise_paper_model_names,
)
from chronoslob.experiments.paper_runner import (
    PaperExperimentSummary,
    run_paper_experiment,
)

__all__ = [
    "ABLATION_RESULTS_COLUMNS",
    "FEATURE_GROUP_PATTERNS",
    "PAPER_ABLATION_VERSION",
    "SUPPORTED_ABLATION_SETS",
    "PaperAblationResult",
    "PaperAblationSpec",
    "PaperAblationSummary",
    "build_ablation_specs",
    "run_paper_ablations",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)

PAPER_ABLATION_VERSION = "phase-h/paper-ablation-runner/v1"

SUPPORTED_ABLATION_SETS: tuple[str, ...] = ("smoke", "standard")

FEATURE_GROUP_PATTERNS: dict[str, tuple[str, ...]] = {
    "all": (),
    "top_of_book": (
        "bid_price_1",
        "ask_price_1",
        "bid_quantity_1",
        "ask_quantity_1",
    ),
    "imbalance": ("*imbalance*", "*microprice*"),
    "depth_liquidity": ("bid_quantity_*", "ask_quantity_*"),
}

ABLATION_RESULTS_COLUMNS: tuple[str, ...] = (
    "ablation_name",
    "ablation_type",
    "status",
    "model_name",
    "split",
    "horizon",
    "feature_group",
    "lookback",
    "calibration_bins",
    "cost_bps",
    "latency_steps",
    "metric_name",
    "metric_value",
    "source_experiment",
    "warning",
)

_FIXTURE_PATH_MARKERS = ("tests", "fixtures")
_SSL_ABLATION_NAME = "ssl_pretraining_ablation"
_SSL_SKIP_REASON = (
    "no traceable runner support for SSL pretraining/fine-tuning yet; "
    "ssl_transformer is not registered in the paper-runner model registry"
)


# ---------------------------------------------------------------------------
# Typed schemas
# ---------------------------------------------------------------------------


class PaperAblationSpec(BaseModel):
    """Definition of a single controlled ablation."""

    model_config = _MODEL_CONFIG

    name: str
    ablation_type: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    requires_neural_model: bool = False
    always_skip: bool = False
    skip_reason: str | None = None

    @field_validator("name", "ablation_type", "description")
    @classmethod
    def _validate_required_strings(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("ablation spec strings must be non-empty")
        return value.strip()

    @field_validator("skip_reason")
    @classmethod
    def _validate_skip_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("skip_reason must be a non-empty string when provided")
        return value.strip()


class PaperAblationResult(BaseModel):
    """Outcome record for a single ablation."""

    model_config = _MODEL_CONFIG

    name: str
    ablation_type: str
    status: str
    reason: str | None = None
    experiment_dir: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("status")
    @classmethod
    def _validate_status(cls, value: str) -> str:
        if value not in ("run", "skipped"):
            raise ValueError("status must be 'run' or 'skipped'")
        return value


class PaperAblationSummary(BaseModel):
    """Top-level summary returned by :func:`run_paper_ablations`."""

    model_config = _MODEL_CONFIG

    runner_version: str
    created_at: datetime
    base_config: str
    data_path: str
    data_source_kind: str
    output_dir: str
    ablation_set: str
    models_requested: list[str]
    ablations_requested: list[str]
    ablations_run: list[str]
    ablations_skipped: list[str]
    child_experiments: dict[str, str] = Field(default_factory=dict)
    reports_written: list[str] = Field(default_factory=list)
    is_fixture: bool = False
    warnings: list[str] = Field(default_factory=list)
    results: list[PaperAblationResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_fixture_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return all(marker in parts for marker in _FIXTURE_PATH_MARKERS)


def _data_source_kind(path: Path) -> str:
    if _is_fixture_path(path.resolve()):
        return "synthetic_fixture"
    return "local_file"


def _has_neural_model(models: Sequence[str]) -> bool:
    for name in models:
        spec = get_paper_model_spec(name)
        if spec.model_family == "neural":
            return True
    return False


def _load_yaml_payload(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark config must be a YAML mapping: {path}")
    return dict(payload)


def _write_yaml_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            dict(payload),
            sort_keys=True,
            allow_unicode=False,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )


def _apply_parameter_overrides(
    payload: dict[str, Any],
    parameters: Mapping[str, Any],
) -> None:
    """Apply dot-notation overrides to ``payload`` in-place."""
    for raw_key, value in parameters.items():
        keys = str(raw_key).split(".")
        cursor: Any = payload
        for key in keys[:-1]:
            if not isinstance(cursor, dict):
                raise ValueError(
                    f"ablation parameter path {raw_key!r} cannot be applied: "
                    "intermediate node is not a mapping"
                )
            existing = cursor.get(key)
            if not isinstance(existing, dict):
                cursor[key] = {}
            cursor = cursor[key]
        if not isinstance(cursor, dict):
            raise ValueError(
                f"ablation parameter path {raw_key!r} cannot be applied: "
                "leaf parent is not a mapping"
            )
        cursor[keys[-1]] = value


# ---------------------------------------------------------------------------
# Ablation set definitions
# ---------------------------------------------------------------------------


def _smoke_specs() -> list[PaperAblationSpec]:
    return [
        PaperAblationSpec(
            name="baseline",
            ablation_type="baseline",
            description=(
                "Reference run with no parameter changes; serves as the "
                "comparison anchor for the other ablations."
            ),
            parameters={},
        ),
        PaperAblationSpec(
            name="calibration_bins_5",
            ablation_type="calibration_bins",
            description=(
                "Reliability bins computed with 5 bins instead of the "
                "configured default."
            ),
            parameters={"calibration.n_bins": 5},
        ),
        PaperAblationSpec(
            name="cost_0bps",
            ablation_type="cost_bps",
            description=(
                "Execution-aware sensitivity reported with a single "
                "0 bps cost level so the net signal proxy equals the "
                "gross proxy."
            ),
            parameters={"execution_sensitivity.cost_bps": [0.0]},
        ),
        PaperAblationSpec(
            name="cost_1bps",
            ablation_type="cost_bps",
            description=(
                "Execution-aware sensitivity reported with a single "
                "1 bps cost level under explicit cost assumptions."
            ),
            parameters={"execution_sensitivity.cost_bps": [1.0]},
        ),
        PaperAblationSpec(
            name=_SSL_ABLATION_NAME,
            ablation_type="ssl_pretraining",
            description=(
                "SSL pretraining + fine-tuning ablation status report; "
                "always skipped because the paper runner does not yet "
                "support a traceable SSL training path."
            ),
            parameters={},
            always_skip=True,
            skip_reason=_SSL_SKIP_REASON,
        ),
    ]


def _standard_specs() -> list[PaperAblationSpec]:
    specs = _smoke_specs()
    additional: list[PaperAblationSpec] = [
        PaperAblationSpec(
            name="calibration_bins_10",
            ablation_type="calibration_bins",
            description=(
                "Reliability bins computed with 10 bins so the bin "
                "resolution can be compared against coarser settings."
            ),
            parameters={"calibration.n_bins": 10},
        ),
        PaperAblationSpec(
            name="latency_0",
            ablation_type="latency_steps",
            description=(
                "Execution-aware sensitivity with a single 0-row "
                "latency assumption."
            ),
            parameters={"execution_sensitivity.latency_steps": [0]},
        ),
        PaperAblationSpec(
            name="latency_1",
            ablation_type="latency_steps",
            description=(
                "Execution-aware sensitivity with a 1-row latency "
                "assumption: realised proxy returns are pessimistically "
                "shifted by one row."
            ),
            parameters={"execution_sensitivity.latency_steps": [1]},
        ),
        PaperAblationSpec(
            name="horizon_50",
            ablation_type="horizon",
            description=(
                "Horizon override to 50 with label column label_50; "
                "labels are regenerated by the runner from the supplied "
                "frame and the temporal split is preserved."
            ),
            parameters={"horizon": 50, "label_name": "label_50"},
        ),
        PaperAblationSpec(
            name="lookback_2",
            ablation_type="lookback",
            description=(
                "Neural lookback window of 2 rows; applies only to "
                "neural paper-runner models that consume windows."
            ),
            parameters={"neural_settings.lookback": 2},
            requires_neural_model=True,
        ),
        PaperAblationSpec(
            name="lookback_4",
            ablation_type="lookback",
            description=(
                "Neural lookback window of 4 rows; applies only to "
                "neural paper-runner models that consume windows."
            ),
            parameters={"neural_settings.lookback": 4},
            requires_neural_model=True,
        ),
        PaperAblationSpec(
            name="feature_top_of_book",
            ablation_type="feature_group",
            description=(
                "Feature subset restricted to top-of-book price and "
                "quantity columns through deterministic column-name "
                "patterns; preprocessing remains fit on train rows only."
            ),
            parameters={
                "feature_patterns": list(FEATURE_GROUP_PATTERNS["top_of_book"]),
            },
        ),
        PaperAblationSpec(
            name="feature_imbalance",
            ablation_type="feature_group",
            description=(
                "Feature subset restricted to imbalance and microprice "
                "columns through column-name patterns. Skipped when too "
                "few matching columns exist on the supplied data."
            ),
            parameters={
                "feature_patterns": list(FEATURE_GROUP_PATTERNS["imbalance"]),
            },
        ),
        PaperAblationSpec(
            name="feature_depth_liquidity",
            ablation_type="feature_group",
            description=(
                "Feature subset restricted to depth and liquidity "
                "columns (bid/ask quantity levels) through column-name "
                "patterns."
            ),
            parameters={
                "feature_patterns": list(FEATURE_GROUP_PATTERNS["depth_liquidity"]),
            },
        ),
    ]
    specs.extend(additional)
    return specs


def build_ablation_specs(ablation_set: str) -> list[PaperAblationSpec]:
    """Return the ablation specs for ``ablation_set``.

    Raises ``ValueError`` for unsupported set names.
    """
    cleaned = str(ablation_set).strip().lower()
    if cleaned == "smoke":
        return _smoke_specs()
    if cleaned == "standard":
        return _standard_specs()
    raise ValueError(
        f"unsupported ablation set {ablation_set!r}; supported sets: "
        f"{list(SUPPORTED_ABLATION_SETS)}"
    )


# ---------------------------------------------------------------------------
# Per-spec execution
# ---------------------------------------------------------------------------


def _resolve_runtime_skip(
    spec: PaperAblationSpec,
    *,
    has_neural: bool,
) -> str | None:
    if spec.always_skip:
        return spec.skip_reason or "ablation marked as always skipped"
    if spec.requires_neural_model and not has_neural:
        return (
            "no neural model in the requested model list; "
            "lookback ablations apply only to neural paper-runner models"
        )
    return None


def _is_reportable_child_warning(warning: str, *, build_plots: bool) -> bool:
    return build_plots or not warning.startswith("optional artefact missing: plots/")


def _run_child_experiment(
    *,
    spec: PaperAblationSpec,
    base_payload: Mapping[str, Any],
    data_path: Path,
    child_dir: Path,
    models: Sequence[str],
    build_plots: bool,
) -> PaperExperimentSummary:
    """Run a single ablation as a child paper experiment."""
    payload = deepcopy(dict(base_payload))
    if spec.parameters:
        _apply_parameter_overrides(payload, spec.parameters)

    if child_dir.exists():
        shutil.rmtree(child_dir)
    child_dir.mkdir(parents=True, exist_ok=True)

    ablation_config_path = child_dir / "_ablation_config.yaml"
    _write_yaml_payload(ablation_config_path, payload)

    return run_paper_experiment(
        config_path=ablation_config_path,
        data_path=data_path,
        out_dir=child_dir,
        models=list(models),
        overwrite=True,
        build_plots=build_plots,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _spec_runtime_columns(
    spec: PaperAblationSpec,
    *,
    base_config: FI2010BenchmarkConfig,
) -> dict[str, Any]:
    """Return the canonical runtime descriptors for one spec."""
    params = spec.parameters
    feature_group = "all"
    feature_patterns = params.get("feature_patterns")
    if feature_patterns is not None:
        for name, patterns in FEATURE_GROUP_PATTERNS.items():
            if list(patterns) == list(feature_patterns):
                feature_group = name
                break
        else:
            feature_group = "custom"
    return {
        "horizon": params.get("horizon", base_config.horizon),
        "feature_group": feature_group,
        "lookback": params.get(
            "neural_settings.lookback",
            base_config.neural_settings.lookback,
        ),
        "calibration_bins": params.get(
            "calibration.n_bins",
            base_config.calibration.n_bins,
        ),
        "cost_bps": params.get(
            "execution_sensitivity.cost_bps",
            list(base_config.execution_sensitivity.cost_bps),
        ),
        "latency_steps": params.get(
            "execution_sensitivity.latency_steps",
            list(base_config.execution_sensitivity.latency_steps),
        ),
    }


def _format_listlike(value: Any) -> str:
    if value is None:
        return "not_applicable"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _format_int(value: Any) -> str:
    if value is None:
        return "not_applicable"
    if isinstance(value, bool):
        return "not_applicable"
    if isinstance(value, (int, float)):
        return str(int(value))
    return str(value)


def _format_feature_group(value: Any) -> str:
    if value is None:
        return "all"
    return str(value)


def _row_from_metric(
    *,
    spec: PaperAblationSpec,
    runtime: Mapping[str, Any],
    model_name: str,
    horizon: int,
    metric_name: str,
    metric_value: float,
    source_experiment: str,
    warning: str = "none",
) -> dict[str, Any]:
    spec_horizon = runtime.get("horizon")
    effective_horizon = horizon if spec_horizon is None else int(spec_horizon)
    return {
        "ablation_name": spec.name,
        "ablation_type": spec.ablation_type,
        "status": "run",
        "model_name": model_name,
        "split": "test",
        "horizon": int(effective_horizon),
        "feature_group": _format_feature_group(runtime.get("feature_group")),
        "lookback": _format_int(runtime.get("lookback")),
        "calibration_bins": _format_int(runtime.get("calibration_bins")),
        "cost_bps": _format_listlike(runtime.get("cost_bps")),
        "latency_steps": _format_listlike(runtime.get("latency_steps")),
        "metric_name": str(metric_name),
        "metric_value": float(metric_value),
        "source_experiment": source_experiment,
        "warning": warning or "none",
    }


def _row_for_skip(
    *,
    spec: PaperAblationSpec,
    runtime: Mapping[str, Any],
    base_horizon: int,
    reason: str,
) -> dict[str, Any]:
    spec_horizon = runtime.get("horizon")
    effective_horizon = base_horizon if spec_horizon is None else int(spec_horizon)
    return {
        "ablation_name": spec.name,
        "ablation_type": spec.ablation_type,
        "status": "skipped",
        "model_name": "not_applicable",
        "split": "not_applicable",
        "horizon": int(effective_horizon),
        "feature_group": _format_feature_group(runtime.get("feature_group")),
        "lookback": _format_int(runtime.get("lookback")),
        "calibration_bins": _format_int(runtime.get("calibration_bins")),
        "cost_bps": _format_listlike(runtime.get("cost_bps")),
        "latency_steps": _format_listlike(runtime.get("latency_steps")),
        "metric_name": "status",
        "metric_value": 0.0,
        "source_experiment": "not_applicable",
        "warning": reason,
    }


def _write_ablation_results_csv(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    materialised = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialised:
        frame = pd.DataFrame(columns=list(ABLATION_RESULTS_COLUMNS))
    else:
        frame = pd.DataFrame(materialised)
        for column in ABLATION_RESULTS_COLUMNS:
            if column not in frame.columns:
                frame[column] = ""
        frame = frame.loc[:, list(ABLATION_RESULTS_COLUMNS)]
    frame.to_csv(path, index=False)


def _finite_numeric_value(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _rows_from_calibration_bins(
    *,
    spec: PaperAblationSpec,
    runtime: Mapping[str, Any],
    child_dir: Path,
    source_experiment: str,
) -> list[dict[str, Any]]:
    path = child_dir / "calibration_bins.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    required = {"model_name", "split", "bin_index"}
    if frame.empty or not required.issubset(frame.columns):
        return []

    rows: list[dict[str, Any]] = []
    metric_columns = ("count", "mean_confidence", "accuracy", "confidence_gap")
    for _, record in frame.iterrows():
        model_name = str(record["model_name"])
        split = str(record["split"])
        bin_index = int(record["bin_index"])
        for metric_column in metric_columns:
            if metric_column not in frame.columns:
                continue
            metric_value = _finite_numeric_value(record[metric_column])
            if metric_value is None:
                continue
            rows.append(
                _row_from_metric(
                    spec=spec,
                    runtime=runtime,
                    model_name=model_name,
                    horizon=int(runtime["horizon"]),
                    metric_name=f"calibration.bin_{bin_index}.{metric_column}",
                    metric_value=metric_value,
                    source_experiment=source_experiment,
                )
            )
            rows[-1]["split"] = split
    return rows


def _rows_from_execution_sensitivity(
    *,
    spec: PaperAblationSpec,
    runtime: Mapping[str, Any],
    child_dir: Path,
    source_experiment: str,
) -> list[dict[str, Any]]:
    path = child_dir / "execution_sensitivity.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    required = {
        "model_name",
        "split",
        "confidence_threshold",
        "cost_bps",
        "latency_steps",
    }
    if frame.empty or not required.issubset(frame.columns):
        return []

    metric_columns = (
        "eligible_predictions",
        "trade_count_proxy",
        "turnover_proxy",
        "gross_signal_return_proxy",
        "cost_proxy",
        "net_signal_return_proxy",
        "hit_rate_proxy",
    )
    rows: list[dict[str, Any]] = []
    for _, record in frame.iterrows():
        model_name = str(record["model_name"])
        split = str(record["split"])
        threshold = _finite_numeric_value(record["confidence_threshold"])
        cost_bps = _finite_numeric_value(record["cost_bps"])
        latency_steps = _finite_numeric_value(record["latency_steps"])
        if threshold is None or cost_bps is None or latency_steps is None:
            continue
        row_runtime = dict(runtime)
        row_runtime["cost_bps"] = cost_bps
        row_runtime["latency_steps"] = int(latency_steps)
        for metric_column in metric_columns:
            if metric_column not in frame.columns:
                continue
            metric_value = _finite_numeric_value(record[metric_column])
            if metric_value is None:
                continue
            rows.append(
                _row_from_metric(
                    spec=spec,
                    runtime=row_runtime,
                    model_name=model_name,
                    horizon=int(runtime["horizon"]),
                    metric_name=(
                        f"execution.{metric_column}"
                        f"@confidence_threshold={threshold:g}"
                    ),
                    metric_value=metric_value,
                    source_experiment=source_experiment,
                )
            )
            rows[-1]["split"] = split
    return rows


# ---------------------------------------------------------------------------
# Markdown report rendering
# ---------------------------------------------------------------------------


_REPORT_FILE_BY_TYPE: dict[str, str] = {
    "calibration_bins": "calibration_ablation.md",
    "cost_bps": "cost_sensitivity.md",
    "latency_steps": "latency_sensitivity.md",
    "horizon": "horizon_ablation.md",
    "lookback": "lookback_window_ablation.md",
    "feature_group": "feature_group_ablation.md",
}

_REPORT_TITLE_BY_TYPE: dict[str, str] = {
    "calibration_bins": "Calibration-bin Ablation",
    "cost_bps": "Cost-Sensitivity Ablation",
    "latency_steps": "Latency-Sensitivity Ablation",
    "horizon": "Horizon Ablation",
    "lookback": "Lookback-Window Ablation",
    "feature_group": "Feature-Group Ablation",
}

_REPORT_INTRO_BY_TYPE: dict[str, str] = {
    "calibration_bins": (
        "Vary only the calibration bin count used to compute reliability "
        "evidence from stored predictions; everything else is held fixed."
    ),
    "cost_bps": (
        "Vary only the per-trade cost assumption used by the simplified "
        "execution-aware sensitivity analysis; everything else is held fixed. "
        "These are explicit proxy assumptions, not live-execution results."
    ),
    "latency_steps": (
        "Vary only the row-step latency assumption used by the simplified "
        "execution-aware sensitivity analysis; everything else is held fixed. "
        "These are explicit proxy assumptions, not live-execution results."
    ),
    "horizon": (
        "Vary only the prediction horizon and matching label column. Labels "
        "are regenerated for each horizon. The temporal split, preprocessing "
        "and leakage controls remain unchanged."
    ),
    "lookback": (
        "Vary only the neural lookback window. Classical baselines do not "
        "consume windows and therefore record the same metric values across "
        "this ablation; lookback effects are only meaningful when neural "
        "models are included."
    ),
    "feature_group": (
        "Restrict the feature matrix to a column-name pattern group; "
        "preprocessing stays fit on train rows only. The runner fails cleanly "
        "when too few columns match; the `all` group is the baseline child."
    ),
}


def _format_metric_value(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    if abs(value) >= 1e-4 and abs(value) < 1e6:
        return f"{value:.6f}"
    return f"{value:.6g}"


def _render_metric_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Render a small predictive-metric table for run rows."""
    selected_metrics = ("accuracy", "macro_f1")
    headers = ["ablation", "model", "metric", "value", "source"]
    lines: list[str] = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    found = False
    for row in rows:
        if row.get("status") != "run":
            continue
        metric_name = str(row.get("metric_name", ""))
        if metric_name not in selected_metrics:
            continue
        value = row.get("metric_value")
        if not isinstance(value, (int, float)):
            continue
        if isinstance(value, bool):
            continue
        if not math.isfinite(float(value)):
            continue
        found = True
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("ablation_name", "")),
                    str(row.get("model_name", "")),
                    metric_name,
                    _format_metric_value(float(value)),
                    str(row.get("source_experiment", "")),
                ]
            )
            + " |"
        )
    if not found:
        return []
    return lines


def _render_typed_report(
    *,
    ablation_type: str,
    title: str,
    intro: str,
    specs: Sequence[PaperAblationSpec],
    results: Sequence[PaperAblationResult],
    metric_rows: Sequence[Mapping[str, Any]],
    is_fixture: bool,
) -> str:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"Ablation type: `{ablation_type}`")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(intro)
    lines.append("")
    lines.append("## What Changed")
    lines.append("")
    for spec in specs:
        params_str = (
            ", ".join(f"`{key}={value}`" for key, value in spec.parameters.items())
            if spec.parameters
            else "(no parameter changes; reference run)"
        )
        lines.append(f"- `{spec.name}`: {spec.description}")
        lines.append(f"  - parameters: {params_str}")
    lines.append("")
    lines.append("## Held Fixed")
    lines.append("")
    lines.append(
        "- base FI-2010 benchmark preparation config, data path, seed and split "
        "design"
    )
    lines.append("- preprocessing fit on train rows only")
    lines.append("- model registry (no model family added by ablations)")
    lines.append(
        "- experiment artefact contract for each child experiment (validated "
        "before this report was written)"
    )
    lines.append("")
    lines.append("## Artefacts Used")
    lines.append("")
    lines.append("- `ablation_summary.json`")
    lines.append("- `ablation_results.csv`")
    lines.append("- `ablation_manifest.json`")
    for result in results:
        if result.status != "run" or result.experiment_dir is None:
            continue
        lines.append(f"- `{result.experiment_dir}`")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    for spec in specs:
        outcome = next(
            (result for result in results if result.name == spec.name),
            None,
        )
        if outcome is None:
            lines.append(f"- `{spec.name}`: not attempted")
            continue
        if outcome.status == "run":
            lines.append(
                f"- `{spec.name}`: run "
                f"(experiment: `{outcome.experiment_dir}`)"
            )
        else:
            reason = outcome.reason or "skipped"
            lines.append(f"- `{spec.name}`: skipped - {reason}")
    lines.append("")
    table_lines = _render_metric_table(metric_rows)
    if table_lines:
        lines.append("## Key Metric Summary")
        lines.append("")
        lines.extend(table_lines)
        lines.append("")
    lines.append("## Warnings And Limitations")
    lines.append("")
    if is_fixture:
        lines.append(
            "- This ablation summary was generated from the synthetic FI-2010 "
            "fixture. It is a synthetic fixture smoke run only and is not "
            "benchmark evidence."
        )
    lines.append(
        "- Ablation rows do not present trading or live-execution claims. "
        "Execution-aware values are simplified proxy assumptions."
    )
    lines.append(
        "- Aggregated values come directly from stored child-experiment "
        "artefacts and are not edited after the run."
    )
    lines.append("")
    return "\n".join(lines)


def _render_ssl_report(*, is_fixture: bool) -> str:
    lines: list[str] = []
    lines.append("# SSL Pretraining Ablation")
    lines.append("")
    lines.append("Ablation type: `ssl_pretraining`")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "Document the SSL pretraining status explicitly so an unsupported "
        "model family is not hidden inside the ablation grid."
    )
    lines.append("")
    lines.append("## What Changed")
    lines.append("")
    lines.append(
        "- No child experiment is run for SSL pretraining in this suite."
    )
    lines.append(
        "- The ablation is recorded as skipped in `ablation_summary.json`, "
        "`ablation_manifest.json` and `ablation_results.csv`."
    )
    lines.append("")
    lines.append("## Held Fixed")
    lines.append("")
    lines.append("- paper-runner model registry")
    lines.append("- no SSL checkpoint, pretraining config or fine-tuning config")
    lines.append("- held-out evaluation requirement before any SSL result is reported")
    lines.append("")
    lines.append("## Artefacts Used")
    lines.append("")
    lines.append("- `ablation_summary.json`")
    lines.append("- `ablation_results.csv`")
    lines.append("- `ablation_manifest.json`")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("- skipped")
    lines.append("")
    lines.append("## Reason")
    lines.append("")
    lines.append("- reason: no traceable runner support for SSL pretraining/fine-tuning yet")
    lines.append("- " + _SSL_SKIP_REASON)
    lines.append(
        "- `ssl_transformer` is intentionally not registered in the paper "
        "runner model registry, and the ablation runner does not report "
        "SSL results without a run."
    )
    lines.append("")
    lines.append("## Requirements Before Enabling")
    lines.append("")
    lines.append(
        "- a train-only pretraining stage with a stored pretraining config"
    )
    lines.append(
        "- a stored checkpoint or weight-transfer trace so the fine-tuning "
        "stage is reproducible"
    )
    lines.append(
        "- a fine-tuning config that uses the pretrained representation and "
        "is fitted without test-row input"
    )
    lines.append(
        "- a held-out evaluation that validates the artefact contract before "
        "any SSL claim is made"
    )
    lines.append("")
    if is_fixture:
        lines.append(
            "## Smoke Context"
        )
        lines.append("")
        lines.append(
            "- This skip report was produced during a synthetic fixture smoke "
            "run and is not benchmark evidence."
        )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_paper_ablations(
    config_path: Path,
    data_path: Path,
    out_dir: Path,
    *,
    models: Sequence[str],
    ablation_set: str = "smoke",
    overwrite: bool = False,
    build_plots: bool = False,
) -> PaperAblationSummary:
    """Run a controlled paper-experiment ablation suite.

    Parameters
    ----------
    config_path:
        Path to the base FI-2010 paper-runner config.
    data_path:
        Local FI-2010-style file path. Required; never downloaded.
    out_dir:
        Top-level ablation output directory.
    models:
        Sequence of model short names from the paper-runner registry.
    ablation_set:
        Named set of ablations to run. Supported values: ``"smoke"`` and
        ``"standard"``.
    overwrite:
        When ``False``, refuse to write into an existing non-empty
        directory. When ``True``, the existing directory is replaced.
    build_plots:
        Forwarded to each child paper experiment.
    """
    resolved_config_path = Path(config_path)
    resolved_data_path = Path(data_path)
    resolved_out_dir = Path(out_dir)

    if not resolved_config_path.is_file():
        raise FileNotFoundError(
            f"paper ablation config not found: {resolved_config_path}"
        )
    if not resolved_data_path.exists():
        raise FileNotFoundError(
            f"local FI-2010 data path does not exist: {resolved_data_path}"
        )
    if not resolved_data_path.is_file():
        raise FileNotFoundError(
            f"local FI-2010 data path is not a regular file: {resolved_data_path}"
        )

    ablation_set_name = str(ablation_set).strip().lower()
    if ablation_set_name not in SUPPORTED_ABLATION_SETS:
        raise ValueError(
            f"unsupported ablation set {ablation_set!r}; supported sets: "
            f"{list(SUPPORTED_ABLATION_SETS)}"
        )

    requested_models = normalise_paper_model_names(list(models))
    has_neural = _has_neural_model(requested_models)

    if resolved_out_dir.exists():
        if not resolved_out_dir.is_dir():
            raise FileExistsError(
                f"output path exists and is not a directory: {resolved_out_dir}"
            )
        if any(resolved_out_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    "refusing to write into a non-empty output directory; "
                    "pass overwrite=True to replace it: "
                    f"{resolved_out_dir}"
                )
            shutil.rmtree(resolved_out_dir)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    # Validate the base config eagerly so config-level problems surface here.
    base_config: FI2010BenchmarkConfig = load_benchmark_config(resolved_config_path)
    base_payload = _load_yaml_payload(resolved_config_path)

    specs = build_ablation_specs(ablation_set_name)
    created_at = datetime.now(UTC)
    experiments_root = resolved_out_dir / "experiments"
    reports_root = resolved_out_dir / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    results: list[PaperAblationResult] = []
    csv_rows: list[dict[str, Any]] = []
    child_experiments: dict[str, str] = {}
    summary_warnings: list[str] = []
    is_fixture = _is_fixture_path(resolved_data_path.resolve())
    if is_fixture:
        summary_warnings.append(
            "synthetic fixture smoke run only; not benchmark evidence"
        )

    for spec in specs:
        runtime = _spec_runtime_columns(spec, base_config=base_config)
        skip_reason = _resolve_runtime_skip(spec, has_neural=has_neural)
        if skip_reason is not None:
            results.append(
                PaperAblationResult(
                    name=spec.name,
                    ablation_type=spec.ablation_type,
                    status="skipped",
                    reason=skip_reason,
                    experiment_dir=None,
                    parameters=dict(spec.parameters),
                    warnings=[skip_reason],
                )
            )
            csv_rows.append(
                _row_for_skip(
                    spec=spec,
                    runtime=runtime,
                    base_horizon=base_config.horizon,
                    reason=skip_reason,
                )
            )
            continue

        child_dir = experiments_root / spec.name
        try:
            child_summary = _run_child_experiment(
                spec=spec,
                base_payload=base_payload,
                data_path=resolved_data_path,
                child_dir=child_dir,
                models=requested_models,
                build_plots=build_plots,
            )
        except (
            FileNotFoundError,
            FileExistsError,
            ValueError,
            RuntimeError,
            TypeError,
            OSError,
        ) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            if child_dir.exists():
                shutil.rmtree(child_dir)
            results.append(
                PaperAblationResult(
                    name=spec.name,
                    ablation_type=spec.ablation_type,
                    status="skipped",
                    reason=reason,
                    experiment_dir=None,
                    parameters=dict(spec.parameters),
                    warnings=[reason],
                )
            )
            csv_rows.append(
                _row_for_skip(
                    spec=spec,
                    runtime=runtime,
                    base_horizon=base_config.horizon,
                    reason=reason,
                )
            )
            summary_warnings.append(f"ablation {spec.name!r} skipped: {reason}")
            continue

        child_relative = child_dir.relative_to(resolved_out_dir).as_posix()
        child_experiments[spec.name] = child_relative
        child_warnings: list[str] = list(child_summary.warnings)
        for warning in child_warnings:
            if not _is_reportable_child_warning(
                warning,
                build_plots=build_plots,
            ):
                continue
            summary_warnings.append(f"ablation {spec.name!r}: {warning}")
        results.append(
            PaperAblationResult(
                name=spec.name,
                ablation_type=spec.ablation_type,
                status="run",
                reason=None,
                experiment_dir=child_relative,
                parameters=dict(spec.parameters),
                warnings=child_warnings,
            )
        )

        for outcome in child_summary.outcomes:
            for metric_name, raw_value in outcome.metrics.items():
                if not isinstance(raw_value, (int, float)):
                    continue
                if isinstance(raw_value, bool):
                    continue
                metric_value = float(raw_value)
                if not math.isfinite(metric_value):
                    continue
                csv_rows.append(
                    _row_from_metric(
                        spec=spec,
                        runtime=runtime,
                        model_name=outcome.model_name,
                        horizon=outcome.horizon,
                        metric_name=metric_name,
                        metric_value=metric_value,
                        source_experiment=child_relative,
                    )
                )
        csv_rows.extend(
            _rows_from_calibration_bins(
                spec=spec,
                runtime=runtime,
                child_dir=child_dir,
                source_experiment=child_relative,
            )
        )
        csv_rows.extend(
            _rows_from_execution_sensitivity(
                spec=spec,
                runtime=runtime,
                child_dir=child_dir,
                source_experiment=child_relative,
            )
        )

    # Write ablation_results.csv
    results_csv_path = resolved_out_dir / "ablation_results.csv"
    _write_ablation_results_csv(csv_rows, results_csv_path)

    # Decide which typed reports to write based on observed ablation types
    observed_types = {spec.ablation_type for spec in specs}
    reports_written: list[str] = []

    for ablation_type, filename in _REPORT_FILE_BY_TYPE.items():
        if ablation_type not in observed_types:
            continue
        title = _REPORT_TITLE_BY_TYPE[ablation_type]
        intro = _REPORT_INTRO_BY_TYPE[ablation_type]
        typed_specs = [spec for spec in specs if spec.ablation_type == ablation_type]
        typed_results = [
            result for result in results if result.ablation_type == ablation_type
        ]
        typed_metric_rows = [
            row for row in csv_rows if row.get("ablation_type") == ablation_type
        ]
        if ablation_type == "feature_group":
            typed_metric_rows.extend(
                row for row in csv_rows if row.get("ablation_name") == "baseline"
            )
        report_text = _render_typed_report(
            ablation_type=ablation_type,
            title=title,
            intro=intro,
            specs=typed_specs,
            results=typed_results,
            metric_rows=typed_metric_rows,
            is_fixture=is_fixture,
        )
        report_path = reports_root / filename
        report_path.write_text(report_text, encoding="utf-8")
        reports_written.append(f"reports/{filename}")

    # SSL pretraining ablation report is always written
    ssl_report_path = reports_root / "ssl_pretraining_ablation.md"
    ssl_report_path.write_text(
        _render_ssl_report(is_fixture=is_fixture),
        encoding="utf-8",
    )
    reports_written.append("reports/ssl_pretraining_ablation.md")
    reports_written = sorted(set(reports_written))

    # Build manifest with config and traceability metadata
    manifest_payload: dict[str, Any] = {
        "runner_version": PAPER_ABLATION_VERSION,
        "created_at": created_at.isoformat(),
        "base_config": str(resolved_config_path),
        "data_path": str(resolved_data_path),
        "data_source_kind": _data_source_kind(resolved_data_path),
        "output_dir": str(resolved_out_dir),
        "ablation_set": ablation_set_name,
        "models_requested": list(requested_models),
        "ablations_requested": [spec.name for spec in specs],
        "ablations_run": [
            result.name for result in results if result.status == "run"
        ],
        "ablations_skipped": [
            result.name for result in results if result.status == "skipped"
        ],
        "child_experiments": dict(child_experiments),
        "reports_written": list(reports_written),
        "is_fixture": is_fixture,
        "specs": [
            {
                "name": spec.name,
                "ablation_type": spec.ablation_type,
                "description": spec.description,
                "parameters": dict(spec.parameters),
                "requires_neural_model": spec.requires_neural_model,
                "always_skip": spec.always_skip,
                "skip_reason": spec.skip_reason,
            }
            for spec in specs
        ],
        "results": [result.model_dump(mode="json") for result in results],
        "warnings": list(summary_warnings),
    }
    manifest_path = resolved_out_dir / "ablation_manifest.json"
    manifest_path.write_text(stable_json_dumps(manifest_payload), encoding="utf-8")

    summary = PaperAblationSummary(
        runner_version=PAPER_ABLATION_VERSION,
        created_at=created_at,
        base_config=str(resolved_config_path),
        data_path=str(resolved_data_path),
        data_source_kind=_data_source_kind(resolved_data_path),
        output_dir=str(resolved_out_dir),
        ablation_set=ablation_set_name,
        models_requested=list(requested_models),
        ablations_requested=[spec.name for spec in specs],
        ablations_run=[
            result.name for result in results if result.status == "run"
        ],
        ablations_skipped=[
            result.name for result in results if result.status == "skipped"
        ],
        child_experiments=dict(child_experiments),
        reports_written=list(reports_written),
        is_fixture=is_fixture,
        warnings=list(summary_warnings),
        results=results,
    )

    summary_path = resolved_out_dir / "ablation_summary.json"
    summary_path.write_text(stable_json_dumps(summary), encoding="utf-8")

    return summary
