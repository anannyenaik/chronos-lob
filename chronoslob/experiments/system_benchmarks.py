"""Local systems benchmark orchestration for ChronosLOB.

The systems benchmark suite measures research-platform plumbing under
explicit local conditions. It records loader throughput, feature-generation
speed, paper-runner wall-clock timing, tiny CPU inference latency and a
Python-level resource profile. Measurements from bundled fixtures are labelled
as smoke measurements only and are not benchmark evidence.
"""

from __future__ import annotations

import csv
import math
import os
import platform
import shutil
import sys
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, TypeVar

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from chronoslob import __version__
from chronoslob.data.fi2010 import FI2010Config, FI2010Dataset, load_fi2010
from chronoslob.data.schemas import ensure_utc_datetime
from chronoslob.experiments.artifacts import validate_experiment_directory
from chronoslob.experiments.fi2010_benchmark import (
    FI2010BenchmarkConfig,
    load_benchmark_config,
)
from chronoslob.experiments.manifests import sha256_file, stable_json_dumps
from chronoslob.experiments.model_registry import normalise_paper_model_names
from chronoslob.experiments.paper_runner import run_paper_experiment
from chronoslob.features.pipeline import (
    FeaturePipelineConfig,
    build_feature_frame_from_fi2010,
    validate_feature_frame,
)
from chronoslob.models.preprocessing import select_feature_columns
from chronoslob.training.experiment import get_git_commit

__all__ = [
    "SUPPORTED_SYSTEM_BENCHMARK_SETS",
    "SYSTEM_BENCHMARK_RESULTS_COLUMNS",
    "SystemBenchmarkMetric",
    "SystemBenchmarkSummary",
    "run_system_benchmarks",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)

SYSTEM_BENCHMARK_VERSION = "phase-i/system-benchmark-runner/v1"
SUPPORTED_SYSTEM_BENCHMARK_SETS: tuple[str, ...] = ("smoke", "standard")

SYSTEM_BENCHMARK_RESULTS_COLUMNS: tuple[str, ...] = (
    "benchmark_name",
    "benchmark_set",
    "status",
    "metric_name",
    "metric_value",
    "metric_unit",
    "rows",
    "models",
    "source",
    "warning",
)

_FIXTURE_PATH_MARKERS = ("tests", "fixtures")
_NO_WARNING = "none"

T = TypeVar("T")


@dataclass(frozen=True)
class _TimedResult:
    elapsed_seconds: float
    value: Any


class SystemBenchmarkMetric(BaseModel):
    """One CSV-shaped metric row from the systems benchmark suite."""

    model_config = _MODEL_CONFIG

    benchmark_name: str
    benchmark_set: str
    status: Literal["run", "skipped"]
    metric_name: str
    metric_value: float
    metric_unit: str
    rows: int = 0
    models: str = "not_applicable"
    source: str
    warning: str = _NO_WARNING

    @field_validator(
        "benchmark_name",
        "benchmark_set",
        "metric_name",
        "metric_unit",
        "models",
        "source",
        "warning",
    )
    @classmethod
    def _validate_non_empty_string(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("system benchmark strings must be non-empty")
        return value.strip()

    @field_validator("metric_value")
    @classmethod
    def _validate_metric_value(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("metric_value must be a finite number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("metric_value must be finite")
        return numeric

    @field_validator("rows")
    @classmethod
    def _validate_rows(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("rows must be an integer")
        if value < 0:
            raise ValueError("rows must be non-negative")
        return value


class SystemBenchmarkSummary(BaseModel):
    """Top-level summary returned by :func:`run_system_benchmarks`."""

    model_config = _MODEL_CONFIG

    runner_version: str
    created_at: datetime
    benchmark_set: str
    config_path: str
    data_path: str
    output_dir: str
    models_requested: list[str]
    benchmarks_run: list[str]
    benchmarks_skipped: list[str]
    reports_written: list[str]
    child_experiments: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    results: list[SystemBenchmarkMetric] = Field(default_factory=list)

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        return ensure_utc_datetime(value)

    @field_validator("runner_version", "benchmark_set", "config_path", "data_path", "output_dir")
    @classmethod
    def _validate_required_string(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("summary fields must be non-empty strings")
        return value.strip()


def _is_fixture_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return all(marker in parts for marker in _FIXTURE_PATH_MARKERS)


def _data_source_kind(path: Path) -> str:
    return "synthetic_fixture" if _is_fixture_path(path.resolve()) else "local_file"


def _display_path(path: Path) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(candidate)


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    value = numerator / denominator
    if not math.isfinite(value):
        return None
    return value


def _validate_finite_non_negative(value: float, *, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _timed_call(func: Callable[[], T]) -> _TimedResult:
    start = time.perf_counter()
    value = func()
    elapsed = time.perf_counter() - start
    return _TimedResult(
        elapsed_seconds=_validate_finite_non_negative(
            elapsed,
            name="elapsed_seconds",
        ),
        value=value,
    )


def _model_label(models: Sequence[str] | None) -> str:
    if not models:
        return "not_applicable"
    return ",".join(models)


def _metric(
    *,
    benchmark_name: str,
    benchmark_set: str,
    metric_name: str,
    metric_value: float,
    metric_unit: str,
    rows: int = 0,
    models: Sequence[str] | None = None,
    source: str,
    warning: str = _NO_WARNING,
) -> SystemBenchmarkMetric:
    return SystemBenchmarkMetric(
        benchmark_name=benchmark_name,
        benchmark_set=benchmark_set,
        status="run",
        metric_name=metric_name,
        metric_value=_validate_finite_non_negative(
            metric_value,
            name=f"{benchmark_name}.{metric_name}",
        ),
        metric_unit=metric_unit,
        rows=rows,
        models=_model_label(models),
        source=source,
        warning=warning,
    )


def _skipped_metric(
    *,
    benchmark_name: str,
    benchmark_set: str,
    source: str,
    warning: str,
    models: Sequence[str] | None = None,
) -> SystemBenchmarkMetric:
    return SystemBenchmarkMetric(
        benchmark_name=benchmark_name,
        benchmark_set=benchmark_set,
        status="skipped",
        metric_name="status",
        metric_value=0.0,
        metric_unit="not_applicable",
        rows=0,
        models=_model_label(models),
        source=source,
        warning=warning,
    )


def _validate_benchmark_set(benchmark_set: str) -> str:
    cleaned = str(benchmark_set).strip().lower()
    if cleaned not in SUPPORTED_SYSTEM_BENCHMARK_SETS:
        raise ValueError(
            f"unsupported systems benchmark set {benchmark_set!r}; "
            f"supported sets: {list(SUPPORTED_SYSTEM_BENCHMARK_SETS)}"
        )
    return cleaned


def _resolve_models(
    models: Sequence[str] | None,
    *,
    benchmark_set: str,
) -> tuple[str, ...]:
    if models is None:
        return normalise_paper_model_names(["majority", "logistic"])
    if benchmark_set == "standard":
        return normalise_paper_model_names(models)
    return normalise_paper_model_names(models)


def _load_dataset(config: FI2010BenchmarkConfig, data_path: Path) -> FI2010Dataset:
    fi2010_config = FI2010Config(
        path=data_path,
        timestamp_column=config.timestamp_column,
        split_column=config.split_column,
        label_columns=list(config.label_columns),
        price_level_count=config.price_level_count,
    )
    return load_fi2010(fi2010_config)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return select_feature_columns(frame, reject_label_like=True)


def _run_loader_throughput(
    *,
    benchmark_set: str,
    config: FI2010BenchmarkConfig,
    data_path: Path,
) -> tuple[list[SystemBenchmarkMetric], FI2010Dataset, list[str]]:
    benchmark_name = "loader_throughput"
    warnings: list[str] = []
    timed = _timed_call(lambda: _load_dataset(config, data_path))
    dataset = timed.value
    if not isinstance(dataset, FI2010Dataset):
        raise TypeError("FI-2010 loader returned an unexpected object")

    rows_loaded = int(dataset.n_rows)
    metrics = [
        _metric(
            benchmark_name=benchmark_name,
            benchmark_set=benchmark_set,
            metric_name="elapsed_seconds",
            metric_value=timed.elapsed_seconds,
            metric_unit="seconds",
            rows=rows_loaded,
            source="chronoslob.data.fi2010.load_fi2010",
        ),
        _metric(
            benchmark_name=benchmark_name,
            benchmark_set=benchmark_set,
            metric_name="rows_loaded",
            metric_value=float(rows_loaded),
            metric_unit="rows",
            rows=rows_loaded,
            source="chronoslob.data.fi2010.load_fi2010",
        ),
    ]
    throughput = _safe_divide(float(rows_loaded), timed.elapsed_seconds)
    if throughput is None:
        warnings.append("loader row throughput could not be computed safely")
    else:
        metrics.append(
            _metric(
                benchmark_name=benchmark_name,
                benchmark_set=benchmark_set,
                metric_name="rows_per_second",
                metric_value=throughput,
                metric_unit="rows/second",
                rows=rows_loaded,
                source="chronoslob.data.fi2010.load_fi2010",
            )
        )
    return metrics, dataset, warnings


def _run_feature_generation_speed(
    *,
    benchmark_set: str,
    dataset: FI2010Dataset,
) -> tuple[list[SystemBenchmarkMetric], pd.DataFrame, list[str]]:
    benchmark_name = "feature_generation_speed"
    warnings: list[str] = []
    timed = _timed_call(
        lambda: build_feature_frame_from_fi2010(
            dataset,
            FeaturePipelineConfig(),
        )
    )
    feature_frame = timed.value
    if not isinstance(feature_frame, pd.DataFrame):
        raise TypeError("feature pipeline returned an unexpected object")

    validation = validate_feature_frame(feature_frame, allow_nan=True)
    if not validation.ok:
        warnings.append("feature frame validation reported issues")

    try:
        feature_columns = _feature_columns(feature_frame)
    except ValueError as exc:
        warnings.append(f"feature column selection warning: {exc}")
        feature_columns = []

    rows_processed = len(feature_frame)
    features_generated = rows_processed * len(feature_columns)
    metrics = [
        _metric(
            benchmark_name=benchmark_name,
            benchmark_set=benchmark_set,
            metric_name="elapsed_seconds",
            metric_value=timed.elapsed_seconds,
            metric_unit="seconds",
            rows=rows_processed,
            source="chronoslob.features.pipeline.build_feature_frame_from_fi2010",
        ),
        _metric(
            benchmark_name=benchmark_name,
            benchmark_set=benchmark_set,
            metric_name="rows_processed",
            metric_value=float(rows_processed),
            metric_unit="rows",
            rows=rows_processed,
            source="chronoslob.features.pipeline.build_feature_frame_from_fi2010",
        ),
        _metric(
            benchmark_name=benchmark_name,
            benchmark_set=benchmark_set,
            metric_name="features_generated",
            metric_value=float(features_generated),
            metric_unit="feature_values",
            rows=rows_processed,
            source="chronoslob.features.pipeline.build_feature_frame_from_fi2010",
        ),
    ]
    row_throughput = _safe_divide(float(rows_processed), timed.elapsed_seconds)
    if row_throughput is not None:
        metrics.append(
            _metric(
                benchmark_name=benchmark_name,
                benchmark_set=benchmark_set,
                metric_name="rows_per_second",
                metric_value=row_throughput,
                metric_unit="rows/second",
                rows=rows_processed,
                source="chronoslob.features.pipeline.build_feature_frame_from_fi2010",
            )
        )
    feature_throughput = _safe_divide(float(features_generated), timed.elapsed_seconds)
    if feature_throughput is not None:
        metrics.append(
            _metric(
                benchmark_name=benchmark_name,
                benchmark_set=benchmark_set,
                metric_name="features_per_second",
                metric_value=feature_throughput,
                metric_unit="feature_values/second",
                rows=rows_processed,
                source="chronoslob.features.pipeline.build_feature_frame_from_fi2010",
            )
        )
    return metrics, feature_frame, warnings


def _prediction_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    frame = pd.read_csv(path)
    return len(frame)


def _artefact_count(directory: Path) -> int:
    return sum(1 for path in directory.rglob("*") if path.is_file())


def _run_experiment_runner_timing(
    *,
    benchmark_set: str,
    config_path: Path,
    data_path: Path,
    out_dir: Path,
    models: Sequence[str],
) -> tuple[list[SystemBenchmarkMetric], dict[str, str], list[str]]:
    benchmark_name = "experiment_runner_timing"
    child_dir = out_dir / "child_experiments" / "paper_runner_timing"
    child_dir.parent.mkdir(parents=True, exist_ok=True)

    timed = _timed_call(
        lambda: run_paper_experiment(
            config_path=config_path,
            data_path=data_path,
            out_dir=child_dir,
            models=list(models),
            overwrite=True,
            build_plots=False,
        )
    )
    summary = timed.value
    validation = validate_experiment_directory(child_dir, include_plots=True)
    if not validation.is_valid:
        missing = ", ".join(validation.missing_required)
        raise RuntimeError(
            f"paper-runner timing child experiment failed validation: {missing}"
        )

    warnings = [
        warning
        for warning in summary.warnings
        if not warning.startswith("optional artefact missing: plots/")
    ]
    models_run = list(summary.models_run)
    prediction_rows = _prediction_row_count(child_dir / "predictions.csv")
    count = _artefact_count(child_dir)
    metrics = [
        _metric(
            benchmark_name=benchmark_name,
            benchmark_set=benchmark_set,
            metric_name="elapsed_seconds",
            metric_value=timed.elapsed_seconds,
            metric_unit="seconds",
            rows=prediction_rows,
            models=models,
            source="chronoslob.experiments.paper_runner.run_paper_experiment",
        ),
        _metric(
            benchmark_name=benchmark_name,
            benchmark_set=benchmark_set,
            metric_name="models_requested",
            metric_value=float(len(models)),
            metric_unit="models",
            rows=prediction_rows,
            models=models,
            source="chronoslob.experiments.paper_runner.run_paper_experiment",
        ),
        _metric(
            benchmark_name=benchmark_name,
            benchmark_set=benchmark_set,
            metric_name="models_run",
            metric_value=float(len(models_run)),
            metric_unit="models",
            rows=prediction_rows,
            models=models_run,
            source="chronoslob.experiments.paper_runner.run_paper_experiment",
        ),
        _metric(
            benchmark_name=benchmark_name,
            benchmark_set=benchmark_set,
            metric_name="prediction_rows",
            metric_value=float(prediction_rows),
            metric_unit="rows",
            rows=prediction_rows,
            models=models_run,
            source="chronoslob.experiments.paper_runner.run_paper_experiment",
        ),
        _metric(
            benchmark_name=benchmark_name,
            benchmark_set=benchmark_set,
            metric_name="artefact_count",
            metric_value=float(count),
            metric_unit="files",
            rows=prediction_rows,
            models=models_run,
            source="chronoslob.experiments.paper_runner.run_paper_experiment",
        ),
    ]
    child_experiments = {
        "paper_runner_timing": child_dir.relative_to(out_dir).as_posix(),
    }
    return metrics, child_experiments, warnings


def _timestamps_for_frame(frame: pd.DataFrame, config: FI2010BenchmarkConfig) -> pd.Series:
    if config.timestamp_column is not None and config.timestamp_column in frame.columns:
        return pd.to_datetime(frame[config.timestamp_column], utc=True, errors="raise")
    base = datetime(2000, 1, 1, tzinfo=UTC)
    return pd.Series(
        [base + timedelta(seconds=index) for index in range(len(frame))],
        index=frame.index,
    )


def _raw_sequence_frames(
    dataset: FI2010Dataset,
    config: FI2010BenchmarkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    frame = dataset.frame
    timestamps = _timestamps_for_frame(frame, config=config).reset_index(drop=True)
    feature_columns = list(dataset.feature_columns)
    feature_frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": [dataset.config.symbol] * len(frame),
        }
    )
    for column in feature_columns:
        feature_frame[column] = frame[column].reset_index(drop=True)

    label_frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": [dataset.config.symbol] * len(frame),
            config.label_name: frame[config.label_name].reset_index(drop=True),
        }
    )
    return feature_frame, label_frame, feature_columns


def _run_inference_latency(
    *,
    benchmark_set: str,
    dataset: FI2010Dataset,
    config: FI2010BenchmarkConfig,
) -> tuple[list[SystemBenchmarkMetric], list[str]]:
    benchmark_name = "inference_latency"
    try:
        import torch

        from chronoslob.models.deeplob import DeepLOBConfig, create_deeplob_model
        from chronoslob.training.datasets import (
            SequenceDataset,
            SequenceWindowConfig,
            torch_is_available,
        )
        from chronoslob.training.torch_training import set_torch_deterministic
    except ImportError as exc:
        reason = f"PyTorch inference path unavailable: {exc}"
        return [
            _skipped_metric(
                benchmark_name=benchmark_name,
                benchmark_set=benchmark_set,
                source="deeplob_forward",
                warning=reason,
            )
        ], [reason]

    if not torch_is_available():
        reason = "PyTorch is not importable in this environment"
        return [
            _skipped_metric(
                benchmark_name=benchmark_name,
                benchmark_set=benchmark_set,
                source="deeplob_forward",
                warning=reason,
            )
        ], [reason]

    try:
        feature_frame, label_frame, feature_columns = _raw_sequence_frames(dataset, config)
        if not feature_columns:
            reason = "no FI-2010 feature columns are available for inference timing"
            return [
                _skipped_metric(
                    benchmark_name=benchmark_name,
                    benchmark_set=benchmark_set,
                    source="deeplob_forward",
                    warning=reason,
                )
            ], [reason]

        lookback = min(max(1, int(config.neural_settings.lookback)), len(feature_frame))
        sequence_config = SequenceWindowConfig(
            lookback=lookback,
            target_column=config.label_name,
            feature_columns=list(feature_columns),
            require_contiguous_indices=True,
        )
        sequence_dataset = SequenceDataset(
            feature_frame=feature_frame,
            label_frame=label_frame,
            config=sequence_config,
            feature_columns=list(feature_columns),
        )
        if len(sequence_dataset) == 0:
            reason = "sequence dataset produced no windows for inference timing"
            return [
                _skipped_metric(
                    benchmark_name=benchmark_name,
                    benchmark_set=benchmark_set,
                    source="deeplob_forward",
                    warning=reason,
                )
            ], [reason]

        if config.neural_settings.deterministic:
            set_torch_deterministic(config.seed)
        windows = [sequence_dataset[index]["x"] for index in range(len(sequence_dataset))]
        batch = torch.stack(windows, dim=0).to("cpu")
        model_config = DeepLOBConfig(
            input_features=sequence_dataset.n_features,
            n_classes=max(2, sequence_dataset.n_classes),
            conv_channels=config.neural_settings.deeplob_conv_channels,
            lstm_hidden_size=config.neural_settings.deeplob_lstm_hidden_size,
            dropout=0.0,
            use_batch_norm=False,
        )
        model = create_deeplob_model(model_config).to("cpu")
        model.eval()
        repetitions = 5
        with torch.no_grad():
            _ = model(batch)

        def _forward_passes() -> Any:
            output = None
            with torch.no_grad():
                for _ in range(repetitions):
                    output = model(batch)
            return output

        timed = _timed_call(_forward_passes)
        windows_measured = int(len(sequence_dataset) * repetitions)
        latency = _safe_divide(timed.elapsed_seconds, float(windows_measured))
        if latency is None:
            reason = "inference latency could not be computed safely"
            return [
                _skipped_metric(
                    benchmark_name=benchmark_name,
                    benchmark_set=benchmark_set,
                    source="deeplob_forward",
                    warning=reason,
                )
            ], [reason]

        metrics = [
            _metric(
                benchmark_name=benchmark_name,
                benchmark_set=benchmark_set,
                metric_name="windows_measured",
                metric_value=float(windows_measured),
                metric_unit="windows",
                rows=windows_measured,
                models=["deeplob_style"],
                source="deeplob_forward",
            ),
            _metric(
                benchmark_name=benchmark_name,
                benchmark_set=benchmark_set,
                metric_name="elapsed_seconds",
                metric_value=timed.elapsed_seconds,
                metric_unit="seconds",
                rows=windows_measured,
                models=["deeplob_style"],
                source="deeplob_forward",
            ),
            _metric(
                benchmark_name=benchmark_name,
                benchmark_set=benchmark_set,
                metric_name="latency_seconds_per_window",
                metric_value=latency,
                metric_unit="seconds/window",
                rows=windows_measured,
                models=["deeplob_style"],
                source="deeplob_forward",
            ),
            _metric(
                benchmark_name=benchmark_name,
                benchmark_set=benchmark_set,
                metric_name="latency_ms_per_window",
                metric_value=latency * 1_000.0,
                metric_unit="ms/window",
                rows=windows_measured,
                models=["deeplob_style"],
                source="deeplob_forward",
            ),
        ]
        return metrics, []
    except (ImportError, RuntimeError, ValueError, TypeError, IndexError) as exc:
        reason = f"inference latency skipped: {type(exc).__name__}: {exc}"
        return [
            _skipped_metric(
                benchmark_name=benchmark_name,
                benchmark_set=benchmark_set,
                source="deeplob_forward",
                warning=reason,
            )
        ], [reason]


def _run_memory_profile(
    *,
    benchmark_set: str,
    dataset: FI2010Dataset,
) -> tuple[list[SystemBenchmarkMetric], list[str]]:
    benchmark_name = "memory_profile"
    if tracemalloc.is_tracing():
        reason = "tracemalloc is already active; peak memory would not be isolated"
        return [
            _skipped_metric(
                benchmark_name=benchmark_name,
                benchmark_set=benchmark_set,
                source="tracemalloc",
                warning=reason,
            )
        ], [reason]

    try:
        tracemalloc.start()
        _ = build_feature_frame_from_fi2010(dataset, FeaturePipelineConfig())
        _, peak = tracemalloc.get_traced_memory()
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        reason = f"tracemalloc memory profile skipped: {type(exc).__name__}: {exc}"
        return [
            _skipped_metric(
                benchmark_name=benchmark_name,
                benchmark_set=benchmark_set,
                source="tracemalloc",
                warning=reason,
            )
        ], [reason]
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    peak_bytes = int(peak)
    peak_mb = float(peak_bytes) / (1024.0 * 1024.0)
    rows = int(dataset.n_rows)
    return [
        _metric(
            benchmark_name=benchmark_name,
            benchmark_set=benchmark_set,
            metric_name="peak_memory_bytes",
            metric_value=float(peak_bytes),
            metric_unit="bytes",
            rows=rows,
            source="tracemalloc_feature_generation",
        ),
        _metric(
            benchmark_name=benchmark_name,
            benchmark_set=benchmark_set,
            metric_name="peak_memory_mb",
            metric_value=peak_mb,
            metric_unit="MiB",
            rows=rows,
            source="tracemalloc_feature_generation",
        ),
        _metric(
            benchmark_name=benchmark_name,
            benchmark_set=benchmark_set,
            metric_name="section_measured",
            metric_value=1.0,
            metric_unit="feature_generation_section",
            rows=rows,
            source="tracemalloc_feature_generation",
        ),
    ], []


def _write_results_csv(rows: Sequence[SystemBenchmarkMetric], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SYSTEM_BENCHMARK_RESULTS_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(mode="json"))


def _metrics_for_benchmark(
    rows: Sequence[SystemBenchmarkMetric],
    benchmark_name: str,
) -> list[SystemBenchmarkMetric]:
    return [row for row in rows if row.benchmark_name == benchmark_name]


def _format_metric_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6g}"


def _render_report(
    *,
    title: str,
    purpose: str,
    method: str,
    data_path: Path,
    data_source_kind: str,
    benchmark_set: str,
    rows: Sequence[SystemBenchmarkMetric],
    limitations: Sequence[str],
    smoke_measurement: bool,
) -> str:
    lines: list[str] = [
        f"# {title}",
        "",
        "## Purpose",
        "",
        purpose,
        "",
        "## Measurement Method",
        "",
        method,
        "",
        "## Input Data Source",
        "",
        f"- path: `{_display_path(data_path)}`",
        f"- data source kind: `{data_source_kind}`",
        f"- benchmark set: `{benchmark_set}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Unit | Status | Warning |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        warning = row.warning if row.warning != _NO_WARNING else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    row.metric_name,
                    _format_metric_value(row.metric_value),
                    row.metric_unit,
                    row.status,
                    warning,
                ]
            )
            + " |"
        )
    lines.extend(["", "## Limitations", ""])
    for limitation in limitations:
        lines.append(f"- {limitation}")
    lines.extend(["", "## Smoke Fixture Measurement", ""])
    if smoke_measurement:
        lines.append(
            "Yes. These are smoke measurements for infrastructure validation only; "
            "they are not benchmark evidence and are not representative of a local "
            "FI-2010 benchmark run."
        )
    else:
        lines.append(
            "No. The run used a local benchmark path supplied to this command. "
            "Interpret metrics only with the recorded environment and input provenance."
        )
    lines.append("")
    return "\n".join(lines)


def _write_reports(
    *,
    out_dir: Path,
    data_path: Path,
    data_source_kind: str,
    benchmark_set: str,
    results: Sequence[SystemBenchmarkMetric],
    smoke_measurement: bool,
) -> list[str]:
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    specs: dict[str, tuple[str, str, str, list[str]]] = {
        "loader_throughput": (
            "Loader Throughput",
            "Measure local FI-2010 load timing and row throughput.",
            (
                "The benchmark calls the existing FI-2010 loader once with the "
                "supplied config and local data path, timing only the load call "
                "with `time.perf_counter`."
            ),
            [
                "Throughput is local to the recorded machine, Python environment and input file.",
                "Fixture timings validate plumbing only and must not be compared with real runs.",
            ],
        ),
        "feature_generation_speed": (
            "Feature-Generation Speed",
            "Measure past-only feature-frame construction on the loaded FI-2010 data.",
            (
                "The benchmark calls the existing feature pipeline once on the "
                "loaded FI-2010 dataset and reports rows and generated feature values."
            ),
            [
                "Feature semantics are unchanged; labels are not introduced as features.",
                "The tiny fixture has too few rows to represent production-size workloads.",
            ],
        ),
        "experiment_runner_timing": (
            "Experiment-Runner Timing",
            "Measure wall-clock time for a small paper-runner invocation.",
            (
                "The benchmark runs `run_paper_experiment` into "
                "`child_experiments/paper_runner_timing` and validates the child "
                "experiment artefact directory before reporting timing metrics."
            ),
            [
                "The child run is a real artefact-producing runner output.",
                "Fixture child outputs remain smoke artefacts and are not benchmark evidence.",
            ],
        ),
        "inference_latency": (
            "Inference Latency",
            "Measure tiny CPU neural forward-pass latency per window.",
            (
                "The benchmark builds a small DeepLOB-style model and split-local "
                "sequence windows using existing dataset utilities, then times repeated "
                "CPU forward passes with gradients disabled."
            ),
            [
                "The model is not trained for this benchmark section.",
                "This is inference-path latency, not a production latency claim.",
            ],
        ),
        "memory_profile": (
            "Memory Profile",
            "Measure a Python-level peak memory profile for feature generation.",
            (
                "The benchmark runs feature generation under `tracemalloc` and records "
                "the peak traced Python allocation for that section."
            ),
            [
                "`tracemalloc` does not capture every native allocation made by extensions.",
                "Treat this as a local resource profile, not a full system memory audit.",
            ],
        ),
    }
    filenames = {
        "loader_throughput": "loader_throughput.md",
        "feature_generation_speed": "feature_generation_speed.md",
        "experiment_runner_timing": "experiment_runner_timing.md",
        "inference_latency": "inference_latency.md",
        "memory_profile": "memory_profile.md",
    }
    written: list[str] = []
    for benchmark_name in filenames:
        benchmark_rows = _metrics_for_benchmark(results, benchmark_name)
        if not benchmark_rows:
            continue
        title, purpose, method, limitations = specs[benchmark_name]
        report_path = reports_dir / filenames[benchmark_name]
        report_path.write_text(
            _render_report(
                title=title,
                purpose=purpose,
                method=method,
                data_path=data_path,
                data_source_kind=data_source_kind,
                benchmark_set=benchmark_set,
                rows=benchmark_rows,
                limitations=limitations,
                smoke_measurement=smoke_measurement,
            ),
            encoding="utf-8",
        )
        written.append(f"reports/{filenames[benchmark_name]}")
    return written


def _build_environment_payload(
    *,
    created_at: datetime,
    benchmark_set: str,
    models_requested: Sequence[str],
    data_path: Path,
    config_path: Path,
    data_source_kind: str,
    row_count: int | None,
    warnings: list[str],
) -> dict[str, Any]:
    code_commit = get_git_commit()
    if code_commit is None:
        warnings.append("git commit could not be resolved")
    stat = data_path.stat()
    return {
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "package_version": __version__,
        "cpu_count": os.cpu_count(),
        "process_id": os.getpid(),
        "code_commit": code_commit,
        "config_path": str(config_path),
        "data_source_path": str(data_path),
        "data_source_kind": data_source_kind,
        "data_size_bytes": int(stat.st_size),
        "data_modified_at": datetime.fromtimestamp(stat.st_mtime, UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "data_source_sha256": sha256_file(data_path),
        "data_row_count": row_count,
        "benchmark_set": benchmark_set,
        "models_requested": list(models_requested),
        "runner_version": SYSTEM_BENCHMARK_VERSION,
    }


def _write_json(path: Path, payload: BaseModel | Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json_dumps(payload), encoding="utf-8")


def _result_names_by_status(
    results: Sequence[SystemBenchmarkMetric],
    status: Literal["run", "skipped"],
) -> list[str]:
    names: list[str] = []
    for row in results:
        if row.status != status:
            continue
        if row.benchmark_name not in names:
            names.append(row.benchmark_name)
    return names


def run_system_benchmarks(
    config_path: Path,
    data_path: Path,
    out_dir: Path,
    benchmark_set: str = "smoke",
    models: Sequence[str] | None = None,
    overwrite: bool = False,
) -> SystemBenchmarkSummary:
    """Run local systems benchmarks and write traceable artefacts.

    The runner requires an explicit local FI-2010-style data path. It never
    downloads data and writes all generated artefacts under ``out_dir``.
    """
    resolved_config_path = Path(config_path)
    resolved_data_path = Path(data_path)
    resolved_out_dir = Path(out_dir)

    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"systems benchmark config not found: {resolved_config_path}")
    if not resolved_data_path.exists():
        raise FileNotFoundError(
            f"local FI-2010 data path does not exist: {resolved_data_path}"
        )
    if not resolved_data_path.is_file():
        raise FileNotFoundError(
            f"local FI-2010 data path is not a regular file: {resolved_data_path}"
        )

    benchmark_set_name = _validate_benchmark_set(benchmark_set)
    requested_models = _resolve_models(models, benchmark_set=benchmark_set_name)

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

    config = load_benchmark_config(resolved_config_path)
    created_at = datetime.now(UTC)
    data_source_kind = _data_source_kind(resolved_data_path)
    smoke_measurement = benchmark_set_name == "smoke" or data_source_kind == "synthetic_fixture"
    warnings: list[str] = []
    if smoke_measurement:
        warnings.append(
            "smoke measurement only; fixture timings are not benchmark evidence"
        )
    if benchmark_set_name == "standard" and data_source_kind == "synthetic_fixture":
        warnings.append(
            "standard benchmark set used a synthetic fixture path; results remain smoke artefacts"
        )

    results: list[SystemBenchmarkMetric] = []
    child_experiments: dict[str, str] = {}

    loader_metrics, dataset, loader_warnings = _run_loader_throughput(
        benchmark_set=benchmark_set_name,
        config=config,
        data_path=resolved_data_path,
    )
    results.extend(loader_metrics)
    warnings.extend(loader_warnings)

    feature_metrics, _, feature_warnings = _run_feature_generation_speed(
        benchmark_set=benchmark_set_name,
        dataset=dataset,
    )
    results.extend(feature_metrics)
    warnings.extend(feature_warnings)

    experiment_metrics, experiment_children, experiment_warnings = (
        _run_experiment_runner_timing(
            benchmark_set=benchmark_set_name,
            config_path=resolved_config_path,
            data_path=resolved_data_path,
            out_dir=resolved_out_dir,
            models=requested_models,
        )
    )
    results.extend(experiment_metrics)
    child_experiments.update(experiment_children)
    warnings.extend(
        f"paper-runner timing note: {warning}" for warning in experiment_warnings
    )

    inference_metrics, inference_warnings = _run_inference_latency(
        benchmark_set=benchmark_set_name,
        dataset=dataset,
        config=config,
    )
    results.extend(inference_metrics)
    warnings.extend(inference_warnings)

    memory_metrics, memory_warnings = _run_memory_profile(
        benchmark_set=benchmark_set_name,
        dataset=dataset,
    )
    results.extend(memory_metrics)
    warnings.extend(memory_warnings)

    environment_payload = _build_environment_payload(
        created_at=created_at,
        benchmark_set=benchmark_set_name,
        models_requested=requested_models,
        data_path=resolved_data_path,
        config_path=resolved_config_path,
        data_source_kind=data_source_kind,
        row_count=int(dataset.n_rows),
        warnings=warnings,
    )
    _write_json(resolved_out_dir / "environment.json", environment_payload)

    _write_results_csv(results, resolved_out_dir / "system_benchmark_results.csv")
    reports_written = _write_reports(
        out_dir=resolved_out_dir,
        data_path=resolved_data_path,
        data_source_kind=data_source_kind,
        benchmark_set=benchmark_set_name,
        results=results,
        smoke_measurement=smoke_measurement,
    )

    summary = SystemBenchmarkSummary(
        runner_version=SYSTEM_BENCHMARK_VERSION,
        created_at=created_at,
        benchmark_set=benchmark_set_name,
        config_path=str(resolved_config_path),
        data_path=str(resolved_data_path),
        output_dir=str(resolved_out_dir),
        models_requested=list(requested_models),
        benchmarks_run=_result_names_by_status(results, "run"),
        benchmarks_skipped=_result_names_by_status(results, "skipped"),
        reports_written=reports_written,
        child_experiments=child_experiments,
        warnings=list(dict.fromkeys(warnings)),
        results=results,
    )
    _write_json(resolved_out_dir / "system_benchmark_summary.json", summary)
    return summary
