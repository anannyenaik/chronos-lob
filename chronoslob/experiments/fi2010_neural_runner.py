"""FI-2010 multi-fold supervised neural benchmark runner.

The runner consumes split-aware CSV files produced by
``prepare-fi2010-multifold`` and executes the configured supervised neural
plan on selected folds, seeds, models and lookbacks. It preserves the
official train/test split, uses the official train tail for validation and
writes lightweight aggregate artefacts by default.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from chronoslob import __version__
from chronoslob.experiments.fi2010_benchmark import (
    BenchmarkSplitConfig,
    FI2010BenchmarkConfig,
    PaperNeuralSettings,
    build_benchmark_split,
    effective_split_summary,
)
from chronoslob.experiments.manifests import sha256_file, stable_json_dumps
from chronoslob.experiments.neural_adapters import run_neural_paper_model
from chronoslob.experiments.neural_benchmarking import (
    NeuralBenchmarkConfig,
    NeuralBenchmarkRunPlan,
    NeuralModelSpec,
    generate_neural_run_plan,
    load_neural_benchmark_config,
    validate_supported_neural_models,
)
from chronoslob.models.preprocessing import build_target_vector, select_feature_columns
from chronoslob.training.experiment import get_git_commit
from chronoslob.training.splitters import SplitIndices

__all__ = [
    "FI2010_NEURAL_RUNNER_VERSION",
    "FI2010NeuralBenchmarkSummary",
    "run_fi2010_neural_benchmark",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)

FI2010_NEURAL_RUNNER_VERSION = "phase-e/fi2010-neural-runner/v1"

_COMBINED_CSV_FILENAME_TEMPLATE = "fold{fold}_combined.csv"

RESULTS_BY_FOLD_SEED_COLUMNS: tuple[str, ...] = (
    "fold_id",
    "seed",
    "model_name",
    "lookback",
    "split",
    "accuracy",
    "macro_f1",
    "mcc",
    "brier_score",
    "ece",
    "n_train",
    "n_validation",
    "n_test",
    "status",
    "error",
)

RESULTS_SUMMARY_COLUMNS: tuple[str, ...] = (
    "model_name",
    "lookback",
    "split",
    "fold_count",
    "seed_count",
    "run_count",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "mcc_mean",
    "mcc_std",
    "brier_score_mean",
    "brier_score_std",
    "ece_mean",
    "ece_std",
)

TRAINING_SUMMARY_COLUMNS: tuple[str, ...] = (
    "fold_id",
    "seed",
    "model_name",
    "lookback",
    "device",
    "parameter_count",
    "max_epochs",
    "best_epoch",
    "early_stopped",
    "training_seconds",
    "validation_metric_name",
    "validation_metric_value",
    "status",
    "error",
)

MODEL_CAPACITY_SUMMARY_COLUMNS: tuple[str, ...] = (
    "model_name",
    "lookback",
    "run_count",
    "successful_run_count",
    "parameter_count_min",
    "parameter_count_mean",
    "parameter_count_max",
    "status",
)


class FI2010NeuralBenchmarkSummary(BaseModel):
    """Top-level return value from the FI-2010 neural benchmark runner."""

    model_config = _MODEL_CONFIG

    study_name: str
    dataset_name: str
    task_name: str
    target_horizon: int
    config_path: str
    processed_root: str
    output_dir: str
    folds_requested: list[str]
    folds_completed: list[str]
    models_requested: list[str]
    seeds: list[int]
    lookbacks: list[int]
    max_epochs: int
    run_count: int
    completed_run_count: int
    failure_count: int
    execution_mode: str
    full_benchmark_grid: bool
    full_predictions_written: bool
    checkpoints_written: bool
    artefacts: dict[str, str]
    runner_version: str
    created_at: datetime
    git_commit: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator(
        "study_name",
        "dataset_name",
        "task_name",
        "config_path",
        "processed_root",
        "output_dir",
        "execution_mode",
        "runner_version",
    )
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("summary text fields must be non-empty")
        return value.strip()

    @field_validator("execution_mode")
    @classmethod
    def _validate_execution_mode(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in {"smoke", "benchmark"}:
            raise ValueError("execution_mode must be smoke or benchmark")
        return cleaned


@dataclass(frozen=True)
class _FoldInput:
    fold_id: str
    path: Path
    sha256: str | None


@dataclass(frozen=True)
class _RunContext:
    benchmark_config: FI2010BenchmarkConfig
    split: SplitIndices
    split_summary: Mapping[str, Any]
    feature_columns: list[str]
    all_labels: list[Any]
    class_count_train: int
    class_count_test: int
    test_timestamps: list[str] | None


def _normalise_fold_token(value: str | int) -> str:
    if isinstance(value, bool):
        raise ValueError("fold identifiers must be positive integers or fold_N strings")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("fold identifiers must be positive")
        return f"fold_{value}"
    if not isinstance(value, str) or not value.strip():
        raise ValueError("fold identifiers must be non-empty")
    cleaned = value.strip().lower()
    if cleaned.isdigit():
        number = int(cleaned)
        if number <= 0:
            raise ValueError("fold identifiers must be positive")
        return f"fold_{number}"
    if cleaned.startswith("fold_") and cleaned.removeprefix("fold_").isdigit():
        number = int(cleaned.removeprefix("fold_"))
        if number <= 0:
            raise ValueError("fold identifiers must be positive")
        return f"fold_{number}"
    raise ValueError(f"invalid fold identifier {value!r}; expected fold_N or N")


def _fold_number(fold_id: str) -> int:
    return int(fold_id.removeprefix("fold_"))


def _normalise_folds(
    folds: Sequence[str | int] | str | None,
    *,
    config: NeuralBenchmarkConfig,
) -> tuple[str, ...]:
    if folds is None:
        return config.folds
    if isinstance(folds, str):
        text = folds.strip()
        if not text or text.casefold() == "all":
            return config.folds
        raw: Sequence[str | int] = [token.strip() for token in text.split(",")]
    else:
        raw = folds
    selected = tuple(_normalise_fold_token(item) for item in raw if str(item).strip())
    if not selected:
        raise ValueError("fold selection must not be empty")
    configured = set(config.folds)
    unknown = [fold for fold in selected if fold not in configured]
    if unknown:
        raise ValueError(f"requested folds are not configured: {unknown}")
    return tuple(dict.fromkeys(selected))


def _normalise_int_selection(
    value: Sequence[int] | str | None,
    *,
    configured: Sequence[int],
    field_name: str,
    allow_all_string: bool = True,
) -> tuple[int, ...]:
    if value is None:
        return tuple(configured)
    if isinstance(value, str):
        text = value.strip()
        if allow_all_string and (not text or text.casefold() == "all"):
            return tuple(configured)
        tokens: Sequence[str] = [token.strip() for token in text.split(",")]
        raw_values: list[int] = []
        for token in tokens:
            if not token:
                continue
            try:
                raw_values.append(int(token))
            except ValueError as exc:
                raise ValueError(f"{field_name} entries must be integers") from exc
    else:
        raw_values = [int(item) for item in value]

    cleaned: list[int] = []
    for item in raw_values:
        if item < 0:
            raise ValueError(f"{field_name} entries must be non-negative")
        if field_name == "lookbacks" and item <= 0:
            raise ValueError("lookbacks entries must be positive")
        if item not in cleaned:
            cleaned.append(item)
    if not cleaned:
        raise ValueError(f"{field_name} selection must not be empty")
    return tuple(cleaned)


def _normalise_model_selection(
    models: Sequence[str] | str | None,
    *,
    config: NeuralBenchmarkConfig,
) -> tuple[str, ...]:
    if models is None:
        selected = config.enabled_model_names
    elif isinstance(models, str):
        text = models.strip()
        if not text or text.casefold() == "all":
            selected = config.enabled_model_names
        else:
            selected = tuple(token.strip().lower() for token in text.split(",") if token.strip())
    else:
        selected = tuple(item.strip().lower() for item in models if item.strip())
    cleaned = validate_supported_neural_models(selected)
    enabled = set(config.enabled_model_names)
    disabled = [model for model in cleaned if model not in enabled]
    if disabled:
        raise ValueError(f"requested neural models are not enabled in the config: {disabled}")
    return cleaned


def _ensure_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {path}")
        if any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    "refusing to write into a non-empty output directory; "
                    f"pass overwrite=True to replace it: {path}",
                )
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _fold_csv_path(processed_root: Path, fold_id: str) -> Path:
    return processed_root / _COMBINED_CSV_FILENAME_TEMPLATE.replace(
        "{fold}",
        str(_fold_number(fold_id)),
    )


def _run_dir(out_dir: Path, plan: NeuralBenchmarkRunPlan) -> Path:
    return (
        out_dir
        / "runs"
        / f"{plan.fold_id}_seed_{plan.seed}_{plan.model_name}_lb{plan.lookback}"
    )


def _checkpoint_path(
    out_dir: Path,
    plan: NeuralBenchmarkRunPlan,
    *,
    filename: str,
    enabled: bool,
) -> Path | None:
    if not enabled:
        return None
    return (
        out_dir
        / "checkpoints"
        / f"{plan.fold_id}_seed_{plan.seed}_{plan.model_name}_lb{plan.lookback}"
        / filename
    )


def _write_csv(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    columns: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows), columns=None if columns is None else list(columns))
    if columns is not None and frame.empty:
        frame = pd.DataFrame(columns=list(columns))
    frame.to_csv(path, index=False)


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return None


def _mean(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    value = float(numeric.mean())
    return value if math.isfinite(value) else None


def _std(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) <= 1:
        return 0.0
    value = float(numeric.std(ddof=0))
    return value if math.isfinite(value) else 0.0


def _build_results_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return []
    ok = frame[frame["status"] == "ok"].copy()
    if ok.empty:
        return []
    summary_rows: list[dict[str, Any]] = []
    group_columns = ["model_name", "lookback", "split"]
    for keys, group in ok.groupby(group_columns, sort=True):
        model_name, lookback, split_name = keys
        row: dict[str, Any] = {
            "model_name": str(model_name),
            "lookback": int(lookback),
            "split": str(split_name),
            "fold_count": int(group["fold_id"].nunique()),
            "seed_count": int(group["seed"].nunique()),
            "run_count": len(group),
        }
        for metric in ("accuracy", "macro_f1", "mcc", "brier_score", "ece"):
            row[f"{metric}_mean"] = _mean(group[metric])
            row[f"{metric}_std"] = _std(group[metric])
        summary_rows.append(row)
    return summary_rows


def _build_capacity_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return []
    summary_rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(["model_name", "lookback"], sort=True):
        model_name, lookback = keys
        ok = group[group["status"] == "ok"]
        counts = pd.to_numeric(ok["parameter_count"], errors="coerce").dropna()
        if counts.empty:
            minimum = mean = maximum = None
        else:
            minimum = int(counts.min())
            mean = float(counts.mean())
            maximum = int(counts.max())
        summary_rows.append(
            {
                "model_name": str(model_name),
                "lookback": int(lookback),
                "run_count": len(group),
                "successful_run_count": len(ok),
                "parameter_count_min": minimum,
                "parameter_count_mean": mean,
                "parameter_count_max": maximum,
                "status": "ok" if len(ok) == len(group) else "partial",
            }
        )
    return summary_rows


def _benchmark_config(
    *,
    config: NeuralBenchmarkConfig,
    seed: int,
    fold_path: Path,
    out_dir: Path,
    settings: PaperNeuralSettings,
) -> FI2010BenchmarkConfig:
    return FI2010BenchmarkConfig(
        experiment_name=f"{config.study_name}_neural",
        dataset_name=config.dataset.name,
        task_name="midprice_direction",
        horizon=config.target.horizon,
        split_name="official_column",
        seed=seed,
        local_data_path=str(fold_path),
        output_dir=str(out_dir),
        label_name=config.target.label_column,
        label_columns=(config.target.label_column,),
        timestamp_column=None,
        split_column=config.official_split.split_column,
        price_level_count=10,
        train_fraction=0.7,
        validation_fraction=0.15,
        test_fraction=0.15,
        split=BenchmarkSplitConfig(
            method="official_column",
            split_column=config.official_split.split_column,
            validation_fraction_within_train=(
                config.official_split.validation_fraction_within_train
            ),
            train_value=config.official_split.train_value,
            test_value=config.official_split.test_value,
        ),
        neural_settings=settings,
    )


def _model_int(spec: NeuralModelSpec, field_name: str, default: int) -> int:
    value = getattr(spec, field_name)
    return default if value is None else int(value)


def _neural_settings(
    *,
    config: NeuralBenchmarkConfig,
    plan: NeuralBenchmarkRunPlan,
    checkpoint_path: Path | None,
    checkpoint_enabled: bool,
) -> PaperNeuralSettings:
    spec = config.neural_models[plan.model_name]
    return PaperNeuralSettings(
        supported_models=("deeplob_style", "matrix_transformer"),
        planned_models=(),
        lookback=plan.lookback,
        transformer_window_length=plan.lookback,
        batch_size=plan.batch_size,
        max_epochs=plan.max_epochs,
        early_stopping_patience=plan.early_stopping_patience,
        early_stopping_metric=plan.early_stopping_metric,
        learning_rate=plan.learning_rate,
        weight_decay=plan.weight_decay,
        gradient_clip_norm=plan.gradient_clip_norm,
        device=plan.device_policy,
        deterministic=config.deterministic_seed_handling.enabled,
        checkpoint_enabled=checkpoint_enabled,
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        dropout=plan.dropout,
        deeplob_conv_channels=_model_int(spec, "conv_channels", 16),
        deeplob_lstm_hidden_size=_model_int(spec, "lstm_hidden_size", 32),
        deeplob_use_batch_norm=bool(spec.use_batch_norm),
        transformer_model_dim=_model_int(spec, "model_dim", 16),
        transformer_num_heads=_model_int(spec, "num_heads", 2),
        transformer_num_layers=_model_int(spec, "num_layers", 1),
        transformer_feedforward_dim=_model_int(spec, "feedforward_dim", 32),
    )


def _feature_columns(frame: pd.DataFrame, config: NeuralBenchmarkConfig) -> list[str]:
    excluded = {
        config.official_split.split_column,
        config.target.label_column,
    }
    excluded.update(
        column
        for column in frame.columns
        if str(column).casefold().startswith("label")
    )
    candidate = frame.drop(columns=[column for column in excluded if column in frame.columns])
    columns = select_feature_columns(candidate, reject_label_like=True)
    if not columns:
        raise ValueError("neural benchmark runner requires at least one feature column")
    return columns


def _class_count(
    frame: pd.DataFrame,
    *,
    label_column: str,
    row_indices: Sequence[int],
) -> int:
    labels = {frame.iloc[int(index)][label_column] for index in row_indices}
    return len(labels)


def _test_timestamps(
    frame: pd.DataFrame,
    *,
    indices: Sequence[int],
    timestamp_column: str | None,
) -> list[str] | None:
    if timestamp_column is None or timestamp_column not in frame.columns:
        return None
    values = pd.to_datetime(
        frame.iloc[list(indices)][timestamp_column],
        utc=True,
        errors="raise",
    )
    return [value.isoformat() for value in values]


def _run_context(
    *,
    frame: pd.DataFrame,
    config: NeuralBenchmarkConfig,
    fold_input: _FoldInput,
    plan: NeuralBenchmarkRunPlan,
    out_dir: Path,
    settings: PaperNeuralSettings,
) -> _RunContext:
    benchmark_config = _benchmark_config(
        config=config,
        seed=plan.seed,
        fold_path=fold_input.path,
        out_dir=out_dir,
        settings=settings,
    )
    split = build_benchmark_split(benchmark_config, frame)
    split_summary = effective_split_summary(
        config=benchmark_config,
        frame=frame,
        indices=split,
    ).model_dump(mode="json")
    feature_columns = _feature_columns(frame, config)
    all_target = build_target_vector(frame, target_column=config.target.label_column)
    all_labels = list(all_target.classes or [])
    return _RunContext(
        benchmark_config=benchmark_config,
        split=split,
        split_summary=split_summary,
        feature_columns=feature_columns,
        all_labels=all_labels,
        class_count_train=_class_count(
            frame,
            label_column=config.target.label_column,
            row_indices=split.train,
        ),
        class_count_test=_class_count(
            frame,
            label_column=config.target.label_column,
            row_indices=split.test,
        ),
        test_timestamps=_test_timestamps(
            frame,
            indices=split.test,
            timestamp_column=benchmark_config.timestamp_column,
        ),
    )


def _validation_metric_value(
    metadata: Mapping[str, Any],
    *,
    metric_name: str,
) -> float | None:
    history_raw = metadata.get("training_history")
    if not isinstance(history_raw, Sequence) or isinstance(history_raw, (str, bytes)):
        return None
    history = [item for item in history_raw if isinstance(item, Mapping)]
    if not history:
        return None
    best = next(
        (item for item in reversed(history) if bool(item.get("is_best"))),
        history[-1],
    )
    if metric_name == "validation_macro_f1":
        return _safe_float(best.get("validation_macro_f1"))
    if metric_name == "validation_loss":
        return _safe_float(best.get("validation_loss"))
    return _safe_float(best.get("monitored_value"))


def _result_row(
    *,
    plan: NeuralBenchmarkRunPlan,
    metrics: Mapping[str, Any],
    split_summary: Mapping[str, Any],
    status: str,
    error: str = "",
) -> dict[str, Any]:
    return {
        "fold_id": plan.fold_id,
        "seed": plan.seed,
        "model_name": plan.model_name,
        "lookback": plan.lookback,
        "split": "test",
        "accuracy": _safe_float(metrics.get("accuracy")),
        "macro_f1": _safe_float(metrics.get("macro_f1")),
        "mcc": _safe_float(metrics.get("matthews_corrcoef")),
        "brier_score": _safe_float(metrics.get("brier_score")),
        "ece": _safe_float(
            metrics.get("ece", metrics.get("expected_calibration_error"))
        ),
        "n_train": int(split_summary.get("n_train", 0)),
        "n_validation": int(split_summary.get("n_validation", 0)),
        "n_test": int(split_summary.get("n_test", 0)),
        "status": status,
        "error": error,
    }


def _training_row(
    *,
    plan: NeuralBenchmarkRunPlan,
    metadata: Mapping[str, Any],
    status: str,
    error: str = "",
) -> dict[str, Any]:
    device_raw = metadata.get("device", {})
    device = (
        str(device_raw.get("resolved", plan.device_policy))
        if isinstance(device_raw, Mapping)
        else plan.device_policy
    )
    return {
        "fold_id": plan.fold_id,
        "seed": plan.seed,
        "model_name": plan.model_name,
        "lookback": plan.lookback,
        "device": device,
        "parameter_count": metadata.get("parameter_count"),
        "max_epochs": plan.max_epochs,
        "best_epoch": metadata.get("best_epoch"),
        "early_stopped": bool(metadata.get("early_stopped", False)),
        "training_seconds": _safe_float(metadata.get("training_seconds")),
        "validation_metric_name": plan.early_stopping_metric,
        "validation_metric_value": _validation_metric_value(
            metadata,
            metric_name=plan.early_stopping_metric,
        ),
        "status": status,
        "error": error,
    }


def _failure_result_row(
    *,
    plan: NeuralBenchmarkRunPlan,
    split_summary: Mapping[str, Any] | None,
    error: str,
) -> dict[str, Any]:
    return _result_row(
        plan=plan,
        metrics={},
        split_summary={} if split_summary is None else split_summary,
        status="failed",
        error=error,
    )


def _failure_training_row(
    *,
    plan: NeuralBenchmarkRunPlan,
    error: str,
) -> dict[str, Any]:
    return {
        "fold_id": plan.fold_id,
        "seed": plan.seed,
        "model_name": plan.model_name,
        "lookback": plan.lookback,
        "device": plan.device_policy,
        "parameter_count": None,
        "max_epochs": plan.max_epochs,
        "best_epoch": None,
        "early_stopped": False,
        "training_seconds": None,
        "validation_metric_name": plan.early_stopping_metric,
        "validation_metric_value": None,
        "status": "failed",
        "error": error,
    }


def _write_predictions(
    *,
    rows: Sequence[Mapping[str, Any]],
    run_dir: Path,
) -> bool:
    if not rows:
        return False
    _write_csv(rows, run_dir / "predictions.csv")
    return True


def _write_run_result(
    *,
    run_dir: Path,
    payload: Mapping[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(
        stable_json_dumps(payload),
        encoding="utf-8",
    )


def _write_model_card(
    *,
    run_dir: Path,
    plan: NeuralBenchmarkRunPlan,
    status: str,
    split_summary: Mapping[str, Any] | None,
    full_predictions_written: bool,
    checkpoint_written: bool,
    error: str = "",
) -> None:
    lines = [
        f"# FI-2010 Neural Run: {plan.run_id}",
        "",
        "## Scope",
        "",
        "- runner: supervised neural FI-2010 benchmark execution",
        f"- fold: `{plan.fold_id}`",
        f"- seed: {plan.seed}",
        f"- model: `{plan.model_name}`",
        f"- lookback: {plan.lookback}",
        f"- status: `{status}`",
        "",
        "## Split",
        "",
    ]
    if split_summary is None:
        lines.append("- split summary unavailable because the run failed before split construction")
    else:
        lines.extend(
            [
                f"- split method: `{split_summary.get('split_method')}`",
                f"- train rows: {split_summary.get('n_train')}",
                f"- validation rows: {split_summary.get('n_validation')}",
                f"- official test rows: {split_summary.get('n_test')}",
                "- validation is carved only from official train rows",
                "- final metrics are evaluated on official test rows",
            ]
        )
    lines.extend(
        [
            "",
            "## Artefact Policy",
            "",
            f"- full predictions written: {'yes' if full_predictions_written else 'no'}",
            f"- checkpoint written: {'yes' if checkpoint_written else 'no'}",
            "",
            "## Boundaries",
            "",
            "- No self-supervised model is run here.",
            "- No trading or deployment claim is made.",
        ]
    )
    if error:
        lines.extend(["", "## Failure", "", f"- {error}"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model_card.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plan_with_output_paths(
    *,
    plan: Sequence[NeuralBenchmarkRunPlan],
    out_dir: Path,
    checkpoint_filename: str,
    checkpoint_enabled: bool,
) -> tuple[NeuralBenchmarkRunPlan, ...]:
    adjusted: list[NeuralBenchmarkRunPlan] = []
    for item in plan:
        checkpoint = _checkpoint_path(
            out_dir,
            item,
            filename=checkpoint_filename,
            enabled=checkpoint_enabled,
        )
        adjusted.append(
            item.model_copy(
                update={
                    "output_dir": str(_run_dir(out_dir, item)),
                    "checkpoint_path": None if checkpoint is None else str(checkpoint),
                }
            )
        )
    return tuple(adjusted)


def _execution_mode(
    *,
    config: NeuralBenchmarkConfig,
    selected_folds: Sequence[str],
    selected_models: Sequence[str],
    selected_seeds: Sequence[int],
    selected_lookbacks: Sequence[int],
    max_epochs: int,
) -> tuple[str, bool]:
    full_grid = (
        tuple(selected_folds) == tuple(config.folds)
        and tuple(selected_models) == tuple(config.enabled_model_names)
        and tuple(selected_seeds) == tuple(config.seeds)
        and tuple(selected_lookbacks) == tuple(config.lookbacks)
        and int(max_epochs) == int(config.training.max_epochs)
    )
    if full_grid and config.is_benchmark_mode:
        return "benchmark", True
    return "smoke", False


def _build_environment_payload() -> dict[str, str]:
    import platform

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_version": __version__,
    }


def run_fi2010_neural_benchmark(
    config_path: str | Path,
    *,
    processed_root: str | Path,
    out_dir: str | Path,
    folds: Sequence[str | int] | str | None = None,
    models: Sequence[str] | str | None = None,
    seeds: Sequence[int] | str | None = None,
    lookbacks: Sequence[int] | str | None = None,
    max_epochs: int | None = None,
    overwrite: bool = False,
    fail_fast: bool = False,
    write_full_predictions: bool | None = None,
    write_checkpoints: bool | None = None,
    allow_full_benchmark: bool = False,
) -> FI2010NeuralBenchmarkSummary:
    """Run selected supervised neural FI-2010 benchmark configurations."""
    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"neural benchmark config not found: {resolved_config_path}")
    config = load_neural_benchmark_config(resolved_config_path)

    selected_folds = _normalise_folds(folds, config=config)
    selected_models = _normalise_model_selection(models, config=config)
    selected_seeds = _normalise_int_selection(
        seeds,
        configured=config.seeds,
        field_name="seeds",
    )
    selected_lookbacks = _normalise_int_selection(
        lookbacks,
        configured=config.lookbacks,
        field_name="lookbacks",
    )
    resolved_max_epochs = (
        config.training.max_epochs if max_epochs is None else int(max_epochs)
    )
    if resolved_max_epochs <= 0:
        raise ValueError("max_epochs must be positive")

    execution_mode, full_grid = _execution_mode(
        config=config,
        selected_folds=selected_folds,
        selected_models=selected_models,
        selected_seeds=selected_seeds,
        selected_lookbacks=selected_lookbacks,
        max_epochs=resolved_max_epochs,
    )
    if full_grid and not allow_full_benchmark:
        raise ValueError(
            "refusing to run the full benchmark grid without allow_full_benchmark=True",
        )

    resolved_out_dir = Path(out_dir)
    _ensure_output_dir(resolved_out_dir, overwrite=overwrite)
    resolved_processed = Path(processed_root)

    run_config = config.model_copy(
        deep=True,
        update={
            "seeds": selected_seeds,
            "training": config.training.model_copy(
                update={"max_epochs": resolved_max_epochs}
            ),
            "artefacts": config.artefacts.model_copy(
                update={
                    "output_root": str(resolved_out_dir / "runs"),
                    "checkpoint_root": str(resolved_out_dir / "checkpoints"),
                }
            ),
        },
    )
    raw_plan = generate_neural_run_plan(
        run_config,
        folds=selected_folds,
        models=selected_models,
        lookbacks=selected_lookbacks,
    )
    prediction_enabled = (
        config.artefacts.write_full_predictions_by_default
        if write_full_predictions is None
        else bool(write_full_predictions)
    )
    checkpoint_enabled = (
        config.checkpoint_policy.enabled
        and config.checkpoint_policy.write_by_default
        and config.artefacts.write_checkpoints_by_default
        if write_checkpoints is None
        else bool(write_checkpoints)
    )
    plan = _plan_with_output_paths(
        plan=raw_plan,
        out_dir=resolved_out_dir,
        checkpoint_filename=config.checkpoint_policy.filename,
        checkpoint_enabled=checkpoint_enabled,
    )

    run_plan_rows = [item.to_csv_row() for item in plan]
    _write_csv(run_plan_rows, resolved_out_dir / "run_plan.csv")

    created_at = datetime.now(UTC)
    result_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    folds_completed: list[str] = []
    checkpoints_written = False
    predictions_written = False

    for item in plan:
        fold_input = _FoldInput(
            fold_id=item.fold_id,
            path=_fold_csv_path(resolved_processed, item.fold_id),
            sha256=None,
        )
        run_directory = Path(item.output_dir)
        split_summary: Mapping[str, Any] | None = None
        try:
            if not fold_input.path.is_file():
                raise FileNotFoundError(
                    f"{item.fold_id} processed CSV is missing: {fold_input.path}"
                )
            fold_input = _FoldInput(
                fold_id=item.fold_id,
                path=fold_input.path,
                sha256=sha256_file(fold_input.path),
            )
            frame = pd.read_csv(fold_input.path)
            checkpoint = None if item.checkpoint_path is None else Path(item.checkpoint_path)
            settings = _neural_settings(
                config=config,
                plan=item,
                checkpoint_path=checkpoint,
                checkpoint_enabled=checkpoint_enabled,
            )
            context = _run_context(
                frame=frame,
                config=config,
                fold_input=fold_input,
                plan=item,
                out_dir=resolved_out_dir,
                settings=settings,
            )
            split_summary = context.split_summary
            result = run_neural_paper_model(
                model_name=item.model_name,
                frame=frame,
                config=context.benchmark_config,
                data_path=fold_input.path,
                split=context.split,
                feature_columns=context.feature_columns,
                all_labels=context.all_labels,
                class_count_train=context.class_count_train,
                class_count_test=context.class_count_test,
                test_timestamps=context.test_timestamps,
                settings=settings,
            )
            result_row = _result_row(
                plan=item,
                metrics=result.metrics,
                split_summary=context.split_summary,
                status="ok",
            )
            training_row = _training_row(
                plan=item,
                metadata=result.metadata,
                status="ok",
            )
            result_rows.append(result_row)
            training_rows.append(training_row)
            prediction_file_written = False
            if prediction_enabled:
                prediction_file_written = _write_predictions(
                    rows=result.prediction_rows,
                    run_dir=run_directory,
                )
                predictions_written = predictions_written or prediction_file_written
            checkpoint_written = bool(
                item.checkpoint_path is not None and Path(item.checkpoint_path).is_file()
            )
            checkpoints_written = checkpoints_written or checkpoint_written
            _write_run_result(
                run_dir=run_directory,
                payload={
                    "run_id": item.run_id,
                    "status": "ok",
                    "fold_csv": str(fold_input.path),
                    "fold_csv_sha256": fold_input.sha256,
                    "result": result_row,
                    "training": training_row,
                    "split_summary": context.split_summary,
                    "feature_count": len(context.feature_columns),
                    "model_metadata": result.metadata,
                    "full_predictions_written": prediction_file_written,
                    "checkpoint_written": checkpoint_written,
                    "runner_version": FI2010_NEURAL_RUNNER_VERSION,
                },
            )
            _write_model_card(
                run_dir=run_directory,
                plan=item,
                status="ok",
                split_summary=context.split_summary,
                full_predictions_written=prediction_file_written,
                checkpoint_written=checkpoint_written,
            )
            if item.fold_id not in folds_completed:
                folds_completed.append(item.fold_id)
        except (
            FileNotFoundError,
            ImportError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            error = f"{type(exc).__name__}: {exc}"
            failure = {
                "run_id": item.run_id,
                "fold_id": item.fold_id,
                "seed": item.seed,
                "model_name": item.model_name,
                "lookback": item.lookback,
                "error": error,
            }
            failures.append(failure)
            result_rows.append(
                _failure_result_row(
                    plan=item,
                    split_summary=split_summary,
                    error=error,
                )
            )
            training_rows.append(_failure_training_row(plan=item, error=error))
            _write_run_result(
                run_dir=run_directory,
                payload={
                    "run_id": item.run_id,
                    "status": "failed",
                    "result": result_rows[-1],
                    "training": training_rows[-1],
                    "split_summary": split_summary,
                    "error": error,
                    "runner_version": FI2010_NEURAL_RUNNER_VERSION,
                },
            )
            _write_model_card(
                run_dir=run_directory,
                plan=item,
                status="failed",
                split_summary=split_summary,
                full_predictions_written=False,
                checkpoint_written=False,
                error=error,
            )
            if fail_fast:
                break

    _write_csv(
        result_rows,
        resolved_out_dir / "results_by_fold_seed.csv",
        RESULTS_BY_FOLD_SEED_COLUMNS,
    )
    result_summary_rows = _build_results_summary(result_rows)
    _write_csv(
        result_summary_rows,
        resolved_out_dir / "results_summary.csv",
        RESULTS_SUMMARY_COLUMNS,
    )
    _write_csv(
        training_rows,
        resolved_out_dir / "training_summary.csv",
        TRAINING_SUMMARY_COLUMNS,
    )
    capacity_rows = _build_capacity_summary(training_rows)
    _write_csv(
        capacity_rows,
        resolved_out_dir / "model_capacity_summary.csv",
        MODEL_CAPACITY_SUMMARY_COLUMNS,
    )
    (resolved_out_dir / "model_failures.json").write_text(
        stable_json_dumps({"failure_count": len(failures), "failures": failures}),
        encoding="utf-8",
    )

    artefacts = {
        "summary": "summary.json",
        "run_plan": "run_plan.csv",
        "results_by_fold_seed": "results_by_fold_seed.csv",
        "results_summary": "results_summary.csv",
        "training_summary": "training_summary.csv",
        "model_capacity_summary": "model_capacity_summary.csv",
        "model_failures": "model_failures.json",
        "runs": "runs/",
    }
    summary = FI2010NeuralBenchmarkSummary(
        study_name=config.study_name,
        dataset_name=config.dataset.name,
        task_name="midprice_direction",
        target_horizon=config.target.horizon,
        config_path=str(resolved_config_path),
        processed_root=str(resolved_processed),
        output_dir=str(resolved_out_dir),
        folds_requested=list(selected_folds),
        folds_completed=folds_completed,
        models_requested=list(selected_models),
        seeds=list(selected_seeds),
        lookbacks=list(selected_lookbacks),
        max_epochs=resolved_max_epochs,
        run_count=len(plan),
        completed_run_count=sum(1 for row in result_rows if row["status"] == "ok"),
        failure_count=len(failures),
        execution_mode=execution_mode,
        full_benchmark_grid=full_grid,
        full_predictions_written=predictions_written,
        checkpoints_written=checkpoints_written,
        artefacts=artefacts,
        runner_version=FI2010_NEURAL_RUNNER_VERSION,
        created_at=created_at,
        git_commit=get_git_commit(),
        warnings=[],
    )
    summary_payload = summary.model_dump(mode="json")
    summary_payload["config_sha256"] = sha256_file(resolved_config_path)
    summary_payload["environment"] = _build_environment_payload()
    summary_payload["config_mode"] = config.mode
    summary_payload["checkpoint_policy"] = {
        "enabled_for_this_run": checkpoint_enabled,
        "write_by_default": config.artefacts.write_checkpoints_by_default,
    }
    summary_payload["prediction_policy"] = {
        "enabled_for_this_run": prediction_enabled,
        "write_by_default": config.artefacts.write_full_predictions_by_default,
    }
    (resolved_out_dir / "summary.json").write_text(
        stable_json_dumps(summary_payload),
        encoding="utf-8",
    )
    return summary
