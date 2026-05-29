"""FI-2010 self-supervised pretraining and fine-tuning benchmark runner.

This runner adds a genuine ``ssl_transformer`` path on top of the leakage-safe
FI-2010 evaluation. For each selected fold, seed and lookback it:

1. Builds the official, leakage-safe train/validation/test split and a
   train-only standardised feature matrix.
2. Pretrains a transformer encoder on **training rows only** with a masked-field
   and/or next-field self-supervised objective, reporting the pretraining loss
   and a *train-carved* validation pretraining loss.
3. Saves the pretrained encoder checkpoint, a config snapshot, a metrics JSON,
   the git commit hash and SHA256 hashes for the key artefacts.
4. Fine-tunes a supervised matrix transformer initialised from the pretrained
   encoder on mid-price direction.
5. Trains a supervised baseline of identical architecture (random init) on the
   same folds, horizons, seeds and preprocessing, and records both results plus
   the comparison.

Pretraining never consults validation or test labels, scalers or statistics
beyond the train-carved validation pretraining loss. The official split-aware
test evaluation is preserved unchanged. No trading, broker or live-execution
functionality is added.
"""

from __future__ import annotations

import math
import platform
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
from chronoslob.experiments.neural_adapters import (
    effective_matrix_window_length,
    run_matrix_transformer_finetune,
)
from chronoslob.experiments.neural_benchmarking import (
    NeuralBenchmarkConfig,
    load_neural_benchmark_config,
    normalise_neural_fold_ids,
)
from chronoslob.models.matrix_ssl import MatrixSSLConfig, create_matrix_ssl_model
from chronoslob.models.preprocessing import (
    TrainOnlyStandardScaler,
    build_target_vector,
    select_feature_columns,
)
from chronoslob.training.experiment import get_git_commit
from chronoslob.training.matrix_ssl_datasets import (
    MatrixSSLWindowDataset,
    MatrixSSLWindowSample,
    bucketise_matrix,
    build_contiguous_windows,
    collate_matrix_ssl_windows,
    fit_feature_bucket_edges,
)
from chronoslob.training.matrix_ssl_experiment import (
    MatrixSSLTrainingConfig,
    fit_matrix_ssl,
    load_pretrained_encoder_state,
    save_pretrained_encoder,
)
from chronoslob.training.splitters import SplitIndices

__all__ = [
    "FI2010_SSL_RUNNER_VERSION",
    "SSL_OBJECTIVE_CHOICES",
    "FI2010SSLBenchmarkSummary",
    "run_fi2010_ssl_neural_benchmark",
]

FI2010_SSL_RUNNER_VERSION = "fi2010-ssl-runner/v1"

SSL_OBJECTIVE_CHOICES: tuple[str, ...] = ("masked_field", "next_field", "both")

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)

_COMBINED_CSV_FILENAME_TEMPLATE = "fold{fold}_combined.csv"

SSL_FINETUNE_MODEL_NAME = "ssl_transformer"
SUPERVISED_BASELINE_MODEL_NAME = "supervised_transformer"

RESULTS_BY_FOLD_SEED_COLUMNS: tuple[str, ...] = (
    "fold_id",
    "seed",
    "model_name",
    "lookback",
    "split",
    "init_source",
    "accuracy",
    "macro_f1",
    "mcc",
    "brier_score",
    "ece",
    "mean_confidence",
    "n_train",
    "n_validation",
    "n_test",
    "status",
    "error",
)

PRETRAINING_SUMMARY_COLUMNS: tuple[str, ...] = (
    "fold_id",
    "seed",
    "lookback",
    "objective",
    "mask_probability",
    "next_field_bucket_count",
    "effective_window_length",
    "pretrain_epochs",
    "pretrain_train_windows",
    "pretrain_validation_windows",
    "final_pretrain_train_loss",
    "final_pretrain_validation_loss",
    "encoder_parameter_count",
    "checkpoint_sha256",
    "status",
    "error",
)

COMPARISON_SUMMARY_COLUMNS: tuple[str, ...] = (
    "fold_id",
    "seed",
    "lookback",
    "ssl_macro_f1",
    "supervised_macro_f1",
    "macro_f1_delta",
    "ssl_accuracy",
    "supervised_accuracy",
    "accuracy_delta",
    "status",
)


class FI2010SSLBenchmarkSummary(BaseModel):
    """Top-level return value from the FI-2010 SSL benchmark runner."""

    model_config = _MODEL_CONFIG

    study_name: str
    dataset_name: str
    task_name: str
    target_horizon: int
    config_path: str
    processed_root: str
    output_dir: str
    objective: str
    folds_requested: list[str]
    folds_completed: list[str]
    seeds: list[int]
    lookbacks: list[int]
    max_epochs: int
    pretrain_epochs: int
    run_count: int
    completed_run_count: int
    failure_count: int
    execution_mode: str
    write_full_predictions: bool
    ssl_artefacts_written: bool
    artefacts: dict[str, str]
    runner_version: str
    created_at: datetime
    git_commit: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("execution_mode")
    @classmethod
    def _validate_execution_mode(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in {"smoke", "benchmark"}:
            raise ValueError("execution_mode must be smoke or benchmark")
        return cleaned


@dataclass(frozen=True)
class _SSLObjectiveFlags:
    enable_masked_field: bool
    enable_next_field: bool


def _objective_flags(objective: str) -> _SSLObjectiveFlags:
    cleaned = objective.strip().lower()
    if cleaned == "masked_field":
        return _SSLObjectiveFlags(enable_masked_field=True, enable_next_field=False)
    if cleaned == "next_field":
        return _SSLObjectiveFlags(enable_masked_field=False, enable_next_field=True)
    if cleaned == "both":
        return _SSLObjectiveFlags(enable_masked_field=True, enable_next_field=True)
    raise ValueError(
        f"unsupported SSL objective {objective!r}; supported: "
        f"{list(SSL_OBJECTIVE_CHOICES)}"
    )


def _fold_number(fold_id: str) -> int:
    return int(fold_id.removeprefix("fold_"))


def _fold_csv_path(processed_root: Path, fold_id: str) -> Path:
    return processed_root / _COMBINED_CSV_FILENAME_TEMPLATE.replace(
        "{fold}",
        str(_fold_number(fold_id)),
    )


def _normalise_int_selection(
    value: Sequence[int] | str | None,
    *,
    configured: Sequence[int],
    field_name: str,
    positive: bool,
) -> tuple[int, ...]:
    if value is None:
        return tuple(configured)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.casefold() == "all":
            return tuple(configured)
        tokens: Sequence[str] = [token.strip() for token in text.split(",")]
        raw: list[int] = []
        for token in tokens:
            if not token:
                continue
            try:
                raw.append(int(token))
            except ValueError as exc:
                raise ValueError(f"{field_name} entries must be integers") from exc
    else:
        raw = [int(item) for item in value]
    cleaned: list[int] = []
    for item in raw:
        if item < 0:
            raise ValueError(f"{field_name} entries must be non-negative")
        if positive and item <= 0:
            raise ValueError(f"{field_name} entries must be positive")
        if item not in cleaned:
            cleaned.append(item)
    if not cleaned:
        raise ValueError(f"{field_name} selection must not be empty")
    return tuple(cleaned)


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


def _benchmark_config(
    *,
    config: NeuralBenchmarkConfig,
    seed: int,
    fold_path: Path,
    out_dir: Path,
) -> FI2010BenchmarkConfig:
    return FI2010BenchmarkConfig(
        experiment_name=f"{config.study_name}_ssl",
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
    candidate = frame.drop(
        columns=[column for column in excluded if column in frame.columns]
    )
    columns = select_feature_columns(candidate, reject_label_like=True)
    if not columns:
        raise ValueError("SSL runner requires at least one feature column")
    return columns


def _standardised_matrix(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    train_indices: Sequence[int],
) -> Any:
    values = frame.loc[:, list(feature_columns)].to_numpy(dtype=float, copy=True)
    train_values = values[list(train_indices), :]
    scaler = TrainOnlyStandardScaler()
    scaler.fit(train_values)
    return scaler.transform(values)


def _windows_for_partition(
    *,
    n_rows: int,
    window_length: int,
    allowed_indices: Sequence[int],
) -> list[MatrixSSLWindowSample]:
    return build_contiguous_windows(
        n_rows=n_rows,
        window_length=window_length,
        allowed_indices=allowed_indices,
    )


def _matrix_ssl_config(
    *,
    config: NeuralBenchmarkConfig,
    input_features: int,
    window_length: int,
    flags: _SSLObjectiveFlags,
    mask_probability: float,
    bucket_count: int,
) -> MatrixSSLConfig:
    spec = config.neural_models["matrix_transformer"]
    dropout = (
        config.training.dropout if spec.dropout is None else float(spec.dropout)
    )
    return MatrixSSLConfig(
        input_features=input_features,
        model_dim=int(spec.model_dim or 16),
        num_heads=int(spec.num_heads or 2),
        num_layers=int(spec.num_layers or 1),
        feedforward_dim=int(spec.feedforward_dim or 32),
        dropout=dropout,
        max_sequence_length=window_length,
        enable_masked_field=flags.enable_masked_field,
        enable_next_field=flags.enable_next_field,
        mask_probability=mask_probability,
        next_field_bucket_count=bucket_count,
    )


def _finetune_settings(
    *,
    config: NeuralBenchmarkConfig,
    lookback: int,
    max_epochs: int,
    batch_size: int,
    device: str,
) -> PaperNeuralSettings:
    spec = config.neural_models["matrix_transformer"]
    dropout = (
        config.training.dropout if spec.dropout is None else float(spec.dropout)
    )
    return PaperNeuralSettings(
        supported_models=("deeplob_style", "matrix_transformer"),
        planned_models=("ssl_transformer",),
        lookback=lookback,
        transformer_window_length=lookback,
        batch_size=batch_size,
        max_epochs=max_epochs,
        early_stopping_patience=config.training.early_stopping_patience,
        early_stopping_metric=config.training.early_stopping_metric,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        gradient_clip_norm=config.training.gradient_clip_norm,
        device=device,
        deterministic=config.deterministic_seed_handling.enabled,
        dropout=dropout,
        transformer_model_dim=int(spec.model_dim or 16),
        transformer_num_heads=int(spec.num_heads or 2),
        transformer_num_layers=int(spec.num_layers or 1),
        transformer_feedforward_dim=int(spec.feedforward_dim or 32),
    )


@dataclass(frozen=True)
class _RunInputs:
    frame: pd.DataFrame
    config: FI2010BenchmarkConfig
    split: SplitIndices
    split_summary: dict[str, Any]
    feature_columns: list[str]
    all_labels: list[Any]
    class_count_train: int
    class_count_test: int
    effective_window: int


def _prepare_run_inputs(
    *,
    frame: pd.DataFrame,
    config: NeuralBenchmarkConfig,
    fold_path: Path,
    seed: int,
    lookback: int,
    out_dir: Path,
) -> _RunInputs:
    benchmark_config = _benchmark_config(
        config=config,
        seed=seed,
        fold_path=fold_path,
        out_dir=out_dir,
    )
    split = build_benchmark_split(benchmark_config, frame)
    split_summary = effective_split_summary(
        config=benchmark_config,
        frame=frame,
        indices=split,
    ).model_dump(mode="json")
    feature_columns = _feature_columns(frame, config)
    target = build_target_vector(frame, target_column=config.target.label_column)
    all_labels = list(target.classes or [])

    def _class_count(rows: Sequence[int]) -> int:
        label_column = config.target.label_column
        return len({frame.iloc[int(index)][label_column] for index in rows})

    effective_window = effective_matrix_window_length(
        configured_window_length=lookback,
        split=split,
    )
    return _RunInputs(
        frame=frame,
        config=benchmark_config,
        split=split,
        split_summary=split_summary,
        feature_columns=feature_columns,
        all_labels=all_labels,
        class_count_train=_class_count(split.train),
        class_count_test=_class_count(split.test),
        effective_window=effective_window,
    )


def _pretrain_and_checkpoint(
    *,
    inputs: _RunInputs,
    config: NeuralBenchmarkConfig,
    seed: int,
    lookback: int,
    flags: _SSLObjectiveFlags,
    mask_probability: float,
    bucket_count: int,
    pretrain_epochs: int,
    batch_size: int,
    device: str,
    pretrain_dir: Path,
    git_commit: str | None,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    standardised = _standardised_matrix(
        inputs.frame,
        feature_columns=inputs.feature_columns,
        train_indices=inputs.split.train,
    )
    n_rows = int(standardised.shape[0])
    input_features = int(standardised.shape[1])

    train_windows = _windows_for_partition(
        n_rows=n_rows,
        window_length=inputs.effective_window,
        allowed_indices=inputs.split.train,
    )
    if not train_windows:
        raise ValueError(
            "no SSL training windows could be built from the training split"
        )
    validation_windows = _windows_for_partition(
        n_rows=n_rows,
        window_length=inputs.effective_window,
        allowed_indices=inputs.split.validation,
    )

    bucket_matrix = None
    feature_edges: list[list[float]] | None = None
    if flags.enable_next_field:
        feature_edges = fit_feature_bucket_edges(
            standardised,
            train_indices=inputs.split.train,
            bucket_count=bucket_count,
        )
        bucket_matrix = bucketise_matrix(
            standardised,
            feature_edges=feature_edges,
            bucket_count=bucket_count,
        )

    def _build_dataset(
        windows: Sequence[MatrixSSLWindowSample],
    ) -> MatrixSSLWindowDataset:
        return MatrixSSLWindowDataset(
            standardised,
            windows,
            bucket_matrix=bucket_matrix,
            mask_probability=mask_probability,
            mask_value=0.0,
            bucket_count=bucket_count,
            ignore_index=-100,
            enable_masked_field=flags.enable_masked_field,
            enable_next_field=flags.enable_next_field,
            base_seed=seed,
        )

    train_loader = DataLoader(
        _build_dataset(train_windows),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_matrix_ssl_windows,
    )
    validation_loader = None
    if validation_windows:
        validation_loader = DataLoader(
            _build_dataset(validation_windows),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_matrix_ssl_windows,
        )

    ssl_config = _matrix_ssl_config(
        config=config,
        input_features=input_features,
        window_length=inputs.effective_window,
        flags=flags,
        mask_probability=mask_probability,
        bucket_count=bucket_count,
    )
    model = create_matrix_ssl_model(ssl_config)
    history = fit_matrix_ssl(
        model,
        train_loader,
        validation_loader,
        MatrixSSLTrainingConfig(
            epochs=pretrain_epochs,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            gradient_clip_norm=config.training.gradient_clip_norm,
            device=device,
            seed=seed,
        ),
    )

    final = history[-1]
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = pretrain_dir / "pretrained_encoder.pt"
    save_pretrained_encoder(
        model,
        encoder_path,
        metadata={
            "fold_objective": list(ssl_config.enabled_objectives()),
            "seed": int(seed),
            "lookback": int(lookback),
            "effective_window": int(inputs.effective_window),
        },
    )

    pretrain_config_payload = {
        "runner_version": FI2010_SSL_RUNNER_VERSION,
        "objective": list(ssl_config.enabled_objectives()),
        "mask_probability": float(mask_probability),
        "next_field_bucket_count": int(bucket_count),
        "effective_window_length": int(inputs.effective_window),
        "input_features": int(input_features),
        "feature_columns": list(inputs.feature_columns),
        "architecture": {
            "model_dim": int(ssl_config.model_dim),
            "num_heads": int(ssl_config.num_heads),
            "num_layers": int(ssl_config.num_layers),
            "feedforward_dim": int(ssl_config.feedforward_dim),
            "dropout": float(ssl_config.dropout),
            "max_sequence_length": int(ssl_config.max_sequence_length),
        },
        "split_summary": inputs.split_summary,
        "pretraining_data_policy": (
            "training rows only; validation pretraining loss is train-carved; "
            "no test rows, labels or scalers are consulted"
        ),
        "label_leakage": "none; SSL objectives use unlabelled standardised features",
    }
    pretrain_config_path = pretrain_dir / "pretrain_config.json"
    pretrain_config_path.write_text(
        stable_json_dumps(pretrain_config_payload),
        encoding="utf-8",
    )

    metrics_payload = {
        "runner_version": FI2010_SSL_RUNNER_VERSION,
        "git_commit": git_commit,
        "objective": list(ssl_config.enabled_objectives()),
        "pretrain_epochs": int(pretrain_epochs),
        "train_window_count": len(train_windows),
        "validation_window_count": len(validation_windows),
        "encoder_parameter_count": int(model.n_parameters()),
        "history": [
            {
                "epoch": int(item.epoch),
                "train_loss": float(item.train_loss),
                "train_loss_components": dict(item.train_loss_components),
                "validation_loss": (
                    None
                    if item.validation_loss is None
                    else float(item.validation_loss)
                ),
                "validation_loss_components": dict(
                    item.validation_loss_components
                ),
            }
            for item in history
        ],
        "final_train_loss": float(final.train_loss),
        "final_validation_loss": (
            None if final.validation_loss is None else float(final.validation_loss)
        ),
    }
    pretrain_metrics_path = pretrain_dir / "pretrain_metrics.json"
    pretrain_metrics_path.write_text(
        stable_json_dumps(metrics_payload),
        encoding="utf-8",
    )

    artefact_files = {
        "pretrained_encoder.pt": encoder_path,
        "pretrain_config.json": pretrain_config_path,
        "pretrain_metrics.json": pretrain_metrics_path,
    }
    artefact_hashes = {
        name: sha256_file(path) for name, path in artefact_files.items()
    }
    manifest_payload = {
        "runner_version": FI2010_SSL_RUNNER_VERSION,
        "checkpoint_version": "fi2010-matrix-ssl/encoder/v1",
        "git_commit": git_commit,
        "objective": list(ssl_config.enabled_objectives()),
        "encoder_parameter_count": int(model.n_parameters()),
        "sha256": artefact_hashes,
        "artefacts": {name: path.name for name, path in artefact_files.items()},
    }
    manifest_path = pretrain_dir / "artefact_manifest.json"
    manifest_path.write_text(stable_json_dumps(manifest_payload), encoding="utf-8")

    encoder_state = load_pretrained_encoder_state(encoder_path)
    return {
        "encoder_state": encoder_state,
        "encoder_path": encoder_path,
        "checkpoint_sha256": artefact_hashes["pretrained_encoder.pt"],
        "train_window_count": len(train_windows),
        "validation_window_count": len(validation_windows),
        "final_train_loss": float(final.train_loss),
        "final_validation_loss": (
            None if final.validation_loss is None else float(final.validation_loss)
        ),
        "encoder_parameter_count": int(model.n_parameters()),
        "objective": list(ssl_config.enabled_objectives()),
        "manifest_path": manifest_path,
    }


def _result_row(
    *,
    fold_id: str,
    seed: int,
    lookback: int,
    model_name: str,
    init_source: str,
    metrics: Mapping[str, Any],
    split_summary: Mapping[str, Any],
    status: str,
    error: str = "",
) -> dict[str, Any]:
    return {
        "fold_id": fold_id,
        "seed": seed,
        "model_name": model_name,
        "lookback": lookback,
        "split": "test",
        "init_source": init_source,
        "accuracy": _safe_float(metrics.get("accuracy")),
        "macro_f1": _safe_float(metrics.get("macro_f1")),
        "mcc": _safe_float(metrics.get("matthews_corrcoef")),
        "brier_score": _safe_float(metrics.get("brier_score")),
        "ece": _safe_float(
            metrics.get("ece", metrics.get("expected_calibration_error"))
        ),
        "mean_confidence": _safe_float(metrics.get("mean_confidence")),
        "n_train": int(split_summary.get("n_train", 0)),
        "n_validation": int(split_summary.get("n_validation", 0)),
        "n_test": int(split_summary.get("n_test", 0)),
        "status": status,
        "error": error,
    }


def _write_predictions(rows: Sequence[Mapping[str, Any]], path: Path) -> bool:
    if not rows:
        return False
    _write_csv(rows, path)
    return True


def run_fi2010_ssl_neural_benchmark(
    config_path: str | Path,
    *,
    processed_root: str | Path,
    out_dir: str | Path,
    folds: Sequence[str | int] | str | None = None,
    seeds: Sequence[int] | str | None = None,
    lookbacks: Sequence[int] | str | None = None,
    objective: str = "masked_field",
    mask_probability: float = 0.15,
    next_field_bucket_count: int = 3,
    pretrain_epochs: int = 1,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    device: str = "cpu",
    overwrite: bool = False,
    fail_fast: bool = False,
    write_full_predictions: bool = True,
) -> FI2010SSLBenchmarkSummary:
    """Run the FI-2010 SSL pretraining and fine-tuning benchmark."""
    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_file():
        raise FileNotFoundError(f"SSL benchmark config not found: {resolved_config_path}")
    config = load_neural_benchmark_config(resolved_config_path)
    if "matrix_transformer" not in config.enabled_model_names:
        raise ValueError(
            "the SSL runner requires the 'matrix_transformer' model to be enabled "
            "in the config so the fine-tuned and baseline architectures match"
        )

    flags = _objective_flags(objective)
    if isinstance(mask_probability, bool) or not isinstance(
        mask_probability, (int, float)
    ):
        raise ValueError("mask_probability must be a finite number")
    mask_probability = float(mask_probability)
    if not 0.0 <= mask_probability <= 1.0:
        raise ValueError("mask_probability must satisfy 0 <= mask_probability <= 1")
    if flags.enable_masked_field and mask_probability <= 0.0:
        raise ValueError(
            "mask_probability must be positive when the masked-field objective is "
            "selected"
        )
    if isinstance(next_field_bucket_count, bool) or not isinstance(
        next_field_bucket_count, int
    ):
        raise ValueError("next_field_bucket_count must be an integer")
    if next_field_bucket_count < 2:
        raise ValueError("next_field_bucket_count must be >= 2")
    if isinstance(pretrain_epochs, bool) or not isinstance(pretrain_epochs, int):
        raise ValueError("pretrain_epochs must be an integer")
    if pretrain_epochs <= 0:
        raise ValueError("pretrain_epochs must be positive")

    selected_folds = normalise_neural_fold_ids(folds, config=config)
    selected_seeds = _normalise_int_selection(
        seeds,
        configured=config.seeds,
        field_name="seeds",
        positive=False,
    )
    selected_lookbacks = _normalise_int_selection(
        lookbacks,
        configured=config.lookbacks,
        field_name="lookbacks",
        positive=True,
    )
    resolved_max_epochs = (
        config.training.max_epochs if max_epochs is None else int(max_epochs)
    )
    if resolved_max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    resolved_batch_size = (
        config.training.batch_size if batch_size is None else int(batch_size)
    )
    if resolved_batch_size <= 0:
        raise ValueError("batch_size must be positive")

    full_grid = (
        tuple(selected_folds) == tuple(config.folds)
        and tuple(selected_seeds) == tuple(config.seeds)
        and tuple(selected_lookbacks) == tuple(config.lookbacks)
        and resolved_max_epochs == int(config.training.max_epochs)
    )
    execution_mode = "benchmark" if (full_grid and config.is_benchmark_mode) else "smoke"

    resolved_out_dir = Path(out_dir)
    _ensure_output_dir(resolved_out_dir, overwrite=overwrite)
    resolved_processed = Path(processed_root)
    git_commit = get_git_commit()

    run_plan_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    pretraining_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    folds_completed: list[str] = []
    ssl_artefacts_written = False
    predictions_written = False
    stop_requested = False

    for fold_id in selected_folds:
        fold_path = _fold_csv_path(resolved_processed, fold_id)
        for seed in selected_seeds:
            for lookback in selected_lookbacks:
                run_id = f"{fold_id}_seed_{seed}_lb{lookback}"
                run_plan_rows.append(
                    {
                        "run_id": run_id,
                        "fold_id": fold_id,
                        "seed": seed,
                        "lookback": lookback,
                        "objective": objective,
                        "pretrain_epochs": pretrain_epochs,
                        "max_epochs": resolved_max_epochs,
                    }
                )
                run_dir = resolved_out_dir / "runs" / run_id
                split_summary: dict[str, Any] = {}
                try:
                    if not fold_path.is_file():
                        raise FileNotFoundError(
                            f"{fold_id} processed CSV is missing: {fold_path}"
                        )
                    frame = pd.read_csv(fold_path)
                    inputs = _prepare_run_inputs(
                        frame=frame,
                        config=config,
                        fold_path=fold_path,
                        seed=seed,
                        lookback=lookback,
                        out_dir=resolved_out_dir,
                    )
                    split_summary = inputs.split_summary
                    pretrain = _pretrain_and_checkpoint(
                        inputs=inputs,
                        config=config,
                        seed=seed,
                        lookback=lookback,
                        flags=flags,
                        mask_probability=mask_probability,
                        bucket_count=next_field_bucket_count,
                        pretrain_epochs=pretrain_epochs,
                        batch_size=resolved_batch_size,
                        device=device,
                        pretrain_dir=run_dir / "pretrain",
                        git_commit=git_commit,
                    )
                    ssl_artefacts_written = True

                    settings = _finetune_settings(
                        config=config,
                        lookback=lookback,
                        max_epochs=resolved_max_epochs,
                        batch_size=resolved_batch_size,
                        device=device,
                    )
                    ssl_result = run_matrix_transformer_finetune(
                        model_name=SSL_FINETUNE_MODEL_NAME,
                        frame=frame,
                        config=inputs.config,
                        split=inputs.split,
                        feature_columns=inputs.feature_columns,
                        all_labels=inputs.all_labels,
                        class_count_train=inputs.class_count_train,
                        class_count_test=inputs.class_count_test,
                        test_timestamps=None,
                        settings=settings,
                        pretrained_encoder_state=pretrain["encoder_state"],
                        init_source="ssl_pretrained",
                        model_type="ssl_finetuned_matrix_transformer",
                    )
                    baseline_result = run_matrix_transformer_finetune(
                        model_name=SUPERVISED_BASELINE_MODEL_NAME,
                        frame=frame,
                        config=inputs.config,
                        split=inputs.split,
                        feature_columns=inputs.feature_columns,
                        all_labels=inputs.all_labels,
                        class_count_train=inputs.class_count_train,
                        class_count_test=inputs.class_count_test,
                        test_timestamps=None,
                        settings=settings,
                        pretrained_encoder_state=None,
                        init_source="random_init",
                        model_type="normalised_matrix_transformer",
                    )

                    ssl_row = _result_row(
                        fold_id=fold_id,
                        seed=seed,
                        lookback=lookback,
                        model_name=SSL_FINETUNE_MODEL_NAME,
                        init_source="ssl_pretrained",
                        metrics=ssl_result.metrics,
                        split_summary=split_summary,
                        status="ok",
                    )
                    baseline_row = _result_row(
                        fold_id=fold_id,
                        seed=seed,
                        lookback=lookback,
                        model_name=SUPERVISED_BASELINE_MODEL_NAME,
                        init_source="random_init",
                        metrics=baseline_result.metrics,
                        split_summary=split_summary,
                        status="ok",
                    )
                    result_rows.extend([ssl_row, baseline_row])

                    if write_full_predictions:
                        ssl_dir = run_dir / SSL_FINETUNE_MODEL_NAME
                        base_dir = run_dir / SUPERVISED_BASELINE_MODEL_NAME
                        ssl_dir.mkdir(parents=True, exist_ok=True)
                        base_dir.mkdir(parents=True, exist_ok=True)
                        wrote_ssl = _write_predictions(
                            ssl_result.prediction_rows, ssl_dir / "predictions.csv"
                        )
                        wrote_base = _write_predictions(
                            baseline_result.prediction_rows,
                            base_dir / "predictions.csv",
                        )
                        predictions_written = (
                            predictions_written or wrote_ssl or wrote_base
                        )

                    ssl_macro = _safe_float(ssl_result.metrics.get("macro_f1"))
                    base_macro = _safe_float(baseline_result.metrics.get("macro_f1"))
                    ssl_acc = _safe_float(ssl_result.metrics.get("accuracy"))
                    base_acc = _safe_float(baseline_result.metrics.get("accuracy"))
                    comparison_rows.append(
                        {
                            "fold_id": fold_id,
                            "seed": seed,
                            "lookback": lookback,
                            "ssl_macro_f1": ssl_macro,
                            "supervised_macro_f1": base_macro,
                            "macro_f1_delta": (
                                None
                                if ssl_macro is None or base_macro is None
                                else ssl_macro - base_macro
                            ),
                            "ssl_accuracy": ssl_acc,
                            "supervised_accuracy": base_acc,
                            "accuracy_delta": (
                                None
                                if ssl_acc is None or base_acc is None
                                else ssl_acc - base_acc
                            ),
                            "status": "ok",
                        }
                    )
                    pretraining_rows.append(
                        {
                            "fold_id": fold_id,
                            "seed": seed,
                            "lookback": lookback,
                            "objective": objective,
                            "mask_probability": mask_probability,
                            "next_field_bucket_count": next_field_bucket_count,
                            "effective_window_length": inputs.effective_window,
                            "pretrain_epochs": pretrain_epochs,
                            "pretrain_train_windows": pretrain["train_window_count"],
                            "pretrain_validation_windows": pretrain[
                                "validation_window_count"
                            ],
                            "final_pretrain_train_loss": pretrain["final_train_loss"],
                            "final_pretrain_validation_loss": pretrain[
                                "final_validation_loss"
                            ],
                            "encoder_parameter_count": pretrain[
                                "encoder_parameter_count"
                            ],
                            "checkpoint_sha256": pretrain["checkpoint_sha256"],
                            "status": "ok",
                            "error": "",
                        }
                    )
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "comparison.json").write_text(
                        stable_json_dumps(
                            {
                                "run_id": run_id,
                                "ssl_transformer": ssl_row,
                                "supervised_transformer": baseline_row,
                                "ssl_init_metadata": ssl_result.metadata.get(
                                    "pretraining", {}
                                ),
                                "pretrain_checkpoint_sha256": pretrain[
                                    "checkpoint_sha256"
                                ],
                            }
                        ),
                        encoding="utf-8",
                    )
                    if fold_id not in folds_completed:
                        folds_completed.append(fold_id)
                except (
                    FileNotFoundError,
                    ImportError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    IndexError,
                ) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    failures.append({"run_id": run_id, "error": error})
                    pretraining_rows.append(
                        {
                            "fold_id": fold_id,
                            "seed": seed,
                            "lookback": lookback,
                            "objective": objective,
                            "mask_probability": mask_probability,
                            "next_field_bucket_count": next_field_bucket_count,
                            "effective_window_length": None,
                            "pretrain_epochs": pretrain_epochs,
                            "pretrain_train_windows": None,
                            "pretrain_validation_windows": None,
                            "final_pretrain_train_loss": None,
                            "final_pretrain_validation_loss": None,
                            "encoder_parameter_count": None,
                            "checkpoint_sha256": None,
                            "status": "failed",
                            "error": error,
                        }
                    )
                    comparison_rows.append(
                        {
                            "fold_id": fold_id,
                            "seed": seed,
                            "lookback": lookback,
                            "ssl_macro_f1": None,
                            "supervised_macro_f1": None,
                            "macro_f1_delta": None,
                            "ssl_accuracy": None,
                            "supervised_accuracy": None,
                            "accuracy_delta": None,
                            "status": "failed",
                        }
                    )
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "comparison.json").write_text(
                        stable_json_dumps({"run_id": run_id, "error": error}),
                        encoding="utf-8",
                    )
                    if fail_fast:
                        stop_requested = True
                        break
            if stop_requested:
                break
        if stop_requested:
            break

    _write_csv(run_plan_rows, resolved_out_dir / "run_plan.csv")
    _write_csv(
        result_rows,
        resolved_out_dir / "results_by_fold_seed.csv",
        RESULTS_BY_FOLD_SEED_COLUMNS,
    )
    results_summary_rows = _build_results_summary(result_rows)
    _write_csv(
        results_summary_rows,
        resolved_out_dir / "results_summary.csv",
        ("model_name", "run_count", "accuracy_mean", "macro_f1_mean", "mcc_mean"),
    )
    _write_csv(
        pretraining_rows,
        resolved_out_dir / "ssl_pretraining_summary.csv",
        PRETRAINING_SUMMARY_COLUMNS,
    )
    _write_csv(
        comparison_rows,
        resolved_out_dir / "comparison_summary.csv",
        COMPARISON_SUMMARY_COLUMNS,
    )
    (resolved_out_dir / "model_failures.json").write_text(
        stable_json_dumps({"failure_count": len(failures), "failures": failures}),
        encoding="utf-8",
    )

    run_count = len(run_plan_rows)
    completed = sum(1 for row in result_rows if row["status"] == "ok") // 2
    artefacts = {
        "summary": "summary.json",
        "run_plan": "run_plan.csv",
        "results_by_fold_seed": "results_by_fold_seed.csv",
        "results_summary": "results_summary.csv",
        "ssl_pretraining_summary": "ssl_pretraining_summary.csv",
        "comparison_summary": "comparison_summary.csv",
        "model_failures": "model_failures.json",
        "runs": "runs/",
    }
    summary = FI2010SSLBenchmarkSummary(
        study_name=config.study_name,
        dataset_name=config.dataset.name,
        task_name="midprice_direction",
        target_horizon=config.target.horizon,
        config_path=str(resolved_config_path),
        processed_root=str(resolved_processed),
        output_dir=str(resolved_out_dir),
        objective=objective,
        folds_requested=list(selected_folds),
        folds_completed=folds_completed,
        seeds=list(selected_seeds),
        lookbacks=list(selected_lookbacks),
        max_epochs=resolved_max_epochs,
        pretrain_epochs=pretrain_epochs,
        run_count=run_count,
        completed_run_count=completed,
        failure_count=len(failures),
        execution_mode=execution_mode,
        write_full_predictions=bool(write_full_predictions),
        ssl_artefacts_written=ssl_artefacts_written,
        artefacts=artefacts,
        runner_version=FI2010_SSL_RUNNER_VERSION,
        created_at=datetime.now(UTC),
        git_commit=git_commit,
        warnings=[],
    )
    summary_payload = summary.model_dump(mode="json")
    summary_payload["config_sha256"] = sha256_file(resolved_config_path)
    summary_payload["predictions_written"] = predictions_written
    summary_payload["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_version": __version__,
    }
    summary_payload["leakage_policy"] = (
        "SSL pretraining uses training rows only; the validation pretraining "
        "loss is carved from training rows; the official split-aware test "
        "evaluation is preserved."
    )
    (resolved_out_dir / "summary.json").write_text(
        stable_json_dumps(summary_payload),
        encoding="utf-8",
    )
    return summary


def _build_results_summary(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return []
    ok = frame[frame["status"] == "ok"]
    if ok.empty:
        return []
    summary_rows: list[dict[str, Any]] = []
    for model_name, group in ok.groupby("model_name", sort=True):
        summary_rows.append(
            {
                "model_name": str(model_name),
                "run_count": len(group),
                "accuracy_mean": _column_mean(group["accuracy"]),
                "macro_f1_mean": _column_mean(group["macro_f1"]),
                "mcc_mean": _column_mean(group["mcc"]),
            }
        )
    return summary_rows


def _column_mean(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    value = float(numeric.mean())
    return value if math.isfinite(value) else None
