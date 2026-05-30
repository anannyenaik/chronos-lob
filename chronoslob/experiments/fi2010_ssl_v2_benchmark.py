"""FI-2010 SSL-v2 benchmark runner.

This runner evaluates the failure-analysis-motivated SSL-v2 objective against a
matched supervised baseline and, when available, the first-generation masked
reconstruction baseline. It keeps the scope explicit and stores conservative
artefacts for downstream analysis and claim audit.
"""

from __future__ import annotations

import platform
import shutil
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from chronoslob import __version__
from chronoslob.experiments.fi2010_neural_grid import (
    write_neural_grid_aggregate_artifacts,
)
from chronoslob.experiments.fi2010_neural_proper_training import (
    FI2010ProperTrainingRunSpec,
)
from chronoslob.experiments.fi2010_neural_proper_training import (
    _best_validation_score as _proper_best_validation_score,
)
from chronoslob.experiments.fi2010_neural_proper_training import (
    _canonical_prediction_rows as _proper_prediction_rows,
)
from chronoslob.experiments.fi2010_neural_proper_training import (
    _curve_rows as _proper_curve_rows,
)
from chronoslob.experiments.fi2010_neural_proper_training import (
    _failure_row as _proper_failure_row,
)
from chronoslob.experiments.fi2010_neural_proper_training import (
    _hash_payload as _proper_hash_payload,
)
from chronoslob.experiments.fi2010_neural_proper_training import (
    _int_or_none as _proper_int_or_none,
)
from chronoslob.experiments.fi2010_neural_proper_training import (
    _read_json as _proper_read_json,
)
from chronoslob.experiments.fi2010_neural_proper_training import (
    _relative_path as _proper_relative_path,
)
from chronoslob.experiments.fi2010_neural_proper_training import (
    _resolve_metrics as _proper_resolve_metrics,
)
from chronoslob.experiments.fi2010_neural_proper_training import (
    _result_row_from_payload as _proper_result_row_from_payload,
)
from chronoslob.experiments.fi2010_neural_proper_training import (
    _safe_float as _proper_safe_float,
)
from chronoslob.experiments.fi2010_neural_proper_training import (
    _training_row_from_payload as _proper_training_row_from_payload,
)
from chronoslob.experiments.fi2010_neural_proper_training import (
    _write_csv as _proper_write_csv,
)
from chronoslob.experiments.fi2010_neural_proper_training import (
    _write_run_log as _proper_write_run_log,
)
from chronoslob.experiments.fi2010_neural_proper_training import (
    _write_run_manifest as _proper_write_run_manifest,
)
from chronoslob.experiments.fi2010_neural_proper_training import (
    _write_status as _proper_write_status,
)
from chronoslob.experiments.fi2010_ssl_runner import (
    _standardised_matrix as _ssl_standardised_matrix,
)
from chronoslob.experiments.fi2010_ssl_runner import (
    build_ssl_finetune_settings,
    prepare_fi2010_run_inputs,
    pretrain_matrix_ssl_encoder,
    ssl_objective_flags,
)
from chronoslob.experiments.manifests import sha256_file, stable_json_dumps
from chronoslob.experiments.neural_adapters import run_matrix_transformer_finetune
from chronoslob.experiments.neural_benchmarking import (
    NeuralBenchmarkConfig,
    NeuralTargetConfig,
    load_neural_benchmark_config,
)
from chronoslob.models.matrix_ssl_v2 import MatrixSSLV2Config, create_matrix_ssl_v2_model
from chronoslob.training.experiment import get_git_commit
from chronoslob.training.matrix_ssl_v2_datasets import (
    MatrixSSLV2WindowDataset,
    MatrixSSLV2WindowSample,
    build_ssl_v2_auxiliary_label_matrix,
    build_ssl_v2_windows,
    collate_matrix_ssl_v2_windows,
    fit_ssl_v2_auxiliary_label_spec,
    infer_ssl_v2_feature_groups,
)
from chronoslob.training.matrix_ssl_v2_experiment import (
    MatrixSSLV2EpochResult,
    MatrixSSLV2TrainingConfig,
    fit_matrix_ssl_v2,
    load_pretrained_ssl_v2_encoder_state,
    save_pretrained_ssl_v2_encoder,
)

__all__ = [
    "FI2010_SSL_V2_BENCHMARK_VERSION",
    "SSL_V2_OBJECTIVE_CHOICES",
    "FI2010SSLV2RunSpec",
    "FI2010SSLV2Summary",
    "expand_ssl_v2_specs",
    "run_fi2010_ssl_v2_benchmark",
]

FI2010_SSL_V2_BENCHMARK_VERSION = "fi2010-ssl-v2-benchmark/v1"

SSL_V2_DEFAULT_OUT_DIR = Path("experiments/fi2010_ssl_v2_benchmark")
SSL_V2_DEFAULT_BASELINE_DIR = Path("experiments/fi2010_neural_proper_training_subset_v2")
SSL_V2_OBJECTIVE_CHOICES: tuple[str, ...] = (
    "supervised",
    "masked_reconstruction",
    "market_state_multitask",
)
SSL_V2_DEFAULT_FOLDS: tuple[int, ...] = (1,)
SSL_V2_DEFAULT_HORIZONS: tuple[int, ...] = (10, 50)
SSL_V2_DEFAULT_SEEDS: tuple[int, ...] = (0,)
SSL_V2_DEFAULT_LOOKBACKS: tuple[int, ...] = (50,)
SSL_V2_COMPLETE_TARGET_FOLDS = {1, 2, 3}
SSL_V2_TARGET_HORIZONS = {10, 50}
SSL_V2_TARGET_SEEDS = {0}
SSL_V2_TARGET_LOOKBACKS = {50}
SSL_V2_TARGET_MAX_EPOCHS = 25
SSL_V2_TARGET_PATIENCE = 5

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)
_SUCCESS_STATUSES = {"completed", "skipped_existing", "imported_existing"}
_COMPARISON_COLUMNS: tuple[str, ...] = (
    "fold",
    "horizon",
    "seed",
    "lookback",
    "model_family",
    "ssl_objective",
    "comparison",
    "supervised_run_id",
    "ssl_run_id",
    "supervised_macro_f1",
    "ssl_macro_f1",
    "delta_macro_f1",
    "supervised_accuracy",
    "ssl_accuracy",
    "delta_accuracy",
    "supervised_mcc",
    "ssl_mcc",
    "delta_mcc",
    "supervised_ece",
    "ssl_ece",
    "delta_ece",
    "supervised_brier_score",
    "ssl_brier_score",
    "delta_brier_score",
    "supervised_nll",
    "ssl_nll",
    "delta_nll",
    "macro_f1_outcome",
    "ece_outcome",
    "mcc_outcome",
    "status",
    "reason",
)
_TRAINING_SUMMARY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "fold",
    "horizon",
    "seed",
    "lookback",
    "objective",
    "pretraining_objective",
    "max_epochs",
    "epochs_ran",
    "best_epoch",
    "monitored_metric",
    "best_validation_score",
    "early_stopping_patience",
    "early_stopped",
    "training_seconds",
    "test_macro_f1",
    "test_accuracy",
    "test_mcc",
    "test_ece",
    "status",
)
_PRETRAINING_CURVE_COLUMNS: tuple[str, ...] = (
    "run_id",
    "fold",
    "horizon",
    "seed",
    "lookback",
    "epoch",
    "train_loss",
    "validation_loss",
    "train_reconstruction",
    "train_future_spread_widening",
    "train_future_volatility",
    "train_future_return",
    "train_future_imbalance",
    "train_contrastive",
    "train_total",
    "validation_reconstruction",
    "validation_future_spread_widening",
    "validation_future_volatility",
    "validation_future_return",
    "validation_future_imbalance",
    "validation_contrastive",
    "validation_total",
)


class FI2010SSLV2Summary(BaseModel):
    """Top-level summary returned by the SSL-v2 benchmark runner."""

    model_config = _MODEL_CONFIG

    output_dir: str
    config_path: str
    processed_root: str
    baseline_source_dir: str | None
    evidence_level: str
    scope_label: str
    folds: list[int]
    horizons: list[int]
    seeds: list[int]
    lookbacks: list[int]
    objectives: list[str]
    pretrain_epochs: int
    max_epochs: int
    early_stopping_patience: int
    batch_size: int
    device: str
    smoke_test: bool
    run_count: int
    completed_run_count: int
    failed_run_count: int
    imported_baseline_count: int
    missing_pair_count: int
    artefacts: dict[str, str]
    runner_version: str
    created_at: datetime
    git_commit: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("evidence_level")
    @classmethod
    def _validate_evidence_level(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in {"complete_real", "partial_real", "smoke_test_only"}:
            raise ValueError(
                "evidence_level must be complete_real, partial_real or "
                "smoke_test_only"
            )
        return cleaned


@dataclass(frozen=True)
class FI2010SSLV2RunSpec:
    """One run in the SSL-v2 benchmark scope."""

    fold: int
    horizon: int
    seed: int
    lookback: int
    objective: str

    @property
    def fold_id(self) -> str:
        return f"fold_{self.fold}"

    @property
    def run_id(self) -> str:
        return (
            f"{self.fold_id}__h{self.horizon}__seed_{self.seed}"
            f"__lb{self.lookback}__{self.objective}"
        )

    @property
    def model_family(self) -> str:
        return "matrix_transformer"

    @property
    def pretraining_objective(self) -> str:
        return "none" if self.objective == "supervised" else self.objective

    def run_dir(self, out_dir: Path) -> Path:
        return (
            Path(out_dir)
            / "runs"
            / self.fold_id
            / f"horizon_{self.horizon}"
            / f"seed_{self.seed}"
            / f"lookback_{self.lookback}"
            / self.objective
        )


def expand_ssl_v2_specs(
    *,
    folds: Sequence[int | str] | str | None = None,
    horizons: Sequence[int] | str | None = None,
    seeds: Sequence[int] | str | None = None,
    lookbacks: Sequence[int] | str | None = None,
    objectives: Sequence[str] | str | None = None,
    smoke_test: bool = False,
) -> tuple[FI2010SSLV2RunSpec, ...]:
    """Expand a deterministic SSL-v2 run grid."""
    selected_folds = _normalise_folds(folds, default=SSL_V2_DEFAULT_FOLDS)
    selected_horizons = _normalise_ints(
        horizons,
        default=SSL_V2_DEFAULT_HORIZONS,
        field_name="horizons",
        positive=True,
    )
    selected_seeds = _normalise_ints(
        seeds,
        default=SSL_V2_DEFAULT_SEEDS,
        field_name="seeds",
        positive=False,
    )
    selected_lookbacks = _normalise_ints(
        lookbacks,
        default=SSL_V2_DEFAULT_LOOKBACKS,
        field_name="lookbacks",
        positive=True,
    )
    selected_objectives = _normalise_objectives(objectives)
    if smoke_test:
        selected_folds = selected_folds[:1]
        selected_horizons = selected_horizons[:1]
        selected_seeds = selected_seeds[:1]
        selected_lookbacks = selected_lookbacks[:1]
    specs: list[FI2010SSLV2RunSpec] = []
    for fold in selected_folds:
        for horizon in selected_horizons:
            for seed in selected_seeds:
                for lookback in selected_lookbacks:
                    for objective in selected_objectives:
                        specs.append(
                            FI2010SSLV2RunSpec(
                                fold=fold,
                                horizon=horizon,
                                seed=seed,
                                lookback=lookback,
                                objective=objective,
                            )
                        )
    return tuple(specs)


def run_fi2010_ssl_v2_benchmark(
    config_path: str | Path = Path("configs/experiments/fi2010_neural_proper_training.yaml"),
    *,
    processed_root: str | Path,
    out_dir: str | Path = SSL_V2_DEFAULT_OUT_DIR,
    baseline_source_dir: str | Path | None = SSL_V2_DEFAULT_BASELINE_DIR,
    folds: Sequence[int | str] | str | None = None,
    horizons: Sequence[int] | str | None = None,
    seeds: Sequence[int] | str | None = None,
    lookbacks: Sequence[int] | str | None = None,
    objectives: Sequence[str] | str | None = None,
    pretrain_epochs: int = 5,
    max_epochs: int | None = None,
    patience: int | None = None,
    batch_size: int | None = None,
    mask_probability: float = 0.30,
    future_bucket_count: int = 3,
    contrastive: bool = False,
    device: str = "cpu",
    reuse_completed: bool = True,
    import_existing_baselines: bool = True,
    smoke_test: bool = False,
) -> FI2010SSLV2Summary:
    """Run and aggregate the FI-2010 SSL-v2 benchmark."""
    resolved_config = Path(config_path)
    if not resolved_config.is_file():
        raise FileNotFoundError(f"SSL-v2 config not found: {resolved_config}")
    base_config = load_neural_benchmark_config(resolved_config)
    if "matrix_transformer" not in base_config.enabled_model_names:
        raise ValueError("SSL-v2 benchmark requires the matrix_transformer model")
    resolved_out = Path(out_dir)
    resolved_out.mkdir(parents=True, exist_ok=True)
    resolved_processed = Path(processed_root)
    resolved_baseline_source = Path(baseline_source_dir) if baseline_source_dir else None

    resolved_max_epochs = (
        int(base_config.training.max_epochs) if max_epochs is None else int(max_epochs)
    )
    resolved_patience = (
        int(base_config.training.early_stopping_patience) if patience is None else int(patience)
    )
    resolved_batch_size = (
        int(base_config.training.batch_size) if batch_size is None else int(batch_size)
    )
    if pretrain_epochs <= 0:
        raise ValueError("pretrain_epochs must be positive")
    if resolved_max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if resolved_patience < 0:
        raise ValueError("patience must be non-negative")
    if resolved_batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if future_bucket_count < 2:
        raise ValueError("future_bucket_count must be >= 2")
    if smoke_test:
        device = "cpu"
        pretrain_epochs = min(int(pretrain_epochs), 1)
        resolved_max_epochs = min(resolved_max_epochs, 1)

    config = base_config.model_copy(
        update={
            "training": base_config.training.model_copy(
                update={
                    "max_epochs": resolved_max_epochs,
                    "early_stopping_patience": resolved_patience,
                    "batch_size": resolved_batch_size,
                }
            )
        }
    )
    specs = expand_ssl_v2_specs(
        folds=folds,
        horizons=horizons,
        seeds=seeds,
        lookbacks=lookbacks,
        objectives=objectives,
        smoke_test=smoke_test,
    )
    if not specs:
        raise ValueError("SSL-v2 selection produced no runs")

    _proper_write_csv(
        [_run_plan_row(spec, resolved_out) for spec in specs],
        resolved_out / "run_plan.csv",
    )

    selected_folds = tuple(dict.fromkeys(spec.fold for spec in specs))
    selected_horizons = tuple(dict.fromkeys(spec.horizon for spec in specs))
    selected_seeds = tuple(dict.fromkeys(spec.seed for spec in specs))
    selected_lookbacks = tuple(dict.fromkeys(spec.lookback for spec in specs))
    selected_objectives = tuple(dict.fromkeys(spec.objective for spec in specs))
    git_commit = get_git_commit()
    result_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    pretraining_curve_rows: list[dict[str, Any]] = []
    imported_baselines = 0
    horizon_configs: dict[int, NeuralBenchmarkConfig] = {}

    for spec in specs:
        run_dir = spec.run_dir(resolved_out)
        if reuse_completed and _completed_run_present(run_dir):
            payload = _proper_read_json(run_dir / "metrics.json")
            payload["status"] = "skipped_existing"
            result_rows.append(
                _proper_result_row_from_payload(payload, out_dir=resolved_out, run_dir=run_dir)
            )
            training_rows.append(_proper_training_row_from_payload(payload))
            _proper_write_status(run_dir, "skipped_existing")
            _proper_write_run_log(
                run_dir,
                {
                    "run_id": spec.run_id,
                    "status": "skipped_existing",
                    "reason": "existing completed run reused",
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )
            _proper_write_run_manifest(run_dir)
            continue

        horizon_config = horizon_configs.get(spec.horizon)
        if horizon_config is None:
            horizon_config = config.model_copy(
                update={
                    "target": NeuralTargetConfig(
                        horizon=spec.horizon,
                        label_column=f"label_{spec.horizon}",
                    )
                }
            )
            horizon_configs[spec.horizon] = horizon_config

        if run_dir.exists():
            _remove_run_dir(run_dir, root=resolved_out)
        try:
            imported = None
            if (
                import_existing_baselines
                and spec.objective in {"supervised", "masked_reconstruction"}
                and resolved_baseline_source is not None
            ):
                imported = _import_existing_baseline_run(
                    spec,
                    baseline_root=resolved_baseline_source,
                    out_dir=resolved_out,
                    run_dir=run_dir,
                )
            if imported is not None:
                row, training_row = imported
                imported_baselines += 1
            elif spec.objective == "market_state_multitask":
                row, training_row, curves = _execute_ssl_v2_run_spec(
                    spec,
                    config=horizon_config,
                    processed_root=resolved_processed,
                    out_dir=resolved_out,
                    run_dir=run_dir,
                    pretrain_epochs=pretrain_epochs,
                    max_epochs=resolved_max_epochs,
                    patience=resolved_patience,
                    batch_size=resolved_batch_size,
                    mask_probability=mask_probability,
                    future_bucket_count=future_bucket_count,
                    contrastive=contrastive,
                    device=device,
                    monitored_metric=config.training.early_stopping_metric,
                    smoke_test=smoke_test,
                    git_commit=git_commit,
                )
                pretraining_curve_rows.extend(curves)
            else:
                row, training_row = _execute_v1_or_supervised_run_spec(
                    spec,
                    config=horizon_config,
                    processed_root=resolved_processed,
                    out_dir=resolved_out,
                    run_dir=run_dir,
                    pretrain_epochs=pretrain_epochs,
                    max_epochs=resolved_max_epochs,
                    patience=resolved_patience,
                    batch_size=resolved_batch_size,
                    mask_probability=0.15,
                    bucket_count=future_bucket_count,
                    device=device,
                    monitored_metric=config.training.early_stopping_metric,
                    smoke_test=smoke_test,
                    git_commit=git_commit,
                )
            result_rows.append(row)
            training_rows.append(training_row)
        except Exception as exc:
            trace = traceback.format_exc()
            reason = f"{type(exc).__name__}: {exc}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "failure.json").write_text(
                stable_json_dumps(
                    {
                        "run_id": spec.run_id,
                        "reason": reason,
                        "traceback": trace,
                        "git_commit": git_commit,
                    }
                ),
                encoding="utf-8",
            )
            _proper_write_status(run_dir, "failed")
            _proper_write_run_manifest(run_dir)
            failure_rows.append(
                _proper_failure_row(
                    FI2010ProperTrainingRunSpec(
                        fold=spec.fold,
                        horizon=spec.horizon,
                        seed=spec.seed,
                        lookback=spec.lookback,
                        objective=spec.objective,
                    ),
                    status="failed",
                    reason=reason,
                    traceback_text=trace,
                    invalidates=True,
                )
            )

    aggregate_rows, _, _ = write_neural_grid_aggregate_artifacts(
        resolved_out,
        result_rows=result_rows,
        failure_rows=failure_rows,
    )
    comparison_rows, missing_rows = build_ssl_v2_comparison_rows(result_rows)
    _proper_write_csv(
        comparison_rows,
        resolved_out / "ssl_v2_comparison.csv",
        _COMPARISON_COLUMNS,
    )
    _proper_write_csv(
        missing_rows,
        resolved_out / "missing_pairs.csv",
        _COMPARISON_COLUMNS,
    )
    _proper_write_csv(
        training_rows,
        resolved_out / "training_curves_summary.csv",
        _TRAINING_SUMMARY_COLUMNS,
    )
    _proper_write_csv(
        pretraining_curve_rows,
        resolved_out / "ssl_v2_loss_components.csv",
        _PRETRAINING_CURVE_COLUMNS,
    )

    evidence_level, scope_label = _scope_labels(
        folds=selected_folds,
        horizons=selected_horizons,
        seeds=selected_seeds,
        lookbacks=selected_lookbacks,
        objectives=selected_objectives,
        max_epochs=resolved_max_epochs,
        patience=resolved_patience,
        smoke_test=smoke_test,
        completed=sum(1 for row in result_rows if row.get("status") in _SUCCESS_STATUSES),
        planned=len(specs),
    )
    completed = sum(1 for row in result_rows if row.get("status") in _SUCCESS_STATUSES)
    failed = sum(1 for row in failure_rows if row.get("status") == "failed")
    artefacts = {
        "summary": "summary.json",
        "readme": "README.md",
        "run_plan": "run_plan.csv",
        "results_summary": "results_summary.csv",
        "aggregate_summary": "aggregate_summary.csv",
        "aggregate_summary_json": "aggregate_summary.json",
        "ssl_v2_comparison": "ssl_v2_comparison.csv",
        "ssl_comparison": "ssl_comparison.csv",
        "missing_pairs": "missing_pairs.csv",
        "failures": "failures.csv",
        "training_curves_summary": "training_curves_summary.csv",
        "ssl_v2_loss_components": "ssl_v2_loss_components.csv",
        "config_snapshot": "config_snapshot.json",
        "sha256_manifest": "sha256_manifest.json",
        "runs": "runs/",
    }
    warnings: list[str] = []
    if imported_baselines:
        warnings.append(
            f"{imported_baselines} supervised/SSL-v1 baseline run(s) were imported "
            "from existing proper-training artefacts rather than retrained."
        )
    _write_readme(
        resolved_out,
        folds=selected_folds,
        horizons=selected_horizons,
        seeds=selected_seeds,
        lookbacks=selected_lookbacks,
        objectives=selected_objectives,
        evidence_level=evidence_level,
        scope_label=scope_label,
        max_epochs=resolved_max_epochs,
        patience=resolved_patience,
        pretrain_epochs=pretrain_epochs,
        imported_baselines=imported_baselines,
    )
    _write_config_snapshot(
        resolved_out,
        config_path=resolved_config,
        processed_root=resolved_processed,
        baseline_source_dir=resolved_baseline_source,
        folds=selected_folds,
        horizons=selected_horizons,
        seeds=selected_seeds,
        lookbacks=selected_lookbacks,
        objectives=selected_objectives,
        max_epochs=resolved_max_epochs,
        patience=resolved_patience,
        batch_size=resolved_batch_size,
        pretrain_epochs=pretrain_epochs,
        mask_probability=mask_probability,
        future_bucket_count=future_bucket_count,
        contrastive=contrastive,
        evidence_level=evidence_level,
        scope_label=scope_label,
    )

    summary = FI2010SSLV2Summary(
        output_dir=str(resolved_out),
        config_path=str(resolved_config),
        processed_root=str(resolved_processed),
        baseline_source_dir=str(resolved_baseline_source) if resolved_baseline_source else None,
        evidence_level=evidence_level,
        scope_label=scope_label,
        folds=list(selected_folds),
        horizons=list(selected_horizons),
        seeds=list(selected_seeds),
        lookbacks=list(selected_lookbacks),
        objectives=list(selected_objectives),
        pretrain_epochs=pretrain_epochs,
        max_epochs=resolved_max_epochs,
        early_stopping_patience=resolved_patience,
        batch_size=resolved_batch_size,
        device=device,
        smoke_test=smoke_test,
        run_count=len(specs),
        completed_run_count=completed,
        failed_run_count=failed,
        imported_baseline_count=imported_baselines,
        missing_pair_count=len(missing_rows),
        artefacts=artefacts,
        runner_version=FI2010_SSL_V2_BENCHMARK_VERSION,
        created_at=datetime.now(UTC),
        git_commit=git_commit,
        warnings=warnings,
    )
    summary_payload = summary.model_dump(mode="json")
    summary_payload["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_version": __version__,
    }
    summary_payload["aggregate_rows"] = len(aggregate_rows)
    summary_payload["comparison_rows"] = len(comparison_rows)
    summary_payload["claim_boundary"] = (
        "SSL-v2 evidence is limited to the exact stored folds, horizons, seeds, "
        "lookbacks and objectives; no broad SSL, calibration or trading claim follows."
    )
    (resolved_out / "summary.json").write_text(
        stable_json_dumps(summary_payload),
        encoding="utf-8",
    )
    _write_sha256_manifest(resolved_out)
    return summary


def _execute_ssl_v2_run_spec(
    spec: FI2010SSLV2RunSpec,
    *,
    config: NeuralBenchmarkConfig,
    processed_root: Path,
    out_dir: Path,
    run_dir: Path,
    pretrain_epochs: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    mask_probability: float,
    future_bucket_count: int,
    contrastive: bool,
    device: str,
    monitored_metric: str,
    smoke_test: bool,
    git_commit: str | None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    started_at = datetime.now(UTC)
    fold_path = processed_root / f"fold{spec.fold}_combined.csv"
    if not fold_path.is_file():
        raise FileNotFoundError(f"{spec.fold_id} processed CSV is missing: {fold_path}")
    frame = pd.read_csv(fold_path)
    inputs = prepare_fi2010_run_inputs(
        frame=frame,
        config=config,
        fold_path=fold_path,
        seed=spec.seed,
        lookback=spec.lookback,
        out_dir=run_dir,
    )
    settings = build_ssl_finetune_settings(
        config=config,
        lookback=spec.lookback,
        max_epochs=max_epochs,
        batch_size=batch_size,
        device=device,
    )
    pretrain = _pretrain_matrix_ssl_v2_encoder(
        inputs=inputs,
        config=config,
        seed=spec.seed,
        lookback=spec.lookback,
        horizon=spec.horizon,
        mask_probability=mask_probability,
        bucket_count=future_bucket_count,
        contrastive=contrastive,
        pretrain_epochs=pretrain_epochs,
        batch_size=batch_size,
        device=device,
        pretrain_dir=run_dir / "pretrain",
        git_commit=git_commit,
    )
    result = run_matrix_transformer_finetune(
        model_name=spec.objective,
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
        init_source="ssl_v2_pretrained",
        model_type="ssl_v2_finetuned_matrix_transformer",
    )
    predictions = _proper_prediction_rows(result, spec=cast(Any, spec))
    _proper_write_csv(predictions, run_dir / "predictions.csv")
    curve_rows = _proper_curve_rows(result, learning_rate=settings.learning_rate)
    _proper_write_csv(curve_rows, run_dir / "curves.csv")
    (run_dir / "curves.json").write_text(
        stable_json_dumps({"run_id": spec.run_id, "curves": curve_rows}),
        encoding="utf-8",
    )
    metrics = _proper_resolve_metrics(result.metrics, predictions)
    architecture_hash = _proper_hash_payload(
        {
            "model_family": spec.model_family,
            "model_dim": int(settings.transformer_model_dim),
            "num_heads": int(settings.transformer_num_heads),
            "num_layers": int(settings.transformer_num_layers),
            "feedforward_dim": int(settings.transformer_feedforward_dim),
            "dropout": float(settings.dropout),
            "effective_window": int(inputs.effective_window),
        }
    )
    preprocessing_hash = _proper_hash_payload(
        {
            "split": "official_column",
            "feature_scaler": "train_only_standardised_matrix",
            "windowing": "split_contained_contiguous",
            "horizon": spec.horizon,
            "label_column": f"label_{spec.horizon}",
            "lookback": spec.lookback,
            "effective_window": int(inputs.effective_window),
        }
    )
    metadata = result.metadata
    best_epoch = _proper_int_or_none(metadata.get("best_epoch"))
    best_validation_score = _proper_best_validation_score(curve_rows, best_epoch=best_epoch)
    epochs_ran = _proper_int_or_none(metadata.get("epochs_ran")) or len(curve_rows)
    early_stopped = bool(metadata.get("early_stopped", False))
    training_seconds = _proper_safe_float(metadata.get("training_seconds"))
    config_snapshot = {
        "run_id": spec.run_id,
        "runner_version": FI2010_SSL_V2_BENCHMARK_VERSION,
        "fold": spec.fold,
        "horizon": spec.horizon,
        "seed": spec.seed,
        "lookback": spec.lookback,
        "objective": spec.objective,
        "init_source": "ssl_v2_pretrained",
        "model_type": "ssl_v2_finetuned_matrix_transformer",
        "max_epochs": int(max_epochs),
        "early_stopping_patience": int(patience),
        "early_stopping_metric": monitored_metric,
        "learning_rate": float(settings.learning_rate),
        "weight_decay": float(settings.weight_decay),
        "dropout": float(settings.dropout),
        "batch_size": int(batch_size),
        "effective_window": int(inputs.effective_window),
        "pretrain_epochs": int(pretrain_epochs),
        "mask_probability": float(mask_probability),
        "future_bucket_count": int(future_bucket_count),
        "contrastive": bool(contrastive),
        "device": device,
        "smoke_test": bool(smoke_test),
        "split_summary": inputs.split_summary,
    }
    (run_dir / "config.json").write_text(stable_json_dumps(config_snapshot), encoding="utf-8")
    (run_dir / "git_commit.txt").write_text(git_commit or "", encoding="utf-8")
    metric_payload = {
        "runner_version": FI2010_SSL_V2_BENCHMARK_VERSION,
        "subset_kind": "ssl_v2_benchmark",
        "run_id": spec.run_id,
        "status": "completed",
        "fold": spec.fold,
        "fold_id": spec.fold_id,
        "horizon": spec.horizon,
        "seed": spec.seed,
        "lookback": spec.lookback,
        "model_family": spec.model_family,
        "objective": spec.objective,
        "pretraining_objective": spec.pretraining_objective,
        "init_source": "ssl_v2_pretrained",
        "model_type": "ssl_v2_finetuned_matrix_transformer",
        "metrics": metrics,
        "training": {
            "max_epochs": int(max_epochs),
            "epochs_ran": int(epochs_ran),
            "best_epoch": best_epoch,
            "monitored_metric": monitored_metric,
            "best_validation_score": best_validation_score,
            "early_stopping_patience": int(patience),
            "early_stopped": early_stopped,
            "training_seconds": training_seconds,
            "validation_only_model_selection": True,
            "best_checkpoint_restored_before_test": True,
        },
        "ssl_v2_pretraining": pretrain["metadata"],
        "checkpoint_hash": pretrain["checkpoint_sha256"],
        "prediction_file": _proper_relative_path(run_dir / "predictions.csv", out_dir),
        "curves_file": _proper_relative_path(run_dir / "curves.csv", out_dir),
        "architecture_hash": architecture_hash,
        "preprocessing_hash": preprocessing_hash,
        "git_commit": git_commit,
    }
    (run_dir / "metrics.json").write_text(stable_json_dumps(metric_payload), encoding="utf-8")
    _proper_write_status(run_dir, "completed")
    _proper_write_run_log(
        run_dir,
        {
            "run_id": spec.run_id,
            "status": "completed",
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now(UTC).isoformat(),
        },
    )
    _proper_write_run_manifest(run_dir)
    return (
        _proper_result_row_from_payload(metric_payload, out_dir=out_dir, run_dir=run_dir),
        _proper_training_row_from_payload(metric_payload),
        [
            dict(
                row,
                run_id=spec.run_id,
                fold=spec.fold,
                horizon=spec.horizon,
                seed=spec.seed,
                lookback=spec.lookback,
            )
            for row in cast(list[dict[str, Any]], pretrain["curve_rows"])
        ],
    )


def _pretrain_matrix_ssl_v2_encoder(
    *,
    inputs: Any,
    config: NeuralBenchmarkConfig,
    seed: int,
    lookback: int,
    horizon: int,
    mask_probability: float,
    bucket_count: int,
    contrastive: bool,
    pretrain_epochs: int,
    batch_size: int,
    device: str,
    pretrain_dir: Path,
    git_commit: str | None,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader

    standardised = _ssl_standardised_matrix(
        inputs.frame,
        feature_columns=inputs.feature_columns,
        train_indices=inputs.split.train,
    )
    n_rows = int(standardised.shape[0])
    input_features = int(standardised.shape[1])
    future_offset = int(horizon)
    label_spec = fit_ssl_v2_auxiliary_label_spec(
        inputs.frame,
        train_indices=inputs.split.train,
        future_offset=future_offset,
        bucket_count=bucket_count,
    )
    auxiliary_labels = build_ssl_v2_auxiliary_label_matrix(inputs.frame, label_spec)
    feature_groups = infer_ssl_v2_feature_groups(inputs.feature_columns)
    train_windows = build_ssl_v2_windows(
        n_rows=n_rows,
        window_length=inputs.effective_window,
        allowed_indices=inputs.split.train,
        future_offset=future_offset,
    )
    if not train_windows:
        raise ValueError("no SSL-v2 training windows could be built from the training split")
    validation_windows = build_ssl_v2_windows(
        n_rows=n_rows,
        window_length=inputs.effective_window,
        allowed_indices=inputs.split.validation,
        future_offset=future_offset,
    )

    def _build_dataset(windows: Sequence[MatrixSSLV2WindowSample]) -> MatrixSSLV2WindowDataset:
        return MatrixSSLV2WindowDataset(
            standardised,
            windows,
            feature_groups=feature_groups,
            auxiliary_labels=auxiliary_labels,
            mask_probability=mask_probability,
            mask_value=0.0,
            ignore_index=-100,
            enable_contrastive=contrastive,
            base_seed=seed,
        )

    train_loader = DataLoader(
        _build_dataset(train_windows),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_matrix_ssl_v2_windows,
    )
    validation_loader = None
    if validation_windows:
        validation_loader = DataLoader(
            _build_dataset(validation_windows),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_matrix_ssl_v2_windows,
        )
    spec = config.neural_models["matrix_transformer"]
    dropout = config.training.dropout if spec.dropout is None else float(spec.dropout)
    ssl_config = MatrixSSLV2Config(
        input_features=input_features,
        model_dim=int(spec.model_dim or 16),
        num_heads=int(spec.num_heads or 2),
        num_layers=int(spec.num_layers or 1),
        feedforward_dim=int(spec.feedforward_dim or 32),
        dropout=dropout,
        max_sequence_length=inputs.effective_window,
        enable_contrastive=contrastive,
        mask_probability=mask_probability,
        future_bucket_count=bucket_count,
    )
    model = create_matrix_ssl_v2_model(ssl_config)
    history = fit_matrix_ssl_v2(
        model,
        train_loader,
        validation_loader,
        MatrixSSLV2TrainingConfig(
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
    save_pretrained_ssl_v2_encoder(
        model,
        encoder_path,
        metadata={
            "objective": "market_state_multitask",
            "seed": int(seed),
            "lookback": int(lookback),
            "horizon": int(horizon),
            "future_offset": int(future_offset),
            "effective_window": int(inputs.effective_window),
        },
    )
    curve_rows = _pretraining_curve_rows(history)
    _proper_write_csv(curve_rows, pretrain_dir / "ssl_v2_pretraining_curves.csv")
    (pretrain_dir / "ssl_v2_pretraining_curves.json").write_text(
        stable_json_dumps({"history": curve_rows}),
        encoding="utf-8",
    )
    label_spec_payload = {
        "future_offset": int(label_spec.future_offset),
        "bucket_count": int(label_spec.bucket_count),
        "volatility_edges": list(label_spec.volatility_edges),
        "return_edges": list(label_spec.return_edges),
        "imbalance_edges": list(label_spec.imbalance_edges),
        "train_current_count": len(label_spec.train_current_indices),
        "train_future_count": len(label_spec.train_future_indices),
        "label_policy": "current and future target rows must both be inside training split",
    }
    (pretrain_dir / "ssl_v2_auxiliary_label_spec.json").write_text(
        stable_json_dumps(label_spec_payload),
        encoding="utf-8",
    )
    pretrain_config = {
        "runner_version": FI2010_SSL_V2_BENCHMARK_VERSION,
        "objective": "market_state_multitask",
        "enabled_objectives": list(ssl_config.enabled_objectives()),
        "mask_probability": float(mask_probability),
        "structured_feature_groups": {
            name: list(indices) for name, indices in feature_groups.items()
        },
        "future_bucket_count": int(bucket_count),
        "contrastive": bool(contrastive),
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
        "split_policy": {
            "pretraining_train_rows": len(inputs.split.train),
            "train_window_count": len(train_windows),
            "validation_window_count": len(validation_windows),
            "train_only_auxiliary_bin_fitting": True,
            "future_labels_strictly_after_window_end": True,
        },
        "git_commit": git_commit,
    }
    (pretrain_dir / "pretrain_config.json").write_text(
        stable_json_dumps(pretrain_config),
        encoding="utf-8",
    )
    manifest = {
        "checkpoint": str(encoder_path.name),
        "checkpoint_sha256": sha256_file(encoder_path),
        "files": {
            path.name: sha256_file(path)
            for path in pretrain_dir.iterdir()
            if path.is_file()
        },
    }
    (pretrain_dir / "artefact_manifest.json").write_text(
        stable_json_dumps(manifest),
        encoding="utf-8",
    )
    metadata = {
        "objective": "market_state_multitask",
        "pretrain_epochs": int(pretrain_epochs),
        "mask_probability": float(mask_probability),
        "future_bucket_count": int(bucket_count),
        "contrastive": bool(contrastive),
        "train_window_count": len(train_windows),
        "validation_window_count": len(validation_windows),
        "final_train_loss": float(final.train_loss),
        "final_validation_loss": final.validation_loss,
        "final_train_loss_components": dict(final.train_loss_components),
        "final_validation_loss_components": dict(final.validation_loss_components),
        "encoder_parameter_count": int(model.n_parameters()),
        "checkpoint_sha256": sha256_file(encoder_path),
        "data_policy": (
            "training windows only for optimisation; auxiliary bin edges fitted "
            "on train-only current/future pairs"
        ),
        "auxiliary_label_spec": label_spec_payload,
    }
    return {
        "encoder_state": load_pretrained_ssl_v2_encoder_state(encoder_path),
        "checkpoint_sha256": sha256_file(encoder_path),
        "metadata": metadata,
        "curve_rows": curve_rows,
    }


def _execute_v1_or_supervised_run_spec(
    spec: FI2010SSLV2RunSpec,
    *,
    config: NeuralBenchmarkConfig,
    processed_root: Path,
    out_dir: Path,
    run_dir: Path,
    pretrain_epochs: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    mask_probability: float,
    bucket_count: int,
    device: str,
    monitored_metric: str,
    smoke_test: bool,
    git_commit: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    proper_spec = FI2010ProperTrainingRunSpec(
        fold=spec.fold,
        horizon=spec.horizon,
        seed=spec.seed,
        lookback=spec.lookback,
        objective=spec.objective,
    )
    from chronoslob.experiments.fi2010_neural_proper_training import (
        _execute_run_spec as _proper_execute_run_spec,
    )

    return _proper_execute_run_spec(
        proper_spec,
        config=config,
        processed_root=processed_root,
        out_dir=out_dir,
        run_dir=run_dir,
        pretrain_epochs=pretrain_epochs,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        mask_probability=mask_probability,
        bucket_count=bucket_count,
        device=device,
        monitored_metric=monitored_metric,
        smoke_test=smoke_test,
        git_commit=git_commit,
    )


def _import_existing_baseline_run(
    spec: FI2010SSLV2RunSpec,
    *,
    baseline_root: Path,
    out_dir: Path,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    source = (
        baseline_root
        / "runs"
        / spec.fold_id
        / f"horizon_{spec.horizon}"
        / f"seed_{spec.seed}"
        / f"lookback_{spec.lookback}"
        / spec.objective
    )
    if not (source / "metrics.json").is_file():
        return None
    if run_dir.exists():
        _remove_run_dir(run_dir, root=out_dir)
    shutil.copytree(source, run_dir)
    payload = _proper_read_json(run_dir / "metrics.json")
    payload["status"] = "completed"
    payload["runner_version"] = FI2010_SSL_V2_BENCHMARK_VERSION
    payload["subset_kind"] = "ssl_v2_benchmark"
    payload["imported_from"] = _proper_relative_path(source, Path.cwd())
    payload["prediction_file"] = _proper_relative_path(run_dir / "predictions.csv", out_dir)
    payload["curves_file"] = _proper_relative_path(run_dir / "curves.csv", out_dir)
    (run_dir / "metrics.json").write_text(stable_json_dumps(payload), encoding="utf-8")
    _proper_write_status(run_dir, "completed")
    _proper_write_run_log(
        run_dir,
        {
            "run_id": spec.run_id,
            "status": "completed",
            "source": _proper_relative_path(source, Path.cwd()),
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    _proper_write_run_manifest(run_dir)
    return (
        _proper_result_row_from_payload(payload, out_dir=out_dir, run_dir=run_dir),
        _proper_training_row_from_payload(payload),
    )


def build_ssl_v2_comparison_rows(
    result_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build supervised-vs-SSL-v1/v2 comparison rows for all stored objectives."""
    successes = [dict(row) for row in result_rows if row.get("status") in _SUCCESS_STATUSES]
    by_key: dict[tuple[int, int, int, int, str, str], dict[str, Any]] = {}
    grouped_keys: set[tuple[int, int, int, int, str]] = set()
    objectives: set[str] = set()
    for row in successes:
        pretraining_objective = str(row.get("pretraining_objective"))
        objective = (
            "supervised"
            if pretraining_objective == "none"
            else pretraining_objective
        )
        key = (
            int(float(row["fold"])),
            int(float(row["horizon"])),
            int(float(row["seed"])),
            int(float(row["lookback"])),
            str(row.get("model_family", "")),
        )
        grouped_keys.add(key)
        by_key[(*key, objective)] = row
        if objective != "supervised":
            objectives.add(objective)
    comparison_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for key in sorted(grouped_keys):
        supervised = by_key.get((*key, "supervised"))
        for objective in sorted(objectives):
            ssl = by_key.get((*key, objective))
            row = _comparison_base_row(key=key, objective=objective, supervised=supervised, ssl=ssl)
            if supervised is None or ssl is None:
                missing = "supervised" if supervised is None else objective
                row.update({"status": "missing_pair", "reason": f"missing matched {missing} run"})
                comparison_rows.append(row)
                missing_rows.append(dict(row))
                continue
            if (
                supervised.get("architecture_hash") != ssl.get("architecture_hash")
                or supervised.get("preprocessing_hash") != ssl.get("preprocessing_hash")
            ):
                row.update(
                    {
                        "status": "unmatched_metadata",
                        "reason": "architecture or preprocessing hash differs; deltas refused",
                    }
                )
                comparison_rows.append(row)
                missing_rows.append(dict(row))
                continue
            _fill_comparison_deltas(row, supervised=supervised, ssl=ssl)
            row.update({"status": "matched", "reason": ""})
            comparison_rows.append(row)
    return comparison_rows, missing_rows


def _comparison_base_row(
    *,
    key: tuple[int, int, int, int, str],
    objective: str,
    supervised: Mapping[str, Any] | None,
    ssl: Mapping[str, Any] | None,
) -> dict[str, Any]:
    fold, horizon, seed, lookback, model_family = key
    return {
        "fold": fold,
        "horizon": horizon,
        "seed": seed,
        "lookback": lookback,
        "model_family": model_family,
        "ssl_objective": objective,
        "comparison": f"supervised_vs_{objective}",
        "supervised_run_id": None if supervised is None else supervised.get("run_id"),
        "ssl_run_id": None if ssl is None else ssl.get("run_id"),
        "supervised_macro_f1": None,
        "ssl_macro_f1": None,
        "delta_macro_f1": None,
        "supervised_accuracy": None,
        "ssl_accuracy": None,
        "delta_accuracy": None,
        "supervised_mcc": None,
        "ssl_mcc": None,
        "delta_mcc": None,
        "supervised_ece": None,
        "ssl_ece": None,
        "delta_ece": None,
        "supervised_brier_score": None,
        "ssl_brier_score": None,
        "delta_brier_score": None,
        "supervised_nll": None,
        "ssl_nll": None,
        "delta_nll": None,
        "macro_f1_outcome": None,
        "ece_outcome": None,
        "mcc_outcome": None,
        "status": "pending",
        "reason": "",
    }


def _fill_comparison_deltas(
    row: dict[str, Any],
    *,
    supervised: Mapping[str, Any],
    ssl: Mapping[str, Any],
) -> None:
    metrics = (
        ("macro_f1", "macro_f1", True),
        ("accuracy", "accuracy", True),
        ("mcc", "mcc", True),
        ("ece", "ece", False),
        ("brier_score", "brier_score", False),
        ("nll", "nll", False),
    )
    for source_name, column_name, higher_is_better in metrics:
        supervised_value = _proper_safe_float(supervised.get(source_name))
        ssl_value = _proper_safe_float(ssl.get(source_name))
        row[f"supervised_{column_name}"] = supervised_value
        row[f"ssl_{column_name}"] = ssl_value
        delta = (
            None
            if supervised_value is None or ssl_value is None
            else ssl_value - supervised_value
        )
        row[f"delta_{column_name}"] = delta
        if source_name == "macro_f1":
            row["macro_f1_outcome"] = _delta_outcome(delta, higher_is_better=higher_is_better)
        if source_name == "ece":
            row["ece_outcome"] = _delta_outcome(delta, higher_is_better=higher_is_better)
        if source_name == "mcc":
            row["mcc_outcome"] = _delta_outcome(delta, higher_is_better=higher_is_better)


def _delta_outcome(value: float | None, *, higher_is_better: bool) -> str | None:
    if value is None:
        return None
    if abs(value) <= 1e-12:
        return "tie"
    improved = value > 0.0 if higher_is_better else value < 0.0
    return "win" if improved else "loss"


def _pretraining_curve_rows(
    history: Sequence[MatrixSSLV2EpochResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in history:
        row: dict[str, Any] = {
            "epoch": int(item.epoch),
            "train_loss": float(item.train_loss),
            "validation_loss": item.validation_loss,
        }
        for component in (
            "reconstruction",
            "future_spread_widening",
            "future_volatility",
            "future_return",
            "future_imbalance",
            "contrastive",
            "total",
        ):
            row[f"train_{component}"] = item.train_loss_components.get(component)
            row[f"validation_{component}"] = item.validation_loss_components.get(component)
        rows.append(row)
    return rows


def _completed_run_present(run_dir: Path) -> bool:
    required = ("metrics.json", "curves.csv", "config.json", "status.txt")
    return all((run_dir / name).is_file() for name in required) and (
        run_dir / "status.txt"
    ).read_text(encoding="utf-8").strip() in _SUCCESS_STATUSES


def _write_sha256_manifest(out_dir: Path) -> None:
    hashes: dict[str, str] = {}
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.json":
            hashes[_proper_relative_path(path, out_dir)] = sha256_file(path)
    (out_dir / "sha256_manifest.json").write_text(
        stable_json_dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "runner_version": FI2010_SSL_V2_BENCHMARK_VERSION,
                "sha256": hashes,
            }
        ),
        encoding="utf-8",
    )


def _write_config_snapshot(
    out_dir: Path,
    *,
    config_path: Path,
    processed_root: Path,
    baseline_source_dir: Path | None,
    folds: Sequence[int],
    horizons: Sequence[int],
    seeds: Sequence[int],
    lookbacks: Sequence[int],
    objectives: Sequence[str],
    max_epochs: int,
    patience: int,
    batch_size: int,
    pretrain_epochs: int,
    mask_probability: float,
    future_bucket_count: int,
    contrastive: bool,
    evidence_level: str,
    scope_label: str,
) -> None:
    payload = {
        "runner_version": FI2010_SSL_V2_BENCHMARK_VERSION,
        "config_path": str(config_path),
        "processed_root": str(processed_root),
        "baseline_source_dir": str(baseline_source_dir) if baseline_source_dir else None,
        "folds": list(folds),
        "horizons": list(horizons),
        "seeds": list(seeds),
        "lookbacks": list(lookbacks),
        "objectives": list(objectives),
        "pretrain_epochs": int(pretrain_epochs),
        "max_epochs": int(max_epochs),
        "early_stopping_patience": int(patience),
        "batch_size": int(batch_size),
        "mask_probability": float(mask_probability),
        "future_bucket_count": int(future_bucket_count),
        "contrastive": bool(contrastive),
        "evidence_level": evidence_level,
        "scope_label": scope_label,
        "claim_boundary": "second-generation SSL objective, scoped comparison only",
    }
    (out_dir / "config_snapshot.json").write_text(stable_json_dumps(payload), encoding="utf-8")


def _write_readme(
    out_dir: Path,
    *,
    folds: Sequence[int],
    horizons: Sequence[int],
    seeds: Sequence[int],
    lookbacks: Sequence[int],
    objectives: Sequence[str],
    evidence_level: str,
    scope_label: str,
    max_epochs: int,
    patience: int,
    pretrain_epochs: int,
    imported_baselines: int,
) -> None:
    lines = [
        "# FI-2010 SSL-v2 Benchmark",
        "",
        "This directory is generated by `run-fi2010-ssl-v2-benchmark`.",
        "It evaluates a failure-analysis-motivated second-generation SSL objective "
        "against matched supervised and first-generation SSL baselines where stored "
        "or run in the same benchmark directory.",
        "",
        "## Scope",
        "",
        f"- folds: {', '.join(str(item) for item in folds)}",
        f"- horizons: {', '.join(str(item) for item in horizons)}",
        f"- seeds: {', '.join(str(item) for item in seeds)}",
        f"- lookbacks: {', '.join(str(item) for item in lookbacks)}",
        f"- objectives: {', '.join(objectives)}",
        f"- SSL-v2 pretrain epochs: {pretrain_epochs}",
        f"- fine-tune max epochs: {max_epochs}",
        f"- early-stopping patience: {patience}",
        f"- evidence level: {evidence_level}",
        f"- scope label: {scope_label}",
        f"- imported baseline runs: {imported_baselines}",
        "",
        "## Claim Boundaries",
        "",
        "- Predictive metric deltas are limited to this exact stored scope.",
        "- Calibration improvement requires ECE and Brier deltas to improve.",
        "- Broad SSL improvement, profitability, live trading and general market "
        "representation claims are not made by these artefacts.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _scope_labels(
    *,
    folds: Sequence[int],
    horizons: Sequence[int],
    seeds: Sequence[int],
    lookbacks: Sequence[int],
    objectives: Sequence[str],
    max_epochs: int,
    patience: int,
    smoke_test: bool,
    completed: int,
    planned: int,
) -> tuple[str, str]:
    if smoke_test:
        return "smoke_test_only", "smoke_code_path_only"
    complete_target = (
        set(folds).issuperset(SSL_V2_COMPLETE_TARGET_FOLDS)
        and set(horizons) == SSL_V2_TARGET_HORIZONS
        and set(seeds) == SSL_V2_TARGET_SEEDS
        and set(lookbacks) == SSL_V2_TARGET_LOOKBACKS
        and set(objectives).issuperset({"supervised", "market_state_multitask"})
        and max_epochs >= SSL_V2_TARGET_MAX_EPOCHS
        and patience >= SSL_V2_TARGET_PATIENCE
        and completed == planned
    )
    if complete_target:
        return "complete_real", "folds_1_3_h10_h50_complete_real"
    return "partial_real", "limited_ssl_v2_partial_real_slice"


def _run_plan_row(spec: FI2010SSLV2RunSpec, out_dir: Path) -> dict[str, Any]:
    return {
        "run_id": spec.run_id,
        "fold": spec.fold,
        "fold_id": spec.fold_id,
        "horizon": spec.horizon,
        "seed": spec.seed,
        "lookback": spec.lookback,
        "objective": spec.objective,
        "pretraining_objective": spec.pretraining_objective,
        "run_dir": _proper_relative_path(spec.run_dir(out_dir), out_dir),
    }


def _remove_run_dir(run_dir: Path, *, root: Path) -> None:
    resolved_root = root.resolve(strict=False)
    resolved_run = run_dir.resolve(strict=False)
    try:
        resolved_run.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"refusing to remove run outside output root: {run_dir}") from exc
    shutil.rmtree(resolved_run)


def _normalise_folds(
    value: Sequence[int | str] | str | None,
    *,
    default: Sequence[int],
) -> tuple[int, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "all":
            return tuple(default)
        tokens: Sequence[int | str] = [token.strip() for token in text.split(",")]
    else:
        tokens = value
    cleaned: list[int] = []
    for token in tokens:
        number = _fold_number(token)
        if number not in cleaned:
            cleaned.append(number)
    if not cleaned:
        raise ValueError("folds selection must not be empty")
    return tuple(cleaned)


def _normalise_ints(
    value: Sequence[int] | str | None,
    *,
    default: Sequence[int],
    field_name: str,
    positive: bool,
) -> tuple[int, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "all":
            return tuple(default)
        tokens: Sequence[int | str] = [token.strip() for token in text.split(",")]
    else:
        tokens = value
    cleaned: list[int] = []
    for token in tokens:
        try:
            number = int(token)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} entries must be integers") from exc
        if positive and number <= 0:
            raise ValueError(f"{field_name} entries must be positive")
        if not positive and number < 0:
            raise ValueError(f"{field_name} entries must be non-negative")
        if number not in cleaned:
            cleaned.append(number)
    if not cleaned:
        raise ValueError(f"{field_name} selection must not be empty")
    return tuple(cleaned)


def _normalise_objectives(value: Sequence[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return SSL_V2_OBJECTIVE_CHOICES
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "all":
            return SSL_V2_OBJECTIVE_CHOICES
        tokens = [token.strip() for token in text.split(",")]
    else:
        tokens = [str(token).strip() for token in value]
    cleaned: list[str] = []
    for token in tokens:
        objective = token.lower()
        if objective not in SSL_V2_OBJECTIVE_CHOICES:
            raise ValueError(
                f"unsupported SSL-v2 objective {token!r}; supported: "
                f"{list(SSL_V2_OBJECTIVE_CHOICES)}"
            )
        if objective not in cleaned:
            cleaned.append(objective)
    if not cleaned:
        raise ValueError("objectives selection must not be empty")
    return tuple(cleaned)


def _fold_number(value: int | str) -> int:
    if isinstance(value, bool):
        raise ValueError("fold identifiers must be positive integers")
    if isinstance(value, int):
        number = value
    else:
        token = value.strip().lower()
        if token.startswith("fold_"):
            token = token.removeprefix("fold_")
        if not token.isdigit():
            raise ValueError(f"invalid fold identifier: {value!r}")
        number = int(token)
    if number <= 0:
        raise ValueError("fold identifiers must be positive")
    return number


_RESERVED_V1_PRETRAIN = pretrain_matrix_ssl_encoder
_RESERVED_V1_FLAGS = ssl_objective_flags
