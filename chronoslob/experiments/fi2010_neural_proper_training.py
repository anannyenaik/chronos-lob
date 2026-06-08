"""FI-2010 proper-training neural subset: longer training with early stopping.

This runner produces longer-training neural modelling evidence that is reported
**separately** from the one-epoch matched full grid. It is not a replacement for
that grid; the one-epoch grid remains the matched comparison and infrastructure
evidence, and this subset assesses whether the neural models stay credible under
a more realistic training budget.

For each fold/horizon/seed/lookback/objective it:

- builds the official leakage-safe train/validation/test split and a train-only
  standardised matrix, reusing the SSL benchmark's input preparation,
- for SSL objectives, pretrains a transformer encoder on **training rows only**,
- trains (supervised) or fine-tunes (SSL) a matrix transformer with
  validation-only early stopping, restoring the best validation checkpoint
  before the single official test evaluation,
- persists per-epoch training curves, the best epoch, the monitored validation
  metric and the held-out test metrics.

Supervised and SSL runs in the same fold/horizon/seed/lookback cell share an
identical architecture, preprocessing, seed and lookback, so the SSL deltas are
matched. The runner reports predictive and calibration metrics only and makes no
trading, profitability, PnL or state-of-the-art claim. If SSL does not help, the
deltas simply show that.
"""

from __future__ import annotations

import math
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
from chronoslob.experiments.fi2010_benchmark import PaperNeuralSettings
from chronoslob.experiments.fi2010_neural_grid import (
    build_ssl_comparison_rows,
    write_neural_grid_aggregate_artifacts,
)
from chronoslob.experiments.fi2010_ssl_runner import (
    build_ssl_finetune_settings,
    prepare_fi2010_run_inputs,
    pretrain_matrix_ssl_encoder,
    ssl_objective_flags,
)
from chronoslob.experiments.manifests import sha256_file, stable_json_dumps
from chronoslob.experiments.neural_adapters import (
    NeuralPaperModelResult,
    run_matrix_transformer_finetune,
    run_neural_paper_model,
)
from chronoslob.experiments.neural_benchmarking import (
    NeuralBenchmarkConfig,
    NeuralTargetConfig,
    load_neural_benchmark_config,
)
from chronoslob.training.experiment import get_git_commit
from chronoslob.utils.paths import project_root

__all__ = [
    "FI2010_PROPER_TRAINING_VERSION",
    "PROPER_TRAINING_MODEL_CHOICES",
    "PROPER_TRAINING_OBJECTIVE_CHOICES",
    "FI2010ProperTrainingRunSpec",
    "FI2010ProperTrainingSummary",
    "expand_proper_training_specs",
    "run_fi2010_neural_proper_training_subset",
]

FI2010_PROPER_TRAINING_VERSION = "fi2010-neural-proper-training-subset/v1"

PT_DEFAULT_OUT_DIR = Path("experiments/fi2010_neural_proper_training_subset_v2")
PT_DEFAULT_FOLDS: tuple[int, ...] = (1, 2, 3, 4, 5)
PT_DEFAULT_HORIZONS: tuple[int, ...] = (10, 20, 50)
PT_DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
PT_DEFAULT_LOOKBACKS: tuple[int, ...] = (50,)
PROPER_TRAINING_OBJECTIVE_CHOICES: tuple[str, ...] = (
    "supervised",
    "masked_reconstruction",
    "next_field",
)
PROPER_TRAINING_MODEL_CHOICES: tuple[str, ...] = (
    "matrix_transformer",
    "deeplob_style",
)
_BROADER_TARGET_FOLDS = {1, 2, 3, 4, 5}
_BROADER_TARGET_HORIZONS = {10, 50}
_BROADER_TARGET_SEEDS = {0, 1, 2}
_BROADER_TARGET_LOOKBACKS = {20, 50, 100}
_BROADER_TARGET_MODELS = set(PROPER_TRAINING_MODEL_CHOICES)
_BROADER_TARGET_OBJECTIVES = {"supervised"}

_PRIMARY_TARGET_FOLDS = {1, 2, 3, 4, 5}
_PRIMARY_TARGET_HORIZONS = {10, 50}
_PRIMARY_TARGET_SEEDS = {0}
_PRIMARY_TARGET_LOOKBACKS = {50}
_PRIMARY_TARGET_OBJECTIVES = set(PROPER_TRAINING_OBJECTIVE_CHOICES)
_PRIMARY_TARGET_MAX_EPOCHS = 25
_PRIMARY_TARGET_PATIENCE = 5
_FALLBACK_TARGET_FOLDS = {1, 2, 3}

_SUCCESS_STATUSES = {"completed", "skipped_existing"}
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)
_EPS = 1e-12

_RESULT_SUMMARY_COLUMNS: tuple[str, ...] = (
    "fold",
    "horizon",
    "seed",
    "lookback",
    "model_family",
    "pretraining_objective",
    "accuracy",
    "macro_f1",
    "mcc",
    "ece",
    "brier_score",
    "nll",
    "class_f1_down",
    "class_f1_stationary",
    "class_f1_up",
    "checkpoint_hash",
    "prediction_file",
    "status",
    "run_id",
    "run_dir",
    "architecture_hash",
    "preprocessing_hash",
)

_FAILURE_COLUMNS: tuple[str, ...] = (
    "fold",
    "horizon",
    "seed",
    "model_family",
    "objective",
    "reason",
    "traceback",
    "invalidates_aggregate_claims",
    "status",
    "run_id",
)

_CURVE_COLUMNS: tuple[str, ...] = (
    "epoch",
    "train_loss",
    "validation_loss",
    "validation_accuracy",
    "validation_macro_f1",
    "validation_mcc",
    "monitored_value",
    "learning_rate",
    "is_best",
    "early_stop",
)

_TRAINING_SUMMARY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "fold",
    "horizon",
    "seed",
    "lookback",
    "model_family",
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

_PREDICTION_COLUMNS: tuple[str, ...] = (
    "row_id",
    "sample_id",
    "fold",
    "horizon",
    "seed",
    "lookback",
    "model_family",
    "pretraining_objective",
    "split",
    "y_true",
    "y_pred",
    "prob_down",
    "prob_stationary",
    "prob_up",
    "confidence",
)


class FI2010ProperTrainingSummary(BaseModel):
    """Top-level summary returned by the proper-training subset runner."""

    model_config = _MODEL_CONFIG

    output_dir: str
    config_path: str
    processed_root: str
    subset_kind: str = "proper_training_subset"
    folds: list[int]
    horizons: list[int]
    seeds: list[int]
    lookbacks: list[int]
    models: list[str]
    objectives: list[str]
    pretrain_epochs: int
    max_epochs: int
    early_stopping_patience: int
    early_stopping_metric: str
    batch_size: int
    device: str
    smoke_test: bool
    execution_mode: str
    run_count: int
    completed_run_count: int
    failed_run_count: int
    skipped_existing_count: int
    missing_pair_count: int
    planned_scope_complete: bool
    target_scope_complete: bool
    scope_label: str
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
class FI2010ProperTrainingRunSpec:
    """One fold/horizon/seed/lookback/model/objective proper-training run."""

    fold: int
    horizon: int
    seed: int
    lookback: int
    objective: str
    model_name: str = "matrix_transformer"

    @property
    def fold_id(self) -> str:
        return f"fold_{self.fold}"

    @property
    def run_id(self) -> str:
        model_token = (
            "" if self.model_name == "matrix_transformer" else f"__{self.model_name}"
        )
        return (
            f"{self.fold_id}__h{self.horizon}__seed_{self.seed}"
            f"__lb{self.lookback}{model_token}__{self.objective}"
        )

    @property
    def model_family(self) -> str:
        return self.model_name

    @property
    def pretraining_objective(self) -> str:
        return "none" if self.objective == "supervised" else self.objective

    def run_dir(self, out_dir: Path) -> Path:
        root = (
            Path(out_dir)
            / "runs"
            / self.fold_id
            / f"horizon_{self.horizon}"
            / f"seed_{self.seed}"
            / f"lookback_{self.lookback}"
        )
        if self.model_name == "matrix_transformer":
            return root / self.objective
        return root / self.model_name / self.objective


def expand_proper_training_specs(
    *,
    folds: Sequence[int | str] | str | None = None,
    horizons: Sequence[int] | str | None = None,
    seeds: Sequence[int] | str | None = None,
    lookbacks: Sequence[int] | str | None = None,
    models: Sequence[str] | str | None = None,
    objectives: Sequence[str] | str | None = None,
    smoke_test: bool = False,
) -> tuple[FI2010ProperTrainingRunSpec, ...]:
    """Expand the deterministic proper-training run grid."""
    selected_folds = _normalise_folds(folds, default=PT_DEFAULT_FOLDS)
    selected_horizons = _normalise_ints(
        horizons, default=PT_DEFAULT_HORIZONS, field_name="horizons", positive=True
    )
    selected_seeds = _normalise_ints(
        seeds, default=PT_DEFAULT_SEEDS, field_name="seeds", positive=False
    )
    selected_lookbacks = _normalise_ints(
        lookbacks, default=PT_DEFAULT_LOOKBACKS, field_name="lookbacks", positive=True
    )
    selected_models = _normalise_models(models)
    selected_objectives = _normalise_objectives(objectives)
    if any(
        model != "matrix_transformer" and objective != "supervised"
        for model in selected_models
        for objective in selected_objectives
    ):
        raise ValueError(
            "DeepLOB-style proper training supports the supervised objective only; "
            "SSL objectives require matrix_transformer"
        )

    if smoke_test:
        selected_folds = selected_folds[:1]
        selected_horizons = selected_horizons[:1]
        selected_seeds = selected_seeds[:1]
        selected_lookbacks = selected_lookbacks[:1]

    specs: list[FI2010ProperTrainingRunSpec] = []
    for fold in selected_folds:
        for horizon in selected_horizons:
            for seed in selected_seeds:
                for lookback in selected_lookbacks:
                    for model_name in selected_models:
                        for objective in selected_objectives:
                            specs.append(
                                FI2010ProperTrainingRunSpec(
                                    fold=fold,
                                    horizon=horizon,
                                    seed=seed,
                                    lookback=lookback,
                                    model_name=model_name,
                                    objective=objective,
                                )
                            )
    return tuple(specs)


def run_fi2010_neural_proper_training_subset(
    config_path: str | Path = Path("configs/experiments/fi2010_neural_proper_training.yaml"),
    *,
    processed_root: str | Path,
    out_dir: str | Path = PT_DEFAULT_OUT_DIR,
    folds: Sequence[int | str] | str | None = None,
    horizons: Sequence[int] | str | None = None,
    seeds: Sequence[int] | str | None = None,
    lookbacks: Sequence[int] | str | None = None,
    models: Sequence[str] | str | None = None,
    objectives: Sequence[str] | str | None = None,
    pretrain_epochs: int = 10,
    max_epochs: int | None = None,
    patience: int | None = None,
    batch_size: int | None = None,
    mask_probability: float = 0.15,
    next_field_bucket_count: int = 3,
    device: str = "cpu",
    reuse_completed: bool = True,
    smoke_test: bool = False,
) -> FI2010ProperTrainingSummary:
    """Run and aggregate the FI-2010 proper-training neural subset."""
    resolved_config = Path(config_path)
    if not resolved_config.is_file():
        raise FileNotFoundError(f"proper-training config not found: {resolved_config}")
    base_config = load_neural_benchmark_config(resolved_config)
    if "matrix_transformer" not in base_config.enabled_model_names:
        raise ValueError(
            "the proper-training subset requires the 'matrix_transformer' model "
            "to be enabled so supervised and SSL architectures match"
        )
    selected_models = _normalise_models(models)
    unavailable_models = [
        model for model in selected_models if model not in base_config.enabled_model_names
    ]
    if unavailable_models:
        raise ValueError(
            "proper-training models are not enabled in the config: "
            + ", ".join(unavailable_models)
        )

    resolved_out = Path(out_dir)
    resolved_out.mkdir(parents=True, exist_ok=True)
    resolved_processed = Path(processed_root)

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

    if smoke_test:
        device = "cpu"

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
    monitored_metric = config.training.early_stopping_metric

    specs = expand_proper_training_specs(
        folds=folds,
        horizons=horizons,
        seeds=seeds,
        lookbacks=lookbacks,
        models=selected_models,
        objectives=objectives,
        smoke_test=smoke_test,
    )
    if not specs:
        raise ValueError("proper-training selection produced no runs")

    _write_csv(
        [_run_plan_row(spec, resolved_out) for spec in specs],
        resolved_out / "run_plan.csv",
    )

    selected_folds = tuple(dict.fromkeys(spec.fold for spec in specs))
    selected_horizons = tuple(dict.fromkeys(spec.horizon for spec in specs))
    selected_seeds = tuple(dict.fromkeys(spec.seed for spec in specs))
    selected_lookbacks = tuple(dict.fromkeys(spec.lookback for spec in specs))
    selected_models = tuple(dict.fromkeys(spec.model_name for spec in specs))
    selected_objectives = tuple(dict.fromkeys(spec.objective for spec in specs))

    git_commit = get_git_commit()
    result_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    horizon_configs: dict[int, NeuralBenchmarkConfig] = {}
    warnings: list[str] = []
    config_mismatch_rerun_count = 0

    for spec in specs:
        run_dir = spec.run_dir(resolved_out)
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
        expected_signature = _expected_reuse_signature(
            spec,
            config=horizon_config,
            max_epochs=resolved_max_epochs,
            patience=resolved_patience,
            batch_size=resolved_batch_size,
            pretrain_epochs=pretrain_epochs,
            device=device,
            smoke_test=smoke_test,
            mask_probability=mask_probability,
            bucket_count=next_field_bucket_count,
        )
        if reuse_completed and _completed_run_present(
            run_dir,
            expected_signature=expected_signature,
        ):
            row, training_row = _load_existing_rows(run_dir, out_dir=resolved_out)
            result_rows.append(row)
            training_rows.append(training_row)
            _write_status(run_dir, "skipped_existing")
            _write_run_log(
                run_dir,
                {
                    "run_id": spec.run_id,
                    "status": "skipped_existing",
                    "reason": "existing completed run reused",
                    "created_at": datetime.now(UTC).isoformat(),
                },
            )
            _write_run_manifest(run_dir)
            failure_rows.append(
                _failure_row(
                    spec,
                    status="skipped_existing",
                    reason="existing completed run reused",
                    traceback_text="",
                    invalidates=False,
                )
            )
            continue

        if reuse_completed and _completed_run_present(run_dir):
            config_mismatch_rerun_count += 1
        if run_dir.exists():
            _remove_run_dir(run_dir, root=resolved_out)
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            row, training_row = _execute_run_spec(
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
                bucket_count=next_field_bucket_count,
                device=device,
                monitored_metric=monitored_metric,
                smoke_test=smoke_test,
                git_commit=git_commit,
            )
            result_rows.append(row)
            training_rows.append(training_row)
        except Exception as exc:
            trace = traceback.format_exc()
            reason = f"{type(exc).__name__}: {exc}"
            _write_failure_artefacts(
                spec,
                run_dir=run_dir,
                reason=reason,
                traceback_text=trace,
                git_commit=git_commit,
            )
            failure_rows.append(
                _failure_row(
                    spec,
                    status="failed",
                    reason=reason,
                    traceback_text=trace,
                    invalidates=True,
                )
            )

    aggregate_rows, comparison_rows, missing_pairs = write_neural_grid_aggregate_artifacts(
        resolved_out,
        result_rows=result_rows,
        failure_rows=failure_rows,
        expect_ssl_pairs=any(
            objective != "supervised" for objective in selected_objectives
        ),
    )
    _write_proper_aggregate_summary_json(
        resolved_out,
        aggregate_rows=aggregate_rows,
        missing_pair_count=len(missing_pairs),
        result_rows=result_rows,
        failure_rows=failure_rows,
    )
    _write_csv(
        training_rows,
        resolved_out / "training_curves_summary.csv",
        _TRAINING_SUMMARY_COLUMNS,
    )

    completed = sum(1 for row in result_rows if row.get("status") in _SUCCESS_STATUSES)
    skipped = sum(1 for row in result_rows if row.get("status") == "skipped_existing")
    failed = sum(1 for row in failure_rows if row.get("status") == "failed")
    planned_complete = completed == len(specs) and failed == 0
    primary_complete = _primary_target_complete(
        folds=selected_folds,
        horizons=selected_horizons,
        seeds=selected_seeds,
        lookbacks=selected_lookbacks,
        models=selected_models,
        objectives=selected_objectives,
        max_epochs=resolved_max_epochs,
        patience=resolved_patience,
        planned_complete=planned_complete,
        smoke_test=smoke_test,
    )
    scope_label = _scope_label(
        folds=selected_folds,
        horizons=selected_horizons,
        seeds=selected_seeds,
        lookbacks=selected_lookbacks,
        models=selected_models,
        objectives=selected_objectives,
        max_epochs=resolved_max_epochs,
        patience=resolved_patience,
        planned_complete=planned_complete,
        primary_complete=primary_complete,
        smoke_test=smoke_test,
    )
    evidence_level = (
        "smoke_test_only"
        if smoke_test
        else "complete_real"
        if primary_complete
        else "partial_real"
    )
    if config_mismatch_rerun_count:
        warnings.append(
            f"{config_mismatch_rerun_count} completed run(s) had a mismatched reuse "
            "signature and were rerun instead of reused."
        )

    _write_subset_readme(
        resolved_out,
        folds=selected_folds,
        horizons=selected_horizons,
        seeds=selected_seeds,
        lookbacks=selected_lookbacks,
        models=selected_models,
        objectives=selected_objectives,
        max_epochs=resolved_max_epochs,
        patience=resolved_patience,
        monitored_metric=monitored_metric,
        pretrain_epochs=pretrain_epochs,
        smoke_test=smoke_test,
        evidence_level=evidence_level,
        scope_label=scope_label,
    )
    _write_root_config_snapshot(
        resolved_out,
        config_path=resolved_config,
        processed_root=resolved_processed,
        folds=selected_folds,
        horizons=selected_horizons,
        seeds=selected_seeds,
        lookbacks=selected_lookbacks,
        models=selected_models,
        objectives=selected_objectives,
        max_epochs=resolved_max_epochs,
        patience=resolved_patience,
        pretrain_epochs=pretrain_epochs,
        batch_size=resolved_batch_size,
        device=device,
        smoke_test=smoke_test,
        evidence_level=evidence_level,
        scope_label=scope_label,
        planned_scope_complete=planned_complete,
        primary_scope_complete=primary_complete,
    )

    artefacts = {
        "summary": "summary.json",
        "config_snapshot": "config_snapshot.json",
        "run_plan": "run_plan.csv",
        "results_summary": "results_summary.csv",
        "aggregate_summary": "aggregate_summary.csv",
        "aggregate_summary_json": "aggregate_summary.json",
        "training_curves_summary": "training_curves_summary.csv",
        "ssl_comparison": "ssl_comparison.csv",
        "missing_pairs": "missing_pairs.csv",
        "failures": "failures.csv",
        "runs": "runs/",
        "readme": "README.md",
        "sha256_manifest": "sha256_manifest.json",
    }
    summary = FI2010ProperTrainingSummary(
        output_dir=str(resolved_out),
        config_path=str(resolved_config),
        processed_root=str(resolved_processed),
        folds=list(selected_folds),
        horizons=list(selected_horizons),
        seeds=list(selected_seeds),
        lookbacks=list(selected_lookbacks),
        models=list(selected_models),
        objectives=list(selected_objectives),
        pretrain_epochs=int(pretrain_epochs),
        max_epochs=resolved_max_epochs,
        early_stopping_patience=resolved_patience,
        early_stopping_metric=monitored_metric,
        batch_size=resolved_batch_size,
        device=device,
        smoke_test=smoke_test,
        execution_mode="smoke" if smoke_test else "benchmark",
        run_count=len(specs),
        completed_run_count=completed,
        failed_run_count=failed,
        skipped_existing_count=skipped,
        missing_pair_count=len(missing_pairs),
        planned_scope_complete=planned_complete,
        target_scope_complete=primary_complete,
        scope_label=scope_label,
        artefacts=artefacts,
        runner_version=FI2010_PROPER_TRAINING_VERSION,
        created_at=datetime.now(UTC),
        git_commit=git_commit,
        warnings=warnings,
    )
    payload = summary.model_dump(mode="json")
    payload["early_stopping"] = {
        "metric": monitored_metric,
        "patience": resolved_patience,
        "validation_only_model_selection": True,
        "restore_best_checkpoint_before_test": True,
        "no_test_set_selection": True,
    }
    payload["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_version": __version__,
    }
    payload["aggregate_rows"] = len(aggregate_rows)
    payload["comparison_rows"] = len(comparison_rows)
    payload["evidence_level"] = evidence_level
    payload["scope"] = _scope_metadata(
        scope_label=scope_label,
        evidence_level=evidence_level,
        planned_scope_complete=planned_complete,
        primary_scope_complete=primary_complete,
    )
    (resolved_out / "summary.json").write_text(stable_json_dumps(payload), encoding="utf-8")
    _write_output_manifest(resolved_out)
    return summary


def _primary_target_complete(
    *,
    folds: Sequence[int],
    horizons: Sequence[int],
    seeds: Sequence[int],
    lookbacks: Sequence[int],
    models: Sequence[str],
    objectives: Sequence[str],
    max_epochs: int,
    patience: int,
    planned_complete: bool,
    smoke_test: bool,
) -> bool:
    if smoke_test or not planned_complete:
        return False
    legacy_primary = (
        set(folds).issuperset(_PRIMARY_TARGET_FOLDS)
        and set(horizons).issuperset(_PRIMARY_TARGET_HORIZONS)
        and set(seeds).issuperset(_PRIMARY_TARGET_SEEDS)
        and set(lookbacks).issuperset(_PRIMARY_TARGET_LOOKBACKS)
        and "matrix_transformer" in set(models)
        and set(objectives).issuperset(_PRIMARY_TARGET_OBJECTIVES)
        and int(max_epochs) >= _PRIMARY_TARGET_MAX_EPOCHS
        and int(patience) >= _PRIMARY_TARGET_PATIENCE
    )
    broader_primary = (
        set(folds).issuperset(_BROADER_TARGET_FOLDS)
        and set(horizons).issuperset(_BROADER_TARGET_HORIZONS)
        and set(seeds).issuperset(_BROADER_TARGET_SEEDS)
        and set(lookbacks).issuperset(_BROADER_TARGET_LOOKBACKS)
        and set(models).issuperset(_BROADER_TARGET_MODELS)
        and set(objectives) == _BROADER_TARGET_OBJECTIVES
        and int(max_epochs) >= _PRIMARY_TARGET_MAX_EPOCHS
        and int(patience) >= _PRIMARY_TARGET_PATIENCE
    )
    return legacy_primary or broader_primary


def _scope_label(
    *,
    folds: Sequence[int],
    horizons: Sequence[int],
    seeds: Sequence[int],
    lookbacks: Sequence[int],
    models: Sequence[str],
    objectives: Sequence[str],
    max_epochs: int,
    patience: int,
    planned_complete: bool,
    primary_complete: bool,
    smoke_test: bool,
) -> str:
    if smoke_test:
        return "smoke_test_only"
    if (
        primary_complete
        and set(models).issuperset(_BROADER_TARGET_MODELS)
        and set(objectives) == _BROADER_TARGET_OBJECTIVES
    ):
        return "broader_proper_training_complete"
    if primary_complete:
        return "primary_credible_minimum"
    if (
        planned_complete
        and set(folds) == _FALLBACK_TARGET_FOLDS
        and set(horizons) == _PRIMARY_TARGET_HORIZONS
        and set(seeds) == _PRIMARY_TARGET_SEEDS
        and set(objectives) == _PRIMARY_TARGET_OBJECTIVES
        and int(max_epochs) >= _PRIMARY_TARGET_MAX_EPOCHS
        and int(patience) >= _PRIMARY_TARGET_PATIENCE
    ):
        if set(lookbacks) == _PRIMARY_TARGET_LOOKBACKS:
            return "fallback_credible_slice"
        return "fallback_limited_lookback_slice"
    if max_epochs <= 2 or max(lookbacks, default=0) < 20:
        return "tiny_or_limited_partial_slice"
    if len(folds) <= 1 or len(horizons) <= 1:
        return "limited_partial_real_slice"
    return "documented_partial_real"


def _scope_metadata(
    *,
    scope_label: str,
    evidence_level: str,
    planned_scope_complete: bool,
    primary_scope_complete: bool,
) -> dict[str, Any]:
    return {
        "scope_label": scope_label,
        "evidence_level": evidence_level,
        "planned_scope_complete": bool(planned_scope_complete),
        "primary_scope_complete": bool(primary_scope_complete),
        "primary_complete_real_target": {
            "folds": sorted(_PRIMARY_TARGET_FOLDS),
            "horizons": sorted(_PRIMARY_TARGET_HORIZONS),
            "seeds": sorted(_PRIMARY_TARGET_SEEDS),
            "lookbacks": sorted(_PRIMARY_TARGET_LOOKBACKS),
            "objectives": list(PROPER_TRAINING_OBJECTIVE_CHOICES),
            "min_max_epochs": _PRIMARY_TARGET_MAX_EPOCHS,
            "min_patience": _PRIMARY_TARGET_PATIENCE,
        },
        "broader_complete_real_target": {
            "folds": sorted(_BROADER_TARGET_FOLDS),
            "horizons": sorted(_BROADER_TARGET_HORIZONS),
            "seeds": sorted(_BROADER_TARGET_SEEDS),
            "lookbacks": sorted(_BROADER_TARGET_LOOKBACKS),
            "models": sorted(_BROADER_TARGET_MODELS),
            "objectives": sorted(_BROADER_TARGET_OBJECTIVES),
            "min_max_epochs": _PRIMARY_TARGET_MAX_EPOCHS,
            "min_patience": _PRIMARY_TARGET_PATIENCE,
        },
        "fallback_partial_real_target": {
            "folds": sorted(_FALLBACK_TARGET_FOLDS),
            "horizons": sorted(_PRIMARY_TARGET_HORIZONS),
            "seeds": sorted(_PRIMARY_TARGET_SEEDS),
            "lookbacks": sorted(_PRIMARY_TARGET_LOOKBACKS),
            "objectives": list(PROPER_TRAINING_OBJECTIVE_CHOICES),
            "min_max_epochs": _PRIMARY_TARGET_MAX_EPOCHS,
            "min_patience": _PRIMARY_TARGET_PATIENCE,
        },
    }


def _expected_reuse_signature(
    spec: FI2010ProperTrainingRunSpec,
    *,
    config: NeuralBenchmarkConfig,
    max_epochs: int,
    patience: int,
    batch_size: int,
    pretrain_epochs: int,
    device: str,
    smoke_test: bool,
    mask_probability: float,
    bucket_count: int,
) -> dict[str, Any]:
    model_spec = config.neural_models[spec.model_name]
    dropout = config.training.dropout if model_spec.dropout is None else float(model_spec.dropout)
    signature: dict[str, Any] = {
        "runner_version": FI2010_PROPER_TRAINING_VERSION,
        "run_id": spec.run_id,
        "fold": spec.fold,
        "horizon": spec.horizon,
        "seed": spec.seed,
        "lookback": spec.lookback,
        "model_family": spec.model_family,
        "objective": spec.objective,
        "pretraining_objective": spec.pretraining_objective,
        "max_epochs": int(max_epochs),
        "early_stopping_patience": int(patience),
        "early_stopping_metric": config.training.early_stopping_metric,
        "learning_rate": float(config.training.learning_rate),
        "weight_decay": float(config.training.weight_decay),
        "dropout": float(dropout),
        "batch_size": int(batch_size),
        "pretrain_epochs": 0 if spec.objective == "supervised" else int(pretrain_epochs),
        "device": str(device),
        "smoke_test": bool(smoke_test),
    }
    if spec.model_name == "matrix_transformer":
        signature.update(
            {
                "transformer_model_dim": int(model_spec.model_dim or 16),
                "transformer_num_heads": int(model_spec.num_heads or 2),
                "transformer_num_layers": int(model_spec.num_layers or 1),
                "transformer_feedforward_dim": int(model_spec.feedforward_dim or 32),
            }
        )
    else:
        signature.update(
            {
                "deeplob_conv_channels": int(model_spec.conv_channels or 16),
                "deeplob_lstm_hidden_size": int(model_spec.lstm_hidden_size or 32),
                "deeplob_use_batch_norm": bool(model_spec.use_batch_norm),
            }
        )
    if spec.objective != "supervised":
        signature["mask_probability"] = float(mask_probability)
        signature["next_field_bucket_count"] = int(bucket_count)
    return signature


def _proper_training_settings(
    *,
    config: NeuralBenchmarkConfig,
    model_name: str,
    lookback: int,
    max_epochs: int,
    batch_size: int,
    device: str,
) -> PaperNeuralSettings:
    spec = config.neural_models[model_name]
    dropout = config.training.dropout if spec.dropout is None else float(spec.dropout)
    return PaperNeuralSettings(
        supported_models=PROPER_TRAINING_MODEL_CHOICES,
        planned_models=(),
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
        deeplob_conv_channels=int(spec.conv_channels or 16),
        deeplob_lstm_hidden_size=int(spec.lstm_hidden_size or 32),
        deeplob_use_batch_norm=bool(spec.use_batch_norm),
        transformer_model_dim=int(spec.model_dim or 16),
        transformer_num_heads=int(spec.num_heads or 2),
        transformer_num_layers=int(spec.num_layers or 1),
        transformer_feedforward_dim=int(spec.feedforward_dim or 32),
    )


def _execute_run_spec(
    spec: FI2010ProperTrainingRunSpec,
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
    settings = (
        build_ssl_finetune_settings(
            config=config,
            lookback=spec.lookback,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
        )
        if spec.model_name == "matrix_transformer"
        else _proper_training_settings(
            config=config,
            model_name=spec.model_name,
            lookback=spec.lookback,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
        )
    )

    pretrain_payload: dict[str, Any] | None = None
    encoder_state = None
    checkpoint_hash = None
    if spec.model_name != "matrix_transformer" and spec.objective != "supervised":
        raise ValueError(
            "DeepLOB-style proper training supports the supervised objective only"
        )
    if spec.objective == "supervised":
        init_source = "random_init"
        model_type = (
            "normalised_matrix_transformer"
            if spec.model_name == "matrix_transformer"
            else "deeplob_style"
        )
    else:
        flags = ssl_objective_flags(_ssl_runner_objective(spec.objective))
        pretrain = pretrain_matrix_ssl_encoder(
            inputs=inputs,
            config=config,
            seed=spec.seed,
            lookback=spec.lookback,
            flags=flags,
            mask_probability=mask_probability,
            bucket_count=bucket_count,
            pretrain_epochs=pretrain_epochs,
            batch_size=batch_size,
            device=device,
            pretrain_dir=run_dir / "pretrain",
            git_commit=git_commit,
        )
        encoder_state = pretrain["encoder_state"]
        checkpoint_hash = pretrain["checkpoint_sha256"]
        pretrain_payload = {
            "objective": pretrain.get("objective"),
            "pretrain_epochs": int(pretrain_epochs),
            "mask_probability": float(mask_probability),
            "next_field_bucket_count": int(bucket_count),
            "train_window_count": pretrain.get("train_window_count"),
            "validation_window_count": pretrain.get("validation_window_count"),
            "final_train_loss": pretrain.get("final_train_loss"),
            "final_validation_loss": pretrain.get("final_validation_loss"),
            "encoder_parameter_count": pretrain.get("encoder_parameter_count"),
            "checkpoint_sha256": checkpoint_hash,
            "data_policy": "training rows only; train-carved validation diagnostics",
        }
        init_source = "ssl_pretrained"
        model_type = "ssl_finetuned_matrix_transformer"

    if spec.model_name == "matrix_transformer":
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
            pretrained_encoder_state=encoder_state,
            init_source=init_source,
            model_type=model_type,
        )
    else:
        result = run_neural_paper_model(
            model_name=spec.model_name,
            frame=frame,
            config=inputs.config,
            data_path=fold_path,
            split=inputs.split,
            feature_columns=inputs.feature_columns,
            all_labels=inputs.all_labels,
            class_count_train=inputs.class_count_train,
            class_count_test=inputs.class_count_test,
            test_timestamps=None,
            settings=settings,
        )

    predictions = _canonical_prediction_rows(result, spec=spec)
    _write_csv(predictions, run_dir / "predictions.csv", _PREDICTION_COLUMNS)
    curve_rows = _curve_rows(result, learning_rate=settings.learning_rate)
    _write_csv(curve_rows, run_dir / "curves.csv", _CURVE_COLUMNS)
    (run_dir / "curves.json").write_text(
        stable_json_dumps({"run_id": spec.run_id, "curves": curve_rows}),
        encoding="utf-8",
    )

    metrics = _resolve_metrics(result.metrics, predictions)
    architecture = {
        "model_family": spec.model_family,
        "dropout": float(settings.dropout),
        "effective_window": int(inputs.effective_window),
    }
    if spec.model_name == "matrix_transformer":
        architecture.update(
            {
                "model_dim": int(settings.transformer_model_dim),
                "num_heads": int(settings.transformer_num_heads),
                "num_layers": int(settings.transformer_num_layers),
                "feedforward_dim": int(settings.transformer_feedforward_dim),
            }
        )
    else:
        architecture.update(
            {
                "conv_channels": int(settings.deeplob_conv_channels),
                "lstm_hidden_size": int(settings.deeplob_lstm_hidden_size),
                "use_batch_norm": bool(settings.deeplob_use_batch_norm),
            }
        )
    architecture_hash = _hash_payload(architecture)
    preprocessing_hash = _hash_payload(
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
    best_epoch = _int_or_none(metadata.get("best_epoch"))
    best_validation_score = _best_validation_score(curve_rows, best_epoch=best_epoch)
    epochs_ran = _int_or_none(metadata.get("epochs_ran")) or len(curve_rows)
    early_stopped = bool(metadata.get("early_stopped", False))
    training_seconds = _safe_float(metadata.get("training_seconds"))
    reuse_signature = _expected_reuse_signature(
        spec,
        config=config,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        pretrain_epochs=pretrain_epochs,
        device=device,
        smoke_test=smoke_test,
        mask_probability=mask_probability,
        bucket_count=bucket_count,
    )

    config_snapshot = {
        "run_id": spec.run_id,
        "reuse_signature": reuse_signature,
        "fold": spec.fold,
        "horizon": spec.horizon,
        "seed": spec.seed,
        "lookback": spec.lookback,
        "model_family": spec.model_family,
        "objective": spec.objective,
        "init_source": init_source,
        "model_type": model_type,
        "max_epochs": int(max_epochs),
        "early_stopping_patience": int(patience),
        "early_stopping_metric": monitored_metric,
        "learning_rate": float(settings.learning_rate),
        "weight_decay": float(settings.weight_decay),
        "dropout": float(settings.dropout),
        "batch_size": int(batch_size),
        "effective_window": int(inputs.effective_window),
        "pretrain_epochs": int(pretrain_epochs) if pretrain_payload else 0,
        "device": device,
        "smoke_test": bool(smoke_test),
        "split_summary": inputs.split_summary,
    }
    config_snapshot.update(architecture)
    (run_dir / "config.json").write_text(stable_json_dumps(config_snapshot), encoding="utf-8")
    (run_dir / "git_commit.txt").write_text(git_commit or "", encoding="utf-8")

    metric_payload = {
        "runner_version": FI2010_PROPER_TRAINING_VERSION,
        "subset_kind": "proper_training_subset",
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
        "init_source": init_source,
        "model_type": model_type,
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
        "ssl_pretraining": pretrain_payload,
        "checkpoint_hash": checkpoint_hash,
        "prediction_file": _relative_path(run_dir / "predictions.csv", out_dir),
        "curves_file": _relative_path(run_dir / "curves.csv", out_dir),
        "architecture_hash": architecture_hash,
        "preprocessing_hash": preprocessing_hash,
        "git_commit": git_commit,
        "class_label_mapping": {
            "prob_up": "FI-2010 label 1",
            "prob_stationary": "FI-2010 label 2",
            "prob_down": "FI-2010 label 3",
        },
    }
    (run_dir / "metrics.json").write_text(stable_json_dumps(metric_payload), encoding="utf-8")
    _write_status(run_dir, "completed")
    _write_run_log(
        run_dir,
        {
            "run_id": spec.run_id,
            "status": "completed",
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now(UTC).isoformat(),
        },
    )
    _write_run_manifest(run_dir)
    result_row = _result_row_from_payload(metric_payload, out_dir=out_dir, run_dir=run_dir)
    training_row = _training_row_from_payload(metric_payload)
    return result_row, training_row


# ---------------------------------------------------------------------------
# Result / training-row construction
# ---------------------------------------------------------------------------


def _result_row_from_payload(
    payload: Mapping[str, Any],
    *,
    out_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    metrics = cast(Mapping[str, Any], payload["metrics"])
    return {
        "fold": int(payload["fold"]),
        "horizon": int(payload["horizon"]),
        "seed": int(payload["seed"]),
        "lookback": int(payload["lookback"]),
        "model_family": str(payload["model_family"]),
        "pretraining_objective": str(payload["pretraining_objective"]),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "mcc": metrics.get("mcc"),
        "ece": metrics.get("ece"),
        "brier_score": metrics.get("brier_score"),
        "nll": metrics.get("nll"),
        "class_f1_down": metrics.get("class_f1_down"),
        "class_f1_stationary": metrics.get("class_f1_stationary"),
        "class_f1_up": metrics.get("class_f1_up"),
        "checkpoint_hash": payload.get("checkpoint_hash"),
        "prediction_file": payload.get("prediction_file"),
        "status": str(payload.get("status", "completed")),
        "run_id": payload.get("run_id"),
        "run_dir": _relative_path(run_dir, out_dir),
        "architecture_hash": payload.get("architecture_hash"),
        "preprocessing_hash": payload.get("preprocessing_hash"),
    }


def _training_row_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = cast(Mapping[str, Any], payload["metrics"])
    training = cast(Mapping[str, Any], payload["training"])
    return {
        "run_id": payload.get("run_id"),
        "fold": int(payload["fold"]),
        "horizon": int(payload["horizon"]),
        "seed": int(payload["seed"]),
        "lookback": int(payload["lookback"]),
        "model_family": str(payload["model_family"]),
        "objective": str(payload["objective"]),
        "pretraining_objective": str(payload["pretraining_objective"]),
        "max_epochs": training.get("max_epochs"),
        "epochs_ran": training.get("epochs_ran"),
        "best_epoch": training.get("best_epoch"),
        "monitored_metric": training.get("monitored_metric"),
        "best_validation_score": training.get("best_validation_score"),
        "early_stopping_patience": training.get("early_stopping_patience"),
        "early_stopped": training.get("early_stopped"),
        "training_seconds": training.get("training_seconds"),
        "test_macro_f1": metrics.get("macro_f1"),
        "test_accuracy": metrics.get("accuracy"),
        "test_mcc": metrics.get("mcc"),
        "test_ece": metrics.get("ece"),
        "status": str(payload.get("status", "completed")),
    }


def _load_existing_rows(
    run_dir: Path,
    *,
    out_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _read_json(run_dir / "metrics.json")
    payload["status"] = "skipped_existing"
    result_row = _result_row_from_payload(payload, out_dir=out_dir, run_dir=run_dir)
    training_row = _training_row_from_payload(payload)
    return result_row, training_row


# ---------------------------------------------------------------------------
# Predictions, curves and metrics
# ---------------------------------------------------------------------------


def _canonical_prediction_rows(
    result: NeuralPaperModelResult,
    *,
    spec: FI2010ProperTrainingRunSpec,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for position, raw in enumerate(result.prediction_rows):
        prob_up = _prob_value(raw, 1)
        prob_stationary = _prob_value(raw, 2)
        prob_down = _prob_value(raw, 3)
        confidence = _safe_float(raw.get("confidence"))
        if confidence is None:
            confidence = _max_probability(prob_up, prob_stationary, prob_down)
        rows.append(
            {
                "row_id": _json_scalar(raw.get("row_index")),
                "sample_id": int(position),
                "fold": int(spec.fold),
                "horizon": int(spec.horizon),
                "seed": int(spec.seed),
                "lookback": int(spec.lookback),
                "model_family": spec.model_family,
                "pretraining_objective": spec.pretraining_objective,
                "split": str(raw.get("split", "test")),
                "y_true": _json_scalar(raw.get("label")),
                "y_pred": _json_scalar(raw.get("prediction")),
                "prob_down": prob_down,
                "prob_stationary": prob_stationary,
                "prob_up": prob_up,
                "confidence": confidence,
            }
        )
    if not rows:
        raise ValueError(f"{spec.model_family} produced no test predictions")
    return rows


def _curve_rows(
    result: NeuralPaperModelResult,
    *,
    learning_rate: float,
) -> list[dict[str, Any]]:
    history = result.metadata.get("training_history")
    rows: list[dict[str, Any]] = []
    if not isinstance(history, Sequence):
        return rows
    for item in history:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            {
                "epoch": _int_or_none(item.get("epoch")),
                "train_loss": _safe_float(item.get("train_loss")),
                "validation_loss": _safe_float(item.get("validation_loss")),
                "validation_accuracy": _safe_float(item.get("validation_accuracy")),
                "validation_macro_f1": _safe_float(item.get("validation_macro_f1")),
                "validation_mcc": _safe_float(item.get("validation_mcc")),
                "monitored_value": _safe_float(item.get("monitored_value")),
                "learning_rate": float(learning_rate),
                "is_best": bool(item.get("is_best", False)),
                "early_stop": bool(item.get("early_stop", False)),
            }
        )
    return rows


def _best_validation_score(
    curve_rows: Sequence[Mapping[str, Any]],
    *,
    best_epoch: int | None,
) -> float | None:
    if best_epoch is not None:
        for row in curve_rows:
            if row.get("epoch") == best_epoch:
                return _safe_float(row.get("monitored_value"))
    best: float | None = None
    for row in curve_rows:
        if bool(row.get("is_best")):
            best = _safe_float(row.get("monitored_value"))
    return best


def _resolve_metrics(
    raw_metrics: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    derived = _derive_class_metrics(predictions)
    return {
        "accuracy": _first_not_none(
            _safe_float(raw_metrics.get("accuracy")),
            derived["accuracy"],
        ),
        "macro_f1": _first_not_none(
            _safe_float(raw_metrics.get("macro_f1")),
            derived["macro_f1"],
        ),
        "mcc": _safe_float(raw_metrics.get("matthews_corrcoef")),
        "ece": _first_not_none(
            _safe_float(raw_metrics.get("expected_calibration_error")),
            _safe_float(raw_metrics.get("ece")),
            derived["ece"],
        ),
        "brier_score": _first_not_none(
            _safe_float(raw_metrics.get("brier_score")),
            derived["brier_score"],
        ),
        "nll": _first_not_none(
            _safe_float(raw_metrics.get("log_loss")),
            _safe_float(raw_metrics.get("nll")),
            derived["nll"],
        ),
        "class_f1_down": derived["class_f1_down"],
        "class_f1_stationary": derived["class_f1_stationary"],
        "class_f1_up": derived["class_f1_up"],
    }


def _derive_class_metrics(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    true = [_direction_label(row.get("y_true")) for row in predictions]
    pred = [_direction_label(row.get("y_pred")) for row in predictions]
    empty: dict[str, float | None] = {
        "accuracy": None,
        "macro_f1": None,
        "class_f1_down": None,
        "class_f1_stationary": None,
        "class_f1_up": None,
        "brier_score": None,
        "nll": None,
        "ece": None,
    }
    if any(item is None for item in true) or any(item is None for item in pred):
        return empty
    true_labels = cast(list[str], true)
    pred_labels = cast(list[str], pred)
    classes = ("down", "stationary", "up")
    f1_by_class = {
        label: _binary_f1(true_labels, pred_labels, positive_label=label) for label in classes
    }
    accuracy = sum(
        1 for actual, predicted in zip(true_labels, pred_labels, strict=True) if actual == predicted
    ) / len(true_labels)
    probabilities = _probability_matrix(predictions)
    brier = nll = ece = None
    if probabilities is not None:
        brier = _brier_score(true_labels, probabilities)
        nll = _negative_log_likelihood(true_labels, probabilities)
        ece = _ece(true_labels, pred_labels, probabilities)
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(sum(f1_by_class.values()) / len(f1_by_class)),
        "class_f1_down": f1_by_class["down"],
        "class_f1_stationary": f1_by_class["stationary"],
        "class_f1_up": f1_by_class["up"],
        "brier_score": brier,
        "nll": nll,
        "ece": ece,
    }


# ---------------------------------------------------------------------------
# Failure / status artefacts
# ---------------------------------------------------------------------------


def _write_failure_artefacts(
    spec: FI2010ProperTrainingRunSpec,
    *,
    run_dir: Path,
    reason: str,
    traceback_text: str,
    git_commit: str | None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "git_commit.txt").write_text(git_commit or "", encoding="utf-8")
    payload = {
        "runner_version": FI2010_PROPER_TRAINING_VERSION,
        "subset_kind": "proper_training_subset",
        "run_id": spec.run_id,
        "status": "failed",
        "fold": spec.fold,
        "horizon": spec.horizon,
        "seed": spec.seed,
        "lookback": spec.lookback,
        "model_family": spec.model_family,
        "objective": spec.objective,
        "pretraining_objective": spec.pretraining_objective,
        "error": reason,
        "git_commit": git_commit,
    }
    (run_dir / "metrics.json").write_text(stable_json_dumps(payload), encoding="utf-8")
    _write_status(run_dir, "failed")
    _write_run_log(
        run_dir,
        {
            "run_id": spec.run_id,
            "status": "failed",
            "reason": reason,
            "traceback": traceback_text,
            "ended_at": datetime.now(UTC).isoformat(),
        },
    )
    _write_run_manifest(run_dir)


def _failure_row(
    spec: FI2010ProperTrainingRunSpec,
    *,
    status: str,
    reason: str,
    traceback_text: str,
    invalidates: bool,
) -> dict[str, Any]:
    return {
        "fold": int(spec.fold),
        "horizon": int(spec.horizon),
        "seed": int(spec.seed),
        "model_family": spec.model_family,
        "objective": spec.objective,
        "reason": reason,
        "traceback": traceback_text,
        "invalidates_aggregate_claims": bool(invalidates),
        "status": status,
        "run_id": spec.run_id,
    }


def _write_run_manifest(run_dir: Path) -> None:
    files = [
        path
        for path in (
            run_dir / "config.json",
            run_dir / "metrics.json",
            run_dir / "predictions.csv",
            run_dir / "curves.csv",
            run_dir / "curves.json",
            run_dir / "git_commit.txt",
            run_dir / "run_log.json",
            run_dir / "status.txt",
        )
        if path.is_file()
    ]
    payload = {
        "runner_version": FI2010_PROPER_TRAINING_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "sha256": {path.name: sha256_file(path) for path in files},
        "artefacts": {path.name: path.name for path in files},
    }
    (run_dir / "sha256_manifest.json").write_text(stable_json_dumps(payload), encoding="utf-8")


def _write_output_manifest(out_dir: Path) -> None:
    root_manifest = (out_dir / "sha256_manifest.json").resolve(strict=False)
    files = [
        path
        for path in sorted(Path(out_dir).rglob("*"))
        if path.is_file() and path.resolve(strict=False) != root_manifest
    ]
    payload = {
        "runner_version": FI2010_PROPER_TRAINING_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "sha256": {
            _relative_path(path, out_dir): sha256_file(path)
            for path in files
        },
    }
    (out_dir / "sha256_manifest.json").write_text(
        stable_json_dumps(payload),
        encoding="utf-8",
    )


def _write_root_config_snapshot(
    out_dir: Path,
    *,
    config_path: Path,
    processed_root: Path,
    folds: Sequence[int],
    horizons: Sequence[int],
    seeds: Sequence[int],
    lookbacks: Sequence[int],
    models: Sequence[str],
    objectives: Sequence[str],
    max_epochs: int,
    patience: int,
    pretrain_epochs: int,
    batch_size: int,
    device: str,
    smoke_test: bool,
    evidence_level: str,
    scope_label: str,
    planned_scope_complete: bool,
    primary_scope_complete: bool,
) -> None:
    payload = {
        "runner_version": FI2010_PROPER_TRAINING_VERSION,
        "config_path": str(config_path),
        "processed_root": str(processed_root),
        "folds": list(folds),
        "horizons": list(horizons),
        "seeds": list(seeds),
        "lookbacks": list(lookbacks),
        "models": list(models),
        "objectives": list(objectives),
        "max_epochs": int(max_epochs),
        "early_stopping_patience": int(patience),
        "pretrain_epochs": int(pretrain_epochs),
        "batch_size": int(batch_size),
        "device": device,
        "smoke_test": bool(smoke_test),
        "evidence_level": evidence_level,
        "scope": _scope_metadata(
            scope_label=scope_label,
            evidence_level=evidence_level,
            planned_scope_complete=planned_scope_complete,
            primary_scope_complete=primary_scope_complete,
        ),
    }
    (out_dir / "config_snapshot.json").write_text(
        stable_json_dumps(payload),
        encoding="utf-8",
    )


def _write_proper_aggregate_summary_json(
    out_dir: Path,
    *,
    aggregate_rows: Sequence[Mapping[str, Any]],
    missing_pair_count: int,
    result_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
) -> None:
    payload = {
        "runner_version": FI2010_PROPER_TRAINING_VERSION,
        "subset_kind": "proper_training_subset",
        "created_at": datetime.now(UTC).isoformat(),
        "paths": {
            "results_summary": "results_summary.csv",
            "aggregate_summary": "aggregate_summary.csv",
            "failures": "failures.csv",
            "ssl_comparison": "ssl_comparison.csv",
            "missing_pairs": "missing_pairs.csv",
            "training_curves_summary": "training_curves_summary.csv",
        },
        "completed_run_count": len(
            [row for row in result_rows if row.get("status") in _SUCCESS_STATUSES]
        ),
        "failed_run_count": len([row for row in failure_rows if row.get("status") == "failed"]),
        "skipped_existing_count": len(
            [row for row in failure_rows if row.get("status") == "skipped_existing"]
        ),
        "missing_pair_count": int(missing_pair_count),
        "aggregate": [dict(row) for row in aggregate_rows],
    }
    (out_dir / "aggregate_summary.json").write_text(
        stable_json_dumps(payload),
        encoding="utf-8",
    )


def _write_status(run_dir: Path, status: str) -> None:
    if status not in {"completed", "failed", "skipped_existing"}:
        raise ValueError(f"unsupported run status: {status}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.txt").write_text(status + "\n", encoding="utf-8")


def _write_run_log(run_dir: Path, payload: Mapping[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_log.json").write_text(stable_json_dumps(dict(payload)), encoding="utf-8")


def _completed_run_present(
    run_dir: Path,
    *,
    expected_signature: Mapping[str, Any] | None = None,
) -> bool:
    status_path = run_dir / "status.txt"
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.csv"
    curves_path = run_dir / "curves.csv"
    curves_json_path = run_dir / "curves.json"
    config_path = run_dir / "config.json"
    if not (
        status_path.is_file()
        and metrics_path.is_file()
        and predictions_path.is_file()
        and curves_path.is_file()
        and curves_json_path.is_file()
        and config_path.is_file()
    ):
        return False
    if status_path.read_text(encoding="utf-8").strip() not in _SUCCESS_STATUSES:
        return False
    if expected_signature is None:
        return True
    return _config_signature_matches(config_path, expected_signature)


def _config_signature_matches(
    config_path: Path,
    expected_signature: Mapping[str, Any],
) -> bool:
    try:
        payload = _read_json(config_path)
    except (OSError, ValueError):
        return False
    raw_signature = payload.get("reuse_signature")
    if isinstance(raw_signature, Mapping):
        existing = dict(raw_signature)
        return all(
            _signature_values_equal(existing.get(key), value)
            for key, value in expected_signature.items()
        )
    comparable_keys = set(expected_signature).intersection(payload)
    required_legacy_keys = {
        "run_id",
        "fold",
        "horizon",
        "seed",
        "lookback",
        "objective",
        "max_epochs",
        "early_stopping_patience",
        "early_stopping_metric",
        "learning_rate",
        "weight_decay",
        "dropout",
        "batch_size",
        "transformer_model_dim",
        "transformer_num_heads",
        "transformer_num_layers",
        "transformer_feedforward_dim",
        "pretrain_epochs",
        "device",
        "smoke_test",
    }
    if not required_legacy_keys.issubset(comparable_keys):
        return False
    return all(
        _signature_values_equal(payload.get(key), expected_signature.get(key))
        for key in comparable_keys
    )


def _signature_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return _bool_value(left) == _bool_value(right)
    left_float = _safe_float(left)
    right_float = _safe_float(right)
    if left_float is not None and right_float is not None:
        return abs(left_float - right_float) <= 1e-12
    return str(left) == str(right)


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned == "true":
            return True
        if cleaned == "false":
            return False
    return bool(value)


def _remove_run_dir(run_dir: Path, *, root: Path) -> None:
    resolved_root = root.resolve(strict=False)
    resolved_run = run_dir.resolve(strict=False)
    try:
        resolved_run.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"refusing to remove run outside output root: {run_dir}") from exc
    shutil.rmtree(resolved_run)


def _run_plan_row(
    spec: FI2010ProperTrainingRunSpec,
    out_dir: Path,
) -> dict[str, Any]:
    return {
        "run_id": spec.run_id,
        "fold": spec.fold,
        "fold_id": spec.fold_id,
        "horizon": spec.horizon,
        "seed": spec.seed,
        "lookback": spec.lookback,
        "model_family": spec.model_family,
        "objective": spec.objective,
        "pretraining_objective": spec.pretraining_objective,
        "run_dir": _relative_path(spec.run_dir(out_dir), out_dir),
    }


def _write_subset_readme(
    out_dir: Path,
    *,
    folds: Sequence[int],
    horizons: Sequence[int],
    seeds: Sequence[int],
    lookbacks: Sequence[int],
    models: Sequence[str],
    objectives: Sequence[str],
    max_epochs: int,
    patience: int,
    monitored_metric: str,
    pretrain_epochs: int,
    smoke_test: bool,
    evidence_level: str,
    scope_label: str,
) -> None:
    lines = [
        "# FI-2010 Proper-Training Neural Subset",
        "",
        "This directory is generated by `run-fi2010-neural-proper-training-subset`.",
        "It is longer-training neural modelling evidence and is reported separately "
        "from the one-epoch matched full grid.",
        "",
        "The one-epoch full grid remains the matched comparison and infrastructure "
        "evidence.",
        "This subset is used to assess whether the neural models remain credible "
        "under a more realistic training budget with early stopping and "
        "validation-only model selection.",
        "",
        "## Training Protocol",
        "",
        f"- max epochs: {max_epochs}",
        f"- early stopping metric: {monitored_metric} (validation only)",
        f"- early stopping patience: {patience}",
        "- best validation checkpoint is restored before the single official test evaluation",
        "- no model is ever selected on test metrics",
        f"- SSL pretraining epochs (SSL objectives only): {pretrain_epochs}",
        "- SSL pretraining consumes official training rows only",
        "",
        "## Scope",
        "",
        f"- folds: {', '.join(str(item) for item in folds)}",
        f"- horizons: {', '.join(str(item) for item in horizons)}",
        f"- seeds: {', '.join(str(item) for item in seeds)}",
        f"- lookbacks: {', '.join(str(item) for item in lookbacks)}",
        f"- models: {', '.join(models)}",
        f"- objectives: {', '.join(objectives)}",
        f"- smoke test: {'yes' if smoke_test else 'no'}",
        f"- evidence level: {evidence_level}",
        f"- scope label: {scope_label}",
        "",
        "## Outputs",
        "",
        "- per-run predictions: `runs/**/predictions.csv`",
        "- per-run training curves: `runs/**/curves.csv`, `runs/**/curves.json`",
        "- per-run metrics, best epoch and SHA256 hashes: `runs/**/metrics.json`, "
        "`runs/**/sha256_manifest.json`",
        "- root config snapshot and SHA256 inventory: `config_snapshot.json`, "
        "`sha256_manifest.json`",
        "- completed-run table: `results_summary.csv`",
        "- training-curve summary (best epoch, early stopping): `training_curves_summary.csv`",
        "- grouped aggregate table: `aggregate_summary.csv`, `aggregate_summary.json`",
        "- matched supervised-vs-SSL deltas: `ssl_comparison.csv`",
        "- failed or reused-existing runs: `failures.csv`",
        "",
        "## Limitations",
        "",
        "- Smoke-test artefacts are code-path checks only and are not empirical evidence.",
        "- A documented partial scope is classified `partial_real`; the report and "
        "evidence pack say exactly what was run.",
        "- SSL pretraining is compared under matched conditions; the artefacts do "
        "not support a broad SSL improvement claim unless the matched deltas show "
        "it.",
        "- The runner reports predictive and calibration metrics only; it makes no "
        "trading, profitability or state-of-the-art claim.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


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
        return PROPER_TRAINING_OBJECTIVE_CHOICES
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "all":
            return PROPER_TRAINING_OBJECTIVE_CHOICES
        tokens = [token.strip() for token in text.split(",")]
    else:
        tokens = [str(token).strip() for token in value]
    cleaned: list[str] = []
    for token in tokens:
        objective = token.lower()
        if objective not in PROPER_TRAINING_OBJECTIVE_CHOICES:
            raise ValueError(
                f"unsupported objective {token!r}; supported: "
                f"{list(PROPER_TRAINING_OBJECTIVE_CHOICES)}"
            )
        if objective not in cleaned:
            cleaned.append(objective)
    if not cleaned:
        raise ValueError("objectives selection must not be empty")
    return tuple(cleaned)


def _normalise_models(value: Sequence[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ("matrix_transformer",)
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "all":
            return PROPER_TRAINING_MODEL_CHOICES
        tokens = [token.strip() for token in text.split(",")]
    else:
        tokens = [str(token).strip() for token in value]
    cleaned: list[str] = []
    for token in tokens:
        model_name = token.lower()
        if model_name not in PROPER_TRAINING_MODEL_CHOICES:
            raise ValueError(
                f"unsupported proper-training model {token!r}; supported: "
                f"{list(PROPER_TRAINING_MODEL_CHOICES)}"
            )
        if model_name not in cleaned:
            cleaned.append(model_name)
    if not cleaned:
        raise ValueError("models selection must not be empty")
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


def _ssl_runner_objective(objective: str) -> str:
    if objective == "masked_reconstruction":
        return "masked_field"
    if objective == "next_field":
        return "next_field"
    raise ValueError(f"objective has no SSL runner mapping: {objective}")


# ---------------------------------------------------------------------------
# Numeric and IO helpers
# ---------------------------------------------------------------------------


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


def _read_json(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], payload)


def _hash_payload(payload: Mapping[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(stable_json_dumps(dict(payload)).encode("utf-8")).hexdigest()


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    return numeric if math.isfinite(numeric) else None


def _first_not_none(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _int_or_none(value: Any) -> int | None:
    numeric = _safe_float(value)
    return None if numeric is None else int(numeric)


def _json_scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    item = value.item() if hasattr(value, "item") else value
    if isinstance(item, float) and item.is_integer():
        return int(item)
    return item


def _prob_value(row: Mapping[str, Any], fi_label: int) -> float | None:
    candidates = (
        f"probability_{fi_label}",
        f"probability_{fi_label}.0",
        f"probability_{float(fi_label)}",
    )
    for key in candidates:
        if key in row:
            value = _safe_float(row.get(key))
            if value is not None:
                return value
    for key, value in row.items():
        text = str(key)
        if not text.startswith("probability_"):
            continue
        suffix = text.removeprefix("probability_")
        suffix_number = _safe_float(suffix)
        if suffix_number is not None and int(suffix_number) == fi_label:
            return _safe_float(value)
    return None


def _max_probability(*values: float | None) -> float | None:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return float(max(numeric))


def _direction_label(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    aliases = {
        "1": "up",
        "1.0": "up",
        "up": "up",
        "2": "stationary",
        "2.0": "stationary",
        "stationary": "stationary",
        "3": "down",
        "3.0": "down",
        "down": "down",
    }
    return aliases.get(text)


def _binary_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    positive_label: str,
) -> float:
    tp = sum(
        1
        for actual, predicted in zip(y_true, y_pred, strict=True)
        if actual == positive_label and predicted == positive_label
    )
    fp = sum(
        1
        for actual, predicted in zip(y_true, y_pred, strict=True)
        if actual != positive_label and predicted == positive_label
    )
    fn = sum(
        1
        for actual, predicted in zip(y_true, y_pred, strict=True)
        if actual == positive_label and predicted != positive_label
    )
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else float(2 * tp / denom)


def _probability_matrix(
    predictions: Sequence[Mapping[str, Any]],
) -> list[list[float]] | None:
    matrix: list[list[float]] = []
    for row in predictions:
        values = [
            _safe_float(row.get("prob_up")),
            _safe_float(row.get("prob_stationary")),
            _safe_float(row.get("prob_down")),
        ]
        if any(value is None for value in values):
            return None
        numeric = cast(list[float], values)
        total = sum(numeric)
        if total <= 0:
            return None
        matrix.append([value / total for value in numeric])
    return matrix


def _brier_score(
    y_true: Sequence[str],
    probabilities: Sequence[Sequence[float]],
) -> float:
    order = ("up", "stationary", "down")
    index_of = {label: position for position, label in enumerate(order)}
    total = 0.0
    for actual, row in zip(y_true, probabilities, strict=True):
        target = index_of[actual]
        total += sum(
            (probability - (1.0 if position == target else 0.0)) ** 2
            for position, probability in enumerate(row)
        )
    return float(total / len(y_true))


def _negative_log_likelihood(
    y_true: Sequence[str],
    probabilities: Sequence[Sequence[float]],
) -> float:
    order = ("up", "stationary", "down")
    index_of = {label: position for position, label in enumerate(order)}
    total = 0.0
    for actual, row in zip(y_true, probabilities, strict=True):
        total -= math.log(max(float(row[index_of[actual]]), _EPS))
    return float(total / len(y_true))


def _ece(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    probabilities: Sequence[Sequence[float]],
    *,
    n_bins: int = 10,
) -> float:
    confidences = [max(row) for row in probabilities]
    correct = [
        1.0 if actual == predicted else 0.0
        for actual, predicted in zip(y_true, y_pred, strict=True)
    ]
    total = len(y_true)
    ece = 0.0
    for index in range(n_bins):
        lo = index / n_bins
        hi = (index + 1) / n_bins
        selected = [
            pos
            for pos, confidence in enumerate(confidences)
            if (lo <= confidence <= hi if index == 0 else lo < confidence <= hi)
        ]
        if not selected:
            continue
        bin_acc = sum(correct[pos] for pos in selected) / len(selected)
        bin_conf = sum(confidences[pos] for pos in selected) / len(selected)
        ece += abs(bin_acc - bin_conf) * len(selected) / total
    return float(ece)


def _relative_path(path: Path, root: Path) -> str:
    candidate = Path(path)
    try:
        return (
            candidate.resolve(strict=False).relative_to(Path(root).resolve(strict=False)).as_posix()
        )
    except ValueError:
        try:
            return (
                candidate.resolve(strict=False)
                .relative_to(project_root().resolve(strict=False))
                .as_posix()
            )
        except ValueError:
            return candidate.as_posix()


# Re-export the matched comparison builder so callers (and tests) can reuse the
# same supervised-vs-SSL pairing logic as the one-epoch full grid.
build_proper_training_ssl_comparison = build_ssl_comparison_rows

