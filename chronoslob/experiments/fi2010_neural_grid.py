"""Full FI-2010 supervised-vs-SSL neural evidence grid.

The grid runner is deliberately an orchestration layer. It keeps horizon
selection config-bound by materialising per-horizon neural benchmark configs,
then delegates model fitting to the existing supervised neural runner and the
existing SSL pretraining/fine-tuning runner. The artefacts written here are a
canonical aggregate layer over those lower-level run outputs.
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
from typing import Any, Literal, cast

import pandas as pd
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator

from chronoslob import __version__
from chronoslob.analysis.fi2010_label_mapping import (
    FI2010_CANONICAL_CLASS_ORDER,
    probability_columns_for_order,
)
from chronoslob.experiments.manifests import sha256_file, stable_json_dumps
from chronoslob.training.experiment import get_git_commit
from chronoslob.utils.paths import project_root

__all__ = [
    "FI2010_NEURAL_FULL_GRID_VERSION",
    "GRID_OBJECTIVE_CHOICES",
    "FI2010NeuralFullGridSummary",
    "FI2010NeuralGridRunSpec",
    "build_horizon_config_payload",
    "build_ssl_comparison_rows",
    "expand_fi2010_neural_grid",
    "run_fi2010_neural_full_grid",
    "write_horizon_config",
    "write_neural_grid_aggregate_artifacts",
]

FI2010_NEURAL_FULL_GRID_VERSION = "fi2010-neural-full-grid/v1"

GRID_FOLDS: tuple[int, ...] = (1, 2, 3, 4, 5)
GRID_HORIZONS: tuple[int, ...] = (10, 20, 50)
GRID_DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
GRID_DEFAULT_LOOKBACKS: tuple[int, ...] = (20,)
GRID_OBJECTIVE_CHOICES: tuple[str, ...] = (
    "supervised",
    "masked_reconstruction",
    "next_field",
)

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

_AGGREGATE_COLUMNS: tuple[str, ...] = (
    "horizon",
    "lookback",
    "model_family",
    "pretraining_objective",
    "completed_run_count",
    "failed_run_count",
    "mean_accuracy",
    "std_accuracy",
    "mean_macro_f1",
    "std_macro_f1",
    "mean_mcc",
    "std_mcc",
    "mean_ece",
    "std_ece",
    "mean_brier_score",
    "mean_nll",
)

_FAILURE_COLUMNS: tuple[str, ...] = (
    "fold",
    "horizon",
    "seed",
    "objective",
    "reason",
    "traceback",
    "invalidates_aggregate_claims",
    "status",
    "run_id",
)

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


class FI2010NeuralFullGridSummary(BaseModel):
    """Top-level summary returned by the full neural grid runner."""

    model_config = _MODEL_CONFIG

    output_dir: str
    config_path: str
    processed_root: str
    folds: list[int]
    horizons: list[int]
    seeds: list[int]
    lookbacks: list[int]
    objectives: list[str]
    pretrain_epochs: int
    max_epochs: int
    batch_size: int
    device: str
    smoke_test: bool
    execution_mode: str
    run_count: int
    completed_run_count: int
    failed_run_count: int
    skipped_existing_count: int
    missing_pair_count: int
    core_grid_complete: bool
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
class FI2010NeuralGridRunSpec:
    """One fold/horizon/seed/lookback/objective run in the neural evidence grid."""

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
        if self.objective == "supervised":
            return "none"
        return self.objective

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

    def to_csv_row(self, out_dir: Path) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "fold": self.fold,
            "fold_id": self.fold_id,
            "horizon": self.horizon,
            "seed": self.seed,
            "lookback": self.lookback,
            "model_family": self.model_family,
            "objective": self.objective,
            "pretraining_objective": self.pretraining_objective,
            "run_dir": _relative_path(self.run_dir(out_dir), out_dir),
        }


def expand_fi2010_neural_grid(
    *,
    folds: Sequence[int | str] | str | None = None,
    horizons: Sequence[int] | str | None = None,
    seeds: Sequence[int] | str | None = None,
    lookbacks: Sequence[int] | str | None = None,
    objectives: Sequence[str] | str | None = None,
    smoke_test: bool = False,
) -> tuple[FI2010NeuralGridRunSpec, ...]:
    """Expand the deterministic FI-2010 neural evidence grid."""
    selected_folds = _normalise_folds(folds, default=GRID_FOLDS)
    selected_horizons = _normalise_int_selection(
        horizons,
        default=GRID_HORIZONS,
        field_name="horizons",
        positive=True,
    )
    selected_seeds = _normalise_int_selection(
        seeds,
        default=GRID_DEFAULT_SEEDS,
        field_name="seeds",
        positive=False,
    )
    selected_lookbacks = _normalise_int_selection(
        lookbacks,
        default=GRID_DEFAULT_LOOKBACKS,
        field_name="lookbacks",
        positive=True,
    )
    selected_objectives = _normalise_objectives(objectives)

    if smoke_test:
        selected_folds = selected_folds[:1]
        selected_horizons = selected_horizons[:1]
        selected_seeds = selected_seeds[:1]
        selected_lookbacks = selected_lookbacks[:1]

    specs: list[FI2010NeuralGridRunSpec] = []
    for fold in selected_folds:
        for horizon in selected_horizons:
            for seed in selected_seeds:
                for lookback in selected_lookbacks:
                    for objective in selected_objectives:
                        specs.append(
                            FI2010NeuralGridRunSpec(
                                fold=fold,
                                horizon=horizon,
                                seed=seed,
                                lookback=lookback,
                                objective=objective,
                            )
                        )
    return tuple(specs)


def build_horizon_config_payload(
    base_config_path: str | Path,
    *,
    horizon: int,
    folds: Sequence[int | str],
    seeds: Sequence[int],
    lookbacks: Sequence[int],
    max_epochs: int,
    batch_size: int,
    device: str,
    smoke_test: bool,
) -> dict[str, Any]:
    """Return a config payload whose target horizon selects ``label_<horizon>``."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    path = Path(base_config_path)
    payload_raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload_raw, Mapping):
        raise ValueError(f"neural grid config must be a YAML mapping: {path}")
    payload = dict(payload_raw)
    neural_models = payload.get("neural_models")
    if not isinstance(neural_models, Mapping) or "matrix_transformer" not in neural_models:
        raise ValueError("full neural grid requires a matrix_transformer config")

    payload["folds"] = [_fold_id(fold) for fold in folds]
    payload["seeds"] = [int(seed) for seed in seeds]
    payload["lookbacks"] = [int(lookback) for lookback in lookbacks]
    payload["target"] = {
        "horizon": int(horizon),
        "label_column": f"label_{int(horizon)}",
    }
    training = dict(cast(Mapping[str, Any], payload.get("training", {})))
    training["max_epochs"] = int(max_epochs)
    training["batch_size"] = int(batch_size)
    payload["training"] = training
    payload["device_selection"] = device
    if smoke_test:
        payload["mode"] = "smoke"
        payload["benchmark_note"] = (
            "Tiny full-grid smoke execution; these artefacts exercise code paths "
            "and are not empirical evidence."
        )
    return payload


def write_horizon_config(
    base_config_path: str | Path,
    output_path: str | Path,
    *,
    horizon: int,
    folds: Sequence[int | str],
    seeds: Sequence[int],
    lookbacks: Sequence[int],
    max_epochs: int,
    batch_size: int,
    device: str,
    smoke_test: bool,
) -> Path:
    """Write and return a per-horizon neural benchmark config snapshot."""
    payload = build_horizon_config_payload(
        base_config_path,
        horizon=horizon,
        folds=folds,
        seeds=seeds,
        lookbacks=lookbacks,
        max_epochs=max_epochs,
        batch_size=batch_size,
        device=device,
        smoke_test=smoke_test,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def run_fi2010_neural_full_grid(
    config_path: str | Path = Path("configs/experiments/fi2010_neural_serious.yaml"),
    *,
    processed_root: str | Path,
    out_dir: str | Path = Path("experiments/fi2010_neural_full_grid"),
    folds: Sequence[int | str] | str | None = None,
    horizons: Sequence[int] | str | None = None,
    seeds: Sequence[int] | str | None = None,
    lookbacks: Sequence[int] | str | None = None,
    objectives: Sequence[str] | str | None = None,
    pretrain_epochs: int = 1,
    max_epochs: int = 1,
    batch_size: int = 16,
    device: str = "cpu",
    reuse_completed: bool = True,
    smoke_test: bool = False,
) -> FI2010NeuralFullGridSummary:
    """Run and aggregate the full FI-2010 supervised-vs-SSL neural grid."""
    resolved_config = Path(config_path)
    if not resolved_config.is_file():
        raise FileNotFoundError(f"neural grid config not found: {resolved_config}")
    resolved_out = Path(out_dir)
    resolved_out.mkdir(parents=True, exist_ok=True)
    resolved_processed = Path(processed_root)

    if pretrain_epochs <= 0:
        raise ValueError("pretrain_epochs must be positive")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    selected_specs = expand_fi2010_neural_grid(
        folds=folds,
        horizons=horizons,
        seeds=seeds,
        lookbacks=lookbacks,
        objectives=objectives,
        smoke_test=smoke_test,
    )
    if not selected_specs:
        raise ValueError("neural grid selection produced no runs")

    if smoke_test:
        device = "cpu"
        pretrain_epochs = min(int(pretrain_epochs), 1)
        max_epochs = min(int(max_epochs), 1)

    _write_csv(
        [spec.to_csv_row(resolved_out) for spec in selected_specs],
        resolved_out / "run_plan.csv",
    )

    selected_folds = tuple(dict.fromkeys(spec.fold for spec in selected_specs))
    selected_horizons = tuple(dict.fromkeys(spec.horizon for spec in selected_specs))
    selected_seeds = tuple(dict.fromkeys(spec.seed for spec in selected_specs))
    selected_lookbacks = tuple(dict.fromkeys(spec.lookback for spec in selected_specs))
    for horizon in selected_horizons:
        write_horizon_config(
            resolved_config,
            resolved_out / "configs" / f"horizon_{horizon}" / "config.yaml",
            horizon=horizon,
            folds=selected_folds,
            seeds=selected_seeds,
            lookbacks=selected_lookbacks,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            smoke_test=smoke_test,
        )

    git_commit = get_git_commit()
    result_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for spec in selected_specs:
        run_dir = spec.run_dir(resolved_out)
        if reuse_completed and _completed_run_present(run_dir):
            row = _load_existing_result_row(run_dir, out_dir=resolved_out)
            row["status"] = "skipped_existing"
            result_rows.append(row)
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

        if run_dir.exists():
            _remove_run_dir(run_dir, root=resolved_out)
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            row = _execute_run_spec(
                spec,
                base_config_path=resolved_config,
                processed_root=resolved_processed,
                out_dir=resolved_out,
                run_dir=run_dir,
                pretrain_epochs=pretrain_epochs,
                max_epochs=max_epochs,
                batch_size=batch_size,
                device=device,
                smoke_test=smoke_test,
                git_commit=git_commit,
            )
            result_rows.append(row)
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
    )
    _write_grid_readme(
        resolved_out,
        folds=selected_folds,
        horizons=selected_horizons,
        seeds=selected_seeds,
        lookbacks=selected_lookbacks,
        objectives=tuple(dict.fromkeys(spec.objective for spec in selected_specs)),
        smoke_test=smoke_test,
    )

    completed = sum(1 for row in result_rows if row.get("status") in _SUCCESS_STATUSES)
    skipped = sum(1 for row in result_rows if row.get("status") == "skipped_existing")
    failed = sum(1 for row in failure_rows if row.get("status") == "failed")
    selected_objectives = tuple(dict.fromkeys(spec.objective for spec in selected_specs))
    core_complete = (
        not smoke_test
        and set(selected_folds) == set(GRID_FOLDS)
        and set(selected_horizons) == set(GRID_HORIZONS)
        and set(selected_objectives) == set(GRID_OBJECTIVE_CHOICES)
        and completed == len(selected_specs)
        and failed == 0
    )
    artefacts = {
        "summary": "summary.json",
        "run_plan": "run_plan.csv",
        "results_summary": "results_summary.csv",
        "aggregate_summary": "aggregate_summary.csv",
        "aggregate_summary_json": "aggregate_summary.json",
        "failures": "failures.csv",
        "ssl_comparison": "ssl_comparison.csv",
        "missing_pairs": "missing_pairs.csv",
        "runs": "runs/",
        "readme": "README.md",
    }
    summary = FI2010NeuralFullGridSummary(
        output_dir=str(resolved_out),
        config_path=str(resolved_config),
        processed_root=str(resolved_processed),
        folds=list(selected_folds),
        horizons=list(selected_horizons),
        seeds=list(selected_seeds),
        lookbacks=list(selected_lookbacks),
        objectives=list(selected_objectives),
        pretrain_epochs=pretrain_epochs,
        max_epochs=max_epochs,
        batch_size=batch_size,
        device=device,
        smoke_test=smoke_test,
        execution_mode="smoke" if smoke_test else "benchmark",
        run_count=len(selected_specs),
        completed_run_count=completed,
        failed_run_count=failed,
        skipped_existing_count=skipped,
        missing_pair_count=len(missing_pairs),
        core_grid_complete=core_complete,
        artefacts=artefacts,
        runner_version=FI2010_NEURAL_FULL_GRID_VERSION,
        created_at=datetime.now(UTC),
        git_commit=git_commit,
        warnings=[],
    )
    summary_payload = summary.model_dump(mode="json")
    summary_payload["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_version": __version__,
    }
    summary_payload["aggregate_rows"] = len(aggregate_rows)
    summary_payload["comparison_rows"] = len(comparison_rows)
    summary_payload["evidence_level"] = (
        "smoke_test_only"
        if smoke_test
        else "full_grid_complete"
        if core_complete
        else "partial_real_grid"
    )
    (resolved_out / "summary.json").write_text(
        stable_json_dumps(summary_payload),
        encoding="utf-8",
    )
    return summary


def write_neural_grid_aggregate_artifacts(
    out_dir: str | Path,
    *,
    result_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Write canonical result, aggregate, failure and comparison tables."""
    root = Path(out_dir)
    result_payloads = [dict(row) for row in result_rows]
    failure_payloads = [dict(row) for row in failure_rows]
    _write_csv(result_payloads, root / "results_summary.csv", _RESULT_SUMMARY_COLUMNS)
    _write_csv(failure_payloads, root / "failures.csv", _FAILURE_COLUMNS)

    aggregate_rows = _build_aggregate_rows(result_payloads, failure_payloads)
    _write_csv(aggregate_rows, root / "aggregate_summary.csv", _AGGREGATE_COLUMNS)

    comparison_rows, missing_rows = build_ssl_comparison_rows(result_payloads)
    _write_csv(comparison_rows, root / "ssl_comparison.csv", _COMPARISON_COLUMNS)
    _write_csv(missing_rows, root / "missing_pairs.csv", _COMPARISON_COLUMNS)

    aggregate_json = {
        "runner_version": FI2010_NEURAL_FULL_GRID_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "paths": {
            "results_summary": "results_summary.csv",
            "aggregate_summary": "aggregate_summary.csv",
            "failures": "failures.csv",
            "ssl_comparison": "ssl_comparison.csv",
            "missing_pairs": "missing_pairs.csv",
        },
        "completed_run_count": len(
            [row for row in result_payloads if row.get("status") in _SUCCESS_STATUSES]
        ),
        "failed_run_count": len(
            [row for row in failure_payloads if row.get("status") == "failed"]
        ),
        "skipped_existing_count": len(
            [row for row in failure_payloads if row.get("status") == "skipped_existing"]
        ),
        "missing_pair_count": len(missing_rows),
        "aggregate": aggregate_rows,
    }
    (root / "aggregate_summary.json").write_text(
        stable_json_dumps(aggregate_json),
        encoding="utf-8",
    )
    return aggregate_rows, comparison_rows, missing_rows


def build_ssl_comparison_rows(
    result_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build matched supervised-vs-SSL comparisons, recording missing pairs."""
    successes = [
        dict(row)
        for row in result_rows
        if row.get("status") in _SUCCESS_STATUSES
    ]
    by_key: dict[tuple[int, int, int, int, str, str], dict[str, Any]] = {}
    grouped_keys: set[tuple[int, int, int, int, str]] = set()
    for row in successes:
        objective = _row_objective(row)
        key = (
            _row_int_required(row, "fold"),
            _row_int_required(row, "horizon"),
            _row_int_required(row, "seed"),
            _row_int_required(row, "lookback"),
            str(row.get("model_family", "")),
        )
        grouped_keys.add(key)
        by_key[(*key, objective)] = row

    comparison_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for key in sorted(grouped_keys):
        supervised = by_key.get((*key, "supervised"))
        for ssl_objective in ("masked_reconstruction", "next_field"):
            ssl = by_key.get((*key, ssl_objective))
            row = _comparison_base_row(
                key=key,
                ssl_objective=ssl_objective,
                supervised=supervised,
                ssl=ssl,
            )
            if supervised is None or ssl is None:
                missing = "supervised" if supervised is None else ssl_objective
                row.update(
                    {
                        "status": "missing_pair",
                        "reason": f"missing matched {missing} run",
                    }
                )
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
                        "reason": (
                            "architecture or preprocessing hash differs; "
                            "deltas refused"
                        ),
                    }
                )
                comparison_rows.append(row)
                missing_rows.append(dict(row))
                continue
            _fill_comparison_deltas(row, supervised=supervised, ssl=ssl)
            row.update({"status": "matched", "reason": ""})
            comparison_rows.append(row)
    return comparison_rows, missing_rows


def _execute_run_spec(
    spec: FI2010NeuralGridRunSpec,
    *,
    base_config_path: Path,
    processed_root: Path,
    out_dir: Path,
    run_dir: Path,
    pretrain_epochs: int,
    max_epochs: int,
    batch_size: int,
    device: str,
    smoke_test: bool,
    git_commit: str | None,
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    config_snapshot = write_horizon_config(
        base_config_path,
        run_dir / "config.yaml",
        horizon=spec.horizon,
        folds=[spec.fold],
        seeds=[spec.seed],
        lookbacks=[spec.lookback],
        max_epochs=max_epochs,
        batch_size=batch_size,
        device=device,
        smoke_test=smoke_test,
    )
    (run_dir / "git_commit.txt").write_text(
        "" if git_commit is None else git_commit,
        encoding="utf-8",
    )
    raw_dir = run_dir / "runner"
    if spec.objective == "supervised":
        runner_summary = _call_supervised_runner(
            config_path=config_snapshot,
            processed_root=processed_root,
            out_dir=raw_dir,
            fold_id=spec.fold_id,
            seed=spec.seed,
            lookback=spec.lookback,
            max_epochs=max_epochs,
        )
        raw_result = _extract_supervised_raw_result(raw_dir, spec=spec)
    else:
        runner_summary = _call_ssl_runner(
            config_path=config_snapshot,
            processed_root=processed_root,
            out_dir=raw_dir,
            fold_id=spec.fold_id,
            seed=spec.seed,
            lookback=spec.lookback,
            ssl_objective=_ssl_runner_objective(spec.objective),
            pretrain_epochs=pretrain_epochs,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
        )
        raw_result = _extract_ssl_raw_result(raw_dir, spec=spec)

    predictions = _normalise_prediction_file(
        raw_result["prediction_path"],
        run_dir / "predictions.csv",
        spec=spec,
    )
    metric_payload = _build_metric_payload(
        spec=spec,
        raw_metrics=raw_result["metrics"],
        predictions=predictions,
        checkpoint_hash=raw_result.get("checkpoint_hash"),
        checkpoint_path=raw_result.get("checkpoint_path"),
        prediction_path=run_dir / "predictions.csv",
        out_dir=out_dir,
        git_commit=git_commit,
        config_path=config_snapshot,
    )
    (run_dir / "metrics.json").write_text(
        stable_json_dumps(metric_payload),
        encoding="utf-8",
    )
    _write_status(run_dir, "completed")
    _write_run_log(
        run_dir,
        {
            "run_id": spec.run_id,
            "status": "completed",
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now(UTC).isoformat(),
            "lower_runner_summary": _summary_to_json(runner_summary),
            "lower_runner_dir": _relative_path(raw_dir, out_dir),
        },
    )
    _write_run_manifest(run_dir)
    return _result_row_from_metrics(metric_payload, status="completed")


def _call_supervised_runner(
    *,
    config_path: Path,
    processed_root: Path,
    out_dir: Path,
    fold_id: str,
    seed: int,
    lookback: int,
    max_epochs: int,
) -> Any:
    from chronoslob.experiments.fi2010_neural_runner import (
        run_fi2010_neural_benchmark,
    )

    return run_fi2010_neural_benchmark(
        config_path,
        processed_root=processed_root,
        out_dir=out_dir,
        folds=[fold_id],
        models=["matrix_transformer"],
        seeds=[seed],
        lookbacks=[lookback],
        max_epochs=max_epochs,
        overwrite=True,
        fail_fast=False,
        write_full_predictions=True,
        write_checkpoints=True,
        allow_full_benchmark=True,
    )


def _call_ssl_runner(
    *,
    config_path: Path,
    processed_root: Path,
    out_dir: Path,
    fold_id: str,
    seed: int,
    lookback: int,
    ssl_objective: str,
    pretrain_epochs: int,
    max_epochs: int,
    batch_size: int,
    device: str,
) -> Any:
    from chronoslob.experiments.fi2010_ssl_runner import (
        run_fi2010_ssl_neural_benchmark,
    )

    return run_fi2010_ssl_neural_benchmark(
        config_path,
        processed_root=processed_root,
        out_dir=out_dir,
        folds=[fold_id],
        seeds=[seed],
        lookbacks=[lookback],
        objective=ssl_objective,
        pretrain_epochs=pretrain_epochs,
        max_epochs=max_epochs,
        batch_size=batch_size,
        device=device,
        overwrite=True,
        fail_fast=False,
        write_full_predictions=True,
    )


def _extract_supervised_raw_result(
    raw_dir: Path,
    *,
    spec: FI2010NeuralGridRunSpec,
) -> dict[str, Any]:
    rows = _read_csv_dicts(raw_dir / "results_by_fold_seed.csv")
    match = _single_row(
        rows,
        lambda row: row.get("model_name") == "matrix_transformer"
        and row.get("status") == "ok",
        "completed supervised matrix_transformer result",
    )
    prediction_path = (
        raw_dir
        / "runs"
        / f"{spec.fold_id}_seed_{spec.seed}_matrix_transformer_lb{spec.lookback}"
        / "predictions.csv"
    )
    checkpoint_path = (
        raw_dir
        / "checkpoints"
        / f"{spec.fold_id}_seed_{spec.seed}_matrix_transformer_lb{spec.lookback}"
        / "best_model.pt"
    )
    if not prediction_path.is_file():
        raise FileNotFoundError(f"supervised prediction file missing: {prediction_path}")
    checkpoint_hash = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    return {
        "metrics": match,
        "prediction_path": prediction_path,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path.is_file() else None,
        "checkpoint_hash": checkpoint_hash,
    }


def _extract_ssl_raw_result(
    raw_dir: Path,
    *,
    spec: FI2010NeuralGridRunSpec,
) -> dict[str, Any]:
    rows = _read_csv_dicts(raw_dir / "results_by_fold_seed.csv")
    match = _single_row(
        rows,
        lambda row: row.get("model_name") == "ssl_transformer"
        and row.get("status") == "ok",
        "completed ssl_transformer result",
    )
    prediction_path = (
        raw_dir
        / "runs"
        / f"{spec.fold_id}_seed_{spec.seed}_lb{spec.lookback}"
        / "ssl_transformer"
        / "predictions.csv"
    )
    checkpoint_path = (
        raw_dir
        / "runs"
        / f"{spec.fold_id}_seed_{spec.seed}_lb{spec.lookback}"
        / "pretrain"
        / "pretrained_encoder.pt"
    )
    if not prediction_path.is_file():
        raise FileNotFoundError(f"SSL prediction file missing: {prediction_path}")
    checkpoint_hash = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    return {
        "metrics": match,
        "prediction_path": prediction_path,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path.is_file() else None,
        "checkpoint_hash": checkpoint_hash,
    }


def _normalise_prediction_file(
    source_path: Path,
    output_path: Path,
    *,
    spec: FI2010NeuralGridRunSpec,
) -> list[dict[str, Any]]:
    frame = pd.read_csv(source_path)
    rows: list[dict[str, Any]] = []
    for position, raw in frame.iterrows():
        row_map = {str(key): raw[key] for key in frame.columns}
        row_id = _first_present(row_map, ("row_id", "sample_id", "row_index"))
        prob_down = _probability_value(row_map, "down")
        prob_stationary = _probability_value(row_map, "stationary")
        prob_up = _probability_value(row_map, "up")
        confidence = _safe_float(row_map.get("confidence"))
        if confidence is None:
            confidence = _max_probability(prob_down, prob_stationary, prob_up)
        rows.append(
            {
                "row_id": _json_scalar(row_id),
                "sample_id": int(position),
                "fold": int(spec.fold),
                "horizon": int(spec.horizon),
                "seed": int(spec.seed),
                "lookback": int(spec.lookback),
                "model_family": spec.model_family,
                "pretraining_objective": spec.pretraining_objective,
                "split": str(row_map.get("split", "test")),
                "y_true": _json_scalar(_first_present(row_map, ("y_true", "label"))),
                "y_pred": _json_scalar(
                    _first_present(row_map, ("y_pred", "prediction"))
                ),
                "prob_down": prob_down,
                "prob_stationary": prob_stationary,
                "prob_up": prob_up,
                "confidence": confidence,
                "raw_model_name": row_map.get("model_name"),
            }
        )
    if not rows:
        raise ValueError(f"prediction file contains no rows: {source_path}")
    _write_csv(rows, output_path, (*_PREDICTION_COLUMNS, "raw_model_name"))
    return rows


def _build_metric_payload(
    *,
    spec: FI2010NeuralGridRunSpec,
    raw_metrics: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    checkpoint_hash: Any,
    checkpoint_path: Any,
    prediction_path: Path,
    out_dir: Path,
    git_commit: str | None,
    config_path: Path,
) -> dict[str, Any]:
    derived = _derive_prediction_metrics(predictions)
    metrics = {
        "accuracy": _safe_float(raw_metrics.get("accuracy")) or derived["accuracy"],
        "macro_f1": _safe_float(raw_metrics.get("macro_f1")) or derived["macro_f1"],
        "mcc": _safe_float(raw_metrics.get("mcc")) or _safe_float(
            raw_metrics.get("matthews_corrcoef")
        ),
        "ece": _safe_float(raw_metrics.get("ece"))
        or _safe_float(raw_metrics.get("expected_calibration_error"))
        or derived["ece"],
        "brier_score": _safe_float(raw_metrics.get("brier_score"))
        or derived["brier_score"],
        "nll": _safe_float(raw_metrics.get("nll"))
        or _safe_float(raw_metrics.get("log_loss"))
        or derived["nll"],
        "class_f1_down": derived["class_f1_down"],
        "class_f1_stationary": derived["class_f1_stationary"],
        "class_f1_up": derived["class_f1_up"],
    }
    architecture_hash = _hash_payload(
        {
            "model_family": spec.model_family,
            "config_sha256": sha256_file(config_path),
        }
    )
    preprocessing_hash = _hash_payload(
        {
            "split": "official_column",
            "feature_scaler": "train_only_standardised_matrix",
            "windowing": "split_contained_contiguous",
            "horizon": spec.horizon,
            "label_column": f"label_{spec.horizon}",
        }
    )
    return {
        "runner_version": FI2010_NEURAL_FULL_GRID_VERSION,
        "run_id": spec.run_id,
        "status": "completed",
        "fold": int(spec.fold),
        "fold_id": spec.fold_id,
        "horizon": int(spec.horizon),
        "seed": int(spec.seed),
        "lookback": int(spec.lookback),
        "model_family": spec.model_family,
        "objective": spec.objective,
        "pretraining_objective": spec.pretraining_objective,
        "metrics": metrics,
        "checkpoint_hash": checkpoint_hash,
        "checkpoint_path": checkpoint_path,
        "prediction_file": _relative_path(prediction_path, out_dir),
        "git_commit": git_commit,
        "architecture_hash": architecture_hash,
        "preprocessing_hash": preprocessing_hash,
        "class_label_mapping": {
            "prob_up": "FI-2010 label 1",
            "prob_stationary": "FI-2010 label 2",
            "prob_down": "FI-2010 label 3",
        },
    }


def _derive_prediction_metrics(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    true = [_direction_label(row.get("y_true")) for row in predictions]
    pred = [_direction_label(row.get("y_pred")) for row in predictions]
    if any(item is None for item in true) or any(item is None for item in pred):
        return {
            "accuracy": None,
            "macro_f1": None,
            "class_f1_down": None,
            "class_f1_stationary": None,
            "class_f1_up": None,
            "brier_score": None,
            "nll": None,
            "ece": None,
        }
    true_labels = cast(list[str], true)
    pred_labels = cast(list[str], pred)
    class_labels = list(FI2010_CANONICAL_CLASS_ORDER)
    f1_by_class = {
        label: _binary_f1(true_labels, pred_labels, positive_label=label)
        for label in class_labels
    }
    accuracy = sum(
        1
        for actual, predicted in zip(true_labels, pred_labels, strict=True)
        if actual == predicted
    ) / len(true_labels)
    probabilities = _probability_matrix(predictions)
    brier = nll = ece = None
    if probabilities is not None:
        brier = _brier_score(true_labels, probabilities, labels=class_labels)
        nll = _negative_log_likelihood(true_labels, probabilities, labels=class_labels)
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


def _build_aggregate_rows(
    result_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    frame = pd.DataFrame([dict(row) for row in result_rows])
    failure_frame = pd.DataFrame([dict(row) for row in failure_rows])
    if frame.empty:
        return []
    ok = frame[frame["status"].isin(_SUCCESS_STATUSES)].copy()
    if ok.empty:
        return []
    rows: list[dict[str, Any]] = []
    group_columns = ["horizon", "lookback", "model_family", "pretraining_objective"]
    for keys, group in ok.groupby(group_columns, sort=True):
        horizon, lookback, model_family, objective = keys
        failed_count = _failed_count_for_group(
            failure_frame,
            horizon=int(horizon),
            lookback=int(lookback),
            objective=str(objective),
        )
        row: dict[str, Any] = {
            "horizon": int(horizon),
            "lookback": int(lookback),
            "model_family": str(model_family),
            "pretraining_objective": str(objective),
            "completed_run_count": len(group),
            "failed_run_count": failed_count,
        }
        for metric in ("accuracy", "macro_f1", "mcc", "ece"):
            row[f"mean_{metric}"] = _mean(group[metric])
            row[f"std_{metric}"] = _std(group[metric])
        row["mean_brier_score"] = _mean(group["brier_score"])
        row["mean_nll"] = _mean(group["nll"])
        rows.append(row)
    return rows


def _failed_count_for_group(
    failure_frame: pd.DataFrame,
    *,
    horizon: int,
    lookback: int,
    objective: str,
) -> int:
    if failure_frame.empty:
        return 0
    objective_token = "supervised" if objective == "none" else objective
    subset = failure_frame[
        (pd.to_numeric(failure_frame["horizon"], errors="coerce") == horizon)
        & (pd.to_numeric(failure_frame.get("lookback", 0), errors="coerce") == lookback)
        & (failure_frame["objective"].astype(str) == objective_token)
        & (failure_frame["status"].astype(str) == "failed")
    ]
    return len(subset)


def _comparison_base_row(
    *,
    key: tuple[int, int, int, int, str],
    ssl_objective: str,
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
        "ssl_objective": ssl_objective,
        "comparison": f"supervised_vs_{ssl_objective}",
        "supervised_run_id": "" if supervised is None else supervised.get("run_id", ""),
        "ssl_run_id": "" if ssl is None else ssl.get("run_id", ""),
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
        "macro_f1_outcome": "",
        "ece_outcome": "",
        "mcc_outcome": "",
        "status": "",
        "reason": "",
    }


def _fill_comparison_deltas(
    row: dict[str, Any],
    *,
    supervised: Mapping[str, Any],
    ssl: Mapping[str, Any],
) -> None:
    for metric in ("macro_f1", "accuracy", "mcc", "ece", "brier_score", "nll"):
        base = _safe_float(supervised.get(metric))
        candidate = _safe_float(ssl.get(metric))
        row[f"supervised_{metric}"] = base
        row[f"ssl_{metric}"] = candidate
        row[f"delta_{metric}"] = (
            None if base is None or candidate is None else candidate - base
        )
    row["macro_f1_outcome"] = _higher_is_better_outcome(row["delta_macro_f1"])
    row["mcc_outcome"] = _higher_is_better_outcome(row["delta_mcc"])
    row["ece_outcome"] = _lower_is_better_outcome(row["delta_ece"])


def _result_row_from_metrics(
    payload: Mapping[str, Any],
    *,
    status: str,
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
        "status": status,
        "run_id": payload.get("run_id"),
        "run_dir": _relative_path(Path(payload.get("prediction_file", ".")).parent, Path(".")),
        "architecture_hash": payload.get("architecture_hash"),
        "preprocessing_hash": payload.get("preprocessing_hash"),
    }


def _load_existing_result_row(run_dir: Path, *, out_dir: Path) -> dict[str, Any]:
    payload = _read_json(run_dir / "metrics.json")
    row = _result_row_from_metrics(payload, status="skipped_existing")
    row["run_dir"] = _relative_path(run_dir, out_dir)
    return row


def _write_failure_artefacts(
    spec: FI2010NeuralGridRunSpec,
    *,
    run_dir: Path,
    reason: str,
    traceback_text: str,
    git_commit: str | None,
) -> None:
    (run_dir / "git_commit.txt").write_text(
        "" if git_commit is None else git_commit,
        encoding="utf-8",
    )
    payload = {
        "runner_version": FI2010_NEURAL_FULL_GRID_VERSION,
        "run_id": spec.run_id,
        "status": "failed",
        "fold": spec.fold,
        "fold_id": spec.fold_id,
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
    spec: FI2010NeuralGridRunSpec,
    *,
    status: Literal["failed", "skipped_existing"],
    reason: str,
    traceback_text: str,
    invalidates: bool,
) -> dict[str, Any]:
    return {
        "fold": int(spec.fold),
        "horizon": int(spec.horizon),
        "seed": int(spec.seed),
        "lookback": int(spec.lookback),
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
            run_dir / "config.yaml",
            run_dir / "metrics.json",
            run_dir / "predictions.csv",
            run_dir / "git_commit.txt",
            run_dir / "run_log.json",
            run_dir / "status.txt",
        )
        if path.is_file()
    ]
    payload = {
        "runner_version": FI2010_NEURAL_FULL_GRID_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "sha256": {path.name: sha256_file(path) for path in files},
        "artefacts": {path.name: path.name for path in files},
    }
    (run_dir / "sha256_manifest.json").write_text(
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


def _completed_run_present(run_dir: Path) -> bool:
    status_path = run_dir / "status.txt"
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.csv"
    if not status_path.is_file() or not metrics_path.is_file() or not predictions_path.is_file():
        return False
    status = status_path.read_text(encoding="utf-8").strip()
    return status in _SUCCESS_STATUSES


def _remove_run_dir(run_dir: Path, *, root: Path) -> None:
    resolved_root = root.resolve(strict=False)
    resolved_run = run_dir.resolve(strict=False)
    try:
        resolved_run.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"refusing to remove run outside output root: {run_dir}") from exc
    shutil.rmtree(resolved_run)


def _write_grid_readme(
    out_dir: Path,
    *,
    folds: Sequence[int],
    horizons: Sequence[int],
    seeds: Sequence[int],
    lookbacks: Sequence[int],
    objectives: Sequence[str],
    smoke_test: bool,
) -> None:
    lines = [
        "# FI-2010 Neural Full Grid Artefacts",
        "",
        "This directory is generated by `run-fi2010-neural-full-grid`.",
        "It stores a matched, one-epoch supervised-vs-SSL comparison grid; it is "
        "separate from the reduced-scope 25-epoch neural benchmark.",
        "",
        "## What Was Run",
        "",
        "- supervised `matrix_transformer` runs with no pretraining",
        "- SSL `matrix_transformer` runs with masked-reconstruction pretraining",
        "- SSL `matrix_transformer` runs with next-field pretraining",
        "- all model fitting is delegated to the existing leakage-safe supervised and SSL runners",
        "",
        "## Grid",
        "",
        f"- folds: {', '.join(str(item) for item in folds)}",
        f"- horizons: {', '.join(str(item) for item in horizons)}",
        f"- seeds: {', '.join(str(item) for item in seeds)}",
        f"- lookbacks: {', '.join(str(item) for item in lookbacks)}",
        f"- objectives: {', '.join(objectives)}",
        f"- smoke test: {'yes' if smoke_test else 'no'}",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m chronoslob.cli run-fi2010-neural-full-grid \\",
        "  --config configs/experiments/fi2010_neural_serious.yaml \\",
        "  --processed-root data/processed/fi2010 \\",
        "  --folds 1,2,3,4,5 \\",
        "  --horizons 10,20,50 \\",
        "  --seeds 0,1,2 \\",
        "  --lookbacks 20 \\",
        "  --objectives supervised,masked_reconstruction,next_field \\",
        "  --pretrain-epochs 1 \\",
        "  --max-epochs 1 \\",
        "  --batch-size 256 \\",
        "  --device cuda \\",
        "  --out experiments/fi2010_neural_full_grid",
        "```",
        "",
        "## Outputs",
        "",
        "- per-run predictions: `runs/**/predictions.csv`",
        "- per-run metrics and hashes: `runs/**/metrics.json`, `runs/**/sha256_manifest.json`",
        "- completed-run table: `results_summary.csv`",
        "- grouped aggregate table: `aggregate_summary.csv` and `aggregate_summary.json`",
        "- failed or reused-existing runs: `failures.csv`",
        "- matched supervised-vs-SSL deltas: `ssl_comparison.csv`",
        "",
        "## Limitations",
        "",
        "- Smoke-test artefacts are code-path checks only and are not empirical evidence.",
        "- Failed runs remain explicit and limit aggregate interpretation.",
        "- One-epoch full-grid scores are controlled comparison diagnostics, not "
        "performance-maximising neural benchmark results.",
        "- SSL pretraining is compared under matched conditions; the artefacts do "
        "not support a broad SSL improvement claim unless aggregate deltas show it.",
        "- The runner reports predictive and calibration metrics only; it makes no "
        "trading or profitability claim.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _normalise_int_selection(
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
        return GRID_OBJECTIVE_CHOICES
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "all":
            return GRID_OBJECTIVE_CHOICES
        tokens = [token.strip() for token in text.split(",")]
    else:
        tokens = [token.strip() for token in value]
    cleaned: list[str] = []
    for token in tokens:
        objective = token.lower()
        if objective not in GRID_OBJECTIVE_CHOICES:
            raise ValueError(
                f"unsupported objective {token!r}; supported: {list(GRID_OBJECTIVE_CHOICES)}"
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


def _fold_id(value: int | str) -> str:
    return f"fold_{_fold_number(value)}"


def _ssl_runner_objective(objective: str) -> str:
    if objective == "masked_reconstruction":
        return "masked_field"
    if objective == "next_field":
        return "next_field"
    raise ValueError(f"objective has no SSL runner mapping: {objective}")


def _row_objective(row: Mapping[str, Any]) -> str:
    objective = str(row.get("pretraining_objective", "")).strip()
    return "supervised" if objective == "none" else objective


def _single_row(
    rows: Sequence[Mapping[str, str]],
    predicate: Any,
    description: str,
) -> dict[str, str]:
    matches = [dict(row) for row in rows if predicate(row)]
    if not matches:
        raise RuntimeError(f"no {description} found")
    if len(matches) > 1:
        raise RuntimeError(f"multiple {description} rows found")
    return matches[0]


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str).fillna("")
    return [
        {str(column): str(row[column]) for column in frame.columns}
        for _, row in frame.iterrows()
    ]


def _read_json(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], payload)


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


def _summary_to_json(summary: Any) -> Any:
    model_dump = getattr(summary, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return str(summary)


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


def _hash_payload(payload: Mapping[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(stable_json_dumps(dict(payload)).encode("utf-8")).hexdigest()


def _first_present(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and not pd.isna(row[key]):
            return row[key]
    return None


def _json_scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    item = value.item() if hasattr(value, "item") else value
    if isinstance(item, float) and item.is_integer():
        return int(item)
    return item


def _probability_value(row: Mapping[str, Any], direction: str) -> float | None:
    aliases = {
        "up": ("prob_up", "probability_up", "probability_1", "probability_1.0"),
        "stationary": (
            "prob_stationary",
            "probability_stationary",
            "probability_2",
            "probability_2.0",
        ),
        "down": ("prob_down", "probability_down", "probability_3", "probability_3.0"),
    }[direction]
    for key in aliases:
        value = _safe_float(row.get(key))
        if value is not None:
            return value
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
        "unchanged": "stationary",
        "flat": "stationary",
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
    try:
        probability_columns = probability_columns_for_order(
            ("prob_up", "prob_stationary", "prob_down"),
            class_order=FI2010_CANONICAL_CLASS_ORDER,
        )
    except ValueError:
        return None
    for row in predictions:
        values = [_safe_float(row.get(column)) for column in probability_columns]
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
    *,
    labels: Sequence[str],
) -> float:
    label_to_index = {label: index for index, label in enumerate(labels)}
    total = 0.0
    for actual, row in zip(y_true, probabilities, strict=True):
        index = label_to_index[actual]
        total += sum(
            (probability - (1.0 if position == index else 0.0)) ** 2
            for position, probability in enumerate(row)
        )
    return float(total / len(y_true))


def _negative_log_likelihood(
    y_true: Sequence[str],
    probabilities: Sequence[Sequence[float]],
    *,
    labels: Sequence[str],
) -> float:
    label_to_index = {label: index for index, label in enumerate(labels)}
    total = 0.0
    for actual, row in zip(y_true, probabilities, strict=True):
        total -= math.log(max(float(row[label_to_index[actual]]), _EPS))
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


def _higher_is_better_outcome(delta: Any) -> str:
    value = _safe_float(delta)
    if value is None:
        return "n/a"
    if abs(value) <= _EPS:
        return "tie"
    return "win" if value > 0 else "loss"


def _lower_is_better_outcome(delta: Any) -> str:
    value = _safe_float(delta)
    if value is None:
        return "n/a"
    if abs(value) <= _EPS:
        return "tie"
    return "win" if value < 0 else "loss"


def _row_int_required(row: Mapping[str, Any], key: str) -> int:
    value = _safe_float(row.get(key))
    if value is None:
        raise ValueError(f"row is missing integer field {key!r}")
    return int(value)


def _relative_path(path: Path, root: Path) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve(strict=False).relative_to(
            Path(root).resolve(strict=False)
        ).as_posix()
    except ValueError:
        try:
            return candidate.resolve(strict=False).relative_to(
                project_root().resolve(strict=False)
            ).as_posix()
        except ValueError:
            return candidate.as_posix()
