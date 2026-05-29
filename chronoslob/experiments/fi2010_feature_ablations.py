"""Leakage-safe FI-2010 microstructure feature ablation runner."""

from __future__ import annotations

import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from chronoslob import __version__
from chronoslob.analysis.fi2010_label_mapping import (
    FI2010_CANONICAL_CLASS_ORDER,
    FI2010_CLASS_TO_RAW_LABEL,
    canonical_class_name,
)
from chronoslob.experiments.manifests import sha256_file, stable_json_dumps
from chronoslob.experiments.model_registry import (
    build_paper_baseline_config,
    get_paper_model_spec,
)
from chronoslob.features.microstructure_fi2010 import (
    DEFAULT_FEATURE_GROUPS,
    build_microstructure_feature_artifacts,
    build_microstructure_features,
    default_label_columns,
)
from chronoslob.features.registry import (
    FeatureRegistryError,
    get_feature_group,
    proxy_group_names,
    unsupported_group_names,
)
from chronoslob.models.baselines import BaseBaselineModel, create_baseline_model
from chronoslob.models.preprocessing import TrainOnlyStandardScaler
from chronoslob.training.experiment import get_git_commit
from chronoslob.training.metrics import compute_classification_metrics

__all__ = [
    "ABLATION_MODES",
    "CLASSICAL_FEATURE_ABLATION_MODELS",
    "FEATURE_ABLATION_VERSION",
    "FeatureAblationSummary",
    "build_aggregate_summary",
    "build_feature_delta_summary",
    "expand_ablation_specs",
    "run_fi2010_feature_ablations",
]

FEATURE_ABLATION_VERSION = "fi2010-microstructure-feature-ablations/v2"

ABLATION_MODES: tuple[str, ...] = (
    "all_features",
    "remove_one_group",
    "only_one_group",
    "raw_lob_only",
    "derived_microstructure_only",
    "no_proxy_features",
)
CLASSICAL_FEATURE_ABLATION_MODELS: tuple[str, ...] = (
    "logistic",
    "ridge",
    "elastic_net",
    "gradient_boosting",
)
RESULTS_SUMMARY_COLUMNS: tuple[str, ...] = (
    "fold",
    "horizon",
    "seed",
    "model",
    "ablation_mode",
    "feature_group",
    "features_used",
    "proxy_features_used",
    "unsupported_groups",
    "accuracy",
    "macro_f1",
    "mcc",
    "ece",
    "brier_score",
    "status",
)
AGGREGATE_SUMMARY_COLUMNS: tuple[str, ...] = (
    "horizon",
    "model",
    "ablation_mode",
    "feature_group",
    "completed_runs",
    "failed_runs",
    "mean_accuracy",
    "std_accuracy",
    "mean_macro_f1",
    "std_macro_f1",
    "mean_mcc",
    "std_mcc",
    "mean_ece",
)
DELTA_SUMMARY_COLUMNS: tuple[str, ...] = (
    "fold",
    "horizon",
    "seed",
    "model",
    "ablation_mode",
    "feature_group",
    "baseline_accuracy",
    "accuracy",
    "delta_accuracy",
    "baseline_macro_f1",
    "macro_f1",
    "delta_macro_f1",
    "baseline_mcc",
    "mcc",
    "delta_mcc",
    "baseline_ece",
    "ece",
    "delta_ece",
    "interpretation",
)
_RAW_GROUPS = ("price_levels", "size_levels", "top_of_book")
_LABELS = (1, 2, 3)
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)


class FeatureAblationSummary(BaseModel):
    """Top-level summary returned by the feature-ablation runner."""

    model_config = _MODEL_CONFIG

    output_dir: str
    run_count: int
    completed_run_count: int
    failed_run_count: int
    skipped_existing_count: int = 0
    folds: list[str]
    horizons: list[int]
    seeds: list[int]
    models: list[str]
    feature_groups: list[str]
    ablation_modes: list[str]
    smoke_test: bool
    artefacts: dict[str, str]
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime
    runner_version: str

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value


@dataclass(frozen=True)
class AblationSpec:
    """One concrete feature subset to evaluate."""

    mode: str
    feature_group: str
    groups: tuple[str, ...]


@dataclass(frozen=True)
class _FoldInput:
    fold: str
    path: Path
    frame: pd.DataFrame
    sha256: str | None


@dataclass(frozen=True)
class _SplitIndices:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


@dataclass(frozen=True)
class _PreparedModel:
    model: BaseBaselineModel
    scaler: TrainOnlyStandardScaler | None


def expand_ablation_specs(
    group_columns: Mapping[str, Sequence[str]],
    *,
    feature_groups: Sequence[str] | None = None,
    ablation_modes: Sequence[str] | None = None,
) -> tuple[AblationSpec, ...]:
    """Expand requested ablation modes into concrete group subsets."""
    selected_modes = _normalise_modes(ablation_modes)
    selected_groups = (
        tuple(feature_groups)
        if feature_groups is not None
        else tuple(group for group in DEFAULT_FEATURE_GROUPS if group in group_columns)
    )
    available = tuple(group for group in selected_groups if group_columns.get(group))
    specs: list[AblationSpec] = []
    for mode in selected_modes:
        if mode == "all_features":
            specs.append(AblationSpec(mode=mode, feature_group="all", groups=available))
        elif mode == "remove_one_group":
            for group in available:
                remaining = tuple(item for item in available if item != group)
                specs.append(AblationSpec(mode=mode, feature_group=group, groups=remaining))
        elif mode == "only_one_group":
            for group in available:
                specs.append(AblationSpec(mode=mode, feature_group=group, groups=(group,)))
        elif mode == "raw_lob_only":
            groups = tuple(group for group in _RAW_GROUPS if group in available)
            specs.append(AblationSpec(mode=mode, feature_group="raw_lob", groups=groups))
        elif mode == "derived_microstructure_only":
            groups = tuple(group for group in available if group not in _RAW_GROUPS)
            specs.append(
                AblationSpec(
                    mode=mode,
                    feature_group="derived_microstructure",
                    groups=groups,
                )
            )
        elif mode == "no_proxy_features":
            proxies = set(proxy_group_names())
            groups = tuple(group for group in available if group not in proxies)
            specs.append(AblationSpec(mode=mode, feature_group="no_proxy", groups=groups))
    return tuple(specs)


def run_fi2010_feature_ablations(
    *,
    out_dir: str | Path,
    config_path: str | Path | None = None,
    processed_root: str | Path | None = None,
    data_path: str | Path | None = None,
    folds: Sequence[str | int] | str | None = None,
    horizons: Sequence[int] | str | None = None,
    seeds: Sequence[int] | str | None = None,
    models: Sequence[str] | str | None = None,
    feature_groups: Sequence[str] | str | None = None,
    ablation_modes: Sequence[str] | str | None = None,
    reuse_completed: bool = True,
    strict: bool = True,
    smoke_test: bool = False,
) -> FeatureAblationSummary:
    """Run classical FI-2010 feature ablations and write artefacts."""
    output_dir = Path(out_dir)
    _prepare_output_dir(output_dir, reuse_completed=reuse_completed)
    selected_folds = _parse_folds(folds)
    selected_horizons = _parse_ints(horizons, default=(10,), positive=True)
    selected_seeds = _parse_ints(seeds, default=(0,), positive=False)
    selected_models = _normalise_models(models, smoke_test=smoke_test)
    selected_groups = _normalise_groups(feature_groups)
    selected_modes = _normalise_modes(ablation_modes)
    warnings: list[str] = []

    fold_inputs = _resolve_fold_inputs(
        config_path=Path(config_path) if config_path is not None else None,
        processed_root=Path(processed_root) if processed_root is not None else None,
        data_path=Path(data_path) if data_path is not None else None,
        folds=selected_folds,
        smoke_test=smoke_test,
        warnings=warnings,
    )

    all_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    run_entries: list[dict[str, Any]] = []

    for fold_input in fold_inputs:
        label_columns = default_label_columns(fold_input.frame.columns)
        try:
            feature_result = build_microstructure_features(
                fold_input.frame,
                feature_groups=selected_groups,
                label_columns=label_columns,
                partition_columns=_partition_columns(fold_input.frame),
                strict=strict,
            )
        except (FeatureRegistryError, ValueError, TypeError) as exc:
            for horizon in selected_horizons:
                for seed in selected_seeds:
                    for model in selected_models:
                        row = _failed_result_row(
                            fold=fold_input.fold,
                            horizon=horizon,
                            seed=seed,
                            model=model,
                            mode="feature_build",
                            feature_group="all",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        all_rows.append(row)
                        failure_rows.append(row)
            continue
        _write_fold_features(
            output_dir,
            fold_input=fold_input,
            feature_result=feature_result,
            strict=strict,
        )
        specs = expand_ablation_specs(
            feature_result.group_columns,
            feature_groups=selected_groups,
            ablation_modes=selected_modes,
        )
        unsupported = _unsupported_summary(feature_result.group_manifest)
        for horizon in selected_horizons:
            label_column = f"label_{horizon}"
            if label_column not in fold_input.frame.columns:
                for seed in selected_seeds:
                    for model in selected_models:
                        for spec in specs:
                            row = _failed_result_row(
                                fold=fold_input.fold,
                                horizon=horizon,
                                seed=seed,
                                model=model,
                                mode=spec.mode,
                                feature_group=spec.feature_group,
                                error=f"label column missing: {label_column}",
                            )
                            all_rows.append(row)
                            failure_rows.append(row)
                continue
            split = _split_indices(fold_input.frame)
            y_all = _raw_label_series(fold_input.frame[label_column])
            for seed in selected_seeds:
                for model_name in selected_models:
                    for spec in specs:
                        run_result = _run_one_spec(
                            output_dir=output_dir,
                            fold_input=fold_input,
                            feature_frame=feature_result.features,
                            group_columns=feature_result.group_columns,
                            spec=spec,
                            horizon=horizon,
                            label_column=label_column,
                            y_all=y_all,
                            split=split,
                            seed=seed,
                            model_name=model_name,
                            unsupported_groups=unsupported,
                            reuse_completed=reuse_completed,
                        )
                        all_rows.append(run_result["row"])
                        run_entries.append(run_result["entry"])
                        if run_result["row"]["status"] != "completed":
                            failure_rows.append(run_result["row"])

    result_path = output_dir / "results_summary.csv"
    _write_csv(all_rows, result_path, RESULTS_SUMMARY_COLUMNS)
    aggregate_rows = build_aggregate_summary(all_rows)
    aggregate_path = output_dir / "aggregate_summary.csv"
    _write_csv(aggregate_rows, aggregate_path, AGGREGATE_SUMMARY_COLUMNS)
    delta_rows = build_feature_delta_summary(all_rows)
    delta_path = output_dir / "feature_delta_summary.csv"
    _write_csv(delta_rows, delta_path, DELTA_SUMMARY_COLUMNS)
    failures_path = output_dir / "failures.json"
    failures_path.write_text(
        stable_json_dumps({"failure_count": len(failure_rows), "failures": failure_rows}),
        encoding="utf-8",
    )
    manifest_path = _write_root_manifest(
        output_dir=output_dir,
        run_entries=run_entries,
        input_folds=fold_inputs,
    )

    artefacts = {
        "summary": "summary.json",
        "results_summary": "results_summary.csv",
        "aggregate_summary": "aggregate_summary.csv",
        "feature_delta_summary": "feature_delta_summary.csv",
        "failures": "failures.json",
        "manifest": manifest_path.name,
        "runs": "runs/",
    }
    summary = FeatureAblationSummary(
        output_dir=str(output_dir),
        run_count=len(all_rows),
        completed_run_count=sum(1 for row in all_rows if row.get("status") == "completed"),
        failed_run_count=sum(1 for row in all_rows if row.get("status") != "completed"),
        skipped_existing_count=sum(
            1 for entry in run_entries if entry.get("status") == "skipped_existing"
        ),
        folds=[fold.fold for fold in fold_inputs],
        horizons=list(selected_horizons),
        seeds=list(selected_seeds),
        models=list(selected_models),
        feature_groups=list(selected_groups),
        ablation_modes=list(selected_modes),
        smoke_test=smoke_test,
        artefacts=artefacts,
        warnings=warnings,
        created_at=datetime.now(UTC),
        runner_version=FEATURE_ABLATION_VERSION,
    )
    summary_payload = summary.model_dump(mode="json")
    summary_payload["package_version"] = __version__
    summary_payload["git_commit"] = get_git_commit()
    summary_payload["unsupported_groups"] = list(unsupported_group_names())
    summary_payload["proxy_groups"] = list(proxy_group_names())
    summary_payload["claim_boundary"] = (
        "Feature ablations are leakage-safe diagnostics over FI-2010 snapshots. "
        "Snapshot deltas are proxies, not true order-flow imbalance."
    )
    (output_dir / "summary.json").write_text(
        stable_json_dumps(summary_payload),
        encoding="utf-8",
    )
    _write_sha256_manifest(output_dir)
    return summary


def build_aggregate_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate result rows by horizon/model/mode/group."""
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return []
    out: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["horizon", "model", "ablation_mode", "feature_group"],
        dropna=False,
        sort=True,
    ):
        horizon, model, mode, feature_group = keys
        completed = group[group["status"] == "completed"]
        failed = group[group["status"] != "completed"]
        row: dict[str, Any] = {
            "horizon": _optional_int(horizon),
            "model": str(model),
            "ablation_mode": str(mode),
            "feature_group": str(feature_group),
            "completed_runs": len(completed),
            "failed_runs": len(failed),
        }
        for metric in ("accuracy", "macro_f1", "mcc", "ece"):
            values = pd.to_numeric(completed.get(metric), errors="coerce").dropna()
            row[f"mean_{metric}"] = _mean(values)
            if metric != "ece":
                row[f"std_{metric}"] = _std(values)
        out.append(row)
    return out


def build_feature_delta_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compare each ablation against the matched all-features baseline."""
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return []
    completed = frame[frame["status"] == "completed"].copy()
    if completed.empty:
        return []
    baselines = completed[
        (completed["ablation_mode"] == "all_features") & (completed["feature_group"] == "all")
    ].copy()
    baseline_by_key: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for _, row in baselines.iterrows():
        key = (
            str(row.get("fold")),
            str(row.get("horizon")),
            str(row.get("seed")),
            str(row.get("model")),
        )
        baseline_by_key[key] = row.to_dict()
    delta_rows: list[dict[str, Any]] = []
    for _, row in completed.iterrows():
        key = (
            str(row.get("fold")),
            str(row.get("horizon")),
            str(row.get("seed")),
            str(row.get("model")),
        )
        baseline = baseline_by_key.get(key)
        if baseline is None:
            delta_rows.append(_insufficient_delta_row(row.to_dict()))
            continue
        payload: dict[str, Any] = {
            "fold": row.get("fold"),
            "horizon": _optional_int(row.get("horizon")),
            "seed": _optional_int(row.get("seed")),
            "model": row.get("model"),
            "ablation_mode": row.get("ablation_mode"),
            "feature_group": row.get("feature_group"),
        }
        for metric in ("accuracy", "macro_f1", "mcc", "ece"):
            base_value = _finite_float(baseline.get(metric))
            value = _finite_float(row.get(metric))
            payload[f"baseline_{metric}"] = base_value
            payload[metric] = value
            payload[f"delta_{metric}"] = (
                None if base_value is None or value is None else value - base_value
            )
        payload["interpretation"] = _interpret_delta(payload.get("delta_macro_f1"))
        delta_rows.append(payload)
    return delta_rows


def _run_one_spec(
    *,
    output_dir: Path,
    fold_input: _FoldInput,
    feature_frame: pd.DataFrame,
    group_columns: Mapping[str, Sequence[str]],
    spec: AblationSpec,
    horizon: int,
    label_column: str,
    y_all: pd.Series,
    split: _SplitIndices,
    seed: int,
    model_name: str,
    unsupported_groups: str,
    reuse_completed: bool,
) -> dict[str, Any]:
    run_dir = (
        output_dir
        / "runs"
        / _run_id(
            fold=fold_input.fold,
            horizon=horizon,
            seed=seed,
            model=model_name,
            spec=spec,
        )
    )
    status_path = run_dir / "status.json"
    metrics_path = run_dir / "metrics.json"
    if reuse_completed and status_path.is_file() and metrics_path.is_file():
        try:
            status_payload = _read_json(status_path)
            metrics_payload = _read_json(metrics_path)
            if status_payload.get("status") == "completed":
                return {
                    "row": dict(metrics_payload["results_summary_row"]),
                    "entry": _run_entry(run_dir, status="skipped_existing"),
                }
        except (OSError, ValueError, KeyError, TypeError):
            pass
    run_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = _columns_for_spec(group_columns, spec)
    proxy_columns = _proxy_columns_for_spec(group_columns, spec)
    config_snapshot = {
        "runner_version": FEATURE_ABLATION_VERSION,
        "fold": fold_input.fold,
        "horizon": horizon,
        "seed": seed,
        "model": model_name,
        "label_column": label_column,
        "ablation_mode": spec.mode,
        "feature_group": spec.feature_group,
        "feature_groups_used": list(spec.groups),
        "feature_columns": list(feature_columns),
        "proxy_feature_columns": list(proxy_columns),
    }
    (run_dir / "config_snapshot.json").write_text(
        stable_json_dumps(config_snapshot),
        encoding="utf-8",
    )
    (run_dir / "feature_groups.json").write_text(
        stable_json_dumps(
            {
                "feature_groups_used": list(spec.groups),
                "feature_columns": list(feature_columns),
                "proxy_feature_columns": list(proxy_columns),
                "unsupported_groups": unsupported_groups,
            }
        ),
        encoding="utf-8",
    )
    if not feature_columns:
        return _record_run_failure(
            run_dir,
            fold=fold_input.fold,
            horizon=horizon,
            seed=seed,
            model=model_name,
            spec=spec,
            unsupported_groups=unsupported_groups,
            reason="ablation spec has no feature columns",
        )
    try:
        prepared = _fit_model(
            feature_frame,
            y_all,
            feature_columns=feature_columns,
            train_indices=split.train,
            model_name=model_name,
            seed=seed,
        )
        row, predictions = _evaluate_model(
            fold_input=fold_input,
            feature_frame=feature_frame,
            y_all=y_all,
            feature_columns=feature_columns,
            proxy_columns=proxy_columns,
            split=split,
            prepared=prepared,
            model_name=model_name,
            horizon=horizon,
            seed=seed,
            spec=spec,
            unsupported_groups=unsupported_groups,
        )
        predictions_path = run_dir / "predictions.csv"
        predictions.to_csv(predictions_path, index=False)
        metrics_payload = {
            "status": "completed",
            "results_summary_row": row,
            "metrics": {
                "accuracy": row["accuracy"],
                "macro_f1": row["macro_f1"],
                "mcc": row["mcc"],
                "ece": row["ece"],
                "brier_score": row["brier_score"],
            },
            "prediction_file": "predictions.csv",
        }
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(stable_json_dumps(metrics_payload), encoding="utf-8")
        status_path.write_text(
            stable_json_dumps({"status": "completed", "failure_reason": ""}),
            encoding="utf-8",
        )
        _write_run_hash_manifest(run_dir)
        return {"row": row, "entry": _run_entry(run_dir, status="completed")}
    except (ImportError, RuntimeError, ValueError, TypeError, FloatingPointError) as exc:
        return _record_run_failure(
            run_dir,
            fold=fold_input.fold,
            horizon=horizon,
            seed=seed,
            model=model_name,
            spec=spec,
            unsupported_groups=unsupported_groups,
            reason=f"{type(exc).__name__}: {exc}",
        )


def _fit_model(
    feature_frame: pd.DataFrame,
    y_all: pd.Series,
    *,
    feature_columns: Sequence[str],
    train_indices: Sequence[int],
    model_name: str,
    seed: int,
) -> _PreparedModel:
    spec = get_paper_model_spec(model_name)
    model = create_baseline_model(build_paper_baseline_config(model_name, seed=seed))
    x_train_raw = (
        feature_frame.iloc[list(train_indices)].loc[:, list(feature_columns)].to_numpy(dtype=float)
    )
    y_train = y_all.iloc[list(train_indices)].to_numpy(dtype=int)
    scaler: TrainOnlyStandardScaler | None = None
    if spec.requires_standardisation:
        scaler = TrainOnlyStandardScaler()
        x_train = scaler.fit_transform(x_train_raw)
    else:
        x_train = x_train_raw
    model.fit(x_train, y_train)
    return _PreparedModel(model=model, scaler=scaler)


def _evaluate_model(
    *,
    fold_input: _FoldInput,
    feature_frame: pd.DataFrame,
    y_all: pd.Series,
    feature_columns: Sequence[str],
    proxy_columns: Sequence[str],
    split: _SplitIndices,
    prepared: _PreparedModel,
    model_name: str,
    horizon: int,
    seed: int,
    spec: AblationSpec,
    unsupported_groups: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    test_indices = list(split.test)
    if not test_indices:
        test_indices = list(split.validation)
    if not test_indices:
        raise ValueError("test/validation split contains no rows")
    x_raw = feature_frame.iloc[test_indices].loc[:, list(feature_columns)].to_numpy(dtype=float)
    x = prepared.scaler.transform(x_raw) if prepared.scaler is not None else x_raw
    y_true = y_all.iloc[test_indices].to_numpy(dtype=int)
    y_pred = prepared.model.predict(x).astype(int)
    probabilities = _predict_probabilities(prepared.model, x)
    metrics = compute_classification_metrics(
        y_true.tolist(),
        y_pred.tolist(),
        y_proba=probabilities,
        labels=list(_LABELS),
    )
    ece = _ece(y_true, probabilities) if probabilities is not None else None
    brier = _multiclass_brier(y_true, probabilities) if probabilities is not None else None
    row = {
        "fold": fold_input.fold,
        "horizon": horizon,
        "seed": seed,
        "model": model_name,
        "ablation_mode": spec.mode,
        "feature_group": spec.feature_group,
        "features_used": len(feature_columns),
        "proxy_features_used": len(proxy_columns),
        "unsupported_groups": unsupported_groups,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "mcc": metrics.matthews_corrcoef,
        "ece": ece,
        "brier_score": brier,
        "status": "completed",
    }
    predictions = _prediction_frame(
        fold_input=fold_input,
        feature_frame=feature_frame,
        row_indices=test_indices,
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        model_name=model_name,
        horizon=horizon,
        seed=seed,
        spec=spec,
        feature_columns=feature_columns,
        proxy_columns=proxy_columns,
    )
    return row, predictions


def _prediction_frame(
    *,
    fold_input: _FoldInput,
    feature_frame: pd.DataFrame,
    row_indices: Sequence[int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
    model_name: str,
    horizon: int,
    seed: int,
    spec: AblationSpec,
    feature_columns: Sequence[str],
    proxy_columns: Sequence[str],
) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "row_id": feature_frame.iloc[list(row_indices)]["row_id"].to_numpy(dtype=int),
            "fold": fold_input.fold,
            "horizon": horizon,
            "seed": seed,
            "lookback": 0,
            "model_family": model_name,
            "pretraining_objective": "feature_ablation",
            "ablation_mode": spec.mode,
            "feature_group": spec.feature_group,
            "feature_groups_used": ",".join(spec.groups),
            "features_used": len(feature_columns),
            "proxy_features_used": len(proxy_columns),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    if "split" in fold_input.frame.columns:
        base["partition"] = fold_input.frame.iloc[list(row_indices)]["split"].to_numpy(copy=True)
        base["split"] = base["partition"]
    else:
        base["partition"] = "test"
    if probabilities is not None:
        for index, class_name in enumerate(FI2010_CANONICAL_CLASS_ORDER):
            base[f"prob_{class_name}"] = probabilities[:, index]
        base["confidence"] = probabilities.max(axis=1)
    else:
        for class_name in FI2010_CANONICAL_CLASS_ORDER:
            base[f"prob_{class_name}"] = np.nan
        base["confidence"] = np.nan
    context_map = {
        "midprice": "mid_price",
        "spread": "spread",
        "relative_spread": "relative_spread",
        "best_bid_size": "bid_depth_1",
        "best_ask_size": "ask_depth_1",
        "top_of_book_imbalance": "imbalance",
    }
    for source_column, output_column in context_map.items():
        if source_column in feature_frame.columns:
            base[output_column] = feature_frame.iloc[list(row_indices)][source_column].to_numpy(
                copy=True
            )
    return base


def _predict_probabilities(model: BaseBaselineModel, x: np.ndarray) -> np.ndarray | None:
    raw = model.predict_proba(x)
    if raw is None:
        return None
    classes = _model_classes(model)
    if not classes:
        return None
    aligned = np.zeros((raw.shape[0], len(_LABELS)), dtype=float)
    class_to_index = {int(label): index for index, label in enumerate(classes)}
    for output_index, raw_label in enumerate(_LABELS):
        source_index = class_to_index.get(int(raw_label))
        if source_index is not None and source_index < raw.shape[1]:
            aligned[:, output_index] = raw[:, source_index]
    row_sums = aligned.sum(axis=1)
    positive = row_sums > 0.0
    aligned[positive] = aligned[positive] / row_sums[positive, None]
    return aligned


def _model_classes(model: BaseBaselineModel) -> list[int]:
    if hasattr(model, "classes_"):
        raw_classes = model.classes_
        return [int(value) for value in list(raw_classes)]
    estimator = getattr(model, "estimator", None)
    if estimator is not None and hasattr(estimator, "classes_"):
        return [int(value) for value in list(estimator.classes_)]
    return []


def _ece(y_true: np.ndarray, probabilities: np.ndarray | None, *, bins: int = 10) -> float | None:
    if probabilities is None or probabilities.size == 0:
        return None
    confidence = probabilities.max(axis=1)
    predicted = np.asarray(_LABELS, dtype=int)[probabilities.argmax(axis=1)]
    correct = (predicted == y_true).astype(float)
    total = len(y_true)
    if total == 0:
        return None
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for left, right in pairwise(edges):
        if right == 1.0:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if not bool(mask.any()):
            continue
        bin_confidence = float(confidence[mask].mean())
        bin_accuracy = float(correct[mask].mean())
        ece += float(mask.mean()) * abs(bin_accuracy - bin_confidence)
    return ece


def _multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray | None) -> float | None:
    if probabilities is None or probabilities.size == 0:
        return None
    target = np.zeros_like(probabilities, dtype=float)
    label_to_index = {label: index for index, label in enumerate(_LABELS)}
    for row_index, label in enumerate(y_true):
        target[row_index, label_to_index[int(label)]] = 1.0
    return float(np.mean(np.sum((probabilities - target) ** 2, axis=1)))


def _record_run_failure(
    run_dir: Path,
    *,
    fold: str,
    horizon: int,
    seed: int,
    model: str,
    spec: AblationSpec,
    unsupported_groups: str,
    reason: str,
) -> dict[str, Any]:
    row = _failed_result_row(
        fold=fold,
        horizon=horizon,
        seed=seed,
        model=model,
        mode=spec.mode,
        feature_group=spec.feature_group,
        error=reason,
        unsupported_groups=unsupported_groups,
    )
    (run_dir / "metrics.json").write_text(
        stable_json_dumps({"status": "failed", "results_summary_row": row}),
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(
        stable_json_dumps({"status": "failed", "failure_reason": reason}),
        encoding="utf-8",
    )
    _write_run_hash_manifest(run_dir)
    return {"row": row, "entry": _run_entry(run_dir, status="failed")}


def _failed_result_row(
    *,
    fold: str,
    horizon: int,
    seed: int,
    model: str,
    mode: str,
    feature_group: str,
    error: str,
    unsupported_groups: str = "",
) -> dict[str, Any]:
    return {
        "fold": fold,
        "horizon": horizon,
        "seed": seed,
        "model": model,
        "ablation_mode": mode,
        "feature_group": feature_group,
        "features_used": 0,
        "proxy_features_used": 0,
        "unsupported_groups": unsupported_groups,
        "accuracy": None,
        "macro_f1": None,
        "mcc": None,
        "ece": None,
        "brier_score": None,
        "status": f"failed: {error}",
    }


def _columns_for_spec(
    group_columns: Mapping[str, Sequence[str]],
    spec: AblationSpec,
) -> tuple[str, ...]:
    columns: list[str] = []
    for group in spec.groups:
        columns.extend(str(column) for column in group_columns.get(group, ()))
    return tuple(dict.fromkeys(columns))


def _proxy_columns_for_spec(
    group_columns: Mapping[str, Sequence[str]],
    spec: AblationSpec,
) -> tuple[str, ...]:
    proxies = set(proxy_group_names())
    columns: list[str] = []
    for group in spec.groups:
        if group in proxies:
            columns.extend(str(column) for column in group_columns.get(group, ()))
    return tuple(dict.fromkeys(columns))


def _split_indices(frame: pd.DataFrame) -> _SplitIndices:
    if "split" in frame.columns:
        split_values = frame["split"].astype(str).str.lower()
        train_all = [int(index) for index in frame.index[split_values == "train"].tolist()]
        test = [int(index) for index in frame.index[split_values == "test"].tolist()]
        if not train_all:
            train_all = [int(index) for index in frame.index[split_values != "test"].tolist()]
        validation_count = 0 if len(train_all) < 4 else max(1, round(len(train_all) * 0.2))
        validation = tuple(train_all[-validation_count:]) if validation_count else ()
        train = tuple(train_all[:-validation_count]) if validation_count else tuple(train_all)
        if not test:
            test = list(validation)
        return _SplitIndices(train=train, validation=validation, test=tuple(test))
    n_rows = len(frame)
    train_end = max(1, int(n_rows * 0.6))
    validation_end = max(train_end + 1, int(n_rows * 0.8)) if n_rows >= 3 else train_end
    validation_end = min(validation_end, n_rows)
    return _SplitIndices(
        train=tuple(range(0, train_end)),
        validation=tuple(range(train_end, validation_end)),
        test=tuple(range(validation_end, n_rows)) or tuple(range(train_end, validation_end)),
    )


def _raw_label_series(series: pd.Series) -> pd.Series:
    def convert(value: Any) -> int:
        class_name = canonical_class_name(value)
        return FI2010_CLASS_TO_RAW_LABEL[class_name]

    return series.apply(convert).astype(int)


def _resolve_fold_inputs(
    *,
    config_path: Path | None,
    processed_root: Path | None,
    data_path: Path | None,
    folds: tuple[str, ...] | None,
    smoke_test: bool,
    warnings: list[str],
) -> tuple[_FoldInput, ...]:
    selected_folds = folds or ("fold_1",)
    if data_path is not None and data_path.is_file():
        frame = pd.read_csv(data_path)
        sha = sha256_file(data_path)
        return tuple(
            _FoldInput(fold=fold, path=data_path, frame=frame.copy(), sha256=sha)
            for fold in selected_folds
        )
    if config_path is not None and processed_root is not None and config_path.is_file():
        try:
            from chronoslob.experiments.fi2010_multifold_runner import (
                load_multifold_classical_config,
            )

            config = load_multifold_classical_config(config_path)
            inputs: list[_FoldInput] = []
            for fold in selected_folds:
                number = int(fold.removeprefix("fold_"))
                filename = config.combined_csv_filename_template.replace("{fold}", str(number))
                path = processed_root / filename
                if not path.is_file():
                    warnings.append(f"prepared fold CSV missing: {path}")
                    continue
                inputs.append(
                    _FoldInput(
                        fold=fold,
                        path=path,
                        frame=pd.read_csv(path),
                        sha256=sha256_file(path),
                    )
                )
            if inputs:
                return tuple(inputs)
        except (OSError, ValueError, TypeError) as exc:
            warnings.append(f"could not resolve prepared folds from config: {exc}")
    if smoke_test:
        fixture = Path("tests/fixtures/fi2010/tiny_fi2010_like.csv")
        if fixture.is_file():
            frame = pd.read_csv(fixture)
            return tuple(
                _FoldInput(
                    fold=fold,
                    path=fixture,
                    frame=frame.copy(),
                    sha256=sha256_file(fixture),
                )
                for fold in selected_folds[:1]
            )
    raise FileNotFoundError(
        "no FI-2010 fold inputs were available; provide --data-path, "
        "--processed-root with --config, or run with --smoke-test"
    )


def _prepare_output_dir(path: Path, *, reuse_completed: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {path}")
        if any(path.iterdir()) and not reuse_completed:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_fold_features(
    output_dir: Path,
    *,
    fold_input: _FoldInput,
    feature_result: Any,
    strict: bool,
) -> None:
    fold_dir = output_dir / "features" / fold_input.fold
    if fold_dir.exists():
        shutil.rmtree(fold_dir)
    if fold_input.sha256 is None:
        return
    temp_path = fold_dir / "_input.csv"
    fold_dir.mkdir(parents=True, exist_ok=True)
    fold_input.frame.to_csv(temp_path, index=False)
    try:
        build_microstructure_feature_artifacts(
            temp_path,
            out_dir=fold_dir / "artefacts",
            feature_groups=list(feature_result.group_columns),
            strict=strict,
            overwrite=True,
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [{column: row.get(column) for column in columns} for row in rows], columns=list(columns)
    )
    frame.to_csv(path, index=False)


def _write_root_manifest(
    *,
    output_dir: Path,
    run_entries: Sequence[Mapping[str, Any]],
    input_folds: Sequence[_FoldInput],
) -> Path:
    path = output_dir / "ablation_manifest.json"
    payload = {
        "runner_version": FEATURE_ABLATION_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "input_folds": [
            {"fold": item.fold, "path": str(item.path), "sha256": item.sha256}
            for item in input_folds
        ],
        "runs": list(run_entries),
        "status_counts": {
            "completed": sum(1 for entry in run_entries if entry.get("status") == "completed"),
            "failed": sum(1 for entry in run_entries if entry.get("status") == "failed"),
            "skipped_existing": sum(
                1 for entry in run_entries if entry.get("status") == "skipped_existing"
            ),
        },
    }
    path.write_text(stable_json_dumps(payload), encoding="utf-8")
    return path


def _write_sha256_manifest(output_dir: Path) -> None:
    files: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.json":
            files[str(path.relative_to(output_dir)).replace("\\", "/")] = sha256_file(path)
    (output_dir / "sha256_manifest.json").write_text(
        stable_json_dumps({"files": files}),
        encoding="utf-8",
    )


def _write_run_hash_manifest(run_dir: Path) -> None:
    files = {
        path.name: sha256_file(path)
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "sha256_manifest.json"
    }
    (run_dir / "sha256_manifest.json").write_text(
        stable_json_dumps({"files": files}),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _run_entry(run_dir: Path, *, status: str) -> dict[str, Any]:
    return {"run_dir": str(run_dir), "status": status}


def _run_id(
    *,
    fold: str,
    horizon: int,
    seed: int,
    model: str,
    spec: AblationSpec,
) -> str:
    return (f"{fold}_h{horizon}_s{seed}_{model}_{spec.mode}_{spec.feature_group}").replace("/", "_")


def _unsupported_summary(manifest: Mapping[str, Any]) -> str:
    unsupported = manifest.get("unsupported_groups")
    if not isinstance(unsupported, list):
        return ",".join(unsupported_group_names())
    names = [
        str(item.get("name"))
        for item in unsupported
        if isinstance(item, Mapping) and item.get("name")
    ]
    return ",".join(dict.fromkeys(names))


def _partition_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column for column in ("fold", "fold_id", "split", "partition") if column in frame.columns
    )


def _normalise_models(models: Sequence[str] | str | None, *, smoke_test: bool) -> tuple[str, ...]:
    default = ("logistic",) if smoke_test else CLASSICAL_FEATURE_ABLATION_MODELS
    if models is None:
        raw: Sequence[str] = default
    elif isinstance(models, str):
        text = models.strip()
        raw = (
            default
            if not text or text.lower() == "all"
            else [token.strip() for token in text.split(",")]
        )
    else:
        raw = models
    cleaned: list[str] = []
    for item in raw:
        spec = get_paper_model_spec(str(item).strip().lower())
        if spec.model_family != "classical":
            continue
        if spec.name not in CLASSICAL_FEATURE_ABLATION_MODELS:
            continue
        if spec.name not in cleaned:
            cleaned.append(spec.name)
    if not cleaned:
        raise ValueError("at least one supported classical model must be selected")
    return tuple(cleaned)


def _normalise_modes(modes: Sequence[str] | str | None) -> tuple[str, ...]:
    if modes is None:
        return ABLATION_MODES
    if isinstance(modes, str):
        text = modes.strip()
        raw: Sequence[str] = (
            ABLATION_MODES
            if not text or text.lower() == "all"
            else [token.strip() for token in text.split(",")]
        )
    else:
        raw = modes
    cleaned: list[str] = []
    for item in raw:
        name = str(item).strip().lower()
        if name not in ABLATION_MODES:
            raise ValueError(
                f"unsupported ablation mode {item!r}; supported: {list(ABLATION_MODES)}"
            )
        if name not in cleaned:
            cleaned.append(name)
    return tuple(cleaned)


def _normalise_groups(groups: Sequence[str] | str | None) -> tuple[str, ...]:
    if groups is None:
        return DEFAULT_FEATURE_GROUPS
    if isinstance(groups, str):
        text = groups.strip()
        raw: Sequence[str] = (
            DEFAULT_FEATURE_GROUPS
            if not text or text.lower() == "all"
            else [token.strip() for token in text.split(",")]
        )
    else:
        raw = groups
    cleaned: list[str] = []
    for item in raw:
        name = str(item).strip().lower()
        if not name:
            continue
        get_feature_group(name)
        if name not in cleaned:
            cleaned.append(name)
    if not cleaned:
        raise ValueError("at least one feature group must be selected")
    return tuple(cleaned)


def _parse_folds(value: Sequence[str | int] | str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "all":
            return None
        raw: Sequence[str | int] = [token.strip() for token in text.split(",")]
    else:
        raw = value
    cleaned: list[str] = []
    for item in raw:
        text = str(item).strip().lower()
        if not text:
            continue
        if text.isdigit():
            fold = f"fold_{int(text)}"
        elif text.startswith("fold_") and text.removeprefix("fold_").isdigit():
            fold = f"fold_{int(text.removeprefix('fold_'))}"
        else:
            raise ValueError(f"invalid fold {item!r}")
        if fold not in cleaned:
            cleaned.append(fold)
    return tuple(cleaned) if cleaned else None


def _parse_ints(
    value: Sequence[int] | str | None,
    *,
    default: tuple[int, ...],
    positive: bool,
) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "all":
            return default
        raw: Sequence[Any] = [token.strip() for token in text.split(",")]
    else:
        raw = value
    cleaned: list[int] = []
    for item in raw:
        number = int(item)
        if positive and number <= 0:
            raise ValueError("integer selections must be positive")
        if not positive and number < 0:
            raise ValueError("integer selections must be non-negative")
        if number not in cleaned:
            cleaned.append(number)
    return tuple(cleaned) if cleaned else default


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: pd.Series) -> float | None:
    if values.empty:
        return None
    value = float(values.mean())
    return value if math.isfinite(value) else None


def _std(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    value = float(values.std(ddof=0))
    return value if math.isfinite(value) else 0.0


def _insufficient_delta_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fold": row.get("fold"),
        "horizon": _optional_int(row.get("horizon")),
        "seed": _optional_int(row.get("seed")),
        "model": row.get("model"),
        "ablation_mode": row.get("ablation_mode"),
        "feature_group": row.get("feature_group"),
        "baseline_accuracy": None,
        "accuracy": _finite_float(row.get("accuracy")),
        "delta_accuracy": None,
        "baseline_macro_f1": None,
        "macro_f1": _finite_float(row.get("macro_f1")),
        "delta_macro_f1": None,
        "baseline_mcc": None,
        "mcc": _finite_float(row.get("mcc")),
        "delta_mcc": None,
        "baseline_ece": None,
        "ece": _finite_float(row.get("ece")),
        "delta_ece": None,
        "interpretation": "insufficient evidence",
    }


def _interpret_delta(delta: Any) -> str:
    value = _finite_float(delta)
    if value is None:
        return "insufficient evidence"
    if value > 0.002:
        return "helped"
    if value < -0.002:
        return "hurt"
    return "neutral"
