"""Reproducible FI-2010 neural full-grid figure generation.

The builder consumes stored full-grid artefacts only.  It validates the
FI-2010 label mapping before plotting, writes source CSVs for every generated
figure and records completed and skipped plots in a manifest.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from chronoslob.analysis.fi2010_label_mapping import (
    FI2010_CANONICAL_CLASS_ORDER,
    FI2010_RAW_LABEL_TO_CLASS,
    canonical_class_name,
    probability_columns_for_order,
    validate_class_order,
    validate_classwise_f1_columns,
    validate_confusion_matrix_axis_labels,
    validate_probability_columns,
)
from chronoslob.experiments.manifests import stable_json_dumps
from chronoslob.utils.paths import project_root

__all__ = [
    "DEFAULT_FI2010_FIGURE_OUTPUT_DIR",
    "FI2010_FIGURE_BUILDER_VERSION",
    "FI2010FigureBuildSummary",
    "build_fi2010_neural_figures",
    "build_matched_ssl_delta_rows",
    "select_best_models_by_horizon",
]

FI2010_FIGURE_BUILDER_VERSION = "fi2010-neural-figures/v1"
DEFAULT_FI2010_FIGURE_OUTPUT_DIR = Path("reports/figures/fi2010_neural_full_grid")

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)
_SUCCESS_STATUSES = {"completed", "skipped_existing", "ok"}
_SSL_OBJECTIVES = ("masked_reconstruction", "next_field")
_PLOT_DPI = 160
_PLOT_FIGURE_SIZE = (7.4, 4.6)
_CONFIDENCE_THRESHOLDS: tuple[float, ...] = (
    0.33,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
)


class FI2010FigureBuildSummary(BaseModel):
    """Summary returned after building FI-2010 full-grid figures."""

    model_config = _MODEL_CONFIG

    output_dir: str
    neural_full_grid_dir: str
    manifest_path: str
    label_mapping_audit_path: str
    best_model_selection_path: str
    completed_figures: list[str] = Field(default_factory=list)
    skipped_figures: list[str] = Field(default_factory=list)
    smoke_test: bool
    warnings: list[str] = Field(default_factory=list)
    builder_version: str
    created_at: datetime

    @field_validator("output_dir", "neural_full_grid_dir", "builder_version")
    @classmethod
    def _validate_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("figure summary string fields must be non-empty")
        return value.strip()


def build_fi2010_neural_figures(
    *,
    neural_full_grid_dir: Path,
    out_dir: Path = DEFAULT_FI2010_FIGURE_OUTPUT_DIR,
    execution_v3_dir: Path | None = None,
    models: Sequence[str] | None = None,
    horizons: Sequence[int] | None = None,
    folds: Sequence[int | str] | None = None,
    seeds: Sequence[int] | None = None,
    overwrite: bool = False,
    allow_smoke_test: bool = False,
    strict: bool = True,
) -> FI2010FigureBuildSummary:
    """Build FI-2010 neural/SSL diagnostic figures from stored artefacts."""
    grid_dir = Path(neural_full_grid_dir)
    if not grid_dir.exists():
        raise FileNotFoundError(f"neural full-grid directory does not exist: {grid_dir}")
    if not grid_dir.is_dir():
        raise NotADirectoryError(f"neural full-grid path is not a directory: {grid_dir}")

    summary_path = grid_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"full-grid summary.json is missing: {summary_path}")
    grid_summary = _read_json(summary_path)
    smoke_test = _is_smoke_summary(grid_summary)
    if smoke_test and not allow_smoke_test:
        raise ValueError(
            "FI-2010 figure generation refuses smoke-test artefacts unless "
            "allow_smoke_test=True"
        )

    output_dir = Path(out_dir)
    _ensure_output_dir(output_dir, input_dir=grid_dir, overwrite=overwrite)
    source_dir = output_dir / "source_data"
    metadata_dir = output_dir / "metadata"
    source_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    results = _load_results(grid_dir, warnings=warnings)
    filtered_results = _filter_result_rows(
        results,
        models=models,
        horizons=horizons,
        folds=folds,
        seeds=seeds,
    )
    predictions_raw = _load_prediction_frame(filtered_results, grid_dir, warnings=warnings)
    label_audit = _build_label_mapping_audit(
        predictions_raw,
        filtered_results,
        warnings=warnings,
    )
    audit_path = output_dir / "label_mapping_audit.json"
    audit_path.write_text(stable_json_dumps(label_audit), encoding="utf-8")
    if strict and label_audit["status"] != "pass":
        raise ValueError(
            "FI-2010 label mapping audit failed in strict mode; see "
            f"{audit_path}"
        )

    predictions = _canonicalise_predictions(predictions_raw, warnings=warnings)
    execution_v3 = _resolve_execution_v3_dir(execution_v3_dir, grid_dir)
    best_selection = select_best_models_by_horizon(filtered_results)
    best_path = output_dir / "best_model_selection.json"
    best_path.write_text(stable_json_dumps(best_selection), encoding="utf-8")

    entries: list[dict[str, Any]] = []
    _build_confusion_matrix_figures(
        entries=entries,
        predictions=predictions,
        best_selection=best_selection,
        out_dir=output_dir,
        source_dir=source_dir,
        metadata_dir=metadata_dir,
        smoke_test=smoke_test,
    )
    _build_reliability_figure(
        entries=entries,
        predictions=predictions,
        out_dir=output_dir,
        source_dir=source_dir,
        metadata_dir=metadata_dir,
        smoke_test=smoke_test,
    )
    _build_macro_f1_by_fold_figure(
        entries=entries,
        results=filtered_results,
        out_dir=output_dir,
        source_dir=source_dir,
        metadata_dir=metadata_dir,
        smoke_test=smoke_test,
    )
    _build_metric_by_horizon_figure(
        entries=entries,
        results=filtered_results,
        metric="macro_f1",
        figure_id="macro_f1_by_horizon",
        title="Macro-F1 Across Horizons",
        ylabel="mean macro-F1",
        out_dir=output_dir,
        source_dir=source_dir,
        metadata_dir=metadata_dir,
        smoke_test=smoke_test,
    )
    _build_metric_by_horizon_figure(
        entries=entries,
        results=filtered_results,
        metric="ece",
        figure_id="ece_by_horizon",
        title="ECE Across Horizons",
        ylabel="mean ECE",
        out_dir=output_dir,
        source_dir=source_dir,
        metadata_dir=metadata_dir,
        smoke_test=smoke_test,
    )
    _build_ssl_delta_figure(
        entries=entries,
        results=filtered_results,
        out_dir=output_dir,
        source_dir=source_dir,
        metadata_dir=metadata_dir,
        smoke_test=smoke_test,
    )
    _build_threshold_fraction_figure(
        entries=entries,
        predictions=predictions,
        out_dir=output_dir,
        source_dir=source_dir,
        metadata_dir=metadata_dir,
        smoke_test=smoke_test,
    )
    _build_threshold_macro_f1_figure(
        entries=entries,
        predictions=predictions,
        out_dir=output_dir,
        source_dir=source_dir,
        metadata_dir=metadata_dir,
        smoke_test=smoke_test,
    )
    _build_cost_proxy_figure(
        entries=entries,
        grid_dir=grid_dir,
        execution_v3_dir=execution_v3,
        out_dir=output_dir,
        source_dir=source_dir,
        metadata_dir=metadata_dir,
        smoke_test=smoke_test,
    )
    _build_execution_v3_figures(
        entries=entries,
        execution_v3_dir=execution_v3,
        out_dir=output_dir,
        source_dir=source_dir,
        metadata_dir=metadata_dir,
        smoke_test=smoke_test,
    )
    _build_regime_breakdown_figure(
        entries=entries,
        predictions=predictions,
        out_dir=output_dir,
        source_dir=source_dir,
        metadata_dir=metadata_dir,
        smoke_test=smoke_test,
    )

    completed = [entry["figure_id"] for entry in entries if entry["status"] == "completed"]
    skipped = [entry["figure_id"] for entry in entries if entry["status"] == "skipped"]
    manifest = {
        "builder_version": FI2010_FIGURE_BUILDER_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "neural_full_grid_dir": _display_path(grid_dir),
        "execution_v3_dir": None if execution_v3 is None else _display_path(execution_v3),
        "output_dir": _display_path(output_dir),
        "smoke_test": smoke_test,
        "allow_smoke_test": bool(allow_smoke_test),
        "strict": bool(strict),
        "label_mapping_audit": _display_path(audit_path),
        "best_model_selection": _display_path(best_path),
        "figures": entries,
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(stable_json_dumps(manifest), encoding="utf-8")

    return FI2010FigureBuildSummary(
        output_dir=str(output_dir),
        neural_full_grid_dir=str(grid_dir),
        manifest_path=str(manifest_path),
        label_mapping_audit_path=str(audit_path),
        best_model_selection_path=str(best_path),
        completed_figures=completed,
        skipped_figures=skipped,
        smoke_test=smoke_test,
        warnings=warnings + [
            str(entry["reason"]) for entry in entries if entry["status"] == "skipped"
        ],
        builder_version=FI2010_FIGURE_BUILDER_VERSION,
        created_at=datetime.now(UTC),
    )


def select_best_models_by_horizon(results: pd.DataFrame) -> dict[str, Any]:
    """Select the best model/objective per horizon using conservative rules."""
    if results.empty:
        return {
            "selection_metric": "mean_macro_f1",
            "tie_breakers": ["lower_mean_ece", "lower_macro_f1_variance"],
            "selected": [],
            "reason": "no completed result rows available",
        }
    frame = _result_frame_with_keys(results)
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["horizon", "model_family", "pretraining_objective", "lookback"],
        dropna=False,
        sort=True,
    ):
        horizon, model_family, pretraining_objective, lookback = keys
        macro = pd.to_numeric(group["macro_f1"], errors="coerce").dropna()
        if macro.empty:
            continue
        ece = pd.to_numeric(group.get("ece"), errors="coerce").dropna()
        rows.append(
            {
                "horizon": int(float(horizon)),
                "model_family": str(model_family),
                "pretraining_objective": str(pretraining_objective),
                "objective_label": _objective_label(pretraining_objective),
                "lookback": _optional_int(lookback),
                "model_key": _model_key(model_family, pretraining_objective),
                "mean_macro_f1": float(macro.mean()),
                "mean_ece": float(ece.mean()) if not ece.empty else math.inf,
                "macro_f1_variance": float(macro.var(ddof=0)) if len(macro) > 1 else 0.0,
                "run_count": len(group),
                "folds": _sorted_unique_ints(group.get("fold")),
                "seeds": _sorted_unique_ints(group.get("seed")),
            }
        )
    selected: list[dict[str, Any]] = []
    if not rows:
        return {
            "selection_metric": "mean_macro_f1",
            "tie_breakers": ["lower_mean_ece", "lower_macro_f1_variance"],
            "selected": [],
            "reason": "completed rows did not contain finite macro-F1",
        }
    candidates = pd.DataFrame(rows)
    for horizon, group in candidates.groupby("horizon", sort=True):
        ordered = group.sort_values(
            by=["mean_macro_f1", "mean_ece", "macro_f1_variance", "model_key"],
            ascending=[False, True, True, True],
        )
        top = dict(ordered.iloc[0].to_dict())
        top["horizon"] = int(horizon)
        top["reason"] = (
            "selected by highest mean macro-F1; ties prefer lower ECE, then "
            "lower macro-F1 variance"
        )
        selected.append(_json_ready(top))
    return {
        "selection_metric": "mean_macro_f1",
        "tie_breakers": ["lower_mean_ece", "lower_macro_f1_variance"],
        "selected": selected,
        "candidate_count": len(rows),
    }


def build_matched_ssl_delta_rows(results: pd.DataFrame) -> list[dict[str, Any]]:
    """Build matched supervised-vs-SSL deltas from completed result rows only."""
    if results.empty:
        return []
    frame = _result_frame_with_keys(results)
    rows: list[dict[str, Any]] = []
    key_columns = ["fold", "horizon", "seed", "lookback", "model_family"]
    supervised = frame[frame["objective_label"] == "supervised"]
    for _, ssl_row in frame[frame["objective_label"].isin(_SSL_OBJECTIVES)].iterrows():
        mask = pd.Series(True, index=supervised.index)
        for column in key_columns:
            mask &= supervised[column].astype(str) == str(ssl_row[column])
        matches = supervised.loc[mask]
        if len(matches) != 1:
            continue
        base = matches.iloc[0]
        delta_row: dict[str, Any] = {
            "fold": _optional_int(ssl_row["fold"]),
            "horizon": _optional_int(ssl_row["horizon"]),
            "seed": _optional_int(ssl_row["seed"]),
            "lookback": _optional_int(ssl_row["lookback"]),
            "model_family": str(ssl_row["model_family"]),
            "ssl_objective": str(ssl_row["objective_label"]),
            "supervised_run_id": str(base.get("run_id", "")),
            "ssl_run_id": str(ssl_row.get("run_id", "")),
        }
        for metric in ("macro_f1", "mcc", "ece"):
            supervised_value = _finite_float(base.get(metric))
            ssl_value = _finite_float(ssl_row.get(metric))
            delta_row[f"supervised_{metric}"] = supervised_value
            delta_row[f"ssl_{metric}"] = ssl_value
            delta_row[f"delta_{metric}"] = (
                None
                if supervised_value is None or ssl_value is None
                else ssl_value - supervised_value
            )
        rows.append(delta_row)
    return rows


def _build_confusion_matrix_figures(
    *,
    entries: list[dict[str, Any]],
    predictions: pd.DataFrame,
    best_selection: Mapping[str, Any],
    out_dir: Path,
    source_dir: Path,
    metadata_dir: Path,
    smoke_test: bool,
) -> None:
    selected = best_selection.get("selected")
    if predictions.empty or not isinstance(selected, list) or not selected:
        _skip_entry(
            entries,
            figure_id="confusion_matrix_best_by_horizon",
            title="Confusion Matrix Best Model By Horizon",
            reason="prediction artefacts or best-model selection unavailable",
            smoke_test=smoke_test,
        )
        return
    wrote_any = False
    for item in selected:
        if not isinstance(item, Mapping):
            continue
        horizon = _optional_int(item.get("horizon"))
        if horizon is None:
            continue
        subset = predictions[
            (predictions["horizon"].astype(str) == str(horizon))
            & (predictions["model_family"].astype(str) == str(item.get("model_family")))
            & (
                predictions["pretraining_objective"].astype(str)
                == str(item.get("pretraining_objective"))
            )
            & (predictions["lookback"].astype(str) == str(item.get("lookback")))
        ].copy()
        if subset.empty:
            _skip_entry(
                entries,
                figure_id=f"confusion_matrix_h{horizon}",
                title=f"Confusion Matrix H{horizon}",
                reason="selected model has no prediction rows",
                smoke_test=smoke_test,
            )
            continue
        rows = _confusion_rows(subset)
        source_path = source_dir / f"confusion_matrix_h{horizon}.csv"
        pd.DataFrame(rows).to_csv(source_path, index=False)
        figure_id = f"confusion_matrix_h{horizon}"
        title = f"Confusion Matrix H{horizon}"
        target = out_dir / f"{figure_id}.png"
        metadata_path = _write_metadata(
            metadata_dir,
            figure_id=figure_id,
            title=title,
            source_path=source_path,
            smoke_test=smoke_test,
        )
        _plot_confusion(rows, target, title=_smoke_title(title, smoke_test))
        _complete_entry(
            entries,
            figure_id=figure_id,
            title=title,
            file_path=target,
            source_path=source_path,
            metadata_path=metadata_path,
            frame=subset,
            smoke_test=smoke_test,
        )
        wrote_any = True
    if not wrote_any:
        _skip_entry(
            entries,
            figure_id="confusion_matrix_best_by_horizon",
            title="Confusion Matrix Best Model By Horizon",
            reason="no selected horizon produced a confusion matrix",
            smoke_test=smoke_test,
        )


def _build_reliability_figure(
    *,
    entries: list[dict[str, Any]],
    predictions: pd.DataFrame,
    out_dir: Path,
    source_dir: Path,
    metadata_dir: Path,
    smoke_test: bool,
) -> None:
    figure_id = "reliability_curve"
    title = "Reliability Curve"
    if predictions.empty:
        _skip_entry(
            entries,
            figure_id=figure_id,
            title=title,
            reason="prediction artefacts not available",
            smoke_test=smoke_test,
        )
        return
    rows = _reliability_rows(predictions)
    if not rows:
        _skip_entry(
            entries,
            figure_id=figure_id,
            title=title,
            reason="no non-empty reliability bins could be computed",
            smoke_test=smoke_test,
        )
        return
    source_path = source_dir / f"{figure_id}.csv"
    source = pd.DataFrame(rows)
    source.to_csv(source_path, index=False)
    target = out_dir / f"{figure_id}.png"
    metadata_path = _write_metadata(
        metadata_dir,
        figure_id=figure_id,
        title=title,
        source_path=source_path,
        smoke_test=smoke_test,
    )
    _plot_reliability(source, target, title=_smoke_title(title, smoke_test))
    _complete_entry(
        entries,
        figure_id=figure_id,
        title=title,
        file_path=target,
        source_path=source_path,
        metadata_path=metadata_path,
        frame=predictions,
        smoke_test=smoke_test,
    )


def _build_macro_f1_by_fold_figure(
    *,
    entries: list[dict[str, Any]],
    results: pd.DataFrame,
    out_dir: Path,
    source_dir: Path,
    metadata_dir: Path,
    smoke_test: bool,
) -> None:
    figure_id = "macro_f1_by_fold"
    title = "Macro-F1 Across Folds"
    if results.empty or "macro_f1" not in results.columns:
        _skip_entry(
            entries,
            figure_id=figure_id,
            title=title,
            reason="completed result rows with macro-F1 are unavailable",
            smoke_test=smoke_test,
        )
        return
    frame = _result_frame_with_keys(results)
    frame["macro_f1"] = pd.to_numeric(frame["macro_f1"], errors="coerce")
    source = frame.dropna(subset=["macro_f1"]).copy()
    if source.empty:
        _skip_entry(
            entries,
            figure_id=figure_id,
            title=title,
            reason="macro-F1 rows are non-finite",
            smoke_test=smoke_test,
        )
        return
    source_path = source_dir / f"{figure_id}.csv"
    source.to_csv(source_path, index=False)
    target = out_dir / f"{figure_id}.png"
    metadata_path = _write_metadata(
        metadata_dir,
        figure_id=figure_id,
        title=title,
        source_path=source_path,
        smoke_test=smoke_test,
    )
    _plot_fold_metric(source, target, title=_smoke_title(title, smoke_test))
    _complete_entry(
        entries,
        figure_id=figure_id,
        title=title,
        file_path=target,
        source_path=source_path,
        metadata_path=metadata_path,
        frame=source,
        smoke_test=smoke_test,
    )


def _build_metric_by_horizon_figure(
    *,
    entries: list[dict[str, Any]],
    results: pd.DataFrame,
    metric: str,
    figure_id: str,
    title: str,
    ylabel: str,
    out_dir: Path,
    source_dir: Path,
    metadata_dir: Path,
    smoke_test: bool,
) -> None:
    if results.empty or metric not in results.columns:
        _skip_entry(
            entries,
            figure_id=figure_id,
            title=title,
            reason=f"completed result rows with {metric} are unavailable",
            smoke_test=smoke_test,
        )
        return
    source = _horizon_summary(results, metric=metric)
    if source.empty:
        _skip_entry(
            entries,
            figure_id=figure_id,
            title=title,
            reason=f"{metric} rows are non-finite",
            smoke_test=smoke_test,
        )
        return
    source_path = source_dir / f"{figure_id}.csv"
    source.to_csv(source_path, index=False)
    target = out_dir / f"{figure_id}.png"
    metadata_path = _write_metadata(
        metadata_dir,
        figure_id=figure_id,
        title=title,
        source_path=source_path,
        smoke_test=smoke_test,
    )
    _plot_horizon_metric(
        source,
        target,
        title=_smoke_title(title, smoke_test),
        ylabel=ylabel,
    )
    _complete_entry(
        entries,
        figure_id=figure_id,
        title=title,
        file_path=target,
        source_path=source_path,
        metadata_path=metadata_path,
        frame=source,
        smoke_test=smoke_test,
    )


def _build_ssl_delta_figure(
    *,
    entries: list[dict[str, Any]],
    results: pd.DataFrame,
    out_dir: Path,
    source_dir: Path,
    metadata_dir: Path,
    smoke_test: bool,
) -> None:
    figure_id = "ssl_matched_delta"
    title = "Matched SSL Deltas"
    rows = build_matched_ssl_delta_rows(results)
    source_path = source_dir / f"{figure_id}.csv"
    source = pd.DataFrame(rows)
    source.to_csv(source_path, index=False)
    if source.empty:
        _skip_entry(
            entries,
            figure_id=figure_id,
            title=title,
            reason="no matched supervised-vs-SSL fold/horizon/seed/lookback pairs",
            smoke_test=smoke_test,
            source_path=source_path,
        )
        return
    target = out_dir / f"{figure_id}.png"
    metadata_path = _write_metadata(
        metadata_dir,
        figure_id=figure_id,
        title=title,
        source_path=source_path,
        smoke_test=smoke_test,
    )
    _plot_ssl_deltas(source, target, title=_smoke_title(title, smoke_test))
    _complete_entry(
        entries,
        figure_id=figure_id,
        title=title,
        file_path=target,
        source_path=source_path,
        metadata_path=metadata_path,
        frame=source,
        smoke_test=smoke_test,
    )


def _build_threshold_fraction_figure(
    *,
    entries: list[dict[str, Any]],
    predictions: pd.DataFrame,
    out_dir: Path,
    source_dir: Path,
    metadata_dir: Path,
    smoke_test: bool,
) -> None:
    figure_id = "confidence_threshold_eligible_fraction"
    title = "Confidence Threshold Vs Eligible Fraction"
    if predictions.empty:
        _skip_entry(
            entries,
            figure_id=figure_id,
            title=title,
            reason="prediction artefacts not available",
            smoke_test=smoke_test,
        )
        return
    source = pd.DataFrame(_threshold_fraction_rows(predictions))
    source_path = source_dir / f"{figure_id}.csv"
    source.to_csv(source_path, index=False)
    if source.empty:
        _skip_entry(
            entries,
            figure_id=figure_id,
            title=title,
            reason="no confidence rows could be computed",
            smoke_test=smoke_test,
            source_path=source_path,
        )
        return
    target = out_dir / f"{figure_id}.png"
    metadata_path = _write_metadata(
        metadata_dir,
        figure_id=figure_id,
        title=title,
        source_path=source_path,
        smoke_test=smoke_test,
    )
    _plot_threshold_metric(
        source,
        target,
        value_column="eligible_fraction",
        ylabel="eligible sample fraction",
        title=_smoke_title(title, smoke_test),
    )
    _complete_entry(
        entries,
        figure_id=figure_id,
        title=title,
        file_path=target,
        source_path=source_path,
        metadata_path=metadata_path,
        frame=predictions,
        smoke_test=smoke_test,
    )


def _build_threshold_macro_f1_figure(
    *,
    entries: list[dict[str, Any]],
    predictions: pd.DataFrame,
    out_dir: Path,
    source_dir: Path,
    metadata_dir: Path,
    smoke_test: bool,
) -> None:
    figure_id = "confidence_threshold_macro_f1"
    title = "Confidence Threshold Vs Retained Macro-F1"
    if predictions.empty:
        _skip_entry(
            entries,
            figure_id=figure_id,
            title=title,
            reason="prediction artefacts not available",
            smoke_test=smoke_test,
        )
        return
    source = pd.DataFrame(_threshold_macro_f1_rows(predictions))
    source_path = source_dir / f"{figure_id}.csv"
    source.to_csv(source_path, index=False)
    ok = source[source["status"] == "ok"] if "status" in source else source.iloc[0:0]
    if ok.empty:
        _skip_entry(
            entries,
            figure_id=figure_id,
            title=title,
            reason="all thresholds retained too few samples",
            smoke_test=smoke_test,
            source_path=source_path,
        )
        return
    target = out_dir / f"{figure_id}.png"
    metadata_path = _write_metadata(
        metadata_dir,
        figure_id=figure_id,
        title=title,
        source_path=source_path,
        smoke_test=smoke_test,
    )
    _plot_threshold_metric(
        ok,
        target,
        value_column="macro_f1",
        ylabel="macro-F1 on retained samples",
        title=_smoke_title(title, smoke_test),
    )
    _complete_entry(
        entries,
        figure_id=figure_id,
        title=title,
        file_path=target,
        source_path=source_path,
        metadata_path=metadata_path,
        frame=predictions,
        smoke_test=smoke_test,
    )


def _build_cost_proxy_figure(
    *,
    entries: list[dict[str, Any]],
    grid_dir: Path,
    execution_v3_dir: Path | None,
    out_dir: Path,
    source_dir: Path,
    metadata_dir: Path,
    smoke_test: bool,
) -> None:
    figure_id = "cost_adjusted_proxy"
    title = "Cost-Adjusted Proxy Diagnostic"
    candidates = [
        grid_dir / "execution_v3_proxy.csv",
        grid_dir / "cost_adjusted_proxy.csv",
        grid_dir / "execution_proxy_diagnostics.csv",
    ]
    if execution_v3_dir is not None:
        candidates.insert(0, execution_v3_dir / "cost_sensitivity_summary.csv")
    source_input = next((path for path in candidates if path.is_file()), None)
    if source_input is None:
        _skip_entry(
            entries,
            figure_id=figure_id,
            title=title,
            reason="execution v3 artefacts not available",
            smoke_test=smoke_test,
        )
        return
    frame = pd.read_csv(source_input)
    required_value = (
        "net_proxy"
        if "net_proxy" in frame.columns
        else "proxy_value"
        if "proxy_value" in frame.columns
        else "value"
    )
    if required_value not in frame.columns:
        _skip_entry(
            entries,
            figure_id=figure_id,
            title=title,
            reason="proxy artefact lacks a proxy_value/value column",
            smoke_test=smoke_test,
        )
        return
    source = frame.copy()
    source_path = source_dir / f"{figure_id}.csv"
    source.to_csv(source_path, index=False)
    target = out_dir / f"{figure_id}.png"
    metadata_path = _write_metadata(
        metadata_dir,
        figure_id=figure_id,
        title=title,
        source_path=source_path,
        smoke_test=smoke_test,
        extra={"proxy_label": "proxy diagnostics only"},
    )
    _plot_cost_proxy(source, target, value_column=required_value, title=title)
    _complete_entry(
        entries,
        figure_id=figure_id,
        title=title,
        file_path=target,
        source_path=source_path,
        metadata_path=metadata_path,
        frame=source,
        smoke_test=smoke_test,
    )


def _build_execution_v3_figures(
    *,
    entries: list[dict[str, Any]],
    execution_v3_dir: Path | None,
    out_dir: Path,
    source_dir: Path,
    metadata_dir: Path,
    smoke_test: bool,
) -> None:
    figure_specs = (
        (
            "execution_v3_confidence_active_fraction",
            "Confidence Threshold Vs Active Trade Fraction Proxy Diagnostic",
            "confidence_threshold_aggregate.csv",
            "threshold",
            "mean_active_trade_fraction",
            "active trade fraction",
        ),
        (
            "execution_v3_confidence_net_proxy",
            "Confidence Threshold Vs Net Cost-Adjusted Proxy Diagnostic",
            "confidence_threshold_aggregate.csv",
            "threshold",
            "mean_net_cost_adjusted_proxy",
            "net cost-adjusted proxy",
        ),
        (
            "execution_v3_cost_sensitivity",
            "Cost Sensitivity Proxy Diagnostic",
            "cost_sensitivity_summary.csv",
            "fee_bps",
            "net_proxy",
            "net cost-adjusted proxy",
        ),
        (
            "execution_v3_latency_sensitivity",
            "Latency Sensitivity Proxy Diagnostic",
            "latency_sensitivity_summary.csv",
            "latency_step",
            "net_proxy",
            "net cost-adjusted proxy",
        ),
    )
    for figure_id, title, filename, x_column, y_column, ylabel in figure_specs:
        _build_execution_v3_line_figure(
            entries=entries,
            execution_v3_dir=execution_v3_dir,
            filename=filename,
            figure_id=figure_id,
            title=title,
            x_column=x_column,
            y_column=y_column,
            ylabel=ylabel,
            out_dir=out_dir,
            source_dir=source_dir,
            metadata_dir=metadata_dir,
            smoke_test=smoke_test,
        )
    _build_execution_v3_bar_figure(
        entries=entries,
        execution_v3_dir=execution_v3_dir,
        filename="fill_assumption_summary.csv",
        figure_id="execution_v3_fill_assumption_comparison",
        title="Fill Assumption Comparison Proxy Diagnostic",
        category_column="fill_mode",
        value_column="net_proxy",
        ylabel="net cost-adjusted proxy",
        out_dir=out_dir,
        source_dir=source_dir,
        metadata_dir=metadata_dir,
        smoke_test=smoke_test,
    )
    _build_execution_v3_bar_figure(
        entries=entries,
        execution_v3_dir=execution_v3_dir,
        filename="adverse_selection_summary.csv",
        figure_id="execution_v3_adverse_selection_by_confidence",
        title="Adverse Selection By Confidence Bucket Proxy Diagnostic",
        category_column="confidence_bucket",
        value_column="adverse_fraction",
        ylabel="adverse-selection fraction",
        out_dir=out_dir,
        source_dir=source_dir,
        metadata_dir=metadata_dir,
        smoke_test=smoke_test,
    )
    _build_execution_v3_bar_figure(
        entries=entries,
        execution_v3_dir=execution_v3_dir,
        filename="regime_execution_summary.csv",
        figure_id="execution_v3_regime_breakdown",
        title="Regime Execution Breakdown Proxy Diagnostic",
        category_column="regime_label",
        value_column="net_proxy",
        ylabel="net cost-adjusted proxy",
        out_dir=out_dir,
        source_dir=source_dir,
        metadata_dir=metadata_dir,
        smoke_test=smoke_test,
    )


def _build_execution_v3_line_figure(
    *,
    entries: list[dict[str, Any]],
    execution_v3_dir: Path | None,
    filename: str,
    figure_id: str,
    title: str,
    x_column: str,
    y_column: str,
    ylabel: str,
    out_dir: Path,
    source_dir: Path,
    metadata_dir: Path,
    smoke_test: bool,
) -> None:
    frame, reason = _load_execution_v3_source(execution_v3_dir, filename)
    if frame is None:
        _skip_entry(
            entries,
            figure_id=figure_id,
            title=title,
            reason=reason,
            smoke_test=smoke_test,
        )
        return
    source = _execution_v3_numeric_source(frame, x_column=x_column, y_column=y_column)
    source_path = source_dir / f"{figure_id}.csv"
    source.to_csv(source_path, index=False)
    ok = source.dropna(subset=[x_column, y_column]) if y_column in source.columns else source
    if ok.empty:
        _skip_entry(
            entries,
            figure_id=figure_id,
            title=title,
            reason=f"{filename} has no plottable execution-v3 rows",
            smoke_test=smoke_test,
            source_path=source_path,
        )
        return
    target = out_dir / f"{figure_id}.png"
    metadata_path = _write_metadata(
        metadata_dir,
        figure_id=figure_id,
        title=title,
        source_path=source_path,
        smoke_test=smoke_test,
        extra={"proxy_label": "offline execution-aware proxy diagnostic"},
    )
    _plot_execution_v3_lines(
        ok,
        target,
        x_column=x_column,
        y_column=y_column,
        ylabel=ylabel,
        title=title,
    )
    _complete_entry(
        entries,
        figure_id=figure_id,
        title=title,
        file_path=target,
        source_path=source_path,
        metadata_path=metadata_path,
        frame=source,
        smoke_test=smoke_test,
    )


def _build_execution_v3_bar_figure(
    *,
    entries: list[dict[str, Any]],
    execution_v3_dir: Path | None,
    filename: str,
    figure_id: str,
    title: str,
    category_column: str,
    value_column: str,
    ylabel: str,
    out_dir: Path,
    source_dir: Path,
    metadata_dir: Path,
    smoke_test: bool,
) -> None:
    frame, reason = _load_execution_v3_source(execution_v3_dir, filename)
    if frame is None:
        _skip_entry(
            entries,
            figure_id=figure_id,
            title=title,
            reason=reason,
            smoke_test=smoke_test,
        )
        return
    source = _execution_v3_numeric_source(
        frame,
        x_column=category_column,
        y_column=value_column,
    )
    if "status" in source.columns:
        source = source[source["status"].astype(str) == "ok"].copy()
    source_path = source_dir / f"{figure_id}.csv"
    source.to_csv(source_path, index=False)
    if source.empty or category_column not in source.columns or value_column not in source.columns:
        _skip_entry(
            entries,
            figure_id=figure_id,
            title=title,
            reason=f"{filename} has no plottable execution-v3 rows",
            smoke_test=smoke_test,
            source_path=source_path,
        )
        return
    target = out_dir / f"{figure_id}.png"
    metadata_path = _write_metadata(
        metadata_dir,
        figure_id=figure_id,
        title=title,
        source_path=source_path,
        smoke_test=smoke_test,
        extra={"proxy_label": "offline execution-aware proxy diagnostic"},
    )
    _plot_execution_v3_bars(
        source,
        target,
        category_column=category_column,
        value_column=value_column,
        ylabel=ylabel,
        title=title,
    )
    _complete_entry(
        entries,
        figure_id=figure_id,
        title=title,
        file_path=target,
        source_path=source_path,
        metadata_path=metadata_path,
        frame=source,
        smoke_test=smoke_test,
    )


def _build_regime_breakdown_figure(
    *,
    entries: list[dict[str, Any]],
    predictions: pd.DataFrame,
    out_dir: Path,
    source_dir: Path,
    metadata_dir: Path,
    smoke_test: bool,
) -> None:
    figure_id = "regime_breakdown"
    title = "Regime Breakdown"
    if predictions.empty or "regime" not in predictions.columns:
        _skip_entry(
            entries,
            figure_id=figure_id,
            title=title,
            reason="regime labels not present in prediction artefacts",
            smoke_test=smoke_test,
        )
        return
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(["model_key", "regime"], sort=True):
        model_key, regime = keys
        rows.append(
            {
                "model_key": str(model_key),
                "regime": str(regime),
                "sample_count": len(group),
                "macro_f1": _macro_f1(
                    group["y_true_class"].tolist(),
                    group["y_pred_class"].tolist(),
                ),
            }
        )
    source = pd.DataFrame(rows)
    source_path = source_dir / f"{figure_id}.csv"
    source.to_csv(source_path, index=False)
    if source.empty:
        _skip_entry(
            entries,
            figure_id=figure_id,
            title=title,
            reason="regime column had no usable rows",
            smoke_test=smoke_test,
            source_path=source_path,
        )
        return
    target = out_dir / f"{figure_id}.png"
    metadata_path = _write_metadata(
        metadata_dir,
        figure_id=figure_id,
        title=title,
        source_path=source_path,
        smoke_test=smoke_test,
    )
    _plot_regime_breakdown(source, target, title=_smoke_title(title, smoke_test))
    _complete_entry(
        entries,
        figure_id=figure_id,
        title=title,
        file_path=target,
        source_path=source_path,
        metadata_path=metadata_path,
        frame=predictions,
        smoke_test=smoke_test,
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], payload)


def _ensure_output_dir(out_dir: Path, *, input_dir: Path, overwrite: bool) -> None:
    resolved_out = out_dir.resolve(strict=False)
    resolved_input = input_dir.resolve(strict=False)
    if resolved_out == resolved_input or resolved_input.is_relative_to(resolved_out):
        raise ValueError("figure output directory must not contain the input grid")
    if out_dir.exists():
        if not out_dir.is_dir():
            raise FileExistsError(f"figure output path exists and is not a directory: {out_dir}")
        if any(out_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    "refusing to overwrite existing figure directory; "
                    f"pass overwrite=True: {out_dir}"
                )
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _resolve_execution_v3_dir(execution_v3_dir: Path | None, grid_dir: Path) -> Path | None:
    candidates: list[Path] = []
    if execution_v3_dir is not None:
        candidates.append(Path(execution_v3_dir))
    candidates.extend(
        [
            grid_dir / "execution_v3",
            project_root() / "experiments" / "fi2010_execution_v3",
        ]
    )
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if (candidate / "execution_v3_manifest.json").is_file() or (
            candidate / "confidence_threshold_summary.csv"
        ).is_file():
            return candidate
    return None


def _load_execution_v3_source(
    execution_v3_dir: Path | None,
    filename: str,
) -> tuple[pd.DataFrame | None, str]:
    if execution_v3_dir is None:
        return None, "execution v3 artefacts not available"
    path = execution_v3_dir / filename
    if not path.is_file():
        return None, f"execution v3 artefact missing: {_display_path(path)}"
    frame = pd.read_csv(path)
    if frame.empty:
        return None, f"execution v3 artefact is empty: {_display_path(path)}"
    return frame, ""


def _execution_v3_numeric_source(
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
) -> pd.DataFrame:
    source = frame.copy()
    if "model_key" not in source.columns:
        if {"model_family", "pretraining_objective"} <= set(source.columns):
            source["model_key"] = source.apply(
                lambda row: _model_key(
                    row.get("model_family"),
                    row.get("pretraining_objective"),
                ),
                axis=1,
            )
        else:
            source["model_key"] = "execution_v3"
    categorical_columns = {"fill_mode", "confidence_bucket", "regime_label"}
    for column in (x_column, y_column, "horizon", "fee_bps", "spread_multiplier"):
        if column in source.columns and column not in categorical_columns:
            source[column] = pd.to_numeric(source[column], errors="coerce")
    if "status" in source.columns:
        ok = source[source["status"].astype(str).isin({"ok", "completed"})].copy()
        if not ok.empty:
            source = ok
    return source


def _is_smoke_summary(summary: Mapping[str, Any]) -> bool:
    return bool(summary.get("smoke_test")) or str(
        summary.get("execution_mode", "")
    ).lower() == "smoke"


def _load_results(grid_dir: Path, *, warnings: list[str]) -> pd.DataFrame:
    path = grid_dir / "results_summary.csv"
    if not path.is_file():
        warnings.append(f"results_summary.csv missing under {grid_dir}")
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        warnings.append("results_summary.csv is empty")
        return frame
    if "status" in frame.columns:
        frame = frame[frame["status"].astype(str).isin(_SUCCESS_STATUSES)].copy()
    return frame.reset_index(drop=True)


def _filter_result_rows(
    results: pd.DataFrame,
    *,
    models: Sequence[str] | None,
    horizons: Sequence[int] | None,
    folds: Sequence[int | str] | None,
    seeds: Sequence[int] | None,
) -> pd.DataFrame:
    if results.empty:
        return results
    frame = _result_frame_with_keys(results)
    mask = pd.Series(True, index=frame.index)
    if models:
        selected = {token.strip().lower() for token in models if token.strip()}
        if selected and "all" not in selected:
            mask &= frame.apply(lambda row: _matches_model_selection(row, selected), axis=1)
    if horizons:
        selected_horizons = {int(item) for item in horizons}
        mask &= frame["horizon"].apply(lambda value: _optional_int(value) in selected_horizons)
    if folds:
        selected_folds = {_fold_number(item) for item in folds}
        mask &= frame["fold"].apply(lambda value: _optional_int(value) in selected_folds)
    if seeds:
        selected_seeds = {int(item) for item in seeds}
        mask &= frame["seed"].apply(lambda value: _optional_int(value) in selected_seeds)
    return frame.loc[mask].reset_index(drop=True)


def _matches_model_selection(row: Mapping[str, Any], selected: set[str]) -> bool:
    values = {
        str(row.get("model_family", "")).lower(),
        str(row.get("pretraining_objective", "")).lower(),
        str(row.get("objective_label", "")).lower(),
        str(row.get("model_key", "")).lower(),
    }
    return bool(values & selected)


def _result_frame_with_keys(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    if "fold" not in frame.columns and "fold_id" in frame.columns:
        frame["fold"] = frame["fold_id"].apply(_fold_number)
    if "fold" not in frame.columns:
        frame["fold"] = None
    for column in ("horizon", "seed", "lookback"):
        if column not in frame.columns:
            frame[column] = None
    if "model_family" not in frame.columns:
        if "model_name" in frame.columns:
            frame["model_family"] = frame["model_name"].astype(str)
        else:
            frame["model_family"] = "unknown"
    if "pretraining_objective" not in frame.columns:
        frame["pretraining_objective"] = "none"
    frame["objective_label"] = frame["pretraining_objective"].apply(_objective_label)
    frame["model_key"] = frame.apply(
        lambda row: _model_key(row["model_family"], row["pretraining_objective"]),
        axis=1,
    )
    return frame


def _load_prediction_frame(
    results: pd.DataFrame,
    grid_dir: Path,
    *,
    warnings: list[str],
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for _, row in results.iterrows():
        path = _prediction_path(row, grid_dir)
        if path is None or not path.is_file():
            warnings.append(
                "prediction file missing for "
                f"{_model_key(row.get('model_family'), row.get('pretraining_objective'))} "
                f"horizon={row.get('horizon')} fold={row.get('fold')} seed={row.get('seed')}"
            )
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            warnings.append(f"prediction file is empty: {path}")
            continue
        for column in (
            "fold",
            "horizon",
            "seed",
            "lookback",
            "model_family",
            "pretraining_objective",
        ):
            if column not in frame.columns:
                frame[column] = row.get(column)
        if "model_key" not in frame.columns:
            frame["model_key"] = _model_key(
                row.get("model_family"),
                row.get("pretraining_objective"),
            )
        frame["_source_prediction_file"] = _display_path(path)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def _prediction_path(row: Mapping[str, Any], grid_dir: Path) -> Path | None:
    raw = row.get("prediction_file") or row.get("prediction_path")
    if raw is not None and str(raw).strip():
        path = Path(str(raw))
        if path.is_absolute():
            return path
        return grid_dir / path
    run_dir = row.get("run_dir")
    if run_dir is not None and str(run_dir).strip():
        return grid_dir / str(run_dir) / "predictions.csv"
    return None


def _build_label_mapping_audit(
    predictions: pd.DataFrame,
    results: pd.DataFrame,
    *,
    warnings: list[str],
) -> dict[str, Any]:
    audit_warnings = list(warnings)
    errors: list[str] = []
    detected_labels = _detected_prediction_labels(predictions)
    probability_columns = list(predictions.columns) if not predictions.empty else []
    probability_validation = validate_probability_columns(probability_columns)
    class_order_validation = validate_class_order(FI2010_CANONICAL_CLASS_ORDER)
    confusion_validation = validate_confusion_matrix_axis_labels(
        FI2010_CANONICAL_CLASS_ORDER
    )
    f1_validation = validate_classwise_f1_columns(tuple(str(col) for col in results.columns))
    for validation in (
        probability_validation,
        class_order_validation,
        confusion_validation,
        f1_validation,
    ):
        errors.extend(validation.errors)
        audit_warnings.extend(validation.warnings)
    if predictions.empty:
        errors.append("no prediction rows available for label-mapping audit")
    for label in detected_labels:
        try:
            canonical_class_name(label)
        except ValueError as exc:
            errors.append(str(exc))
    details = probability_validation.details or {}
    return {
        "detected_labels": detected_labels,
        "expected_labels": {str(key): value for key, value in FI2010_RAW_LABEL_TO_CLASS.items()},
        "probability_columns_found": details.get("probability_columns_found", []),
        "probability_column_order_used": list(
            details.get("probability_column_order_used", ())
        ),
        "class_order_used_for_metrics": list(FI2010_CANONICAL_CLASS_ORDER),
        "confusion_matrix_axis_labels": list(FI2010_CANONICAL_CLASS_ORDER),
        "classwise_f1_columns_expected": [
            f"class_f1_{label}" for label in FI2010_CANONICAL_CLASS_ORDER
        ],
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": audit_warnings,
    }


def _detected_prediction_labels(predictions: pd.DataFrame) -> list[Any]:
    if predictions.empty:
        return []
    labels: list[Any] = []
    for column in ("y_true", "y_pred", "label", "prediction"):
        if column in predictions.columns:
            labels.extend(
                value for value in predictions[column].dropna().tolist() if str(value) != ""
            )
    return sorted({_json_scalar(value) for value in labels}, key=lambda item: str(item))


def _canonicalise_predictions(
    predictions: pd.DataFrame,
    *,
    warnings: list[str],
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    try:
        probability_columns = probability_columns_for_order(
            tuple(str(col) for col in predictions.columns)
        )
    except ValueError as exc:
        warnings.append(f"prediction probabilities unavailable: {exc}")
        return pd.DataFrame()
    frame = predictions.copy()
    y_true_column = "y_true" if "y_true" in frame.columns else "label"
    y_pred_column = "y_pred" if "y_pred" in frame.columns else "prediction"
    if y_true_column not in frame.columns or y_pred_column not in frame.columns:
        warnings.append("prediction rows do not contain y_true/y_pred or label/prediction")
        return pd.DataFrame()
    try:
        frame["y_true_class"] = frame[y_true_column].apply(canonical_class_name)
        frame["y_pred_class"] = frame[y_pred_column].apply(canonical_class_name)
    except ValueError as exc:
        warnings.append(f"prediction labels unavailable: {exc}")
        return pd.DataFrame()
    for class_name, column in zip(
        FI2010_CANONICAL_CLASS_ORDER,
        probability_columns,
        strict=True,
    ):
        canonical_column = f"prob_{class_name}"
        frame[canonical_column] = pd.to_numeric(frame[column], errors="coerce")
    if "confidence" in frame.columns:
        frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    else:
        frame["confidence"] = frame[
            [f"prob_{label}" for label in FI2010_CANONICAL_CLASS_ORDER]
        ].max(axis=1)
    frame["confidence"] = frame["confidence"].fillna(
        frame[[f"prob_{label}" for label in FI2010_CANONICAL_CLASS_ORDER]].max(axis=1)
    )
    if "model_key" not in frame.columns:
        frame["model_key"] = frame.apply(
            lambda row: _model_key(row.get("model_family"), row.get("pretraining_objective")),
            axis=1,
        )
    return frame.dropna(subset=["confidence", "y_true_class", "y_pred_class"]).reset_index(
        drop=True
    )


def _confusion_rows(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    labels = list(FI2010_CANONICAL_CLASS_ORDER)
    rows: list[dict[str, Any]] = []
    for actual in labels:
        actual_rows = predictions[predictions["y_true_class"] == actual]
        row_total = len(actual_rows)
        for predicted in labels:
            count = int((actual_rows["y_pred_class"] == predicted).sum())
            rows.append(
                {
                    "true_label": actual,
                    "predicted_label": predicted,
                    "count": count,
                    "row_normalised_percent": (
                        None if row_total == 0 else 100.0 * count / row_total
                    ),
                }
            )
    return rows


def _reliability_rows(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_key, group in predictions.groupby("model_key", sort=True):
        total = len(group)
        if total == 0:
            continue
        for bin_index in range(10):
            lower = bin_index / 10
            upper = (bin_index + 1) / 10
            if bin_index == 0:
                selected = group[(group["confidence"] >= lower) & (group["confidence"] <= upper)]
            else:
                selected = group[(group["confidence"] > lower) & (group["confidence"] <= upper)]
            if selected.empty:
                continue
            correct = selected["y_true_class"] == selected["y_pred_class"]
            rows.append(
                {
                    "model_key": str(model_key),
                    "bin_index": bin_index,
                    "confidence_lower": lower,
                    "confidence_upper": upper,
                    "count": len(selected),
                    "fraction": len(selected) / total,
                    "mean_confidence": float(selected["confidence"].mean()),
                    "empirical_accuracy": float(correct.mean()),
                }
            )
    return rows


def _horizon_summary(results: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    frame = _result_frame_with_keys(results)
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    rows: list[dict[str, Any]] = []
    for keys, group in frame.dropna(subset=[metric]).groupby(
        ["horizon", "model_key"],
        sort=True,
    ):
        horizon, model_key = keys
        values = pd.to_numeric(group[metric], errors="coerce").dropna()
        rows.append(
            {
                "horizon": int(float(horizon)),
                "model_key": str(model_key),
                f"mean_{metric}": float(values.mean()),
                f"std_{metric}": float(values.std(ddof=0)) if len(values) > 1 else 0.0,
                "run_count": len(values),
                "folds": ",".join(str(item) for item in _sorted_unique_ints(group["fold"])),
                "seeds": ",".join(str(item) for item in _sorted_unique_ints(group["seed"])),
            }
        )
    return pd.DataFrame(rows)


def _threshold_fraction_rows(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_key, group in predictions.groupby("model_key", sort=True):
        total = len(group)
        for threshold in _CONFIDENCE_THRESHOLDS:
            retained = group[group["confidence"] >= threshold]
            rows.append(
                {
                    "model_key": str(model_key),
                    "threshold": threshold,
                    "eligible_count": len(retained),
                    "total_count": total,
                    "eligible_fraction": 0.0 if total == 0 else len(retained) / total,
                }
            )
    return rows


def _threshold_macro_f1_rows(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    min_retained = len(FI2010_CANONICAL_CLASS_ORDER)
    for model_key, group in predictions.groupby("model_key", sort=True):
        total = len(group)
        for threshold in _CONFIDENCE_THRESHOLDS:
            retained = group[group["confidence"] >= threshold]
            if len(retained) < min_retained:
                rows.append(
                    {
                        "model_key": str(model_key),
                        "threshold": threshold,
                        "retained_count": len(retained),
                        "total_count": total,
                        "retained_fraction": 0.0
                        if total == 0
                        else len(retained) / total,
                        "macro_f1": None,
                        "status": "skipped",
                        "reason": "too few samples remain",
                    }
                )
                continue
            rows.append(
                {
                    "model_key": str(model_key),
                    "threshold": threshold,
                    "retained_count": len(retained),
                    "total_count": total,
                    "retained_fraction": 0.0 if total == 0 else len(retained) / total,
                    "macro_f1": _macro_f1(
                        retained["y_true_class"].tolist(),
                        retained["y_pred_class"].tolist(),
                    ),
                    "status": "ok",
                    "reason": "",
                }
            )
    return rows


def _macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("macro-F1 inputs must have matching lengths")
    if not y_true:
        return 0.0
    scores = [
        _binary_f1(y_true, y_pred, positive_label=label)
        for label in FI2010_CANONICAL_CLASS_ORDER
    ]
    return float(sum(scores) / len(scores))


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


def _import_matplotlib() -> Any:
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("matplotlib is required for FI-2010 figure generation") from exc
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _plot_confusion(rows: Sequence[Mapping[str, Any]], target: Path, *, title: str) -> None:
    plt = _import_matplotlib()
    labels = list(FI2010_CANONICAL_CLASS_ORDER)
    matrix = [
        [
            _row_lookup(rows, actual, predicted, "count") or 0
            for predicted in labels
        ]
        for actual in labels
    ]
    percentages = [
        [
            _row_lookup(rows, actual, predicted, "row_normalised_percent")
            for predicted in labels
        ]
        for actual in labels
    ]
    figure, ax = plt.subplots(figsize=(5.8, 5.2), dpi=_PLOT_DPI)
    image = ax.imshow(matrix, cmap="Blues", aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title)
    for row_index, row in enumerate(matrix):
        for column_index, count in enumerate(row):
            percent = percentages[row_index][column_index]
            label = f"{int(count)}"
            if percent is not None:
                label += f"\n{float(percent):.1f}%"
            ax.text(column_index, row_index, label, ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=ax, shrink=0.78)
    figure.tight_layout()
    figure.savefig(target, dpi=_PLOT_DPI)
    plt.close(figure)


def _plot_reliability(source: pd.DataFrame, target: Path, *, title: str) -> None:
    plt = _import_matplotlib()
    figure, ax = plt.subplots(figsize=_PLOT_FIGURE_SIZE, dpi=_PLOT_DPI)
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", linewidth=1.0)
    for model_key, group in source.groupby("model_key", sort=True):
        ordered = group.sort_values("mean_confidence")
        ax.plot(
            ordered["mean_confidence"],
            ordered["empirical_accuracy"],
            marker="o",
            linewidth=1.4,
            markersize=4,
            label=str(model_key),
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="best", fontsize="x-small", frameon=False)
    figure.tight_layout()
    figure.savefig(target, dpi=_PLOT_DPI)
    plt.close(figure)


def _plot_fold_metric(source: pd.DataFrame, target: Path, *, title: str) -> None:
    plt = _import_matplotlib()
    figure, ax = plt.subplots(figsize=_PLOT_FIGURE_SIZE, dpi=_PLOT_DPI)
    grouped = (
        source.groupby(["model_key", "fold"], sort=True)["macro_f1"]
        .mean()
        .reset_index()
    )
    for model_key, group in grouped.groupby("model_key", sort=True):
        ordered = group.sort_values("fold")
        ax.plot(
            ordered["fold"],
            ordered["macro_f1"],
            marker="o",
            linewidth=1.4,
            label=str(model_key),
        )
    ax.set_xlabel("fold")
    ax.set_ylabel("macro-F1")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="best", fontsize="x-small", frameon=False)
    figure.tight_layout()
    figure.savefig(target, dpi=_PLOT_DPI)
    plt.close(figure)


def _plot_horizon_metric(
    source: pd.DataFrame,
    target: Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    plt = _import_matplotlib()
    figure, ax = plt.subplots(figsize=_PLOT_FIGURE_SIZE, dpi=_PLOT_DPI)
    mean_column = next(column for column in source.columns if column.startswith("mean_"))
    std_column = next(column for column in source.columns if column.startswith("std_"))
    for model_key, group in source.groupby("model_key", sort=True):
        ordered = group.sort_values("horizon")
        ax.errorbar(
            ordered["horizon"],
            ordered[mean_column],
            yerr=ordered[std_column],
            marker="o",
            linewidth=1.4,
            capsize=3,
            label=str(model_key),
        )
    ax.set_xlabel("horizon")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="best", fontsize="x-small", frameon=False)
    figure.tight_layout()
    figure.savefig(target, dpi=_PLOT_DPI)
    plt.close(figure)


def _plot_ssl_deltas(source: pd.DataFrame, target: Path, *, title: str) -> None:
    plt = _import_matplotlib()
    plot_rows: list[dict[str, Any]] = []
    for _, row in source.iterrows():
        label = f"h{row.get('horizon')} {row.get('ssl_objective')}"
        for metric in ("delta_macro_f1", "delta_mcc", "delta_ece"):
            value = _finite_float(row.get(metric))
            if value is None:
                continue
            plot_rows.append({"comparison": label, "metric": metric, "value": value})
    plot_frame = pd.DataFrame(plot_rows)
    figure, ax = plt.subplots(figsize=(8.2, 4.8), dpi=_PLOT_DPI)
    if not plot_frame.empty:
        pivot = plot_frame.pivot_table(index="comparison", columns="metric", values="value")
        pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("matched comparison")
    ax.set_ylabel("SSL minus supervised")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="best", fontsize="x-small", frameon=False)
    figure.tight_layout()
    figure.savefig(target, dpi=_PLOT_DPI)
    plt.close(figure)


def _plot_threshold_metric(
    source: pd.DataFrame,
    target: Path,
    *,
    value_column: str,
    ylabel: str,
    title: str,
) -> None:
    plt = _import_matplotlib()
    figure, ax = plt.subplots(figsize=_PLOT_FIGURE_SIZE, dpi=_PLOT_DPI)
    for model_key, group in source.groupby("model_key", sort=True):
        ordered = group.sort_values("threshold")
        ax.plot(
            ordered["threshold"],
            ordered[value_column],
            marker="o",
            linewidth=1.4,
            label=str(model_key),
        )
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="best", fontsize="x-small", frameon=False)
    figure.tight_layout()
    figure.savefig(target, dpi=_PLOT_DPI)
    plt.close(figure)


def _plot_cost_proxy(
    source: pd.DataFrame,
    target: Path,
    *,
    value_column: str,
    title: str,
) -> None:
    plt = _import_matplotlib()
    figure, ax = plt.subplots(figsize=_PLOT_FIGURE_SIZE, dpi=_PLOT_DPI)
    x_column = "cost_bps" if "cost_bps" in source.columns else "horizon"
    group_column = "model_key" if "model_key" in source.columns else "model_family"
    if group_column not in source.columns:
        source = source.copy()
        source[group_column] = "proxy"
    for label, group in source.groupby(group_column, sort=True):
        ordered = group.sort_values(x_column) if x_column in group.columns else group
        x_values = ordered[x_column] if x_column in ordered.columns else range(len(ordered))
        ax.plot(x_values, ordered[value_column], marker="o", linewidth=1.4, label=str(label))
    ax.set_xlabel(x_column)
    ax.set_ylabel("proxy diagnostic value")
    ax.set_title(f"{title} (proxy diagnostics)")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="best", fontsize="x-small", frameon=False)
    figure.tight_layout()
    figure.savefig(target, dpi=_PLOT_DPI)
    plt.close(figure)


def _plot_regime_breakdown(source: pd.DataFrame, target: Path, *, title: str) -> None:
    plt = _import_matplotlib()
    figure, ax = plt.subplots(figsize=_PLOT_FIGURE_SIZE, dpi=_PLOT_DPI)
    labels = [f"{row.model_key}\n{row.regime}" for row in source.itertuples()]
    ax.bar(labels, source["macro_f1"], color="#4C78A8", edgecolor="black", linewidth=0.4)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("model / regime")
    ax.set_ylabel("macro-F1")
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    figure.tight_layout()
    figure.savefig(target, dpi=_PLOT_DPI)
    plt.close(figure)


def _plot_execution_v3_lines(
    source: pd.DataFrame,
    target: Path,
    *,
    x_column: str,
    y_column: str,
    ylabel: str,
    title: str,
) -> None:
    plt = _import_matplotlib()
    figure, ax = plt.subplots(figsize=_PLOT_FIGURE_SIZE, dpi=_PLOT_DPI)
    plot_source = source.copy()
    series_column = "model_key"
    if "spread_multiplier" in plot_source.columns and x_column == "fee_bps":
        plot_source["series"] = plot_source.apply(
            lambda row: f"{row.get('model_key')} spread x{row.get('spread_multiplier')}",
            axis=1,
        )
        series_column = "series"
    for label, group in plot_source.groupby(series_column, sort=True):
        ordered = group.sort_values(x_column)
        ax.plot(
            ordered[x_column],
            ordered[y_column],
            marker="o",
            linewidth=1.4,
            markersize=4,
            label=str(label),
        )
    ax.set_xlabel(x_column.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="best", fontsize="x-small", frameon=False)
    figure.tight_layout()
    figure.savefig(target, dpi=_PLOT_DPI)
    plt.close(figure)


def _plot_execution_v3_bars(
    source: pd.DataFrame,
    target: Path,
    *,
    category_column: str,
    value_column: str,
    ylabel: str,
    title: str,
) -> None:
    plt = _import_matplotlib()
    frame = source.copy()
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    grouped = (
        frame.dropna(subset=[value_column])
        .groupby(category_column, sort=True)[value_column]
        .mean()
        .reset_index()
    )
    figure, ax = plt.subplots(figsize=_PLOT_FIGURE_SIZE, dpi=_PLOT_DPI)
    ax.bar(
        grouped[category_column].astype(str),
        grouped[value_column],
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlabel(category_column.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    figure.autofmt_xdate(rotation=30, ha="right")
    figure.tight_layout()
    figure.savefig(target, dpi=_PLOT_DPI)
    plt.close(figure)


def _write_metadata(
    metadata_dir: Path,
    *,
    figure_id: str,
    title: str,
    source_path: Path,
    smoke_test: bool,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "figure_id": figure_id,
        "title": title,
        "source_data_path": _display_path(source_path),
        "smoke_test": bool(smoke_test),
        "class_order": list(FI2010_CANONICAL_CLASS_ORDER),
        "created_at": datetime.now(UTC).isoformat(),
    }
    if extra:
        payload.update(dict(extra))
    path = metadata_dir / f"{figure_id}.json"
    path.write_text(stable_json_dumps(payload), encoding="utf-8")
    return path


def _complete_entry(
    entries: list[dict[str, Any]],
    *,
    figure_id: str,
    title: str,
    file_path: Path,
    source_path: Path,
    metadata_path: Path,
    frame: pd.DataFrame,
    smoke_test: bool,
) -> None:
    entries.append(
        {
            "figure_id": figure_id,
            "title": title,
            "file_path": _display_path(file_path),
            "source_data_path": _display_path(source_path),
            "metadata_path": _display_path(metadata_path),
            "models_included": _string_values(frame, "model_key"),
            "horizons_included": _int_values(frame, "horizon"),
            "folds_included": _int_values(frame, "fold"),
            "seeds_included": _int_values(frame, "seed"),
            "smoke_test": bool(smoke_test),
            "status": "completed",
            "reason": "",
        }
    )


def _skip_entry(
    entries: list[dict[str, Any]],
    *,
    figure_id: str,
    title: str,
    reason: str,
    smoke_test: bool,
    source_path: Path | None = None,
) -> None:
    entries.append(
        {
            "figure_id": figure_id,
            "title": title,
            "file_path": None,
            "source_data_path": None if source_path is None else _display_path(source_path),
            "metadata_path": None,
            "models_included": [],
            "horizons_included": [],
            "folds_included": [],
            "seeds_included": [],
            "smoke_test": bool(smoke_test),
            "status": "skipped",
            "reason": reason,
        }
    )


def _row_lookup(
    rows: Sequence[Mapping[str, Any]],
    actual: str,
    predicted: str,
    field: str,
) -> Any:
    for row in rows:
        if row.get("true_label") == actual and row.get("predicted_label") == predicted:
            return row.get(field)
    return None


def _model_key(model_family: Any, pretraining_objective: Any) -> str:
    family = str(model_family or "unknown").strip()
    return f"{family}/{_objective_label(pretraining_objective)}"


def _objective_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"", "none", "nan", "supervised", "random_init"}:
        return "supervised"
    if text == "masked_field":
        return "masked_reconstruction"
    return text


def _fold_number(value: Any) -> int:
    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith("fold_"):
            text = text.removeprefix("fold_")
        return int(float(text))
    return int(float(value))


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
    try:
        return int(float(text))
    except ValueError:
        return None


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _sorted_unique_ints(values: Any) -> list[int]:
    if values is None:
        return []
    items = values.tolist() if hasattr(values, "tolist") else list(values)
    cleaned = {_optional_int(value) for value in items}
    return sorted(value for value in cleaned if value is not None)


def _int_values(frame: pd.DataFrame, column: str) -> list[int]:
    if column not in frame.columns:
        return []
    return _sorted_unique_ints(frame[column])


def _string_values(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        return []
    return sorted({str(value) for value in frame[column].dropna().tolist() if str(value)})


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    # List/tuple cells (e.g. contributing folds/seeds) are serialised
    # element-wise; ``pd.isna`` on an array-like raises in a boolean context.
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    item = value.item() if hasattr(value, "item") else value
    if isinstance(item, float) and item.is_integer():
        return int(item)
    return item


def _json_ready(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_scalar(value) for key, value in row.items()}


def _display_path(path: Path) -> str:
    candidate = Path(path)
    try:
        resolved = candidate.resolve(strict=False)
        root = project_root().resolve(strict=False)
        return resolved.relative_to(root).as_posix()
    except (OSError, ValueError):
        return candidate.as_posix()


def _smoke_title(title: str, smoke_test: bool) -> str:
    return f"{title} (smoke-test diagnostics)" if smoke_test else title
