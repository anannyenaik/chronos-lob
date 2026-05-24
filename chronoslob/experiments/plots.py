"""Deterministic plot generation for paper experiment artefact directories.

This module turns stored experiment artefacts (calibration bins, execution
sensitivity rows, confusion matrices and predictions) into reproducible
PNG plots inside an experiment directory.

The plot builder is deliberately scoped:

* It reads only artefacts that were already written by the paper runner.
* It never refits a model, never invents missing data and never fabricates
  metrics or labels.
* When an optional plot input is missing or invalid it records a clear
  warning and skips that plot rather than writing a placeholder image.
* Regime breakdown plots are only generated when genuine regime data is
  present in the stored artefacts; otherwise the plot is skipped.

Plots are placed under ``<experiment_dir>/plots/`` with stable filenames
so the experiment artefact contract can recognise them as optional
artefacts.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from chronoslob.experiments.manifests import stable_json_dumps

__all__ = [
    "PAPER_PLOT_BUILDER_VERSION",
    "PAPER_PLOT_FILENAMES",
    "PLOT_SUMMARY_FILENAME",
    "PaperPlotSummary",
    "build_paper_experiment_plots",
]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)

PAPER_PLOT_BUILDER_VERSION = "phase-g/paper-plot-builder/v1"
PLOT_SUMMARY_FILENAME = "plot_summary.json"

PAPER_PLOT_FILENAMES: tuple[str, ...] = (
    "reliability_curve.png",
    "cost_sensitivity.png",
    "confusion_matrix.png",
    "regime_breakdown.png",
)

_PLOT_DPI = 120
_PLOT_FIGURE_SIZE = (7.0, 4.5)


class PaperPlotSummary(BaseModel):
    """Summary of plot generation results written as ``plot_summary.json``."""

    model_config = _MODEL_CONFIG

    experiment_dir: str
    created_at: datetime
    builder_version: str
    plots_written: list[str] = Field(default_factory=list)
    plots_skipped: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("experiment_dir", "builder_version")
    @classmethod
    def _validate_required_strings(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("plot summary string fields must be non-empty")
        return value.strip()


# ---------------------------------------------------------------------------
# Matplotlib helpers (lazy import so tests can detect missing dependency)
# ---------------------------------------------------------------------------


def _import_matplotlib() -> Any:
    """Lazily import matplotlib with a headless backend.

    The plot builder is the only consumer of matplotlib in the project,
    so importing it lazily keeps the rest of the package free of the
    dependency at import time.
    """
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "matplotlib is required for paper experiment plot generation. "
            "Install it with `python -m pip install matplotlib` or use the "
            "`[plots]` extra defined in pyproject.toml."
        ) from exc
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _read_csv_if_present(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    if frame.empty:
        return None
    return frame


def _read_json_if_present(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _ensure_plots_dir(experiment_dir: Path) -> Path:
    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def _finite_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _finite_rows(
    frame: pd.DataFrame,
    *,
    numeric_columns: Sequence[str],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    coerced = frame.copy()
    for column in numeric_columns:
        if column not in coerced.columns:
            return coerced.iloc[0:0]
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
    mask = pd.Series(True, index=coerced.index)
    for column in numeric_columns:
        mask &= coerced[column].apply(
            lambda value: isinstance(value, (int, float))
            and math.isfinite(float(value))
        )
    return coerced.loc[mask].reset_index(drop=True)


def _refuse_overwrite(target: Path, overwrite: bool) -> str | None:
    if target.exists() and not overwrite:
        return (
            f"refusing to overwrite existing plot {target.name}; "
            "pass overwrite=True to replace it"
        )
    return None


# ---------------------------------------------------------------------------
# Individual plot builders
# ---------------------------------------------------------------------------


def _build_reliability_curve(
    experiment_dir: Path,
    plots_dir: Path,
    *,
    overwrite: bool,
) -> tuple[bool, str | None]:
    source_path = experiment_dir / "calibration_bins.csv"
    frame = _read_csv_if_present(source_path)
    if frame is None:
        return False, (
            "reliability curve skipped: calibration_bins.csv is missing or empty"
        )
    required = {"model_name", "mean_confidence", "accuracy", "count"}
    if not required.issubset(frame.columns):
        return False, (
            "reliability curve skipped: calibration_bins.csv does not have "
            "the required columns "
            + ", ".join(sorted(required))
        )

    finite = _finite_rows(
        frame,
        numeric_columns=("mean_confidence", "accuracy", "count"),
    )
    finite = finite.loc[finite["count"] > 0]
    if finite.empty:
        return False, (
            "reliability curve skipped: calibration_bins.csv has no rows "
            "with a positive count and finite mean confidence/accuracy"
        )

    target = plots_dir / "reliability_curve.png"
    refusal = _refuse_overwrite(target, overwrite)
    if refusal is not None:
        return False, refusal

    plt = _import_matplotlib()
    figure, ax = plt.subplots(figsize=_PLOT_FIGURE_SIZE, dpi=_PLOT_DPI)
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        color="black",
        linewidth=1.0,
        label="perfect calibration",
    )
    grouped = finite.groupby("model_name", sort=True)
    for model_name, model_rows in grouped:
        model_rows_sorted = model_rows.sort_values("mean_confidence")
        ax.plot(
            model_rows_sorted["mean_confidence"].to_numpy(),
            model_rows_sorted["accuracy"].to_numpy(),
            marker="o",
            linewidth=1.2,
            markersize=4.0,
            label=str(model_name),
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("Reliability curve")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="lower right", fontsize="small", frameon=False)
    figure.tight_layout()
    figure.savefig(target, dpi=_PLOT_DPI)
    plt.close(figure)
    return True, None


def _build_cost_sensitivity(
    experiment_dir: Path,
    plots_dir: Path,
    *,
    overwrite: bool,
) -> tuple[bool, str | None]:
    source_path = experiment_dir / "execution_sensitivity.csv"
    frame = _read_csv_if_present(source_path)
    if frame is None:
        return False, (
            "cost sensitivity skipped: execution_sensitivity.csv is missing or empty"
        )
    required = {
        "model_name",
        "confidence_threshold",
        "cost_bps",
        "net_signal_return_proxy",
    }
    if not required.issubset(frame.columns):
        return False, (
            "cost sensitivity skipped: execution_sensitivity.csv does not have "
            "the required columns "
            + ", ".join(sorted(required))
        )

    finite = _finite_rows(
        frame,
        numeric_columns=(
            "confidence_threshold",
            "cost_bps",
            "net_signal_return_proxy",
        ),
    )
    if finite.empty:
        return False, (
            "cost sensitivity skipped: execution_sensitivity.csv has no rows "
            "with finite cost_bps and net_signal_return_proxy values"
        )

    target = plots_dir / "cost_sensitivity.png"
    refusal = _refuse_overwrite(target, overwrite)
    if refusal is not None:
        return False, refusal

    plt = _import_matplotlib()
    figure, ax = plt.subplots(figsize=_PLOT_FIGURE_SIZE, dpi=_PLOT_DPI)
    model_groups = list(finite.groupby("model_name", sort=True))
    for model_name, model_rows in model_groups:
        threshold_groups = sorted(
            model_rows.groupby("confidence_threshold", sort=True),
            key=lambda item: float(item[0]),
        )
        for threshold, threshold_rows in threshold_groups:
            ordered = threshold_rows.sort_values("cost_bps")
            label = (
                f"{model_name} (threshold={float(threshold):.2f})"
                if len(threshold_groups) > 1
                else str(model_name)
            )
            ax.plot(
                ordered["cost_bps"].to_numpy(),
                ordered["net_signal_return_proxy"].to_numpy(),
                marker="o",
                linewidth=1.2,
                markersize=4.0,
                label=label,
            )
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("cost (bps)")
    ax.set_ylabel("net signal return proxy")
    ax.set_title("Cost-aware signal sensitivity (proxy, not strategy performance)")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    if model_groups:
        ax.legend(loc="best", fontsize="x-small", frameon=False)
    figure.tight_layout()
    figure.savefig(target, dpi=_PLOT_DPI)
    plt.close(figure)
    return True, None


def _confusion_models_from_payload(payload: Any) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    models = payload.get("models")
    if not isinstance(models, list):
        return []
    cleaned: list[Mapping[str, Any]] = []
    for entry in models:
        if not isinstance(entry, Mapping):
            continue
        labels = entry.get("labels")
        matrix = entry.get("matrix")
        if not isinstance(labels, list) or not isinstance(matrix, list):
            continue
        if len(matrix) != len(labels):
            continue
        valid = True
        for row in matrix:
            if not isinstance(row, list) or len(row) != len(labels):
                valid = False
                break
        if not valid:
            continue
        cleaned.append(entry)
    return cleaned


def _build_confusion_matrix(
    experiment_dir: Path,
    plots_dir: Path,
    *,
    overwrite: bool,
) -> tuple[bool, str | None]:
    source_path = experiment_dir / "confusion_matrix.json"
    payload = _read_json_if_present(source_path)
    if payload is None:
        return False, (
            "confusion matrix skipped: confusion_matrix.json is missing or invalid"
        )
    models = _confusion_models_from_payload(payload)
    if not models:
        return False, (
            "confusion matrix skipped: confusion_matrix.json contained no "
            "models with usable labels and matrix entries"
        )

    target = plots_dir / "confusion_matrix.png"
    refusal = _refuse_overwrite(target, overwrite)
    if refusal is not None:
        return False, refusal

    plt = _import_matplotlib()
    n_models = len(models)
    columns = min(2, n_models)
    rows = (n_models + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(_PLOT_FIGURE_SIZE[0] * columns, _PLOT_FIGURE_SIZE[1] * rows),
        dpi=_PLOT_DPI,
        squeeze=False,
    )
    flat_axes = axes.flatten().tolist()
    for index, entry in enumerate(models):
        ax = flat_axes[index]
        labels = [str(label) for label in entry["labels"]]
        matrix_rows = entry["matrix"]
        try:
            matrix_values = [
                [float(value) for value in row] for row in matrix_rows
            ]
        except (TypeError, ValueError):
            ax.set_axis_off()
            continue
        image = ax.imshow(matrix_values, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize="small")
        ax.set_yticklabels(labels, fontsize="small")
        ax.set_xlabel("predicted label")
        ax.set_ylabel("true label")
        title_parts = [str(entry.get("model_name", f"model_{index}"))]
        split_label = entry.get("split")
        if isinstance(split_label, str) and split_label:
            title_parts.append(f"({split_label})")
        ax.set_title(" ".join(title_parts), fontsize="small")
        for row_index, row in enumerate(matrix_values):
            for column_index, value in enumerate(row):
                ax.text(
                    column_index,
                    row_index,
                    f"{int(value)}" if float(value).is_integer() else f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize="xx-small",
                    color="black",
                )
        figure.colorbar(image, ax=ax, shrink=0.7)
    for index in range(n_models, len(flat_axes)):
        flat_axes[index].set_axis_off()
    figure.suptitle("Confusion matrix (from stored artefacts)")
    figure.tight_layout()
    figure.savefig(target, dpi=_PLOT_DPI)
    plt.close(figure)
    return True, None


def _has_regime_breakdown(experiment_dir: Path) -> tuple[bool, str | None]:
    """Detect whether genuine regime-breakdown data exists on disk.

    We treat the breakdown as available only when an explicit regime
    field is present in stored artefacts. We never infer a regime from
    a row number, a timestamp slice or a fabricated bucket.
    """
    predictions_path = experiment_dir / "predictions.csv"
    if predictions_path.is_file():
        try:
            frame = pd.read_csv(predictions_path, nrows=5)
        except (OSError, ValueError, pd.errors.ParserError):
            frame = None
        if frame is not None and "regime" in frame.columns:
            return True, None

    for candidate_name in ("regime_breakdown.json", "regime_summary.json"):
        candidate_path = experiment_dir / candidate_name
        if candidate_path.is_file():
            payload = _read_json_if_present(candidate_path)
            if isinstance(payload, Mapping) and payload:
                return True, None

    for results_name in ("results.json", "runner_summary.json"):
        path = experiment_dir / results_name
        payload = _read_json_if_present(path)
        if not isinstance(payload, Mapping):
            continue
        evidence = payload.get("evidence_streams")
        if not isinstance(evidence, Mapping):
            continue
        robustness = evidence.get("robustness")
        mentions_regime = isinstance(robustness, list) and any(
            isinstance(item, str) and "regime" in item.lower()
            for item in robustness
        )
        if mentions_regime and (experiment_dir / "regime_breakdown.csv").is_file():
            return True, None
    return False, None


def _build_regime_breakdown(
    experiment_dir: Path,
    plots_dir: Path,
    *,
    overwrite: bool,
) -> tuple[bool, str | None]:
    has_data, _ = _has_regime_breakdown(experiment_dir)
    if not has_data:
        return False, (
            "regime breakdown skipped: no genuine regime-breakdown data is "
            "available in stored artefacts; not fabricating regimes from row "
            "numbers or timestamps"
        )

    predictions_path = experiment_dir / "predictions.csv"
    frame = _read_csv_if_present(predictions_path)
    if frame is None or "regime" not in frame.columns or "label" not in frame.columns:
        return False, (
            "regime breakdown skipped: predictions.csv does not have both a "
            "'regime' and a 'label' column to build the breakdown from"
        )

    if "prediction" not in frame.columns:
        return False, (
            "regime breakdown skipped: predictions.csv has no 'prediction' "
            "column to compare against the stored labels"
        )

    target = plots_dir / "regime_breakdown.png"
    refusal = _refuse_overwrite(target, overwrite)
    if refusal is not None:
        return False, refusal

    frame["regime"] = frame["regime"].astype(str)
    accuracy_per_regime = (
        frame.assign(correct=(frame["prediction"] == frame["label"]).astype(float))
        .groupby("regime", sort=True)["correct"]
        .mean()
        .sort_index()
    )
    if accuracy_per_regime.empty:
        return False, (
            "regime breakdown skipped: regime column had no usable rows for the "
            "breakdown"
        )

    plt = _import_matplotlib()
    figure, ax = plt.subplots(figsize=_PLOT_FIGURE_SIZE, dpi=_PLOT_DPI)
    ax.bar(
        accuracy_per_regime.index.tolist(),
        accuracy_per_regime.to_numpy(),
        color="steelblue",
        edgecolor="black",
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("regime")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("Regime breakdown (from stored artefacts)")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    figure.tight_layout()
    figure.savefig(target, dpi=_PLOT_DPI)
    plt.close(figure)
    return True, None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_paper_experiment_plots(
    experiment_dir: Path,
    *,
    overwrite: bool = False,
) -> PaperPlotSummary:
    """Generate plots from stored artefacts and return a typed summary.

    Parameters
    ----------
    experiment_dir:
        Path to a completed paper experiment directory.
    overwrite:
        When ``True``, existing plot files are replaced. When ``False``
        existing plots are kept and a clear warning is recorded.

    Returns
    -------
    PaperPlotSummary
        Typed summary listing the plots that were written, the ones that
        were skipped and any warnings collected along the way.
    """
    resolved_dir = Path(experiment_dir)
    if not resolved_dir.exists():
        raise FileNotFoundError(
            f"paper experiment directory does not exist: {resolved_dir}"
        )
    if not resolved_dir.is_dir():
        raise NotADirectoryError(
            f"paper experiment path is not a directory: {resolved_dir}"
        )

    plots_dir = _ensure_plots_dir(resolved_dir)
    builders: tuple[tuple[str, Any], ...] = (
        ("reliability_curve.png", _build_reliability_curve),
        ("cost_sensitivity.png", _build_cost_sensitivity),
        ("confusion_matrix.png", _build_confusion_matrix),
        ("regime_breakdown.png", _build_regime_breakdown),
    )

    plots_written: list[str] = []
    plots_skipped: list[str] = []
    warnings: list[str] = []

    for filename, builder in builders:
        try:
            wrote, message = builder(
                resolved_dir,
                plots_dir,
                overwrite=overwrite,
            )
        except RuntimeError as exc:
            wrote = False
            message = f"{filename} skipped: {exc}"
        if wrote:
            plots_written.append(f"plots/{filename}")
        else:
            plots_skipped.append(f"plots/{filename}")
            if message is not None:
                warnings.append(message)

    summary = PaperPlotSummary(
        experiment_dir=str(resolved_dir),
        created_at=datetime.now(UTC),
        builder_version=PAPER_PLOT_BUILDER_VERSION,
        plots_written=plots_written,
        plots_skipped=plots_skipped,
        warnings=warnings,
    )
    summary_path = resolved_dir / PLOT_SUMMARY_FILENAME
    summary_path.write_text(stable_json_dumps(summary), encoding="utf-8")
    return summary
