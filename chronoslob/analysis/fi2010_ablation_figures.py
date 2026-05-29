"""Figure builder for FI-2010 microstructure feature ablations."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from chronoslob.experiments.manifests import sha256_file, stable_json_dumps
from chronoslob.utils.paths import project_root

__all__ = [
    "DEFAULT_FI2010_ABLATION_FIGURE_DIR",
    "FI2010_ABLATION_FIGURE_VERSION",
    "FI2010AblationFigureSummary",
    "build_fi2010_ablation_figures",
]

FI2010_ABLATION_FIGURE_VERSION = "fi2010-feature-ablation-figures/v1"
DEFAULT_FI2010_ABLATION_FIGURE_DIR = Path("reports/figures/fi2010_feature_ablations")

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)
_PLOT_DPI = 160
_PLOT_SIZE = (7.6, 4.6)


class FI2010AblationFigureSummary(BaseModel):
    """Summary returned by the ablation figure builder."""

    model_config = _MODEL_CONFIG

    output_dir: str
    ablation_dir: str
    manifest_path: str
    completed_figures: list[str] = Field(default_factory=list)
    skipped_figures: list[str] = Field(default_factory=list)
    smoke_test: bool
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime
    builder_version: str

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value


def build_fi2010_ablation_figures(
    *,
    ablation_dir: Path,
    out_dir: Path = DEFAULT_FI2010_ABLATION_FIGURE_DIR,
    overwrite: bool = False,
    allow_smoke_test: bool = False,
) -> FI2010AblationFigureSummary:
    """Build ablation figures from stored feature-ablation CSV artefacts."""
    input_dir = Path(ablation_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"feature ablation directory missing: {input_dir}")
    summary_path = input_dir / "summary.json"
    summary = _read_json(summary_path) if summary_path.is_file() else {}
    smoke_test = bool(summary.get("smoke_test"))
    if smoke_test and not allow_smoke_test:
        raise ValueError(
            "ablation figure generation refuses smoke-test artefacts unless allow_smoke_test=True"
        )
    output_dir = Path(out_dir)
    _ensure_output_dir(output_dir, input_dir=input_dir, overwrite=overwrite)
    source_dir = output_dir / "source_data"
    metadata_dir = output_dir / "metadata"
    source_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    results = _read_csv_or_empty(input_dir / "results_summary.csv")
    delta = _read_csv_or_empty(input_dir / "feature_delta_summary.csv")
    aggregate = _read_csv_or_empty(input_dir / "aggregate_summary.csv")
    entries: list[dict[str, Any]] = []

    _build_delta_plot(
        entries=entries,
        frame=delta,
        metric="delta_macro_f1",
        figure_id="feature_group_delta_macro_f1",
        title="Feature Group Delta Macro-F1",
        ylabel="ablation minus all-features macro-F1",
        out_dir=output_dir,
        source_dir=source_dir,
        metadata_dir=metadata_dir,
        smoke_test=smoke_test,
    )
    _build_delta_plot(
        entries=entries,
        frame=delta,
        metric="delta_mcc",
        figure_id="feature_group_delta_mcc",
        title="Feature Group Delta MCC",
        ylabel="ablation minus all-features MCC",
        out_dir=output_dir,
        source_dir=source_dir,
        metadata_dir=metadata_dir,
        smoke_test=smoke_test,
    )
    _build_mode_metric_plot(
        entries=entries,
        frame=results,
        mode="only_one_group",
        metric="macro_f1",
        figure_id="only_one_group_comparison",
        title="Only-One-Group Comparison",
        ylabel="test macro-F1",
        out_dir=output_dir,
        source_dir=source_dir,
        metadata_dir=metadata_dir,
        smoke_test=smoke_test,
    )
    _build_delta_plot(
        entries=entries,
        frame=delta[delta.get("ablation_mode", pd.Series(dtype=str)) == "remove_one_group"]
        if not delta.empty
        else delta,
        metric="delta_macro_f1",
        figure_id="remove_one_group_degradation",
        title="Remove-One-Group Degradation",
        ylabel="removed-group macro-F1 delta",
        out_dir=output_dir,
        source_dir=source_dir,
        metadata_dir=metadata_dir,
        smoke_test=smoke_test,
    )
    _build_proxy_comparison(
        entries=entries,
        frame=results,
        out_dir=output_dir,
        source_dir=source_dir,
        metadata_dir=metadata_dir,
        smoke_test=smoke_test,
    )
    _build_horizon_importance(
        entries=entries,
        frame=aggregate,
        out_dir=output_dir,
        source_dir=source_dir,
        metadata_dir=metadata_dir,
        smoke_test=smoke_test,
    )

    completed = [entry["figure_id"] for entry in entries if entry["status"] == "completed"]
    skipped = [entry["figure_id"] for entry in entries if entry["status"] == "skipped"]
    manifest = {
        "builder_version": FI2010_ABLATION_FIGURE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "ablation_dir": _display_path(input_dir),
        "output_dir": _display_path(output_dir),
        "smoke_test": smoke_test,
        "figures": entries,
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(stable_json_dumps(manifest), encoding="utf-8")
    return FI2010AblationFigureSummary(
        output_dir=str(output_dir),
        ablation_dir=str(input_dir),
        manifest_path=str(manifest_path),
        completed_figures=completed,
        skipped_figures=skipped,
        smoke_test=smoke_test,
        warnings=[str(entry["reason"]) for entry in entries if entry["status"] == "skipped"],
        created_at=datetime.now(UTC),
        builder_version=FI2010_ABLATION_FIGURE_VERSION,
    )


def _build_delta_plot(
    *,
    entries: list[dict[str, Any]],
    frame: pd.DataFrame,
    metric: str,
    figure_id: str,
    title: str,
    ylabel: str,
    out_dir: Path,
    source_dir: Path,
    metadata_dir: Path,
    smoke_test: bool,
) -> None:
    if frame.empty or metric not in frame.columns:
        _skip(
            entries,
            figure_id=figure_id,
            title=title,
            reason=f"{metric} rows unavailable",
            smoke_test=smoke_test,
        )
        return
    source = frame.copy()
    source[metric] = pd.to_numeric(source[metric], errors="coerce")
    source = source.dropna(subset=[metric])
    if source.empty:
        _skip(
            entries,
            figure_id=figure_id,
            title=title,
            reason=f"no finite {metric} values",
            smoke_test=smoke_test,
        )
        return
    grouped = (
        source.groupby(["ablation_mode", "feature_group"], sort=True)[metric].mean().reset_index()
    )
    source_path = source_dir / f"{figure_id}.csv"
    grouped.to_csv(source_path, index=False)
    target = out_dir / f"{figure_id}.png"
    metadata_path = _write_metadata(
        metadata_dir,
        figure_id=figure_id,
        title=title,
        source_path=source_path,
        smoke_test=smoke_test,
    )
    _plot_bar(
        grouped,
        target,
        x_column="feature_group",
        y_column=metric,
        color_column="ablation_mode",
        title=_smoke_title(title, smoke_test),
        ylabel=ylabel,
    )
    _complete(
        entries,
        figure_id=figure_id,
        title=title,
        file_path=target,
        source_path=source_path,
        metadata_path=metadata_path,
        frame=grouped,
        smoke_test=smoke_test,
    )


def _build_mode_metric_plot(
    *,
    entries: list[dict[str, Any]],
    frame: pd.DataFrame,
    mode: str,
    metric: str,
    figure_id: str,
    title: str,
    ylabel: str,
    out_dir: Path,
    source_dir: Path,
    metadata_dir: Path,
    smoke_test: bool,
) -> None:
    if frame.empty or metric not in frame.columns or "ablation_mode" not in frame.columns:
        _skip(
            entries,
            figure_id=figure_id,
            title=title,
            reason="mode metric rows unavailable",
            smoke_test=smoke_test,
        )
        return
    source = frame[frame["ablation_mode"].astype(str) == mode].copy()
    source[metric] = pd.to_numeric(source[metric], errors="coerce")
    source = source[source["status"].astype(str) == "completed"].dropna(subset=[metric])
    if source.empty:
        _skip(
            entries,
            figure_id=figure_id,
            title=title,
            reason=f"no completed {mode} rows",
            smoke_test=smoke_test,
        )
        return
    grouped = source.groupby(["feature_group", "model"], sort=True)[metric].mean().reset_index()
    source_path = source_dir / f"{figure_id}.csv"
    grouped.to_csv(source_path, index=False)
    target = out_dir / f"{figure_id}.png"
    metadata_path = _write_metadata(
        metadata_dir,
        figure_id=figure_id,
        title=title,
        source_path=source_path,
        smoke_test=smoke_test,
    )
    _plot_bar(
        grouped,
        target,
        x_column="feature_group",
        y_column=metric,
        color_column="model",
        title=_smoke_title(title, smoke_test),
        ylabel=ylabel,
    )
    _complete(
        entries,
        figure_id=figure_id,
        title=title,
        file_path=target,
        source_path=source_path,
        metadata_path=metadata_path,
        frame=grouped,
        smoke_test=smoke_test,
    )


def _build_proxy_comparison(
    *,
    entries: list[dict[str, Any]],
    frame: pd.DataFrame,
    out_dir: Path,
    source_dir: Path,
    metadata_dir: Path,
    smoke_test: bool,
) -> None:
    figure_id = "proxy_vs_non_proxy_comparison"
    title = "Proxy Versus Non-Proxy Feature Comparison"
    if frame.empty or "ablation_mode" not in frame.columns:
        _skip(
            entries,
            figure_id=figure_id,
            title=title,
            reason="results_summary.csv unavailable",
            smoke_test=smoke_test,
        )
        return
    source = frame[frame["ablation_mode"].isin(["all_features", "no_proxy_features"])].copy()
    source["macro_f1"] = pd.to_numeric(source.get("macro_f1"), errors="coerce")
    source = source[source["status"].astype(str) == "completed"].dropna(subset=["macro_f1"])
    if source.empty:
        _skip(
            entries,
            figure_id=figure_id,
            title=title,
            reason="no completed all/no-proxy rows",
            smoke_test=smoke_test,
        )
        return
    source["feature_set"] = source["ablation_mode"].map(
        {"all_features": "with_proxy_features", "no_proxy_features": "without_proxy_features"}
    )
    grouped = source.groupby(["feature_set", "model"], sort=True)["macro_f1"].mean().reset_index()
    source_path = source_dir / f"{figure_id}.csv"
    grouped.to_csv(source_path, index=False)
    target = out_dir / f"{figure_id}.png"
    metadata_path = _write_metadata(
        metadata_dir,
        figure_id=figure_id,
        title=title,
        source_path=source_path,
        smoke_test=smoke_test,
    )
    _plot_bar(
        grouped,
        target,
        x_column="feature_set",
        y_column="macro_f1",
        color_column="model",
        title=_smoke_title(title, smoke_test),
        ylabel="test macro-F1",
    )
    _complete(
        entries,
        figure_id=figure_id,
        title=title,
        file_path=target,
        source_path=source_path,
        metadata_path=metadata_path,
        frame=grouped,
        smoke_test=smoke_test,
    )


def _build_horizon_importance(
    *,
    entries: list[dict[str, Any]],
    frame: pd.DataFrame,
    out_dir: Path,
    source_dir: Path,
    metadata_dir: Path,
    smoke_test: bool,
) -> None:
    figure_id = "horizon_specific_feature_importance"
    title = "Horizon-Specific Feature Importance"
    if frame.empty or "mean_macro_f1" not in frame.columns:
        _skip(
            entries,
            figure_id=figure_id,
            title=title,
            reason="aggregate_summary.csv unavailable",
            smoke_test=smoke_test,
        )
        return
    source = frame[frame["ablation_mode"].astype(str) == "only_one_group"].copy()
    source["mean_macro_f1"] = pd.to_numeric(source["mean_macro_f1"], errors="coerce")
    source = source.dropna(subset=["mean_macro_f1"])
    if source.empty:
        _skip(
            entries,
            figure_id=figure_id,
            title=title,
            reason="no only-one-group aggregate rows",
            smoke_test=smoke_test,
        )
        return
    grouped = (
        source.groupby(["horizon", "feature_group"], sort=True)["mean_macro_f1"]
        .mean()
        .reset_index()
    )
    source_path = source_dir / f"{figure_id}.csv"
    grouped.to_csv(source_path, index=False)
    target = out_dir / f"{figure_id}.png"
    metadata_path = _write_metadata(
        metadata_dir,
        figure_id=figure_id,
        title=title,
        source_path=source_path,
        smoke_test=smoke_test,
    )
    _plot_bar(
        grouped,
        target,
        x_column="feature_group",
        y_column="mean_macro_f1",
        color_column="horizon",
        title=_smoke_title(title, smoke_test),
        ylabel="mean macro-F1",
    )
    _complete(
        entries,
        figure_id=figure_id,
        title=title,
        file_path=target,
        source_path=source_path,
        metadata_path=metadata_path,
        frame=grouped,
        smoke_test=smoke_test,
    )


def _plot_bar(
    source: pd.DataFrame,
    target: Path,
    *,
    x_column: str,
    y_column: str,
    color_column: str,
    title: str,
    ylabel: str,
) -> None:
    plt = _import_matplotlib()
    frame = source.copy()
    frame[y_column] = pd.to_numeric(frame[y_column], errors="coerce")
    pivot = frame.pivot_table(index=x_column, columns=color_column, values=y_column, aggfunc="mean")
    figure, ax = plt.subplots(figsize=_PLOT_SIZE, dpi=_PLOT_DPI)
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(x_column.replace("_", " "))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="best", fontsize="x-small", frameon=False)
    figure.autofmt_xdate(rotation=30, ha="right")
    figure.tight_layout()
    figure.savefig(target, dpi=_PLOT_DPI)
    plt.close(figure)


def _import_matplotlib() -> Any:
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for ablation figure generation") from exc
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _write_metadata(
    metadata_dir: Path,
    *,
    figure_id: str,
    title: str,
    source_path: Path,
    smoke_test: bool,
) -> Path:
    path = metadata_dir / f"{figure_id}.json"
    path.write_text(
        stable_json_dumps(
            {
                "figure_id": figure_id,
                "title": title,
                "source_data_path": _display_path(source_path),
                "source_sha256": sha256_file(source_path),
                "smoke_test": smoke_test,
                "created_at": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    return path


def _complete(
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
            "row_count": len(frame),
            "smoke_test": smoke_test,
            "status": "completed",
            "reason": "",
        }
    )


def _skip(
    entries: list[dict[str, Any]],
    *,
    figure_id: str,
    title: str,
    reason: str,
    smoke_test: bool,
) -> None:
    entries.append(
        {
            "figure_id": figure_id,
            "title": title,
            "file_path": None,
            "source_data_path": None,
            "metadata_path": None,
            "row_count": 0,
            "smoke_test": smoke_test,
            "status": "skipped",
            "reason": reason,
        }
    )


def _ensure_output_dir(out_dir: Path, *, input_dir: Path, overwrite: bool) -> None:
    resolved_out = out_dir.resolve(strict=False)
    resolved_input = input_dir.resolve(strict=False)
    if resolved_out == resolved_input or resolved_input.is_relative_to(resolved_out):
        raise ValueError("ablation figure output directory must not contain the input directory")
    if out_dir.exists():
        if not out_dir.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {out_dir}")
        if any(out_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"refusing to overwrite non-empty output directory: {out_dir}"
                )
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


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
