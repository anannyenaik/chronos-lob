"""Regenerate the public figures from the committed result summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPOSITORY_ROOT / "experiments" / "selected_results"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "figures"

FIGURE_STEMS = (
    "neural_benchmark_stability",
    "ssl_v2_matched_deltas",
    "feature_ablation",
    "selective_prediction_frontier",
    "execution_stress",
)

MODEL_LABELS = {
    "matrix_transformer": "Matrix transformer",
    "deeplob_style": "DeepLOB-style",
}
OBJECTIVE_LABELS = {
    "supervised": "Supervised",
    "masked_reconstruction": "Masked reconstruction",
    "next_field": "Next-field prediction",
}
FEATURE_LABELS = {
    "snapshot_order_flow_proxy": "Snapshot order-flow proxy",
    "size_levels": "Size levels",
    "spread": "Spread",
    "price_levels": "Price levels",
    "liquidity_concentration": "Liquidity concentration",
    "top_of_book_imbalance": "Top-of-book imbalance",
    "volatility_proxy": "Volatility proxy",
    "depth_imbalance": "Depth imbalance",
    "depth_slope": "Depth slope",
    "microprice": "Microprice",
    "top_of_book": "Top of book",
    "midprice": "Midprice",
}

BLUE = "#2166ac"
RED = "#b2182b"
PURPLE = "#762a83"
GREY = "#4d4d4d"
LIGHT_GREY = "#bdbdbd"
MODEL_COLOURS = {"matrix_transformer": BLUE, "deeplob_style": RED}
OBJECTIVE_COLOURS = {
    "supervised": GREY,
    "masked_reconstruction": BLUE,
    "next_field": PURPLE,
}
SEED_COLOURS = {0: BLUE, 1: RED, 2: PURPLE}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.65,
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_selected_results(results_dir: Path) -> list[Path]:
    """Verify the selected-result inventory and return its files in manifest order."""
    root = results_dir.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"selected-result manifest is missing: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"selected-result manifest is invalid JSON: {manifest_path}") from exc
    if not isinstance(manifest, dict) or manifest.get("manifest_version") != 1:
        raise ValueError("selected-result manifest must use manifest_version 1")

    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("selected-result manifest files must be a list")
    if manifest.get("file_count") != len(entries):
        raise ValueError("selected-result manifest file_count does not match its entries")

    prefix = ("experiments", "selected_results")
    verified: list[Path] = []
    listed_paths: set[Path] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"selected-result manifest entry {index} must be an object")
        raw_path = entry.get("path")
        expected_digest = entry.get("sha256")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"selected-result manifest entry {index} has no valid path")
        if (
            not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_digest)
        ):
            raise ValueError(
                f"selected-result manifest entry {raw_path!r} has no valid SHA-256 digest"
            )

        portable_path = PurePosixPath(raw_path.replace("\\", "/"))
        if portable_path.is_absolute() or ".." in portable_path.parts:
            raise ValueError(f"selected-result manifest path is unsafe: {raw_path!r}")
        if portable_path.parts[:2] != prefix or len(portable_path.parts) < 3:
            raise ValueError(
                "selected-result manifest paths must start with 'experiments/selected_results/'"
            )

        candidate = (root / Path(*portable_path.parts[2:])).resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(f"selected-result manifest path escapes its directory: {raw_path!r}")
        if candidate in listed_paths:
            raise ValueError(f"selected-result manifest path is duplicated: {raw_path!r}")
        if not candidate.is_file():
            raise FileNotFoundError(f"selected result is missing: {candidate}")

        actual_digest = _sha256(candidate)
        if actual_digest != expected_digest:
            raise ValueError(
                f"selected-result SHA-256 mismatch for {candidate}: "
                f"expected {expected_digest}, found {actual_digest}"
            )
        verified.append(candidate)
        listed_paths.add(candidate)

    inventory = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"README.md", "manifest.json"}
    }
    unlisted = sorted(inventory - listed_paths)
    if unlisted:
        relative_paths = ", ".join(path.relative_to(root).as_posix() for path in unlisted)
        raise ValueError(f"selected-result files are absent from the manifest: {relative_paths}")
    return verified


def _read_csv(
    results_dir: Path,
    relative_path: str,
    required_columns: set[str],
) -> pd.DataFrame:
    path = results_dir / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"required result table is missing: {path}")
    frame = pd.read_csv(path)
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"required result table is empty: {path}")
    return frame


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    converted = frame.copy()
    for column in columns:
        converted[column] = pd.to_numeric(converted[column], errors="raise")
    return converted


def _save_figure(figure: Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix in ("png", "pdf"):
        target = output_dir / f"{stem}.{suffix}"
        save_options: dict[str, Any] = {"bbox_inches": "tight"}
        if suffix == "png":
            save_options["dpi"] = 300
            save_options["metadata"] = {
                "Software": "ChronosLOB scripts/plot_results.py"
            }
        else:
            save_options["metadata"] = {
                "Creator": "ChronosLOB scripts/plot_results.py",
                "CreationDate": None,
                "ModDate": None,
            }
        figure.savefig(target, **save_options)
        written.append(target)
    plt.close(figure)
    return written


def plot_neural_benchmark(results_dir: Path, output_dir: Path) -> list[Path]:
    """Plot cell-level variation and descriptive uncertainty for the neural grid."""
    frame = _read_csv(
        results_dir,
        "proper_training/results_summary.csv",
        {
            "fold",
            "horizon",
            "seed",
            "lookback",
            "model_family",
            "macro_f1",
            "status",
        },
    )
    frame = frame[frame["status"].isin({"completed", "skipped_existing"})].copy()
    frame = _numeric(frame, ("fold", "horizon", "seed", "lookback", "macro_f1"))
    if frame.empty:
        raise ValueError("neural benchmark results contain no completed rows")

    horizons = sorted(frame["horizon"].astype(int).unique())
    lookbacks = sorted(frame["lookback"].astype(int).unique())
    figure, axes = plt.subplots(
        1,
        len(horizons),
        figsize=(7.2, 3.45),
        sharey=True,
        constrained_layout=True,
    )
    axis_list = np.atleast_1d(axes).tolist()
    positions = np.arange(len(lookbacks), dtype=float)
    offsets = {"matrix_transformer": -0.09, "deeplob_style": 0.09}

    for axis, horizon in zip(axis_list, horizons, strict=True):
        horizon_rows = frame[frame["horizon"] == horizon]
        for model in MODEL_LABELS:
            model_rows = horizon_rows[horizon_rows["model_family"] == model]
            means: list[float] = []
            standard_deviations: list[float] = []
            for position, lookback in zip(positions, lookbacks, strict=True):
                cell = model_rows[model_rows["lookback"] == lookback]
                values = cell["macro_f1"].to_numpy(dtype=float)
                if values.size < 2:
                    raise ValueError(
                        f"neural cell requires at least two rows: horizon={horizon}, "
                        f"lookback={lookback}, model={model}"
                    )
                means.append(float(values.mean()))
                standard_deviations.append(float(values.std(ddof=1)))
                jitter = (
                    (cell["seed"].to_numpy(dtype=float) - 1.0) * 0.014
                    + (cell["fold"].to_numpy(dtype=float) - 3.0) * 0.004
                )
                axis.scatter(
                    np.full(values.size, position + offsets[model]) + jitter,
                    values,
                    s=11,
                    color=MODEL_COLOURS[model],
                    alpha=0.28,
                    linewidths=0,
                    zorder=2,
                )
            axis.errorbar(
                positions + offsets[model],
                means,
                yerr=standard_deviations,
                fmt="o-",
                color=MODEL_COLOURS[model],
                linewidth=1.2,
                markersize=4.5,
                capsize=2.5,
                label=MODEL_LABELS[model],
                zorder=3,
            )
        axis.set_title(f"Horizon {horizon} events")
        axis.set_xticks(positions, [str(value) for value in lookbacks])
        axis.set_xlabel("Lookback (events)")
        axis.grid(axis="y")
        axis.set_ylim(0.15, 0.88)

    axis_list[0].set_ylabel("Test macro-F1")
    axis_list[-1].legend(frameon=False, loc="upper right")
    figure.suptitle("Neural benchmark stability", fontsize=11, fontweight="bold")
    figure.text(
        0.5,
        -0.01,
        "Small points are fold and seed cells. Large points are means; intervals are ±1 SD "
        "(n=15 per model, lookback and horizon).",
        ha="center",
        va="top",
        fontsize=7.2,
    )
    return _save_figure(figure, output_dir, "neural_benchmark_stability")


def plot_ssl_v2_deltas(results_dir: Path, output_dir: Path) -> list[Path]:
    """Plot all matched SSL-v2 differences, grouped by horizon and seed."""
    frame = _read_csv(
        results_dir,
        "ssl_v2/ssl_v2_comparison.csv",
        {
            "fold",
            "horizon",
            "seed",
            "delta_macro_f1",
            "delta_mcc",
            "delta_ece",
            "delta_brier_score",
            "status",
        },
    )
    frame = frame[frame["status"] == "matched"].copy()
    frame = _numeric(
        frame,
        (
            "fold",
            "horizon",
            "seed",
            "delta_macro_f1",
            "delta_mcc",
            "delta_ece",
            "delta_brier_score",
        ),
    )
    if frame.empty:
        raise ValueError("SSL-v2 comparison contains no matched rows")

    metrics = (
        ("delta_macro_f1", "Macro-F1", "Higher is better"),
        ("delta_mcc", "MCC", "Higher is better"),
        ("delta_ece", "ECE", "Lower is better"),
        ("delta_brier_score", "Brier score", "Lower is better"),
    )
    horizons = sorted(frame["horizon"].astype(int).unique())
    horizon_positions = {horizon: index for index, horizon in enumerate(horizons)}
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))

    for axis, (column, label, direction) in zip(axes.flat, metrics, strict=True):
        for horizon in horizons:
            subset = frame[frame["horizon"] == horizon]
            base = float(horizon_positions[horizon])
            for seed, seed_rows in subset.groupby("seed", sort=True):
                seed_number = int(seed)
                fold_jitter = (seed_rows["fold"].to_numpy(dtype=float) - 3.0) * 0.012
                y_values = np.full(len(seed_rows), base + (seed_number - 1) * 0.075)
                axis.scatter(
                    seed_rows[column].to_numpy(dtype=float),
                    y_values + fold_jitter,
                    s=18,
                    color=SEED_COLOURS[seed_number],
                    alpha=0.68,
                    edgecolors="white",
                    linewidths=0.25,
                    zorder=2,
                )
            values = subset[column].to_numpy(dtype=float)
            axis.errorbar(
                float(values.mean()),
                base,
                xerr=float(values.std(ddof=1)),
                fmt="D",
                color="black",
                markerfacecolor="white",
                markersize=4.5,
                capsize=2.5,
                linewidth=1.0,
                zorder=3,
            )
        axis.axvline(0.0, color=GREY, linewidth=0.8, linestyle="--", zorder=1)
        axis.set_yticks(
            list(horizon_positions.values()),
            [f"Horizon {horizon}" for horizon in horizons],
        )
        axis.invert_yaxis()
        axis.set_title(f"{label} delta")
        axis.set_xlabel(f"SSL-v2 minus supervised ({direction.lower()})")
        axis.grid(axis="x")

    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=SEED_COLOURS[seed],
            markeredgecolor="none",
            label=f"Seed {seed}",
        )
        for seed in sorted(SEED_COLOURS)
    ]
    legend_items.append(
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            label="Horizon mean ±1 SD",
        )
    )
    figure.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "SSL-v2 matched differences",
        fontsize=11,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.012,
        "Each small point is one matched fold and seed pair (n=15 per horizon).",
        ha="center",
        va="top",
        fontsize=7.2,
    )
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.79, hspace=0.52, wspace=0.29)
    return _save_figure(figure, output_dir, "ssl_v2_matched_deltas")


def pooled_feature_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    """Combine horizon-level means and sample SDs without inventing raw observations."""
    required = {
        "feature_group",
        "run_count",
        "mean_delta_macro_f1",
        "std_delta_macro_f1",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"feature delta table is missing required columns: {missing}")
    data = _numeric(
        frame,
        ("run_count", "mean_delta_macro_f1", "std_delta_macro_f1"),
    )
    rows: list[dict[str, float | int | str]] = []
    for feature_group, subset in data.groupby("feature_group", sort=True):
        counts = subset["run_count"].to_numpy(dtype=float)
        means = subset["mean_delta_macro_f1"].to_numpy(dtype=float)
        standard_deviations = subset["std_delta_macro_f1"].to_numpy(dtype=float)
        total_count = int(counts.sum())
        if total_count < 2 or np.any(counts < 2):
            raise ValueError(f"feature group {feature_group!r} lacks enough observations")
        pooled_mean = float(np.average(means, weights=counts))
        sum_squares = np.sum(
            (counts - 1.0) * np.square(standard_deviations)
            + counts * np.square(means - pooled_mean)
        )
        pooled_sd = math.sqrt(float(sum_squares) / (total_count - 1))
        rows.append(
            {
                "feature_group": str(feature_group),
                "run_count": total_count,
                "mean_delta_macro_f1": pooled_mean,
                "std_delta_macro_f1": pooled_sd,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_delta_macro_f1").reset_index(drop=True)


def _draw_forest(axis: Axes, summary: pd.DataFrame) -> None:
    y_positions = np.arange(len(summary), dtype=float)
    axis.errorbar(
        summary["mean_delta_macro_f1"].to_numpy(dtype=float),
        y_positions,
        xerr=summary["std_delta_macro_f1"].to_numpy(dtype=float),
        fmt="o",
        color=BLUE,
        ecolor=LIGHT_GREY,
        elinewidth=1.2,
        capsize=2,
        markersize=4,
        zorder=2,
    )
    axis.set_ylim(-0.7, len(summary) - 0.3)
    axis.invert_yaxis()
    axis.grid(axis="x")


def plot_feature_ablation(results_dir: Path, output_dir: Path) -> list[Path]:
    """Plot ranked feature-removal effects with pooled descriptive intervals."""
    frame = _read_csv(
        results_dir,
        "feature_ablation/feature_delta_by_horizon.csv",
        {
            "feature_group",
            "run_count",
            "mean_delta_macro_f1",
            "std_delta_macro_f1",
        },
    )
    summary = pooled_feature_deltas(frame)
    if len(summary) < 2:
        raise ValueError("feature-ablation figure requires at least two feature groups")

    extreme = summary.iloc[[0]].reset_index(drop=True)
    remaining = summary.iloc[1:].reset_index(drop=True)

    figure, (extreme_axis, detail_axis) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.7),
        gridspec_kw={"height_ratios": (1.0, 4.6), "hspace": 0.32},
    )
    for axis, subset in ((extreme_axis, extreme), (detail_axis, remaining)):
        _draw_forest(axis, subset)
        labels = [
            FEATURE_LABELS.get(value, str(value).replace("_", " ").title())
            for value in subset["feature_group"]
        ]
        y_positions = np.arange(len(subset), dtype=float)
        axis.set_yticks(y_positions, labels)
        axis.axvline(0.0, color=GREY, linewidth=0.8, linestyle="--", zorder=1)
        low = min(
            0.0,
            float(
                (subset["mean_delta_macro_f1"] - subset["std_delta_macro_f1"]).min()
            ),
        )
        high = max(
            0.0,
            float(
                (subset["mean_delta_macro_f1"] + subset["std_delta_macro_f1"]).max()
            ),
        )
        padding = max(0.001, (high - low) * 0.12)
        axis.set_xlim(low - padding, high + padding)
        label_x = axis.get_xlim()[1] - padding * 0.2
        for y_position, run_count in zip(
            y_positions, subset["run_count"].astype(int), strict=True
        ):
            axis.text(
                label_x,
                y_position,
                f"n={run_count}",
                ha="right",
                va="center",
                fontsize=6.8,
                color=GREY,
            )

    extreme_axis.set_title("Dominant effect", loc="left", fontsize=8.5)
    detail_axis.set_title("Expanded scale for remaining groups", loc="left", fontsize=8.5)
    detail_axis.set_xlabel("Change in macro-F1 after feature-group removal")
    figure.suptitle(
        "Feature-ablation effects",
        fontsize=11,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.012,
        "Points are pooled means; intervals are ±1 pooled SD from horizon summaries. "
        "Negative values indicate degradation after removal.",
        ha="center",
        va="bottom",
        fontsize=7.2,
    )
    figure.subplots_adjust(left=0.31, right=0.98, bottom=0.13, top=0.91)
    return _save_figure(figure, output_dir, "feature_ablation")


def plot_selective_prediction(results_dir: Path, output_dir: Path) -> list[Path]:
    """Plot confidence-filtered active fraction against predictive quality."""
    frame = _read_csv(
        results_dir,
        "execution/confidence_threshold_tradeoff.csv",
        {
            "pretraining_objective",
            "horizon",
            "threshold",
            "n_groups",
            "mean_active_fraction",
            "mean_macro_f1",
        },
    )
    frame = _numeric(
        frame,
        ("horizon", "threshold", "n_groups", "mean_active_fraction", "mean_macro_f1"),
    )
    horizons = sorted(frame["horizon"].astype(int).unique())
    figure, axes = plt.subplots(
        1,
        len(horizons),
        figsize=(7.2, 3.25),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axis_list = np.atleast_1d(axes).tolist()

    for axis, horizon in zip(axis_list, horizons, strict=True):
        horizon_rows = frame[frame["horizon"] == horizon]
        for objective in OBJECTIVE_LABELS:
            subset = horizon_rows[
                horizon_rows["pretraining_objective"] == objective
            ].sort_values("threshold")
            if subset.empty:
                continue
            axis.plot(
                subset["mean_active_fraction"],
                subset["mean_macro_f1"],
                marker="o",
                markersize=3.2,
                linewidth=1.1,
                color=OBJECTIVE_COLOURS[objective],
                label=OBJECTIVE_LABELS[objective],
            )
            axis.scatter(
                [float(subset.iloc[0]["mean_active_fraction"])],
                [float(subset.iloc[0]["mean_macro_f1"])],
                marker="s",
                s=25,
                color=OBJECTIVE_COLOURS[objective],
                zorder=3,
            )
            axis.scatter(
                [float(subset.iloc[-1]["mean_active_fraction"])],
                [float(subset.iloc[-1]["mean_macro_f1"])],
                marker="^",
                s=28,
                color=OBJECTIVE_COLOURS[objective],
                zorder=3,
            )
        axis.set_title(f"Horizon {horizon} events")
        axis.set_xlabel("Active fraction")
        axis.grid()
        axis.set_xlim(-0.015, 0.62)

    axis_list[0].set_ylabel("Macro-F1 after filtering")
    axis_list[-1].legend(frameon=False, loc="best")
    figure.suptitle(
        "Selective-prediction frontier",
        fontsize=11,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.015,
        "Squares mark threshold 0.33 and triangles mark 0.95. Each point averages 15 "
        "run groups; dispersion is unavailable.",
        ha="center",
        va="top",
        fontsize=7.2,
    )
    return _save_figure(figure, output_dir, "selective_prediction_frontier")


def _heatmap(
    axis: Axes,
    table: pd.DataFrame,
    *,
    title: str,
    annotation_format: str,
    colour_map: str,
    value_min: float | None = None,
    value_max: float | None = None,
) -> Any:
    image = axis.imshow(
        table.to_numpy(dtype=float),
        aspect="auto",
        cmap=colour_map,
        vmin=value_min,
        vmax=value_max,
    )
    axis.set_xticks(np.arange(len(table.columns)), [str(value) for value in table.columns])
    axis.set_yticks(
        np.arange(len(table.index)),
        [OBJECTIVE_LABELS.get(str(value), str(value)) for value in table.index],
    )
    axis.set_title(title)
    axis.set_xlabel("Horizon (events)")
    axis.spines["top"].set_visible(True)
    axis.spines["right"].set_visible(True)
    for row_index in range(len(table.index)):
        for column_index in range(len(table.columns)):
            value = float(table.iloc[row_index, column_index])
            midpoint = (
                (float(value_min) + float(value_max)) / 2.0
                if value_min is not None and value_max is not None
                else float(np.nanmedian(table.to_numpy(dtype=float)))
            )
            text_colour = "white" if value > midpoint else "#1a1a1a"
            axis.text(
                column_index,
                row_index,
                format(value, annotation_format),
                ha="center",
                va="center",
                fontsize=7.2,
                color=text_colour,
            )
    return image


def plot_execution_stress(results_dir: Path, output_dir: Path) -> list[Path]:
    """Plot the cost and latency stress diagnostics by horizon."""
    frame = _read_csv(
        results_dir,
        "execution/latency_cost_gap.csv",
        {
            "pretraining_objective",
            "horizon",
            "representative_fee_bps",
            "representative_spread_multiplier",
            "representative_cost_degradation_pct",
            "max_cost_fee_bps",
            "max_cost_spread_multiplier",
            "max_cost_degradation_pct",
            "representative_latency_step",
            "latency_degradation_vs_lag0",
        },
    )
    numeric_columns = (
        "horizon",
        "representative_fee_bps",
        "representative_spread_multiplier",
        "representative_cost_degradation_pct",
        "max_cost_fee_bps",
        "max_cost_spread_multiplier",
        "max_cost_degradation_pct",
        "representative_latency_step",
        "latency_degradation_vs_lag0",
    )
    frame = _numeric(frame, numeric_columns)
    objectives = list(OBJECTIVE_LABELS)
    horizons = sorted(frame["horizon"].astype(int).unique())

    def pivot(column: str) -> pd.DataFrame:
        return (
            frame.pivot(
                index="pretraining_objective", columns="horizon", values=column
            )
            .reindex(index=objectives, columns=horizons)
            .astype(float)
        )

    representative = pivot("representative_cost_degradation_pct")
    maximum = pivot("max_cost_degradation_pct")
    latency_loss = -pivot("latency_degradation_vs_lag0")
    if representative.isna().any().any() or maximum.isna().any().any():
        raise ValueError("execution stress table does not contain a complete objective grid")

    representative_fee = float(frame["representative_fee_bps"].iloc[0])
    representative_spread = float(frame["representative_spread_multiplier"].iloc[0])
    maximum_fee = float(frame["max_cost_fee_bps"].iloc[0])
    maximum_spread = float(frame["max_cost_spread_multiplier"].iloc[0])
    latency_step = int(frame["representative_latency_step"].iloc[0])
    cost_max = float(max(representative.to_numpy().max(), maximum.to_numpy().max()))

    figure, axes = plt.subplots(1, 3, figsize=(7.2, 3.25), constrained_layout=True)
    cost_image = _heatmap(
        axes[0],
        representative,
        title=(
            f"Cost degradation (%)\n{representative_fee:g} bps fee, "
            f"{representative_spread:g}x spread"
        ),
        annotation_format=".1f",
        colour_map="Blues",
        value_min=0.0,
        value_max=cost_max,
    )
    _heatmap(
        axes[1],
        maximum,
        title=(
            f"Cost degradation (%)\n{maximum_fee:g} bps fee, "
            f"{maximum_spread:g}x spread"
        ),
        annotation_format=".1f",
        colour_map="Blues",
        value_min=0.0,
        value_max=cost_max,
    )
    latency_image = _heatmap(
        axes[2],
        latency_loss,
        title=f"Signal-proxy loss\nlatency {latency_step} events",
        annotation_format=".0f",
        colour_map="Purples",
        value_min=0.0,
        value_max=float(latency_loss.to_numpy().max()),
    )
    axes[0].set_ylabel("Training objective")
    for axis in axes[1:]:
        axis.tick_params(axis="y", labelleft=False)
    cost_colourbar = figure.colorbar(cost_image, ax=axes[:2], shrink=0.78, pad=0.025)
    cost_colourbar.set_label("Degradation from zero-cost proxy (%)")
    latency_colourbar = figure.colorbar(latency_image, ax=axes[2], shrink=0.78, pad=0.025)
    latency_colourbar.set_label("Loss versus zero latency (proxy units)")
    figure.suptitle(
        "Execution stress diagnostic",
        fontsize=11,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.01,
        "Cells are aggregate offline proxies. The committed summaries retain two cost "
        "settings and one non-zero latency setting; no uncertainty is available.",
        ha="center",
        va="top",
        fontsize=7.2,
    )
    return _save_figure(figure, output_dir, "execution_stress")


def generate_figures(results_dir: Path, output_dir: Path) -> list[Path]:
    """Generate every public figure and return the written paths."""
    verify_selected_results(results_dir)
    _configure_style()
    written: list[Path] = []
    written.extend(plot_neural_benchmark(results_dir, output_dir))
    written.extend(plot_ssl_v2_deltas(results_dir, output_dir))
    written.extend(plot_feature_ablation(results_dir, output_dir))
    written.extend(plot_selective_prediction(results_dir, output_dir))
    written.extend(plot_execution_stress(results_dir, output_dir))
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate ChronosLOB public figures from committed result summaries."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing the committed selected result tables.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG and PDF figures.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify the selected-result manifest without generating figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir.resolve()
    if args.verify_only:
        verified = verify_selected_results(results_dir)
        print(f"Verified {len(verified)} selected result files.")
        return

    written = generate_figures(results_dir, args.output_dir.resolve())
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
