"""Storage-light analysis for the broader proper-training neural benchmark."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from chronoslob.training.metrics import compute_classification_metrics

DEFAULT_THRESHOLDS: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)
METRICS: tuple[str, ...] = ("accuracy", "macro_f1", "mcc", "brier_score", "ece")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"required benchmark table is missing: {path}")
    return pd.read_csv(path)


def _successful_results(source: Path) -> pd.DataFrame:
    frame = _read_csv(source / "results_summary.csv")
    required = {
        "fold",
        "horizon",
        "seed",
        "lookback",
        "model_family",
        "prediction_file",
        "status",
        *METRICS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"results_summary.csv is missing columns: {missing}")
    successful = frame[frame["status"].isin({"completed", "skipped_existing"})].copy()
    if successful.empty:
        raise ValueError("proper-training analysis requires completed benchmark rows")
    return successful


def _summary_table(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    aggregations: dict[str, list[str]] = {
        metric: ["mean", "std"] for metric in METRICS
    }
    grouped = frame.groupby(list(group_columns), dropna=False, sort=True).agg(aggregations)
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    grouped = grouped.reset_index()
    counts = (
        frame.groupby(list(group_columns), dropna=False, sort=True)
        .size()
        .rename("run_count")
        .reset_index()
    )
    return counts.merge(grouped, on=list(group_columns), how="left")


def _prediction_path(source: Path, relative_path: object) -> Path:
    path = Path(str(relative_path))
    return path if path.is_absolute() else source / path


def _confidence_rows(
    source: Path,
    results: pd.DataFrame,
    *,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results.to_dict(orient="records"):
        prediction_path = _prediction_path(source, result["prediction_file"])
        predictions = _read_csv(prediction_path)
        required = {"y_true", "y_pred", "confidence"}
        missing = sorted(required - set(predictions.columns))
        if missing:
            raise ValueError(f"{prediction_path} is missing columns: {missing}")
        if "split" in predictions.columns:
            predictions = predictions[predictions["split"].astype(str) == "test"]
        predictions = predictions.copy()
        predictions["confidence"] = pd.to_numeric(
            predictions["confidence"], errors="coerce"
        )
        predictions = predictions.dropna(subset=["y_true", "y_pred", "confidence"])
        total = len(predictions)
        for threshold in thresholds:
            active = predictions[predictions["confidence"] >= float(threshold)]
            macro_f1 = mcc = None
            if not active.empty:
                metrics = compute_classification_metrics(
                    active["y_true"].tolist(),
                    active["y_pred"].tolist(),
                )
                macro_f1 = float(metrics.macro_f1)
                mcc = float(metrics.matthews_corrcoef)
            rows.append(
                {
                    "run_id": result.get("run_id"),
                    "fold": int(result["fold"]),
                    "horizon": int(result["horizon"]),
                    "seed": int(result["seed"]),
                    "lookback": int(result["lookback"]),
                    "model_family": str(result["model_family"]),
                    "confidence_threshold": float(threshold),
                    "n_total": total,
                    "n_active": len(active),
                    "active_fraction": (len(active) / total) if total else 0.0,
                    "macro_f1": macro_f1,
                    "mcc": mcc,
                }
            )
    return rows


def analyse_proper_neural_benchmark(
    source_dir: str | Path,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Build retained summaries while raw per-run predictions are available."""
    source = Path(source_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"proper-training benchmark directory is missing: {source}")
    results = _successful_results(source)
    results.to_csv(source / "per_run_summary.csv", index=False)

    summary_groups = {
        "fold_summary.csv": ("fold", "model_family", "horizon", "lookback"),
        "seed_summary.csv": ("seed", "model_family", "horizon", "lookback"),
        "lookback_summary.csv": ("lookback", "model_family", "horizon"),
        "model_summary.csv": ("model_family",),
        "horizon_summary.csv": ("horizon", "model_family"),
    }
    for filename, group_columns in summary_groups.items():
        _summary_table(results, group_columns).to_csv(source / filename, index=False)

    confidence = pd.DataFrame(
        _confidence_rows(source, results, thresholds=thresholds)
    )
    confidence.to_csv(source / "confidence_filtered_summary.csv", index=False)
    confidence_groups = [
        "model_family",
        "horizon",
        "lookback",
        "confidence_threshold",
    ]
    confidence_aggregate = (
        confidence.groupby(confidence_groups, dropna=False, sort=True)
        .agg(
            run_count=("run_id", "size"),
            active_fraction_mean=("active_fraction", "mean"),
            active_fraction_std=("active_fraction", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            mcc_mean=("mcc", "mean"),
            mcc_std=("mcc", "std"),
        )
        .reset_index()
    )
    confidence_aggregate.to_csv(
        source / "confidence_filtered_aggregate.csv", index=False
    )

    payload = {
        "analysis": "broader proper-training neural benchmark",
        "completed_run_count": len(results),
        "models": sorted(results["model_family"].astype(str).unique().tolist()),
        "folds": sorted(pd.to_numeric(results["fold"]).astype(int).unique().tolist()),
        "seeds": sorted(pd.to_numeric(results["seed"]).astype(int).unique().tolist()),
        "lookbacks": sorted(
            pd.to_numeric(results["lookback"]).astype(int).unique().tolist()
        ),
        "horizons": sorted(
            pd.to_numeric(results["horizon"]).astype(int).unique().tolist()
        ),
        "confidence_thresholds": [float(value) for value in thresholds],
        "execution_proxy": (
            "active fraction is retained as a selective-prediction coverage proxy; "
            "no live execution, profitability or tradability claim is made"
        ),
        "storage_policy": (
            "retained CSV/JSON summaries are storage-light; per-run predictions, "
            "checkpoints and cluster logs remain excluded"
        ),
    }
    (source / "proper_neural_analysis_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    payload = analyse_proper_neural_benchmark(args.source)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
