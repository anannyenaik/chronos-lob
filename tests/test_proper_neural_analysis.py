"""Tests for storage-light broader proper-neural analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from chronoslob.analysis.proper_neural_analysis import analyse_proper_neural_benchmark


def test_analyse_proper_neural_benchmark_writes_retained_summaries(
    tmp_path: Path,
) -> None:
    source = tmp_path / "proper"
    run = source / "runs" / "matrix"
    run.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "run_id": "run_1",
                "fold": 1,
                "horizon": 10,
                "seed": 0,
                "lookback": 50,
                "model_family": "matrix_transformer",
                "prediction_file": "runs/matrix/predictions.csv",
                "status": "completed",
                "accuracy": 0.5,
                "macro_f1": 0.4,
                "mcc": 0.1,
                "brier_score": 0.6,
                "ece": 0.2,
            }
        ]
    ).to_csv(source / "results_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "split": "test",
                "y_true": 1,
                "y_pred": 1,
                "confidence": 0.9,
            },
            {
                "split": "test",
                "y_true": 2,
                "y_pred": 1,
                "confidence": 0.4,
            },
        ]
    ).to_csv(run / "predictions.csv", index=False)

    payload = analyse_proper_neural_benchmark(source, thresholds=(0.0, 0.5))

    assert payload["completed_run_count"] == 1
    assert payload["models"] == ["matrix_transformer"]
    expected = {
        "per_run_summary.csv",
        "fold_summary.csv",
        "seed_summary.csv",
        "lookback_summary.csv",
        "model_summary.csv",
        "horizon_summary.csv",
        "confidence_filtered_summary.csv",
        "confidence_filtered_aggregate.csv",
        "proper_neural_analysis_summary.json",
    }
    assert expected.issubset({path.name for path in source.iterdir()})
    confidence = pd.read_csv(source / "confidence_filtered_summary.csv")
    high = confidence[confidence["confidence_threshold"] == 0.5].iloc[0]
    assert high["active_fraction"] == 0.5
    assert high["macro_f1"] == 1.0
    summary = json.loads(
        (source / "proper_neural_analysis_summary.json").read_text(encoding="utf-8")
    )
    assert "selective-prediction coverage proxy" in summary["execution_proxy"]
