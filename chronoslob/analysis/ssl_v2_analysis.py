"""Conservative analysis for FI-2010 SSL-v2 benchmark artefacts."""

from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from chronoslob.experiments.manifests import stable_json_dumps
from chronoslob.utils.paths import project_root
from chronoslob.utils.release_text import (
    HAMILTON_PROVENANCE_PARAGRAPH,
    SSL_V2_SCOPE_PARAGRAPH,
)

__all__ = [
    "SSL_V2_ANALYSIS_VERSION",
    "SSLV2AnalysisSummary",
    "analyse_ssl_v2_results",
]

SSL_V2_ANALYSIS_VERSION = "fi2010-ssl-v2-analysis/v1"

# Confidence thresholds for selective-prediction (confidence-filtered) diagnostics.
SSL_V2_CONFIDENCE_THRESHOLDS: tuple[float, ...] = (0.33, 0.50, 0.70, 0.85, 0.95)


@dataclass(frozen=True)
class SSLV2AnalysisSummary:
    """Summary returned by :func:`analyse_ssl_v2_results`."""

    ssl_v2_dir: str
    out_dir: str
    evidence_level: str
    matched_rows: int
    ssl_v2_matched_rows: int
    failure_count: int
    claim_statuses: Mapping[str, str] = field(default_factory=dict)
    artefacts: Mapping[str, str] = field(default_factory=dict)
    confidence_filtered_rows: int = 0
    execution_proxy_available: bool = False


def analyse_ssl_v2_results(
    ssl_v2_dir: str | Path = Path("experiments/fi2010_ssl_v2_benchmark"),
    *,
    out_dir: str | Path = Path("reports/ssl_v2_analysis"),
) -> SSLV2AnalysisSummary:
    """Analyse stored SSL-v2 benchmark outputs and write report artefacts."""
    source = Path(ssl_v2_dir)
    output = Path(out_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"SSL-v2 benchmark directory not found: {source}")
    required = (
        "summary.json",
        "results_summary.csv",
        "aggregate_summary.csv",
        "ssl_v2_comparison.csv",
        "failures.csv",
    )
    for filename in required:
        if not (source / filename).is_file():
            raise FileNotFoundError(f"SSL-v2 required artefact missing: {source / filename}")
    output.mkdir(parents=True, exist_ok=True)
    summary = _read_json(source / "summary.json")
    comparison = pd.read_csv(source / "ssl_v2_comparison.csv")
    aggregate = pd.read_csv(source / "aggregate_summary.csv")
    failures = pd.read_csv(source / "failures.csv")
    loss_components = _read_optional_csv(source / "ssl_v2_loss_components.csv")

    metric_summary = aggregate.copy()
    metric_summary.to_csv(output / "ssl_v2_metric_summary.csv", index=False)
    delta_by_horizon = _delta_table(comparison, group_columns=("ssl_objective", "horizon"))
    delta_by_fold = _delta_table(comparison, group_columns=("ssl_objective", "fold"))
    delta_by_seed = _delta_table(comparison, group_columns=("ssl_objective", "seed"))
    delta_by_fold_horizon = _delta_table(
        comparison, group_columns=("ssl_objective", "fold", "horizon")
    )
    delta_overall = _delta_table(comparison, group_columns=("ssl_objective",))
    delta_by_horizon.to_csv(output / "ssl_v2_delta_by_horizon.csv", index=False)
    delta_by_fold.to_csv(output / "ssl_v2_delta_by_fold.csv", index=False)
    delta_by_seed.to_csv(output / "ssl_v2_delta_by_seed.csv", index=False)
    delta_by_fold_horizon.to_csv(output / "ssl_v2_delta_by_fold_horizon.csv", index=False)
    delta_overall.to_csv(output / "ssl_v2_delta_overall.csv", index=False)
    if loss_components is not None:
        loss_components.to_csv(output / "ssl_v2_loss_components.csv", index=False)
    else:
        pd.DataFrame().to_csv(output / "ssl_v2_loss_components.csv", index=False)

    confidence_filtered, confidence_status = _confidence_filtered_diagnostics(source, comparison)
    confidence_filtered.to_csv(output / "ssl_v2_confidence_filtered.csv", index=False)
    execution_proxy = _execution_proxy_note(confidence_status=confidence_status)
    (output / "ssl_v2_execution_proxy.json").write_text(
        stable_json_dumps(execution_proxy),
        encoding="utf-8",
    )

    claims = _assess_claims(
        summary=summary,
        comparison=comparison,
        loss_components=loss_components,
    )
    claim_statuses = {str(item["claim_id"]): str(item["status"]) for item in claims}
    git_commit = _current_git_commit()
    compute_provenance = (
        _read_json(output / "hamilton_compute_provenance.json")
        if (output / "hamilton_compute_provenance.json").is_file()
        else None
    )
    claim_payload = {
        "analysis_version": SSL_V2_ANALYSIS_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "ssl_v2_dir": source.as_posix(),
        "evidence_level": summary.get("evidence_level"),
        "claims": claims,
    }
    (output / "ssl_v2_claim_assessment.json").write_text(
        stable_json_dumps(claim_payload),
        encoding="utf-8",
    )
    figure_manifest = {
        "analysis_version": SSL_V2_ANALYSIS_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "figures": [],
        "reason": "No figures generated; lightweight CSV deltas are the canonical output.",
    }
    (output / "figure_manifest.json").write_text(
        stable_json_dumps(figure_manifest),
        encoding="utf-8",
    )
    report_text = "\n".join(
        _render_report(
            summary=summary,
            comparison=comparison,
            delta_by_horizon=delta_by_horizon,
            delta_by_fold=delta_by_fold,
            delta_by_seed=delta_by_seed,
            delta_overall=delta_overall,
            confidence_filtered=confidence_filtered,
            confidence_status=confidence_status,
            execution_proxy=execution_proxy,
            claims=claims,
            compute_provenance=compute_provenance,
        )
    )
    (output / "ssl_v2_analysis.md").write_text(report_text + "\n", encoding="utf-8")
    matched_rows = int((comparison["status"] == "matched").sum()) if "status" in comparison else 0
    ssl_v2_matched_rows = (
        int(
            (
                (comparison.get("status") == "matched")
                & (comparison.get("ssl_objective") == "market_state_multitask")
            ).sum()
        )
        if not comparison.empty
        else 0
    )
    failure_count = (
        len(failures[failures.get("status") == "failed"])
        if "status" in failures
        else len(failures)
    )
    artefacts = {
        "report": "ssl_v2_analysis.md",
        "metric_summary": "ssl_v2_metric_summary.csv",
        "delta_by_horizon": "ssl_v2_delta_by_horizon.csv",
        "delta_by_fold": "ssl_v2_delta_by_fold.csv",
        "delta_by_seed": "ssl_v2_delta_by_seed.csv",
        "delta_by_fold_horizon": "ssl_v2_delta_by_fold_horizon.csv",
        "delta_overall": "ssl_v2_delta_overall.csv",
        "confidence_filtered": "ssl_v2_confidence_filtered.csv",
        "execution_proxy_note": "ssl_v2_execution_proxy.json",
        "loss_components": "ssl_v2_loss_components.csv",
        "claim_assessment": "ssl_v2_claim_assessment.json",
        "figure_manifest": "figure_manifest.json",
        "summary": "summary.json",
    }
    if (output / "hamilton_compute_provenance.json").is_file():
        artefacts["compute_provenance"] = "hamilton_compute_provenance.json"
    analysis_summary = {
        "analysis_version": SSL_V2_ANALYSIS_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit,
        "ssl_v2_dir": source.as_posix(),
        "out_dir": output.as_posix(),
        "evidence_level": summary.get("evidence_level"),
        "scope_label": summary.get("scope_label"),
        "folds": summary.get("folds"),
        "horizons": summary.get("horizons"),
        "seeds": summary.get("seeds"),
        "lookbacks": summary.get("lookbacks"),
        "matched_rows": matched_rows,
        "ssl_v2_matched_rows": ssl_v2_matched_rows,
        "failure_count": failure_count,
        "claim_statuses": claim_statuses,
        "confidence_filtered_rows": len(confidence_filtered),
        "confidence_filtered_status": confidence_status,
        "execution_proxy_available": bool(execution_proxy["active_fraction_reported"]),
        "execution_proxy_note": execution_proxy,
        "artefacts": artefacts,
    }
    (output / "summary.json").write_text(stable_json_dumps(analysis_summary), encoding="utf-8")
    return SSLV2AnalysisSummary(
        ssl_v2_dir=source.as_posix(),
        out_dir=output.as_posix(),
        evidence_level=str(summary.get("evidence_level", "missing")),
        matched_rows=matched_rows,
        ssl_v2_matched_rows=ssl_v2_matched_rows,
        failure_count=failure_count,
        claim_statuses=claim_statuses,
        artefacts=artefacts,
        confidence_filtered_rows=len(confidence_filtered),
        execution_proxy_available=bool(execution_proxy["active_fraction_reported"]),
    )


def _read_json(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _current_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root(),
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _read_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    return pd.read_csv(path)


def _delta_table(
    comparison: pd.DataFrame,
    *,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame(
            columns=[
                *group_columns,
                "matched_run_count",
                "mean_delta_macro_f1",
                "mean_delta_mcc",
                "mean_delta_ece",
                "mean_delta_brier_score",
                "macro_f1_wins",
                "ece_wins",
            ]
        )
    matched = comparison[comparison["status"] == "matched"].copy()
    if matched.empty:
        return pd.DataFrame(columns=list(group_columns))
    rows: list[dict[str, Any]] = []
    for key, group in matched.groupby(list(group_columns), dropna=False, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        row = {column: values[index] for index, column in enumerate(group_columns)}
        row.update(
            {
                "matched_run_count": len(group),
                "mean_delta_macro_f1": _mean(group.get("delta_macro_f1")),
                "mean_delta_mcc": _mean(group.get("delta_mcc")),
                "mean_delta_ece": _mean(group.get("delta_ece")),
                "mean_delta_brier_score": _mean(group.get("delta_brier_score")),
                "macro_f1_wins": int((group.get("macro_f1_outcome") == "win").sum()),
                "ece_wins": int((group.get("ece_outcome") == "win").sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _mean(series: Any) -> float | None:
    if series is None:
        return None
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    value = float(numeric.mean())
    return value if math.isfinite(value) else None


def _assess_claims(
    *,
    summary: Mapping[str, Any],
    comparison: pd.DataFrame,
    loss_components: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    evidence_level = str(summary.get("evidence_level", "missing"))
    scope_text = (
        f"folds {summary.get('folds')}, horizons {summary.get('horizons')}, "
        f"seeds {summary.get('seeds')}, lookbacks {summary.get('lookbacks')}; "
        f"evidence level {evidence_level}"
    )
    v2_rows = comparison[
        (comparison.get("ssl_objective") == "market_state_multitask")
        & (comparison.get("status") == "matched")
    ]
    macro_values = pd.to_numeric(v2_rows.get("delta_macro_f1"), errors="coerce").dropna()
    mcc_values = pd.to_numeric(v2_rows.get("delta_mcc"), errors="coerce").dropna()
    ece_values = pd.to_numeric(v2_rows.get("delta_ece"), errors="coerce").dropna()
    brier_values = pd.to_numeric(v2_rows.get("delta_brier_score"), errors="coerce").dropna()
    predictive_supported = (
        not macro_values.empty
        and not mcc_values.empty
        and float(macro_values.mean()) > 0.0
        and float(mcc_values.mean()) > 0.0
        and evidence_level != "smoke_test_only"
    )
    calibration_supported = (
        not ece_values.empty
        and not brier_values.empty
        and float(ece_values.mean()) < 0.0
        and float(brier_values.mean()) < 0.0
        and evidence_level != "smoke_test_only"
    )
    implemented = bool(summary.get("completed_run_count", 0)) or (
        loss_components is not None and not loss_components.empty
    )
    evaluated = not v2_rows.empty and evidence_level != "smoke_test_only"
    return [
        {
            "claim_id": "ssl_v2_objective_implemented",
            "claim_text": "The second-generation market-state-aware SSL objective is implemented.",
            "status": "supported" if implemented else "needs_real_evidence",
            "scope": "code and stored benchmark artefacts",
            "reason": (
                "SSL-v2 benchmark artefacts or loss-component rows are present."
                if implemented
                else "No SSL-v2 run artefacts were found."
            ),
        },
        {
            "claim_id": "ssl_v2_evaluated",
            "claim_text": "SSL-v2 was evaluated in the exact stored FI-2010 scope.",
            "status": "supported" if evaluated else "needs_real_evidence",
            "scope": scope_text,
            "reason": (
                f"{len(v2_rows)} matched supervised-vs-SSL-v2 row(s) are present."
                if evaluated
                else "No non-smoke matched supervised-vs-SSL-v2 row is present."
            ),
        },
        {
            "claim_id": "ssl_v2_predictive_improvement",
            "claim_text": "SSL-v2 improves predictive metrics in the exact stored scope.",
            "status": "supported" if predictive_supported else "unsupported",
            "scope": scope_text,
            "reason": (
                "Mean macro-F1 and MCC deltas are positive for matched SSL-v2 "
                "rows in the exact stored scope."
                if predictive_supported
                else "Matched SSL-v2 macro-F1/MCC deltas do not jointly support improvement."
            ),
        },
        {
            "claim_id": "ssl_v2_calibration_improvement",
            "claim_text": "SSL-v2 improves calibration in the exact stored scope.",
            "status": "supported" if calibration_supported else "unsupported",
            "scope": scope_text,
            "reason": (
                "Mean ECE and Brier deltas improve for matched SSL-v2 rows in "
                "the exact stored scope."
                if calibration_supported
                else "ECE and Brier deltas do not jointly support calibration improvement."
            ),
        },
        {
            "claim_id": "broad_ssl_improvement",
            "claim_text": "SSL improves ChronosLOB overall.",
            "status": "unsupported",
            "scope": "blocked by existing SSL-v1 failure analysis and scoped SSL-v2 evidence",
            "reason": (
                "A scoped SSL-v2 result is not broad enough to overturn the "
                "broad SSL boundary."
            ),
        },
        {
            "claim_id": "foundation_model",
            "claim_text": "ChronosLOB is a foundation model.",
            "status": "forbidden",
            "scope": "not claimed",
            "reason": (
                "The SSL-v2 objective is a scoped pretraining objective, not a "
                "foundation-model claim."
            ),
        },
        {
            "claim_id": "sota",
            "claim_text": "ChronosLOB is state-of-the-art.",
            "status": "forbidden",
            "scope": "not claimed",
            "reason": "No SOTA claim is introduced or supported.",
        },
    ]


_CONFIDENCE_FILTERED_COLUMNS: tuple[str, ...] = (
    "grouping",
    "ssl_objective",
    "horizon",
    "threshold",
    "matched_run_count",
    "mean_ssl_active_fraction",
    "mean_ssl_abstention_fraction",
    "mean_ssl_active_examples",
    "mean_supervised_active_fraction",
    "mean_supervised_macro_f1",
    "mean_ssl_macro_f1",
    "mean_delta_macro_f1",
    "mean_supervised_mcc",
    "mean_ssl_mcc",
    "mean_delta_mcc",
)
_CONFIDENCE_AGGREGATE_COLUMNS: tuple[str, ...] = (
    "ssl_active_fraction",
    "ssl_abstention_fraction",
    "ssl_active_examples",
    "supervised_active_fraction",
    "supervised_macro_f1",
    "ssl_macro_f1",
    "delta_macro_f1",
    "supervised_mcc",
    "ssl_mcc",
    "delta_mcc",
)


def _run_predictions_path(
    source: Path,
    *,
    fold: Any,
    horizon: Any,
    seed: Any,
    lookback: Any,
    objective: str,
) -> Path:
    """Reconstruct a benchmark run's predictions path from comparison keys."""
    return (
        source
        / "runs"
        / f"fold_{int(float(fold))}"
        / f"horizon_{int(float(horizon))}"
        / f"seed_{int(float(seed))}"
        / f"lookback_{int(float(lookback))}"
        / str(objective)
        / "predictions.csv"
    )


def _load_test_predictions(path: Path) -> pd.DataFrame | None:
    """Load the y_true/y_pred/confidence test columns; None when unusable."""
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(
            path,
            usecols=lambda column: column in {"split", "y_true", "y_pred", "confidence"},
        )
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None
    if not {"y_true", "y_pred", "confidence"}.issubset(frame.columns):
        return None
    if "split" in frame.columns:
        frame = frame[frame["split"].astype(str) == "test"]
    frame = frame.copy()
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    frame = frame.dropna(subset=["y_true", "y_pred", "confidence"])
    return frame if not frame.empty else None


def _subset_classification_metrics(frame: pd.DataFrame) -> tuple[float | None, float | None]:
    """Return macro-F1 and MCC over a prediction subset, or None when empty."""
    if frame.empty:
        return None, None
    from chronoslob.training.metrics import compute_classification_metrics

    metrics = compute_classification_metrics(
        frame["y_true"].tolist(),
        frame["y_pred"].tolist(),
    )
    return float(metrics.macro_f1), float(metrics.matthews_corrcoef)


def _threshold_run_stats(predictions: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """Selective-prediction stats for one run at one confidence threshold."""
    total = len(predictions)
    active = predictions[predictions["confidence"] >= float(threshold)]
    n_active = len(active)
    active_fraction = (n_active / total) if total else 0.0
    macro_f1, mcc = (None, None)
    if n_active >= 1:
        macro_f1, mcc = _subset_classification_metrics(active)
    return {
        "n_active": n_active,
        "active_fraction": active_fraction,
        "abstention_fraction": 1.0 - active_fraction,
        "macro_f1": macro_f1,
        "mcc": mcc,
    }


def _confidence_filtered_diagnostics(
    source: Path,
    comparison: pd.DataFrame,
    *,
    thresholds: Sequence[float] = SSL_V2_CONFIDENCE_THRESHOLDS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute summary-light confidence-filtered deltas from on-disk predictions.

    The per-run ``predictions.csv`` files are git-ignored heavy artefacts. When
    they are present (a fresh benchmark in the same workspace) this emits a
    summary-light per-threshold table. When they are absent the table is empty
    and the status records that prediction-level artefacts are required.
    """
    empty = pd.DataFrame(columns=list(_CONFIDENCE_FILTERED_COLUMNS))
    status: dict[str, Any] = {
        "thresholds": [float(value) for value in thresholds],
        "predictions_available": False,
        "runs_with_predictions": 0,
        "runs_missing_predictions": 0,
        "reason": "",
    }
    if comparison.empty or "status" not in comparison.columns:
        status["reason"] = "no comparison rows available"
        return empty, status
    matched = comparison[
        (comparison["status"] == "matched")
        & (comparison.get("ssl_objective") == "market_state_multitask")
    ]
    if matched.empty:
        status["reason"] = "no matched supervised-vs-SSL-v2 rows"
        return empty, status

    records: list[dict[str, Any]] = []
    runs_with = 0
    runs_missing = 0
    for _, row in matched.iterrows():
        objective = str(row.get("ssl_objective"))
        keys = {
            "fold": row.get("fold"),
            "horizon": row.get("horizon"),
            "seed": row.get("seed"),
            "lookback": row.get("lookback"),
        }
        supervised = _load_test_predictions(
            _run_predictions_path(source, **keys, objective="supervised")
        )
        ssl = _load_test_predictions(
            _run_predictions_path(source, **keys, objective=objective)
        )
        if supervised is None or ssl is None:
            runs_missing += 1
            continue
        runs_with += 1
        for threshold in thresholds:
            sup_stats = _threshold_run_stats(supervised, threshold)
            ssl_stats = _threshold_run_stats(ssl, threshold)
            records.append(
                {
                    "ssl_objective": objective,
                    "fold": int(float(keys["fold"])),
                    "horizon": int(float(keys["horizon"])),
                    "seed": int(float(keys["seed"])),
                    "threshold": float(threshold),
                    "ssl_active_fraction": ssl_stats["active_fraction"],
                    "ssl_abstention_fraction": ssl_stats["abstention_fraction"],
                    "ssl_active_examples": ssl_stats["n_active"],
                    "supervised_active_fraction": sup_stats["active_fraction"],
                    "supervised_macro_f1": sup_stats["macro_f1"],
                    "ssl_macro_f1": ssl_stats["macro_f1"],
                    "supervised_mcc": sup_stats["mcc"],
                    "ssl_mcc": ssl_stats["mcc"],
                }
            )

    status["runs_with_predictions"] = runs_with
    status["runs_missing_predictions"] = runs_missing
    if not records:
        status["reason"] = (
            "per-run prediction files are required; benchmark predictions are "
            "stored as git-ignored heavy artefacts, so run the benchmark and this "
            "analysis in the same workspace"
        )
        return empty, status

    runs_df = pd.DataFrame(records)
    runs_df["delta_macro_f1"] = runs_df["ssl_macro_f1"] - runs_df["supervised_macro_f1"]
    runs_df["delta_mcc"] = runs_df["ssl_mcc"] - runs_df["supervised_mcc"]
    status["predictions_available"] = True

    frames: list[pd.DataFrame] = []
    groupings = (
        ("by_horizon", ["ssl_objective", "horizon", "threshold"], True),
        ("overall", ["ssl_objective", "threshold"], False),
    )
    aggregate_columns = list(_CONFIDENCE_AGGREGATE_COLUMNS)
    for grouping_label, group_columns, include_horizon in groupings:
        grouped = runs_df.groupby(group_columns, dropna=False, sort=True)
        means = grouped[aggregate_columns].mean(numeric_only=True)
        counts = grouped.size().rename("matched_run_count")
        merged = means.join(counts).reset_index()
        merged.insert(0, "grouping", grouping_label)
        if not include_horizon:
            merged["horizon"] = ""
        frames.append(merged)
    result = pd.concat(frames, ignore_index=True)
    result = result.rename(
        columns={column: f"mean_{column}" for column in aggregate_columns}
    )
    ordered = [column for column in _CONFIDENCE_FILTERED_COLUMNS if column in result.columns]
    return result[ordered], status


def _execution_proxy_note(*, confidence_status: Mapping[str, Any]) -> dict[str, Any]:
    """Honest execution-aware proxy status plus a storage-light design note."""
    active_fraction_reported = bool(confidence_status.get("predictions_available"))
    return {
        "analysis_version": SSL_V2_ANALYSIS_VERSION,
        "active_fraction_reported": active_fraction_reported,
        "active_fraction_source": "ssl_v2_confidence_filtered.csv",
        "deferred_proxies": [
            "turnover_proxy",
            "cost_adjusted_proxy",
            "latency_sensitivity",
            "adverse_selection_proxy",
        ],
        "computable_from_retained_artefacts": False,
        "prediction_level_artefacts_required": True,
        "reason": (
            "The retained SSL-v2 benchmark artefacts are summary-light. The "
            "turnover, cost-adjusted, latency-sensitivity and adverse-selection "
            "proxies need per-run test predictions, which are stored as "
            "git-ignored heavy artefacts and are not part of the retained set."
        ),
        "execution_hook": "execution_centrepiece",
        "execution_hook_command": "build-execution-centrepiece",
        "execution_hook_inputs": (
            "retained reports/execution_v3_analysis summary-light tables built "
            "from the neural full grid; the centrepiece never reopens raw "
            "predictions"
        ),
        "storage_light_design_note": (
            "Future SSL-v2 runs can emit per-(run, threshold) execution summary "
            "rows at evaluation time - active fraction, signal-change turnover "
            "proxy, cost-adjusted proxy, a latency-step degradation sweep and an "
            "adverse-selection proxy - and persist only the aggregated per-threshold "
            "table, mirroring the execution-v3 *_summary.csv tables, without ever "
            "storing raw per-row predictions. analyse-fi2010-ssl-v2-results would "
            "then aggregate those tables and the execution centrepiece could ingest "
            "them through its existing summary-light interface."
        ),
        "claim_boundary": (
            "Execution proxies are offline signal-quality diagnostics; no PnL, "
            "live-trading, tradability or production execution claim is implied."
        ),
    }


def _render_report(
    *,
    summary: Mapping[str, Any],
    comparison: pd.DataFrame,
    delta_by_horizon: pd.DataFrame,
    delta_by_fold: pd.DataFrame,
    delta_by_seed: pd.DataFrame,
    delta_overall: pd.DataFrame,
    confidence_filtered: pd.DataFrame,
    confidence_status: Mapping[str, Any],
    execution_proxy: Mapping[str, Any],
    claims: Sequence[Mapping[str, Any]],
    compute_provenance: Mapping[str, Any] | None,
) -> list[str]:
    evidence_level = str(summary.get("evidence_level", "missing"))
    scope_label = str(summary.get("scope_label", "missing"))
    v2_rows = comparison[
        (comparison.get("ssl_objective") == "market_state_multitask")
        & (comparison.get("status") == "matched")
    ]
    lines = [
        "# SSL-v2 Analysis",
        "",
        f"Analysis version: `{SSL_V2_ANALYSIS_VERSION}`.",
        "",
        "## Scope",
        "",
        f"- evidence level: `{evidence_level}`",
        f"- scope label: `{scope_label}`",
        f"- folds: {summary.get('folds')}",
        f"- horizons: {summary.get('horizons')}",
        f"- seeds: {summary.get('seeds')}",
        f"- lookbacks: {summary.get('lookbacks')}",
        f"- objectives: {summary.get('objectives')}",
        "",
        "SSL-v2 was added because the first-generation SSL analysis found that random "
        "field reconstruction and next-field prediction did not broadly improve "
        "downstream predictive or calibration metrics.",
        "The current closure covers the exact stored folds, horizons, seeds and "
        "lookbacks listed above.",
        "",
    ]
    if _release_scope_supported(summary=summary, v2_rows=v2_rows, claims=claims):
        lines += [*_wrap(SSL_V2_SCOPE_PARAGRAPH), ""]
    if compute_provenance:
        lines += [*_wrap(HAMILTON_PROVENANCE_PARAGRAPH), ""]
    lines += ["## Predictive Metrics", ""]
    if v2_rows.empty:
        lines.append("No matched supervised-vs-SSL-v2 rows were available.")
    else:
        lines += _markdown_table(
            (
                "horizon",
                "fold",
                "delta_macro_f1",
                "delta_mcc",
                "delta_ece",
                "delta_brier_score",
            ),
            [
                (
                    str(row.get("horizon", "")),
                    str(row.get("fold", "")),
                    _fmt(row.get("delta_macro_f1")),
                    _fmt(row.get("delta_mcc")),
                    _fmt(row.get("delta_ece")),
                    _fmt(row.get("delta_brier_score")),
                )
                for _, row in v2_rows.iterrows()
            ],
        )
    lines += [
        "",
        "## Aggregate, Seed, Horizon and Fold Deltas",
        "",
        "Canonical grouped summaries: `ssl_v2_delta_overall.csv`, "
        "`ssl_v2_delta_by_seed.csv`, `ssl_v2_delta_by_horizon.csv`, "
        "`ssl_v2_delta_by_fold.csv` and `ssl_v2_delta_by_fold_horizon.csv`.",
        "",
    ]
    lines += _aggregate_delta_block(delta_overall)
    mixed_note = _mixed_delta_note(delta_by_seed=delta_by_seed, delta_by_horizon=delta_by_horizon)
    if mixed_note:
        lines += ["", *_wrap(mixed_note)]
    lines += [
        "",
        "## Confidence-Filtered Diagnostics",
        "",
    ]
    lines += _wrap(
        "Selective-prediction deltas versus the matched supervised baseline at "
        "confidence thresholds "
        + ", ".join(f"{value:.2f}" for value in SSL_V2_CONFIDENCE_THRESHOLDS)
        + ". Active fraction, abstention and active-example counts are recorded "
        "in ssl_v2_confidence_filtered.csv."
    )
    lines.append("")
    lines += _confidence_filtered_block(confidence_filtered, confidence_status)
    lines += [
        "",
        "## Execution-Aware Proxy Diagnostics",
        "",
    ]
    lines += _execution_proxy_block(execution_proxy)
    lines += [
        "",
        "## Claim Assessment",
        "",
    ]
    lines += _markdown_table(
        ("claim", "status", "scope"),
        [
            (
                str(claim.get("claim_id", "")),
                str(claim.get("status", "")),
                str(claim.get("scope", "")),
            )
            for claim in claims
        ],
    )
    lines += [
        "",
        "## Conservative Interpretation",
        "",
        "This analysis reports predictive and calibration deltas only. It does not "
        "claim profitability, live trading, broad SSL improvement, market-wide "
        "generalisation, a foundation model, or state-of-the-art performance.",
    ]
    if not delta_by_horizon.empty or not delta_by_fold.empty:
        lines.append("")
        lines.append("Grouped CSV deltas are available for exact numeric inspection.")
    return lines


def _release_scope_supported(
    *,
    summary: Mapping[str, Any],
    v2_rows: pd.DataFrame,
    claims: Sequence[Mapping[str, Any]],
) -> bool:
    statuses = {
        str(claim.get("claim_id")): str(claim.get("status"))
        for claim in claims
        if isinstance(claim, Mapping)
    }
    return (
        summary.get("folds") == [1, 2, 3, 4, 5]
        and summary.get("horizons") == [10, 50]
        and summary.get("seeds") == [0, 1, 2]
        and summary.get("lookbacks") == [50]
        and len(v2_rows) == 30
        and statuses.get("ssl_v2_predictive_improvement") == "supported"
        and statuses.get("ssl_v2_calibration_improvement") == "supported"
    )


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return [header, separator, *body]


def _fmt(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.6f}"


def _wrap(text: str, *, width: int = 110) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width) or [""]


def _aggregate_delta_block(delta_overall: pd.DataFrame) -> list[str]:
    if delta_overall.empty:
        return ["No matched supervised-vs-SSL-v2 rows were available for aggregate deltas."]
    rows: list[tuple[str, ...]] = []
    for _, row in delta_overall.iterrows():
        if str(row.get("ssl_objective")) != "market_state_multitask":
            continue
        rows.append(
            (
                str(int(row.get("matched_run_count", 0) or 0)),
                _fmt(row.get("mean_delta_macro_f1")),
                _fmt(row.get("mean_delta_mcc")),
                _fmt(row.get("mean_delta_ece")),
                _fmt(row.get("mean_delta_brier_score")),
            )
        )
    if not rows:
        return ["No matched SSL-v2 aggregate row was available."]
    return _markdown_table(
        (
            "matched rows",
            "mean delta macro-F1",
            "mean delta MCC",
            "mean delta ECE",
            "mean delta Brier",
        ),
        rows,
    )


def _mixed_delta_note(*, delta_by_seed: pd.DataFrame, delta_by_horizon: pd.DataFrame) -> str:
    def _negative_groups(frame: pd.DataFrame, group: str) -> list[str]:
        if frame.empty:
            return []
        rows = frame[frame.get("ssl_objective") == "market_state_multitask"]
        return [
            str(row.get(group))
            for _, row in rows.iterrows()
            if float(row.get("mean_delta_macro_f1", 0.0)) < 0.0
        ]

    negative_seeds = _negative_groups(delta_by_seed, "seed")
    negative_horizons = _negative_groups(delta_by_horizon, "horizon")
    if not negative_seeds and not negative_horizons:
        return ""
    parts = []
    if negative_seeds:
        parts.append("negative mean macro-F1 for seed(s) " + ", ".join(negative_seeds))
    if negative_horizons:
        parts.append("negative mean macro-F1 for horizon(s) " + ", ".join(negative_horizons))
    return "Aggregate support is not uniform across strata: " + "; ".join(parts) + "."


def _confidence_filtered_block(
    confidence_filtered: pd.DataFrame,
    confidence_status: Mapping[str, Any],
) -> list[str]:
    if confidence_filtered.empty:
        reason = str(confidence_status.get("reason") or "prediction-level artefacts are required")
        return [
            *_wrap("Confidence-filtered diagnostics were not computed: " + reason + "."),
            "",
            *_wrap(
                "They are emitted automatically when per-run predictions are "
                "present in the benchmark runs/ tree at analysis time."
            ),
        ]
    overall = confidence_filtered[
        (confidence_filtered["grouping"] == "overall")
        & (confidence_filtered["ssl_objective"] == "market_state_multitask")
    ]
    if overall.empty:
        return ["No overall confidence-filtered rows were available."]
    rows: list[tuple[str, ...]] = []
    for _, row in overall.sort_values("threshold").iterrows():
        rows.append(
            (
                _fmt(row.get("threshold")),
                _fmt(row.get("mean_ssl_active_fraction")),
                _fmt(row.get("mean_ssl_macro_f1")),
                _fmt(row.get("mean_delta_macro_f1")),
                _fmt(row.get("mean_delta_mcc")),
            )
        )
    return _markdown_table(
        (
            "threshold",
            "SSL-v2 active fraction",
            "SSL-v2 macro-F1",
            "delta macro-F1",
            "delta MCC",
        ),
        rows,
    )


def _execution_proxy_block(execution_proxy: Mapping[str, Any]) -> list[str]:
    deferred = ", ".join(str(item) for item in execution_proxy.get("deferred_proxies", []))
    lines = [
        *_wrap(
            "Active fraction is reported above through the confidence-filtered "
            "diagnostics. The remaining execution-aware proxies are deferred."
        ),
        "",
        f"- deferred proxies: {deferred}",
        f"- computable from retained artefacts: "
        f"{execution_proxy.get('computable_from_retained_artefacts')}",
        f"- prediction-level artefacts required: "
        f"{execution_proxy.get('prediction_level_artefacts_required')}",
        f"- execution hook: `{execution_proxy.get('execution_hook_command')}` "
        f"({execution_proxy.get('execution_hook')})",
        "",
        "Reason:",
        "",
        *_wrap(str(execution_proxy.get("reason"))),
        "",
        "Storage-light design note for future runs:",
        "",
        *_wrap(str(execution_proxy.get("storage_light_design_note"))),
        "",
        *_wrap("Claim boundary: " + str(execution_proxy.get("claim_boundary"))),
    ]
    return lines


_RESERVED_SHUTIL = shutil
