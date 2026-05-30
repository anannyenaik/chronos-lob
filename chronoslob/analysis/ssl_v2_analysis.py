"""Conservative analysis for FI-2010 SSL-v2 benchmark artefacts."""

from __future__ import annotations

import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from chronoslob.experiments.manifests import stable_json_dumps

__all__ = [
    "SSL_V2_ANALYSIS_VERSION",
    "SSLV2AnalysisSummary",
    "analyse_ssl_v2_results",
]

SSL_V2_ANALYSIS_VERSION = "fi2010-ssl-v2-analysis/v1"


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
    delta_by_horizon.to_csv(output / "ssl_v2_delta_by_horizon.csv", index=False)
    delta_by_fold.to_csv(output / "ssl_v2_delta_by_fold.csv", index=False)
    if loss_components is not None:
        loss_components.to_csv(output / "ssl_v2_loss_components.csv", index=False)
    else:
        pd.DataFrame().to_csv(output / "ssl_v2_loss_components.csv", index=False)

    claims = _assess_claims(
        summary=summary,
        comparison=comparison,
        loss_components=loss_components,
    )
    claim_statuses = {str(item["claim_id"]): str(item["status"]) for item in claims}
    claim_payload = {
        "analysis_version": SSL_V2_ANALYSIS_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "ssl_v2_dir": str(source),
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
            claims=claims,
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
        "loss_components": "ssl_v2_loss_components.csv",
        "claim_assessment": "ssl_v2_claim_assessment.json",
        "figure_manifest": "figure_manifest.json",
        "summary": "summary.json",
    }
    analysis_summary = {
        "analysis_version": SSL_V2_ANALYSIS_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "ssl_v2_dir": str(source),
        "out_dir": str(output),
        "evidence_level": summary.get("evidence_level"),
        "scope_label": summary.get("scope_label"),
        "matched_rows": matched_rows,
        "ssl_v2_matched_rows": ssl_v2_matched_rows,
        "failure_count": failure_count,
        "claim_statuses": claim_statuses,
        "artefacts": artefacts,
    }
    (output / "summary.json").write_text(stable_json_dumps(analysis_summary), encoding="utf-8")
    return SSLV2AnalysisSummary(
        ssl_v2_dir=str(source),
        out_dir=str(output),
        evidence_level=str(summary.get("evidence_level", "missing")),
        matched_rows=matched_rows,
        ssl_v2_matched_rows=ssl_v2_matched_rows,
        failure_count=failure_count,
        claim_statuses=claim_statuses,
        artefacts=artefacts,
    )


def _read_json(path: Path) -> dict[str, Any]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


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
            "claim_text": "SSL-v2 was evaluated in the stored FI-2010 scope.",
            "status": "supported" if evaluated else "needs_real_evidence",
            "scope": f"evidence level {evidence_level}",
            "reason": (
                f"{len(v2_rows)} matched supervised-vs-SSL-v2 row(s) are present."
                if evaluated
                else "No non-smoke matched supervised-vs-SSL-v2 row is present."
            ),
        },
        {
            "claim_id": "ssl_v2_predictive_improvement",
            "claim_text": "SSL-v2 improves predictive metrics in the stored scope.",
            "status": "supported" if predictive_supported else "unsupported",
            "scope": f"exact stored scope; evidence level {evidence_level}",
            "reason": (
                "Mean macro-F1 and MCC deltas are positive for matched SSL-v2 rows."
                if predictive_supported
                else "Matched SSL-v2 macro-F1/MCC deltas do not jointly support improvement."
            ),
        },
        {
            "claim_id": "ssl_v2_calibration_improvement",
            "claim_text": "SSL-v2 improves calibration in the stored scope.",
            "status": "supported" if calibration_supported else "unsupported",
            "scope": f"exact stored scope; evidence level {evidence_level}",
            "reason": (
                "Mean ECE and Brier deltas improve for matched SSL-v2 rows."
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


def _render_report(
    *,
    summary: Mapping[str, Any],
    comparison: pd.DataFrame,
    delta_by_horizon: pd.DataFrame,
    delta_by_fold: pd.DataFrame,
    claims: Sequence[Mapping[str, Any]],
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
        "",
        "## Predictive Metrics",
        "",
    ]
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
        "## Horizon and Fold Deltas",
        "",
        "The CSV artefacts `ssl_v2_delta_by_horizon.csv` and "
        "`ssl_v2_delta_by_fold.csv` are the canonical grouped summaries.",
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


_RESERVED_SHUTIL = shutil
