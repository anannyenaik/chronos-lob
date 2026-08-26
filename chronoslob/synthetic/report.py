"""Markdown report and optional figures for the synthetic LOB extension.

The report wording is deliberately conservative: it labels every result as
synthetic and controlled, and restates that nothing here transfers to real
markets or changes FI-2010 limitations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chronoslob.synthetic.benchmark import BenchmarkResult

__all__ = [
    "render_report_markdown",
    "write_figures",
]


def render_report_markdown(
    *,
    summary: Mapping[str, Any],
    replay_quality: Mapping[str, Any],
    feature_summary_rows: Sequence[Mapping[str, Any]],
    label_summary_rows: Sequence[Mapping[str, Any]],
    benchmark: BenchmarkResult,
    regime_diagnostic_rows: Sequence[Mapping[str, Any]],
    claim_assessment: Mapping[str, Any],
) -> str:
    """Render the synthetic LOB extension report as Markdown text."""
    lines: list[str] = [
        "# ChronosLOB Synthetic Event-Level Extension",
        "",
        "This report is generated from a deterministic synthetic limit-order-book",
        "event simulator. It demonstrates event-level pipeline support under",
        "controlled synthetic regimes. It is not real-market evidence, it does not",
        "show tradability or returns, and it does not change any FI-2010 limitation:",
        "FI-2010 snapshots still expose only snapshot proxies, never true event-level",
        "order flow.",
        "",
    ]
    lines.extend(_overview_section(summary))
    lines.extend(_schema_section())
    lines.extend(_regime_section(feature_summary_rows))
    lines.extend(_replay_section(replay_quality))
    lines.extend(_feature_section())
    lines.extend(_label_section(label_summary_rows))
    lines.extend(_benchmark_section(benchmark))
    lines.extend(_diagnostics_section(regime_diagnostic_rows))
    lines.extend(_claim_section(claim_assessment))
    lines.extend(_limitations_section())
    return "\n".join(lines).rstrip() + "\n"


def _overview_section(summary: Mapping[str, Any]) -> list[str]:
    rows = [
        ("generated_at", str(summary.get("generated_at", "n/a"))),
        ("symbol", str(summary.get("symbol", "n/a"))),
        ("seed", str(summary.get("seed", "n/a"))),
        ("event_count", str(summary.get("event_count", "n/a"))),
        ("snapshot_count", str(summary.get("snapshot_count", "n/a"))),
        ("regimes", ", ".join(summary.get("regimes", []))),
        ("target", str(summary.get("target", "n/a"))),
        ("leakage_check_ok", str(summary.get("leakage_ok", "n/a"))),
        ("replay_ok", str(summary.get("replay_ok", "n/a"))),
        ("evidence_level", str(summary.get("evidence_level", "n/a"))),
    ]
    return ["## Overview", "", *_table(("field", "value"), rows), ""]


def _schema_section() -> list[str]:
    rows = [
        ("timestamp", "synthetic strictly increasing event time (UTC)"),
        ("sequence_id", "strictly increasing integer event index"),
        ("event_type", "ADD, CANCEL or TRADE"),
        ("side", "BID or ASK (aggressor side for TRADE)"),
        ("price", "tick-aligned price level"),
        ("quantity", "non-negative size added, cancelled or executed"),
        ("regime_id / regime_name", "known ground-truth regime label"),
        ("latent_mid", "latent reference mid used by the simulator"),
    ]
    return ["## Event Schema", "", *_table(("field", "meaning"), rows), ""]


def _regime_section(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    header = (
        "regime_name",
        "row_count",
        "mean_event_intensity",
        "mean_spread",
        "mean_event_order_flow_imbalance",
        "mean_cancellation_imbalance",
        "mean_trade_imbalance",
    )
    table_rows = [tuple(str(row.get(col, "")) for col in header) for row in rows]
    return [
        "## Regimes And Event-Level Feature Behaviour",
        "",
        "Each regime is a controlled synthetic environment with known intensities",
        "and imbalance. The table reports mean event-level features per regime.",
        "",
        *_table(header, table_rows),
        "",
    ]


def _replay_section(quality: Mapping[str, Any]) -> list[str]:
    rows = [
        ("event_count", str(quality.get("event_count", "n/a"))),
        ("snapshot_count", str(quality.get("snapshot_count", "n/a"))),
        ("crossed_snapshot_count", str(quality.get("crossed_snapshot_count", "n/a"))),
        ("negative_depth_event_count", str(quality.get("negative_depth_event_count", "n/a"))),
        ("sequence_gap_count", str(quality.get("sequence_gap_count", "n/a"))),
        ("ok", str(quality.get("ok", "n/a"))),
    ]
    return [
        "## Deterministic Replay Checks",
        "",
        "Replay rebuilds the book from the event stream alone and validates",
        "invariants: bid below ask, non-negative depth and continuous sequence ids.",
        "",
        *_table(("check", "value"), rows),
        "",
    ]


def _feature_section() -> list[str]:
    bullets = [
        "event_order_flow_imbalance: signed order flow from adds and cancels",
        "cancellation_imbalance: bid-versus-ask cancellation pressure",
        "trade_imbalance: buyer- versus seller-initiated executed volume",
        "event_intensity and add/cancel/trade rates",
        "spread, relative_spread and microprice_offset",
        "depth_imbalance_l1 and depth_imbalance_l5",
        "realised_volatility_proxy from latent mid changes",
    ]
    return [
        "## Supported Event-Level Features",
        "",
        "These features require event messages and are computed only on synthetic",
        "event streams here. FI-2010 snapshots cannot support them.",
        "",
        *[f"- {item}" for item in bullets],
        "",
    ]


def _label_section(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    header = ("label", "class_value", "count", "fraction")
    table_rows = [tuple(str(row.get(col, "")) for col in header) for row in rows]
    return [
        "## Labels",
        "",
        "Labels summarise a window strictly after each feature timestamp, so they",
        "do not leak into contemporaneous features.",
        "",
        *_table(header, table_rows),
        "",
    ]


def _benchmark_section(benchmark: BenchmarkResult) -> list[str]:
    header = ("model_name", "split", "n_samples", "accuracy", "macro_f1", "mcc")
    table_rows = [
        (
            metric.model_name,
            metric.split,
            str(metric.n_samples),
            f"{metric.accuracy:.4f}",
            f"{metric.macro_f1:.4f}",
            f"{metric.mcc:.4f}",
        )
        for metric in [*benchmark.chronological, *benchmark.regime_holdout]
    ]
    return [
        "## Baseline Diagnostics",
        "",
        "Small baselines confirm the synthetic features and labels flow through a",
        "standard train/validate/test protocol. This is platform and data",
        "validation on controlled synthetic regimes, not a real-market result.",
        "",
        f"Target: `{benchmark.target}`. Held-out regimes: "
        f"{', '.join(benchmark.holdout_regimes)}.",
        "",
        *_table(header, table_rows),
        "",
    ]


def _diagnostics_section(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    header = (
        "regime_name",
        "n_samples",
        "accuracy",
        "active_fraction",
        "filtered_accuracy",
        "turnover_proxy",
        "latency_accuracy",
    )
    table_rows = [tuple(str(row.get(col, "")) for col in header) for row in rows]
    return [
        "## Synthetic Regime Stress-Test Diagnostics",
        "",
        "Execution-aware proxy diagnostics broken down by known regime. These are",
        "controlled stress tests on synthetic data, not real-market execution",
        "evidence and not a returns or tradability claim.",
        "",
        *_table(header, table_rows),
        "",
    ]


def _claim_section(claim_assessment: Mapping[str, Any]) -> list[str]:
    claims = claim_assessment.get("claims", {})
    header = ("claim", "status", "reason")
    table_rows = [
        (name, str(payload.get("status", "")), str(payload.get("reason", "")))
        for name, payload in claims.items()
    ]
    return [
        "## Claim Assessment",
        "",
        *_table(header, table_rows),
        "",
    ]


def _limitations_section() -> list[str]:
    bullets = [
        "All data here is synthetic; it is not real-market evidence.",
        "Results do not transfer to real markets and imply no returns.",
        "This extension does not change any FI-2010 limitation.",
        "FI-2010 snapshots still expose only snapshot proxies, not event-level order flow.",
        "No live trading, profitability or tradability is implied.",
    ]
    return ["## Limitations", "", *[f"- {item}" for item in bullets], ""]


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def write_figures(
    out_dir: Path,
    *,
    feature_summary_rows: Sequence[Mapping[str, Any]],
    benchmark: BenchmarkResult,
) -> list[dict[str, Any]]:
    """Write compact figures if matplotlib is available; otherwise skip."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return [
            {"name": name, "status": "skipped", "reason": "matplotlib not installed"}
            for name in ("event_intensity_by_regime", "spread_by_regime", "macro_f1_by_model")
        ]

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    names = [str(row.get("regime_name", "")) for row in feature_summary_rows]
    entries.append(
        _bar_figure(
            plt,
            figures_dir / "event_intensity_by_regime.png",
            "event_intensity_by_regime",
            names,
            [float(row.get("mean_event_intensity", 0.0)) for row in feature_summary_rows],
            "mean event intensity",
        )
    )
    entries.append(
        _bar_figure(
            plt,
            figures_dir / "spread_by_regime.png",
            "spread_by_regime",
            names,
            [float(row.get("mean_spread", 0.0)) for row in feature_summary_rows],
            "mean spread",
        )
    )
    test_metrics = [m for m in benchmark.chronological if m.split == "test"]
    entries.append(
        _bar_figure(
            plt,
            figures_dir / "macro_f1_by_model.png",
            "macro_f1_by_model",
            [m.model_name for m in test_metrics],
            [m.macro_f1 for m in test_metrics],
            "test macro-F1",
        )
    )
    return entries


def _bar_figure(
    plt: Any,
    path: Path,
    name: str,
    labels: Sequence[str],
    values: Sequence[float],
    ylabel: str,
) -> dict[str, Any]:
    figure, axis = plt.subplots(figsize=(5.0, 3.0))
    axis.bar(range(len(values)), values, color="#3b6ea5")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axis.set_ylabel(ylabel)
    axis.set_title(name.replace("_", " "))
    figure.tight_layout()
    figure.savefig(path, dpi=90)
    plt.close(figure)
    return {"name": name, "status": "completed", "path": path.name}
