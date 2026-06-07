"""Markdown report, schema doc and optional figures for the Binance L2 extension.

The wording is deliberately conservative. Binance L2 evidence is aggregated
depth-stream ingestion/replay engineering evidence for a crypto venue. It is
not individual-order event evidence, equity-market evidence, live trading or
profitability or predictive-success evidence, and it complements (does not
replace) the FI-2010 and synthetic evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "render_binance_report_markdown",
    "render_schema_markdown",
    "write_figures",
]


def render_binance_report_markdown(
    *,
    summary: Mapping[str, Any],
    replay_quality: Mapping[str, Any],
    feature_summary_rows: Sequence[Mapping[str, Any]],
    unsupported_features: Sequence[tuple[str, str]],
    claim_assessment: Mapping[str, Any],
) -> str:
    """Render the Binance L2 extension report as Markdown text."""
    fixture_data = bool(summary.get("fixture_data", False))
    if fixture_data:
        intro = [
            "This sample report is generated from small Binance-shaped synthetic",
            "fixtures using the same local snapshot-plus-diff replay contract used",
            "for user-supplied Binance Spot captures. It demonstrates the offline",
            "aggregated depth-stream engineering path; the fixture run is not",
            "exchange-data evidence.",
        ]
        scope = [
            "The extension is scoped to crypto-market data engineering support:",
            "not equity-market evidence, not live trading, and it does not establish",
            "profitability, tradability or predictive success.",
        ]
    else:
        intro = [
            "This report is generated from a Binance Spot L2 depth snapshot plus a",
            "diff-depth update stream, replayed offline into a local order book. It",
            "demonstrates a real captured aggregated depth-stream ingestion and",
            "replay path for a crypto venue.",
        ]
        scope = [
            "It is crypto-market data engineering evidence only: not equity-market evidence,",
            "not live trading, and it does not establish profitability, tradability or predictive",
            "success.",
        ]
    lines: list[str] = [
        "# ChronosLOB Binance Spot Aggregated L2 Replay",
        "",
        *intro,
        *scope,
        "It complements the FI-2010 and synthetic evidence.",
        "",
        "Binance diff-depth updates are aggregated level updates, not individual",
        "order-event data. This extension does not attribute individual trades or",
        "cancellations and does not model queue position from diff-depth alone.",
        "",
    ]
    lines.extend(_overview_section(summary))
    lines.extend(_data_source_section(summary))
    lines.extend(_method_section())
    lines.extend(_replay_section(replay_quality))
    lines.extend(_feature_section(feature_summary_rows))
    lines.extend(_unsupported_section(unsupported_features))
    lines.extend(_storage_section(summary))
    lines.extend(_claim_section(claim_assessment))
    lines.extend(_limitations_section())
    return "\n".join(lines).rstrip() + "\n"


def _overview_section(summary: Mapping[str, Any]) -> list[str]:
    rows = [
        ("generated_at", str(summary.get("generated_at", "n/a"))),
        ("symbol", str(summary.get("symbol", "n/a"))),
        ("venue", str(summary.get("venue", "binance"))),
        ("evidence_level", str(summary.get("evidence_level", "n/a"))),
        ("snapshot_last_update_id", str(summary.get("snapshot_last_update_id", "n/a"))),
        ("diff_event_count", str(summary.get("diff_event_count", "n/a"))),
        ("applied_event_count", str(summary.get("applied_event_count", "n/a"))),
        ("snapshot_count", str(summary.get("snapshot_count", "n/a"))),
        ("feature_row_count", str(summary.get("feature_row_count", "n/a"))),
        ("replay_ok", str(summary.get("replay_ok", "n/a"))),
    ]
    return ["## Overview", "", *_table(("field", "value"), rows), ""]


def _data_source_section(summary: Mapping[str, Any]) -> list[str]:
    rows = [
        ("snapshot_source", str(summary.get("snapshot_source", "n/a"))),
        ("updates_source", str(summary.get("updates_source", "n/a"))),
        ("fixture_data", str(summary.get("fixture_data", "n/a"))),
    ]
    return [
        "## Data Source",
        "",
        "The replay consumes a local REST-style depth snapshot and a local JSONL",
        "stream of diff-depth updates. Tests and the default sample use small",
        "Binance-shaped synthetic fixtures; users may supply their own captured",
        "snapshot and diff files.",
        "",
        *_table(("field", "value"), rows),
        "",
    ]


def _method_section() -> list[str]:
    bullets = [
        "Load a REST-style snapshot and read its lastUpdateId.",
        "Buffer diff-depth updates from the JSONL stream.",
        "Discard events whose final update id u <= snapshot lastUpdateId.",
        "Start at the first event where U <= lastUpdateId + 1 <= u.",
        "Require update-id continuity after start; verify pu == prior u when present.",
        "Upsert a price level for positive quantity; remove it for quantity zero.",
        "Record gaps, stale events and crossed books instead of silently repairing.",
    ]
    return [
        "## Snapshot-Plus-Diff Replay Method",
        "",
        *[f"- {item}" for item in bullets],
        "",
    ]


def _replay_section(quality: Mapping[str, Any]) -> list[str]:
    keys = (
        "started_correctly",
        "event_count",
        "applied_event_count",
        "skipped_stale_count",
        "gap_count",
        "crossed_count",
        "not_initialised_count",
        "invalid_quantity_count",
        "all_snapshots_uncrossed",
        "final_update_id",
        "final_bid_levels",
        "final_ask_levels",
        "ok",
    )
    rows = [(key, str(quality.get(key, "n/a"))) for key in keys]
    return [
        "## Replay Quality And Book Invariants",
        "",
        "Reconstruction rebuilds the book from the snapshot and diff stream and",
        "validates update continuity, stale-event skipping, non-negative depth and",
        "best bid below best ask. Findings are recorded rather than silently fixed.",
        "",
        *_table(("check", "value"), rows),
        "",
    ]


def _feature_section(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    header = ("feature", "mean", "min", "max")
    table_rows = [
        (
            str(row.get("feature", "")),
            str(row.get("mean", "")),
            str(row.get("min", "")),
            str(row.get("max", "")),
        )
        for row in rows
    ]
    return [
        "## Supported Event-Level Features",
        "",
        "These features are computed from aggregated diff-depth level updates and",
        "the reconstructed snapshots. Update and depth imbalances reflect aggregate",
        "level changes, not individual orders.",
        "",
        *_table(header, table_rows),
        "",
    ]


def _unsupported_section(unsupported: Sequence[tuple[str, str]]) -> list[str]:
    rows = [(name, reason) for name, reason in unsupported]
    return [
        "## Unsupported Features",
        "",
        "Diff-depth data alone cannot support the following. They are not computed",
        "and must not be claimed from this extension.",
        "",
        *_table(("feature", "reason"), rows),
        "",
    ]


def _storage_section(summary: Mapping[str, Any]) -> list[str]:
    rows = [
        ("snapshot_bytes", str(summary.get("snapshot_bytes", "n/a"))),
        ("updates_bytes", str(summary.get("updates_bytes", "n/a"))),
        ("retained_artefacts", "small summaries, CSVs and JSON only"),
        ("raw_capture_policy", "large raw captures are git-ignored by default"),
    ]
    return [
        "## Storage Footprint",
        "",
        "Only compact summaries are retained. Large raw captures are written to a",
        "git-ignored local directory and are never committed.",
        "",
        *_table(("field", "value"), rows),
        "",
    ]


def _claim_section(claim_assessment: Mapping[str, Any]) -> list[str]:
    claims = claim_assessment.get("claims", {})
    header = ("claim", "status", "reason")
    table_rows = [
        (name, str(payload.get("status", "")), str(payload.get("reason", "")))
        for name, payload in claims.items()
    ]
    return ["## Claim Assessment", "", *_table(header, table_rows), ""]


def _limitations_section() -> list[str]:
    bullets = [
        "Binance L2 evidence is crypto-market engineering evidence, not equity evidence.",
        "Diff-depth updates are aggregated level updates, not individual order events.",
        "No true trade, cancellation or queue-position attribution is possible here.",
        "It does not establish profitability, tradability or predictive success.",
        "It is not live trading; replay is offline from local files.",
        "It complements FI-2010 and synthetic evidence and changes no FI-2010 limitation.",
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


def render_schema_markdown() -> str:
    """Render a standalone schema document for the Binance L2 extension."""
    snapshot_rows = [
        ("symbol", "instrument symbol, e.g. TESTUSDT"),
        ("last_update_id", "snapshot lastUpdateId from the REST depth payload"),
        ("bids / asks", "price/quantity levels (bids descending, asks ascending)"),
        ("timestamp / received_timestamp", "optional exchange / capture timestamps"),
    ]
    diff_rows = [
        ("event_type", "depthUpdate (Binance 'e')"),
        ("event_time / transaction_time", "Binance 'E' / 'T' epoch-ms timestamps"),
        ("symbol", "instrument symbol (Binance 's')"),
        ("first_update_id", "U: first update id covered by the event"),
        ("final_update_id", "u: final update id covered by the event"),
        ("previous_final_update_id", "pu: prior event final update id when present"),
        ("bids / asks", "aggregated level updates; quantity 0 removes a level"),
        ("received_timestamp", "optional local capture timestamp"),
    ]
    book_rows = [
        ("timestamp", "timestamp of the applied diff event"),
        ("symbol / venue", "instrument symbol and 'binance' venue"),
        ("bids / asks", "reconstructed price levels after applying the diff"),
        ("sequence_id", "final update id of the most recent applied diff"),
    ]
    quality_rows = [
        ("started_correctly", "snapshot/diff bracketing succeeded"),
        ("applied_event_count", "number of diffs applied to the book"),
        ("skipped_stale_count", "diffs discarded as stale"),
        ("gap_count", "update-id continuity gaps recorded"),
        ("crossed_count", "crossed-book findings recorded"),
        ("final_bid_levels / final_ask_levels", "depth of the final reconstructed book"),
        ("ok", "true when no continuity or invariant violation was found"),
    ]
    lines = [
        "# Binance L2 Extension Schema",
        "",
        "Typed schemas for the real captured aggregated L2 replay path. Binance",
        "diff-depth updates are aggregated level updates, not individual order",
        "events.",
        "",
        "## Depth Snapshot",
        "",
        *_table(("field", "meaning"), snapshot_rows),
        "",
        "## Diff-Depth Update",
        "",
        *_table(("field", "meaning"), diff_rows),
        "",
        "## Reconstructed Book Snapshot",
        "",
        *_table(("field", "meaning"), book_rows),
        "",
        "## Replay Quality Report",
        "",
        *_table(("field", "meaning"), quality_rows),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_figures(
    out_dir: Path,
    *,
    feature_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Write compact replay figures if matplotlib is available; otherwise skip."""
    figure_names = (
        "spread_over_replay",
        "depth_imbalance_l1_over_replay",
        "mid_price_over_replay",
    )
    if not feature_rows:
        return [
            {"name": name, "status": "skipped", "reason": "no feature rows to plot"}
            for name in figure_names
        ]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return [
            {"name": name, "status": "skipped", "reason": "matplotlib not installed"}
            for name in figure_names
        ]

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    update_ids = [row.get("update_id", index) for index, row in enumerate(feature_rows)]
    entries = [
        _line_figure(
            plt,
            figures_dir / "spread_over_replay.png",
            "spread_over_replay",
            update_ids,
            [float(row.get("spread", 0.0)) for row in feature_rows],
            "spread",
        ),
        _line_figure(
            plt,
            figures_dir / "depth_imbalance_l1_over_replay.png",
            "depth_imbalance_l1_over_replay",
            update_ids,
            [float(row.get("depth_imbalance_l1", 0.0)) for row in feature_rows],
            "depth imbalance L1",
        ),
        _line_figure(
            plt,
            figures_dir / "mid_price_over_replay.png",
            "mid_price_over_replay",
            update_ids,
            [float(row.get("mid_price", 0.0)) for row in feature_rows],
            "mid price",
        ),
    ]
    return entries


def _line_figure(
    plt: Any,
    path: Path,
    name: str,
    x_values: Sequence[Any],
    y_values: Sequence[float],
    ylabel: str,
) -> dict[str, Any]:
    figure, axis = plt.subplots(figsize=(5.0, 3.0))
    axis.plot(x_values, y_values, marker="o", markersize=3, color="#3b6ea5")
    axis.set_xlabel("applied update index")
    axis.set_ylabel(ylabel)
    axis.set_title(name.replace("_", " "))
    figure.tight_layout()
    figure.savefig(path, dpi=90)
    plt.close(figure)
    return {"name": name, "status": "completed", "path": f"figures/{path.name}"}
