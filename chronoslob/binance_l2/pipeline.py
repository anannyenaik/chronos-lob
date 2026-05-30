"""Orchestrate the Binance L2 snapshot-plus-diff replay extension.

The pipeline loads a local REST-style depth snapshot and a local diff-depth
JSONL stream, reconstructs the book deterministically, computes the supported
event-level features and update-continuity audit, builds a granular replay
quality report and a claim assessment, and writes a compact, storage-light
artefact set under the output directory.

Everything here is offline: inputs are local files and no network call is made.
Binance diff-depth updates are aggregated level updates, not individual order
events, so the artefacts never attribute individual trades, cancellations or
queue positions.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from chronoslob.binance_l2.features import (
    BINANCE_FEATURE_COLUMNS,
    UNSUPPORTED_FEATURES,
    build_binance_feature_frame,
    build_update_continuity_frame,
)
from chronoslob.binance_l2.quality import build_replay_quality_report
from chronoslob.binance_l2.report import (
    render_binance_report_markdown,
    render_schema_markdown,
    write_figures,
)
from chronoslob.book.reconstruction import reconstruct_order_book
from chronoslob.data.binance import (
    load_binance_diff_events_jsonl,
    load_binance_snapshot_json,
)
from chronoslob.utils.paths import project_root

__all__ = [
    "BINANCE_L2_BUILDER_VERSION",
    "BinanceL2Config",
    "BinanceL2Result",
    "default_fixture_paths",
    "run_binance_l2_pipeline",
]

BINANCE_L2_BUILDER_VERSION = "binance-l2-extension/v1"


@dataclass(frozen=True)
class BinanceL2Config:
    """Configuration for a Binance L2 replay-extension run."""

    snapshot_path: Path
    updates_path: Path
    symbol: str | None = None
    max_depth: int | None = None
    window_events: int = 20
    stop_on_gap: bool = True
    allow_crossed: bool = False

    def __post_init__(self) -> None:
        if self.window_events < 1:
            raise ValueError("window_events must be >= 1")
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth must be positive when supplied")


@dataclass(frozen=True)
class BinanceL2Result:
    """Summary of a Binance L2 replay-extension run."""

    out_dir: Path
    files_written: tuple[Path, ...]
    diff_event_count: int
    applied_event_count: int
    snapshot_count: int
    feature_row_count: int
    replay_ok: bool
    summary: dict[str, Any]


def default_fixture_paths() -> tuple[Path, Path]:
    """Return the bundled Binance-shaped fixture snapshot and diff paths."""
    base = project_root() / "tests" / "fixtures" / "binance"
    return base / "synthetic_snapshot.json", base / "synthetic_diff_updates.jsonl"


def _is_fixture_path(path: Path) -> bool:
    parts = {part.lower() for part in Path(path).parts}
    return "tests" in parts and "fixtures" in parts


def run_binance_l2_pipeline(
    out_dir: Path,
    config: BinanceL2Config,
    *,
    make_figures: bool = False,
    overwrite: bool = False,
) -> BinanceL2Result:
    """Run the Binance L2 replay extension and write artefacts to ``out_dir``."""
    out_dir = Path(out_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {out_dir}")
    report_path = out_dir / "binance_l2_report.md"
    if report_path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing Binance L2 report; pass overwrite=True: {report_path}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot = load_binance_snapshot_json(config.snapshot_path, symbol=config.symbol)
    events = load_binance_diff_events_jsonl(config.updates_path)
    result = reconstruct_order_book(
        snapshot,
        events,
        max_depth=config.max_depth,
        stop_on_gap=config.stop_on_gap,
        allow_crossed=config.allow_crossed,
    )

    quality = build_replay_quality_report(result, events)
    feature_frame = build_binance_feature_frame(
        events, result.snapshots, window_events=config.window_events
    )
    continuity_frame = build_update_continuity_frame(events)
    feature_summary_rows = _feature_summary_rows(feature_frame)

    fixture_data = _is_fixture_path(config.snapshot_path) or _is_fixture_path(
        config.updates_path
    )
    claim_assessment = _build_claim_assessment(
        replay_ok=quality.ok,
        fixture_data=fixture_data,
    )
    summary = _build_summary(
        config=config,
        snapshot_symbol=snapshot.symbol,
        snapshot_last_update_id=snapshot.last_update_id,
        quality=quality,
        feature_row_count=len(feature_frame),
        fixture_data=fixture_data,
    )

    files_written = _write_artefacts(
        out_dir=out_dir,
        summary=summary,
        quality=quality.to_dict(),
        feature_frame=feature_frame,
        feature_summary_rows=feature_summary_rows,
        continuity_frame=continuity_frame,
        claim_assessment=claim_assessment,
        make_figures=make_figures,
    )

    return BinanceL2Result(
        out_dir=out_dir,
        files_written=tuple(files_written),
        diff_event_count=len(events),
        applied_event_count=quality.applied_event_count,
        snapshot_count=result.n_snapshots,
        feature_row_count=len(feature_frame),
        replay_ok=quality.ok,
        summary=summary,
    )


def _feature_summary_rows(feature_frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in BINANCE_FEATURE_COLUMNS:
        if column not in feature_frame.columns or feature_frame.empty:
            rows.append({"feature": column, "mean": "", "min": "", "max": "", "std": ""})
            continue
        series = pd.to_numeric(feature_frame[column], errors="coerce").dropna()
        if series.empty:
            rows.append({"feature": column, "mean": "", "min": "", "max": "", "std": ""})
            continue
        rows.append(
            {
                "feature": column,
                "mean": round(float(series.mean()), 6),
                "min": round(float(series.min()), 6),
                "max": round(float(series.max()), 6),
                "std": round(float(series.std(ddof=0)), 6),
            }
        )
    return rows


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


def _build_summary(
    *,
    config: BinanceL2Config,
    snapshot_symbol: str,
    snapshot_last_update_id: int,
    quality: Any,
    feature_row_count: int,
    fixture_data: bool,
) -> dict[str, Any]:
    evidence_level = (
        "binance_l2_fixture_replay" if fixture_data else "binance_l2_real_capture"
    )
    return {
        "builder_version": BINANCE_L2_BUILDER_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _current_git_commit(),
        "venue": "binance",
        "market": "crypto_spot",
        "symbol": snapshot_symbol,
        "evidence_level": evidence_level,
        "fixture_data": fixture_data,
        "snapshot_source": _display_path(config.snapshot_path),
        "updates_source": _display_path(config.updates_path),
        "snapshot_bytes": _file_size(config.snapshot_path),
        "updates_bytes": _file_size(config.updates_path),
        "input_file_hashes": {
            _display_path(config.snapshot_path): _sha256_file(config.snapshot_path),
            _display_path(config.updates_path): _sha256_file(config.updates_path),
        },
        "snapshot_last_update_id": snapshot_last_update_id,
        "diff_event_count": quality.event_count,
        "applied_event_count": quality.applied_event_count,
        "snapshot_count": quality.applied_event_count,
        "skipped_stale_count": quality.skipped_stale_count,
        "gap_count": quality.gap_count,
        "crossed_count": quality.crossed_count,
        "started_correctly": quality.started_correctly,
        "feature_row_count": feature_row_count,
        "window_events": config.window_events,
        "max_depth": config.max_depth,
        "stop_on_gap": config.stop_on_gap,
        "allow_crossed": config.allow_crossed,
        "replay_ok": quality.ok,
        "feature_columns": list(BINANCE_FEATURE_COLUMNS),
        "unsupported_features": [name for name, _ in UNSUPPORTED_FEATURES],
        "aggregated_level_updates": True,
        "live_trading": False,
        "network_calls": 0,
    }


def _build_claim_assessment(*, replay_ok: bool, fixture_data: bool) -> dict[str, Any]:
    pipeline_status = "supported" if replay_ok else "unsupported"
    real_stream_status = "needs_real_evidence" if fixture_data else pipeline_status
    real_stream_reason = (
        "The bundled sample uses Binance-shaped synthetic fixtures; the same "
        "offline parser/replay path accepts user-supplied local Binance captures."
        if fixture_data
        else (
            "Local Binance Spot depth snapshot and aggregated diff-depth updates "
            "were ingested and replayed offline."
        )
    )
    return {
        "builder_version": BINANCE_L2_BUILDER_VERSION,
        "claims": {
            "binance_l2_replay_pipeline": {
                "status": pipeline_status,
                "reason": (
                    "Snapshot-plus-diff replay reconstructs the book and passes the "
                    "continuity and invariant checks."
                ),
            },
            "real_event_level_stream_path": {
                "status": real_stream_status,
                "reason": real_stream_reason,
            },
            "binance_update_continuity_validation": {
                "status": "supported",
                "reason": "Update-id bracketing, stale-event and gap checks are enforced.",
            },
            "binance_order_book_invariants": {
                "status": "supported",
                "reason": "Non-negative depth and best bid below best ask are validated.",
            },
            "real_market_predictive_success": {
                "status": "unsupported",
                "reason": "No predictive or returns evidence is produced by replay.",
            },
            "equity_market_generalisation": {
                "status": "unsupported",
                "reason": "This is crypto-venue data and does not transfer to equities.",
            },
            "live_trading_or_profitability": {
                "status": "forbidden",
                "reason": "Replay is offline; no live trading or returns are implied.",
            },
            "individual_order_queue_position": {
                "status": "forbidden",
                "reason": "Aggregated level updates cannot expose individual queue position.",
            },
            "true_trades_or_cancellations_from_diff_depth": {
                "status": "unsupported",
                "reason": "Diff-depth carries aggregate level changes, not trades or cancels.",
            },
        },
    }


def _write_artefacts(
    *,
    out_dir: Path,
    summary: Mapping[str, Any],
    quality: Mapping[str, Any],
    feature_frame: pd.DataFrame,
    feature_summary_rows: Sequence[Mapping[str, Any]],
    continuity_frame: pd.DataFrame,
    claim_assessment: Mapping[str, Any],
    make_figures: bool,
) -> list[Path]:
    written: list[Path] = []

    written.append(_write_json(out_dir / "summary.json", summary))
    written.append(_write_json(out_dir / "replay_quality.json", quality))
    written.append(_write_json(out_dir / "binance_claim_assessment.json", claim_assessment))

    written.append(
        _write_frame(out_dir / "feature_summary.csv", pd.DataFrame(feature_summary_rows))
    )
    written.append(
        _write_frame(out_dir / "book_snapshot_summary.csv", _book_summary_frame(feature_frame))
    )
    written.append(_write_frame(out_dir / "update_continuity_summary.csv", continuity_frame))

    figure_entries: list[dict[str, Any]] = []
    if make_figures:
        figure_entries = write_figures(out_dir, feature_rows=feature_frame.to_dict("records"))
    written.append(
        _write_json(
            out_dir / "figure_manifest.json",
            {"builder_version": BINANCE_L2_BUILDER_VERSION, "figures": figure_entries},
        )
    )

    schema_path = out_dir / "binance_l2_schema.md"
    schema_path.write_text(render_schema_markdown(), encoding="utf-8")
    written.append(schema_path)

    markdown = render_binance_report_markdown(
        summary=summary,
        replay_quality=quality,
        feature_summary_rows=feature_summary_rows,
        unsupported_features=UNSUPPORTED_FEATURES,
        claim_assessment=claim_assessment,
    )
    report_path = out_dir / "binance_l2_report.md"
    report_path.write_text(markdown, encoding="utf-8")
    written.append(report_path)
    return written


def _book_summary_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "update_id",
        "timestamp",
        "best_bid",
        "best_ask",
        "mid_price",
        "spread",
        "depth_imbalance_l1",
    ]
    present = [column for column in columns if column in feature_frame.columns]
    if feature_frame.empty or not present:
        return pd.DataFrame(columns=columns)
    return feature_frame[present].copy()


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_frame(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False)
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except (OSError, ValueError):
        return Path(path).as_posix()


def _current_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
        )
    except (OSError, ValueError):
        return None
    if completed.returncode != 0:
        return None
    commit = completed.stdout.strip()
    return commit or None
