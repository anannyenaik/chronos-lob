"""Orchestrate the synthetic event-level pipeline and write artefacts.

The pipeline generates a deterministic synthetic event stream, replays it,
computes event-level features and future-horizon labels, runs small baseline
diagnostics and writes a compact, storage-light artefact set under the output
directory. Everything produced is synthetic and clearly labelled as such.
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from chronoslob.synthetic.benchmark import (
    BenchmarkResult,
    diagnostic_predictions,
    run_synthetic_benchmark,
)
from chronoslob.synthetic.diagnostics import (
    regime_execution_diagnostics,
    regime_feature_summary,
)
from chronoslob.synthetic.events import (
    SyntheticEventConfig,
    default_regime_plan,
    generate_synthetic_events,
)
from chronoslob.synthetic.features import build_event_feature_frame
from chronoslob.synthetic.labels import (
    LABEL_COLUMNS,
    build_label_frame,
    validate_no_lookahead_frames,
)
from chronoslob.synthetic.replay import replay_events_to_snapshots
from chronoslob.synthetic.report import render_report_markdown, write_figures

__all__ = [
    "SYNTHETIC_LOB_BUILDER_VERSION",
    "SyntheticLobConfig",
    "SyntheticLobResult",
    "run_synthetic_lob_pipeline",
    "smoke_config",
]

SYNTHETIC_LOB_BUILDER_VERSION = "synthetic-lob-extension/v1"

_EVENTS_SAMPLE_ROWS = 500
_SNAPSHOTS_SAMPLE_ROWS = 300

_DISCRETE_LABEL_COLUMNS: tuple[str, ...] = LABEL_COLUMNS


@dataclass(frozen=True)
class SyntheticLobConfig:
    """Configuration for a synthetic LOB pipeline run."""

    event_config: SyntheticEventConfig = field(default_factory=SyntheticEventConfig)
    snapshot_interval: int = 5
    window_events: int = 50
    horizon: int = 20
    benchmark_seed: int = 0
    holdout_regimes: tuple[str, ...] = ("high_volatility", "cancellation_shock")
    confidence_threshold: float = 0.5
    latency_steps: int = 1
    smoke_test: bool = False

    def __post_init__(self) -> None:
        if self.snapshot_interval < 1:
            raise ValueError("snapshot_interval must be >= 1")
        if self.window_events < 1:
            raise ValueError("window_events must be >= 1")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")


def smoke_config(events_per_regime: int = 120) -> SyntheticLobConfig:
    """Return a tiny, fast configuration for smoke checks and tests."""
    return SyntheticLobConfig(
        event_config=SyntheticEventConfig(
            regime_plan=default_regime_plan(events_per_regime),
            max_levels_per_side=6,
        ),
        snapshot_interval=3,
        window_events=20,
        horizon=8,
        smoke_test=True,
    )


@dataclass(frozen=True)
class SyntheticLobResult:
    """Summary of a synthetic LOB pipeline run."""

    out_dir: Path
    files_written: tuple[Path, ...]
    event_count: int
    snapshot_count: int
    replay_ok: bool
    leakage_ok: bool
    feature_row_count: int
    label_row_count: int
    benchmark: BenchmarkResult
    summary: dict[str, Any]


def run_synthetic_lob_pipeline(
    out_dir: Path,
    config: SyntheticLobConfig | None = None,
    *,
    make_figures: bool = False,
    overwrite: bool = False,
) -> SyntheticLobResult:
    """Run the synthetic event-level pipeline and write artefacts to ``out_dir``."""
    config = config or SyntheticLobConfig()
    out_dir = Path(out_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {out_dir}")
    report_path = out_dir / "synthetic_lob_report.md"
    if report_path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing synthetic report; pass overwrite=True: {report_path}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    generation = generate_synthetic_events(config.event_config)
    replay = replay_events_to_snapshots(
        generation.events,
        tick_size=config.event_config.tick_size,
        max_levels_per_side=config.event_config.max_levels_per_side,
        snapshot_interval=config.snapshot_interval,
        symbol=config.event_config.symbol,
    )
    feature_frame = build_event_feature_frame(
        generation.events,
        replay.snapshots,
        window_events=config.window_events,
    )
    label_frame = build_label_frame(feature_frame, horizon=config.horizon)
    leakage = validate_no_lookahead_frames(label_frame)
    benchmark = run_synthetic_benchmark(
        feature_frame,
        label_frame,
        seed=config.benchmark_seed,
        holdout_regimes=config.holdout_regimes,
    )

    feature_summary = regime_feature_summary(feature_frame)
    diagnostics = diagnostic_predictions(
        feature_frame, label_frame, seed=config.benchmark_seed
    )
    if diagnostics.available:
        regime_diag = regime_execution_diagnostics(
            diagnostics.test_frame,
            diagnostics.predictions,
            diagnostics.confidence,
            confidence_threshold=config.confidence_threshold,
            latency_steps=config.latency_steps,
        )
    else:
        regime_diag = pd.DataFrame()

    label_summary_rows = _label_summary_rows(label_frame)
    claim_assessment = _build_claim_assessment(
        regime_diagnostics_available=diagnostics.available,
        leakage_ok=leakage.ok,
    )
    summary = _build_summary(
        config=config,
        generation_event_count=generation.event_count,
        regime_event_counts=generation.regime_event_counts,
        event_type_counts=generation.event_type_counts,
        snapshot_count=replay.quality.snapshot_count,
        replay_ok=replay.quality.ok,
        leakage_ok=leakage.ok,
        feature_row_count=len(feature_frame),
        label_row_count=len(label_frame),
        benchmark=benchmark,
    )

    files_written = _write_artefacts(
        out_dir=out_dir,
        generation_events=generation.events,
        snapshots=replay.snapshots,
        replay_quality=replay.quality.to_dict(),
        feature_summary=feature_summary,
        label_summary_rows=label_summary_rows,
        benchmark=benchmark,
        regime_diag=regime_diag,
        claim_assessment=claim_assessment,
        summary=summary,
        make_figures=make_figures,
    )

    return SyntheticLobResult(
        out_dir=out_dir,
        files_written=tuple(files_written),
        event_count=generation.event_count,
        snapshot_count=replay.quality.snapshot_count,
        replay_ok=replay.quality.ok,
        leakage_ok=leakage.ok,
        feature_row_count=len(feature_frame),
        label_row_count=len(label_frame),
        benchmark=benchmark,
        summary=summary,
    )


def _write_artefacts(
    *,
    out_dir: Path,
    generation_events: Sequence[Any],
    snapshots: Sequence[Any],
    replay_quality: Mapping[str, Any],
    feature_summary: pd.DataFrame,
    label_summary_rows: Sequence[Mapping[str, Any]],
    benchmark: BenchmarkResult,
    regime_diag: pd.DataFrame,
    claim_assessment: Mapping[str, Any],
    summary: Mapping[str, Any],
    make_figures: bool,
) -> list[Path]:
    written: list[Path] = []

    written.append(_write_json(out_dir / "summary.json", summary))
    written.append(_write_json(out_dir / "synthetic_data_summary.json", summary))
    written.append(_write_json(out_dir / "synthetic_replay_quality.json", replay_quality))
    written.append(_write_json(out_dir / "synthetic_claim_assessment.json", claim_assessment))

    written.append(_write_frame(out_dir / "synthetic_feature_summary.csv", feature_summary))
    written.append(
        _write_rows(
            out_dir / "synthetic_label_summary.csv",
            ("label", "class_value", "count", "fraction"),
            label_summary_rows,
        )
    )
    benchmark_rows = [
        metric.to_row() for metric in [*benchmark.chronological, *benchmark.regime_holdout]
    ]
    written.append(
        _write_rows(
            out_dir / "synthetic_benchmark_summary.csv",
            (
                "model_name",
                "split",
                "n_samples",
                "accuracy",
                "macro_f1",
                "mcc",
                "brier_score",
                "expected_calibration_error",
            ),
            benchmark_rows,
        )
    )
    written.append(_write_frame(out_dir / "synthetic_regime_diagnostics.csv", regime_diag))

    written.append(
        _write_events_sample(out_dir / "synthetic_events_sample.csv", generation_events)
    )
    written.append(
        _write_snapshots_sample(out_dir / "synthetic_snapshots_sample.csv", snapshots)
    )

    figure_entries: list[dict[str, Any]] = []
    if make_figures:
        figure_entries = write_figures(
            out_dir,
            feature_summary_rows=feature_summary.to_dict("records"),
            benchmark=benchmark,
        )
    written.append(
        _write_json(
            out_dir / "figure_manifest.json",
            {"builder_version": SYNTHETIC_LOB_BUILDER_VERSION, "figures": figure_entries},
        )
    )

    markdown = render_report_markdown(
        summary=summary,
        replay_quality=replay_quality,
        feature_summary_rows=feature_summary.to_dict("records"),
        label_summary_rows=label_summary_rows,
        benchmark=benchmark,
        regime_diagnostic_rows=regime_diag.to_dict("records"),
        claim_assessment=claim_assessment,
    )
    report_path = out_dir / "synthetic_lob_report.md"
    report_path.write_text(markdown, encoding="utf-8")
    written.append(report_path)
    return written


def _label_summary_rows(label_frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if label_frame.empty:
        return rows
    total = len(label_frame)
    for column in _DISCRETE_LABEL_COLUMNS:
        if column not in label_frame.columns:
            continue
        counts = label_frame[column].value_counts().sort_index()
        for class_value, count in counts.items():
            rows.append(
                {
                    "label": column,
                    "class_value": int(class_value),
                    "count": int(count),
                    "fraction": round(int(count) / total, 6),
                }
            )
    return rows


def _build_summary(
    *,
    config: SyntheticLobConfig,
    generation_event_count: int,
    regime_event_counts: Mapping[str, int],
    event_type_counts: Mapping[str, int],
    snapshot_count: int,
    replay_ok: bool,
    leakage_ok: bool,
    feature_row_count: int,
    label_row_count: int,
    benchmark: BenchmarkResult,
) -> dict[str, Any]:
    benchmark_run_count = len(benchmark.chronological) + len(benchmark.regime_holdout)
    regimes = [name for name, _ in config.event_config.regime_plan]
    return {
        "builder_version": SYNTHETIC_LOB_BUILDER_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _current_git_commit(),
        "synthetic": True,
        "smoke_test": config.smoke_test,
        "evidence_level": "smoke_test_only" if config.smoke_test else "synthetic_controlled",
        "symbol": config.event_config.symbol,
        "seed": config.event_config.seed,
        "tick_size": config.event_config.tick_size,
        "snapshot_interval": config.snapshot_interval,
        "window_events": config.window_events,
        "horizon": config.horizon,
        "event_count": generation_event_count,
        "snapshot_count": snapshot_count,
        "feature_row_count": feature_row_count,
        "label_row_count": label_row_count,
        "regimes": list(dict.fromkeys(regimes)),
        "regime_event_counts": dict(regime_event_counts),
        "event_type_counts": dict(event_type_counts),
        "replay_ok": replay_ok,
        "leakage_ok": leakage_ok,
        "target": benchmark.target,
        "feature_columns": list(benchmark.feature_columns),
        "label_columns": list(LABEL_COLUMNS),
        "benchmark_models": ["majority", "logistic", "ridge", "gradient_boosting"],
        "holdout_regimes": list(benchmark.holdout_regimes),
        "completed_run_count": benchmark_run_count,
        "failed_run_count": 0,
        "warnings": list(benchmark.warnings),
    }


def _build_claim_assessment(
    *,
    regime_diagnostics_available: bool,
    leakage_ok: bool,
) -> dict[str, Any]:
    diagnostics_status = "supported" if regime_diagnostics_available else "needs_real_evidence"
    return {
        "builder_version": SYNTHETIC_LOB_BUILDER_VERSION,
        "claims": {
            "synthetic_event_level_pipeline": {
                "status": "supported",
                "reason": "Deterministic generation, replay, features and labels are produced.",
            },
            "synthetic_event_level_features": {
                "status": "supported",
                "reason": "Order-flow, cancellation and trade imbalance are computed from events.",
            },
            "synthetic_regime_diagnostics": {
                "status": diagnostics_status,
                "reason": "Per-regime execution-aware proxy diagnostics over known regimes.",
            },
            "no_lookahead_labels": {
                "status": "supported" if leakage_ok else "needs_real_evidence",
                "reason": "Labels use windows strictly after the feature timestamp.",
            },
            "real_market_event_level_generalisation": {
                "status": "unsupported",
                "reason": "All data is synthetic; nothing here transfers to real markets.",
            },
            "synthetic_to_real_transfer": {
                "status": "unsupported",
                "reason": "Synthetic results do not transfer to real markets.",
            },
            "live_trading_or_profitability": {
                "status": "forbidden",
                "reason": "No returns, tradability or live trading is implied or claimed.",
            },
            "fi2010_true_event_level_ofi": {
                "status": "forbidden",
                "reason": (
                    "FI-2010 snapshots expose only proxies, never true event-level "
                    "order flow."
                ),
            },
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_frame(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False)
    return path


def _write_rows(
    path: Path,
    header: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})
    return path


def _write_events_sample(path: Path, events: Sequence[Any]) -> Path:
    header = (
        "sequence_id",
        "event_type",
        "side",
        "price",
        "quantity",
        "regime_id",
        "regime_name",
        "latent_mid",
    )
    rows: list[dict[str, Any]] = []
    for event in events[:_EVENTS_SAMPLE_ROWS]:
        rows.append(
            {
                "sequence_id": event.sequence_id,
                "event_type": event.event_type.value,
                "side": event.side.value if event.side else "",
                "price": event.price,
                "quantity": event.quantity,
                "regime_id": event.metadata.get("regime_id", ""),
                "regime_name": event.metadata.get("regime_name", ""),
                "latent_mid": event.metadata.get("latent_mid", ""),
            }
        )
    return _write_rows(path, header, rows)


def _write_snapshots_sample(path: Path, snapshots: Sequence[Any]) -> Path:
    header = (
        "sequence_id",
        "regime_name",
        "best_bid",
        "best_ask",
        "mid_price",
        "spread",
        "bid_levels",
        "ask_levels",
    )
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots[:_SNAPSHOTS_SAMPLE_ROWS]:
        best_bid = snapshot.best_bid.price if snapshot.best_bid else ""
        best_ask = snapshot.best_ask.price if snapshot.best_ask else ""
        rows.append(
            {
                "sequence_id": snapshot.sequence_id,
                "regime_name": snapshot.metadata.get("regime_name", ""),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid_price": snapshot.mid_price if snapshot.mid_price is not None else "",
                "spread": snapshot.spread if snapshot.spread is not None else "",
                "bid_levels": len(snapshot.bids),
                "ask_levels": len(snapshot.asks),
            }
        )
    return _write_rows(path, header, rows)


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
