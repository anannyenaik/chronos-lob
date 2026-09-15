"""Replay and synthetic-book command implementations."""

from __future__ import annotations

import sys
from pathlib import Path


def _run_synthetic_lob_benchmark_impl(
    *,
    out: Path,
    events_per_regime: int,
    seed: int,
    horizon: int,
    smoke: bool,
    make_figures: bool,
    overwrite: bool,
) -> int:
    """Run the synthetic event-level LOB pipeline and write artefacts."""
    from chronoslob.synthetic.events import SyntheticEventConfig, default_regime_plan
    from chronoslob.synthetic.pipeline import (
        SyntheticLobConfig,
        run_synthetic_lob_pipeline,
        smoke_config,
    )

    if smoke:
        config = smoke_config(events_per_regime=events_per_regime)
    else:
        config = SyntheticLobConfig(
            event_config=SyntheticEventConfig(
                seed=seed,
                regime_plan=default_regime_plan(events_per_regime),
            ),
            horizon=horizon,
            benchmark_seed=seed,
        )

    try:
        result = run_synthetic_lob_pipeline(
            Path(out),
            config,
            make_figures=make_figures,
            overwrite=overwrite,
        )
    except FileExistsError as exc:
        print(f"Refusing to overwrite: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Synthetic LOB pipeline failed: {exc}", file=sys.stderr)
        return 1

    chronological_test = [m for m in result.benchmark.chronological if m.split == "test"]
    print("ChronosLOB synthetic event-level extension")
    print(f"  output directory:   {result.out_dir}")
    print(f"  files written:      {len(result.files_written)}")
    print(f"  events generated:   {result.event_count}")
    print(f"  snapshots:          {result.snapshot_count}")
    print(f"  feature rows:       {result.feature_row_count}")
    print(f"  label rows:         {result.label_row_count}")
    print(f"  replay invariants:  {'ok' if result.replay_ok else 'violations recorded'}")
    print(f"  no-lookahead check: {'ok' if result.leakage_ok else 'violations recorded'}")
    print(f"  regimes:            {', '.join(result.summary['regimes'])}")
    for metric in chronological_test:
        print(
            f"    test {metric.model_name:>17}: "
            f"macro_f1={metric.macro_f1:.4f} accuracy={metric.accuracy:.4f}"
        )
    print("  note: synthetic controlled stress test; not real-market evidence.")
    print("  network calls:      none performed")
    return 0


def _replay_binance_l2_sample_impl(
    *,
    out: Path,
    snapshot: Path | None,
    updates: Path | None,
    symbol: str | None,
    max_depth: int | None,
    window_events: int,
    stop_on_gap: bool,
    allow_crossed: bool,
    make_figures: bool,
    overwrite: bool,
) -> int:
    """Replay a local Binance L2 snapshot-plus-diff sample and write artefacts.

    The command is offline: it reads local files only and makes no network
    calls. When no snapshot/diff paths are supplied it falls back to the small
    bundled Binance-shaped synthetic fixtures.
    """
    from chronoslob.binance_l2.pipeline import (
        BinanceL2Config,
        default_fixture_paths,
        run_binance_l2_pipeline,
    )

    default_snapshot, default_updates = default_fixture_paths()
    snapshot_path = Path(snapshot) if snapshot is not None else default_snapshot
    updates_path = Path(updates) if updates is not None else default_updates

    fixture_marker = str(Path("tests") / "fixtures")
    if any(fixture_marker in str(path) for path in (snapshot_path, updates_path)):
        print(
            "WARNING: replay is running against a Binance-shaped synthetic fixture; "
            "outputs are not real market data."
        )

    try:
        config = BinanceL2Config(
            snapshot_path=snapshot_path,
            updates_path=updates_path,
            symbol=symbol,
            max_depth=max_depth,
            window_events=window_events,
            stop_on_gap=stop_on_gap,
            allow_crossed=allow_crossed,
        )
        result = run_binance_l2_pipeline(
            Path(out),
            config,
            make_figures=make_figures,
            overwrite=overwrite,
        )
    except FileNotFoundError as exc:
        print(f"File not found: {exc}", file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(f"Refusing to overwrite: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Binance L2 replay extension failed: {exc}", file=sys.stderr)
        return 1

    print("ChronosLOB Binance Spot aggregated L2 replay (offline)")
    print(f"  output directory:   {result.out_dir}")
    print(f"  files written:      {len(result.files_written)}")
    print(f"  snapshot path:      {snapshot_path}")
    print(f"  updates path:       {updates_path}")
    print(f"  diff events:        {result.diff_event_count}")
    print(f"  applied events:     {result.applied_event_count}")
    print(f"  snapshots:          {result.snapshot_count}")
    print(f"  feature rows:       {result.feature_row_count}")
    print(f"  replay invariants:  {'ok' if result.replay_ok else 'violations recorded'}")
    print(f"  evidence_level:     {result.summary['evidence_level']}")
    print(
        "  note: aggregated L2 diff-depth replay; crypto-market engineering "
        "evidence, not equity, not live trading, not profitability evidence."
    )
    print("  network calls:      none performed")
    return 0


def _inspect_binance_replay_impl(
    snapshot_path: Path,
    updates_path: Path,
    *,
    symbol: str | None = None,
    max_depth: int | None = None,
    stop_on_gap: bool = True,
    allow_crossed: bool = False,
) -> int:
    """Replay a local Binance-style snapshot and diff fixture.

    The command is intentionally read-only. It loads files from disk only,
    runs the reconstruction and prints a short summary. It writes no
    outputs and makes no network calls.
    """
    from chronoslob.book.replay import (
        ReplayConfig,
        replay_binance_jsonl,
        summarise_replay_result,
    )

    snapshot_path = Path(snapshot_path)
    updates_path = Path(updates_path)

    fixture_marker = Path("tests") / "fixtures"
    fixture_marker_str = str(fixture_marker)
    is_fixture = any(fixture_marker_str in str(path) for path in (snapshot_path, updates_path))
    if is_fixture:
        print(
            "WARNING: replay is running against a synthetic fixture; "
            "outputs are not real market data."
        )

    config = ReplayConfig(
        snapshot_path=snapshot_path,
        updates_path=updates_path,
        symbol=symbol,
        max_depth=max_depth,
        stop_on_gap=stop_on_gap,
        allow_crossed=allow_crossed,
    )
    result = replay_binance_jsonl(config)
    summary = summarise_replay_result(result)

    print("ChronosLOB inspect-binance-replay (offline only)")
    print(f"  snapshot path:    {snapshot_path}")
    print(f"  updates path:     {updates_path}")
    print(f"  ok:               {summary['ok']}")
    print(f"  n_snapshots:      {summary['n_snapshots']}")
    print(f"  final_update_id:  {summary['final_update_id']}")
    print(f"  issue_count:      {summary['issue_count']}")
    print(f"  gap_count:        {summary['gap_count']}")
    print(f"  crossed_count:    {summary['crossed_count']}")

    if result.snapshots:
        final_snapshot = result.snapshots[-1]
        best_bid = final_snapshot.best_bid
        best_ask = final_snapshot.best_ask
        bid_str = f"{best_bid.price}@{best_bid.quantity}" if best_bid is not None else "n/a"
        ask_str = f"{best_ask.price}@{best_ask.quantity}" if best_ask is not None else "n/a"
        print(f"  best bid:         {bid_str}")
        print(f"  best ask:         {ask_str}")
        bid_levels = len(final_snapshot.bids)
        ask_levels = len(final_snapshot.asks)
        print(f"  depth counts:     bids={bid_levels}, asks={ask_levels}")
    else:
        print("  best bid:         not available (no snapshots emitted)")
        print("  best ask:         not available (no snapshots emitted)")

    print("  outputs:          not written (read-only command)")
    print("  network calls:    none performed")

    return 0 if result.ok else 1
