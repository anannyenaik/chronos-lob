"""Tests for the synthetic event-level LOB extension."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from chronoslob.data.schemas import (
    BookEvent,
    EventType,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
)
from chronoslob.synthetic.benchmark import run_synthetic_benchmark
from chronoslob.synthetic.events import (
    SyntheticEventConfig,
    default_regime_plan,
    generate_synthetic_events,
)
from chronoslob.synthetic.features import build_event_feature_frame
from chronoslob.synthetic.labels import build_label_frame, validate_no_lookahead_frames
from chronoslob.synthetic.pipeline import (
    SyntheticLobConfig,
    run_synthetic_lob_pipeline,
    smoke_config,
)
from chronoslob.synthetic.replay import replay_events_to_snapshots

_EPOCH = datetime(2020, 1, 1, tzinfo=UTC)


def _tiny_config(events_per_regime: int = 80) -> SyntheticEventConfig:
    return SyntheticEventConfig(
        seed=3,
        regime_plan=default_regime_plan(events_per_regime),
        max_levels_per_side=6,
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_generation_is_deterministic_for_a_seed() -> None:
    config = _tiny_config()
    first = generate_synthetic_events(config)
    second = generate_synthetic_events(config)
    assert first.event_count == second.event_count
    assert [event.model_dump() for event in first.events] == [
        event.model_dump() for event in second.events
    ]


def test_generation_differs_across_seeds() -> None:
    a = generate_synthetic_events(SyntheticEventConfig(seed=1, regime_plan=default_regime_plan(60)))
    b = generate_synthetic_events(SyntheticEventConfig(seed=2, regime_plan=default_regime_plan(60)))
    assert [e.model_dump() for e in a.events] != [e.model_dump() for e in b.events]


def test_generated_events_satisfy_schema_invariants() -> None:
    result = generate_synthetic_events(_tiny_config())
    previous = -1
    for event in result.events:
        assert event.sequence_id == previous + 1
        previous = event.sequence_id
        assert event.quantity is not None and event.quantity >= 0.0
        assert event.event_type in {EventType.ADD, EventType.CANCEL, EventType.TRADE}
        assert event.metadata["synthetic"] is True
        assert "regime_name" in event.metadata
    # Every planned regime should appear in the stream.
    seen = {str(event.metadata["regime_name"]) for event in result.events}
    for name, _ in result.config.regime_plan:
        assert name in seen


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def test_replay_invariants_hold_for_full_stream() -> None:
    config = _tiny_config(120)
    result = generate_synthetic_events(config)
    replay = replay_events_to_snapshots(
        result.events,
        tick_size=config.tick_size,
        max_levels_per_side=config.max_levels_per_side,
        snapshot_interval=4,
    )
    quality = replay.quality
    assert quality.ok
    assert quality.crossed_snapshot_count == 0
    assert quality.negative_depth_event_count == 0
    assert quality.sequence_gap_count == 0
    assert replay.snapshots
    for snapshot in replay.snapshots:
        if snapshot.best_bid is not None and snapshot.best_ask is not None:
            assert snapshot.best_bid.price < snapshot.best_ask.price


def test_replay_flags_sequence_gap() -> None:
    events = [
        BookEvent(
            timestamp=_EPOCH,
            event_type=EventType.ADD,
            symbol="SYNTH",
            side=Side.BID,
            price=100.0,
            quantity=5.0,
            sequence_id=0,
            metadata={"synthetic": True},
        ),
        BookEvent(
            timestamp=_EPOCH + timedelta(milliseconds=5),
            event_type=EventType.ADD,
            symbol="SYNTH",
            side=Side.ASK,
            price=101.0,
            quantity=5.0,
            sequence_id=5,
            metadata={"synthetic": True},
        ),
    ]
    replay = replay_events_to_snapshots(events, tick_size=0.01, snapshot_interval=1)
    assert replay.quality.sequence_gap_count == 1
    assert not replay.quality.ok


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def _hand_built_stream() -> tuple[list[BookEvent], OrderBookSnapshot]:
    events = [
        _event(0, EventType.ADD, Side.BID, 100.0, 10.0),
        _event(1, EventType.ADD, Side.BID, 99.0, 6.0),
        _event(2, EventType.CANCEL, Side.ASK, 101.0, 4.0),
        _event(3, EventType.TRADE, Side.BID, 101.0, 5.0, aggressor="BID"),
    ]
    snapshot = OrderBookSnapshot(
        timestamp=_EPOCH + timedelta(milliseconds=3),
        symbol="SYNTH",
        venue="synthetic",
        bids=[OrderBookLevel(price=100.0, quantity=10.0)],
        asks=[OrderBookLevel(price=101.0, quantity=4.0)],
        sequence_id=3,
        metadata={"synthetic": True, "regime_id": 0, "regime_name": "stable_liquid"},
    )
    return events, snapshot


def _event(
    seq: int,
    event_type: EventType,
    side: Side,
    price: float,
    quantity: float,
    *,
    aggressor: str | None = None,
) -> BookEvent:
    metadata: dict[str, str | int | float | bool] = {
        "synthetic": True,
        "regime_id": 0,
        "regime_name": "stable_liquid",
        "latent_mid": 100.5,
    }
    if aggressor is not None:
        metadata["aggressor_side"] = aggressor
    return BookEvent(
        timestamp=_EPOCH + timedelta(milliseconds=seq),
        event_type=event_type,
        symbol="SYNTH",
        side=side,
        price=price,
        quantity=quantity,
        sequence_id=seq,
        metadata=metadata,
    )


def test_event_level_features_are_correct_on_hand_built_stream() -> None:
    events, snapshot = _hand_built_stream()
    frame = build_event_feature_frame(events, [snapshot], window_events=10)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["add_rate"] == pytest.approx(0.5)
    assert row["cancel_rate"] == pytest.approx(0.25)
    assert row["trade_rate"] == pytest.approx(0.25)
    # All adds on the bid plus an ask cancellation are buying pressure.
    assert row["event_order_flow_imbalance"] == pytest.approx(1.0)
    # Only ask-side cancellations occurred.
    assert row["cancellation_imbalance"] == pytest.approx(-1.0)
    # Only a buyer-initiated trade occurred.
    assert row["trade_imbalance"] == pytest.approx(1.0)
    assert row["spread"] == pytest.approx(1.0)
    assert row["depth_imbalance_l1"] == pytest.approx((10.0 - 4.0) / 14.0)


# ---------------------------------------------------------------------------
# Labels and leakage
# ---------------------------------------------------------------------------


def test_labels_have_no_lookahead() -> None:
    config = _tiny_config(120)
    result = generate_synthetic_events(config)
    replay = replay_events_to_snapshots(
        result.events, tick_size=config.tick_size, snapshot_interval=4
    )
    features = build_event_feature_frame(result.events, replay.snapshots, window_events=30)
    labels = build_label_frame(features, horizon=10)
    assert not labels.empty
    leakage = validate_no_lookahead_frames(features, labels)
    assert leakage.ok
    # Every label must reference a strictly future snapshot.
    for _, row in labels.iterrows():
        assert row["future_timestamp"] > row["feature_timestamp"]


def test_label_frame_drops_final_horizon_rows() -> None:
    config = _tiny_config(120)
    result = generate_synthetic_events(config)
    replay = replay_events_to_snapshots(
        result.events, tick_size=config.tick_size, snapshot_interval=4
    )
    features = build_event_feature_frame(result.events, replay.snapshots, window_events=30)
    horizon = 12
    labels = build_label_frame(features, horizon=horizon)
    assert len(labels) == len(features) - horizon


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def test_benchmark_runner_smoke() -> None:
    config = _tiny_config(160)
    result = generate_synthetic_events(config)
    replay = replay_events_to_snapshots(
        result.events, tick_size=config.tick_size, snapshot_interval=4
    )
    features = build_event_feature_frame(result.events, replay.snapshots, window_events=30)
    labels = build_label_frame(features, horizon=10)
    benchmark = run_synthetic_benchmark(features, labels, seed=0)
    model_names = {metric.model_name for metric in benchmark.chronological}
    assert {"majority", "logistic", "ridge", "gradient_boosting"} <= model_names
    for metric in benchmark.chronological:
        assert 0.0 <= metric.accuracy <= 1.0
        assert -1.0 <= metric.mcc <= 1.0
    assert benchmark.regime_holdout  # held-out regimes are present in the plan


# ---------------------------------------------------------------------------
# Pipeline + artefacts
# ---------------------------------------------------------------------------


def test_pipeline_writes_all_required_artefacts(tmp_path: Path) -> None:
    out = tmp_path / "synthetic_lob"
    result = run_synthetic_lob_pipeline(out, smoke_config(events_per_regime=120), overwrite=True)
    assert result.replay_ok
    assert result.leakage_ok
    required = {
        "synthetic_lob_report.md",
        "summary.json",
        "synthetic_data_summary.json",
        "synthetic_replay_quality.json",
        "synthetic_feature_summary.csv",
        "synthetic_label_summary.csv",
        "synthetic_benchmark_summary.csv",
        "synthetic_regime_diagnostics.csv",
        "synthetic_claim_assessment.json",
        "figure_manifest.json",
    }
    written = {path.name for path in out.iterdir()}
    assert required <= written
    report = (out / "synthetic_lob_report.md").read_text(encoding="utf-8")
    assert "Synthetic Event-Level Extension" in report
    assert "not real-market evidence" in report.lower()


def test_pipeline_refuses_to_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "synthetic_lob"
    run_synthetic_lob_pipeline(out, smoke_config(events_per_regime=120), overwrite=True)
    with pytest.raises(FileExistsError):
        run_synthetic_lob_pipeline(out, smoke_config(events_per_regime=120), overwrite=False)


def test_pipeline_claim_assessment_blocks_forbidden_claims(tmp_path: Path) -> None:
    out = tmp_path / "synthetic_lob"
    run_synthetic_lob_pipeline(out, smoke_config(events_per_regime=120), overwrite=True)
    payload = json.loads((out / "synthetic_claim_assessment.json").read_text(encoding="utf-8"))
    claims = payload["claims"]
    assert claims["real_market_event_level_generalisation"]["status"] == "unsupported"
    assert claims["synthetic_to_real_transfer"]["status"] == "unsupported"
    assert claims["live_trading_or_profitability"]["status"] == "forbidden"
    assert claims["fi2010_true_event_level_ofi"]["status"] == "forbidden"


# ---------------------------------------------------------------------------
# Evidence pack + final report integration
# ---------------------------------------------------------------------------


def test_evidence_pack_classifies_synthetic_extension(tmp_path: Path) -> None:
    from chronoslob.experiments.evidence_pack import (
        EvidencePackConfig,
        audit_claims,
        discover_artefacts,
    )

    syn = tmp_path / "synthetic_lob_extension"
    config = SyntheticLobConfig(
        event_config=SyntheticEventConfig(regime_plan=default_regime_plan(200)),
        horizon=12,
    )
    run_synthetic_lob_pipeline(syn, config, overwrite=True)

    pack_config = EvidencePackConfig(
        out_dir=tmp_path / "pack",
        synthetic_lob_dir=syn,
        feature_audit_dir=None,
        project_audit_dir=None,
        strict=False,
    )
    records = discover_artefacts(pack_config)
    record = next(r for r in records if r.artefact_name == "synthetic_lob_extension_report")
    assert record.status == "complete_real"

    by_id = {claim.claim_id: claim for claim in audit_claims(records)}
    assert by_id["synthetic.event_level_pipeline"].status == "supported"
    assert by_id["synthetic.event_level_features"].status == "supported"
    assert by_id["synthetic.regime_diagnostics"].status == "supported"
    assert by_id["synthetic.real_market_event_level_generalisation"].status == "unsupported"
    assert by_id["synthetic.live_trading_or_profitability"].status == "forbidden"
    assert by_id["synthetic.fi2010_true_event_level_ofi"].status == "forbidden"


def test_final_report_includes_synthetic_section(tmp_path: Path) -> None:
    from chronoslob.experiments.final_report import build_final_empirical_report

    syn = tmp_path / "synthetic_lob_extension"
    run_synthetic_lob_pipeline(syn, smoke_config(events_per_regime=150), overwrite=True)

    classical = tmp_path / "classical"
    neural = tmp_path / "neural"
    uncertainty = tmp_path / "uncertainty"
    _write_json(classical / "summary.json", {"dataset_name": "FI-2010", "models_requested": ["x"]})
    _write_csv(
        classical / "results_summary.csv",
        ["model_name", "split", "macro_f1_mean"],
        [{"model_name": "x", "split": "test", "macro_f1_mean": 0.3}],
    )
    _write_json(neural / "summary.json", {"max_epochs": 1, "models_requested": ["m"]})
    _write_csv(
        neural / "results_summary.csv",
        ["model_name", "lookback", "split", "macro_f1_mean", "seed_count"],
        [
            {
                "model_name": "m",
                "lookback": 20,
                "split": "test",
                "macro_f1_mean": 0.5,
                "seed_count": 1,
            }
        ],
    )
    _write_json(uncertainty / "summary.json", {"classical": {}, "neural": {}})
    _write_csv(uncertainty / "metric_confidence_intervals.csv", ["source", "metric", "mean"], [])
    _write_csv(uncertainty / "model_ranking.csv", ["source", "rank", "model_name", "mean"], [])

    out = tmp_path / "final.md"
    build_final_empirical_report(
        classical_dir=classical,
        neural_dir=neural,
        uncertainty_dir=uncertainty,
        synthetic_lob_dir=syn,
        out_path=out,
        overwrite=True,
    )
    text = out.read_text(encoding="utf-8")
    assert "## Synthetic Event-Level Extension" in text
    assert "does not provide real-market evidence" in text
    assert "does not change FI-2010 limitations" in text


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, header: list[str], rows: list[dict[str, object]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
