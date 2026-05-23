"""Tests for replaying canonical event logs into research frames."""

from __future__ import annotations

from pathlib import Path

import pytest

from chronoslob.book.event_replay import (
    replay_event_log_to_feature_frame,
    replay_event_log_to_feature_label_frames,
    replay_event_log_to_label_frame,
    snapshots_from_event_log_records,
    write_binance_reconstruction_to_event_log,
)
from chronoslob.book.replay import ReplayConfig, replay_binance_jsonl
from chronoslob.data.event_store import read_event_log_jsonl
from chronoslob.data.schemas import BookEvent, OrderBookSnapshot
from chronoslob.labels.pipeline import LabelPipelineConfig, validate_label_frame

FIXTURES = Path(__file__).parent / "fixtures"
EVENT_LOGS = FIXTURES / "event_logs"
BINANCE = FIXTURES / "binance"
SNAPSHOTS = EVENT_LOGS / "synthetic_snapshots.jsonl"
EVENTS = EVENT_LOGS / "synthetic_events.jsonl"
MIXED = EVENT_LOGS / "synthetic_mixed_log.jsonl"


def _small_label_config() -> LabelPipelineConfig:
    return LabelPipelineConfig(
        horizons=(1, 2),
        include_spread_widening=False,
        fill_horizon=1,
        adverse_evaluation_horizon=2,
    )


def test_snapshots_from_event_log_records_extracts_snapshots() -> None:
    records = read_event_log_jsonl(MIXED)

    snapshots = snapshots_from_event_log_records(records)

    assert len(snapshots) == 2
    assert all(isinstance(snapshot, OrderBookSnapshot) for snapshot in snapshots)


def test_snapshots_from_event_log_records_sort_records_true_sorts() -> None:
    records = read_event_log_jsonl(SNAPSHOTS)
    snapshots = [record for record in records if isinstance(record, OrderBookSnapshot)]
    reversed_records = list(reversed(snapshots[:2]))

    sorted_snapshots = snapshots_from_event_log_records(
        reversed_records,
        sort_records=True,
    )

    assert [snapshot.sequence_id for snapshot in sorted_snapshots] == [1, 2]


def test_no_snapshots_raises() -> None:
    records = read_event_log_jsonl(EVENTS)

    with pytest.raises(ValueError, match="no explicit order book snapshots"):
        snapshots_from_event_log_records(records)


def test_generic_book_event_only_reconstruction_is_clear_error() -> None:
    records = read_event_log_jsonl(EVENTS)

    assert all(isinstance(record, BookEvent) for record in records)
    with pytest.raises(ValueError, match="not implemented in Phase 9"):
        replay_event_log_to_feature_frame(records)


def test_replay_event_log_to_feature_frame_returns_rows() -> None:
    records = read_event_log_jsonl(SNAPSHOTS)

    frame = replay_event_log_to_feature_frame(records)

    assert len(frame) == 6
    assert {"timestamp", "symbol", "mid_price", "spread"}.issubset(frame.columns)


def test_replay_event_log_to_label_frame_returns_labels() -> None:
    records = read_event_log_jsonl(SNAPSHOTS)

    frame = replay_event_log_to_label_frame(
        records,
        label_config=_small_label_config(),
    )

    validation = validate_label_frame(frame)
    assert len(frame) == 4
    assert validation.ok
    assert "future_return_1" in frame.columns
    assert {"horizon_start", "horizon_end"}.issubset(frame.columns)


def test_replay_event_log_to_feature_label_frames_passes_leakage_checks() -> None:
    records = read_event_log_jsonl(SNAPSHOTS)

    feature_frame, label_frame = replay_event_log_to_feature_label_frames(
        records,
        label_config=_small_label_config(),
    )

    assert len(feature_frame) == 6
    assert len(label_frame) == 4


def test_write_binance_reconstruction_to_event_log_writes_snapshots(
    tmp_path: Path,
) -> None:
    result = replay_binance_jsonl(
        ReplayConfig(
            snapshot_path=BINANCE / "synthetic_snapshot.json",
            updates_path=BINANCE / "synthetic_diff_updates.jsonl",
        )
    )
    path = tmp_path / "binance_reconstruction_event_log.jsonl"

    written = write_binance_reconstruction_to_event_log(result, path)
    records = read_event_log_jsonl(written)

    assert len(records) == result.n_snapshots
    assert all(isinstance(record, OrderBookSnapshot) for record in records)
    assert [record.sequence_id for record in records] == [105, 110, 115]


def test_write_binance_reconstruction_to_event_log_rejects_empty_result(
    tmp_path: Path,
) -> None:
    result = replay_binance_jsonl(
        ReplayConfig(
            snapshot_path=BINANCE / "synthetic_snapshot.json",
            updates_path=BINANCE / "synthetic_crossed_updates.jsonl",
        )
    )

    with pytest.raises(ValueError, match="no snapshots"):
        write_binance_reconstruction_to_event_log(
            result,
            tmp_path / "empty.jsonl",
        )
