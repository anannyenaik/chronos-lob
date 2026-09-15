"""Tests for event-log reproducibility manifests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from chronoslob.data.manifests import (
    EventLogManifest,
    create_event_log_manifest,
    read_manifest,
    sha256_file,
    write_manifest,
)

FIXTURES = Path(__file__).parent / "fixtures" / "event_logs"
SNAPSHOTS = FIXTURES / "synthetic_snapshots.jsonl"


def test_sha256_file_stable() -> None:
    first = sha256_file(SNAPSHOTS)
    second = sha256_file(SNAPSHOTS)

    assert first == second
    assert len(first) == 64


def test_create_event_log_manifest_counts_records() -> None:
    manifest = create_event_log_manifest(SNAPSHOTS)

    assert manifest.n_records == 6
    assert manifest.n_book_events == 0
    assert manifest.n_snapshots == 6


def test_manifest_symbols_sorted() -> None:
    manifest = create_event_log_manifest(SNAPSHOTS)

    assert manifest.symbols == ["TESTUSDT"]


def test_manifest_timestamp_range_correct() -> None:
    manifest = create_event_log_manifest(SNAPSHOTS)

    assert manifest.start_timestamp == datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    assert manifest.end_timestamp == datetime(2024, 1, 1, 0, 0, 5, tzinfo=UTC)


def test_manifest_sequence_range_correct() -> None:
    manifest = create_event_log_manifest(SNAPSHOTS)

    assert manifest.min_sequence_id == 1
    assert manifest.max_sequence_id == 6


def test_write_read_manifest_round_trip(tmp_path: Path) -> None:
    manifest = create_event_log_manifest(SNAPSHOTS, metadata={"phase": "phase-9"})
    path = tmp_path / "manifest.json"

    written = write_manifest(path, manifest)
    restored = read_manifest(written)

    assert restored == manifest
    assert restored.metadata == {"phase": "phase-9"}


def test_write_manifest_overwrite_false_raises(tmp_path: Path) -> None:
    manifest = create_event_log_manifest(SNAPSHOTS)
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)

    with pytest.raises(FileExistsError):
        write_manifest(path, manifest)


def test_manifest_metadata_validation_rejects_nested_metadata() -> None:
    with pytest.raises(ValueError, match="unsupported type"):
        create_event_log_manifest(
            SNAPSHOTS,
            metadata={"nested": {"not": "allowed"}},
        )


def test_manifest_created_at_must_be_timezone_aware() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        EventLogManifest(
            path="event_log.jsonl",
            n_records=0,
            n_book_events=0,
            n_snapshots=0,
            symbols=[],
            start_timestamp=None,
            end_timestamp=None,
            min_sequence_id=None,
            max_sequence_id=None,
            sha256="abc",
            created_at=datetime(2024, 1, 1),
        )
