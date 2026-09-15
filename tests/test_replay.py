"""Tests for deterministic local Binance-style replay helpers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from chronoslob.book.reconstruction import ReconstructionStatus
from chronoslob.book.replay import (
    ReplayConfig,
    replay_binance_jsonl,
    summarise_replay_result,
)

FIXTURES = Path(__file__).parent / "fixtures" / "binance"
SNAPSHOT = FIXTURES / "synthetic_snapshot.json"
UPDATES = FIXTURES / "synthetic_diff_updates.jsonl"
GAP_UPDATES = FIXTURES / "synthetic_gap_updates.jsonl"
CROSSED_UPDATES = FIXTURES / "synthetic_crossed_updates.jsonl"


def test_replay_binance_jsonl_valid_fixture() -> None:
    result = replay_binance_jsonl(
        ReplayConfig(
            snapshot_path=SNAPSHOT,
            updates_path=UPDATES,
            symbol="TESTUSDT",
        )
    )

    assert result.ok
    assert result.n_snapshots == 3
    assert result.final_update_id == 115


def test_replay_gap_fixture_reports_gap() -> None:
    result = replay_binance_jsonl(
        ReplayConfig(snapshot_path=SNAPSHOT, updates_path=GAP_UPDATES)
    )

    assert not result.ok
    assert result.gap_count == 1
    assert result.issues[0].status == ReconstructionStatus.GAP_DETECTED


def test_replay_crossed_fixture_reports_crossed_issue() -> None:
    result = replay_binance_jsonl(
        ReplayConfig(snapshot_path=SNAPSHOT, updates_path=CROSSED_UPDATES)
    )

    assert not result.ok
    assert result.crossed_count == 1
    assert result.issues[0].status == ReconstructionStatus.CROSSED_BOOK


def test_summarise_replay_result_keys() -> None:
    result = replay_binance_jsonl(
        ReplayConfig(snapshot_path=SNAPSHOT, updates_path=UPDATES)
    )

    summary = summarise_replay_result(result)

    assert set(summary) == {
        "ok",
        "n_snapshots",
        "final_update_id",
        "gap_count",
        "crossed_count",
        "issue_count",
    }
    assert summary["ok"] is True
    assert summary["n_snapshots"] == 3


def test_cli_inspect_binance_replay_command() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chronoslob.cli",
            "inspect-binance-replay",
            "--snapshot",
            str(SNAPSHOT),
            "--updates",
            str(UPDATES),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "synthetic fixture" in completed.stdout
    assert "ok:               True" in completed.stdout
    assert "n_snapshots:      3" in completed.stdout
    assert "network calls:    none performed" in completed.stdout


def test_replay_writes_no_files(tmp_path: Path) -> None:
    snapshot_copy = tmp_path / "snapshot.json"
    updates_copy = tmp_path / "updates.jsonl"
    snapshot_copy.write_text(SNAPSHOT.read_text(encoding="utf-8"), encoding="utf-8")
    updates_copy.write_text(UPDATES.read_text(encoding="utf-8"), encoding="utf-8")
    before = sorted(path.name for path in tmp_path.iterdir())

    result = replay_binance_jsonl(
        ReplayConfig(snapshot_path=snapshot_copy, updates_path=updates_copy)
    )

    after = sorted(path.name for path in tmp_path.iterdir())
    assert result.ok
    assert after == before
