"""End-to-end tests for replay-to-feature and CLI inspection paths."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from chronoslob.book.event_replay import (
    replay_event_log_to_feature_frame,
    replay_event_log_to_label_frame,
)
from chronoslob.data.event_store import read_event_log_jsonl
from chronoslob.features.pipeline import validate_feature_frame
from chronoslob.labels.pipeline import LabelPipelineConfig, validate_label_frame

FIXTURES = Path(__file__).parent / "fixtures" / "event_logs"
SNAPSHOTS = FIXTURES / "synthetic_snapshots.jsonl"


def _small_label_config() -> LabelPipelineConfig:
    return LabelPipelineConfig(
        horizons=(1, 2),
        include_spread_widening=False,
        fill_horizon=1,
        adverse_evaluation_horizon=2,
    )


def test_fixture_synthetic_snapshots_loads() -> None:
    records = read_event_log_jsonl(SNAPSHOTS)

    assert len(records) == 6
    assert {record.symbol for record in records} == {"TESTUSDT"}


def test_event_log_to_feature_path_works() -> None:
    records = read_event_log_jsonl(SNAPSHOTS)

    frame = replay_event_log_to_feature_frame(records)
    validation = validate_feature_frame(frame)

    assert len(frame) == 6
    assert validation.ok


def test_features_contain_no_label_like_columns() -> None:
    records = read_event_log_jsonl(SNAPSHOTS)
    feature_frame = replay_event_log_to_feature_frame(records)

    feature_columns = {
        str(column)
        for column in feature_frame.columns
        if str(column) not in {"timestamp", "symbol", "split"}
    }

    assert not any(
        column.lower().startswith(("label", "y_", "future_", "target"))
        for column in feature_columns
    )


def test_labels_contain_no_feature_like_columns() -> None:
    records = read_event_log_jsonl(SNAPSHOTS)
    label_frame = replay_event_log_to_label_frame(
        records,
        label_config=_small_label_config(),
    )

    validation = validate_label_frame(label_frame)

    assert validation.ok


def test_feature_label_frames_align_on_timestamp_and_symbol() -> None:
    records = read_event_log_jsonl(SNAPSHOTS)
    feature_frame = replay_event_log_to_feature_frame(records)
    label_frame = replay_event_log_to_label_frame(
        records,
        label_config=_small_label_config(),
    )

    merged = label_frame.merge(
        feature_frame.loc[:, ["timestamp", "symbol"]],
        on=["timestamp", "symbol"],
        how="left",
        indicator=True,
    )

    assert set(merged["_merge"]) == {"both"}


def test_cli_inspect_event_log_subprocess() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chronoslob.cli",
            "inspect-event-log",
            "--path",
            str(SNAPSHOTS),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "synthetic fixture" in completed.stdout
    assert "records:          6" in completed.stdout
    assert "snapshots:        6" in completed.stdout
    assert "sha256 prefix:" in completed.stdout


def test_cli_event_log_to_features_subprocess() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chronoslob.cli",
            "event-log-to-features",
            "--path",
            str(SNAPSHOTS),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "synthetic fixture" in completed.stdout
    assert "rows:                6" in completed.stdout
    assert "validation ok:       True" in completed.stdout
    assert "outputs:             not written" in completed.stdout


def test_cli_inspection_commands_write_no_files() -> None:
    before = sorted(path.name for path in FIXTURES.iterdir())

    for command in ("inspect-event-log", "event-log-to-features"):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "chronoslob.cli",
                command,
                "--path",
                str(SNAPSHOTS),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr

    after = sorted(path.name for path in FIXTURES.iterdir())
    assert after == before
