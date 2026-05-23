"""Tests for purged and embargoed split behaviour."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from chronoslob.data.schemas import LabelRow
from chronoslob.training.splitters import (
    PurgedEmbargoConfig,
    SplitIndices,
    apply_purge_and_embargo,
    apply_row_embargo,
    intervals_overlap,
    label_horizon_end_indices_from_rows,
    make_label_horizon_end_indices_from_frame,
    purge_overlapping_train_indices,
)

T0 = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)


def _time(offset: int) -> datetime:
    return T0 + timedelta(seconds=offset)


def _label_row(index: int, horizon_end_index: int) -> LabelRow:
    return LabelRow(
        timestamp=_time(index),
        symbol="TEST",
        labels={"future_return": 0.0},
        horizon_start=_time(index),
        horizon_end=_time(horizon_end_index),
    )


def test_intervals_overlap_closed_semantics() -> None:
    assert intervals_overlap(0, 2, 2, 4)
    assert not intervals_overlap(0, 2, 2, 4, closed=False)


def test_purging_removes_train_rows_whose_label_horizons_overlap_validation() -> None:
    kept = purge_overlapping_train_indices(
        train_indices=[0, 1, 2, 3],
        evaluation_indices=[4, 5],
        label_horizon_end_by_index={0: 0, 1: 3, 2: 4, 3: 6},
    )

    assert kept == [0, 1]


def test_purging_preserves_non_overlapping_train_rows() -> None:
    kept = purge_overlapping_train_indices(
        train_indices=[0, 1, 2],
        evaluation_indices=[5, 6],
        label_horizon_end_by_index={0: 0, 1: 2, 2: 4},
    )

    assert kept == [0, 1, 2]


def test_missing_horizon_metadata_raises() -> None:
    with pytest.raises(KeyError):
        purge_overlapping_train_indices(
            train_indices=[0, 1],
            evaluation_indices=[2],
            label_horizon_end_by_index={0: 1},
        )


def test_row_embargo_removes_rows_around_validation_block() -> None:
    embargoed = apply_row_embargo(
        train_indices=[0, 1, 2, 3, 8, 9, 10],
        evaluation_indices=[5, 6],
        embargo=2,
    )

    assert embargoed == [0, 1, 2, 9, 10]


def test_apply_purge_and_embargo_validation_only() -> None:
    split = SplitIndices(train=[0, 1, 2, 3, 4], validation=[5, 6], test=[7, 8])
    result = apply_purge_and_embargo(
        split,
        label_horizon_end_by_index={0: 0, 1: 1, 2: 2, 3: 3, 4: 5},
        config=PurgedEmbargoConfig(purge=True, embargo=1),
        purge_against="validation",
    )

    assert result.train == [0, 1, 2, 3]
    assert result.validation == split.validation
    assert result.test == split.test


def test_apply_purge_and_embargo_validation_test() -> None:
    split = SplitIndices(train=[0, 1, 2, 3, 4, 5], validation=[6], test=[7, 8])
    result = apply_purge_and_embargo(
        split,
        label_horizon_end_by_index={0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6},
        config=PurgedEmbargoConfig(purge=True, embargo=0),
        purge_against="validation_test",
    )

    assert result.train == [0, 1, 2, 3, 4]


def test_label_horizon_end_indices_from_rows_maps_label_rows() -> None:
    rows = [_label_row(0, 2), _label_row(1, 3)]
    timestamp_to_index = {_time(index): index for index in range(4)}

    mapped = label_horizon_end_indices_from_rows(rows, timestamp_to_index)

    assert mapped == {0: 2, 1: 3}


def test_make_label_horizon_end_indices_from_frame_maps_timestamps() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [_time(0), _time(1), _time(2), _time(3)],
            "horizon_end": [_time(2), _time(3), _time(3), _time(3)],
        }
    )

    mapped = make_label_horizon_end_indices_from_frame(frame)

    assert mapped == {0: 2, 1: 3, 2: 3, 3: 3}


def test_non_monotonic_label_frame_timestamps_raise() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [_time(1), _time(0)],
            "horizon_end": [_time(1), _time(1)],
        }
    )

    with pytest.raises(ValueError):
        make_label_horizon_end_indices_from_frame(frame)
