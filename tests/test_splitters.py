"""Tests for temporal and walk-forward splitters."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from chronoslob.training.splitters import (
    SplitIndices,
    TemporalSplitConfig,
    WalkForwardSplitConfig,
    temporal_train_validation_test_split,
    walk_forward_splits,
)


def test_temporal_split_fractions_produce_ordered_non_overlapping_indices() -> None:
    split = temporal_train_validation_test_split(100)

    assert split.n_train == 70
    assert split.n_validation == 15
    assert split.n_test == 15
    assert split.train == list(range(70))
    assert split.validation == list(range(70, 85))
    assert split.test == list(range(85, 100))
    split.assert_no_overlap()


def test_temporal_split_min_sizes_are_enforced() -> None:
    with pytest.raises(ValueError):
        temporal_train_validation_test_split(1)


def test_invalid_temporal_fractions_raise() -> None:
    with pytest.raises(ValidationError):
        TemporalSplitConfig(
            train_fraction=0.5,
            validation_fraction=0.4,
            test_fraction=0.2,
        )


def test_walk_forward_expanding_split() -> None:
    folds = walk_forward_splits(
        10,
        WalkForwardSplitConfig(
            train_window=4,
            validation_window=2,
            test_window=1,
            step_size=2,
            expanding_train=True,
        ),
    )

    assert len(folds) == 2
    assert folds[0].train == [0, 1, 2, 3]
    assert folds[0].validation == [4, 5]
    assert folds[0].test == [6]
    assert folds[1].train == [0, 1, 2, 3, 4, 5]
    assert folds[1].validation == [6, 7]
    assert folds[1].test == [8]


def test_walk_forward_rolling_split() -> None:
    folds = walk_forward_splits(
        10,
        WalkForwardSplitConfig(
            train_window=4,
            validation_window=2,
            step_size=2,
            expanding_train=False,
        ),
    )

    assert len(folds) == 3
    assert folds[0].train == [0, 1, 2, 3]
    assert folds[0].validation == [4, 5]
    assert folds[1].train == [2, 3, 4, 5]
    assert folds[1].validation == [6, 7]
    assert folds[2].train == [4, 5, 6, 7]
    assert folds[2].validation == [8, 9]


def test_walk_forward_incomplete_folds_are_skipped() -> None:
    folds = walk_forward_splits(
        8,
        WalkForwardSplitConfig(
            train_window=4,
            validation_window=3,
            step_size=3,
            expanding_train=False,
        ),
    )

    assert len(folds) == 1
    assert folds[0].train == [0, 1, 2, 3]
    assert folds[0].validation == [4, 5, 6]


def test_split_indices_detects_overlap() -> None:
    with pytest.raises(ValueError):
        SplitIndices(train=[0, 1], validation=[1, 2])


def test_split_indices_all_indices_returns_sorted_union() -> None:
    split = SplitIndices(train=[0, 2], validation=[1], test=[3])

    assert split.all_indices() == [0, 1, 2, 3]
    assert split.to_dict() == {
        "train": [0, 2],
        "validation": [1],
        "test": [3],
    }
