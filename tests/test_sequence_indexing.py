"""Tests for past-only sequence-window indexing."""

from __future__ import annotations

import pytest

from chronoslob.training.datasets import (
    SequenceSampleIndex,
    SequenceWindowConfig,
    build_sequence_indices,
)


def _config(lookback: int = 3, **overrides: object) -> SequenceWindowConfig:
    kwargs = {
        "lookback": lookback,
        "target_column": "label",
    }
    kwargs.update(overrides)
    return SequenceWindowConfig(**kwargs)  # type: ignore[arg-type]


def test_build_sequence_indices_basic_lookback() -> None:
    indices = build_sequence_indices(n_rows=5, config=_config(lookback=3))

    assert [sample.target_index for sample in indices] == [2, 3, 4]
    assert [sample.window_start for sample in indices] == [0, 1, 2]
    assert [sample.window_end for sample in indices] == [2, 3, 4]


def test_first_target_index_equals_lookback_minus_one() -> None:
    indices = build_sequence_indices(n_rows=6, config=_config(lookback=4))

    assert indices[0].target_index == 3
    assert indices[0].window_start == 0


def test_window_end_equals_target_index() -> None:
    indices = build_sequence_indices(n_rows=10, config=_config(lookback=2))

    for sample in indices:
        assert sample.window_end == sample.target_index


def test_window_indices_never_exceed_target_index() -> None:
    indices = build_sequence_indices(n_rows=8, config=_config(lookback=3))

    for sample in indices:
        for row in range(sample.window_start, sample.window_end + 1):
            assert row <= sample.target_index


def test_stride_skips_candidate_targets() -> None:
    indices = build_sequence_indices(n_rows=10, config=_config(lookback=2, stride=3))

    assert [sample.target_index for sample in indices] == [1, 4, 7]


def test_allowed_target_indices_restricts_samples() -> None:
    config = _config(lookback=2, require_contiguous_indices=False)

    indices = build_sequence_indices(
        n_rows=10,
        config=config,
        allowed_target_indices=[3, 5, 7],
    )

    assert [sample.target_index for sample in indices] == [3, 5, 7]


def test_require_contiguous_indices_blocks_cross_partition_windows() -> None:
    config = _config(lookback=3, require_contiguous_indices=True)

    # Two contiguous blocks; only target=4 has rows {2,3,4} fully inside the
    # allowed set. target=8 would pull rows 6 and 7 from outside the block.
    indices = build_sequence_indices(
        n_rows=10,
        config=config,
        allowed_target_indices=[2, 3, 4, 8, 9],
    )

    assert [sample.target_index for sample in indices] == [4]


def test_require_contiguous_indices_allows_full_block() -> None:
    config = _config(lookback=2, require_contiguous_indices=True)

    indices = build_sequence_indices(
        n_rows=10,
        config=config,
        allowed_target_indices=[2, 3, 4, 5],
    )

    assert [sample.target_index for sample in indices] == [3, 4, 5]


def test_insufficient_rows_returns_no_samples_when_dropping_incomplete() -> None:
    indices = build_sequence_indices(n_rows=2, config=_config(lookback=5))

    assert indices == []


def test_insufficient_rows_raises_when_not_dropping_incomplete() -> None:
    with pytest.raises(ValueError, match="lookback"):
        build_sequence_indices(
            n_rows=2,
            config=_config(lookback=5, drop_incomplete=False),
        )


def test_zero_rows_returns_empty_list() -> None:
    assert build_sequence_indices(n_rows=0, config=_config(lookback=2)) == []


def test_sequence_sample_index_validation_window_alignment() -> None:
    with pytest.raises(ValueError, match="window_end"):
        SequenceSampleIndex(window_start=0, window_end=3, target_index=4)


def test_sequence_sample_index_validation_negative_rows() -> None:
    with pytest.raises(ValueError):
        SequenceSampleIndex(window_start=-1, window_end=0, target_index=0)


def test_sequence_sample_index_window_start_after_window_end_raises() -> None:
    with pytest.raises(ValueError, match="window_start"):
        SequenceSampleIndex(window_start=5, window_end=3, target_index=3)


def test_sequence_window_config_rejects_non_positive_lookback() -> None:
    with pytest.raises(ValueError):
        SequenceWindowConfig(lookback=0, target_column="label")


def test_sequence_window_config_rejects_empty_target_column() -> None:
    with pytest.raises(ValueError):
        SequenceWindowConfig(lookback=2, target_column="   ")


def test_allowed_target_indices_out_of_range_raises() -> None:
    with pytest.raises(IndexError):
        build_sequence_indices(
            n_rows=5,
            config=_config(lookback=2),
            allowed_target_indices=[3, 10],
        )


def test_sample_indices_sorted_and_deterministic() -> None:
    config = _config(lookback=2, require_contiguous_indices=False)
    first = build_sequence_indices(
        n_rows=8, config=config, allowed_target_indices=[5, 3, 7]
    )
    second = build_sequence_indices(
        n_rows=8, config=config, allowed_target_indices=[7, 5, 3]
    )

    assert [sample.target_index for sample in first] == [3, 5, 7]
    assert first == second
