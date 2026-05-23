"""Tests for supervised multi-task token-window datasets."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from chronoslob.models.tokenisation import (  # noqa: E402
    TokenisationConfig,
    TokenisedRecord,
    TokenSequence,
    build_static_token_vocabulary,
)
from chronoslob.training.multitask_datasets import (  # noqa: E402
    MultiTaskLabelSpec,
    MultiTaskTokenDataset,
    MultiTaskWindowConfig,
    build_multitask_sample_indices,
    collate_multitask_token_windows,
)
from chronoslob.training.token_datasets import TOKEN_WINDOW_FIELD_NAMES  # noqa: E402


def _record(position: int, *, symbol: str = "A") -> TokenisedRecord:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return TokenisedRecord(
        event_type_id=5,
        side_id=6,
        price_bucket_id=7,
        quantity_bucket_id=8,
        time_delta_bucket_id=5,
        context_bucket_id=5,
        source_id=5,
        position=position,
        timestamp=base + timedelta(seconds=position),
        symbol=symbol,
        sequence_id=100 + position,
        record_kind="order_book_snapshot",
        token_role="snapshot_summary",
        source_record_index=position,
        is_snapshot_derived=True,
    )


def _sequence(symbols: tuple[str, ...] = ("A", "A", "A", "B", "B", "B")) -> TokenSequence:
    config = TokenisationConfig()
    records = tuple(_record(position, symbol=symbol) for position, symbol in enumerate(symbols))
    return TokenSequence(
        records=records,
        vocabulary=build_static_token_vocabulary(config),
        config=config,
        input_record_count=len(records),
    )


def _specs() -> tuple[MultiTaskLabelSpec, ...]:
    return (
        MultiTaskLabelSpec("direction", num_classes=3),
        MultiTaskLabelSpec("spread_widening", num_classes=2),
    )


def test_sample_index_construction_from_windows_and_labels() -> None:
    sequence = _sequence()
    label_table = {
        0: {"direction": 0, "spread_widening": 1},
        2: {"direction": 2, "spread_widening": None},
    }

    samples = build_multitask_sample_indices(
        sequence,
        MultiTaskWindowConfig(window_length=3),
        _specs(),
        label_table=label_table,
    )

    assert [sample.target_record_index for sample in samples] == [0, 2]
    assert samples[0].targets == {"direction": 0, "spread_widening": 1}
    assert samples[1].targets == {"direction": 2, "spread_widening": None}


def test_target_alignment_uses_token_window_end_index() -> None:
    sequence = _sequence()
    samples = build_multitask_sample_indices(
        sequence,
        MultiTaskWindowConfig(window_length=3),
        _specs(),
        label_table={3: {"direction": 1, "spread_widening": 0}},
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample.target_record_index == sample.window_index.window_end == 3


def test_timestamp_label_frame_uses_exact_origin_matching() -> None:
    sequence = _sequence()
    timestamp = sequence.records[2].timestamp
    frame = pd.DataFrame(
        {
            "timestamp": [timestamp],
            "symbol": ["A"],
            "direction": [2],
            "spread_widening": [True],
        }
    )

    samples = build_multitask_sample_indices(
        sequence,
        MultiTaskWindowConfig(window_length=2),
        _specs(),
        label_frame=frame,
    )

    assert len(samples) == 1
    assert samples[0].target_record_index == 2
    assert samples[0].targets["direction"] == 2
    assert samples[0].targets["spread_widening"] == 1


def test_samples_do_not_cross_symbol_boundaries() -> None:
    sequence = _sequence()
    label_table = {
        index: {"direction": 1, "spread_widening": 0}
        for index in range(len(sequence.records))
    }

    samples = build_multitask_sample_indices(
        sequence,
        MultiTaskWindowConfig(window_length=4, respect_symbol_boundaries=True),
        _specs(),
        label_table=label_table,
    )
    first_b = next(sample for sample in samples if sample.target_record_index == 3)

    assert first_b.window_index.window_start == 3


def test_samples_do_not_cross_split_boundaries_when_supported() -> None:
    sequence = _sequence(symbols=("A", "A", "A", "A"))
    label_table = {
        index: {"direction": 1, "spread_widening": 0}
        for index in range(len(sequence.records))
    }

    samples = build_multitask_sample_indices(
        sequence,
        MultiTaskWindowConfig(window_length=3, respect_split_boundaries=True),
        _specs(),
        label_table=label_table,
        split_ids=("train", "train", "validation", "validation"),
    )
    first_validation = next(sample for sample in samples if sample.target_record_index == 2)

    assert first_validation.window_index.window_start == 2


def test_missing_and_partially_missing_labels_are_handled() -> None:
    sequence = _sequence()
    samples = build_multitask_sample_indices(
        sequence,
        MultiTaskWindowConfig(window_length=2),
        _specs(),
        label_table={
            1: {"direction": None, "spread_widening": None},
            2: {"direction": 2, "spread_widening": None},
        },
    )

    assert len(samples) == 1
    assert samples[0].target_record_index == 2
    assert samples[0].targets == {"direction": 2, "spread_widening": None}


def test_all_missing_samples_can_be_preserved_when_configured() -> None:
    sequence = _sequence()
    samples = build_multitask_sample_indices(
        sequence,
        MultiTaskWindowConfig(window_length=2, drop_all_missing_samples=False),
        _specs(),
        label_table={1: {"direction": None, "spread_widening": None}},
    )

    assert len(samples) == 6
    assert any(not sample.has_any_target for sample in samples)


def test_dataset_item_shapes_dtypes_and_masks() -> None:
    sequence = _sequence()
    dataset = MultiTaskTokenDataset(
        sequence,
        MultiTaskWindowConfig(window_length=3),
        _specs(),
        label_table={2: {"direction": 2, "spread_widening": None}},
    )

    sample = dataset[0]

    for field_name in TOKEN_WINDOW_FIELD_NAMES:
        assert tuple(sample[field_name].shape) == (3,)
        assert sample[field_name].dtype == torch.long
    assert tuple(sample["attention_mask"].shape) == (3,)
    assert sample["attention_mask"].dtype == torch.bool
    assert sample["targets"]["direction"].item() == 2
    assert sample["targets"]["spread_widening"].item() == -100
    assert sample["target_mask"]["direction"].item() is True
    assert sample["target_mask"]["spread_widening"].item() is False


def test_collate_batches_token_fields_targets_and_masks() -> None:
    sequence = _sequence()
    dataset = MultiTaskTokenDataset(
        sequence,
        MultiTaskWindowConfig(window_length=3),
        _specs(),
        label_table={
            1: {"direction": 1, "spread_widening": 0},
            2: {"direction": 2, "spread_widening": None},
        },
    )

    batch = collate_multitask_token_windows([dataset[0], dataset[1]])

    for field_name in TOKEN_WINDOW_FIELD_NAMES:
        assert tuple(batch[field_name].shape) == (2, 3)
        assert batch[field_name].dtype == torch.long
    assert tuple(batch["attention_mask"].shape) == (2, 3)
    assert batch["attention_mask"].dtype == torch.bool
    assert tuple(batch["targets"]["direction"].shape) == (2,)
    assert batch["targets"]["direction"].dtype == torch.long
    assert batch["target_mask"]["spread_widening"].tolist() == [True, False]


def test_label_values_do_not_appear_in_token_fields() -> None:
    sequence = _sequence()
    specs = (MultiTaskLabelSpec("direction", num_classes=100),)
    dataset = MultiTaskTokenDataset(
        sequence,
        MultiTaskWindowConfig(window_length=2),
        specs,
        label_table={1: {"direction": 97}},
    )
    sample = dataset[0]

    assert sample["targets"]["direction"].item() == 97
    for field_name in TOKEN_WINDOW_FIELD_NAMES:
        assert 97 not in sample[field_name].tolist()
