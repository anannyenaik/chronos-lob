"""Tests for token-window dataset preparation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

torch = pytest.importorskip("torch")

from chronoslob.data.schemas import BookEvent, EventType, Side  # noqa: E402
from chronoslob.models.tokenisation import (  # noqa: E402
    SPECIAL_TOKEN_IDS,
    SpecialToken,
    TokenisationConfig,
    TokenSequence,
    fit_tokenisation_state,
    tokenise_records,
)
from chronoslob.training.token_datasets import (  # noqa: E402
    TOKEN_WINDOW_FIELD_NAMES,
    TokenSequenceDataset,
    TokenWindowConfig,
    build_token_window_indices,
)


def _event(position: int, *, symbol: str = "TESTUSDT") -> BookEvent:
    return BookEvent(
        timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=position),
        event_type=EventType.ADD,
        symbol=symbol,
        side=Side.BID,
        price=100.0 + position,
        quantity=1.0,
        sequence_id=position,
        metadata={"source": "synthetic_train"},
    )


def _sequence(symbols: list[str] | None = None) -> TokenSequence:
    if symbols is None:
        symbols = ["TESTUSDT", "TESTUSDT", "TESTUSDT", "TESTUSDT"]
    records = [_event(index, symbol=symbol) for index, symbol in enumerate(symbols)]
    config = TokenisationConfig()
    state = fit_tokenisation_state(records, config)
    return tokenise_records(records, state.vocabulary, config)


def test_build_token_window_indices() -> None:
    sequence = _sequence()
    indices = build_token_window_indices(
        sequence,
        TokenWindowConfig(window_length=3, drop_incomplete=True),
    )

    assert [(index.window_start, index.window_end) for index in indices] == [
        (0, 2),
        (1, 3),
    ]


def test_token_windows_do_not_cross_symbol_boundaries() -> None:
    sequence = _sequence(["A", "A", "B", "A"])
    indices = build_token_window_indices(
        sequence,
        TokenWindowConfig(window_length=3),
    )

    assert [(index.window_start, index.window_end) for index in indices] == [
        (0, 0),
        (0, 1),
        (2, 2),
        (3, 3),
    ]


def test_token_windows_do_not_cross_split_boundaries() -> None:
    sequence = _sequence()
    split_ids = ["train", "train", "validation", "validation"]
    indices = build_token_window_indices(
        sequence,
        TokenWindowConfig(window_length=3),
        split_ids=split_ids,
    )

    assert [(index.window_start, index.window_end) for index in indices] == [
        (0, 0),
        (0, 1),
        (2, 2),
        (2, 3),
    ]


def test_padding_behaviour_and_attention_mask() -> None:
    sequence = _sequence()
    dataset = TokenSequenceDataset(
        sequence,
        TokenWindowConfig(window_length=4, padding_side="left"),
    )

    sample = dataset[0]
    pad_id = SPECIAL_TOKEN_IDS[SpecialToken.PAD]

    assert sample["event_type"].tolist() == [
        pad_id,
        pad_id,
        pad_id,
        sequence.records[0].event_type_id,
    ]
    assert sample["attention_mask"].tolist() == [False, False, False, True]
    assert sample["window_start"] == 0
    assert sample["window_end"] == 0
    assert sample["anchor_index"] == 0


def test_dataset_output_tensor_shapes_and_dtypes() -> None:
    sequence = _sequence()
    dataset = TokenSequenceDataset(sequence, TokenWindowConfig(window_length=3))

    sample = dataset[2]

    for field_name in TOKEN_WINDOW_FIELD_NAMES:
        assert sample[field_name].shape == (3,)
        assert sample[field_name].dtype == torch.long
    assert sample["attention_mask"].shape == (3,)
    assert sample["attention_mask"].dtype == torch.bool
    assert sample["attention_mask"].tolist() == [True, True, True]


def test_dataset_deterministic_indexing() -> None:
    sequence = _sequence()
    dataset = TokenSequenceDataset(sequence, TokenWindowConfig(window_length=3))

    first = dataset[2]
    second = dataset[2]

    for field_name in TOKEN_WINDOW_FIELD_NAMES:
        assert torch.equal(first[field_name], second[field_name])
    assert torch.equal(first["attention_mask"], second["attention_mask"])
    assert first["window_start"] == second["window_start"]
    assert first["window_end"] == second["window_end"]


def test_tiny_synthetic_sequence_works() -> None:
    sequence = _sequence(["TESTUSDT"])
    dataset = TokenSequenceDataset(sequence, TokenWindowConfig(window_length=3))

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["attention_mask"].tolist() == [False, False, True]
    assert sample["event_type"].shape == (3,)
