"""Tests for deterministic event tokenisation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from chronoslob.data.schemas import (
    BookEvent,
    EventType,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
)
from chronoslob.models.tokenisation import (
    CONTEXT_MISSING,
    EVENT_TYPE_SNAPSHOT,
    EVENT_TYPE_SNAPSHOT_LEVEL,
    PRICE_MISSING,
    QUANTITY_MISSING,
    SIDE_NONE,
    SOURCE_UNKNOWN,
    SPECIAL_TOKEN_IDS,
    SpecialToken,
    TokenField,
    TokenisationConfig,
    build_static_token_vocabulary,
    event_type_token,
    fit_tokenisation_state,
    price_bucket_token,
    quantity_bucket_token,
    side_token,
    source_value_to_token,
    time_delta_bucket_token,
    tokenise_book_event,
    tokenise_records,
    tokenise_snapshot,
)
from chronoslob.training.splitters import SplitIndices


def _snapshot(
    sequence_id: int = 1,
    *,
    timestamp: datetime | None = None,
    source: str = "synthetic_train",
) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        timestamp=timestamp or datetime(2024, 1, 1, tzinfo=UTC),
        symbol="TESTUSDT",
        venue="synthetic",
        sequence_id=sequence_id,
        bids=[
            OrderBookLevel(price=100.0, quantity=1.0),
            OrderBookLevel(price=99.5, quantity=2.0),
        ],
        asks=[
            OrderBookLevel(price=101.0, quantity=1.5),
            OrderBookLevel(price=101.5, quantity=2.5),
        ],
        metadata={"source": source},
    )


def _event(
    sequence_id: int = 1,
    *,
    timestamp: datetime | None = None,
    source: str = "synthetic_train",
) -> BookEvent:
    return BookEvent(
        timestamp=timestamp or datetime(2024, 1, 1, tzinfo=UTC),
        event_type=EventType.ADD,
        symbol="TESTUSDT",
        side=Side.BID,
        price=100.0,
        quantity=0.5,
        sequence_id=sequence_id,
        metadata={"source": source},
    )


def _token(vocabulary, field: TokenField, token_id: int) -> str:
    return vocabulary.id_to_token(field, token_id)


def test_fixed_special_token_ids() -> None:
    assert SPECIAL_TOKEN_IDS[SpecialToken.PAD] == 0
    assert SPECIAL_TOKEN_IDS[SpecialToken.UNK] == 1
    assert SPECIAL_TOKEN_IDS[SpecialToken.BOS] == 2
    assert SPECIAL_TOKEN_IDS[SpecialToken.EOS] == 3
    assert SPECIAL_TOKEN_IDS[SpecialToken.MASK] == 4


def test_deterministic_vocabulary_construction() -> None:
    config = TokenisationConfig(static_source_tokens=("b", "a", "a"))
    first = build_static_token_vocabulary(config)
    second = build_static_token_vocabulary(config)

    assert first.tokens_by_field == second.tokens_by_field
    assert first.id_to_token(TokenField.EVENT_TYPE, 0) == "[PAD]"
    assert first.contains(TokenField.SOURCE, source_value_to_token("a"))
    assert first.contains(TokenField.SOURCE, source_value_to_token("b"))


def test_side_and_event_type_token_mapping() -> None:
    assert side_token(Side.BID) == "bid"
    assert side_token(Side.ASK) == "ask"
    assert side_token(None) == SIDE_NONE
    assert event_type_token(EventType.ADD) == "add"
    assert event_type_token(EventType.CANCEL) == "cancel"
    assert event_type_token(EventType.TRADE) == "trade"
    assert event_type_token(EventType.MODIFY) == "modify"
    assert event_type_token(EventType.DEPTH_UPDATE) == "depth_update"
    assert event_type_token(EventType.SNAPSHOT) == EVENT_TYPE_SNAPSHOT


def test_unknown_token_handling() -> None:
    vocabulary = build_static_token_vocabulary(TokenisationConfig(fit_source_tokens=False))

    assert vocabulary.token_to_id(TokenField.SOURCE, source_value_to_token("unseen")) == 1
    assert vocabulary.token_to_id(TokenField.SOURCE, SOURCE_UNKNOWN) > 4


def test_quantity_bucket_boundaries() -> None:
    config = TokenisationConfig().quantity_buckets

    assert quantity_bucket_token(None, config) == QUANTITY_MISSING
    assert quantity_bucket_token(0.0, config) == "zero"
    assert quantity_bucket_token(0.01, config) == "very_small"
    assert quantity_bucket_token(0.1, config) == "small"
    assert quantity_bucket_token(1.0, config) == "medium"
    assert quantity_bucket_token(10.0, config) == "large"
    assert quantity_bucket_token(10.1, config) == "very_large"


def test_time_delta_bucket_boundaries() -> None:
    config = TokenisationConfig().time_delta_buckets
    base = datetime(2024, 1, 1, tzinfo=UTC)

    assert time_delta_bucket_token(base, None, config) == "missing_or_first"
    assert time_delta_bucket_token(base, base, config) == "zero"
    assert time_delta_bucket_token(base + timedelta(milliseconds=1), base, config) == (
        "lte_1ms"
    )
    assert time_delta_bucket_token(base + timedelta(milliseconds=10), base, config) == (
        "lte_10ms"
    )
    assert time_delta_bucket_token(base + timedelta(milliseconds=100), base, config) == (
        "lte_100ms"
    )
    assert time_delta_bucket_token(base + timedelta(seconds=1), base, config) == (
        "lte_1s"
    )
    assert time_delta_bucket_token(base + timedelta(seconds=10), base, config) == (
        "lte_10s"
    )
    assert time_delta_bucket_token(base + timedelta(seconds=11), base, config) == (
        "gt_10s"
    )


def test_price_bucket_behaviour_relative_to_mid_price() -> None:
    config = TokenisationConfig()
    snapshot = _snapshot()

    assert snapshot.mid_price == 100.5
    bid_token = price_bucket_token(100.0, snapshot.mid_price, config.price_buckets)
    ask_token = price_bucket_token(101.0, snapshot.mid_price, config.price_buckets)

    assert bid_token == "rel_lte_-10bp"
    assert ask_token == "rel_lte_50bp"
    assert price_bucket_token(None, snapshot.mid_price, config.price_buckets) == (
        PRICE_MISSING
    )


def test_snapshot_tokenisation_order_and_level_sides() -> None:
    config = TokenisationConfig(max_levels_per_side=2)
    state = fit_tokenisation_state([_snapshot()], config)
    records = tokenise_snapshot(_snapshot(), state.vocabulary, config)

    event_tokens = [
        _token(state.vocabulary, TokenField.EVENT_TYPE, record.event_type_id)
        for record in records
    ]
    side_tokens = [
        _token(state.vocabulary, TokenField.SIDE, record.side_id) for record in records
    ]

    assert event_tokens == [
        EVENT_TYPE_SNAPSHOT,
        EVENT_TYPE_SNAPSHOT_LEVEL,
        EVENT_TYPE_SNAPSHOT_LEVEL,
        EVENT_TYPE_SNAPSHOT_LEVEL,
        EVENT_TYPE_SNAPSHOT_LEVEL,
    ]
    assert side_tokens == [SIDE_NONE, "bid", "bid", "ask", "ask"]
    assert [record.token_role for record in records] == [
        "snapshot_summary",
        "snapshot_level",
        "snapshot_level",
        "snapshot_level",
        "snapshot_level",
    ]
    assert [record.level_index for record in records] == [None, 0, 1, 0, 1]


def test_snapshot_max_levels_per_side_respected() -> None:
    config = TokenisationConfig(max_levels_per_side=1)
    state = fit_tokenisation_state([_snapshot()], config)

    records = tokenise_snapshot(_snapshot(), state.vocabulary, config)

    assert len(records) == 3
    assert [record.token_role for record in records] == [
        "snapshot_summary",
        "snapshot_level",
        "snapshot_level",
    ]


def test_book_event_tokenisation_emits_expected_field_ids() -> None:
    config = TokenisationConfig()
    event = _event()
    state = fit_tokenisation_state([event], config)

    [record] = tokenise_book_event(
        event,
        state.vocabulary,
        config,
        previous_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        reference_price=100.0,
    )

    assert _token(state.vocabulary, TokenField.EVENT_TYPE, record.event_type_id) == "add"
    assert _token(state.vocabulary, TokenField.SIDE, record.side_id) == "bid"
    assert _token(state.vocabulary, TokenField.PRICE_BUCKET, record.price_bucket_id) == (
        "rel_lte_0bp"
    )
    assert _token(
        state.vocabulary,
        TokenField.QUANTITY_BUCKET,
        record.quantity_bucket_id,
    ) == "medium"
    assert _token(
        state.vocabulary,
        TokenField.CONTEXT_BUCKET,
        record.context_bucket_id,
    ) == CONTEXT_MISSING
    assert record.is_snapshot_derived is False


def test_validation_tokenisation_does_not_expand_vocabulary() -> None:
    train = _event(1, source="train_source")
    validation = _event(
        2,
        timestamp=datetime(2024, 1, 1, 0, 0, 1, tzinfo=UTC),
        source="validation_source",
    )
    config = TokenisationConfig()
    split = SplitIndices(train=[0], validation=[1], test=[])
    state = fit_tokenisation_state([train, validation], config, split_indices=split)
    sequence = tokenise_records([train, validation], state.vocabulary, config)

    assert state.vocabulary.contains(
        TokenField.SOURCE,
        source_value_to_token("train_source"),
    )
    assert not state.vocabulary.contains(
        TokenField.SOURCE,
        source_value_to_token("validation_source"),
    )
    assert sequence.records[0].source_id > 4
    assert sequence.records[1].source_id == SPECIAL_TOKEN_IDS[SpecialToken.UNK]


def test_tokenisation_is_deterministic_across_repeated_runs() -> None:
    records = [
        _snapshot(2, timestamp=datetime(2024, 1, 1, 0, 0, 1, tzinfo=UTC)),
        _snapshot(1, timestamp=datetime(2024, 1, 1, tzinfo=UTC)),
    ]
    config = TokenisationConfig(max_levels_per_side=1)
    first_state = fit_tokenisation_state(records, config)
    second_state = fit_tokenisation_state(records, config)

    first = tokenise_records(records, first_state.vocabulary, config)
    second = tokenise_records(records, second_state.vocabulary, config)

    assert first.vocabulary.tokens_by_field == second.vocabulary.tokens_by_field
    assert [record.field_id_mapping() for record in first.records] == [
        record.field_id_mapping() for record in second.records
    ]
    assert [record.sequence_id for record in first.records] == [1, 1, 1, 2, 2, 2]


def test_naive_timestamp_rejected_during_tokenisation() -> None:
    naive_event = BookEvent.model_construct(
        timestamp=datetime(2024, 1, 1),
        event_type=EventType.ADD,
        symbol="TESTUSDT",
        side=Side.BID,
        price=100.0,
        quantity=1.0,
        sequence_id=1,
        metadata={},
    )
    config = TokenisationConfig()
    vocabulary = build_static_token_vocabulary(config)

    with pytest.raises(ValueError, match="timezone-aware"):
        tokenise_book_event(naive_event, vocabulary, config)
