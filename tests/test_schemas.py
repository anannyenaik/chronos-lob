"""Tests for canonical ChronosLOB data schemas."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from chronoslob.data.schemas import (
    BookEvent,
    DataQualityIssue,
    EventType,
    FeatureRow,
    LabelRow,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
    ensure_utc_datetime,
    is_finite_number,
    validate_metadata,
)

PARIS = timezone(timedelta(hours=2))
T0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Time-validation helpers
# ---------------------------------------------------------------------------


def test_ensure_utc_datetime_accepts_utc_value() -> None:
    result = ensure_utc_datetime(T0)
    assert result is T0
    assert result.tzinfo == UTC


def test_ensure_utc_datetime_normalises_non_utc_to_utc() -> None:
    local = datetime(2024, 1, 1, 14, 0, 0, tzinfo=PARIS)
    result = ensure_utc_datetime(local)
    assert result.tzinfo is not None
    assert result.utcoffset() == timedelta(0)
    assert result == T0


def test_ensure_utc_datetime_rejects_naive() -> None:
    naive = datetime(2024, 1, 1, 12, 0, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        ensure_utc_datetime(naive)


def test_ensure_utc_datetime_rejects_non_datetime() -> None:
    with pytest.raises(TypeError):
        ensure_utc_datetime("2024-01-01T12:00:00Z")  # type: ignore[arg-type]


def test_is_finite_number_behaviour() -> None:
    assert is_finite_number(1) is True
    assert is_finite_number(1.5) is True
    assert is_finite_number(-3.2) is True
    assert is_finite_number(float("nan")) is False
    assert is_finite_number(float("inf")) is False
    assert is_finite_number(float("-inf")) is False
    assert is_finite_number(True) is False
    assert is_finite_number(False) is False
    assert is_finite_number("1.0") is False
    assert is_finite_number(None) is False


def test_validate_metadata_accepts_scalars() -> None:
    cleaned = validate_metadata({"a": 1, "b": 1.5, "c": "x", "d": True})
    assert cleaned == {"a": 1, "b": 1.5, "c": "x", "d": True}


def test_validate_metadata_rejects_nested_objects() -> None:
    with pytest.raises(TypeError):
        validate_metadata({"a": [1, 2, 3]})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        validate_metadata({"a": {"nested": 1}})  # type: ignore[arg-type]


def test_validate_metadata_rejects_nan_and_inf() -> None:
    with pytest.raises(ValueError):
        validate_metadata({"x": float("nan")})
    with pytest.raises(ValueError):
        validate_metadata({"x": float("inf")})


def test_validate_metadata_rejects_empty_keys() -> None:
    with pytest.raises(ValueError):
        validate_metadata({"": "x"})


# ---------------------------------------------------------------------------
# Side / EventType enums
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    ("bid", Side.BID),
    ("BID", Side.BID),
    ("Ask", Side.ASK),
    ("ASK", Side.ASK),
])
def test_side_enum_accepts_string_inputs(value: str, expected: Side) -> None:
    assert Side(value) is expected


def test_side_enum_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        Side("buy")


@pytest.mark.parametrize("value,expected", [
    ("add", EventType.ADD),
    ("CANCEL", EventType.CANCEL),
    ("modify", EventType.MODIFY),
    ("trade", EventType.TRADE),
    ("SNAPSHOT", EventType.SNAPSHOT),
    ("depth_update", EventType.DEPTH_UPDATE),
])
def test_event_type_enum_accepts_string_inputs(value: str, expected: EventType) -> None:
    assert EventType(value) is expected


# ---------------------------------------------------------------------------
# OrderBookLevel
# ---------------------------------------------------------------------------


def test_order_book_level_valid_inputs() -> None:
    level = OrderBookLevel(price=100.5, quantity=2.0, order_count=3)
    assert level.price == 100.5
    assert level.quantity == 2.0
    assert level.order_count == 3


def test_order_book_level_allows_zero_quantity_and_omitted_order_count() -> None:
    level = OrderBookLevel(price=100.0, quantity=0.0)
    assert level.quantity == 0.0
    assert level.order_count is None


@pytest.mark.parametrize("bad_price", [0.0, -0.01, float("nan"), float("inf")])
def test_order_book_level_rejects_invalid_price(bad_price: float) -> None:
    with pytest.raises(ValidationError):
        OrderBookLevel(price=bad_price, quantity=1.0)


@pytest.mark.parametrize("bad_quantity", [-1.0, float("nan"), float("inf")])
def test_order_book_level_rejects_invalid_quantity(bad_quantity: float) -> None:
    with pytest.raises(ValidationError):
        OrderBookLevel(price=100.0, quantity=bad_quantity)


def test_order_book_level_rejects_negative_order_count() -> None:
    with pytest.raises(ValidationError):
        OrderBookLevel(price=100.0, quantity=1.0, order_count=-1)


def test_order_book_level_serialises_to_json_dict() -> None:
    level = OrderBookLevel(price=100.5, quantity=2.0, order_count=3)
    payload = level.model_dump(mode="json")
    assert payload == {"price": 100.5, "quantity": 2.0, "order_count": 3}


# ---------------------------------------------------------------------------
# OrderBookSnapshot
# ---------------------------------------------------------------------------


def _sorted_book() -> OrderBookSnapshot:
    return OrderBookSnapshot(
        timestamp=T0,
        symbol="BTCUSDT",
        bids=[
            OrderBookLevel(price=100.0, quantity=1.0),
            OrderBookLevel(price=99.0, quantity=2.0),
        ],
        asks=[
            OrderBookLevel(price=101.0, quantity=1.5),
            OrderBookLevel(price=102.0, quantity=2.5),
        ],
    )


def test_snapshot_valid_sorted_book() -> None:
    snap = _sorted_book()
    assert snap.best_bid is not None and snap.best_bid.price == 100.0
    assert snap.best_ask is not None and snap.best_ask.price == 101.0
    assert snap.mid_price == pytest.approx(100.5)
    assert snap.spread == pytest.approx(1.0)
    assert snap.is_crossed is False


def test_snapshot_empty_sides_have_none_properties() -> None:
    snap = OrderBookSnapshot(timestamp=T0, symbol="X")
    assert snap.best_bid is None
    assert snap.best_ask is None
    assert snap.mid_price is None
    assert snap.spread is None
    assert snap.is_crossed is False


def test_snapshot_unsorted_bids_fail() -> None:
    with pytest.raises(ValidationError):
        OrderBookSnapshot(
            timestamp=T0,
            symbol="X",
            bids=[
                OrderBookLevel(price=99.0, quantity=1.0),
                OrderBookLevel(price=100.0, quantity=1.0),
            ],
            asks=[OrderBookLevel(price=101.0, quantity=1.0)],
        )


def test_snapshot_unsorted_asks_fail() -> None:
    with pytest.raises(ValidationError):
        OrderBookSnapshot(
            timestamp=T0,
            symbol="X",
            bids=[OrderBookLevel(price=100.0, quantity=1.0)],
            asks=[
                OrderBookLevel(price=102.0, quantity=1.0),
                OrderBookLevel(price=101.0, quantity=1.0),
            ],
        )


def test_snapshot_duplicate_bid_prices_fail() -> None:
    with pytest.raises(ValidationError):
        OrderBookSnapshot(
            timestamp=T0,
            symbol="X",
            bids=[
                OrderBookLevel(price=100.0, quantity=1.0),
                OrderBookLevel(price=100.0, quantity=2.0),
            ],
        )


def test_snapshot_duplicate_ask_prices_fail() -> None:
    with pytest.raises(ValidationError):
        OrderBookSnapshot(
            timestamp=T0,
            symbol="X",
            asks=[
                OrderBookLevel(price=101.0, quantity=1.0),
                OrderBookLevel(price=101.0, quantity=2.0),
            ],
        )


def test_snapshot_crossed_book_is_detectable() -> None:
    snap = OrderBookSnapshot(
        timestamp=T0,
        symbol="X",
        bids=[OrderBookLevel(price=101.0, quantity=1.0)],
        asks=[OrderBookLevel(price=100.0, quantity=1.0)],
    )
    assert snap.is_crossed is True
    with pytest.raises(ValueError, match="crossed"):
        snap.assert_not_crossed()
    snap.assert_not_crossed(allow_crossed=True)


def test_snapshot_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(ValidationError):
        OrderBookSnapshot(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),  # naive
            symbol="X",
        )


def test_snapshot_normalises_non_utc_timestamps() -> None:
    snap = OrderBookSnapshot(
        timestamp=datetime(2024, 1, 1, 14, 0, 0, tzinfo=PARIS),
        symbol="X",
        received_timestamp=datetime(2024, 1, 1, 14, 0, 1, tzinfo=PARIS),
    )
    assert snap.timestamp.utcoffset() == timedelta(0)
    assert snap.received_timestamp is not None
    assert snap.received_timestamp.utcoffset() == timedelta(0)


def test_snapshot_rejects_empty_symbol() -> None:
    with pytest.raises(ValidationError):
        OrderBookSnapshot(timestamp=T0, symbol="")


def test_snapshot_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        OrderBookSnapshot(timestamp=T0, symbol="X", extra_field=1)  # type: ignore[call-arg]


def test_snapshot_round_trip_serialisation() -> None:
    snap = _sorted_book()
    payload = snap.model_dump(mode="json")
    assert payload["symbol"] == "BTCUSDT"
    assert payload["bids"][0]["price"] == 100.0
    # Re-construct from dump to confirm field stability.
    again = OrderBookSnapshot.model_validate(payload)
    assert again.mid_price == snap.mid_price


# ---------------------------------------------------------------------------
# BookEvent
# ---------------------------------------------------------------------------


def test_book_event_valid_add() -> None:
    event = BookEvent(
        timestamp=T0,
        event_type=EventType.ADD,
        symbol="BTCUSDT",
        side=Side.BID,
        price=100.0,
        quantity=1.5,
        order_id="abc",
    )
    assert event.event_type is EventType.ADD
    assert event.side is Side.BID


def test_book_event_add_requires_side() -> None:
    with pytest.raises(ValidationError):
        BookEvent(
            timestamp=T0,
            event_type=EventType.ADD,
            symbol="X",
            price=100.0,
            quantity=1.0,
        )


def test_book_event_trade_without_side_is_allowed() -> None:
    event = BookEvent(
        timestamp=T0,
        event_type=EventType.TRADE,
        symbol="X",
        price=100.0,
        quantity=0.5,
        trade_id="t1",
    )
    assert event.side is None


def test_book_event_snapshot_can_omit_side_price_quantity() -> None:
    event = BookEvent(timestamp=T0, event_type=EventType.SNAPSHOT, symbol="X")
    assert event.side is None
    assert event.price is None
    assert event.quantity is None


@pytest.mark.parametrize("bad_price", [0.0, -1.0, float("nan"), float("inf")])
def test_book_event_rejects_invalid_price(bad_price: float) -> None:
    with pytest.raises(ValidationError):
        BookEvent(
            timestamp=T0,
            event_type=EventType.TRADE,
            symbol="X",
            price=bad_price,
            quantity=1.0,
        )


@pytest.mark.parametrize("bad_quantity", [-1.0, float("nan"), float("inf")])
def test_book_event_rejects_invalid_quantity(bad_quantity: float) -> None:
    with pytest.raises(ValidationError):
        BookEvent(
            timestamp=T0,
            event_type=EventType.TRADE,
            symbol="X",
            price=100.0,
            quantity=bad_quantity,
        )


def test_book_event_metadata_rejects_nested_objects() -> None:
    with pytest.raises(ValidationError):
        BookEvent(
            timestamp=T0,
            event_type=EventType.SNAPSHOT,
            symbol="X",
            metadata={"nested": {"a": 1}},  # type: ignore[dict-item]
        )


def test_book_event_metadata_rejects_nan() -> None:
    with pytest.raises(ValidationError):
        BookEvent(
            timestamp=T0,
            event_type=EventType.SNAPSHOT,
            symbol="X",
            metadata={"x": float("nan")},
        )


def test_book_event_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(ValidationError):
        BookEvent(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),  # naive
            event_type=EventType.SNAPSHOT,
            symbol="X",
        )


def test_book_event_received_timestamp_must_be_aware() -> None:
    with pytest.raises(ValidationError):
        BookEvent(
            timestamp=T0,
            event_type=EventType.SNAPSHOT,
            symbol="X",
            received_timestamp=datetime(2024, 1, 1, 12, 0, 0),  # naive
        )


# ---------------------------------------------------------------------------
# FeatureRow
# ---------------------------------------------------------------------------


def test_feature_row_accepts_finite_values() -> None:
    row = FeatureRow(
        timestamp=T0,
        symbol="X",
        features={"spread": 0.01, "mid": 100.5},
    )
    assert row.features == {"spread": 0.01, "mid": 100.5}


def test_feature_row_rejects_nan() -> None:
    with pytest.raises(ValidationError):
        FeatureRow(timestamp=T0, symbol="X", features={"f": math.nan})


def test_feature_row_rejects_inf() -> None:
    with pytest.raises(ValidationError):
        FeatureRow(timestamp=T0, symbol="X", features={"f": math.inf})


def test_feature_row_rejects_empty_feature_name() -> None:
    with pytest.raises(ValidationError):
        FeatureRow(timestamp=T0, symbol="X", features={"": 0.1})


def test_feature_row_horizon_origin_must_not_be_after_timestamp() -> None:
    with pytest.raises(ValidationError):
        FeatureRow(
            timestamp=T0,
            symbol="X",
            features={"f": 1.0},
            horizon_origin_timestamp=T0 + timedelta(seconds=1),
        )


def test_feature_row_horizon_origin_equal_or_before_timestamp_passes() -> None:
    row = FeatureRow(
        timestamp=T0,
        symbol="X",
        features={"f": 1.0},
        horizon_origin_timestamp=T0 - timedelta(seconds=5),
    )
    assert row.horizon_origin_timestamp == T0 - timedelta(seconds=5)


# ---------------------------------------------------------------------------
# LabelRow
# ---------------------------------------------------------------------------


def test_label_row_valid_future_horizon() -> None:
    row = LabelRow(
        timestamp=T0,
        symbol="X",
        labels={"up": True, "mid_return": 0.001, "regime": "trend"},
        horizon_start=T0,
        horizon_end=T0 + timedelta(seconds=10),
    )
    assert row.labels["up"] is True
    assert row.labels["regime"] == "trend"


def test_label_row_horizon_start_before_timestamp_fails() -> None:
    with pytest.raises(ValidationError):
        LabelRow(
            timestamp=T0,
            symbol="X",
            labels={"up": True},
            horizon_start=T0 - timedelta(seconds=1),
            horizon_end=T0 + timedelta(seconds=10),
        )


def test_label_row_horizon_end_not_after_start_fails() -> None:
    with pytest.raises(ValidationError):
        LabelRow(
            timestamp=T0,
            symbol="X",
            labels={"up": True},
            horizon_start=T0,
            horizon_end=T0,
        )


def test_label_row_rejects_nan_numeric_label() -> None:
    with pytest.raises(ValidationError):
        LabelRow(
            timestamp=T0,
            symbol="X",
            labels={"r": math.nan},
            horizon_start=T0,
            horizon_end=T0 + timedelta(seconds=1),
        )


def test_label_row_rejects_empty_label_name() -> None:
    with pytest.raises(ValidationError):
        LabelRow(
            timestamp=T0,
            symbol="X",
            labels={"": 1.0},
            horizon_start=T0,
            horizon_end=T0 + timedelta(seconds=1),
        )


def test_label_row_requires_aware_timestamps() -> None:
    with pytest.raises(ValidationError):
        LabelRow(
            timestamp=datetime(2024, 1, 1),
            symbol="X",
            labels={"r": 0.0},
            horizon_start=T0,
            horizon_end=T0 + timedelta(seconds=1),
        )


# ---------------------------------------------------------------------------
# DataQualityIssue
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("severity", ["info", "warning", "error", "INFO", " Warning "])
def test_data_quality_issue_accepts_valid_severity(severity: str) -> None:
    issue = DataQualityIssue(severity=severity, code="C1", message="something happened")
    assert issue.severity in {"info", "warning", "error"}


def test_data_quality_issue_rejects_invalid_severity() -> None:
    with pytest.raises(ValidationError):
        DataQualityIssue(severity="critical", code="C1", message="x")


def test_data_quality_issue_requires_non_empty_code_and_message() -> None:
    with pytest.raises(ValidationError):
        DataQualityIssue(severity="info", code="", message="x")
    with pytest.raises(ValidationError):
        DataQualityIssue(severity="info", code="C1", message="")


def test_data_quality_issue_timestamp_must_be_aware_when_provided() -> None:
    with pytest.raises(ValidationError):
        DataQualityIssue(
            severity="info",
            code="C1",
            message="msg",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
        )


def test_data_quality_issue_allows_optional_fields() -> None:
    issue = DataQualityIssue(severity="warning", code="C2", message="msg")
    assert issue.timestamp is None
    assert issue.symbol is None
