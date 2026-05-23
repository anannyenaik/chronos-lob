"""Deterministic event tokenisation for future transformer inputs.

This module converts canonical :class:`BookEvent` and
:class:`OrderBookSnapshot` records into field-wise categorical IDs. It does
not implement trainable embeddings, masking objectives, transformer layers or
any target generation.

Special token IDs are fixed for every token field:

* ``[PAD]`` = 0
* ``[UNK]`` = 1
* ``[BOS]`` = 2
* ``[EOS]`` = 3
* ``[MASK]`` = 4

Snapshot level tokens are deterministic representations derived from an
order-book snapshot. They are not raw exchange-native order events.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from chronoslob.data.event_store import (
    EventLogObject,
    filter_event_log_records,
    read_event_log_jsonl,
)
from chronoslob.data.schemas import (
    BookEvent,
    EventType,
    MetadataScalar,
    OrderBookLevel,
    OrderBookSnapshot,
    Side,
    ensure_utc_datetime,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from chronoslob.training.splitters import SplitIndices

PriceReference = Literal["mid", "best_same_side", "best_opposite_side"]
SpreadMode = Literal["absolute", "relative_mid_bps"]


class SpecialToken(StrEnum):
    """Reserved tokens available in every categorical field vocabulary."""

    PAD = "[PAD]"
    UNK = "[UNK]"
    BOS = "[BOS]"
    EOS = "[EOS]"
    MASK = "[MASK]"


SPECIAL_TOKEN_IDS: Mapping[SpecialToken, int] = {
    SpecialToken.PAD: 0,
    SpecialToken.UNK: 1,
    SpecialToken.BOS: 2,
    SpecialToken.EOS: 3,
    SpecialToken.MASK: 4,
}


class TokenField(StrEnum):
    """Categorical channels emitted for every tokenised record."""

    EVENT_TYPE = "event_type"
    SIDE = "side"
    PRICE_BUCKET = "price_bucket"
    QUANTITY_BUCKET = "quantity_bucket"
    TIME_DELTA_BUCKET = "time_delta_bucket"
    CONTEXT_BUCKET = "context_bucket"
    SOURCE = "source"


TOKEN_FIELDS: tuple[TokenField, ...] = tuple(TokenField)

EVENT_TYPE_SNAPSHOT = "snapshot"
EVENT_TYPE_SNAPSHOT_LEVEL = "snapshot_level"
EVENT_TYPE_UNKNOWN = "unknown_event"
SIDE_NONE = "none"
SIDE_UNKNOWN = "unknown_side"
PRICE_MISSING = "missing_price"
QUANTITY_MISSING = "missing"
QUANTITY_ZERO = "zero"
TIME_DELTA_MISSING = "missing_or_first"
TIME_DELTA_ZERO = "zero"
CONTEXT_MISSING = "missing_context"
CONTEXT_CROSSED = "crossed"
CONTEXT_ZERO = "zero"
SOURCE_UNKNOWN = "unknown_source"


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_thresholds(
    thresholds: Sequence[float],
    *,
    name: str,
    allow_negative: bool = False,
) -> tuple[float, ...]:
    cleaned: list[float] = []
    previous: float | None = None
    for position, value in enumerate(thresholds):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name}[{position}] must be a finite number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{name}[{position}] must be finite")
        if not allow_negative and numeric <= 0.0:
            raise ValueError(f"{name}[{position}] must be strictly positive")
        if previous is not None and numeric <= previous:
            raise ValueError(f"{name} must be strictly increasing")
        cleaned.append(numeric)
        previous = numeric
    if not cleaned:
        raise ValueError(f"{name} must not be empty")
    return tuple(cleaned)


def _validate_labels(labels: Sequence[str], *, name: str) -> tuple[str, ...]:
    cleaned: list[str] = []
    for position, label in enumerate(labels):
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{name}[{position}] must be a non-empty string")
        cleaned.append(label.strip())
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(cleaned)


@dataclass(frozen=True)
class PriceBucketConfig:
    """Fixed relative-price bucket configuration.

    Prices are bucketed in basis points relative to the configured reference
    price. No train, validation or test data are fitted for these boundaries.
    """

    relative_bps_thresholds: tuple[float, ...] = (
        -100.0,
        -50.0,
        -10.0,
        0.0,
        10.0,
        50.0,
        100.0,
    )

    def __post_init__(self) -> None:
        thresholds = _validate_thresholds(
            self.relative_bps_thresholds,
            name="relative_bps_thresholds",
            allow_negative=True,
        )
        object.__setattr__(self, "relative_bps_thresholds", thresholds)

    @property
    def bucket_tokens(self) -> tuple[str, ...]:
        """Return deterministic token labels for the configured buckets."""
        labels = [PRICE_MISSING]
        labels.extend(
            f"rel_lte_{_format_threshold(threshold)}bp"
            for threshold in self.relative_bps_thresholds
        )
        labels.append(
            f"rel_gt_{_format_threshold(self.relative_bps_thresholds[-1])}bp"
        )
        return tuple(labels)


@dataclass(frozen=True)
class QuantityBucketConfig:
    """Fixed log-style quantity bucket configuration."""

    positive_thresholds: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    positive_labels: tuple[str, ...] = (
        "very_small",
        "small",
        "medium",
        "large",
        "very_large",
    )

    def __post_init__(self) -> None:
        thresholds = _validate_thresholds(
            self.positive_thresholds,
            name="positive_thresholds",
        )
        labels = _validate_labels(self.positive_labels, name="positive_labels")
        if len(labels) != len(thresholds) + 1:
            raise ValueError(
                "positive_labels must contain exactly one more entry than "
                "positive_thresholds"
            )
        object.__setattr__(self, "positive_thresholds", thresholds)
        object.__setattr__(self, "positive_labels", labels)

    @property
    def bucket_tokens(self) -> tuple[str, ...]:
        """Return deterministic token labels for quantity buckets."""
        return (QUANTITY_MISSING, QUANTITY_ZERO, *self.positive_labels)


@dataclass(frozen=True)
class TimeDeltaBucketConfig:
    """Fixed bucket configuration for timestamp deltas."""

    thresholds_seconds: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)
    labels: tuple[str, ...] = (
        "lte_1ms",
        "lte_10ms",
        "lte_100ms",
        "lte_1s",
        "lte_10s",
        "gt_10s",
    )

    def __post_init__(self) -> None:
        thresholds = _validate_thresholds(
            self.thresholds_seconds,
            name="thresholds_seconds",
        )
        labels = _validate_labels(self.labels, name="labels")
        if len(labels) != len(thresholds) + 1:
            raise ValueError(
                "labels must contain exactly one more entry than thresholds_seconds"
            )
        object.__setattr__(self, "thresholds_seconds", thresholds)
        object.__setattr__(self, "labels", labels)

    @property
    def bucket_tokens(self) -> tuple[str, ...]:
        """Return deterministic token labels for time-delta buckets."""
        return (TIME_DELTA_MISSING, TIME_DELTA_ZERO, *self.labels)


@dataclass(frozen=True)
class ContextBucketConfig:
    """Fixed spread/context bucket configuration."""

    mode: SpreadMode = "relative_mid_bps"
    thresholds: tuple[float, ...] = (1.0, 5.0, 10.0, 50.0, 100.0)
    labels: tuple[str, ...] = (
        "lte_1bp",
        "lte_5bp",
        "lte_10bp",
        "lte_50bp",
        "lte_100bp",
        "gt_100bp",
    )

    def __post_init__(self) -> None:
        if self.mode not in {"absolute", "relative_mid_bps"}:
            raise ValueError("mode must be 'absolute' or 'relative_mid_bps'")
        thresholds = _validate_thresholds(self.thresholds, name="thresholds")
        labels = _validate_labels(self.labels, name="labels")
        if len(labels) != len(thresholds) + 1:
            raise ValueError("labels must contain exactly one more entry than thresholds")
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "labels", labels)

    @property
    def bucket_tokens(self) -> tuple[str, ...]:
        """Return deterministic token labels for context buckets."""
        return (CONTEXT_MISSING, CONTEXT_CROSSED, CONTEXT_ZERO, *self.labels)


@dataclass(frozen=True)
class TokenisationConfig:
    """Configuration for deterministic canonical-record tokenisation."""

    max_levels_per_side: int = 10
    price_reference: PriceReference = "mid"
    price_buckets: PriceBucketConfig = field(default_factory=PriceBucketConfig)
    quantity_buckets: QuantityBucketConfig = field(default_factory=QuantityBucketConfig)
    time_delta_buckets: TimeDeltaBucketConfig = field(
        default_factory=TimeDeltaBucketConfig
    )
    context_buckets: ContextBucketConfig = field(default_factory=ContextBucketConfig)
    source_metadata_keys: tuple[str, ...] = ("source", "venue", "dataset")
    static_source_tokens: tuple[str, ...] = ()
    fit_source_tokens: bool = True
    sort_records: bool = True
    include_bos: bool = False
    include_eos: bool = False

    def __post_init__(self) -> None:
        _validate_positive_int(self.max_levels_per_side, name="max_levels_per_side")
        if self.price_reference not in {
            "mid",
            "best_same_side",
            "best_opposite_side",
        }:
            raise ValueError(
                "price_reference must be 'mid', 'best_same_side' or "
                "'best_opposite_side'"
            )
        source_keys = _validate_labels(
            self.source_metadata_keys,
            name="source_metadata_keys",
        )
        static_sources = _normalise_source_values(self.static_source_tokens)
        object.__setattr__(self, "source_metadata_keys", source_keys)
        object.__setattr__(self, "static_source_tokens", static_sources)


@dataclass(frozen=True)
class TokenVocabulary:
    """Frozen field-wise token vocabulary with fixed special token IDs."""

    tokens_by_field: Mapping[TokenField, tuple[str, ...]]

    def __post_init__(self) -> None:
        cleaned: dict[TokenField, tuple[str, ...]] = {}
        for field_name in TOKEN_FIELDS:
            if field_name not in self.tokens_by_field:
                raise ValueError(f"tokens_by_field is missing {field_name.value}")
            tokens = tuple(self.tokens_by_field[field_name])
            expected_specials = tuple(token.value for token in SpecialToken)
            if tokens[: len(expected_specials)] != expected_specials:
                raise ValueError(
                    f"{field_name.value} vocabulary must start with fixed "
                    "special tokens"
                )
            if len(tokens) != len(set(tokens)):
                raise ValueError(
                    f"{field_name.value} vocabulary contains duplicate tokens"
                )
            cleaned[field_name] = tokens
        object.__setattr__(self, "tokens_by_field", cleaned)

    def field_size(self, field_name: TokenField | str) -> int:
        """Return the number of tokens in one field vocabulary."""
        field_key = _coerce_field(field_name)
        return len(self.tokens_by_field[field_key])

    def field_sizes(self) -> dict[str, int]:
        """Return vocabulary sizes keyed by public field name."""
        return {
            field_name.value: self.field_size(field_name)
            for field_name in TOKEN_FIELDS
        }

    def token_to_id(
        self,
        field_name: TokenField | str,
        token: str,
        *,
        unknown_token: SpecialToken = SpecialToken.UNK,
    ) -> int:
        """Return a token ID, falling back to ``[UNK]`` by default."""
        field_key = _coerce_field(field_name)
        tokens = self.tokens_by_field[field_key]
        try:
            return tokens.index(token)
        except ValueError:
            return int(SPECIAL_TOKEN_IDS[unknown_token])

    def id_to_token(self, field_name: TokenField | str, token_id: int) -> str:
        """Return the token label for ``token_id`` in ``field_name``."""
        field_key = _coerce_field(field_name)
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("token_id must be an integer")
        tokens = self.tokens_by_field[field_key]
        if token_id < 0 or token_id >= len(tokens):
            raise IndexError(
                f"token_id {token_id} is outside {field_key.value} vocabulary"
            )
        return tokens[token_id]

    def contains(self, field_name: TokenField | str, token: str) -> bool:
        """Return whether ``token`` exists in one field vocabulary."""
        field_key = _coerce_field(field_name)
        return token in self.tokens_by_field[field_key]

    def with_added_tokens(
        self,
        field_name: TokenField | str,
        tokens: Iterable[str],
    ) -> TokenVocabulary:
        """Return a new vocabulary with sorted new tokens appended."""
        field_key = _coerce_field(field_name)
        existing = list(self.tokens_by_field[field_key])
        existing_set = set(existing)
        additions = sorted({token for token in tokens if token not in existing_set})
        if not additions:
            return self
        updated = dict(self.tokens_by_field)
        updated[field_key] = tuple(existing + additions)
        return TokenVocabulary(tokens_by_field=updated)


@dataclass(frozen=True)
class TokenisedRecord:
    """One field-wise token representation for a canonical market-data record."""

    event_type_id: int
    side_id: int
    price_bucket_id: int
    quantity_bucket_id: int
    time_delta_bucket_id: int
    context_bucket_id: int
    source_id: int
    position: int
    timestamp: datetime | None
    symbol: str | None
    sequence_id: int | None
    record_kind: str
    token_role: str
    source_record_index: int | None = None
    level_index: int | None = None
    source_value: str | None = None
    is_snapshot_derived: bool = False

    def __post_init__(self) -> None:
        for name in (
            "event_type_id",
            "side_id",
            "price_bucket_id",
            "quantity_bucket_id",
            "time_delta_bucket_id",
            "context_bucket_id",
            "source_id",
        ):
            _validate_non_negative_int(int(getattr(self, name)), name=name)
        _validate_non_negative_int(self.position, name="position")
        if self.timestamp is not None:
            object.__setattr__(self, "timestamp", ensure_utc_datetime(self.timestamp))

    def field_id_mapping(self) -> dict[str, int]:
        """Return categorical IDs keyed by transformer input field name."""
        return {
            TokenField.EVENT_TYPE.value: self.event_type_id,
            TokenField.SIDE.value: self.side_id,
            TokenField.PRICE_BUCKET.value: self.price_bucket_id,
            TokenField.QUANTITY_BUCKET.value: self.quantity_bucket_id,
            TokenField.TIME_DELTA_BUCKET.value: self.time_delta_bucket_id,
            TokenField.CONTEXT_BUCKET.value: self.context_bucket_id,
            TokenField.SOURCE.value: self.source_id,
        }


@dataclass(frozen=True)
class TokenSequence:
    """A deterministic sequence of field-wise tokenised records."""

    records: tuple[TokenisedRecord, ...]
    vocabulary: TokenVocabulary
    config: TokenisationConfig
    input_record_count: int
    source_path: Path | None = None

    def __len__(self) -> int:
        """Return the number of tokenised records."""
        return len(self.records)

    @property
    def has_snapshot_derived_tokens(self) -> bool:
        """Return whether the sequence contains snapshot-derived tokens."""
        return any(record.is_snapshot_derived for record in self.records)

    @property
    def field_sizes(self) -> dict[str, int]:
        """Return vocabulary sizes for every token field."""
        return self.vocabulary.field_sizes()


@dataclass(frozen=True)
class TokenisationFitState:
    """Frozen train-only state used by tokenisation."""

    vocabulary: TokenVocabulary
    config: TokenisationConfig
    fitted_source_tokens: tuple[str, ...] = ()
    n_fit_records: int = 0


def _coerce_field(field_name: TokenField | str) -> TokenField:
    if isinstance(field_name, TokenField):
        return field_name
    try:
        return TokenField(field_name)
    except ValueError as exc:
        raise ValueError(f"unknown token field {field_name!r}") from exc


def _format_threshold(value: float) -> str:
    if value == 0:
        return "0"
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p").replace("-", "minus_")


def _special_tokens() -> tuple[str, ...]:
    return tuple(token.value for token in SpecialToken)


def _deduplicate_preserving_order(tokens: Iterable[str]) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if not token or token in seen:
            continue
        cleaned.append(token)
        seen.add(token)
    return tuple(cleaned)


def _normalise_source_value(value: MetadataScalar | str) -> str | None:
    text = str(value).strip()
    return text if text else None


def _normalise_source_values(values: Iterable[MetadataScalar | str]) -> tuple[str, ...]:
    cleaned = [
        source
        for value in values
        if (source := _normalise_source_value(value)) is not None
    ]
    return tuple(sorted(set(cleaned)))


def source_value_to_token(value: str) -> str:
    """Return the deterministic source-vocabulary token for ``value``."""
    return f"source:{value}"


def build_static_token_vocabulary(config: TokenisationConfig) -> TokenVocabulary:
    """Build a deterministic vocabulary from static config only."""
    special = _special_tokens()
    event_type_tokens = (
        EVENT_TYPE_SNAPSHOT,
        EVENT_TYPE_SNAPSHOT_LEVEL,
        "add",
        "cancel",
        "trade",
        "modify",
        "depth_update",
        EVENT_TYPE_UNKNOWN,
    )
    side_tokens = ("bid", "ask", SIDE_NONE, SIDE_UNKNOWN)
    source_tokens = (
        SOURCE_UNKNOWN,
        *(source_value_to_token(value) for value in config.static_source_tokens),
    )
    tokens_by_field: dict[TokenField, tuple[str, ...]] = {
        TokenField.EVENT_TYPE: _deduplicate_preserving_order(
            (*special, *event_type_tokens)
        ),
        TokenField.SIDE: _deduplicate_preserving_order((*special, *side_tokens)),
        TokenField.PRICE_BUCKET: _deduplicate_preserving_order(
            (*special, *config.price_buckets.bucket_tokens)
        ),
        TokenField.QUANTITY_BUCKET: _deduplicate_preserving_order(
            (*special, *config.quantity_buckets.bucket_tokens)
        ),
        TokenField.TIME_DELTA_BUCKET: _deduplicate_preserving_order(
            (*special, *config.time_delta_buckets.bucket_tokens)
        ),
        TokenField.CONTEXT_BUCKET: _deduplicate_preserving_order(
            (*special, *config.context_buckets.bucket_tokens)
        ),
        TokenField.SOURCE: _deduplicate_preserving_order((*special, *source_tokens)),
    }
    return TokenVocabulary(tokens_by_field=tokens_by_field)


def _record_source_value(
    record: BookEvent | OrderBookSnapshot,
    config: TokenisationConfig,
) -> str | None:
    for key in config.source_metadata_keys:
        value = record.metadata.get(key)
        if value is not None:
            source = _normalise_source_value(value)
            if source is not None:
                return source
    if isinstance(record, OrderBookSnapshot) and record.venue is not None:
        return _normalise_source_value(record.venue)
    return None


def fit_tokenisation_state(
    records: Sequence[BookEvent | OrderBookSnapshot],
    config: TokenisationConfig,
    split_indices: SplitIndices | None = None,
) -> TokenisationFitState:
    """Fit train-only tokenisation state.

    The current implementation keeps price, quantity, time and context buckets
    fixed. If ``config.fit_source_tokens`` is true, source vocabulary entries
    are fitted only from training rows when ``split_indices`` is supplied.
    """
    base_vocabulary = build_static_token_vocabulary(config)
    record_list = list(records)
    if split_indices is None:
        fit_indices = list(range(len(record_list)))
    else:
        fit_indices = list(split_indices.train)
        for position, index in enumerate(fit_indices):
            if index < 0 or index >= len(record_list):
                raise IndexError(
                    f"split_indices.train[{position}]={index} is out of range "
                    f"[0, {len(record_list)})"
                )

    fitted_sources: tuple[str, ...] = ()
    vocabulary = base_vocabulary
    if config.fit_source_tokens:
        fitted_sources = _normalise_source_values(
            source
            for index in fit_indices
            if (source := _record_source_value(record_list[index], config)) is not None
        )
        source_tokens = [source_value_to_token(source) for source in fitted_sources]
        vocabulary = vocabulary.with_added_tokens(TokenField.SOURCE, source_tokens)

    return TokenisationFitState(
        vocabulary=vocabulary,
        config=config,
        fitted_source_tokens=fitted_sources,
        n_fit_records=len(fit_indices),
    )


def event_type_token(event_type: EventType | None) -> str:
    """Map a canonical event type to its deterministic token label."""
    if event_type is None:
        return EVENT_TYPE_UNKNOWN
    if event_type is EventType.ADD:
        return "add"
    if event_type is EventType.CANCEL:
        return "cancel"
    if event_type is EventType.TRADE:
        return "trade"
    if event_type is EventType.MODIFY:
        return "modify"
    if event_type is EventType.DEPTH_UPDATE:
        return "depth_update"
    if event_type is EventType.SNAPSHOT:
        return EVENT_TYPE_SNAPSHOT
    return EVENT_TYPE_UNKNOWN


def side_token(side: Side | None) -> str:
    """Map a canonical side to its deterministic token label."""
    if side is None:
        return SIDE_NONE
    if side is Side.BID:
        return "bid"
    if side is Side.ASK:
        return "ask"
    return SIDE_UNKNOWN


def quantity_bucket_token(
    quantity: float | None,
    config: QuantityBucketConfig,
) -> str:
    """Return the fixed quantity bucket token for ``quantity``."""
    if quantity is None:
        return QUANTITY_MISSING
    if isinstance(quantity, bool) or not isinstance(quantity, (int, float)):
        raise TypeError("quantity must be a finite number or None")
    numeric = float(quantity)
    if not math.isfinite(numeric):
        raise ValueError("quantity must be finite")
    if numeric < 0.0:
        raise ValueError("quantity must be non-negative")
    if numeric == 0.0:
        return QUANTITY_ZERO
    for threshold, label in zip(
        config.positive_thresholds,
        config.positive_labels,
        strict=False,
    ):
        if numeric <= threshold:
            return label
    return config.positive_labels[-1]


def price_bucket_token(
    price: float | None,
    reference_price: float | None,
    config: PriceBucketConfig,
) -> str:
    """Return the fixed relative-price bucket token for ``price``."""
    if price is None or reference_price is None:
        return PRICE_MISSING
    if isinstance(price, bool) or not isinstance(price, (int, float)):
        raise TypeError("price must be a finite number or None")
    if isinstance(reference_price, bool) or not isinstance(reference_price, (int, float)):
        raise TypeError("reference_price must be a finite number or None")
    price_value = float(price)
    reference_value = float(reference_price)
    if not math.isfinite(price_value) or not math.isfinite(reference_value):
        raise ValueError("price and reference_price must be finite")
    if reference_value <= 0.0:
        return PRICE_MISSING
    relative_bps = 10_000.0 * (price_value - reference_value) / reference_value
    for threshold in config.relative_bps_thresholds:
        if relative_bps <= threshold:
            return f"rel_lte_{_format_threshold(threshold)}bp"
    return f"rel_gt_{_format_threshold(config.relative_bps_thresholds[-1])}bp"


def time_delta_bucket_token(
    timestamp: datetime,
    previous_timestamp: datetime | None,
    config: TimeDeltaBucketConfig,
) -> str:
    """Return the fixed time-delta bucket token for ``timestamp``."""
    current = ensure_utc_datetime(timestamp)
    if previous_timestamp is None:
        return TIME_DELTA_MISSING
    previous = ensure_utc_datetime(previous_timestamp)
    delta_seconds = (current - previous).total_seconds()
    if delta_seconds < 0.0:
        raise ValueError("timestamp must not be before previous_timestamp")
    if delta_seconds == 0.0:
        return TIME_DELTA_ZERO
    for threshold, label in zip(
        config.thresholds_seconds,
        config.labels,
        strict=False,
    ):
        if delta_seconds <= threshold:
            return label
    return config.labels[-1]


def context_bucket_token(
    spread: float | None,
    mid_price: float | None,
    config: ContextBucketConfig,
) -> str:
    """Return the fixed spread/context bucket token."""
    if spread is None:
        return CONTEXT_MISSING
    if isinstance(spread, bool) or not isinstance(spread, (int, float)):
        raise TypeError("spread must be a finite number or None")
    spread_value = float(spread)
    if not math.isfinite(spread_value):
        raise ValueError("spread must be finite")
    if spread_value < 0.0:
        return CONTEXT_CROSSED
    if spread_value == 0.0:
        return CONTEXT_ZERO
    if config.mode == "relative_mid_bps":
        if mid_price is None or mid_price <= 0.0:
            return CONTEXT_MISSING
        bucket_value = 10_000.0 * spread_value / mid_price
    else:
        bucket_value = spread_value
    for threshold, label in zip(config.thresholds, config.labels, strict=False):
        if bucket_value <= threshold:
            return label
    return config.labels[-1]


def _event_context_from_metadata(
    metadata: Mapping[str, MetadataScalar],
    config: ContextBucketConfig,
) -> str:
    if config.mode == "relative_mid_bps":
        for key in ("relative_spread_bps", "spread_bps"):
            value = metadata.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return _bucket_positive_context_value(float(value), config)
        return CONTEXT_MISSING

    value = metadata.get("spread")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return context_bucket_token(float(value), None, config)
    return CONTEXT_MISSING


def _bucket_positive_context_value(value: float, config: ContextBucketConfig) -> str:
    if not math.isfinite(value):
        raise ValueError("context value must be finite")
    if value < 0.0:
        return CONTEXT_CROSSED
    if value == 0.0:
        return CONTEXT_ZERO
    for threshold, label in zip(config.thresholds, config.labels, strict=False):
        if value <= threshold:
            return label
    return config.labels[-1]


def _snapshot_reference_price(
    snapshot: OrderBookSnapshot,
    side: Side,
    config: TokenisationConfig,
) -> float | None:
    if config.price_reference == "mid":
        return snapshot.mid_price
    if config.price_reference == "best_same_side":
        best_same = snapshot.best_bid if side is Side.BID else snapshot.best_ask
        return best_same.price if best_same is not None else None
    if config.price_reference == "best_opposite_side":
        best_opposite = snapshot.best_ask if side is Side.BID else snapshot.best_bid
        return best_opposite.price if best_opposite is not None else None
    raise ValueError(f"unsupported price_reference {config.price_reference!r}")


def _source_id(
    record: BookEvent | OrderBookSnapshot,
    vocabulary: TokenVocabulary,
    config: TokenisationConfig,
) -> tuple[int, str | None]:
    source_value = _record_source_value(record, config)
    if source_value is None:
        return (
            vocabulary.token_to_id(TokenField.SOURCE, SOURCE_UNKNOWN),
            None,
        )
    source_token = source_value_to_token(source_value)
    return vocabulary.token_to_id(TokenField.SOURCE, source_token), source_value


def _make_tokenised_record(
    *,
    vocabulary: TokenVocabulary,
    event_token: str,
    side_value_token: str,
    price_token: str,
    quantity_token: str,
    time_delta_token: str,
    context_token: str,
    source_id: int,
    position: int,
    timestamp: datetime,
    symbol: str,
    sequence_id: int | None,
    record_kind: str,
    token_role: str,
    source_record_index: int | None,
    level_index: int | None,
    source_value: str | None,
    is_snapshot_derived: bool,
) -> TokenisedRecord:
    return TokenisedRecord(
        event_type_id=vocabulary.token_to_id(TokenField.EVENT_TYPE, event_token),
        side_id=vocabulary.token_to_id(TokenField.SIDE, side_value_token),
        price_bucket_id=vocabulary.token_to_id(TokenField.PRICE_BUCKET, price_token),
        quantity_bucket_id=vocabulary.token_to_id(
            TokenField.QUANTITY_BUCKET,
            quantity_token,
        ),
        time_delta_bucket_id=vocabulary.token_to_id(
            TokenField.TIME_DELTA_BUCKET,
            time_delta_token,
        ),
        context_bucket_id=vocabulary.token_to_id(
            TokenField.CONTEXT_BUCKET,
            context_token,
        ),
        source_id=source_id,
        position=position,
        timestamp=timestamp,
        symbol=symbol,
        sequence_id=sequence_id,
        record_kind=record_kind,
        token_role=token_role,
        source_record_index=source_record_index,
        level_index=level_index,
        source_value=source_value,
        is_snapshot_derived=is_snapshot_derived,
    )


def tokenise_book_event(
    event: BookEvent,
    vocabulary: TokenVocabulary,
    config: TokenisationConfig,
    previous_timestamp: datetime | None = None,
    reference_price: float | None = None,
) -> list[TokenisedRecord]:
    """Tokenise one canonical :class:`BookEvent` directly.

    The function does not reconstruct the full book and does not infer future
    context. Spread/context is used only when safely present in the event's
    current metadata.
    """
    timestamp = ensure_utc_datetime(event.timestamp)
    source_id, source_value = _source_id(event, vocabulary, config)
    return [
        _make_tokenised_record(
            vocabulary=vocabulary,
            event_token=event_type_token(event.event_type),
            side_value_token=side_token(event.side),
            price_token=price_bucket_token(
                event.price,
                reference_price,
                config.price_buckets,
            ),
            quantity_token=quantity_bucket_token(
                event.quantity,
                config.quantity_buckets,
            ),
            time_delta_token=time_delta_bucket_token(
                timestamp,
                previous_timestamp,
                config.time_delta_buckets,
            ),
            context_token=_event_context_from_metadata(
                event.metadata,
                config.context_buckets,
            ),
            source_id=source_id,
            position=0,
            timestamp=timestamp,
            symbol=event.symbol,
            sequence_id=event.sequence_id,
            record_kind="book_event",
            token_role="event",
            source_record_index=None,
            level_index=None,
            source_value=source_value,
            is_snapshot_derived=False,
        )
    ]


def tokenise_snapshot(
    snapshot: OrderBookSnapshot,
    vocabulary: TokenVocabulary,
    config: TokenisationConfig,
    previous_timestamp: datetime | None = None,
) -> list[TokenisedRecord]:
    """Tokenise one :class:`OrderBookSnapshot` into deterministic records.

    The first emitted record is a snapshot summary. It is followed by
    synthetic snapshot-level representations: bids from best to deeper levels,
    then asks from best to deeper levels.
    """
    timestamp = ensure_utc_datetime(snapshot.timestamp)
    time_token = time_delta_bucket_token(
        timestamp,
        previous_timestamp,
        config.time_delta_buckets,
    )
    context_token = context_bucket_token(
        snapshot.spread,
        snapshot.mid_price,
        config.context_buckets,
    )
    source_id, source_value = _source_id(snapshot, vocabulary, config)
    records = [
        _make_tokenised_record(
            vocabulary=vocabulary,
            event_token=EVENT_TYPE_SNAPSHOT,
            side_value_token=SIDE_NONE,
            price_token=PRICE_MISSING,
            quantity_token=QUANTITY_MISSING,
            time_delta_token=time_token,
            context_token=context_token,
            source_id=source_id,
            position=0,
            timestamp=timestamp,
            symbol=snapshot.symbol,
            sequence_id=snapshot.sequence_id,
            record_kind="order_book_snapshot",
            token_role="snapshot_summary",
            source_record_index=None,
            level_index=None,
            source_value=source_value,
            is_snapshot_derived=True,
        )
    ]
    records.extend(
        _tokenise_snapshot_levels(
            snapshot=snapshot,
            side=Side.BID,
            levels=snapshot.bids[: config.max_levels_per_side],
            vocabulary=vocabulary,
            config=config,
            time_token=time_token,
            context_token=context_token,
            source_id=source_id,
            source_value=source_value,
            position_offset=len(records),
        )
    )
    records.extend(
        _tokenise_snapshot_levels(
            snapshot=snapshot,
            side=Side.ASK,
            levels=snapshot.asks[: config.max_levels_per_side],
            vocabulary=vocabulary,
            config=config,
            time_token=time_token,
            context_token=context_token,
            source_id=source_id,
            source_value=source_value,
            position_offset=len(records),
        )
    )
    return records


def _tokenise_snapshot_levels(
    *,
    snapshot: OrderBookSnapshot,
    side: Side,
    levels: Sequence[OrderBookLevel],
    vocabulary: TokenVocabulary,
    config: TokenisationConfig,
    time_token: str,
    context_token: str,
    source_id: int,
    source_value: str | None,
    position_offset: int,
) -> list[TokenisedRecord]:
    reference_price = _snapshot_reference_price(snapshot, side, config)
    output: list[TokenisedRecord] = []
    for level_index, level in enumerate(levels):
        output.append(
            _make_tokenised_record(
                vocabulary=vocabulary,
                event_token=EVENT_TYPE_SNAPSHOT_LEVEL,
                side_value_token=side_token(side),
                price_token=price_bucket_token(
                    level.price,
                    reference_price,
                    config.price_buckets,
                ),
                quantity_token=quantity_bucket_token(
                    level.quantity,
                    config.quantity_buckets,
                ),
                time_delta_token=time_token,
                context_token=context_token,
                source_id=source_id,
                position=position_offset + level_index,
                timestamp=snapshot.timestamp,
                symbol=snapshot.symbol,
                sequence_id=snapshot.sequence_id,
                record_kind="order_book_snapshot",
                token_role="snapshot_level",
                source_record_index=None,
                level_index=level_index,
                source_value=source_value,
                is_snapshot_derived=True,
            )
        )
    return output


def _record_sort_key(
    indexed_record: tuple[int, BookEvent | OrderBookSnapshot],
) -> tuple[datetime, bool, int, int]:
    original_index, record = indexed_record
    timestamp = ensure_utc_datetime(record.timestamp)
    sequence_id = record.sequence_id
    return (
        timestamp,
        sequence_id is None,
        0 if sequence_id is None else sequence_id,
        original_index,
    )


def _special_record(
    token: SpecialToken,
    *,
    position: int,
    token_role: str,
) -> TokenisedRecord:
    token_id = int(SPECIAL_TOKEN_IDS[token])
    return TokenisedRecord(
        event_type_id=token_id,
        side_id=token_id,
        price_bucket_id=token_id,
        quantity_bucket_id=token_id,
        time_delta_bucket_id=token_id,
        context_bucket_id=token_id,
        source_id=token_id,
        position=position,
        timestamp=None,
        symbol=None,
        sequence_id=None,
        record_kind="special",
        token_role=token_role,
        source_record_index=None,
        level_index=None,
        source_value=None,
        is_snapshot_derived=False,
    )


def tokenise_records(
    records: Sequence[BookEvent | OrderBookSnapshot],
    vocabulary: TokenVocabulary,
    config: TokenisationConfig,
) -> TokenSequence:
    """Tokenise canonical records into one deterministic token sequence."""
    indexed_records = list(enumerate(records))
    if config.sort_records:
        indexed_records = sorted(indexed_records, key=_record_sort_key)

    output: list[TokenisedRecord] = []
    if config.include_bos:
        output.append(
            _special_record(SpecialToken.BOS, position=0, token_role="bos")
        )

    previous_by_symbol: dict[str, datetime] = {}
    for source_record_index, record in indexed_records:
        timestamp = ensure_utc_datetime(record.timestamp)
        previous_timestamp = previous_by_symbol.get(record.symbol)
        if isinstance(record, OrderBookSnapshot):
            tokenised = tokenise_snapshot(
                record,
                vocabulary,
                config,
                previous_timestamp,
            )
        elif isinstance(record, BookEvent):
            tokenised = tokenise_book_event(
                record,
                vocabulary,
                config,
                previous_timestamp,
            )
        else:
            raise TypeError(
                "records must contain BookEvent or OrderBookSnapshot objects; "
                f"got {type(record).__name__}"
            )
        for tokenised_record in tokenised:
            output.append(
                replace(
                    tokenised_record,
                    position=len(output),
                    source_record_index=source_record_index,
                )
            )
        previous_by_symbol[record.symbol] = timestamp

    if config.include_eos:
        output.append(
            _special_record(
                SpecialToken.EOS,
                position=len(output),
                token_role="eos",
            )
        )

    return TokenSequence(
        records=tuple(output),
        vocabulary=vocabulary,
        config=config,
        input_record_count=len(records),
    )


def tokenise_event_log(
    path: Path,
    config: TokenisationConfig,
    symbol: str | None = None,
) -> TokenSequence:
    """Read and tokenise a local canonical JSONL event log."""
    file_path = Path(path)
    records: list[EventLogObject] = read_event_log_jsonl(file_path)
    if symbol is not None:
        records = filter_event_log_records(records, symbol=symbol)
    state = fit_tokenisation_state(records, config)
    sequence = tokenise_records(records, state.vocabulary, config)
    return TokenSequence(
        records=sequence.records,
        vocabulary=sequence.vocabulary,
        config=sequence.config,
        input_record_count=sequence.input_record_count,
        source_path=file_path,
    )


__all__ = [
    "CONTEXT_CROSSED",
    "CONTEXT_MISSING",
    "CONTEXT_ZERO",
    "EVENT_TYPE_SNAPSHOT",
    "EVENT_TYPE_SNAPSHOT_LEVEL",
    "EVENT_TYPE_UNKNOWN",
    "PRICE_MISSING",
    "QUANTITY_MISSING",
    "QUANTITY_ZERO",
    "SIDE_NONE",
    "SIDE_UNKNOWN",
    "SOURCE_UNKNOWN",
    "SPECIAL_TOKEN_IDS",
    "TOKEN_FIELDS",
    "ContextBucketConfig",
    "PriceBucketConfig",
    "QuantityBucketConfig",
    "SpecialToken",
    "TimeDeltaBucketConfig",
    "TokenField",
    "TokenSequence",
    "TokenVocabulary",
    "TokenisationConfig",
    "TokenisationFitState",
    "TokenisedRecord",
    "build_static_token_vocabulary",
    "context_bucket_token",
    "event_type_token",
    "fit_tokenisation_state",
    "price_bucket_token",
    "quantity_bucket_token",
    "side_token",
    "source_value_to_token",
    "time_delta_bucket_token",
    "tokenise_book_event",
    "tokenise_event_log",
    "tokenise_records",
    "tokenise_snapshot",
]
