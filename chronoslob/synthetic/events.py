"""Deterministic synthetic limit-order-book event generation.

The generator emits a strictly increasing stream of canonical
:class:`~chronoslob.data.schemas.BookEvent` records (``ADD``, ``CANCEL`` and
``TRADE``) under a sequence of known regimes. It maintains its own coherent
book while emitting so that the stream never produces a crossed book on replay
and every event carries its ground-truth regime label.

All output is synthetic. It exists to validate event-level pipeline support and
to provide controlled stress tests; it is not real-market data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import numpy as np

from chronoslob.data.schemas import BookEvent, EventType, Side

__all__ = [
    "REGIME_LIBRARY",
    "RegimeSpec",
    "SyntheticEventConfig",
    "SyntheticGenerationResult",
    "default_regime_plan",
    "generate_synthetic_events",
]

# Fixed epoch for synthetic timestamps. The absolute value is arbitrary; only
# the strictly increasing ordering matters for the event-level pipeline.
_EPOCH = datetime(2020, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class RegimeSpec:
    """Controlled generation parameters for one known synthetic regime.

    Intensities are unnormalised relative weights for sampling the next event
    type. ``buy_fraction`` controls bid/ask and aggressor imbalance.
    ``spread_ticks`` and ``volatility_ticks`` shape the half-spread placement
    and the latent mid random walk. ``depth_levels`` controls how far from the
    touch new liquidity is concentrated.
    """

    name: str
    regime_id: int
    add_intensity: float
    cancel_intensity: float
    trade_intensity: float
    buy_fraction: float
    spread_ticks: int
    volatility_ticks: float
    depth_levels: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("regime name must be non-empty")
        for value in (self.add_intensity, self.cancel_intensity, self.trade_intensity):
            if value < 0.0:
                raise ValueError("regime intensities must be non-negative")
        if (self.add_intensity + self.cancel_intensity + self.trade_intensity) <= 0.0:
            raise ValueError("regime must have at least one positive intensity")
        if not 0.0 <= self.buy_fraction <= 1.0:
            raise ValueError("buy_fraction must be in [0, 1]")
        if self.spread_ticks < 1:
            raise ValueError("spread_ticks must be >= 1")
        if self.volatility_ticks < 0.0:
            raise ValueError("volatility_ticks must be non-negative")
        if self.depth_levels < 1:
            raise ValueError("depth_levels must be >= 1")


# Library of known regimes. Each entry has a stable integer id so labels and
# diagnostics can group by regime deterministically.
REGIME_LIBRARY: dict[str, RegimeSpec] = {
    "stable_liquid": RegimeSpec(
        name="stable_liquid",
        regime_id=0,
        add_intensity=6.0,
        cancel_intensity=3.0,
        trade_intensity=1.0,
        buy_fraction=0.5,
        spread_ticks=2,
        volatility_ticks=0.4,
        depth_levels=5,
    ),
    "high_volatility": RegimeSpec(
        name="high_volatility",
        regime_id=1,
        add_intensity=5.0,
        cancel_intensity=4.0,
        trade_intensity=2.0,
        buy_fraction=0.5,
        spread_ticks=3,
        volatility_ticks=1.8,
        depth_levels=6,
    ),
    "low_liquidity": RegimeSpec(
        name="low_liquidity",
        regime_id=2,
        add_intensity=3.0,
        cancel_intensity=4.0,
        trade_intensity=1.0,
        buy_fraction=0.5,
        spread_ticks=4,
        volatility_ticks=0.9,
        depth_levels=3,
    ),
    "wide_spread": RegimeSpec(
        name="wide_spread",
        regime_id=3,
        add_intensity=5.0,
        cancel_intensity=3.0,
        trade_intensity=1.0,
        buy_fraction=0.5,
        spread_ticks=8,
        volatility_ticks=0.7,
        depth_levels=5,
    ),
    "buy_pressure": RegimeSpec(
        name="buy_pressure",
        regime_id=4,
        add_intensity=6.0,
        cancel_intensity=3.0,
        trade_intensity=2.0,
        buy_fraction=0.72,
        spread_ticks=2,
        volatility_ticks=0.8,
        depth_levels=5,
    ),
    "sell_pressure": RegimeSpec(
        name="sell_pressure",
        regime_id=5,
        add_intensity=6.0,
        cancel_intensity=3.0,
        trade_intensity=2.0,
        buy_fraction=0.28,
        spread_ticks=2,
        volatility_ticks=0.8,
        depth_levels=5,
    ),
    "cancellation_shock": RegimeSpec(
        name="cancellation_shock",
        regime_id=6,
        add_intensity=3.0,
        cancel_intensity=9.0,
        trade_intensity=1.0,
        buy_fraction=0.5,
        spread_ticks=3,
        volatility_ticks=1.0,
        depth_levels=4,
    ),
}


def default_regime_plan(events_per_regime: int) -> tuple[tuple[str, int], ...]:
    """Return the default ordered regime plan used by the demo pipeline."""
    if events_per_regime < 1:
        raise ValueError("events_per_regime must be >= 1")
    order = (
        "stable_liquid",
        "buy_pressure",
        "high_volatility",
        "sell_pressure",
        "low_liquidity",
        "wide_spread",
        "cancellation_shock",
    )
    return tuple((name, events_per_regime) for name in order)


@dataclass(frozen=True)
class SyntheticEventConfig:
    """Configuration for deterministic synthetic event generation."""

    symbol: str = "SYNTH"
    seed: int = 0
    tick_size: float = 0.01
    initial_mid_ticks: int = 10_000
    base_quantity: float = 5.0
    max_levels_per_side: int = 10
    regime_plan: tuple[tuple[str, int], ...] = field(
        default_factory=lambda: default_regime_plan(3_000)
    )

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must be non-empty")
        if self.tick_size <= 0.0:
            raise ValueError("tick_size must be positive")
        if self.initial_mid_ticks <= self.max_levels_per_side + 10:
            raise ValueError("initial_mid_ticks must be comfortably positive")
        if self.base_quantity <= 0.0:
            raise ValueError("base_quantity must be positive")
        if self.max_levels_per_side < 1:
            raise ValueError("max_levels_per_side must be >= 1")
        if not self.regime_plan:
            raise ValueError("regime_plan must contain at least one regime")
        for name, count in self.regime_plan:
            if name not in REGIME_LIBRARY:
                raise ValueError(f"unknown regime {name!r}")
            if count < 1:
                raise ValueError("regime event counts must be >= 1")

    @property
    def total_events(self) -> int:
        """Return the total number of events the plan will emit."""
        return sum(count for _, count in self.regime_plan)


@dataclass(frozen=True)
class SyntheticGenerationResult:
    """Result of a deterministic synthetic event generation run."""

    events: list[BookEvent]
    config: SyntheticEventConfig
    regime_event_counts: dict[str, int]
    event_type_counts: dict[str, int]

    @property
    def event_count(self) -> int:
        """Return the number of generated events."""
        return len(self.events)


class _GeneratorBook:
    """Minimal integer-tick book used while emitting a coherent stream.

    The book keeps every resting level (no silent trimming) so that the emitted
    event stream alone fully determines the book on replay. New orders are
    clamped by the caller so the book never crosses.
    """

    def __init__(self) -> None:
        self.bids: dict[int, float] = {}
        self.asks: dict[int, float] = {}

    def best_bid(self) -> int | None:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> int | None:
        return min(self.asks) if self.asks else None

    def add(self, side: Side, price_tick: int, quantity: float) -> None:
        book = self.bids if side is Side.BID else self.asks
        book[price_tick] = book.get(price_tick, 0.0) + quantity

    def reduce(self, side: Side, price_tick: int, quantity: float) -> float:
        book = self.bids if side is Side.BID else self.asks
        resting = book.get(price_tick)
        if resting is None:
            return 0.0
        removed = min(resting, quantity)
        remaining = resting - removed
        if remaining <= 1e-9:
            del book[price_tick]
        else:
            book[price_tick] = remaining
        return removed


def generate_synthetic_events(
    config: SyntheticEventConfig | None = None,
) -> SyntheticGenerationResult:
    """Generate a deterministic synthetic event stream for ``config``.

    The same ``config`` (notably the same ``seed``) always yields byte-for-byte
    identical events. The emitted stream is internally coherent: bid prices stay
    strictly below ask prices, quantities are positive and the sequence id is
    strictly increasing.
    """
    config = config or SyntheticEventConfig()
    rng = np.random.default_rng(config.seed)
    book = _GeneratorBook()

    events: list[BookEvent] = []
    regime_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {"ADD": 0, "CANCEL": 0, "TRADE": 0}
    sequence_id = _seed_initial_book(book, config, rng, events, type_counts, regime_counts)
    mid = float(config.initial_mid_ticks)

    for regime_name, count in config.regime_plan:
        spec = REGIME_LIBRARY[regime_name]
        weights = np.array(
            [spec.add_intensity, spec.cancel_intensity, spec.trade_intensity],
            dtype=float,
        )
        weights = weights / weights.sum()
        emitted = 0
        guard = 0
        max_guard = count * 12 + 64
        while emitted < count and guard < max_guard:
            guard += 1
            mid += float(rng.normal(0.0, spec.volatility_ticks))
            mid = max(mid, float(config.max_levels_per_side + 5))
            choice = int(rng.choice(3, p=weights))
            event = _emit_event(
                choice=choice,
                spec=spec,
                book=book,
                config=config,
                rng=rng,
                mid=mid,
                sequence_id=sequence_id,
            )
            if event is None:
                continue
            events.append(event)
            type_counts[event.event_type.value] += 1
            regime_counts[regime_name] = regime_counts.get(regime_name, 0) + 1
            sequence_id += 1
            emitted += 1

    return SyntheticGenerationResult(
        events=events,
        config=config,
        regime_event_counts=regime_counts,
        event_type_counts=type_counts,
    )


def _seed_initial_book(
    book: _GeneratorBook,
    config: SyntheticEventConfig,
    rng: np.random.Generator,
    events: list[BookEvent],
    type_counts: dict[str, int],
    regime_counts: dict[str, int],
) -> int:
    """Seed a symmetric initial book, emitting the liquidity as ADD events.

    Emitting the seed as events keeps replay fully determined by the stream.
    Returns the next free sequence id.
    """
    center = config.initial_mid_ticks
    spec = REGIME_LIBRARY[config.regime_plan[0][0]]
    half_spread = 1
    sequence_id = 0
    levels = min(config.max_levels_per_side, 5)
    for depth in range(levels):
        for side, offset in (
            (Side.BID, -half_spread - depth),
            (Side.ASK, half_spread + depth),
        ):
            quantity = round(config.base_quantity * (1.0 + float(rng.random())), 4)
            price_tick = center + offset
            book.add(side, price_tick, quantity)
            events.append(
                _build_event(
                    EventType.ADD,
                    side,
                    price_tick,
                    quantity,
                    spec,
                    config,
                    sequence_id,
                    float(center),
                )
            )
            type_counts["ADD"] += 1
            regime_counts[spec.name] = regime_counts.get(spec.name, 0) + 1
            sequence_id += 1
    return sequence_id


def _emit_event(
    *,
    choice: int,
    spec: RegimeSpec,
    book: _GeneratorBook,
    config: SyntheticEventConfig,
    rng: np.random.Generator,
    mid: float,
    sequence_id: int,
) -> BookEvent | None:
    if choice == 0:
        return _emit_add(spec, book, config, rng, mid, sequence_id)
    if choice == 1:
        return _emit_cancel(spec, book, config, rng, mid, sequence_id)
    return _emit_trade(spec, book, config, rng, mid, sequence_id)


def _quantity(config: SyntheticEventConfig, rng: np.random.Generator) -> float:
    return round(config.base_quantity * (1.0 + float(rng.exponential(0.6))), 4)


def _emit_add(
    spec: RegimeSpec,
    book: _GeneratorBook,
    config: SyntheticEventConfig,
    rng: np.random.Generator,
    mid: float,
    sequence_id: int,
) -> BookEvent:
    center = round(mid)
    half_spread = max(1, spec.spread_ticks // 2)
    side = Side.BID if float(rng.random()) < spec.buy_fraction else Side.ASK
    depth = int(rng.integers(0, spec.depth_levels))
    if side is Side.BID:
        price_tick = center - half_spread - depth
        best_ask = book.best_ask()
        if best_ask is not None:
            # Clamp below the best ask so the book never crosses on replay.
            price_tick = min(price_tick, best_ask - 1)
    else:
        price_tick = center + half_spread + depth
        best_bid = book.best_bid()
        if best_bid is not None:
            price_tick = max(price_tick, best_bid + 1)
    quantity = _quantity(config, rng)
    book.add(side, price_tick, quantity)
    return _build_event(
        EventType.ADD, side, price_tick, quantity, spec, config, sequence_id, mid
    )


def _emit_cancel(
    spec: RegimeSpec,
    book: _GeneratorBook,
    config: SyntheticEventConfig,
    rng: np.random.Generator,
    mid: float,
    sequence_id: int,
) -> BookEvent | None:
    side = Side.BID if float(rng.random()) < spec.buy_fraction else Side.ASK
    resting = book.bids if side is Side.BID else book.asks
    if not resting:
        side = Side.ASK if side is Side.BID else Side.BID
        resting = book.bids if side is Side.BID else book.asks
    if not resting:
        return None
    prices = sorted(resting)
    price_tick = int(prices[int(rng.integers(0, len(prices)))])
    available = resting[price_tick]
    fraction = 0.4 + 0.6 * float(rng.random())
    quantity = round(min(available, max(available * fraction, 1e-3)), 4)
    removed = book.reduce(side, price_tick, quantity)
    if removed <= 0.0:
        return None
    return _build_event(
        EventType.CANCEL, side, price_tick, round(removed, 4), spec, config, sequence_id, mid
    )


def _emit_trade(
    spec: RegimeSpec,
    book: _GeneratorBook,
    config: SyntheticEventConfig,
    rng: np.random.Generator,
    mid: float,
    sequence_id: int,
) -> BookEvent | None:
    # Aggressor BID (buyer-initiated) lifts the resting ASK; aggressor ASK
    # (seller-initiated) hits the resting BID.
    aggressor = Side.BID if float(rng.random()) < spec.buy_fraction else Side.ASK
    passive = Side.ASK if aggressor is Side.BID else Side.BID
    resting = book.asks if passive is Side.ASK else book.bids
    if not resting:
        return None
    price_tick = min(resting) if passive is Side.ASK else max(resting)
    available = resting[price_tick]
    quantity = round(min(available, _quantity(config, rng)), 4)
    removed = book.reduce(passive, price_tick, quantity)
    if removed <= 0.0:
        return None
    return _build_event(
        EventType.TRADE,
        aggressor,
        price_tick,
        round(removed, 4),
        spec,
        config,
        sequence_id,
        mid,
        aggressor_side=aggressor.value,
    )


def _build_event(
    event_type: EventType,
    side: Side,
    price_tick: int,
    quantity: float,
    spec: RegimeSpec,
    config: SyntheticEventConfig,
    sequence_id: int,
    mid: float,
    *,
    aggressor_side: str | None = None,
) -> BookEvent:
    metadata: dict[str, str | int | float | bool] = {
        "regime_id": spec.regime_id,
        "regime_name": spec.name,
        "latent_mid": round(mid * config.tick_size, 6),
        "synthetic": True,
    }
    if aggressor_side is not None:
        metadata["aggressor_side"] = aggressor_side
    return BookEvent(
        timestamp=_EPOCH + timedelta(milliseconds=sequence_id),
        event_type=event_type,
        symbol=config.symbol,
        side=side,
        price=round(price_tick * config.tick_size, 6),
        quantity=quantity,
        sequence_id=sequence_id,
        metadata=metadata,
    )


def regime_ids_in_plan(plan: Sequence[tuple[str, int]]) -> tuple[int, ...]:
    """Return the distinct ordered regime ids referenced by ``plan``."""
    seen: list[int] = []
    for name, _ in plan:
        regime_id = REGIME_LIBRARY[name].regime_id
        if regime_id not in seen:
            seen.append(regime_id)
    return tuple(seen)
