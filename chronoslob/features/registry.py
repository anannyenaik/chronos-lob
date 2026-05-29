"""Microstructure feature-group registry for FI-2010 experiments.

FI-2010 NoAuction snapshots expose normalised limit-order-book levels and
labels.  They do not expose event messages, trades, cancellations or queue
position.  This registry therefore separates supported snapshot features from
unsupported event-level concepts and labels snapshot-to-snapshot changes as
proxies rather than true order-flow features.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

FeatureKind = Literal["raw", "derived", "rolling", "proxy", "unsupported"]
ResolutionStatus = Literal["available", "missing_columns", "unsupported"]

__all__ = [
    "DEFAULT_FI2010_LEVELS",
    "DEFAULT_REGISTRY_GROUPS",
    "FeatureGroupResolution",
    "FeatureGroupSpec",
    "FeatureRegistryError",
    "available_lob_levels",
    "feature_groups_for_columns",
    "feature_manifest",
    "get_feature_group",
    "group_names",
    "proxy_group_names",
    "supported_fi2010_group_names",
    "unsupported_group_names",
    "validate_requested_groups",
]

DEFAULT_FI2010_LEVELS = 10


class FeatureRegistryError(ValueError):
    """Raised when strict registry validation fails."""


@dataclass(frozen=True)
class FeatureGroupSpec:
    """Static registry entry for a microstructure feature group."""

    name: str
    description: str
    required_source_columns: tuple[str, ...]
    generated_columns: tuple[str, ...]
    kind: FeatureKind
    requires_past_context: bool
    valid_for_fi2010: bool
    limitations: tuple[str, ...]

    @property
    def is_proxy(self) -> bool:
        """Return whether this group is a labelled proxy."""
        return self.kind == "proxy"


@dataclass(frozen=True)
class FeatureGroupResolution:
    """Resolved group-to-column mapping for one concrete input schema."""

    name: str
    description: str
    source_columns: tuple[str, ...]
    generated_columns: tuple[str, ...]
    kind: FeatureKind
    requires_past_context: bool
    valid_for_fi2010: bool
    limitations: tuple[str, ...]
    status: ResolutionStatus
    reason: str

    @property
    def has_columns(self) -> bool:
        """Return whether the group maps to at least one source or output column."""
        return bool(self.source_columns or self.generated_columns)

    @property
    def is_proxy(self) -> bool:
        """Return whether the resolved group is a labelled proxy."""
        return self.kind == "proxy"


def _level_columns(prefix: str, *, levels: int = DEFAULT_FI2010_LEVELS) -> tuple[str, ...]:
    return tuple(f"{prefix}{level}" for level in range(1, levels + 1))


def _book_columns(
    *,
    price: bool,
    size: bool,
    levels: int = DEFAULT_FI2010_LEVELS,
) -> tuple[str, ...]:
    columns: list[str] = []
    for level in range(1, levels + 1):
        if price:
            columns.extend((f"bid_price_{level}", f"ask_price_{level}"))
        if size:
            columns.extend((f"bid_quantity_{level}", f"ask_quantity_{level}"))
    return tuple(columns)


def _snapshot_delta_columns(levels: int = DEFAULT_FI2010_LEVELS) -> tuple[str, ...]:
    return tuple(
        f"snapshot_delta_{column}" for column in _book_columns(price=True, size=True, levels=levels)
    )


DEFAULT_REGISTRY_GROUPS: tuple[FeatureGroupSpec, ...] = (
    FeatureGroupSpec(
        name="price_levels",
        description="Raw bid and ask price columns at each visible book level.",
        required_source_columns=_book_columns(price=True, size=False),
        generated_columns=_book_columns(price=True, size=False),
        kind="raw",
        requires_past_context=False,
        valid_for_fi2010=True,
        limitations=(
            "FI-2010 NoAuction prices are commonly normalised benchmark values, "
            "not necessarily exchange-native prices.",
        ),
    ),
    FeatureGroupSpec(
        name="size_levels",
        description="Raw bid and ask size columns at each visible book level.",
        required_source_columns=_book_columns(price=False, size=True),
        generated_columns=_book_columns(price=False, size=True),
        kind="raw",
        requires_past_context=False,
        valid_for_fi2010=True,
        limitations=(
            "Size columns may be named bid_quantity_N/ask_quantity_N or "
            "bid_size_N/ask_size_N depending on the mirror.",
        ),
    ),
    FeatureGroupSpec(
        name="top_of_book",
        description="Best bid/ask prices and sizes.",
        required_source_columns=(
            "bid_price_1",
            "ask_price_1",
            "bid_quantity_1",
            "ask_quantity_1",
        ),
        generated_columns=(
            "best_bid_price",
            "best_ask_price",
            "best_bid_size",
            "best_ask_size",
        ),
        kind="raw",
        requires_past_context=False,
        valid_for_fi2010=True,
        limitations=("Top-of-book values remain snapshot values, not event updates.",),
    ),
    FeatureGroupSpec(
        name="spread",
        description="Absolute and relative spread from best ask and best bid.",
        required_source_columns=("bid_price_1", "ask_price_1"),
        generated_columns=("spread", "relative_spread"),
        kind="derived",
        requires_past_context=False,
        valid_for_fi2010=True,
        limitations=(
            "Relative spread is emitted only when the contemporaneous midprice is finite "
            "and non-zero.",
        ),
    ),
    FeatureGroupSpec(
        name="midprice",
        description="Midpoint of best bid and best ask.",
        required_source_columns=("bid_price_1", "ask_price_1"),
        generated_columns=("midprice",),
        kind="derived",
        requires_past_context=False,
        valid_for_fi2010=True,
        limitations=("Computed from the current snapshot only.",),
    ),
    FeatureGroupSpec(
        name="microprice",
        description="Top-of-book size-weighted price.",
        required_source_columns=(
            "bid_price_1",
            "ask_price_1",
            "bid_quantity_1",
            "ask_quantity_1",
        ),
        generated_columns=("microprice",),
        kind="derived",
        requires_past_context=False,
        valid_for_fi2010=True,
        limitations=("Undefined denominators are emitted as missing values before final filling.",),
    ),
    FeatureGroupSpec(
        name="top_of_book_imbalance",
        description="Level-1 bid/ask size imbalance.",
        required_source_columns=("bid_quantity_1", "ask_quantity_1"),
        generated_columns=("top_of_book_imbalance",),
        kind="derived",
        requires_past_context=False,
        valid_for_fi2010=True,
        limitations=("Uses displayed level-1 size only.",),
    ),
    FeatureGroupSpec(
        name="depth_imbalance",
        description="Multi-level displayed-depth imbalance.",
        required_source_columns=_book_columns(price=False, size=True),
        generated_columns=(
            "depth_imbalance_l1",
            "depth_imbalance_l5",
            "depth_imbalance_l10",
        ),
        kind="derived",
        requires_past_context=False,
        valid_for_fi2010=True,
        limitations=("Uses visible snapshot depth only; no hidden liquidity is observed.",),
    ),
    FeatureGroupSpec(
        name="depth_slope",
        description="Visible-depth slope and concentration proxy across levels.",
        required_source_columns=_book_columns(price=False, size=True),
        generated_columns=(
            "bid_depth_slope",
            "ask_depth_slope",
            "depth_slope_imbalance",
        ),
        kind="derived",
        requires_past_context=False,
        valid_for_fi2010=True,
        limitations=("A snapshot-level shape proxy, not queue dynamics.",),
    ),
    FeatureGroupSpec(
        name="liquidity_concentration",
        description="Fraction of visible depth concentrated near the top levels.",
        required_source_columns=_book_columns(price=False, size=True),
        generated_columns=(
            "liquidity_concentration_top1",
            "liquidity_concentration_top5",
        ),
        kind="derived",
        requires_past_context=False,
        valid_for_fi2010=True,
        limitations=("Only displayed FI-2010 levels are considered.",),
    ),
    FeatureGroupSpec(
        name="snapshot_order_flow_proxy",
        description=(
            "Snapshot-to-snapshot changes in visible prices and sizes. This is a proxy, "
            "not true order-flow imbalance."
        ),
        required_source_columns=_book_columns(price=True, size=True),
        generated_columns=_snapshot_delta_columns(),
        kind="proxy",
        requires_past_context=True,
        valid_for_fi2010=True,
        limitations=(
            "Does not identify submissions, cancellations or trades.",
            "Deltas must be reset at fold and partition boundaries.",
        ),
    ),
    FeatureGroupSpec(
        name="volatility_proxy",
        description="Rolling realised-volatility proxy from past midprice changes.",
        required_source_columns=("bid_price_1", "ask_price_1"),
        generated_columns=("volatility_proxy",),
        kind="rolling",
        requires_past_context=True,
        valid_for_fi2010=True,
        limitations=("Uses current and past rows only; no future horizon columns are used.",),
    ),
    FeatureGroupSpec(
        name="time_context",
        description="Timestamp/session context when a real timestamp or session column exists.",
        required_source_columns=("timestamp",),
        generated_columns=("time_of_day_seconds", "session_position"),
        kind="derived",
        requires_past_context=False,
        valid_for_fi2010=False,
        limitations=(
            "Canonical FI-2010 NoAuction matrices do not include true timestamp/session "
            "information; this group is skipped unless such columns are explicitly present.",
        ),
    ),
    FeatureGroupSpec(
        name="true_order_flow_imbalance",
        description="True event-level order-flow imbalance.",
        required_source_columns=(),
        generated_columns=(),
        kind="unsupported",
        requires_past_context=True,
        valid_for_fi2010=False,
        limitations=("Unsupported for normalised FI-2010 snapshots without event messages.",),
    ),
    FeatureGroupSpec(
        name="cancellation_imbalance",
        description="Cancellation imbalance from event messages.",
        required_source_columns=(),
        generated_columns=(),
        kind="unsupported",
        requires_past_context=True,
        valid_for_fi2010=False,
        limitations=("Unsupported unless cancellation events are directly observed.",),
    ),
    FeatureGroupSpec(
        name="trade_imbalance",
        description="Trade imbalance from executed trade events.",
        required_source_columns=(),
        generated_columns=(),
        kind="unsupported",
        requires_past_context=True,
        valid_for_fi2010=False,
        limitations=("Unsupported unless trades are directly observed.",),
    ),
    FeatureGroupSpec(
        name="queue_position",
        description="Queue-position and fill-priority information.",
        required_source_columns=(),
        generated_columns=(),
        kind="unsupported",
        requires_past_context=True,
        valid_for_fi2010=False,
        limitations=("Unsupported for FI-2010 snapshots; queue position is not observed.",),
    ),
)

_REGISTRY: dict[str, FeatureGroupSpec] = {group.name: group for group in DEFAULT_REGISTRY_GROUPS}


def group_names() -> tuple[str, ...]:
    """Return all registered group names."""
    return tuple(_REGISTRY)


def supported_fi2010_group_names() -> tuple[str, ...]:
    """Return groups that are conceptually valid for FI-2010 snapshots."""
    return tuple(
        group.name
        for group in DEFAULT_REGISTRY_GROUPS
        if group.valid_for_fi2010 and group.kind != "unsupported"
    )


def unsupported_group_names() -> tuple[str, ...]:
    """Return groups that are explicitly unsupported for canonical FI-2010."""
    return tuple(
        group.name
        for group in DEFAULT_REGISTRY_GROUPS
        if not group.valid_for_fi2010 or group.kind == "unsupported"
    )


def proxy_group_names() -> tuple[str, ...]:
    """Return registered proxy group names."""
    return tuple(group.name for group in DEFAULT_REGISTRY_GROUPS if group.is_proxy)


def get_feature_group(name: str) -> FeatureGroupSpec:
    """Return the registry spec for ``name``."""
    cleaned = name.strip().lower()
    if not cleaned:
        raise FeatureRegistryError("feature group name must be non-empty")
    try:
        return _REGISTRY[cleaned]
    except KeyError as exc:
        raise FeatureRegistryError(
            f"unknown feature group {name!r}; supported registry names: {list(group_names())}"
        ) from exc


def _available_set(columns: Sequence[str]) -> set[str]:
    return {str(column) for column in columns}


def _quantity_column(available: set[str], side: str, level: int) -> str | None:
    primary = f"{side}_quantity_{level}"
    alias = f"{side}_size_{level}"
    if primary in available:
        return primary
    if alias in available:
        return alias
    return None


def _price_column(available: set[str], side: str, level: int) -> str | None:
    candidate = f"{side}_price_{level}"
    return candidate if candidate in available else None


def available_lob_levels(
    columns: Sequence[str], *, max_level: int = DEFAULT_FI2010_LEVELS
) -> tuple[int, ...]:
    """Return levels with bid/ask price and size columns available."""
    available = _available_set(columns)
    levels: list[int] = []
    for level in range(1, max_level + 1):
        if (
            _price_column(available, "bid", level) is not None
            and _price_column(available, "ask", level) is not None
            and _quantity_column(available, "bid", level) is not None
            and _quantity_column(available, "ask", level) is not None
        ):
            levels.append(level)
    return tuple(levels)


def _size_columns(available: set[str], levels: Sequence[int]) -> tuple[str, ...]:
    columns: list[str] = []
    for level in levels:
        bid = _quantity_column(available, "bid", level)
        ask = _quantity_column(available, "ask", level)
        if bid is not None and ask is not None:
            columns.extend((bid, ask))
    return tuple(columns)


def _price_columns(available: set[str], levels: Sequence[int]) -> tuple[str, ...]:
    columns: list[str] = []
    for level in levels:
        bid = _price_column(available, "bid", level)
        ask = _price_column(available, "ask", level)
        if bid is not None and ask is not None:
            columns.extend((bid, ask))
    return tuple(columns)


def _resolve_group(name: str, columns: Sequence[str]) -> FeatureGroupResolution:
    spec = get_feature_group(name)
    available = _available_set(columns)
    levels = available_lob_levels(columns)

    if spec.kind == "unsupported":
        return FeatureGroupResolution(
            name=spec.name,
            description=spec.description,
            source_columns=(),
            generated_columns=(),
            kind=spec.kind,
            requires_past_context=spec.requires_past_context,
            valid_for_fi2010=spec.valid_for_fi2010,
            limitations=spec.limitations,
            status="unsupported",
            reason="event-level source fields are not available in FI-2010 snapshots",
        )

    if spec.name == "price_levels":
        source = _price_columns(available, levels)
        generated = source
    elif spec.name == "size_levels":
        source = _size_columns(available, levels)
        generated = source
    elif spec.name == "top_of_book":
        if 1 in levels:
            bid_size = _quantity_column(available, "bid", 1)
            ask_size = _quantity_column(available, "ask", 1)
            source = (
                "bid_price_1",
                "ask_price_1",
                bid_size or "bid_quantity_1",
                ask_size or "ask_quantity_1",
            )
            generated = spec.generated_columns
        else:
            source = ()
            generated = ()
    elif spec.name in {"spread", "midprice", "volatility_proxy"}:
        source = (
            ("bid_price_1", "ask_price_1") if {"bid_price_1", "ask_price_1"} <= available else ()
        )
        generated = spec.generated_columns if source else ()
    elif spec.name in {"microprice", "top_of_book_imbalance"}:
        bid_size = _quantity_column(available, "bid", 1)
        ask_size = _quantity_column(available, "ask", 1)
        if bid_size is not None and ask_size is not None:
            if spec.name == "microprice":
                has_prices = {"bid_price_1", "ask_price_1"} <= available
                source = ("bid_price_1", "ask_price_1", bid_size, ask_size) if has_prices else ()
            else:
                source = (bid_size, ask_size)
        else:
            source = ()
        generated = spec.generated_columns if source else ()
    elif spec.name == "depth_imbalance":
        source = _size_columns(available, levels)
        generated = tuple(
            f"depth_imbalance_l{depth}" for depth in (1, 5, 10) if len(levels) >= depth
        )
    elif spec.name == "depth_slope":
        source = _size_columns(available, levels)
        generated = spec.generated_columns if len(levels) >= 2 else ()
    elif spec.name == "liquidity_concentration":
        source = _size_columns(available, levels)
        generated_list = ["liquidity_concentration_top1"] if levels else []
        if len(levels) >= 5:
            generated_list.append("liquidity_concentration_top5")
        generated = tuple(generated_list)
    elif spec.name == "snapshot_order_flow_proxy":
        source = (*_price_columns(available, levels), *_size_columns(available, levels))
        generated = tuple(f"snapshot_delta_{column}" for column in source)
    elif spec.name == "time_context":
        timestamp_candidates = tuple(
            column
            for column in ("timestamp", "event_time", "datetime", "session_id")
            if column in available
        )
        source = timestamp_candidates
        generated = spec.generated_columns if source else ()
    else:
        source = tuple(column for column in spec.required_source_columns if column in available)
        generated = spec.generated_columns if source else ()

    if source or generated:
        status: ResolutionStatus = "available"
        reason = "resolved"
    else:
        status = "missing_columns"
        reason = "required source columns were not present"

    return FeatureGroupResolution(
        name=spec.name,
        description=spec.description,
        source_columns=tuple(dict.fromkeys(source)),
        generated_columns=tuple(dict.fromkeys(generated)),
        kind=spec.kind,
        requires_past_context=spec.requires_past_context,
        valid_for_fi2010=spec.valid_for_fi2010,
        limitations=spec.limitations,
        status=status,
        reason=reason,
    )


def _normalise_requested_groups(requested_groups: Sequence[str] | str | None) -> tuple[str, ...]:
    if requested_groups is None:
        return supported_fi2010_group_names()
    if isinstance(requested_groups, str):
        text = requested_groups.strip()
        if not text or text.lower() == "all":
            return supported_fi2010_group_names()
        raw: Sequence[str] = [token.strip() for token in text.split(",")]
    else:
        raw = requested_groups
    cleaned: list[str] = []
    for group in raw:
        name = str(group).strip().lower()
        if not name:
            continue
        get_feature_group(name)
        if name not in cleaned:
            cleaned.append(name)
    if not cleaned:
        raise FeatureRegistryError("at least one feature group must be requested")
    return tuple(cleaned)


def feature_groups_for_columns(
    columns: Sequence[str],
    requested_groups: Sequence[str] | str | None = None,
    *,
    strict: bool = False,
) -> dict[str, FeatureGroupResolution]:
    """Resolve requested feature groups against concrete input columns."""
    requested = _normalise_requested_groups(requested_groups)
    resolutions = {name: _resolve_group(name, columns) for name in requested}
    if strict:
        validate_requested_groups(resolutions)
    return resolutions


def validate_requested_groups(
    resolutions: Mapping[str, FeatureGroupResolution],
) -> None:
    """Fail when a requested group is unsupported or maps to no columns."""
    failures: list[str] = []
    for name, resolution in resolutions.items():
        if resolution.status == "unsupported":
            failures.append(f"{name}: unsupported for FI-2010 ({resolution.reason})")
        elif not resolution.has_columns:
            failures.append(f"{name}: maps to no source or generated columns")
    if failures:
        raise FeatureRegistryError("; ".join(failures))


def feature_manifest(
    columns: Sequence[str],
    requested_groups: Sequence[str] | str | None = None,
    *,
    strict: bool = False,
) -> dict[str, object]:
    """Return a JSON-ready registry manifest for ``columns``."""
    resolutions = feature_groups_for_columns(
        columns,
        requested_groups=requested_groups,
        strict=strict,
    )
    groups = []
    unsupported = []
    proxies = []
    for resolution in resolutions.values():
        payload = {
            "name": resolution.name,
            "description": resolution.description,
            "required_source_columns": list(
                get_feature_group(resolution.name).required_source_columns
            ),
            "source_columns": list(resolution.source_columns),
            "generated_columns": list(resolution.generated_columns),
            "kind": resolution.kind,
            "requires_past_context": resolution.requires_past_context,
            "valid_for_fi2010": resolution.valid_for_fi2010,
            "limitations": list(resolution.limitations),
            "status": resolution.status,
            "reason": resolution.reason,
        }
        groups.append(payload)
        if resolution.status == "unsupported" or not resolution.valid_for_fi2010:
            unsupported.append(payload)
        if resolution.is_proxy:
            proxies.append(payload)
    return {
        "registry_version": "microstructure-feature-registry/v2",
        "groups": groups,
        "unsupported_groups": unsupported,
        "proxy_groups": proxies,
    }
