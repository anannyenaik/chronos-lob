"""Future spread-widening labels."""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "compute_future_spread_change",
    "compute_spread_widening_label",
    "compute_spread_widening_labels",
]


def _validate_horizon(horizon: int) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise TypeError("horizon must be an integer")
    if horizon <= 0:
        raise ValueError(f"horizon must be strictly positive; got {horizon!r}")
    return horizon


def _validate_index(index: int) -> int:
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("index must be an integer")
    if index < 0:
        raise ValueError(f"index must be non-negative; got {index!r}")
    return index


def _validate_missing(missing: str) -> str:
    if missing not in {"drop", "none"}:
        raise ValueError("missing must be one of {'drop', 'none'}")
    return missing


def _coerce_spreads(spreads: Sequence[float]) -> list[float]:
    cleaned: list[float] = []
    for position, value in enumerate(spreads):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                "spreads must contain finite numeric values; "
                f"got {type(value).__name__} at index {position}"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"spreads[{position}] must be finite; got {value!r}")
        if numeric < 0.0:
            raise ValueError(
                f"spreads[{position}] must be non-negative; got {numeric!r}"
            )
        cleaned.append(numeric)
    return cleaned


def _validate_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a finite number")
    numeric = float(threshold)
    if not math.isfinite(numeric):
        raise ValueError(f"threshold must be finite; got {threshold!r}")
    if numeric < 0.0:
        raise ValueError(f"threshold must be non-negative; got {numeric!r}")
    return numeric


def compute_future_spread_change(
    spreads: Sequence[float],
    index: int,
    horizon: int,
) -> float:
    """Return ``spread[index + horizon] - spread[index]``."""
    cleaned = _coerce_spreads(spreads)
    idx = _validate_index(index)
    h = _validate_horizon(horizon)
    future_index = idx + h
    if idx >= len(cleaned):
        raise IndexError(f"index {idx} is outside spreads of length {len(cleaned)}")
    if future_index >= len(cleaned):
        raise IndexError(
            "insufficient future data for requested horizon: "
            f"index {idx} + horizon {h} >= length {len(cleaned)}"
        )
    return cleaned[future_index] - cleaned[idx]


def compute_spread_widening_label(
    spreads: Sequence[float],
    index: int,
    horizon: int,
    threshold: float = 0.0,
) -> bool:
    """Return ``True`` when the future spread change exceeds ``threshold``."""
    cutoff = _validate_threshold(threshold)
    return compute_future_spread_change(spreads, index, horizon) > cutoff


def compute_spread_widening_labels(
    spreads: Sequence[float],
    horizon: int,
    threshold: float = 0.0,
    *,
    missing: str = "drop",
) -> list[bool | None]:
    """Return spread-widening labels under an explicit missing policy."""
    cleaned = _coerce_spreads(spreads)
    h = _validate_horizon(horizon)
    cutoff = _validate_threshold(threshold)
    policy = _validate_missing(missing)
    output: list[bool | None] = []
    for idx in range(len(cleaned)):
        if idx + h >= len(cleaned):
            if policy == "none":
                output.append(None)
            continue
        output.append(compute_future_spread_change(cleaned, idx, h) > cutoff)
    return output
