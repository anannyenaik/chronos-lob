"""Future realised-volatility labels.

These labels summarise market-state outcomes after timestamp ``t``. The
realised-volatility convention matches ``chronoslob.features.volatility``:
``sqrt(sum(log_return ** 2))`` over the requested price interval.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise

__all__ = [
    "classify_volatility_labels",
    "compute_future_realised_volatility",
    "compute_future_volatility_series",
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


def _coerce_positive_finite_prices(mid_prices: Sequence[float]) -> list[float]:
    cleaned: list[float] = []
    for position, value in enumerate(mid_prices):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                "mid_prices must contain finite numeric prices; "
                f"got {type(value).__name__} at index {position}"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"mid_prices[{position}] must be finite; got {value!r}")
        if numeric <= 0.0:
            raise ValueError(
                f"mid_prices[{position}] must be strictly positive; got {numeric!r}"
            )
        cleaned.append(numeric)
    return cleaned


def _coerce_finite_non_negative(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite; got {value!r}")
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative; got {numeric!r}")
    return numeric


def compute_future_realised_volatility(
    mid_prices: Sequence[float],
    index: int,
    horizon: int,
) -> float:
    """Return future realised volatility over ``index:index + horizon``.

    The interval uses prices from ``index`` through ``index + horizon``
    inclusive, producing ``horizon`` future log returns.
    """
    cleaned = _coerce_positive_finite_prices(mid_prices)
    idx = _validate_index(index)
    h = _validate_horizon(horizon)
    future_index = idx + h
    if idx >= len(cleaned):
        raise IndexError(f"index {idx} is outside mid_prices of length {len(cleaned)}")
    if future_index >= len(cleaned):
        raise IndexError(
            "insufficient future data for requested horizon: "
            f"index {idx} + horizon {h} >= length {len(cleaned)}"
        )
    prices = cleaned[idx : future_index + 1]
    squared_sum = 0.0
    for previous, current in pairwise(prices):
        log_return = math.log(current / previous)
        squared_sum += log_return * log_return
    return math.sqrt(squared_sum)


def compute_future_volatility_series(
    mid_prices: Sequence[float],
    horizon: int,
    *,
    missing: str = "drop",
) -> list[float | None]:
    """Return future realised-volatility labels for each eligible row."""
    cleaned = _coerce_positive_finite_prices(mid_prices)
    h = _validate_horizon(horizon)
    policy = _validate_missing(missing)
    output: list[float | None] = []
    for idx in range(len(cleaned)):
        if idx + h >= len(cleaned):
            if policy == "none":
                output.append(None)
            continue
        output.append(compute_future_realised_volatility(cleaned, idx, h))
    return output


def classify_volatility_labels(
    future_volatility: Sequence[float],
    low_threshold: float,
    high_threshold: float,
) -> list[str]:
    """Classify future volatility into low, medium or high regimes."""
    low = _coerce_finite_non_negative(low_threshold, name="low_threshold")
    high = _coerce_finite_non_negative(high_threshold, name="high_threshold")
    if low > high:
        raise ValueError("low_threshold must be <= high_threshold")

    labels: list[str] = []
    for position, value in enumerate(future_volatility):
        vol = _coerce_finite_non_negative(
            value, name=f"future_volatility[{position}]"
        )
        if vol <= low:
            labels.append("low_volatility")
        elif vol >= high:
            labels.append("high_volatility")
        else:
            labels.append("medium_volatility")
    return labels
