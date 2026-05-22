"""Past-only volatility and event-intensity features.

All functions in this module operate purely on history. None of them
peek at observations after the index at which they are evaluated, so
they are safe to embed in feature pipelines without leakage as long as
the caller hands them a window ending at or before ``t``.

Volatility uses ``log(price_t / price_{t-1})`` returns and the realised
volatility is computed as the root-sum-of-squares of log returns, which
is the standard estimator for realised volatility on a fixed sample.

Event intensity is computed in events-per-second over a trailing window
ending at the latest timestamp. Timezone-naive timestamps are rejected
to make synthetic vs. real-time inputs unambiguous.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from itertools import pairwise

__all__ = [
    "compute_event_intensity",
    "compute_log_returns",
    "compute_realised_volatility",
    "compute_rolling_event_intensity",
    "compute_rolling_realised_volatility",
]


def _validate_window(window: int | None, *, allow_none: bool = True) -> int | None:
    if window is None:
        if allow_none:
            return None
        raise ValueError("window must be provided")
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be an int or None")
    if window < 2:
        raise ValueError(f"window must be >= 2; got {window!r}")
    return window


def _ensure_positive_finite_prices(prices: Sequence[float]) -> list[float]:
    cleaned: list[float] = []
    for idx, value in enumerate(prices):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"prices[{idx}] must be a finite number; got {type(value).__name__}"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"prices[{idx}] must be finite; got {value!r}")
        if numeric <= 0.0:
            raise ValueError(
                f"prices[{idx}] must be strictly positive for log returns; "
                f"got {numeric!r}"
            )
        cleaned.append(numeric)
    return cleaned


def compute_log_returns(prices: Sequence[float]) -> list[float]:
    """Return log returns ``log(p_t / p_{t-1})`` for the input price series.

    The output has length ``len(prices) - 1``. All prices must be finite
    and strictly positive. Returns an empty list when fewer than two
    prices are supplied.
    """
    cleaned = _ensure_positive_finite_prices(prices)
    if len(cleaned) < 2:
        return []
    out: list[float] = []
    for previous, current in pairwise(cleaned):
        out.append(math.log(current / previous))
    return out


def compute_realised_volatility(
    prices: Sequence[float],
    window: int | None = None,
) -> float:
    """Return realised volatility as ``sqrt(sum(log_return^2))``.

    When ``window`` is supplied, only the last ``window`` prices are
    used. Requires at least two prices in the chosen sample.
    """
    cleaned = _ensure_positive_finite_prices(prices)
    validated_window = _validate_window(window)
    if validated_window is not None:
        cleaned = cleaned[-validated_window:]
    if len(cleaned) < 2:
        raise ValueError(
            "realised volatility requires at least 2 prices; "
            f"got {len(cleaned)}"
        )
    returns = compute_log_returns(cleaned)
    return math.sqrt(sum(r * r for r in returns))


def compute_rolling_realised_volatility(
    prices: Sequence[float],
    window: int,
    *,
    fill_value: float = math.nan,
) -> list[float]:
    """Return a per-index past-only realised-volatility series.

    At index ``t`` the function uses the most recent ``window`` prices
    in ``prices[: t + 1]`` (i.e. only data available up to and including
    ``t``). Positions with fewer than two usable prices return
    ``fill_value`` (default NaN).
    """
    validated_window = _validate_window(window, allow_none=False)
    # Validate everything up-front so partial computation cannot mask bad data.
    cleaned = _ensure_positive_finite_prices(prices)
    assert validated_window is not None
    out: list[float] = []
    for t in range(len(cleaned)):
        start = max(0, t + 1 - validated_window)
        sample = cleaned[start : t + 1]
        if len(sample) < 2:
            out.append(fill_value)
            continue
        returns = compute_log_returns(sample)
        out.append(math.sqrt(sum(r * r for r in returns)))
    return out


def _validate_aware_timestamps(timestamps: Sequence[datetime]) -> list[datetime]:
    cleaned: list[datetime] = []
    previous: datetime | None = None
    for idx, ts in enumerate(timestamps):
        if not isinstance(ts, datetime):
            raise TypeError(
                f"timestamps[{idx}] must be a datetime; got {type(ts).__name__}"
            )
        if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
            raise ValueError(
                f"timestamps[{idx}] must be timezone-aware; got naive {ts!r}"
            )
        if previous is not None and ts < previous:
            raise ValueError(
                "timestamps must be non-decreasing; "
                f"index {idx} ({ts!r}) precedes index {idx - 1} ({previous!r})"
            )
        cleaned.append(ts)
        previous = ts
    return cleaned


def _validate_positive_window_seconds(window_seconds: float) -> float:
    if isinstance(window_seconds, bool) or not isinstance(window_seconds, (int, float)):
        raise TypeError("window_seconds must be a number")
    value = float(window_seconds)
    if not math.isfinite(value):
        raise ValueError(f"window_seconds must be finite; got {window_seconds!r}")
    if value <= 0.0:
        raise ValueError(
            f"window_seconds must be strictly positive; got {window_seconds!r}"
        )
    return value


def compute_event_intensity(
    timestamps: Sequence[datetime],
    window_seconds: float,
) -> float:
    """Return events-per-second in a trailing window ending at the last ts.

    The window is inclusive of its endpoints. Returns ``0.0`` for an
    empty timestamp sequence.
    """
    cleaned = _validate_aware_timestamps(timestamps)
    window = _validate_positive_window_seconds(window_seconds)
    if not cleaned:
        return 0.0
    end_ts = cleaned[-1]
    count = 0
    for ts in cleaned:
        delta = (end_ts - ts).total_seconds()
        if 0.0 <= delta <= window:
            count += 1
    return count / window


def compute_rolling_event_intensity(
    timestamps: Sequence[datetime],
    window_seconds: float,
) -> list[float]:
    """Return per-index trailing event intensity (events/sec).

    For each index ``t`` the function counts how many timestamps lie in
    ``[timestamps[t] - window_seconds, timestamps[t]]`` and divides by
    the window size. Empty input yields an empty list.
    """
    cleaned = _validate_aware_timestamps(timestamps)
    window = _validate_positive_window_seconds(window_seconds)
    out: list[float] = []
    # Two-pointer sweep because timestamps are non-decreasing.
    left = 0
    for right, current in enumerate(cleaned):
        threshold_seconds = (current - cleaned[left]).total_seconds()
        while left <= right and threshold_seconds > window:
            left += 1
            if left <= right:
                threshold_seconds = (current - cleaned[left]).total_seconds()
        count = right - left + 1
        out.append(count / window)
    return out
