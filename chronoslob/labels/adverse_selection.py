"""Adverse-selection proxy labels.

These labels are simplified research proxies. They first require a passive
fill proxy, then ask whether the future mid-price moved against the passive
side by more than an explicit return threshold. They do not prove real fills,
inventory costs, market impact or realised execution performance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from chronoslob.data.schemas import OrderBookSnapshot, Side
from chronoslob.labels.fill_probability import compute_passive_fill_proxy

__all__ = [
    "compute_adverse_selection_after_fill_proxy",
    "compute_adverse_selection_proxy_series",
]


def _validate_horizon(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be strictly positive; got {value!r}")
    return value


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


def _validate_side(side: Side) -> Side:
    try:
        return Side(side)
    except ValueError as exc:
        raise ValueError("side must be Side.BID or Side.ASK") from exc


def _validate_threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("adverse_return_threshold must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("adverse_return_threshold must be finite")
    if numeric < 0.0:
        raise ValueError("adverse_return_threshold must be non-negative")
    return numeric


def _mid_price(snapshot: OrderBookSnapshot) -> float:
    mid = snapshot.mid_price
    if mid is None:
        raise ValueError("top-of-book bid and ask levels are required")
    if not math.isfinite(mid) or mid <= 0.0:
        raise ValueError(f"mid-price must be finite and positive; got {mid!r}")
    return mid


def compute_adverse_selection_after_fill_proxy(
    snapshots: Sequence[OrderBookSnapshot],
    index: int,
    fill_horizon: int,
    evaluation_horizon: int,
    side: Side,
    *,
    adverse_return_threshold: float = 0.0,
) -> bool:
    """Return whether a proxy fill is followed by an adverse mid-price move.

    For passive buys, adverse selection means the evaluation mid-price is
    below the current mid-price by more than ``adverse_return_threshold``.
    For passive sells, it means the evaluation mid-price is above the current
    mid-price by more than that threshold. The threshold is applied to simple
    mid-price returns.
    """
    seq = list(snapshots)
    idx = _validate_index(index)
    fill_h = _validate_horizon(fill_horizon, name="fill_horizon")
    eval_h = _validate_horizon(evaluation_horizon, name="evaluation_horizon")
    if eval_h < fill_h:
        raise ValueError("evaluation_horizon must be >= fill_horizon")
    resolved_side = _validate_side(side)
    threshold = _validate_threshold(adverse_return_threshold)
    future_index = idx + eval_h
    if idx >= len(seq):
        raise IndexError(f"index {idx} is outside snapshots of length {len(seq)}")
    if future_index >= len(seq):
        raise IndexError(
            "insufficient future data for requested evaluation horizon: "
            f"index {idx} + horizon {eval_h} >= length {len(seq)}"
        )

    filled = compute_passive_fill_proxy(seq, idx, fill_h, resolved_side)
    if not filled:
        return False

    current_mid = _mid_price(seq[idx])
    future_mid = _mid_price(seq[future_index])
    simple_return = future_mid / current_mid - 1.0
    if resolved_side is Side.BID:
        return simple_return < -threshold
    return simple_return > threshold


def compute_adverse_selection_proxy_series(
    snapshots: Sequence[OrderBookSnapshot],
    fill_horizon: int,
    evaluation_horizon: int,
    side: Side,
    *,
    adverse_return_threshold: float = 0.0,
    missing: str = "drop",
) -> list[bool | None]:
    """Return adverse-selection proxy labels under an explicit missing policy."""
    seq = list(snapshots)
    fill_h = _validate_horizon(fill_horizon, name="fill_horizon")
    eval_h = _validate_horizon(evaluation_horizon, name="evaluation_horizon")
    if eval_h < fill_h:
        raise ValueError("evaluation_horizon must be >= fill_horizon")
    resolved_side = _validate_side(side)
    threshold = _validate_threshold(adverse_return_threshold)
    policy = _validate_missing(missing)

    output: list[bool | None] = []
    for idx in range(len(seq)):
        if idx + eval_h >= len(seq):
            if policy == "none":
                output.append(None)
            continue
        output.append(
            compute_adverse_selection_after_fill_proxy(
                seq,
                idx,
                fill_h,
                eval_h,
                resolved_side,
                adverse_return_threshold=threshold,
            )
        )
    return output
