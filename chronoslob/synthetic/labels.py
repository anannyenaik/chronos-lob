"""Future-horizon labels for the synthetic event-level pipeline.

Labels summarise a window strictly *after* each feature timestamp. A label row
decided at snapshot ``t`` depends only on snapshots in ``(t, t + horizon]``, so
it never leaks into the contemporaneous features at ``t``. The companion
:func:`validate_no_lookahead_frames` check enforces this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "LABEL_COLUMNS",
    "LeakageResult",
    "build_label_frame",
    "validate_no_lookahead_frames",
]

# Classification / regression label columns produced for each retained row.
LABEL_COLUMNS: tuple[str, ...] = (
    "future_mid_direction",
    "future_return_bucket",
    "volatility_regime",
    "spread_widening",
    "adverse_selection_proxy",
    "regime_label",
    "next_regime_id",
)

_DIRECTION_DOWN = 0
_DIRECTION_FLAT = 1
_DIRECTION_UP = 2


class LeakageResult:
    """Outcome of a no-lookahead validation over feature/label frames."""

    def __init__(self, *, ok: bool, checked_rows: int, violations: list[str]) -> None:
        self.ok = ok
        self.checked_rows = checked_rows
        self.violations = violations

    def summary(self) -> dict[str, object]:
        """Return a JSON-serialisable summary."""
        return {
            "ok": self.ok,
            "checked_rows": self.checked_rows,
            "violation_count": len(self.violations),
            "violations": list(self.violations[:10]),
        }


def build_label_frame(
    feature_frame: pd.DataFrame,
    *,
    horizon: int = 20,
    flat_threshold: float = 5e-5,
    adverse_threshold: float = 2e-4,
) -> pd.DataFrame:
    """Build a future-horizon label frame aligned to ``feature_frame`` rows.

    ``horizon`` is measured in snapshot steps. The final ``horizon`` rows are
    dropped because they have no complete future window.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    required = {"sequence_id", "timestamp", "mid_price", "spread", "regime_id"}
    missing = required - set(feature_frame.columns)
    if missing:
        raise ValueError(f"feature_frame missing columns: {sorted(missing)}")
    if len(feature_frame) <= horizon:
        return pd.DataFrame(columns=_label_output_columns())

    frame = feature_frame.reset_index(drop=True)
    mid = frame["mid_price"].to_numpy(dtype=float)
    spread = frame["spread"].to_numpy(dtype=float)
    regime = frame["regime_id"].to_numpy(dtype=int)
    vol = frame.get(
        "realised_volatility_proxy",
        pd.Series(np.zeros(len(frame))),
    ).to_numpy(dtype=float)

    usable = len(frame) - horizon
    future_mid = mid[horizon : horizon + usable]
    current_mid = mid[:usable]
    future_spread = spread[horizon : horizon + usable]
    current_spread = spread[:usable]
    future_regime = regime[horizon : horizon + usable]
    future_vol = vol[horizon : horizon + usable]

    with np.errstate(divide="ignore", invalid="ignore"):
        future_return = np.where(
            current_mid > 0.0, (future_mid - current_mid) / current_mid, 0.0
        )

    direction = np.full(usable, _DIRECTION_FLAT, dtype=int)
    direction[future_return > flat_threshold] = _DIRECTION_UP
    direction[future_return < -flat_threshold] = _DIRECTION_DOWN

    return_bucket = _tercile_bucket(future_return)
    vol_median = float(np.median(future_vol)) if usable else 0.0
    volatility_regime = (future_vol > vol_median).astype(int)
    spread_widening = (future_spread > current_spread).astype(int)
    adverse_selection = (np.abs(future_return) > adverse_threshold).astype(int)

    output = pd.DataFrame(
        {
            "sequence_id": frame["sequence_id"].to_numpy()[:usable].astype(int),
            "feature_timestamp": frame["timestamp"].to_numpy()[:usable],
            "future_timestamp": frame["timestamp"].to_numpy()[horizon : horizon + usable],
            "horizon": horizon,
            "future_return": future_return,
            "future_mid_direction": direction,
            "future_return_bucket": return_bucket,
            "volatility_regime": volatility_regime,
            "spread_widening": spread_widening,
            "adverse_selection_proxy": adverse_selection,
            "regime_label": regime[:usable].astype(int),
            "next_regime_id": future_regime.astype(int),
        }
    )
    return output


def _label_output_columns() -> list[str]:
    return [
        "sequence_id",
        "feature_timestamp",
        "future_timestamp",
        "horizon",
        "future_return",
        *LABEL_COLUMNS,
    ]


def _tercile_bucket(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=int)
    lower = float(np.quantile(values, 1.0 / 3.0))
    upper = float(np.quantile(values, 2.0 / 3.0))
    bucket = np.full(values.shape, 1, dtype=int)
    bucket[values <= lower] = 0
    bucket[values >= upper] = 2
    return bucket


def validate_no_lookahead_frames(
    feature_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
) -> LeakageResult:
    """Verify every label references a strictly future snapshot.

    The check confirms that, for each label row, the ``future_timestamp`` used to
    compute the label is strictly after the ``feature_timestamp`` at which the
    decision is made.
    """
    violations: list[str] = []
    if label_frame.empty:
        return LeakageResult(ok=True, checked_rows=0, violations=[])
    feature_times = label_frame["feature_timestamp"].astype(str).to_numpy()
    future_times = label_frame["future_timestamp"].astype(str).to_numpy()
    for index in range(len(label_frame)):
        if not future_times[index] > feature_times[index]:
            violations.append(
                f"row {index}: future_timestamp {future_times[index]!r} is not "
                f"after feature_timestamp {feature_times[index]!r}"
            )
    return LeakageResult(
        ok=not violations,
        checked_rows=len(label_frame),
        violations=violations,
    )
