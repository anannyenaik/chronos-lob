"""Feature pipeline that assembles snapshot-level feature rows and frames.

The pipeline is intentionally thin: it composes the small feature
functions exposed elsewhere in :mod:`chronoslob.features` into either a
single :class:`~chronoslob.data.schemas.FeatureRow` or a pandas
DataFrame of timestamped feature columns.

Design constraints:

* Every feature is past-only. The pipeline never reads any element of
  the input sequence after the index it is computing.
* Synthetic timestamps (those produced by the FI-2010 loader when the
  source file has no real timestamp column) cannot be used for time
  windows unless the caller explicitly opts in. We default to *skip*
  time-window features in that case and record the skip in the frame
  metadata.
* Labels never enter feature columns. The pipeline refuses to forward
  configured label columns into feature output.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

from chronoslob.data.schemas import DataQualityIssue, FeatureRow, OrderBookSnapshot
from chronoslob.data.validation import DataValidationResult
from chronoslob.features.imbalance import (
    compute_depth,
    compute_level_imbalances,
)
from chronoslob.features.microprice import compute_snapshot_price_features
from chronoslob.features.order_flow import (
    compute_order_flow_imbalance_from_snapshots,
)
from chronoslob.features.volatility import (
    compute_rolling_event_intensity,
    compute_rolling_realised_volatility,
)

if TYPE_CHECKING:
    from chronoslob.data.fi2010 import FI2010Dataset

__all__ = [
    "FeaturePipelineConfig",
    "build_feature_frame_from_fi2010",
    "build_feature_frame_from_snapshots",
    "build_features_from_snapshot",
    "validate_feature_frame",
]


class FeaturePipelineConfig(BaseModel):
    """Configuration for the feature pipeline.

    Fields:

    * ``depths`` -- depths at which to compute depth-imbalance features.
    * ``include_price_features`` / ``include_imbalance_features`` /
      ``include_order_flow`` / ``include_volatility`` /
      ``include_regime_features`` -- toggle whole feature families.
    * ``volatility_window`` -- look-back length (in snapshots) for the
      rolling realised volatility.
    * ``event_intensity_window_seconds`` -- trailing window for the
      rolling event-intensity feature.
    * ``allow_synthetic_timestamps_for_time_features`` -- enable time
      features even when snapshot metadata flags timestamps as
      synthetic. Defaults to ``False`` so callers cannot accidentally
      treat synthetic time as real.
    * ``allow_partial_features`` -- if ``False``, raise when an
      individual feature cannot be computed; if ``True`` skip it and
      record the skip in metadata.
    """

    model_config = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)

    depths: tuple[int, ...] = (1, 5, 10)
    include_price_features: bool = True
    include_imbalance_features: bool = True
    include_order_flow: bool = True
    include_volatility: bool = True
    include_regime_features: bool = True
    volatility_window: int = 20
    event_intensity_window_seconds: float = 60.0
    allow_synthetic_timestamps_for_time_features: bool = False
    allow_partial_features: bool = True

    @field_validator("depths")
    @classmethod
    def _validate_depths(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("depths must contain at least one positive integer")
        for entry in value:
            if isinstance(entry, bool) or not isinstance(entry, int):
                raise TypeError(f"depths must be ints; got {type(entry).__name__}")
            if entry <= 0:
                raise ValueError(f"depths must be strictly positive; got {entry!r}")
        return value

    @field_validator("volatility_window")
    @classmethod
    def _validate_volatility_window(cls, value: int) -> int:
        if value < 2:
            raise ValueError(f"volatility_window must be >= 2; got {value!r}")
        return value

    @field_validator("event_intensity_window_seconds")
    @classmethod
    def _validate_event_window(cls, value: float) -> float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("event_intensity_window_seconds must be a number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("event_intensity_window_seconds must be finite")
        if numeric <= 0.0:
            raise ValueError(
                f"event_intensity_window_seconds must be > 0; got {value!r}"
            )
        return numeric


# ---------------------------------------------------------------------------
# Single-snapshot feature builder
# ---------------------------------------------------------------------------


def _snapshot_is_synthetic(snapshot: OrderBookSnapshot) -> bool:
    value = snapshot.metadata.get("synthetic_time")
    if isinstance(value, bool):
        return value
    return False


def _build_snapshot_features(
    snapshot: OrderBookSnapshot,
    previous_snapshot: OrderBookSnapshot | None,
    config: FeaturePipelineConfig,
) -> tuple[dict[str, float], list[str]]:
    features: dict[str, float] = {}
    skipped: list[str] = []
    if config.include_price_features:
        try:
            features.update(compute_snapshot_price_features(snapshot))
        except ValueError:
            if not config.allow_partial_features:
                raise
            skipped.append("price_features")
    if config.include_imbalance_features:
        try:
            features.update(
                compute_level_imbalances(snapshot, depths=config.depths)
            )
        except ValueError:
            if not config.allow_partial_features:
                raise
            skipped.append("imbalance_features")
        # Add total depth across all levels as a convenience feature.
        try:
            features["total_depth"] = compute_depth(
                snapshot.bids
            ) + compute_depth(snapshot.asks)
        except ValueError:
            if not config.allow_partial_features:
                raise
            skipped.append("total_depth")
    if config.include_order_flow and previous_snapshot is not None:
        try:
            features["order_flow_imbalance"] = (
                compute_order_flow_imbalance_from_snapshots(
                    previous_snapshot, snapshot
                )
            )
        except ValueError:
            if not config.allow_partial_features:
                raise
            skipped.append("order_flow_imbalance")
    return features, skipped


def build_features_from_snapshot(
    snapshot: OrderBookSnapshot,
    previous_snapshot: OrderBookSnapshot | None = None,
    config: FeaturePipelineConfig | None = None,
) -> FeatureRow:
    """Build a :class:`FeatureRow` for a single snapshot.

    The optional ``previous_snapshot`` enables the simple top-of-book
    order-flow imbalance contribution when ``config.include_order_flow``
    is set. When omitted, OFI is simply not included.

    The returned ``metadata`` carries:

    * ``source = "snapshot"``;
    * ``synthetic_time`` mirrored from the snapshot if present;
    * ``partial_features`` = ``True`` when at least one feature family
      was skipped;
    * ``skipped`` listing the skipped feature families (comma-separated).
    """
    if not isinstance(snapshot, OrderBookSnapshot):
        raise TypeError("snapshot must be an OrderBookSnapshot")
    if previous_snapshot is not None and not isinstance(
        previous_snapshot, OrderBookSnapshot
    ):
        raise TypeError("previous_snapshot must be an OrderBookSnapshot or None")
    if config is None:
        config = FeaturePipelineConfig()
    features, skipped = _build_snapshot_features(
        snapshot, previous_snapshot, config
    )
    metadata: dict[str, str | int | float | bool] = {
        "source": "snapshot",
        "partial_features": bool(skipped),
    }
    if "synthetic_time" in snapshot.metadata:
        metadata["synthetic_time"] = bool(snapshot.metadata["synthetic_time"])
    if skipped:
        metadata["skipped"] = ",".join(skipped)
    return FeatureRow(
        timestamp=snapshot.timestamp,
        symbol=snapshot.symbol,
        features=features,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Sequence-level builders
# ---------------------------------------------------------------------------


def _check_timestamps_non_decreasing(snapshots: Sequence[OrderBookSnapshot]) -> None:
    for i in range(1, len(snapshots)):
        if snapshots[i].timestamp < snapshots[i - 1].timestamp:
            raise ValueError(
                "snapshots must be ordered by non-decreasing timestamp; "
                f"index {i} ({snapshots[i].timestamp!r}) precedes "
                f"index {i - 1} ({snapshots[i - 1].timestamp!r})"
            )


def _build_per_snapshot_rows(
    snapshots: Sequence[OrderBookSnapshot],
    config: FeaturePipelineConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    previous: OrderBookSnapshot | None = None
    for snapshot in snapshots:
        feature_row = build_features_from_snapshot(
            snapshot, previous_snapshot=previous, config=config
        )
        record: dict[str, object] = {
            "timestamp": feature_row.timestamp,
            "symbol": feature_row.symbol,
        }
        record.update(feature_row.features)
        rows.append(record)
        previous = snapshot
    return rows


def build_feature_frame_from_snapshots(
    snapshots: Sequence[OrderBookSnapshot],
    config: FeaturePipelineConfig | None = None,
) -> pd.DataFrame:
    """Return a pandas DataFrame of past-only features for ``snapshots``.

    The returned frame includes one row per snapshot plus rolling
    realised volatility computed from ``mid_price``. When timestamps are
    real (not synthetic), the rolling event-intensity feature is also
    included. Synthetic timestamps are detected via the
    ``synthetic_time`` field in each snapshot's metadata; when any
    snapshot is flagged synthetic and the config does not opt in, time
    features are skipped and the skip is recorded in ``frame.attrs``.

    Validation:

    * raises ``ValueError`` if ``snapshots`` is empty;
    * raises ``ValueError`` if timestamps are not non-decreasing.
    """
    if config is None:
        config = FeaturePipelineConfig()
    seq = list(snapshots)
    if not seq:
        raise ValueError("snapshots must be non-empty")
    _check_timestamps_non_decreasing(seq)

    records = _build_per_snapshot_rows(seq, config)
    frame = pd.DataFrame.from_records(records)

    skipped_time_features = False
    if config.include_volatility and "mid_price" in frame.columns:
        mid_prices = frame["mid_price"].to_list()
        # The rolling realised volatility helper validates positivity. We
        # therefore feed it only when every mid_price is positive/finite.
        positive = all(
            isinstance(p, (int, float))
            and not isinstance(p, bool)
            and math.isfinite(p)
            and p > 0.0
            for p in mid_prices
        )
        if positive:
            frame["realised_volatility"] = compute_rolling_realised_volatility(
                mid_prices, window=config.volatility_window
            )

    any_synthetic = any(_snapshot_is_synthetic(s) for s in seq)
    if config.include_volatility:
        # Event intensity is a *time* feature; we gate it on synthetic time.
        if any_synthetic and not config.allow_synthetic_timestamps_for_time_features:
            skipped_time_features = True
        else:
            timestamps = [s.timestamp for s in seq]
            frame["event_intensity"] = compute_rolling_event_intensity(
                timestamps,
                window_seconds=config.event_intensity_window_seconds,
            )

    frame.attrs["synthetic_time"] = bool(any_synthetic)
    frame.attrs["skipped_time_features"] = bool(skipped_time_features)
    return frame


def build_feature_frame_from_fi2010(
    dataset: FI2010Dataset,
    config: FeaturePipelineConfig | None = None,
) -> pd.DataFrame:
    """Build a feature frame from an :class:`FI2010Dataset`.

    The dataset's snapshot rows are produced via
    :meth:`FI2010Dataset.to_snapshot_rows` and then handed to
    :func:`build_feature_frame_from_snapshots`. Label columns configured
    on the dataset are *never* included in the returned frame.

    Raises ``ValueError`` if the FI-2010 file does not carry the LOB
    columns required to materialise snapshots.
    """
    if config is None:
        config = FeaturePipelineConfig()
    snapshots = dataset.to_snapshot_rows()
    if not snapshots:
        raise ValueError(
            "FI-2010 dataset has no snapshots; the file is missing the "
            "configured bid/ask price and quantity columns or is empty"
        )
    frame = build_feature_frame_from_snapshots(snapshots, config=config)

    # Defence-in-depth: even though we never write label columns into the
    # feature frame, we drop them here in case a future refactor changes
    # this. Label columns are taken from the dataset's resolved list and
    # also matched against the conventional prefixes.
    for column in list(frame.columns):
        if column in dataset.label_columns:
            frame = frame.drop(columns=[column])
            continue
        if column.startswith("label") or column.startswith("y_"):
            frame = frame.drop(columns=[column])

    split_series = dataset.split_values()
    if split_series is not None:
        # Preserve as a non-feature column so downstream code can route
        # rows into train/test partitions without leaking it as a feature.
        # The column lives outside the feature columns: callers should
        # exclude ``split`` when serialising features.
        limit = min(len(split_series), len(frame))
        frame["split"] = split_series.iloc[:limit].reset_index(drop=True)
    return frame


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


_LABEL_LIKE_PREFIXES = ("label", "y_")


def validate_feature_frame(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    allow_nan: bool = True,
) -> DataValidationResult:
    """Validate a feature DataFrame.

    Checks:

    * non-empty frame;
    * ``timestamp`` and ``symbol`` columns are present;
    * feature columns are numeric;
    * no infinite values in feature columns;
    * NaN values only when ``allow_nan=True``;
    * no obvious label columns (names starting with ``label`` or
      ``y_``) are reported among the feature columns when
      ``feature_columns`` is left to the default.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    issues: list[DataQualityIssue] = []

    def _error(
        code: str,
        message: str,
        metadata: dict[str, str | int | float | bool] | None = None,
    ) -> None:
        issues.append(
            DataQualityIssue(
                severity="error",
                code=code,
                message=message,
                metadata=metadata or {},
            )
        )

    n_rows, n_columns = frame.shape
    if n_rows == 0:
        _error(code="empty_frame", message="feature frame contains no rows")
    for required in ("timestamp", "symbol"):
        if required not in frame.columns:
            _error(
                code="missing_required_column",
                message=f"feature frame is missing required column {required!r}",
                metadata={"column": required},
            )

    if feature_columns is None:
        candidates = [
            c
            for c in frame.columns
            if c not in {"timestamp", "symbol", "split"}
        ]
        for column in candidates:
            lowered = column.lower()
            if any(lowered.startswith(prefix) for prefix in _LABEL_LIKE_PREFIXES):
                _error(
                    code="label_like_column_in_features",
                    message=(
                        f"column {column!r} looks like a label "
                        "(prefix 'label' or 'y_') and must not be used as a feature"
                    ),
                    metadata={"column": column},
                )
        feature_cols_to_check = candidates
    else:
        feature_cols_to_check = list(feature_columns)

    for column in feature_cols_to_check:
        if column not in frame.columns:
            _error(
                code="missing_feature_column",
                message=f"feature column {column!r} is missing",
                metadata={"column": column},
            )
            continue
        series = frame[column]
        if not pd.api.types.is_numeric_dtype(series):
            _error(
                code="non_numeric_feature",
                message=f"feature column {column!r} is not numeric",
                metadata={"column": column},
            )
            continue
        try:
            has_inf = bool((series.abs() == math.inf).any())
        except TypeError:
            has_inf = False
        if has_inf:
            _error(
                code="inf_in_feature",
                message=f"feature column {column!r} contains infinite values",
                metadata={"column": column},
            )
        if not allow_nan and series.isna().any():
            _error(
                code="nan_in_feature",
                message=f"feature column {column!r} contains NaN values",
                metadata={"column": column},
            )

    return DataValidationResult(
        ok=not any(issue.severity == "error" for issue in issues),
        issues=issues,
        n_rows=n_rows,
        n_columns=n_columns,
    )
