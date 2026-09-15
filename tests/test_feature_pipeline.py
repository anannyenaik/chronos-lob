"""Tests for chronoslob.features.pipeline."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from chronoslob.data.fi2010 import FI2010Config, load_fi2010
from chronoslob.data.schemas import FeatureRow, OrderBookLevel, OrderBookSnapshot
from chronoslob.features.pipeline import (
    FeaturePipelineConfig,
    build_feature_frame_from_fi2010,
    build_feature_frame_from_snapshots,
    build_features_from_snapshot,
    validate_feature_frame,
)

T0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "fi2010" / "tiny_fi2010_like.csv"
)
LABEL_COLUMNS = ["label_10", "label_50", "label_100"]


def _snapshot(
    t: datetime,
    bid_price: float,
    bid_qty: float,
    ask_price: float,
    ask_qty: float,
    *,
    second_bid: tuple[float, float] | None = None,
    second_ask: tuple[float, float] | None = None,
    synthetic_time: bool = False,
) -> OrderBookSnapshot:
    bids = [OrderBookLevel(price=bid_price, quantity=bid_qty)]
    if second_bid is not None:
        bids.append(OrderBookLevel(price=second_bid[0], quantity=second_bid[1]))
    asks = [OrderBookLevel(price=ask_price, quantity=ask_qty)]
    if second_ask is not None:
        asks.append(OrderBookLevel(price=second_ask[0], quantity=second_ask[1]))
    return OrderBookSnapshot(
        timestamp=t,
        symbol="TEST",
        bids=bids,
        asks=asks,
        metadata={"synthetic_time": synthetic_time, "source": "test"},
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_pipeline_config_rejects_invalid_depths() -> None:
    with pytest.raises(ValueError):
        FeaturePipelineConfig(depths=())
    with pytest.raises(ValueError):
        FeaturePipelineConfig(depths=(0,))


def test_pipeline_config_rejects_invalid_window() -> None:
    with pytest.raises(ValueError):
        FeaturePipelineConfig(volatility_window=1)
    with pytest.raises(ValueError):
        FeaturePipelineConfig(event_intensity_window_seconds=0.0)


# ---------------------------------------------------------------------------
# build_features_from_snapshot
# ---------------------------------------------------------------------------


def test_build_features_from_snapshot_returns_feature_row() -> None:
    snap = _snapshot(T0, 100.0, 2.0, 101.0, 3.0)
    row = build_features_from_snapshot(snap)
    assert isinstance(row, FeatureRow)
    assert row.symbol == "TEST"
    assert row.timestamp == T0
    assert "mid_price" in row.features
    assert "spread" in row.features
    assert "queue_imbalance" in row.features
    assert "bid_depth_1" in row.features


def test_build_features_from_snapshot_with_previous_adds_ofi() -> None:
    previous = _snapshot(T0, 100.0, 2.0, 101.0, 2.0)
    current = _snapshot(
        T0 + timedelta(seconds=1),
        bid_price=100.5,
        bid_qty=4.0,
        ask_price=101.0,
        ask_qty=2.0,
    )
    row = build_features_from_snapshot(current, previous_snapshot=previous)
    assert "order_flow_imbalance" in row.features
    assert row.features["order_flow_imbalance"] == pytest.approx(4.0)


def test_build_features_from_snapshot_marks_synthetic_in_metadata() -> None:
    snap = _snapshot(T0, 100.0, 2.0, 101.0, 3.0, synthetic_time=True)
    row = build_features_from_snapshot(snap)
    assert row.metadata.get("synthetic_time") is True


# ---------------------------------------------------------------------------
# build_feature_frame_from_snapshots
# ---------------------------------------------------------------------------


def test_build_feature_frame_returns_one_row_per_snapshot() -> None:
    snaps = [
        _snapshot(T0 + timedelta(seconds=i), 100.0 + i * 0.1, 2.0, 101.0 + i * 0.1, 2.0)
        for i in range(5)
    ]
    frame = build_feature_frame_from_snapshots(snaps)
    assert len(frame) == 5
    assert "timestamp" in frame.columns
    assert "symbol" in frame.columns
    assert "mid_price" in frame.columns


def test_build_feature_frame_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_feature_frame_from_snapshots([])


def test_build_feature_frame_rejects_unordered_timestamps() -> None:
    snaps = [
        _snapshot(T0 + timedelta(seconds=2), 100.0, 2.0, 101.0, 2.0),
        _snapshot(T0, 100.0, 2.0, 101.0, 2.0),
    ]
    with pytest.raises(ValueError, match="non-decreasing"):
        build_feature_frame_from_snapshots(snaps)


def test_build_feature_frame_no_label_columns() -> None:
    snaps = [
        _snapshot(T0 + timedelta(seconds=i), 100.0, 2.0, 101.0, 2.0)
        for i in range(3)
    ]
    frame = build_feature_frame_from_snapshots(snaps)
    for column in frame.columns:
        assert not column.lower().startswith("label")
        assert not column.lower().startswith("y_")


def test_build_feature_frame_rolling_volatility_does_not_use_future() -> None:
    snaps = [
        _snapshot(T0 + timedelta(seconds=0), 100.0, 2.0, 101.0, 2.0),
        _snapshot(T0 + timedelta(seconds=1), 100.0, 2.0, 200.0, 2.0),
        _snapshot(T0 + timedelta(seconds=2), 100.0, 2.0, 101.0, 2.0),
    ]
    frame = build_feature_frame_from_snapshots(
        snaps, FeaturePipelineConfig(volatility_window=2)
    )
    # At t=0 there is no past so realised_volatility should be NaN.
    assert math.isnan(frame.iloc[0]["realised_volatility"])
    # The value at t=2 must be computed only from positions 1..2 (window=2).
    mid_at_1 = frame.iloc[1]["mid_price"]
    mid_at_2 = frame.iloc[2]["mid_price"]
    expected_last = math.sqrt(math.log(mid_at_2 / mid_at_1) ** 2)
    assert frame.iloc[2]["realised_volatility"] == pytest.approx(expected_last)


def test_build_feature_frame_skips_event_intensity_for_synthetic_time() -> None:
    snaps = [
        _snapshot(
            T0 + timedelta(seconds=i),
            100.0 + i * 0.1,
            2.0,
            101.0 + i * 0.1,
            2.0,
            synthetic_time=True,
        )
        for i in range(3)
    ]
    frame = build_feature_frame_from_snapshots(snaps)
    assert frame.attrs["synthetic_time"] is True
    assert frame.attrs["skipped_time_features"] is True
    assert "event_intensity" not in frame.columns


def test_build_feature_frame_includes_event_intensity_when_allowed() -> None:
    snaps = [
        _snapshot(
            T0 + timedelta(seconds=i),
            100.0 + i * 0.1,
            2.0,
            101.0 + i * 0.1,
            2.0,
            synthetic_time=True,
        )
        for i in range(3)
    ]
    frame = build_feature_frame_from_snapshots(
        snaps,
        FeaturePipelineConfig(allow_synthetic_timestamps_for_time_features=True),
    )
    assert "event_intensity" in frame.columns


# ---------------------------------------------------------------------------
# build_feature_frame_from_fi2010
# ---------------------------------------------------------------------------


def _fi2010_dataset():
    config = FI2010Config(
        path=FIXTURE_PATH,
        timestamp_column="timestamp",
        split_column="split",
        label_columns=list(LABEL_COLUMNS),
        price_level_count=2,
    )
    return load_fi2010(config)


def test_build_feature_frame_from_fi2010_runs_on_fixture() -> None:
    dataset = _fi2010_dataset()
    frame = build_feature_frame_from_fi2010(dataset)
    assert len(frame) == dataset.n_rows
    assert "mid_price" in frame.columns
    assert "spread" in frame.columns


def test_build_feature_frame_from_fi2010_excludes_label_columns() -> None:
    dataset = _fi2010_dataset()
    frame = build_feature_frame_from_fi2010(dataset)
    for column in LABEL_COLUMNS:
        assert column not in frame.columns


def test_build_feature_frame_from_fi2010_preserves_split_as_non_feature() -> None:
    dataset = _fi2010_dataset()
    frame = build_feature_frame_from_fi2010(dataset)
    assert "split" in frame.columns


def test_build_feature_frame_from_fi2010_raises_when_no_snapshots() -> None:
    import pandas as pd

    from chronoslob.data.fi2010 import FI2010Dataset

    frame = pd.DataFrame({"foo": [1.0, 2.0]})
    config = FI2010Config(
        path=FIXTURE_PATH,
        feature_columns=["foo"],
        allow_missing_lob_columns=True,
    )
    dataset = FI2010Dataset(
        frame=frame,
        config=config,
        feature_columns=["foo"],
        label_columns=[],
    )
    with pytest.raises(ValueError, match="no snapshots"):
        build_feature_frame_from_fi2010(dataset)


# ---------------------------------------------------------------------------
# validate_feature_frame
# ---------------------------------------------------------------------------


def test_validate_feature_frame_passes_for_clean_frame() -> None:
    snaps = [
        _snapshot(T0 + timedelta(seconds=i), 100.0, 2.0, 101.0, 2.0)
        for i in range(3)
    ]
    frame = build_feature_frame_from_snapshots(snaps)
    result = validate_feature_frame(frame)
    assert result.ok


def test_validate_feature_frame_catches_infinity() -> None:
    snaps = [
        _snapshot(T0 + timedelta(seconds=i), 100.0, 2.0, 101.0, 2.0)
        for i in range(3)
    ]
    frame = build_feature_frame_from_snapshots(snaps)
    frame.loc[0, "mid_price"] = np.inf
    result = validate_feature_frame(frame)
    assert not result.ok
    assert any(issue.code == "inf_in_feature" for issue in result.issues)


def test_validate_feature_frame_catches_obvious_label_columns() -> None:
    snaps = [
        _snapshot(T0 + timedelta(seconds=i), 100.0, 2.0, 101.0, 2.0)
        for i in range(3)
    ]
    frame = build_feature_frame_from_snapshots(snaps)
    frame["label_10"] = [0, 1, 2]
    result = validate_feature_frame(frame)
    assert not result.ok
    assert any(
        issue.code == "label_like_column_in_features" for issue in result.issues
    )


def test_validate_feature_frame_disallow_nan_by_request() -> None:
    snaps = [
        _snapshot(T0 + timedelta(seconds=i), 100.0, 2.0, 101.0, 2.0)
        for i in range(3)
    ]
    frame = build_feature_frame_from_snapshots(snaps)
    # Default rolling volatility carries NaN for the first row.
    assert frame["realised_volatility"].isna().any()
    result = validate_feature_frame(frame, allow_nan=False)
    assert not result.ok
    assert any(issue.code == "nan_in_feature" for issue in result.issues)
