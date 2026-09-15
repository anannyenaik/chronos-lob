"""Explicit no-look-ahead and feature/label separation tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from chronoslob.data.schemas import FeatureRow, LabelRow, OrderBookLevel, OrderBookSnapshot
from chronoslob.features.pipeline import (
    FeaturePipelineConfig,
    build_feature_frame_from_snapshots,
    build_features_from_snapshot,
)
from chronoslob.labels.leakage import (
    assert_feature_label_separation,
    assert_no_future_feature_timestamps,
    assert_temporal_label_alignment,
    validate_no_lookahead,
)
from chronoslob.labels.pipeline import (
    LabelPipelineConfig,
    build_label_frame_from_snapshots,
    build_label_rows_from_snapshots,
)

T0 = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)


def _snapshot(offset: int, bid_price: float, ask_price: float) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        timestamp=T0 + timedelta(seconds=offset),
        symbol="TEST",
        bids=[OrderBookLevel(price=bid_price, quantity=10.0)],
        asks=[OrderBookLevel(price=ask_price, quantity=10.0)],
    )


def _snapshots() -> list[OrderBookSnapshot]:
    return [
        _snapshot(0, 100.0, 101.0),
        _snapshot(1, 100.1, 101.1),
        _snapshot(2, 100.2, 101.2),
        _snapshot(3, 100.0, 101.0),
    ]


def _label_config() -> LabelPipelineConfig:
    return LabelPipelineConfig(
        horizons=(1,),
        include_fill_probability=False,
        include_adverse_selection=False,
    )


def test_feature_and_label_frames_from_same_snapshots_pass_separation() -> None:
    snapshots = _snapshots()
    feature_frame = build_feature_frame_from_snapshots(
        snapshots,
        FeaturePipelineConfig(volatility_window=2),
    )
    label_frame = build_label_frame_from_snapshots(snapshots, _label_config())
    result = assert_feature_label_separation(feature_frame, label_frame)
    assert result.ok


def test_feature_frame_containing_label_prefix_fails() -> None:
    feature_frame = build_feature_frame_from_snapshots(_snapshots())
    label_frame = build_label_frame_from_snapshots(_snapshots(), _label_config())
    feature_frame["label_10"] = [0] * len(feature_frame)
    result = assert_feature_label_separation(feature_frame, label_frame)
    assert not result.ok
    assert any(issue.code == "label_like_column_in_features" for issue in result.issues)


def test_feature_frame_containing_future_return_fails() -> None:
    feature_frame = build_feature_frame_from_snapshots(_snapshots())
    label_frame = build_label_frame_from_snapshots(_snapshots(), _label_config())
    feature_frame["future_return_10"] = [0.0] * len(feature_frame)
    result = assert_feature_label_separation(feature_frame, label_frame)
    assert not result.ok
    assert any(issue.code == "label_like_column_in_features" for issue in result.issues)


def test_label_row_with_invalid_horizon_fails_temporal_alignment() -> None:
    row = LabelRow.model_construct(
        timestamp=T0,
        symbol="TEST",
        labels={"future_return_1": 0.0},
        horizon_start=T0 - timedelta(seconds=1),
        horizon_end=T0,
        metadata={},
    )
    result = assert_temporal_label_alignment([row])
    assert not result.ok
    assert any(issue.code == "horizon_start_before_timestamp" for issue in result.issues)


def test_feature_row_with_future_origin_timestamp_fails() -> None:
    row = FeatureRow.model_construct(
        timestamp=T0,
        symbol="TEST",
        features={"mid_price": 100.0},
        horizon_origin_timestamp=T0 + timedelta(seconds=1),
        metadata={},
    )
    result = assert_no_future_feature_timestamps([row])
    assert not result.ok
    assert any(issue.code == "feature_origin_after_timestamp" for issue in result.issues)


def test_validate_no_lookahead_combines_checks() -> None:
    feature_frame = build_feature_frame_from_snapshots(_snapshots())
    label_rows = build_label_rows_from_snapshots(_snapshots(), _label_config())
    label_frame = build_label_frame_from_snapshots(_snapshots(), _label_config())
    feature_frame["future_return_1"] = [0.0] * len(feature_frame)
    invalid_row = LabelRow.model_construct(
        timestamp=T0,
        symbol="TEST",
        labels={"future_return_1": 0.0},
        horizon_start=T0 - timedelta(seconds=1),
        horizon_end=T0,
        metadata={},
    )
    result = validate_no_lookahead(
        feature_frame,
        label_frame,
        label_rows=[*label_rows, invalid_row],
    )
    assert not result.ok
    assert result.error_count >= 2


def test_generated_feature_rows_do_not_include_future_label_columns() -> None:
    rows = [
        build_features_from_snapshot(snapshot)
        for snapshot in _snapshots()
    ]
    for row in rows:
        for column in row.features:
            assert not column.startswith("future_")
            assert not column.startswith("label")
            assert not column.startswith("target")


def test_generated_labels_do_not_alter_feature_frames() -> None:
    feature_frame = build_feature_frame_from_snapshots(_snapshots())
    before = feature_frame.copy(deep=True)
    build_label_frame_from_snapshots(_snapshots(), _label_config())
    pd.testing.assert_frame_equal(feature_frame, before)


def test_label_timestamps_and_future_horizons_are_explicit() -> None:
    rows = build_label_rows_from_snapshots(_snapshots(), _label_config())
    for row in rows:
        assert row.horizon_start >= row.timestamp
        assert row.horizon_end > row.horizon_start
