"""Tests for the label pipeline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from chronoslob.data.fi2010 import FI2010Config, load_fi2010
from chronoslob.data.schemas import LabelRow, OrderBookLevel, OrderBookSnapshot
from chronoslob.labels.pipeline import (
    LabelPipelineConfig,
    build_label_frame_from_fi2010,
    build_label_frame_from_snapshots,
    build_label_rows_from_snapshots,
    validate_label_frame,
)

T0 = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "fi2010" / "tiny_fi2010_like.csv"
)
LABEL_COLUMNS = ["label_10", "label_50", "label_100"]


def _snapshot(
    offset: int,
    bid_price: float,
    bid_quantity: float,
    ask_price: float,
    ask_quantity: float,
) -> OrderBookSnapshot:
    return OrderBookSnapshot(
        timestamp=T0 + timedelta(seconds=offset),
        symbol="TEST",
        bids=[OrderBookLevel(price=bid_price, quantity=bid_quantity)],
        asks=[OrderBookLevel(price=ask_price, quantity=ask_quantity)],
    )


def _snapshots() -> list[OrderBookSnapshot]:
    return [
        _snapshot(0, 100.0, 10.0, 101.0, 10.0),
        _snapshot(1, 100.0, 9.0, 101.0, 9.0),
        _snapshot(2, 100.5, 8.0, 101.5, 8.0),
        _snapshot(3, 100.2, 8.0, 101.2, 8.0),
    ]


def _config() -> LabelPipelineConfig:
    return LabelPipelineConfig(
        horizons=(1, 2),
        fill_horizon=1,
        adverse_evaluation_horizon=2,
        return_threshold=0.0,
    )


def _fi2010_dataset_with_labels():
    config = FI2010Config(
        path=FIXTURE_PATH,
        timestamp_column="timestamp",
        split_column="split",
        label_columns=list(LABEL_COLUMNS),
        price_level_count=2,
    )
    return load_fi2010(config)


def test_build_label_rows_from_snapshots_returns_label_rows() -> None:
    rows = build_label_rows_from_snapshots(_snapshots(), _config())
    assert rows
    assert all(isinstance(row, LabelRow) for row in rows)


def test_label_names_include_requested_horizons() -> None:
    row = build_label_rows_from_snapshots(_snapshots(), _config())[0]
    assert "future_return_1" in row.labels
    assert "direction_2" in row.labels
    assert "return_quantile_2" in row.labels
    assert "future_volatility_1" in row.labels
    assert "spread_widening_2" in row.labels
    assert "bid_fill_proxy_1" in row.labels
    assert "ask_adverse_selection_proxy_2" in row.labels


def test_label_horizon_start_and_end_are_correct() -> None:
    row = build_label_rows_from_snapshots(_snapshots(), _config())[0]
    assert row.timestamp == T0
    assert row.horizon_start == T0 + timedelta(seconds=1)
    assert row.horizon_end == T0 + timedelta(seconds=2)


def test_build_label_frame_from_snapshots_returns_expected_columns() -> None:
    frame = build_label_frame_from_snapshots(_snapshots(), _config())
    assert {"timestamp", "symbol", "horizon_start", "horizon_end"}.issubset(
        frame.columns
    )
    assert "future_return_1" in frame.columns
    assert "direction_1" in frame.columns


def test_build_label_frame_from_fi2010_extracts_existing_fixture_labels() -> None:
    dataset = _fi2010_dataset_with_labels()
    frame = build_label_frame_from_fi2010(dataset, prefer_existing_labels=True)
    assert len(frame) == dataset.n_rows
    for column in LABEL_COLUMNS:
        assert column in frame.columns
    assert set(frame["label_source"]) == {"fi2010_existing_labels"}


def test_build_label_frame_from_fi2010_generates_when_labels_absent() -> None:
    config = FI2010Config(
        path=FIXTURE_PATH,
        timestamp_column="timestamp",
        split_column="split",
        price_level_count=2,
    )
    dataset = load_fi2010(config)
    frame = build_label_frame_from_fi2010(
        dataset,
        LabelPipelineConfig(
            horizons=(1,),
            fill_horizon=1,
            adverse_evaluation_horizon=1,
        ),
        prefer_existing_labels=True,
    )
    assert len(frame) == dataset.n_rows - 1
    assert "future_return_1" in frame.columns
    assert "label_10" not in frame.columns


def test_label_frame_does_not_contain_feature_columns() -> None:
    frame = build_label_frame_from_snapshots(_snapshots(), _config())
    for column in ("mid_price", "spread", "microprice", "bid_depth_1"):
        assert column not in frame.columns


def test_validate_label_frame_catches_infinity() -> None:
    frame = build_label_frame_from_snapshots(_snapshots(), _config())
    frame.loc[0, "future_return_1"] = np.inf
    result = validate_label_frame(frame)
    assert not result.ok
    assert any(issue.code == "inf_in_label" for issue in result.issues)


def test_validate_label_frame_catches_suspicious_feature_columns() -> None:
    frame = build_label_frame_from_snapshots(_snapshots(), _config())
    frame["mid_price"] = [100.0] * len(frame)
    result = validate_label_frame(frame)
    assert not result.ok
    assert any(
        issue.code == "feature_like_column_in_labels" for issue in result.issues
    )


def test_validate_label_frame_accepts_existing_fi2010_benchmark_labels() -> None:
    dataset = _fi2010_dataset_with_labels()
    frame = build_label_frame_from_fi2010(dataset, prefer_existing_labels=True)
    result = validate_label_frame(frame)
    assert result.ok


def test_validate_label_frame_catches_empty_frame() -> None:
    result = validate_label_frame(pd.DataFrame())
    assert not result.ok
    assert any(issue.code == "empty_frame" for issue in result.issues)
