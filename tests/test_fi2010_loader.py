"""Tests for the FI-2010 local loader and validation helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronoslob.data.fi2010 import (
    FI2010Config,
    FI2010Dataset,
    build_snapshot_from_row,
    infer_fi2010_columns,
    load_fi2010,
)
from chronoslob.data.schemas import OrderBookSnapshot
from chronoslob.data.validation import (
    DataValidationError,
    DataValidationResult,
    validate_fi2010_dataset,
    validate_numeric_frame,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "fi2010" / "tiny_fi2010_like.csv"
)
LABEL_COLUMNS = ["label_10", "label_50", "label_100"]


def _make_default_config(path: Path | None = None) -> FI2010Config:
    return FI2010Config(
        path=path or FIXTURE_PATH,
        timestamp_column="timestamp",
        split_column="split",
        label_columns=list(LABEL_COLUMNS),
        price_level_count=2,
    )


# ---------------------------------------------------------------------------
# FI2010Config validation
# ---------------------------------------------------------------------------


def test_config_rejects_empty_path() -> None:
    with pytest.raises((ValueError, TypeError)):
        FI2010Config(path="")


def test_config_rejects_non_positive_price_level_count() -> None:
    with pytest.raises(ValueError):
        FI2010Config(path=FIXTURE_PATH, price_level_count=0)


def test_config_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError):
        FI2010Config(path=FIXTURE_PATH, symbol="")


def test_config_rejects_empty_delimiter() -> None:
    with pytest.raises(ValueError):
        FI2010Config(path=FIXTURE_PATH, delimiter="")


def test_config_rejects_overlapping_feature_and_label_columns() -> None:
    with pytest.raises(ValueError):
        FI2010Config(
            path=FIXTURE_PATH,
            feature_columns=["a", "b"],
            label_columns=["b", "c"],
        )


def test_config_accepts_str_path_and_coerces() -> None:
    config = FI2010Config(path=str(FIXTURE_PATH))
    assert isinstance(config.path, Path)
    assert config.path == FIXTURE_PATH


# ---------------------------------------------------------------------------
# Missing file
# ---------------------------------------------------------------------------


def test_load_fi2010_raises_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.csv"
    config = FI2010Config(path=missing)
    with pytest.raises(FileNotFoundError):
        load_fi2010(config)


def test_load_fi2010_raises_for_directory_path(tmp_path: Path) -> None:
    config = FI2010Config(path=tmp_path)
    with pytest.raises(FileNotFoundError):
        load_fi2010(config)


# ---------------------------------------------------------------------------
# Loading the tiny fixture
# ---------------------------------------------------------------------------


def test_load_tiny_fixture_succeeds() -> None:
    dataset = load_fi2010(_make_default_config())
    assert isinstance(dataset, FI2010Dataset)
    assert dataset.n_rows == 6


def test_features_and_labels_inferred_when_configured() -> None:
    dataset = load_fi2010(_make_default_config())
    assert dataset.label_columns == LABEL_COLUMNS
    # Features should exclude timestamp, split and label columns and be numeric.
    for column in dataset.feature_columns:
        assert column not in {"timestamp", "split", *LABEL_COLUMNS}
    assert "bid_price_1" in dataset.feature_columns
    assert "ask_quantity_2" in dataset.feature_columns


def test_get_features_returns_only_features() -> None:
    dataset = load_fi2010(_make_default_config())
    features = dataset.get_features()
    assert list(features.columns) == dataset.feature_columns
    assert not any(c in features.columns for c in LABEL_COLUMNS)


def test_get_labels_returns_only_labels() -> None:
    dataset = load_fi2010(_make_default_config())
    labels = dataset.get_labels()
    assert list(labels.columns) == LABEL_COLUMNS


def test_dataset_sizing_properties() -> None:
    dataset = load_fi2010(_make_default_config())
    assert dataset.n_rows == 6
    assert dataset.n_features == len(dataset.feature_columns)
    assert dataset.n_labels == len(LABEL_COLUMNS)
    assert dataset.has_labels is True


def test_dataset_without_labels_reports_no_labels(tmp_path: Path) -> None:
    config = FI2010Config(
        path=FIXTURE_PATH,
        timestamp_column="timestamp",
        split_column="split",
        price_level_count=2,
    )
    dataset = load_fi2010(config)
    assert dataset.has_labels is False
    assert dataset.n_labels == 0


def test_split_values_returns_series() -> None:
    dataset = load_fi2010(_make_default_config())
    split = dataset.split_values()
    assert split is not None
    assert set(split.unique()) == {"train", "test"}


def test_split_values_none_when_unconfigured() -> None:
    config = FI2010Config(
        path=FIXTURE_PATH,
        timestamp_column="timestamp",
        label_columns=list(LABEL_COLUMNS),
        price_level_count=2,
    )
    dataset = load_fi2010(config)
    assert dataset.split_values() is None


def test_describe_returns_stable_metadata() -> None:
    dataset = load_fi2010(_make_default_config())
    summary = dataset.describe()
    assert summary["symbol"] == "FI2010"
    assert summary["n_rows"] == 6
    assert summary["has_labels"] is True
    assert summary["has_split_column"] is True
    assert summary["has_timestamp_column"] is True


# ---------------------------------------------------------------------------
# Column inference
# ---------------------------------------------------------------------------


def test_infer_columns_with_no_explicit_config() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0],
            "b": [3.0, 4.0],
            "c": ["x", "y"],
        }
    )
    config = FI2010Config(path=FIXTURE_PATH)
    features, labels = infer_fi2010_columns(frame, config)
    assert features == ["a", "b"]
    assert labels == []


def test_infer_columns_with_explicit_labels_only() -> None:
    frame = pd.DataFrame(
        {
            "f1": [1.0, 2.0],
            "f2": [3.0, 4.0],
            "label_a": [0, 1],
            "timestamp": ["2024-01-01", "2024-01-02"],
        }
    )
    config = FI2010Config(
        path=FIXTURE_PATH,
        timestamp_column="timestamp",
        label_columns=["label_a"],
    )
    features, labels = infer_fi2010_columns(frame, config)
    assert "label_a" not in features
    assert "timestamp" not in features
    assert labels == ["label_a"]


# ---------------------------------------------------------------------------
# Validation: NaN, infinity, missing
# ---------------------------------------------------------------------------


def test_validation_catches_nan(tmp_path: Path) -> None:
    bad = tmp_path / "nan.csv"
    bad.write_text(
        "bid_price_1,bid_quantity_1,ask_price_1,ask_quantity_1,label\n"
        "1.0,2.0,3.0,4.0,0\n"
        ",2.5,3.5,4.5,1\n",
        encoding="utf-8",
    )
    config = FI2010Config(
        path=bad,
        label_columns=["label"],
        price_level_count=1,
    )
    with pytest.raises(DataValidationError):
        load_fi2010(config)


def test_validation_catches_infinity(tmp_path: Path) -> None:
    bad = tmp_path / "inf.csv"
    bad.write_text(
        "bid_price_1,bid_quantity_1,ask_price_1,ask_quantity_1,label\n"
        "1.0,2.0,3.0,4.0,0\n"
        "inf,2.5,3.5,4.5,1\n",
        encoding="utf-8",
    )
    config = FI2010Config(
        path=bad,
        label_columns=["label"],
        price_level_count=1,
    )
    with pytest.raises(DataValidationError):
        load_fi2010(config)


def test_validate_numeric_frame_catches_missing_required_columns() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    result = validate_numeric_frame(frame, required_columns=["a", "b"])
    assert result.ok is False
    codes = {issue.code for issue in result.issues}
    assert "missing_required_columns" in codes


def test_validate_numeric_frame_catches_inf() -> None:
    frame = pd.DataFrame({"a": [1.0, np.inf]})
    result = validate_numeric_frame(frame)
    assert result.ok is False
    codes = {issue.code for issue in result.issues}
    assert "inf_in_numeric_column" in codes


def test_validate_numeric_frame_catches_nan() -> None:
    frame = pd.DataFrame({"a": [1.0, np.nan]})
    result = validate_numeric_frame(frame)
    assert result.ok is False


def test_validate_numeric_frame_allows_nan_when_enabled() -> None:
    frame = pd.DataFrame({"a": [1.0, np.nan]})
    result = validate_numeric_frame(frame, allow_nan=True)
    nan_codes = [
        issue.code
        for issue in result.issues
        if issue.code == "nan_in_numeric_column"
    ]
    assert nan_codes == []


def test_validate_numeric_frame_rejects_empty_frame() -> None:
    frame = pd.DataFrame({"a": pd.Series([], dtype="float64")})
    result = validate_numeric_frame(frame)
    assert result.ok is False
    codes = {issue.code for issue in result.issues}
    assert "empty_frame" in codes


def test_data_validation_result_summary_and_raise() -> None:
    frame = pd.DataFrame({"a": [1.0, np.nan]})
    result: DataValidationResult = validate_numeric_frame(frame)
    summary = result.summary()
    assert summary["ok"] is False
    assert summary["error_count"] >= 1
    with pytest.raises(DataValidationError):
        result.raise_if_errors()


# ---------------------------------------------------------------------------
# has_header=False creates col_0, col_1 names
# ---------------------------------------------------------------------------


def test_headerless_loading_creates_positional_columns(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    path.write_text(
        "1.0,2.0,3.0,4.0\n"
        "1.1,2.1,3.1,4.1\n"
        "1.2,2.2,3.2,4.2\n",
        encoding="utf-8",
    )
    config = FI2010Config(path=path, has_header=False, price_level_count=1)
    dataset = load_fi2010(config)
    assert list(dataset.frame.columns) == ["col_0", "col_1", "col_2", "col_3"]
    assert dataset.feature_columns == ["col_0", "col_1", "col_2", "col_3"]


# ---------------------------------------------------------------------------
# Snapshot conversion
# ---------------------------------------------------------------------------


def test_to_snapshot_rows_builds_snapshots_when_lob_columns_present() -> None:
    dataset = load_fi2010(_make_default_config())
    snapshots = dataset.to_snapshot_rows()
    assert len(snapshots) == dataset.n_rows
    assert all(isinstance(snap, OrderBookSnapshot) for snap in snapshots)

    first = snapshots[0]
    assert first.symbol == "FI2010"
    assert len(first.bids) == 2
    assert len(first.asks) == 2
    # Bid ordering is strictly descending.
    assert first.bids[0].price > first.bids[1].price
    # Ask ordering is strictly ascending.
    assert first.asks[0].price < first.asks[1].price


def test_to_snapshot_rows_uses_real_timestamps_when_available() -> None:
    dataset = load_fi2010(_make_default_config())
    snapshots = dataset.to_snapshot_rows(max_rows=1)
    assert snapshots[0].timestamp == datetime(
        2024, 1, 2, 9, 30, 0, tzinfo=UTC
    )
    assert snapshots[0].metadata["synthetic_time"] is False


def test_to_snapshot_rows_synthetic_time_when_no_timestamp_column(
    tmp_path: Path,
) -> None:
    path = tmp_path / "no_ts.csv"
    path.write_text(
        "bid_price_1,bid_quantity_1,ask_price_1,ask_quantity_1,label\n"
        "99.0,10,100.0,11,0\n"
        "99.1,10,100.1,11,1\n",
        encoding="utf-8",
    )
    config = FI2010Config(
        path=path,
        label_columns=["label"],
        price_level_count=1,
    )
    dataset = load_fi2010(config)
    snapshots = dataset.to_snapshot_rows()
    assert len(snapshots) == 2
    assert all(snap.metadata["synthetic_time"] is True for snap in snapshots)
    # Synthetic timestamps must still be timezone-aware.
    for snap in snapshots:
        assert snap.timestamp.tzinfo is not None


def test_to_snapshot_rows_returns_empty_when_lob_absent_and_allowed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "no_lob.csv"
    path.write_text(
        "feature_a,feature_b,label\n"
        "1.0,2.0,0\n"
        "1.1,2.1,1\n",
        encoding="utf-8",
    )
    config = FI2010Config(
        path=path,
        label_columns=["label"],
        price_level_count=1,
        allow_missing_lob_columns=True,
    )
    dataset = load_fi2010(config)
    assert dataset.to_snapshot_rows() == []


def test_to_snapshot_rows_raises_when_lob_absent_and_not_allowed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "no_lob.csv"
    path.write_text(
        "feature_a,feature_b,label\n"
        "1.0,2.0,0\n"
        "1.1,2.1,1\n",
        encoding="utf-8",
    )
    config = FI2010Config(
        path=path,
        label_columns=["label"],
        price_level_count=1,
        allow_missing_lob_columns=False,
    )
    dataset = load_fi2010(config)
    with pytest.raises(ValueError):
        dataset.to_snapshot_rows()


def test_to_snapshot_rows_max_rows_limits_output() -> None:
    dataset = load_fi2010(_make_default_config())
    snapshots = dataset.to_snapshot_rows(max_rows=2)
    assert len(snapshots) == 2


def test_to_snapshot_rows_accepts_size_alias(tmp_path: Path) -> None:
    path = tmp_path / "alias.csv"
    path.write_text(
        "bid_price_1,bid_size_1,ask_price_1,ask_size_1,label\n"
        "99.0,10,100.0,11,0\n"
        "99.1,10,100.1,11,1\n",
        encoding="utf-8",
    )
    config = FI2010Config(
        path=path,
        label_columns=["label"],
        price_level_count=1,
    )
    dataset = load_fi2010(config)
    snapshots = dataset.to_snapshot_rows()
    assert len(snapshots) == 2
    assert snapshots[0].bids[0].quantity == 10.0


def test_build_snapshot_from_row_uses_provided_timestamp() -> None:
    row = pd.Series(
        {
            "bid_price_1": 99.0,
            "bid_quantity_1": 10.0,
            "ask_price_1": 100.0,
            "ask_quantity_1": 11.0,
        }
    )
    ts = datetime(2024, 5, 1, 0, 0, 0, tzinfo=UTC)
    config = FI2010Config(path=FIXTURE_PATH, price_level_count=1)
    snapshot = build_snapshot_from_row(row, config, ts)
    assert snapshot is not None
    assert snapshot.timestamp == ts
    assert snapshot.best_bid is not None
    assert snapshot.best_bid.price == 99.0


def test_build_snapshot_from_row_returns_none_when_columns_absent() -> None:
    row = pd.Series({"feature": 1.0})
    ts = datetime(2024, 5, 1, 0, 0, 0, tzinfo=UTC)
    config = FI2010Config(path=FIXTURE_PATH, price_level_count=1)
    assert build_snapshot_from_row(row, config, ts) is None


def test_build_snapshot_from_row_rejects_naive_timestamp() -> None:
    row = pd.Series(
        {
            "bid_price_1": 99.0,
            "bid_quantity_1": 10.0,
            "ask_price_1": 100.0,
            "ask_quantity_1": 11.0,
        }
    )
    config = FI2010Config(path=FIXTURE_PATH, price_level_count=1)
    with pytest.raises(ValueError):
        build_snapshot_from_row(row, config, datetime(2024, 5, 1, 0, 0, 0))


# ---------------------------------------------------------------------------
# validate_fi2010_dataset surfaces issues on a hand-built dataset
# ---------------------------------------------------------------------------


def test_validate_fi2010_dataset_reports_missing_feature_column() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    config = FI2010Config(path=FIXTURE_PATH, price_level_count=1)
    dataset = FI2010Dataset(
        frame=frame,
        config=config,
        feature_columns=["a", "missing"],
        label_columns=[],
    )
    result = validate_fi2010_dataset(dataset)
    codes = {issue.code for issue in result.issues}
    assert "missing_feature_column" in codes
    assert result.ok is False


def test_validate_fi2010_dataset_reports_incomplete_lob_level() -> None:
    frame = pd.DataFrame(
        {
            "bid_price_1": [1.0, 2.0],
            # Missing bid_quantity_1, ask_price_1, ask_quantity_1
        }
    )
    config = FI2010Config(path=FIXTURE_PATH, price_level_count=1)
    dataset = FI2010Dataset(
        frame=frame,
        config=config,
        feature_columns=["bid_price_1"],
        label_columns=[],
    )
    result = validate_fi2010_dataset(dataset)
    codes = {issue.code for issue in result.issues}
    assert "incomplete_lob_level" in codes
