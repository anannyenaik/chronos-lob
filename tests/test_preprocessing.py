"""Tests for baseline preprocessing utilities."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from chronoslob.models.preprocessing import (
    FeatureMatrix,
    TrainOnlyStandardScaler,
    align_feature_label_frames,
    build_feature_matrix,
    build_target_vector,
    select_feature_columns,
)


def _timestamps(n_rows: int) -> list[datetime]:
    start = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
    return [start + timedelta(seconds=index) for index in range(n_rows)]


def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": _timestamps(3),
            "symbol": ["TEST"] * 3,
            "split": ["train", "train", "validation"],
            "mid_price": [100.0, 101.0, 102.0],
            "spread": [1.0, 1.1, 1.2],
            "venue": ["X", "X", "X"],
        }
    )


def test_select_feature_columns_excludes_metadata_columns() -> None:
    columns = select_feature_columns(_feature_frame())

    assert columns == ["mid_price", "spread"]


def test_select_feature_columns_rejects_label_like_columns() -> None:
    frame = _feature_frame()
    frame["label_10"] = [1, 2, 1]

    with pytest.raises(ValueError, match="look like labels"):
        select_feature_columns(frame)


def test_build_feature_matrix_validates_shape_and_names() -> None:
    matrix = build_feature_matrix(
        _feature_frame(),
        feature_columns=["mid_price", "spread"],
        row_indices=[0, 2],
    )

    assert matrix.x.shape == (2, 2)
    assert matrix.feature_names == ["mid_price", "spread"]
    assert matrix.row_indices == [0, 2]

    with pytest.raises(ValueError, match="feature_names length"):
        FeatureMatrix(
            x=np.asarray([[1.0, 2.0]]),
            feature_names=["one"],
            row_indices=[0],
        )


def test_build_feature_matrix_rejects_nan_and_inf_by_default() -> None:
    frame = _feature_frame()
    frame.loc[0, "mid_price"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        build_feature_matrix(frame, feature_columns=["mid_price"])

    frame = _feature_frame()
    frame.loc[0, "mid_price"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        build_feature_matrix(frame, feature_columns=["mid_price"])


def test_build_target_vector_extracts_target_and_classes() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": _timestamps(3),
            "symbol": ["TEST"] * 3,
            "direction_1": ["up", "down", "up"],
        }
    )

    target = build_target_vector(frame, target_column="direction_1", row_indices=[0, 1])

    assert target.y.tolist() == ["up", "down"]
    assert target.target_name == "direction_1"
    assert target.row_indices == [0, 1]
    assert target.classes == ["down", "up"]


def test_build_target_vector_rejects_missing_and_nan_targets() -> None:
    frame = pd.DataFrame({"target": [1.0, np.nan]})

    with pytest.raises(ValueError, match="missing"):
        build_target_vector(frame, target_column="missing")
    with pytest.raises(ValueError, match="NaN"):
        build_target_vector(frame, target_column="target")


def test_align_feature_label_frames_joins_on_timestamp_and_symbol() -> None:
    feature_frame = _feature_frame()
    label_frame = pd.DataFrame(
        {
            "timestamp": _timestamps(3)[1:],
            "symbol": ["TEST", "TEST"],
            "direction_1": [0, 1],
        }
    )

    aligned = align_feature_label_frames(feature_frame, label_frame)

    assert aligned["timestamp"].tolist() == _timestamps(3)[1:]
    assert "mid_price" in aligned.columns
    assert "direction_1" in aligned.columns


def test_align_feature_label_frames_rejects_overlapping_non_key_columns() -> None:
    label_frame = pd.DataFrame(
        {
            "timestamp": _timestamps(3),
            "symbol": ["TEST"] * 3,
            "split": ["train", "train", "validation"],
            "direction_1": [0, 1, 0],
        }
    )

    with pytest.raises(ValueError, match="overlapping non-key"):
        align_feature_label_frames(_feature_frame(), label_frame)


def test_train_only_standard_scaler_fit_transform_behaviour() -> None:
    scaler = TrainOnlyStandardScaler()
    train = np.asarray([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])

    transformed_train = scaler.fit_transform(train)
    validation = scaler.transform(np.asarray([[7.0, 22.0]]))

    assert scaler.is_fitted
    assert scaler.mean_.tolist() == [3.0, 14.0]
    assert transformed_train.shape == train.shape
    assert validation.shape == (1, 2)


def test_train_only_standard_scaler_transform_before_fit_raises() -> None:
    scaler = TrainOnlyStandardScaler()

    with pytest.raises(ValueError, match="fitted"):
        scaler.transform(np.asarray([[1.0]]))


def test_train_only_standard_scaler_stats_do_not_change_on_transform() -> None:
    scaler = TrainOnlyStandardScaler()
    scaler.fit(np.asarray([[0.0], [2.0], [4.0]]))
    mean_before = scaler.mean_.copy()
    scale_before = scaler.scale_.copy()

    scaler.transform(np.asarray([[100.0], [200.0]]))

    assert scaler.mean_.tolist() == mean_before.tolist()
    assert scaler.scale_.tolist() == scale_before.tolist()
