"""Tests for the PyTorch SequenceDataset and tensor standardiser."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from chronoslob.training.datasets import (  # noqa: E402
    SequenceDataset,
    SequenceWindowConfig,
    TorchSequenceStandardiser,
    encode_target_values,
)


def _make_frames(n_rows: int = 10, n_features: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [start + timedelta(seconds=index) for index in range(n_rows)]
    feature_data: dict[str, list[float]] = {
        "timestamp": timestamps,
        "symbol": ["TEST"] * n_rows,
    }
    for feature_index in range(n_features):
        feature_data[f"feat_{feature_index}"] = [
            float(row + feature_index) for row in range(n_rows)
        ]
    feature_frame = pd.DataFrame(feature_data)

    labels = [int(row % 3) for row in range(n_rows)]
    label_frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["TEST"] * n_rows,
            "label": labels,
        }
    )
    return feature_frame, label_frame


def test_sequence_dataset_length_and_item_format() -> None:
    feature_frame, label_frame = _make_frames(n_rows=6, n_features=2)
    config = SequenceWindowConfig(lookback=3, target_column="label")

    dataset = SequenceDataset(feature_frame, label_frame, config)

    assert len(dataset) == 4  # rows 2,3,4,5
    sample = dataset[0]
    assert set(sample.keys()) == {
        "x",
        "y",
        "target_index",
        "window_start",
        "window_end",
    }


def test_sequence_dataset_x_shape_is_lookback_by_n_features() -> None:
    feature_frame, label_frame = _make_frames(n_rows=6, n_features=4)
    config = SequenceWindowConfig(lookback=3, target_column="label")

    dataset = SequenceDataset(feature_frame, label_frame, config)
    sample = dataset[0]

    assert tuple(sample["x"].shape) == (3, dataset.n_features)
    assert dataset.n_features == 4


def test_sequence_dataset_y_is_scalar_long_tensor() -> None:
    feature_frame, label_frame = _make_frames(n_rows=6)
    config = SequenceWindowConfig(lookback=2, target_column="label")

    dataset = SequenceDataset(feature_frame, label_frame, config)
    sample = dataset[0]

    assert sample["y"].dtype == torch.long
    assert sample["y"].ndim == 0


def test_class_mapping_is_deterministic_and_sorted() -> None:
    feature_frame, label_frame = _make_frames(n_rows=9)
    config = SequenceWindowConfig(lookback=2, target_column="label")

    dataset = SequenceDataset(feature_frame, label_frame, config)

    assert dataset.class_to_index == {0: 0, 1: 1, 2: 2}
    assert dataset.index_to_class == {0: 0, 1: 1, 2: 2}


def test_validation_can_reuse_train_class_to_index() -> None:
    feature_frame, label_frame = _make_frames(n_rows=12)
    config = SequenceWindowConfig(lookback=2, target_column="label")

    train_dataset = SequenceDataset(
        feature_frame,
        label_frame,
        config,
        allowed_target_indices=list(range(0, 9)),
    )
    validation_dataset = SequenceDataset(
        feature_frame,
        label_frame,
        config,
        class_to_index=train_dataset.class_to_index,
        allowed_target_indices=list(range(9, 12)),
    )

    assert validation_dataset.class_to_index == train_dataset.class_to_index


def test_unseen_validation_class_raises() -> None:
    feature_frame, label_frame = _make_frames(n_rows=6)
    config = SequenceWindowConfig(lookback=2, target_column="label")

    # Train mapping covers only {0, 1}; validation row 5 has class 2 (5 % 3 = 2).
    train_mapping = {0: 0, 1: 1}

    with pytest.raises(ValueError, match="class_to_index"):
        SequenceDataset(
            feature_frame,
            label_frame,
            config,
            class_to_index=train_mapping,
            allowed_target_indices=[4, 5],
        )


def test_label_like_feature_column_is_rejected() -> None:
    feature_frame, label_frame = _make_frames(n_rows=6)
    feature_frame = feature_frame.copy()
    feature_frame["label_leak"] = np.arange(len(feature_frame))
    config = SequenceWindowConfig(lookback=2, target_column="label")

    with pytest.raises(ValueError, match="label"):
        SequenceDataset(feature_frame, label_frame, config)


def test_target_alignment_matches_aligned_row() -> None:
    feature_frame, label_frame = _make_frames(n_rows=6)
    config = SequenceWindowConfig(lookback=2, target_column="label")

    dataset = SequenceDataset(feature_frame, label_frame, config)
    sample = dataset[0]
    expected_class = label_frame.loc[sample["target_index"], "label"]

    assert int(sample["y"].item()) == dataset.class_to_index[int(expected_class)]


def test_no_future_rows_in_x_window() -> None:
    feature_frame, label_frame = _make_frames(n_rows=8, n_features=2)
    config = SequenceWindowConfig(lookback=3, target_column="label")

    dataset = SequenceDataset(feature_frame, label_frame, config)

    aligned_features = feature_frame.loc[
        :, [column for column in feature_frame.columns if column.startswith("feat_")]
    ].to_numpy(dtype=np.float64)
    for sample in (dataset[index] for index in range(len(dataset))):
        target = int(sample["target_index"])
        window_start = int(sample["window_start"])
        window_end = int(sample["window_end"])
        assert window_end == target
        expected = aligned_features[window_start : window_end + 1, :]
        np.testing.assert_allclose(sample["x"].cpu().numpy(), expected)


def test_dataset_does_not_mutate_input_frames() -> None:
    feature_frame, label_frame = _make_frames(n_rows=6)
    feature_snapshot = feature_frame.copy(deep=True)
    label_snapshot = label_frame.copy(deep=True)
    config = SequenceWindowConfig(lookback=2, target_column="label")

    SequenceDataset(feature_frame, label_frame, config)

    pd.testing.assert_frame_equal(feature_frame, feature_snapshot)
    pd.testing.assert_frame_equal(label_frame, label_snapshot)


def test_encode_target_values_infers_sorted_mapping() -> None:
    encoded, mapping = encode_target_values([2, 1, 1, 3])

    assert mapping == {1: 0, 2: 1, 3: 2}
    assert encoded.tolist() == [1, 0, 0, 2]


def test_encode_target_values_uses_supplied_mapping() -> None:
    encoded, mapping = encode_target_values([1, 2], class_to_index={1: 5, 2: 9})

    assert mapping == {1: 5, 2: 9}
    assert encoded.tolist() == [5, 9]


def test_encode_target_values_raises_on_missing_class() -> None:
    with pytest.raises(ValueError):
        encode_target_values([1, 4], class_to_index={1: 0, 2: 1})


def test_standardiser_fit_from_frame_and_transform() -> None:
    frame = pd.DataFrame({"a": [0.0, 2.0, 4.0], "b": [1.0, 3.0, 5.0]})
    standardiser = TorchSequenceStandardiser()
    standardiser.fit_from_feature_frame(frame, ["a", "b"])

    x = torch.tensor([[4.0, 5.0], [0.0, 1.0]], dtype=torch.float32)
    transformed = standardiser.transform_tensor(x)

    expected_mean = torch.tensor([2.0, 3.0], dtype=torch.float64)
    expected_std = torch.tensor(
        [float(np.std([0.0, 2.0, 4.0])), float(np.std([1.0, 3.0, 5.0]))],
        dtype=torch.float64,
    )
    np.testing.assert_allclose(standardiser.mean.numpy(), expected_mean.numpy())
    np.testing.assert_allclose(standardiser.std.numpy(), expected_std.numpy())
    expected = (x.double() - expected_mean) / expected_std
    np.testing.assert_allclose(
        transformed.cpu().numpy(), expected.float().cpu().numpy(), atol=1e-6
    )


def test_standardiser_transform_before_fit_raises() -> None:
    standardiser = TorchSequenceStandardiser()
    with pytest.raises(ValueError, match="fitted"):
        standardiser.transform_tensor(torch.zeros((2, 3)))


def test_standardiser_zero_std_replaced_with_one() -> None:
    constant = torch.ones((4, 2), dtype=torch.float32)
    standardiser = TorchSequenceStandardiser().fit_from_sequences(constant)

    assert standardiser.std.tolist() == [1.0, 1.0]
    transformed = standardiser.transform_tensor(constant)
    assert torch.allclose(transformed, torch.zeros_like(constant))


def test_standardiser_does_not_mutate_input_tensor() -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    snapshot = x.clone()
    standardiser = TorchSequenceStandardiser().fit_from_sequences(x)
    standardiser.transform_tensor(x)

    assert torch.equal(x, snapshot)


def test_standardiser_rejects_nan_input() -> None:
    standardiser = TorchSequenceStandardiser()
    with pytest.raises(ValueError, match="NaN"):
        standardiser.fit_from_sequences(torch.tensor([[float("nan"), 1.0]]))


def test_standardiser_rejects_inf_input() -> None:
    standardiser = TorchSequenceStandardiser()
    with pytest.raises(ValueError, match="infinite"):
        standardiser.fit_from_sequences(torch.tensor([[float("inf"), 1.0]]))


def test_sequence_dataset_exposes_metadata_properties() -> None:
    feature_frame, label_frame = _make_frames(n_rows=6, n_features=2)
    config = SequenceWindowConfig(lookback=2, target_column="label")
    dataset = SequenceDataset(feature_frame, label_frame, config)

    assert dataset.feature_columns == ["feat_0", "feat_1"]
    assert dataset.n_classes == 3
    assert dataset.n_features == 2
    assert len(dataset.sample_indices) == len(dataset)
