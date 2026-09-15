"""Tests for sequence DataLoader configuration and factory helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from chronoslob.training.dataloaders import (  # noqa: E402
    DataLoaderConfig,
    build_dataloaders_for_split,
    create_sequence_dataloader,
)
from chronoslob.training.datasets import (  # noqa: E402
    SequenceDataset,
    SequenceWindowConfig,
)
from chronoslob.training.splitters import (  # noqa: E402
    SplitIndices,
    TemporalSplitConfig,
    temporal_train_validation_test_split,
)


def _make_frames(n_rows: int = 30, n_features: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [start + timedelta(seconds=index) for index in range(n_rows)]
    features: dict[str, list[float]] = {
        "timestamp": timestamps,
        "symbol": ["TEST"] * n_rows,
    }
    for feature_index in range(n_features):
        features[f"feat_{feature_index}"] = [
            float(row + feature_index) for row in range(n_rows)
        ]
    feature_frame = pd.DataFrame(features)

    label_frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["TEST"] * n_rows,
            "label": [row % 3 for row in range(n_rows)],
        }
    )
    return feature_frame, label_frame


def test_dataloader_config_validates_batch_size() -> None:
    with pytest.raises(ValueError):
        DataLoaderConfig(batch_size=0)


def test_dataloader_config_validates_num_workers() -> None:
    with pytest.raises(ValueError):
        DataLoaderConfig(num_workers=-1)


def test_dataloader_config_default_shuffle_is_false() -> None:
    config = DataLoaderConfig()

    assert config.shuffle is False


def test_create_sequence_dataloader_returns_batches() -> None:
    feature_frame, label_frame = _make_frames(n_rows=10)
    config = SequenceWindowConfig(lookback=2, target_column="label")
    dataset = SequenceDataset(feature_frame, label_frame, config)

    loader = create_sequence_dataloader(dataset, DataLoaderConfig(batch_size=3))
    batch = next(iter(loader))

    assert tuple(batch["x"].shape) == (3, 2, dataset.n_features)
    assert tuple(batch["y"].shape) == (3,)


def test_create_sequence_dataloader_default_shuffle_false_preserves_order() -> None:
    feature_frame, label_frame = _make_frames(n_rows=10)
    config = SequenceWindowConfig(lookback=2, target_column="label")
    dataset = SequenceDataset(feature_frame, label_frame, config)

    loader = create_sequence_dataloader(dataset)

    target_indices: list[int] = []
    for batch in loader:
        target_indices.extend(batch["target_index"].tolist())
    expected_targets = [sample.target_index for sample in dataset.sample_indices]
    assert target_indices == expected_targets


def test_build_dataloaders_for_split_creates_train_validation_test_loaders() -> None:
    feature_frame, label_frame = _make_frames(n_rows=30)
    split = temporal_train_validation_test_split(
        30,
        TemporalSplitConfig(
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
            min_train_size=1,
            min_validation_size=1,
            min_test_size=1,
        ),
    )
    sequence_config = SequenceWindowConfig(lookback=3, target_column="label")

    loaders = build_dataloaders_for_split(
        feature_frame,
        label_frame,
        split,
        sequence_config,
        DataLoaderConfig(batch_size=4),
    )

    assert set(loaders.keys()) == {"train", "validation", "test"}
    for key in loaders:
        batch = next(iter(loaders[key]))
        assert tuple(batch["x"].shape[1:]) == (3, loaders[key].dataset.n_features)


def test_dataloader_windows_stay_inside_split_partitions() -> None:
    feature_frame, label_frame = _make_frames(n_rows=30)
    split = temporal_train_validation_test_split(
        30,
        TemporalSplitConfig(
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
            min_train_size=1,
            min_validation_size=1,
            min_test_size=1,
        ),
    )
    sequence_config = SequenceWindowConfig(lookback=3, target_column="label")
    loaders = build_dataloaders_for_split(
        feature_frame,
        label_frame,
        split,
        sequence_config,
        DataLoaderConfig(batch_size=2),
    )

    partition_indices = {
        "train": set(split.train),
        "validation": set(split.validation),
        "test": set(split.test),
    }
    for partition, loader in loaders.items():
        for sample in loader.dataset.sample_indices:
            window = range(sample.window_start, sample.window_end + 1)
            assert all(
                row in partition_indices[partition] for row in window
            ), f"window crosses {partition} partition: {list(window)}"


def test_validation_and_test_loaders_reuse_train_class_to_index() -> None:
    feature_frame, label_frame = _make_frames(n_rows=30)
    split = temporal_train_validation_test_split(
        30,
        TemporalSplitConfig(
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
            min_train_size=1,
            min_validation_size=1,
            min_test_size=1,
        ),
    )
    sequence_config = SequenceWindowConfig(lookback=3, target_column="label")
    loaders = build_dataloaders_for_split(
        feature_frame,
        label_frame,
        split,
        sequence_config,
        DataLoaderConfig(batch_size=4),
    )

    train_mapping = loaders["train"].dataset.class_to_index
    assert loaders["validation"].dataset.class_to_index == train_mapping
    assert loaders["test"].dataset.class_to_index == train_mapping


def test_tiny_fi2010_fixture_smoke_path_builds_loaders() -> None:
    fixture_path = Path("tests/fixtures/fi2010/tiny_fi2010_like.csv")
    if not fixture_path.exists():
        pytest.skip("FI-2010 fixture missing")

    from chronoslob.data.fi2010 import FI2010Config, load_fi2010
    from chronoslob.features.pipeline import (
        FeaturePipelineConfig,
        build_feature_frame_from_fi2010,
    )
    from chronoslob.labels.pipeline import build_label_frame_from_fi2010
    from chronoslob.models.preprocessing import align_feature_label_frames
    from chronoslob.training.datasets import encode_target_values

    dataset = load_fi2010(
        FI2010Config(
            path=fixture_path,
            timestamp_column="timestamp",
            split_column="split",
            label_columns=["label_10", "label_50", "label_100"],
            price_level_count=2,
        )
    )
    feature_frame = build_feature_frame_from_fi2010(
        dataset,
        FeaturePipelineConfig(
            include_order_flow=False,
            include_volatility=False,
        ),
    )
    labels = build_label_frame_from_fi2010(dataset, prefer_existing_labels=True)
    label_frame = labels.loc[:, ["timestamp", "symbol", "label_10"]]

    aligned = align_feature_label_frames(feature_frame, label_frame)
    split = temporal_train_validation_test_split(
        len(aligned),
        TemporalSplitConfig(
            train_fraction=0.5,
            validation_fraction=0.34,
            test_fraction=0.16,
            min_train_size=1,
            min_validation_size=1,
            min_test_size=0,
        ),
    )
    sequence_config = SequenceWindowConfig(lookback=2, target_column="label_10")
    _, full_mapping = encode_target_values(aligned.loc[:, "label_10"].tolist())

    loaders = build_dataloaders_for_split(
        feature_frame,
        label_frame,
        split,
        sequence_config,
        DataLoaderConfig(batch_size=2),
        class_to_index=full_mapping,
    )

    assert "train" in loaders
    assert "validation" in loaders
    train_batch = next(iter(loaders["train"]))
    assert train_batch["x"].ndim == 3
    assert train_batch["x"].shape[1] == 2


def test_build_dataloaders_raises_on_invalid_split_type() -> None:
    feature_frame, label_frame = _make_frames(n_rows=10)
    sequence_config = SequenceWindowConfig(lookback=2, target_column="label")

    with pytest.raises(TypeError):
        build_dataloaders_for_split(
            feature_frame,
            label_frame,
            split={"train": [0, 1]},  # type: ignore[arg-type]
            sequence_config=sequence_config,
        )


def test_build_dataloaders_raises_when_train_has_too_few_rows_for_lookback() -> None:
    feature_frame, label_frame = _make_frames(n_rows=10)
    sequence_config = SequenceWindowConfig(lookback=4, target_column="label")
    split = SplitIndices(train=[0, 1], validation=[2, 3, 4, 5], test=[])

    with pytest.raises(ValueError, match="training split has no usable"):
        build_dataloaders_for_split(
            feature_frame,
            label_frame,
            split,
            sequence_config,
        )
