"""DataLoader factories for sequence datasets.

These helpers wrap :class:`torch.utils.data.DataLoader` so that temporal
data is never shuffled by default and so that train/validation/test
loaders share a single class-to-index mapping inferred from training rows
only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

from chronoslob.training.batching import collate_fixed_length_batch
from chronoslob.training.datasets import (
    SequenceDataset,
    SequenceWindowConfig,
    encode_target_values,
)
from chronoslob.training.splitters import SplitIndices

try:  # pragma: no cover - exercised when torch is unavailable
    import torch
    from torch.utils.data import DataLoader

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]
    _TORCH_AVAILABLE = False


__all__ = [
    "DataLoaderConfig",
    "build_dataloaders_for_split",
    "create_sequence_dataloader",
]


_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=False, validate_assignment=True)


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for dataloaders. Install the 'torch' "
            "optional dependency: pip install -e '.[torch]'"
        )
    return torch


class DataLoaderConfig(BaseModel):
    """Configuration for a sequence-aware :class:`DataLoader`.

    ``shuffle`` defaults to ``False`` because reshuffling temporal samples
    breaks downstream evaluation. Callers must explicitly opt in to
    shuffling when training a stationary classifier.
    """

    model_config = _MODEL_CONFIG

    batch_size: int = 32
    shuffle: bool = False
    drop_last: bool = False
    num_workers: int = 0
    pin_memory: bool = False

    @field_validator("batch_size")
    @classmethod
    def _validate_batch_size(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("batch_size must be an integer")
        if value <= 0:
            raise ValueError("batch_size must be positive")
        return value

    @field_validator("num_workers")
    @classmethod
    def _validate_num_workers(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("num_workers must be an integer")
        if value < 0:
            raise ValueError("num_workers must be non-negative")
        return value


def create_sequence_dataloader(
    dataset: SequenceDataset,
    config: DataLoaderConfig | None = None,
) -> Any:
    """Wrap ``dataset`` in a :class:`DataLoader` with safe defaults.

    The default :class:`DataLoaderConfig` keeps ``shuffle=False`` so that
    temporal order is preserved unless the caller explicitly chooses to
    randomise.
    """
    _require_torch()
    if not isinstance(dataset, SequenceDataset):
        raise TypeError("dataset must be a SequenceDataset")
    loader_config = config if config is not None else DataLoaderConfig()
    return DataLoader(
        dataset,
        batch_size=loader_config.batch_size,
        shuffle=loader_config.shuffle,
        drop_last=loader_config.drop_last,
        num_workers=loader_config.num_workers,
        pin_memory=loader_config.pin_memory,
        collate_fn=collate_fixed_length_batch,
    )


def _resolve_aligned_target_values(
    feature_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
    target_column: str,
    train_indices: Sequence[int],
) -> tuple[Sequence[object], dict[object, int]]:
    from chronoslob.models.preprocessing import align_feature_label_frames

    aligned = align_feature_label_frames(feature_frame, label_frame)
    if target_column not in aligned.columns:
        raise ValueError(
            f"target column {target_column!r} is missing from aligned frame"
        )
    if not train_indices:
        raise ValueError("train indices must be non-empty to infer class mapping")
    n_rows = len(aligned)
    for index in train_indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("train indices must contain integers")
        if index < 0 or index >= n_rows:
            raise IndexError(
                f"train index {index} is out of range [0, {n_rows})"
            )
    train_values = aligned.iloc[list(train_indices)].loc[:, target_column].tolist()
    _, mapping = encode_target_values(train_values)
    return train_values, mapping


def build_dataloaders_for_split(
    feature_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
    split: SplitIndices,
    sequence_config: SequenceWindowConfig,
    loader_config: DataLoaderConfig | None = None,
    *,
    require_split_contained_windows: bool = True,
    class_to_index: Mapping[object, int] | None = None,
) -> dict[str, Any]:
    """Build train/validation/test :class:`DataLoader` objects.

    The train class-to-index mapping is inferred from training rows only
    and shared across validation and test loaders, so unseen
    validation/test classes raise rather than being silently re-mapped.
    Windows for each split are constrained to ``allowed_target_indices``
    drawn from that split, and ``require_split_contained_windows`` forces
    the full window to remain inside the same partition by default.
    """
    _require_torch()
    if not isinstance(feature_frame, pd.DataFrame):
        raise TypeError("feature_frame must be a pandas DataFrame")
    if not isinstance(label_frame, pd.DataFrame):
        raise TypeError("label_frame must be a pandas DataFrame")
    if not isinstance(split, SplitIndices):
        raise TypeError("split must be a SplitIndices instance")
    if not isinstance(sequence_config, SequenceWindowConfig):
        raise TypeError("sequence_config must be a SequenceWindowConfig instance")

    sequence_config_with_contiguous = (
        sequence_config
        if sequence_config.require_contiguous_indices == require_split_contained_windows
        else SequenceWindowConfig(
            lookback=sequence_config.lookback,
            target_column=sequence_config.target_column,
            feature_columns=sequence_config.feature_columns,
            stride=sequence_config.stride,
            drop_incomplete=sequence_config.drop_incomplete,
            require_contiguous_indices=require_split_contained_windows,
        )
    )

    if class_to_index is None:
        _, train_mapping = _resolve_aligned_target_values(
            feature_frame,
            label_frame,
            sequence_config.target_column,
            split.train,
        )
    else:
        train_mapping = dict(class_to_index)

    loaders: dict[str, Any] = {}

    def _build(partition_name: str, allowed: Sequence[int]) -> SequenceDataset | None:
        if not allowed:
            return None
        try:
            dataset = SequenceDataset(
                feature_frame=feature_frame,
                label_frame=label_frame,
                config=sequence_config_with_contiguous,
                feature_columns=sequence_config.feature_columns,
                class_to_index=train_mapping,
                allowed_target_indices=list(allowed),
            )
        except ValueError as exc:
            raise ValueError(
                f"failed to build dataset for {partition_name!r}: {exc}"
            ) from exc
        return dataset

    train_dataset = _build("train", split.train)
    if train_dataset is None:
        raise ValueError("training split contains no rows")
    if len(train_dataset) == 0:
        raise ValueError(
            "training split has no usable sequence samples after lookback filtering"
        )
    loaders["train"] = create_sequence_dataloader(train_dataset, loader_config)

    validation_dataset = _build("validation", split.validation)
    if validation_dataset is None:
        raise ValueError("validation split contains no rows")
    if len(validation_dataset) == 0:
        raise ValueError(
            "validation split has no usable sequence samples after lookback filtering"
        )
    validation_loader_config = (
        loader_config
        if loader_config is None
        else DataLoaderConfig(
            batch_size=loader_config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=loader_config.num_workers,
            pin_memory=loader_config.pin_memory,
        )
    )
    loaders["validation"] = create_sequence_dataloader(
        validation_dataset,
        validation_loader_config,
    )

    if split.test:
        test_dataset = _build("test", split.test)
        if test_dataset is not None and len(test_dataset) > 0:
            test_loader_config = (
                loader_config
                if loader_config is None
                else DataLoaderConfig(
                    batch_size=loader_config.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=loader_config.num_workers,
                    pin_memory=loader_config.pin_memory,
                )
            )
            loaders["test"] = create_sequence_dataloader(
                test_dataset,
                test_loader_config,
            )
    return loaders
