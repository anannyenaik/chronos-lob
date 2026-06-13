"""PyTorch sequence-window datasets and train-only tensor standardisation.

This module supplies the data layer required by future supervised sequence
models such as DeepLOB-style baselines. It is intentionally narrow: it does
not implement model architectures, training loops or self-supervised
objectives. The indexing rules below enforce that the feature window ending
at row ``t`` may only contain rows ``<= t`` and that the target label for
that sample belongs to timestamp ``t``.

PyTorch is treated as an optional dependency. Public Dataset classes import
``torch`` directly and will raise a clear ``ImportError`` if the package is
not installed. Tests that exercise tensors use ``pytest.importorskip``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from chronoslob.models.preprocessing import (
    align_feature_label_frames,
    build_feature_matrix,
    build_target_vector,
    select_feature_columns,
)

try:  # pragma: no cover - exercised when torch is unavailable
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    torch = None  # type: ignore[assignment]
    _TorchDataset = object  # type: ignore[assignment,misc]
    _TORCH_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    import torch as _torch_typing  # noqa: F401

__all__ = [
    "SequenceDataset",
    "SequenceSampleIndex",
    "SequenceWindowConfig",
    "TorchSequenceStandardiser",
    "build_sequence_indices",
    "encode_target_values",
    "infer_torch_feature_columns",
    "torch_is_available",
]


def torch_is_available() -> bool:
    """Return ``True`` if PyTorch is importable in the current environment."""
    return _TORCH_AVAILABLE


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for this functionality. Install the 'torch' "
            "optional dependency: pip install -e '.[torch]'"
        )
    return torch


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class SequenceSampleIndex:
    """Indices for one supervised sequence sample.

    ``window_start`` and ``window_end`` are inclusive row indices into the
    aligned feature/label frame. ``target_index`` is the row supplying the
    label. For past-only supervised windows ``window_end`` must equal
    ``target_index`` so no future rows can leak into the feature window.
    """

    window_start: int
    window_end: int
    target_index: int

    def __post_init__(self) -> None:
        """Validate non-negativity and the past-only window invariant."""
        _validate_non_negative_int(self.window_start, name="window_start")
        _validate_non_negative_int(self.window_end, name="window_end")
        _validate_non_negative_int(self.target_index, name="target_index")
        if self.window_start > self.window_end:
            raise ValueError("window_start must be <= window_end")
        if self.window_end != self.target_index:
            raise ValueError(
                "window_end must equal target_index for past-only supervised windows"
            )

    @property
    def window_slice(self) -> slice:
        """Return a Python ``slice`` covering the inclusive window."""
        return slice(self.window_start, self.window_end + 1)

    @property
    def lookback(self) -> int:
        """Number of rows in the window (inclusive)."""
        return self.window_end - self.window_start + 1


@dataclass(frozen=True)
class SequenceWindowConfig:
    """Configuration for sequence-window indexing.

    ``lookback`` is the number of rows in each window. ``target_column`` is
    the column in the label frame that supplies ``y``. ``feature_columns``
    optionally pins the model features. ``stride`` controls the step
    between candidate target indices. ``drop_incomplete`` keeps the
    indexer from emitting samples whose window starts before row zero.
    ``require_contiguous_indices`` forces every row in a sample's window to
    lie inside the ``allowed_target_indices`` set when one is supplied;
    this prevents a validation sample from pulling training rows into its
    window unless the caller has explicitly authorised it.
    """

    lookback: int
    target_column: str
    feature_columns: list[str] | None = None
    stride: int = 1
    drop_incomplete: bool = True
    require_contiguous_indices: bool = True

    def __post_init__(self) -> None:
        """Validate window size, target column and stride."""
        _validate_positive_int(self.lookback, name="lookback")
        _validate_positive_int(self.stride, name="stride")
        if not isinstance(self.target_column, str) or not self.target_column.strip():
            raise ValueError("target_column must be a non-empty string")
        if self.feature_columns is not None:
            if not isinstance(self.feature_columns, list):
                raise TypeError("feature_columns must be a list of strings")
            for position, column in enumerate(self.feature_columns):
                if not isinstance(column, str) or not column.strip():
                    raise ValueError(
                        f"feature_columns[{position}] must be a non-empty string"
                    )


def build_sequence_indices(
    n_rows: int,
    config: SequenceWindowConfig,
    *,
    allowed_target_indices: Sequence[int] | None = None,
) -> list[SequenceSampleIndex]:
    """Build deterministic past-only sequence sample indices.

    A sample with ``target_index = t`` uses feature rows
    ``[t - lookback + 1, ..., t]``. The first valid target is
    ``lookback - 1``. ``stride`` is applied across candidate target indices.

    If ``allowed_target_indices`` is supplied, only target indices in that
    set are emitted. When ``require_contiguous_indices`` is true, the
    entire window must also lie inside that set; this guards against
    validation/test windows pulling training rows by accident.
    """
    rows = _validate_non_negative_int(n_rows, name="n_rows")
    if rows == 0:
        return []
    if config.lookback > rows:
        if config.drop_incomplete:
            return []
        raise ValueError(
            f"lookback={config.lookback} is larger than n_rows={rows}"
        )

    if allowed_target_indices is None:
        candidates = range(config.lookback - 1, rows)
        allowed_set: frozenset[int] | None = None
    else:
        cleaned: list[int] = []
        for position, index in enumerate(allowed_target_indices):
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError(
                    f"allowed_target_indices[{position}] must be an integer"
                )
            if index < 0 or index >= rows:
                raise IndexError(
                    f"allowed_target_indices[{position}]={index} is out of range "
                    f"[0, {rows})"
                )
            cleaned.append(index)
        allowed_set = frozenset(cleaned)
        candidates = sorted(allowed_set)  # type: ignore[assignment]

    samples: list[SequenceSampleIndex] = []
    last_emitted: int | None = None
    first_valid = config.lookback - 1
    for target_index in candidates:
        if target_index < first_valid:
            continue
        if last_emitted is not None and (target_index - last_emitted) < config.stride:
            continue
        window_start = target_index - config.lookback + 1
        if window_start < 0:
            continue
        if allowed_set is not None and config.require_contiguous_indices:
            window_indices = range(window_start, target_index + 1)
            if any(index not in allowed_set for index in window_indices):
                continue
        samples.append(
            SequenceSampleIndex(
                window_start=window_start,
                window_end=target_index,
                target_index=target_index,
            )
        )
        last_emitted = target_index
    return samples


def infer_torch_feature_columns(
    frame: pd.DataFrame,
    explicit_columns: list[str] | None = None,
) -> list[str]:
    """Resolve the model feature columns for a sequence dataset.

    When ``explicit_columns`` is supplied it is validated against the frame
    and rejected if any column looks label-like. Otherwise
    :func:`chronoslob.models.preprocessing.select_feature_columns` chooses
    numeric, non-metadata columns and rejects label-like names.
    """
    if explicit_columns is None:
        return select_feature_columns(frame)

    if not isinstance(explicit_columns, list):
        raise TypeError("explicit_columns must be a list of column names")
    missing = [column for column in explicit_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"explicit feature columns are missing from frame: {missing}")
    # Reuse select_feature_columns to enforce the label-like rejection rule.
    select_feature_columns(
        frame.loc[:, explicit_columns],
        reject_label_like=True,
    )
    return list(explicit_columns)


def _to_hashable_class(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    return value


def _sort_class_keys(values: Sequence[object]) -> list[object]:
    cleaned = [_to_hashable_class(value) for value in values]
    unique: list[object] = []
    seen: set[object] = set()
    for value in cleaned:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    try:
        unique.sort()
    except TypeError:
        unique.sort(key=lambda item: (type(item).__name__, repr(item)))
    return unique


def encode_target_values(
    values: Sequence[object] | np.ndarray | pd.Series,
    *,
    class_to_index: Mapping[object, int] | None = None,
) -> tuple[np.ndarray, dict[object, int]]:
    """Encode classification labels into integer class indices.

    If ``class_to_index`` is supplied it is used as-is; values not present in
    that mapping raise ``ValueError`` rather than being silently re-mapped.
    Otherwise a deterministic sorted mapping is inferred. Returns a tuple
    of ``(integer_array, mapping)``.
    """
    if not isinstance(values, (Sequence, np.ndarray, pd.Series)):
        raise TypeError("values must be a sequence")

    raw = list(values)
    cleaned_values = [_to_hashable_class(value) for value in raw]

    if class_to_index is None:
        sorted_classes = _sort_class_keys(cleaned_values)
        mapping: dict[object, int] = {
            class_value: index for index, class_value in enumerate(sorted_classes)
        }
    else:
        mapping = {
            _to_hashable_class(class_value): int(index)
            for class_value, index in class_to_index.items()
        }
        for index_value in mapping.values():
            if index_value < 0:
                raise ValueError("class_to_index values must be non-negative integers")

    encoded = np.empty(len(cleaned_values), dtype=np.int64)
    for position, value in enumerate(cleaned_values):
        if value not in mapping:
            raise ValueError(
                f"value {value!r} at position {position} is not in class_to_index"
            )
        encoded[position] = mapping[value]
    return encoded, dict(mapping)


class SequenceDataset(_TorchDataset):
    """Past-only supervised sequence dataset for LOB/feature frames.

    The dataset aligns ``feature_frame`` and ``label_frame`` via
    :func:`chronoslob.models.preprocessing.align_feature_label_frames`,
    resolves feature columns, builds the feature matrix and target vector
    using the existing preprocessing helpers and finally constructs
    sequence-window samples. ``__getitem__`` returns a mapping with the
    feature window ``x`` of shape ``[lookback, n_features]``, the integer
    target ``y`` and the inclusive window indices.
    """

    def __init__(
        self,
        feature_frame: pd.DataFrame,
        label_frame: pd.DataFrame,
        config: SequenceWindowConfig,
        *,
        sample_indices: Sequence[SequenceSampleIndex] | None = None,
        feature_columns: list[str] | None = None,
        class_to_index: Mapping[object, int] | None = None,
        dtype: Any = None,
        allowed_target_indices: Sequence[int] | None = None,
    ) -> None:
        """Build a sequence dataset from aligned feature and label frames."""
        torch_module = _require_torch()
        if not isinstance(feature_frame, pd.DataFrame):
            raise TypeError("feature_frame must be a pandas DataFrame")
        if not isinstance(label_frame, pd.DataFrame):
            raise TypeError("label_frame must be a pandas DataFrame")
        if dtype is None:
            dtype = torch_module.float32

        explicit_columns = (
            config.feature_columns if feature_columns is None else list(feature_columns)
        )
        resolved_columns = infer_torch_feature_columns(feature_frame, explicit_columns)
        if not resolved_columns:
            raise ValueError("sequence dataset requires at least one feature column")

        aligned = align_feature_label_frames(feature_frame, label_frame)
        if aligned.empty:
            raise ValueError("feature and label frames have no aligned rows")

        feature_matrix = build_feature_matrix(
            aligned,
            feature_columns=resolved_columns,
        )
        target_vector = build_target_vector(
            aligned,
            target_column=config.target_column,
        )

        if sample_indices is None:
            indices = build_sequence_indices(
                n_rows=feature_matrix.x.shape[0],
                config=config,
                allowed_target_indices=allowed_target_indices,
            )
        else:
            indices = list(sample_indices)
            for sample in indices:
                if not isinstance(sample, SequenceSampleIndex):
                    raise TypeError(
                        "sample_indices must contain SequenceSampleIndex objects"
                    )
                if sample.window_end >= feature_matrix.x.shape[0]:
                    raise IndexError(
                        f"sample window_end={sample.window_end} exceeds aligned "
                        f"row count {feature_matrix.x.shape[0]}"
                    )

        if class_to_index is None:
            encoded_targets, mapping = encode_target_values(
                target_vector.y.tolist()
            )
        else:
            sampled_target_rows = sorted({sample.target_index for sample in indices})
            sampled_values = [target_vector.y[index] for index in sampled_target_rows]
            _, mapping = encode_target_values(
                sampled_values,
                class_to_index=class_to_index,
            )
            encoded_targets = np.full(target_vector.y.shape[0], -1, dtype=np.int64)
            for row_index, raw_value in zip(
                sampled_target_rows, sampled_values, strict=True
            ):
                encoded_targets[row_index] = mapping[_to_hashable_class(raw_value)]

        self._aligned_frame = aligned
        self._feature_matrix = feature_matrix.x
        self._encoded_targets = encoded_targets
        self._sample_indices = indices
        self._feature_columns = list(resolved_columns)
        self._class_to_index = mapping
        self._index_to_class = {index: cls for cls, index in mapping.items()}
        self._dtype = dtype
        self._config = config

    def __len__(self) -> int:
        """Number of past-only supervised sequence samples."""
        return len(self._sample_indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        """Return one sample as a dictionary of tensors and indices."""
        torch_module = _require_torch()
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError("sequence dataset indices must be integers")
        if item < 0:
            item = len(self) + item
        if item < 0 or item >= len(self):
            raise IndexError("sequence dataset index out of range")

        sample = self._sample_indices[item]
        window = self._feature_matrix[sample.window_slice, :]
        if window.shape[0] != self._config.lookback:
            raise RuntimeError(
                "internal error: window has unexpected length "
                f"{window.shape[0]} != lookback {self._config.lookback}"
            )
        x_tensor = torch_module.from_numpy(np.asarray(window, dtype=np.float64)).to(
            self._dtype
        )
        y_value = int(self._encoded_targets[sample.target_index])
        y_tensor = torch_module.tensor(y_value, dtype=torch_module.long)
        return {
            "x": x_tensor,
            "y": y_tensor,
            "target_index": int(sample.target_index),
            "window_start": int(sample.window_start),
            "window_end": int(sample.window_end),
        }

    @property
    def feature_columns(self) -> list[str]:
        """List of feature column names backing the ``x`` tensor."""
        return list(self._feature_columns)

    @property
    def class_to_index(self) -> dict[object, int]:
        """Deterministic class-value to integer class-index mapping."""
        return dict(self._class_to_index)

    @property
    def index_to_class(self) -> dict[int, object]:
        """Inverse mapping from integer class index back to class value."""
        return dict(self._index_to_class)

    @property
    def n_features(self) -> int:
        """Number of feature columns in each window."""
        return len(self._feature_columns)

    @property
    def n_classes(self) -> int:
        """Number of distinct classes covered by ``class_to_index``."""
        return len(self._class_to_index)

    @property
    def sample_indices(self) -> list[SequenceSampleIndex]:
        """Return a defensive copy of the sequence sample indices."""
        return list(self._sample_indices)

    @property
    def aligned_frame(self) -> pd.DataFrame:
        """Aligned feature/label frame underlying the dataset."""
        return self._aligned_frame


class TorchSequenceStandardiser:
    """Train-only mean/std standardiser for sequence feature tensors.

    Statistics are fitted from training data supplied explicitly by the
    caller, either a pandas feature frame or a stacked sequence tensor.
    The transform always operates on a copy. ``fit`` must be called before
    ``transform``; calling ``transform`` first raises ``ValueError``. Zero
    standard deviations are replaced with ``1.0`` to avoid division by zero
    so downstream models still see numerically stable inputs.
    """

    def __init__(self) -> None:
        self._mean: Any = None
        self._std: Any = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Return whether the standardiser has been fitted."""
        return self._is_fitted

    @property
    def mean(self) -> Any:
        """Return a copy of the fitted feature-wise mean tensor."""
        self._raise_if_unfitted()
        return self._mean.clone()

    @property
    def std(self) -> Any:
        """Return a copy of the fitted feature-wise standard deviation."""
        self._raise_if_unfitted()
        return self._std.clone()

    def _raise_if_unfitted(self) -> None:
        if not self._is_fitted:
            raise ValueError(
                "TorchSequenceStandardiser must be fitted before transform"
            )

    @staticmethod
    def _validate_finite(tensor: Any, *, name: str) -> None:
        torch_module = _require_torch()
        if not torch_module.is_tensor(tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if torch_module.isnan(tensor).any().item():
            raise ValueError(f"{name} must not contain NaN values")
        if torch_module.isinf(tensor).any().item():
            raise ValueError(f"{name} must not contain infinite values")

    def fit_from_feature_frame(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
    ) -> TorchSequenceStandardiser:
        """Fit per-feature mean and std from a training feature frame."""
        torch_module = _require_torch()
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        if not isinstance(feature_columns, list) or not feature_columns:
            raise ValueError("feature_columns must be a non-empty list")
        missing = [column for column in feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"frame is missing feature columns: {missing}")
        values = frame.loc[:, feature_columns].to_numpy(dtype=np.float64, copy=True)
        if values.shape[0] == 0:
            raise ValueError("frame must contain at least one row for fitting")
        tensor = torch_module.from_numpy(values).to(torch_module.float64)
        self._validate_finite(tensor, name="values")
        return self._fit_from_2d(tensor)

    def fit_from_sequences(self, x: Any) -> TorchSequenceStandardiser:
        """Fit per-feature mean and std from a stacked training tensor.

        ``x`` may be either a 2D tensor ``[n_samples, n_features]`` or a
        3D tensor ``[n_samples, lookback, n_features]``.
        """
        torch_module = _require_torch()
        self._validate_finite(x, name="x")
        if x.ndim == 3:
            flat = x.reshape(-1, x.shape[-1])
        elif x.ndim == 2:
            flat = x
        else:
            raise ValueError("x must be a 2D or 3D tensor")
        if flat.shape[0] == 0:
            raise ValueError("x must contain at least one row for fitting")
        return self._fit_from_2d(flat.to(torch_module.float64))

    def _fit_from_2d(self, flat: Any) -> TorchSequenceStandardiser:
        torch_module = _require_torch()
        mean = flat.mean(dim=0)
        std = flat.std(dim=0, unbiased=False)
        zero_mask = std == 0
        if zero_mask.any().item():
            replacement = torch_module.ones_like(std)
            std = torch_module.where(zero_mask, replacement, std)
        self._mean = mean.to(torch_module.float64)
        self._std = std.to(torch_module.float64)
        self._is_fitted = True
        return self

    def transform_tensor(self, x: Any) -> Any:
        """Standardise ``x`` using previously fitted statistics."""
        torch_module = _require_torch()
        self._raise_if_unfitted()
        self._validate_finite(x, name="x")
        if x.shape[-1] != self._mean.shape[0]:
            raise ValueError(
                "feature dimension mismatch: expected "
                f"{self._mean.shape[0]} but got {x.shape[-1]}"
            )
        original_dtype = x.dtype
        promoted = x.to(torch_module.float64)
        result = (promoted - self._mean) / self._std
        return result.to(original_dtype)

    def fit_transform_tensor(self, x: Any) -> Any:
        """Fit on ``x`` and return its standardised values."""
        self.fit_from_sequences(x)
        return self.transform_tensor(x)
