"""Self-supervised window datasets for normalised FI-2010 matrices.

This module turns a train-only standardised feature matrix into masked and
next-field self-supervised samples. All statistics used for masking and
bucketisation are fitted on the training partition only:

* ``fit_feature_bucket_edges`` fits one :class:`TrainOnlyQuantileBinner` per
  feature column on training rows and returns the resulting edges.
* ``bucketise_matrix`` applies those train-fitted edges to every row.
* :class:`MatrixSSLWindowDataset` emits deterministic masked windows together
  with masked-reconstruction targets and next-field bucket labels.

No validation or test rows, scalers, edges or statistics are consulted by any
function here. The caller supplies window index pairs that lie wholly inside a
single partition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from chronoslob.training.splitters import TrainOnlyQuantileBinner

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
    "MatrixSSLWindowDataset",
    "MatrixSSLWindowSample",
    "bucketise_matrix",
    "build_contiguous_windows",
    "collate_matrix_ssl_windows",
    "fit_feature_bucket_edges",
]


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for matrix SSL datasets. Install the 'torch' "
            "optional dependency: pip install -e '.[torch]'"
        )
    return torch


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _as_standardised_matrix(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ValueError("matrix must be 2D [n_rows, n_features]")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("matrix must contain at least one row and column")
    if not np.isfinite(array).all():
        raise ValueError("matrix must contain only finite values")
    return array


def _bucket_quantiles(bucket_count: int) -> tuple[float, ...]:
    _validate_positive_int(bucket_count, name="bucket_count")
    if bucket_count < 2:
        raise ValueError("bucket_count must be >= 2")
    return tuple(
        float(index) / float(bucket_count) for index in range(1, bucket_count)
    )


def fit_feature_bucket_edges(
    matrix: np.ndarray,
    *,
    train_indices: Sequence[int],
    bucket_count: int,
) -> list[list[float]]:
    """Fit per-feature quantile bucket edges on training rows only.

    Returns one edge list per feature column. The edges are derived purely
    from rows in ``train_indices`` so no validation or test statistics leak
    into the next-field objective.
    """
    array = _as_standardised_matrix(matrix)
    quantiles = _bucket_quantiles(bucket_count)
    rows: list[int] = []
    for position, index in enumerate(train_indices):
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"train_indices[{position}] must be an integer")
        if index < 0 or index >= array.shape[0]:
            raise IndexError(
                f"train_indices[{position}]={index} is out of range "
                f"[0, {array.shape[0]})"
            )
        rows.append(index)
    if not rows:
        raise ValueError("train_indices must not be empty")

    train_values = array[np.asarray(rows, dtype=int), :]
    edges: list[list[float]] = []
    for column in range(array.shape[1]):
        binner = TrainOnlyQuantileBinner(quantiles=quantiles)
        binner.fit([float(value) for value in train_values[:, column].tolist()])
        edges.append(list(binner.bin_edges_ or []))
    return edges


def bucketise_matrix(
    matrix: np.ndarray,
    *,
    feature_edges: Sequence[Sequence[float]],
    bucket_count: int,
) -> np.ndarray:
    """Assign train-fitted bucket indices to every entry of ``matrix``.

    Returns an integer array ``[n_rows, n_features]`` with values in
    ``[0, bucket_count)``.
    """
    array = _as_standardised_matrix(matrix)
    _validate_positive_int(bucket_count, name="bucket_count")
    if len(feature_edges) != array.shape[1]:
        raise ValueError(
            "feature_edges length must match the feature dimension: "
            f"{len(feature_edges)} != {array.shape[1]}"
        )
    quantiles = _bucket_quantiles(bucket_count)
    buckets = np.zeros(array.shape, dtype=np.int64)
    for column, edges in enumerate(feature_edges):
        binner = TrainOnlyQuantileBinner(quantiles=quantiles)
        binner.bin_edges_ = [float(edge) for edge in edges]
        assigned = binner.transform(
            [float(value) for value in array[:, column].tolist()]
        )
        clipped = np.clip(np.asarray(assigned, dtype=np.int64), 0, bucket_count - 1)
        buckets[:, column] = clipped
    return buckets


@dataclass(frozen=True)
class MatrixSSLWindowSample:
    """Inclusive window bounds into the standardised matrix."""

    window_start: int
    window_end: int

    def __post_init__(self) -> None:
        if isinstance(self.window_start, bool) or not isinstance(
            self.window_start, int
        ):
            raise TypeError("window_start must be an integer")
        if isinstance(self.window_end, bool) or not isinstance(
            self.window_end, int
        ):
            raise TypeError("window_end must be an integer")
        if self.window_start < 0:
            raise ValueError("window_start must be non-negative")
        if self.window_end < self.window_start:
            raise ValueError("window_end must be >= window_start")


def build_contiguous_windows(
    *,
    n_rows: int,
    window_length: int,
    allowed_indices: Sequence[int],
) -> list[MatrixSSLWindowSample]:
    """Build fixed-length windows that lie wholly inside ``allowed_indices``.

    A window ends at row ``end`` (which must be in ``allowed_indices``) and
    spans ``[end - window_length + 1, end]``. Every row in the span must also
    be allowed, so windows never cross a partition boundary. This guards the
    pretraining data path against pulling validation or test rows into a
    training window.
    """
    _validate_positive_int(window_length, name="window_length")
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows <= 0:
        raise ValueError("n_rows must be a positive integer")
    allowed: set[int] = set()
    for position, index in enumerate(allowed_indices):
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"allowed_indices[{position}] must be an integer")
        if index < 0 or index >= n_rows:
            raise IndexError(
                f"allowed_indices[{position}]={index} is out of range [0, {n_rows})"
            )
        allowed.add(index)
    windows: list[MatrixSSLWindowSample] = []
    for end in sorted(allowed):
        start = end - window_length + 1
        if start < 0:
            continue
        if all((start + offset) in allowed for offset in range(window_length)):
            windows.append(
                MatrixSSLWindowSample(window_start=start, window_end=end)
            )
    return windows


class MatrixSSLWindowDataset(_TorchDataset):
    """Deterministic masked/next-field SSL windows over a standardised matrix.

    Each item is one fixed-length window of the standardised feature matrix.
    Masking is deterministic per item (seeded from ``base_seed`` plus the item
    index) so re-iterating the dataset yields identical masks, which keeps the
    pretraining loop reproducible without consuming the global RNG state.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        windows: Sequence[MatrixSSLWindowSample],
        *,
        bucket_matrix: np.ndarray | None,
        mask_probability: float,
        mask_value: float,
        bucket_count: int,
        ignore_index: int,
        enable_masked_field: bool,
        enable_next_field: bool,
        base_seed: int,
    ) -> None:
        _require_torch()
        self._matrix = _as_standardised_matrix(matrix)
        self._n_rows = int(self._matrix.shape[0])
        self._n_features = int(self._matrix.shape[1])
        if not windows:
            raise ValueError("windows must not be empty")
        self._windows: list[MatrixSSLWindowSample] = []
        window_length: int | None = None
        for sample in windows:
            if not isinstance(sample, MatrixSSLWindowSample):
                raise TypeError("windows must contain MatrixSSLWindowSample objects")
            if sample.window_end >= self._n_rows:
                raise IndexError(
                    f"window_end={sample.window_end} exceeds matrix row count "
                    f"{self._n_rows}"
                )
            length = sample.window_end - sample.window_start + 1
            if window_length is None:
                window_length = length
            elif length != window_length:
                raise ValueError("all SSL windows must share the same length")
            self._windows.append(sample)
        self._window_length = int(window_length or 0)

        if not (enable_masked_field or enable_next_field):
            raise ValueError("at least one SSL objective must be enabled")
        if enable_next_field:
            if bucket_matrix is None:
                raise ValueError(
                    "bucket_matrix is required when the next-field objective is "
                    "enabled"
                )
            bucket_array = np.asarray(bucket_matrix, dtype=np.int64)
            if bucket_array.shape != self._matrix.shape:
                raise ValueError(
                    "bucket_matrix shape must match the standardised matrix"
                )
            self._bucket_matrix: np.ndarray | None = bucket_array
        else:
            self._bucket_matrix = None

        self._mask_probability = float(mask_probability)
        self._mask_value = float(mask_value)
        self._bucket_count = _validate_positive_int(bucket_count, name="bucket_count")
        self._ignore_index = int(ignore_index)
        self._enable_masked_field = bool(enable_masked_field)
        self._enable_next_field = bool(enable_next_field)
        self._base_seed = int(base_seed)

    def __len__(self) -> int:
        return len(self._windows)

    @property
    def window_length(self) -> int:
        """Common inclusive window length of every sample."""
        return self._window_length

    @property
    def n_features(self) -> int:
        """Number of feature channels per window position."""
        return self._n_features

    def _window_values(self, sample: MatrixSSLWindowSample) -> np.ndarray:
        return self._matrix[sample.window_start : sample.window_end + 1, :].astype(
            np.float64, copy=True
        )

    def _build_mask(self, item: int) -> np.ndarray:
        rng = np.random.default_rng(self._base_seed + int(item))
        mask = rng.random((self._window_length, self._n_features)) < (
            self._mask_probability
        )
        if not mask.any():
            # Force at least one masked entry so the reconstruction loss is
            # always defined for this window.
            flat_index = int(
                rng.integers(0, self._window_length * self._n_features)
            )
            row = flat_index // self._n_features
            col = flat_index % self._n_features
            mask[row, col] = True
        return mask

    def _next_labels(self, sample: MatrixSSLWindowSample) -> np.ndarray:
        assert self._bucket_matrix is not None
        labels = np.full(
            (self._window_length, self._n_features),
            self._ignore_index,
            dtype=np.int64,
        )
        for offset in range(self._window_length - 1):
            source_row = sample.window_start + offset + 1
            labels[offset, :] = self._bucket_matrix[source_row, :]
        return labels

    def __getitem__(self, item: int) -> dict[str, Any]:
        torch_module = _require_torch()
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError("dataset indices must be integers")
        if item < 0:
            item = len(self) + item
        if item < 0 or item >= len(self):
            raise IndexError("matrix SSL dataset index out of range")

        sample = self._windows[item]
        target = self._window_values(sample)
        x_values = target.copy()

        if self._enable_masked_field and self._mask_probability > 0.0:
            mask = self._build_mask(item)
            x_values[mask] = self._mask_value
        else:
            mask = np.zeros(
                (self._window_length, self._n_features), dtype=bool
            )

        record: dict[str, Any] = {
            "x": torch_module.from_numpy(x_values).to(torch_module.float32),
            "masked_target": torch_module.from_numpy(target).to(
                torch_module.float32
            ),
            "mask": torch_module.from_numpy(mask),
        }
        if self._enable_next_field:
            record["next_bucket_labels"] = torch_module.from_numpy(
                self._next_labels(sample)
            ).to(torch_module.long)
        return record


def collate_matrix_ssl_windows(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Stack masked SSL window samples into batched tensors."""
    torch_module = _require_torch()
    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch: dict[str, Any] = {
        "x": torch_module.stack([sample["x"] for sample in samples], dim=0),
        "masked_target": torch_module.stack(
            [sample["masked_target"] for sample in samples], dim=0
        ),
        "mask": torch_module.stack([sample["mask"] for sample in samples], dim=0),
    }
    if all("next_bucket_labels" in sample for sample in samples):
        batch["next_bucket_labels"] = torch_module.stack(
            [sample["next_bucket_labels"] for sample in samples], dim=0
        )
    return batch
