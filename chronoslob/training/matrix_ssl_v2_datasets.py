"""Dataset utilities for the market-state-aware matrix SSL-v2 objective.

All fitting helpers in this module are train-only by construction: auxiliary
bucket edges are fitted from rows whose current timestamp and future target
timestamp both lie inside the supplied training partition. Window builders also
require the full input window and the future target row to remain inside one
partition, so pretraining windows cannot cross fold boundaries.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

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
    "DEFAULT_SSL_V2_FEATURE_GROUPS",
    "MatrixSSLV2WindowDataset",
    "MatrixSSLV2WindowSample",
    "SSLV2AuxiliaryLabelSpec",
    "build_ssl_v2_auxiliary_label_matrix",
    "build_ssl_v2_windows",
    "collate_matrix_ssl_v2_windows",
    "fit_ssl_v2_auxiliary_label_spec",
    "infer_ssl_v2_feature_groups",
]

DEFAULT_SSL_V2_FEATURE_GROUPS: tuple[str, ...] = (
    "price_depth",
    "imbalance",
    "spread_microprice",
    "temporal_context",
)


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for matrix SSL-v2 datasets. Install the 'torch' "
            "optional dependency: pip install -e '.[torch]'"
        )
    return torch


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_index_sequence(
    indices: Sequence[int],
    *,
    n_rows: int,
    name: str,
) -> list[int]:
    cleaned: list[int] = []
    for position, index in enumerate(indices):
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"{name}[{position}] must be an integer")
        if index < 0 or index >= n_rows:
            raise IndexError(f"{name}[{position}]={index} is out of range [0, {n_rows})")
        cleaned.append(index)
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{name} must not contain duplicates")
    return sorted(cleaned)


def _as_matrix(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ValueError("matrix must be 2D [n_rows, n_features]")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("matrix must contain at least one row and column")
    if not np.isfinite(array).all():
        raise ValueError("matrix must contain only finite values")
    return array


def _quantiles(bucket_count: int) -> tuple[float, ...]:
    _validate_positive_int(bucket_count, name="bucket_count")
    if bucket_count < 2:
        raise ValueError("bucket_count must be >= 2")
    return tuple(index / float(bucket_count) for index in range(1, bucket_count))


def _bucketise(values: np.ndarray, edges: Sequence[float], *, bucket_count: int) -> np.ndarray:
    binner = TrainOnlyQuantileBinner(quantiles=_quantiles(bucket_count))
    binner.bin_edges_ = [float(edge) for edge in edges]
    assigned = binner.transform([float(value) for value in values.tolist()])
    return np.clip(np.asarray(assigned, dtype=np.int64), 0, bucket_count - 1)


@dataclass(frozen=True)
class SSLV2AuxiliaryLabelSpec:
    """Train-fitted auxiliary label configuration for SSL-v2."""

    future_offset: int
    bucket_count: int
    volatility_edges: tuple[float, ...]
    return_edges: tuple[float, ...]
    imbalance_edges: tuple[float, ...]
    train_current_indices: tuple[int, ...]
    train_future_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        _validate_positive_int(self.future_offset, name="future_offset")
        _validate_positive_int(self.bucket_count, name="bucket_count")
        if self.bucket_count < 2:
            raise ValueError("bucket_count must be >= 2")
        expected_edges = self.bucket_count - 1
        for name in ("volatility_edges", "return_edges", "imbalance_edges"):
            edges = tuple(float(value) for value in getattr(self, name))
            if len(edges) != expected_edges:
                raise ValueError(
                    f"{name} must contain {expected_edges} train-fitted edges"
                )
            if not all(math.isfinite(edge) for edge in edges):
                raise ValueError(f"{name} must contain finite edges")


@dataclass(frozen=True)
class MatrixSSLV2WindowSample:
    """Window and future-target indices for one SSL-v2 sample."""

    window_start: int
    window_end: int
    future_index: int

    def __post_init__(self) -> None:
        for field_name in ("window_start", "window_end", "future_index"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.window_end < self.window_start:
            raise ValueError("window_end must be >= window_start")
        if self.future_index <= self.window_end:
            raise ValueError("future_index must be strictly after window_end")


def infer_ssl_v2_feature_groups(feature_columns: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Infer coherent SSL-v2 masking groups from feature column names."""
    if not feature_columns:
        raise ValueError("feature_columns must not be empty")
    groups: dict[str, list[int]] = {name: [] for name in DEFAULT_SSL_V2_FEATURE_GROUPS}
    assigned: set[int] = set()
    for index, raw_name in enumerate(feature_columns):
        name = str(raw_name).strip().lower()
        if not name:
            raise ValueError(f"feature_columns[{index}] must be non-empty")
        if any(token in name for token in ("spread", "microprice", "midprice")):
            groups["spread_microprice"].append(index)
            assigned.add(index)
        elif any(token in name for token in ("imbalance", "ofi")):
            groups["imbalance"].append(index)
            assigned.add(index)
        elif (
            name.startswith(("bid_", "ask_"))
            or "price" in name
            or "quantity" in name
            or "qty" in name
            or "depth" in name
        ):
            groups["price_depth"].append(index)
            assigned.add(index)
    for index in range(len(feature_columns)):
        if index not in assigned:
            groups["temporal_context"].append(index)
    return {
        group_name: tuple(indices)
        for group_name, indices in groups.items()
        if indices
    }


def build_ssl_v2_windows(
    *,
    n_rows: int,
    window_length: int,
    allowed_indices: Sequence[int],
    future_offset: int,
) -> list[MatrixSSLV2WindowSample]:
    """Build windows whose input rows and future target row share a partition."""
    _validate_positive_int(window_length, name="window_length")
    _validate_positive_int(future_offset, name="future_offset")
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows <= 0:
        raise ValueError("n_rows must be a positive integer")
    allowed = set(
        _validate_index_sequence(allowed_indices, n_rows=n_rows, name="allowed_indices")
    )
    windows: list[MatrixSSLV2WindowSample] = []
    for end in sorted(allowed):
        start = end - window_length + 1
        future_index = end + future_offset
        if start < 0 or future_index >= n_rows or future_index not in allowed:
            continue
        if all((start + offset) in allowed for offset in range(window_length)):
            windows.append(
                MatrixSSLV2WindowSample(
                    window_start=start,
                    window_end=end,
                    future_index=future_index,
                )
            )
    return windows


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        column
        for column in frame.columns
        if str(column).lower() == "split" or str(column).lower().startswith("label")
    }
    return [
        str(column)
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _column_values(frame: pd.DataFrame, candidates: Sequence[str], fallback: int) -> np.ndarray:
    lower_to_column = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        column = lower_to_column.get(candidate.lower())
        if column is not None:
            return frame[column].to_numpy(dtype=float, copy=True)
    numeric = _numeric_columns(frame)
    if fallback >= len(numeric):
        raise ValueError(
            "SSL-v2 auxiliary labels require bid/ask price and quantity columns "
            "or at least four numeric feature columns"
        )
    return frame[numeric[fallback]].to_numpy(dtype=float, copy=True)


def _lob_proxy_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    bid_price = _column_values(frame, ("bid_price_1", "bid_px_1", "bid1_price"), 0)
    bid_quantity = _column_values(frame, ("bid_quantity_1", "bid_qty_1", "bid1_qty"), 1)
    ask_price = _column_values(frame, ("ask_price_1", "ask_px_1", "ask1_price"), 2)
    ask_quantity = _column_values(frame, ("ask_quantity_1", "ask_qty_1", "ask1_qty"), 3)
    mid = (bid_price + ask_price) / 2.0
    spread = ask_price - bid_price
    denominator = bid_quantity + ask_quantity
    imbalance = np.divide(
        bid_quantity - ask_quantity,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=np.abs(denominator) > 1e-12,
    )
    for name, values in {
        "mid": mid,
        "spread": spread,
        "imbalance": imbalance,
    }.items():
        if not np.isfinite(values).all():
            raise ValueError(f"{name} proxy contains non-finite values")
    return {"mid": mid, "spread": spread, "imbalance": imbalance}


def _future_values(
    frame: pd.DataFrame,
    *,
    current_indices: Sequence[int],
    future_offset: int,
) -> dict[str, np.ndarray]:
    proxies = _lob_proxy_arrays(frame)
    current = np.asarray(list(current_indices), dtype=np.int64)
    future = current + int(future_offset)
    if len(current) == 0:
        raise ValueError("current_indices must not be empty")
    if int(future.max()) >= len(frame):
        raise IndexError("future index exceeds frame length")
    future_return = proxies["mid"][future] - proxies["mid"][current]
    abs_return = np.abs(future_return)
    volatility = abs_return.copy()
    if future_offset > 1:
        realised: list[float] = []
        for index in current.tolist():
            path = proxies["mid"][index + 1 : index + future_offset + 1]
            diffs = np.diff(np.concatenate(([proxies["mid"][index]], path)))
            realised.append(float(np.std(diffs)) if len(diffs) > 1 else abs(float(diffs[0])))
        volatility = np.asarray(realised, dtype=float)
    return {
        "spread_widening": (
            proxies["spread"][future] > proxies["spread"][current]
        ).astype(np.int64),
        "volatility": volatility,
        "return": future_return,
        "imbalance": proxies["imbalance"][future],
    }


def fit_ssl_v2_auxiliary_label_spec(
    frame: pd.DataFrame,
    *,
    train_indices: Sequence[int],
    future_offset: int,
    bucket_count: int = 3,
) -> SSLV2AuxiliaryLabelSpec:
    """Fit SSL-v2 auxiliary bucket edges on training rows only."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("frame must not be empty")
    _validate_positive_int(future_offset, name="future_offset")
    _validate_positive_int(bucket_count, name="bucket_count")
    train = _validate_index_sequence(train_indices, n_rows=len(frame), name="train_indices")
    train_set = set(train)
    current = [
        index
        for index in train
        if index + future_offset < len(frame) and index + future_offset in train_set
    ]
    if not current:
        raise ValueError(
            "no train-only current/future pairs are available for SSL-v2 labels"
        )
    values = _future_values(frame, current_indices=current, future_offset=future_offset)
    quantiles = _quantiles(bucket_count)

    def _fit_edges(name: str) -> tuple[float, ...]:
        binner = TrainOnlyQuantileBinner(quantiles=quantiles)
        binner.fit([float(value) for value in values[name].tolist()])
        return tuple(float(edge) for edge in binner.bin_edges_ or [])

    return SSLV2AuxiliaryLabelSpec(
        future_offset=future_offset,
        bucket_count=bucket_count,
        volatility_edges=_fit_edges("volatility"),
        return_edges=_fit_edges("return"),
        imbalance_edges=_fit_edges("imbalance"),
        train_current_indices=tuple(current),
        train_future_indices=tuple(index + future_offset for index in current),
    )


def build_ssl_v2_auxiliary_label_matrix(
    frame: pd.DataFrame,
    spec: SSLV2AuxiliaryLabelSpec,
    *,
    ignore_index: int = -100,
) -> dict[str, np.ndarray]:
    """Build per-row future-state labels using train-fitted SSL-v2 edges."""
    if not isinstance(spec, SSLV2AuxiliaryLabelSpec):
        raise TypeError("spec must be an SSLV2AuxiliaryLabelSpec")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    n_rows = len(frame)
    valid_current = list(range(0, max(0, n_rows - spec.future_offset)))
    labels = {
        "future_spread_widening": np.full(n_rows, ignore_index, dtype=np.int64),
        "future_volatility": np.full(n_rows, ignore_index, dtype=np.int64),
        "future_return": np.full(n_rows, ignore_index, dtype=np.int64),
        "future_imbalance": np.full(n_rows, ignore_index, dtype=np.int64),
        "contrastive_regime": np.full(n_rows, ignore_index, dtype=np.int64),
    }
    if not valid_current:
        return labels
    values = _future_values(
        frame,
        current_indices=valid_current,
        future_offset=spec.future_offset,
    )
    current = np.asarray(valid_current, dtype=np.int64)
    labels["future_spread_widening"][current] = values["spread_widening"]
    labels["future_volatility"][current] = _bucketise(
        values["volatility"],
        spec.volatility_edges,
        bucket_count=spec.bucket_count,
    )
    labels["future_return"][current] = _bucketise(
        values["return"],
        spec.return_edges,
        bucket_count=spec.bucket_count,
    )
    labels["future_imbalance"][current] = _bucketise(
        values["imbalance"],
        spec.imbalance_edges,
        bucket_count=spec.bucket_count,
    )
    labels["contrastive_regime"][current] = labels["future_volatility"][current]
    return labels


class MatrixSSLV2WindowDataset(_TorchDataset):
    """Deterministic structured-mask SSL-v2 windows over a standardised matrix."""

    def __init__(
        self,
        matrix: np.ndarray,
        windows: Sequence[MatrixSSLV2WindowSample],
        *,
        feature_groups: Mapping[str, Sequence[int]],
        auxiliary_labels: Mapping[str, np.ndarray],
        mask_probability: float,
        mask_value: float,
        ignore_index: int,
        enable_contrastive: bool,
        base_seed: int,
    ) -> None:
        _require_torch()
        self._matrix = _as_matrix(matrix)
        self._n_rows = int(self._matrix.shape[0])
        self._n_features = int(self._matrix.shape[1])
        if not windows:
            raise ValueError("windows must not be empty")
        self._windows = self._validate_windows(windows)
        self._window_length = (
            self._windows[0].window_end - self._windows[0].window_start + 1
        )
        self._feature_groups = self._validate_feature_groups(feature_groups)
        self._enable_contrastive = bool(enable_contrastive)
        self._auxiliary_labels = self._validate_auxiliary_labels(auxiliary_labels)
        if isinstance(mask_probability, bool) or not isinstance(
            mask_probability, (int, float)
        ):
            raise TypeError("mask_probability must be a float")
        self._mask_probability = float(mask_probability)
        if not 0.0 <= self._mask_probability <= 1.0:
            raise ValueError("mask_probability must satisfy 0 <= p <= 1")
        if self._mask_probability <= 0.0:
            raise ValueError("mask_probability must be positive for SSL-v2 masking")
        self._mask_value = float(mask_value)
        self._ignore_index = int(ignore_index)
        self._base_seed = int(base_seed)

    def _validate_windows(
        self,
        windows: Sequence[MatrixSSLV2WindowSample],
    ) -> list[MatrixSSLV2WindowSample]:
        validated: list[MatrixSSLV2WindowSample] = []
        window_length: int | None = None
        for sample in windows:
            if not isinstance(sample, MatrixSSLV2WindowSample):
                raise TypeError("windows must contain MatrixSSLV2WindowSample objects")
            if sample.future_index >= self._n_rows:
                raise IndexError("future_index exceeds matrix row count")
            length = sample.window_end - sample.window_start + 1
            if window_length is None:
                window_length = length
            elif length != window_length:
                raise ValueError("all SSL-v2 windows must share one length")
            validated.append(sample)
        return validated

    def _validate_feature_groups(
        self,
        feature_groups: Mapping[str, Sequence[int]],
    ) -> dict[str, tuple[int, ...]]:
        if not isinstance(feature_groups, Mapping):
            raise TypeError("feature_groups must be a mapping")
        cleaned: dict[str, tuple[int, ...]] = {}
        seen: set[int] = set()
        for group_name, raw_indices in feature_groups.items():
            indices = []
            for index in raw_indices:
                if isinstance(index, bool) or not isinstance(index, int):
                    raise TypeError(f"feature group {group_name!r} contains non-int index")
                if index < 0 or index >= self._n_features:
                    raise IndexError(
                        f"feature group {group_name!r} index {index} out of range"
                    )
                indices.append(index)
            if indices:
                cleaned[str(group_name)] = tuple(indices)
                seen.update(indices)
        if not cleaned:
            raise ValueError("feature_groups must contain at least one non-empty group")
        if not seen:
            raise ValueError("feature_groups selected no feature channels")
        return cleaned

    def _validate_auxiliary_labels(
        self,
        auxiliary_labels: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        required = {
            "future_spread_widening",
            "future_volatility",
            "future_return",
            "future_imbalance",
        }
        if self._enable_contrastive:
            required.add("contrastive_regime")
        missing = required - set(auxiliary_labels)
        if missing:
            raise ValueError(f"auxiliary_labels missing required keys: {sorted(missing)}")
        labels: dict[str, np.ndarray] = {}
        for key, raw_values in auxiliary_labels.items():
            values = np.asarray(raw_values, dtype=np.int64)
            if values.shape != (self._n_rows,):
                raise ValueError(
                    f"auxiliary label {key!r} shape {values.shape} must be {(self._n_rows,)}"
                )
            labels[str(key)] = values
        return labels

    def __len__(self) -> int:
        return len(self._windows)

    @property
    def window_length(self) -> int:
        """Common inclusive window length."""
        return self._window_length

    @property
    def n_features(self) -> int:
        """Number of feature channels."""
        return self._n_features

    def _window_values(self, sample: MatrixSSLV2WindowSample) -> np.ndarray:
        return self._matrix[sample.window_start : sample.window_end + 1, :].astype(
            np.float64,
            copy=True,
        )

    def _build_structured_mask(self, item: int) -> np.ndarray:
        rng = np.random.default_rng(self._base_seed + int(item))
        mask = np.zeros((self._window_length, self._n_features), dtype=bool)
        group_names = list(self._feature_groups)
        selected = [
            group_name
            for group_name in group_names
            if rng.random() < self._mask_probability
        ]
        if not selected:
            selected = [group_names[int(rng.integers(0, len(group_names)))]]
        span = max(1, math.ceil(self._window_length * self._mask_probability))
        for group_name in selected:
            indices = self._feature_groups[group_name]
            start = int(rng.integers(0, self._window_length - span + 1))
            mask[start : start + span, list(indices)] = True
        if not mask.any():
            group_name = group_names[0]
            mask[0, list(self._feature_groups[group_name])] = True
        return mask

    def __getitem__(self, item: int) -> dict[str, Any]:
        torch_module = _require_torch()
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError("dataset indices must be integers")
        if item < 0:
            item = len(self) + item
        if item < 0 or item >= len(self):
            raise IndexError("matrix SSL-v2 dataset index out of range")
        sample = self._windows[item]
        target = self._window_values(sample)
        x_values = target.copy()
        mask = self._build_structured_mask(item)
        x_values[mask] = self._mask_value
        future_labels = {
            key: torch_module.tensor(
                int(values[sample.window_end]),
                dtype=torch_module.long,
            )
            for key, values in self._auxiliary_labels.items()
            if key.startswith("future_")
        }
        record: dict[str, Any] = {
            "x": torch_module.from_numpy(x_values).to(torch_module.float32),
            "masked_target": torch_module.from_numpy(target).to(torch_module.float32),
            "mask": torch_module.from_numpy(mask),
            "future_labels": future_labels,
        }
        if self._enable_contrastive:
            record["contrastive_labels"] = torch_module.tensor(
                int(self._auxiliary_labels["contrastive_regime"][sample.window_end]),
                dtype=torch_module.long,
            )
        return record


def collate_matrix_ssl_v2_windows(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack SSL-v2 window samples into batched tensors."""
    torch_module = _require_torch()
    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch: dict[str, Any] = {
        "x": torch_module.stack([sample["x"] for sample in samples], dim=0),
        "masked_target": torch_module.stack(
            [sample["masked_target"] for sample in samples],
            dim=0,
        ),
        "mask": torch_module.stack([sample["mask"] for sample in samples], dim=0),
    }
    future_keys = sorted(samples[0]["future_labels"])
    batch["future_labels"] = {
        key: torch_module.stack(
            [sample["future_labels"][key] for sample in samples],
            dim=0,
        )
        for key in future_keys
    }
    if all("contrastive_labels" in sample for sample in samples):
        batch["contrastive_labels"] = torch_module.stack(
            [sample["contrastive_labels"] for sample in samples],
            dim=0,
        )
    return batch
