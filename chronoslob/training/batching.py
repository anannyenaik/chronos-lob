"""Collation helpers for fixed-length and variable-length sequence batches.

The helpers in this module are deliberately small: they validate input
shapes, stack tensors and return mappings that future PyTorch training
loops can consume. They do not move tensors to CUDA, do not cast dtypes
implicitly and do not generate any random state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

try:  # pragma: no cover - exercised when torch is unavailable
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


__all__ = [
    "collate_fixed_length_batch",
    "collate_variable_length_batch",
    "pad_variable_length_sequences",
]


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for batching helpers. Install the 'torch' "
            "optional dependency: pip install -e '.[torch]'"
        )
    return torch


def _stack_index_tensor(values: Sequence[Any], *, name: str) -> Any:
    torch_module = _require_torch()
    return torch_module.tensor([int(value) for value in values], dtype=torch_module.long)


def collate_fixed_length_batch(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Collate fixed-length sequence samples into a single batch.

    Each sample must be a mapping containing tensors ``x`` of shape
    ``[lookback, n_features]`` and scalar ``y`` plus the integer window
    indices. The function returns a dictionary with batched tensors of
    shape ``[batch, lookback, n_features]`` and ``[batch]``.
    """
    torch_module = _require_torch()
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("collate_fixed_length_batch received an empty batch")

    expected_shape: tuple[int, ...] | None = None
    x_tensors: list[Any] = []
    y_values: list[Any] = []
    target_indices: list[int] = []
    window_starts: list[int] = []
    window_ends: list[int] = []
    for position, sample in enumerate(sample_list):
        if not isinstance(sample, Mapping):
            raise TypeError(f"sample at position {position} must be a mapping")
        required = {"x", "y", "target_index", "window_start", "window_end"}
        missing = required - set(sample.keys())
        if missing:
            raise KeyError(
                f"sample at position {position} is missing keys: {sorted(missing)}"
            )
        x_value = sample["x"]
        if not torch_module.is_tensor(x_value):
            raise TypeError(f"sample[{position}]['x'] must be a torch.Tensor")
        if x_value.ndim != 2:
            raise ValueError(
                f"sample[{position}]['x'] must be 2D [lookback, n_features]"
            )
        shape = tuple(x_value.shape)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(
                "fixed-length batch contains mismatched x shapes: "
                f"{shape} vs {expected_shape}"
            )
        x_tensors.append(x_value)

        y_value = sample["y"]
        if not torch_module.is_tensor(y_value):
            raise TypeError(f"sample[{position}]['y'] must be a torch.Tensor")
        if y_value.ndim != 0:
            raise ValueError(f"sample[{position}]['y'] must be a scalar tensor")
        y_values.append(y_value)
        target_indices.append(int(sample["target_index"]))
        window_starts.append(int(sample["window_start"]))
        window_ends.append(int(sample["window_end"]))

    x_batch = torch_module.stack(x_tensors, dim=0)
    y_batch = torch_module.stack(y_values, dim=0)
    return {
        "x": x_batch,
        "y": y_batch,
        "target_index": _stack_index_tensor(target_indices, name="target_index"),
        "window_start": _stack_index_tensor(window_starts, name="window_start"),
        "window_end": _stack_index_tensor(window_ends, name="window_end"),
    }


def pad_variable_length_sequences(
    sequences: Sequence[Any],
    *,
    padding_value: float = 0.0,
) -> tuple[Any, Any]:
    """Pad a list of ``[seq_len, n_features]`` tensors to a common length.

    Returns a tuple ``(padded, mask)``. ``padded`` has shape
    ``[batch, max_len, n_features]`` and ``mask`` is a boolean tensor of
    shape ``[batch, max_len]`` where ``True`` marks valid (non-padded)
    tokens.
    """
    torch_module = _require_torch()
    sequences_list = list(sequences)
    if not sequences_list:
        raise ValueError("pad_variable_length_sequences received no sequences")

    feature_dim: int | None = None
    lengths: list[int] = []
    for position, tensor in enumerate(sequences_list):
        if not torch_module.is_tensor(tensor):
            raise TypeError(f"sequences[{position}] must be a torch.Tensor")
        if tensor.ndim != 2:
            raise ValueError(
                f"sequences[{position}] must be 2D [seq_len, n_features]"
            )
        if feature_dim is None:
            feature_dim = int(tensor.shape[1])
        elif int(tensor.shape[1]) != feature_dim:
            raise ValueError(
                "variable-length sequences have mismatched feature "
                f"dimensions: {tensor.shape[1]} vs {feature_dim}"
            )
        if tensor.shape[0] <= 0:
            raise ValueError(
                f"sequences[{position}] must have at least one row"
            )
        lengths.append(int(tensor.shape[0]))

    assert feature_dim is not None  # for type narrowing
    max_len = max(lengths)
    dtype = sequences_list[0].dtype
    padded = torch_module.full(
        (len(sequences_list), max_len, feature_dim),
        float(padding_value),
        dtype=dtype,
    )
    mask = torch_module.zeros(
        (len(sequences_list), max_len),
        dtype=torch_module.bool,
    )
    for index, tensor in enumerate(sequences_list):
        length = lengths[index]
        padded[index, :length, :] = tensor
        mask[index, :length] = True
    return padded, mask


def collate_variable_length_batch(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Collate variable-length sequence samples with a boolean mask.

    Each sample must contain a 2D ``x`` tensor and a scalar ``y`` tensor.
    The returned mapping mirrors :func:`collate_fixed_length_batch` but
    additionally includes ``mask`` of shape ``[batch, max_len]``.
    """
    torch_module = _require_torch()
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("collate_variable_length_batch received an empty batch")

    sequences: list[Any] = []
    y_values: list[Any] = []
    target_indices: list[int] = []
    window_starts: list[int] = []
    window_ends: list[int] = []
    for position, sample in enumerate(sample_list):
        if not isinstance(sample, Mapping):
            raise TypeError(f"sample at position {position} must be a mapping")
        required = {"x", "y", "target_index", "window_start", "window_end"}
        missing = required - set(sample.keys())
        if missing:
            raise KeyError(
                f"sample at position {position} is missing keys: {sorted(missing)}"
            )
        sequences.append(sample["x"])
        y_value = sample["y"]
        if not torch_module.is_tensor(y_value):
            raise TypeError(f"sample[{position}]['y'] must be a torch.Tensor")
        if y_value.ndim != 0:
            raise ValueError(f"sample[{position}]['y'] must be a scalar tensor")
        y_values.append(y_value)
        target_indices.append(int(sample["target_index"]))
        window_starts.append(int(sample["window_start"]))
        window_ends.append(int(sample["window_end"]))

    padded, mask = pad_variable_length_sequences(sequences)
    return {
        "x": padded,
        "mask": mask,
        "y": torch_module.stack(y_values, dim=0),
        "target_index": _stack_index_tensor(target_indices, name="target_index"),
        "window_start": _stack_index_tensor(window_starts, name="window_start"),
        "window_end": _stack_index_tensor(window_ends, name="window_end"),
    }
