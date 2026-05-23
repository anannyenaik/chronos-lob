"""Batching helpers for field-wise token windows.

These helpers stack or pad token-window dictionaries. They do not create
targets, random masks or model inputs beyond deterministic categorical fields
and boolean attention masks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from chronoslob.models.tokenisation import SPECIAL_TOKEN_IDS, SpecialToken
from chronoslob.training.token_datasets import TOKEN_WINDOW_FIELD_NAMES

try:  # pragma: no cover - exercised when torch is unavailable
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

PaddingSide = Literal["left", "right"]
_METADATA_KEYS = ("window_start", "window_end", "anchor_index")


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for token batching helpers. Install the "
            "'torch' optional dependency: pip install -e '.[torch]'"
        )
    return torch


def _sample_list(samples: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    sample_values = list(samples)
    if not sample_values:
        raise ValueError("token batching received an empty batch")
    for position, sample in enumerate(sample_values):
        if not isinstance(sample, Mapping):
            raise TypeError(f"sample at position {position} must be a mapping")
    return sample_values


def _require_field_tensor(
    sample: Mapping[str, Any],
    *,
    key: str,
    position: int,
) -> Any:
    torch_module = _require_torch()
    if key not in sample:
        raise KeyError(f"sample at position {position} is missing key {key!r}")
    tensor = sample[key]
    if not torch_module.is_tensor(tensor):
        raise TypeError(f"sample[{position}][{key!r}] must be a torch.Tensor")
    if tensor.ndim != 1:
        raise ValueError(f"sample[{position}][{key!r}] must be 1D")
    return tensor


def _require_attention_mask(sample: Mapping[str, Any], *, position: int) -> Any:
    torch_module = _require_torch()
    mask = _require_field_tensor(sample, key="attention_mask", position=position)
    if mask.dtype != torch_module.bool:
        raise TypeError(f"sample[{position}]['attention_mask'] must be bool")
    return mask


def _validate_categorical_field(tensor: Any, *, key: str, position: int) -> None:
    torch_module = _require_torch()
    if tensor.dtype != torch_module.long:
        raise TypeError(f"sample[{position}][{key!r}] must have dtype torch.long")


def _stack_optional_metadata(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    torch_module = _require_torch()
    metadata: dict[str, Any] = {}
    for key in _METADATA_KEYS:
        if all(key in sample for sample in samples):
            metadata[key] = torch_module.tensor(
                [int(sample[key]) for sample in samples],
                dtype=torch_module.long,
            )
    return metadata


def collate_token_windows(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack fixed-length token-window samples into a batch."""
    torch_module = _require_torch()
    sample_values = _sample_list(samples)
    expected_length: int | None = None
    batched: dict[str, list[Any]] = {
        field_name: [] for field_name in TOKEN_WINDOW_FIELD_NAMES
    }
    masks: list[Any] = []

    for position, sample in enumerate(sample_values):
        mask = _require_attention_mask(sample, position=position)
        if expected_length is None:
            expected_length = int(mask.shape[0])
        elif int(mask.shape[0]) != expected_length:
            raise ValueError("token windows have mismatched attention_mask lengths")
        masks.append(mask)

        for field_name in TOKEN_WINDOW_FIELD_NAMES:
            tensor = _require_field_tensor(sample, key=field_name, position=position)
            _validate_categorical_field(
                tensor,
                key=field_name,
                position=position,
            )
            if int(tensor.shape[0]) != expected_length:
                raise ValueError(
                    f"sample[{position}][{field_name!r}] length does not match "
                    "attention_mask"
                )
            batched[field_name].append(tensor)

    output = {
        field_name: torch_module.stack(tensors, dim=0)
        for field_name, tensors in batched.items()
    }
    output["attention_mask"] = torch_module.stack(masks, dim=0)
    output.update(_stack_optional_metadata(sample_values))
    return output


def pad_variable_length_token_windows(
    samples: Sequence[Mapping[str, Any]],
    *,
    pad_token_id: int | None = None,
    padding_side: PaddingSide = "right",
) -> dict[str, Any]:
    """Pad variable-length token-window samples to the longest sample."""
    torch_module = _require_torch()
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'")
    sample_values = _sample_list(samples)
    if pad_token_id is None:
        pad_token_id = int(SPECIAL_TOKEN_IDS[SpecialToken.PAD])

    lengths: list[int] = []
    field_tensors: dict[str, list[Any]] = {
        field_name: [] for field_name in TOKEN_WINDOW_FIELD_NAMES
    }
    masks: list[Any] = []
    for position, sample in enumerate(sample_values):
        sample_length: int | None = None
        for field_name in TOKEN_WINDOW_FIELD_NAMES:
            tensor = _require_field_tensor(sample, key=field_name, position=position)
            _validate_categorical_field(
                tensor,
                key=field_name,
                position=position,
            )
            if sample_length is None:
                sample_length = int(tensor.shape[0])
            elif int(tensor.shape[0]) != sample_length:
                raise ValueError(
                    f"sample[{position}] has inconsistent categorical field lengths"
                )
            field_tensors[field_name].append(tensor)
        assert sample_length is not None
        if "attention_mask" in sample:
            mask = _require_attention_mask(sample, position=position)
            if int(mask.shape[0]) != sample_length:
                raise ValueError(
                    f"sample[{position}]['attention_mask'] length does not match "
                    "categorical fields"
                )
        else:
            mask = torch_module.ones((sample_length,), dtype=torch_module.bool)
        lengths.append(sample_length)
        masks.append(mask)

    max_length = max(lengths)
    output: dict[str, Any] = {}
    for field_name, tensors in field_tensors.items():
        padded = torch_module.full(
            (len(sample_values), max_length),
            int(pad_token_id),
            dtype=torch_module.long,
        )
        for row_index, tensor in enumerate(tensors):
            length = lengths[row_index]
            start = max_length - length if padding_side == "left" else 0
            padded[row_index, start : start + length] = tensor
        output[field_name] = padded

    attention = torch_module.zeros(
        (len(sample_values), max_length),
        dtype=torch_module.bool,
    )
    for row_index, mask in enumerate(masks):
        length = lengths[row_index]
        start = max_length - length if padding_side == "left" else 0
        attention[row_index, start : start + length] = mask
    output["attention_mask"] = attention
    output.update(_stack_optional_metadata(sample_values))
    return output


__all__ = [
    "collate_token_windows",
    "pad_variable_length_token_windows",
]
