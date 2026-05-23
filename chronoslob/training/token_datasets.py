"""PyTorch-compatible datasets for tokenised event sequences.

The dataset in this module prepares fixed-length, field-wise categorical
windows for future transformer models. It does not generate labels, masking
targets, self-supervised objectives or trainable embeddings.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from chronoslob.models.tokenisation import (
    SPECIAL_TOKEN_IDS,
    TOKEN_FIELDS,
    SpecialToken,
    TokenField,
    TokenisedRecord,
    TokenSequence,
)

try:  # pragma: no cover - exercised when torch is unavailable
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    torch = None  # type: ignore[assignment]
    _TorchDataset = object  # type: ignore[assignment,misc]
    _TORCH_AVAILABLE = False

TOKEN_WINDOW_FIELD_NAMES: tuple[str, ...] = tuple(field.value for field in TOKEN_FIELDS)

SplitId = str | int | None
PaddingSide = Literal["left", "right"]


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for token sequence datasets. Install the "
            "'torch' optional dependency: pip install -e '.[torch]'"
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
class TokenWindowConfig:
    """Configuration for deterministic token-window indexing."""

    window_length: int
    stride: int = 1
    drop_incomplete: bool = False
    respect_symbol_boundaries: bool = True
    respect_split_boundaries: bool = True
    padding_side: PaddingSide = "left"

    def __post_init__(self) -> None:
        _validate_positive_int(self.window_length, name="window_length")
        _validate_positive_int(self.stride, name="stride")
        if self.padding_side not in {"left", "right"}:
            raise ValueError("padding_side must be 'left' or 'right'")


@dataclass(frozen=True)
class TokenWindowIndex:
    """Inclusive row indices for one token-window sample."""

    window_start: int
    window_end: int
    anchor_index: int

    def __post_init__(self) -> None:
        _validate_non_negative_int(self.window_start, name="window_start")
        _validate_non_negative_int(self.window_end, name="window_end")
        _validate_non_negative_int(self.anchor_index, name="anchor_index")
        if self.window_start > self.window_end:
            raise ValueError("window_start must be <= window_end")
        if self.anchor_index != self.window_end:
            raise ValueError("anchor_index must equal window_end")

    @property
    def window_slice(self) -> slice:
        """Return a Python slice covering the inclusive token window."""
        return slice(self.window_start, self.window_end + 1)

    @property
    def real_length(self) -> int:
        """Return the number of non-padding tokens represented by the index."""
        return self.window_end - self.window_start + 1


def _records_from_sequence(
    sequence: TokenSequence | Sequence[TokenisedRecord],
) -> tuple[TokenisedRecord, ...]:
    if isinstance(sequence, TokenSequence):
        return sequence.records
    records = tuple(sequence)
    for position, record in enumerate(records):
        if not isinstance(record, TokenisedRecord):
            raise TypeError(
                "sequence must contain TokenisedRecord objects; "
                f"got {type(record).__name__} at position {position}"
            )
    return records


def _validate_split_ids(
    split_ids: Sequence[SplitId] | None,
    *,
    n_records: int,
) -> tuple[SplitId, ...] | None:
    if split_ids is None:
        return None
    cleaned = tuple(split_ids)
    if len(cleaned) != n_records:
        raise ValueError(
            f"split_ids length {len(cleaned)} does not match token count {n_records}"
        )
    for position, value in enumerate(cleaned):
        if value is not None and isinstance(value, bool):
            raise TypeError(f"split_ids[{position}] must not be a boolean")
        if value is not None and not isinstance(value, (str, int)):
            raise TypeError(f"split_ids[{position}] must be a string, int or None")
    return cleaned


def _can_extend_window(
    records: Sequence[TokenisedRecord],
    split_ids: Sequence[SplitId] | None,
    config: TokenWindowConfig,
    *,
    candidate_index: int,
    anchor_index: int,
) -> bool:
    if (
        config.respect_symbol_boundaries
        and records[candidate_index].symbol != records[anchor_index].symbol
    ):
        return False
    return not (
        config.respect_split_boundaries
        and split_ids is not None
        and split_ids[candidate_index] != split_ids[anchor_index]
    )


def build_token_window_indices(
    sequence: TokenSequence | Sequence[TokenisedRecord],
    config: TokenWindowConfig,
    *,
    split_ids: Sequence[SplitId] | None = None,
) -> list[TokenWindowIndex]:
    """Build deterministic past-only token-window indices.

    Windows end at ``anchor_index`` and walk backwards up to
    ``window_length`` tokens. By default they stop before crossing symbol or
    supplied split-id boundaries. Incomplete leading windows are retained and
    padded later unless ``drop_incomplete`` is true.
    """
    records = _records_from_sequence(sequence)
    split_id_values = _validate_split_ids(split_ids, n_records=len(records))
    if not records:
        return []

    indices: list[TokenWindowIndex] = []
    for anchor_index in range(0, len(records), config.stride):
        window_start = anchor_index
        while anchor_index - window_start + 1 < config.window_length:
            candidate = window_start - 1
            if candidate < 0:
                break
            if not _can_extend_window(
                records,
                split_id_values,
                config,
                candidate_index=candidate,
                anchor_index=anchor_index,
            ):
                break
            window_start = candidate

        index = TokenWindowIndex(
            window_start=window_start,
            window_end=anchor_index,
            anchor_index=anchor_index,
        )
        if config.drop_incomplete and index.real_length < config.window_length:
            continue
        indices.append(index)
    return indices


def _field_value(record: TokenisedRecord, field_name: TokenField) -> int:
    if field_name is TokenField.EVENT_TYPE:
        return record.event_type_id
    if field_name is TokenField.SIDE:
        return record.side_id
    if field_name is TokenField.PRICE_BUCKET:
        return record.price_bucket_id
    if field_name is TokenField.QUANTITY_BUCKET:
        return record.quantity_bucket_id
    if field_name is TokenField.TIME_DELTA_BUCKET:
        return record.time_delta_bucket_id
    if field_name is TokenField.CONTEXT_BUCKET:
        return record.context_bucket_id
    if field_name is TokenField.SOURCE:
        return record.source_id
    raise ValueError(f"unsupported token field {field_name!r}")


class TokenSequenceDataset(_TorchDataset):
    """Fixed-length token-window dataset for future transformer inputs."""

    def __init__(
        self,
        sequence: TokenSequence,
        config: TokenWindowConfig,
        *,
        window_indices: Sequence[TokenWindowIndex] | None = None,
        split_ids: Sequence[SplitId] | None = None,
    ) -> None:
        _require_torch()
        if not isinstance(sequence, TokenSequence):
            raise TypeError("sequence must be a TokenSequence")
        self._sequence = sequence
        self._config = config
        if window_indices is None:
            indices = build_token_window_indices(
                sequence,
                config,
                split_ids=split_ids,
            )
        else:
            indices = list(window_indices)
            for position, index in enumerate(indices):
                if not isinstance(index, TokenWindowIndex):
                    raise TypeError(
                        "window_indices must contain TokenWindowIndex objects; "
                        f"got {type(index).__name__} at position {position}"
                    )
                if index.window_end >= len(sequence.records):
                    raise IndexError(
                        f"window_indices[{position}].window_end is outside "
                        "the token sequence"
                    )
                if index.real_length > config.window_length:
                    raise ValueError(
                        f"window_indices[{position}] is longer than window_length"
                    )
        self._window_indices = tuple(indices)

    def __len__(self) -> int:
        """Return the number of token windows."""
        return len(self._window_indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        """Return one fixed-length token window as field-wise tensors."""
        torch_module = _require_torch()
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError("token dataset indices must be integers")
        if item < 0:
            item = len(self) + item
        if item < 0 or item >= len(self):
            raise IndexError("token dataset index out of range")

        index = self._window_indices[item]
        records = self._sequence.records[index.window_slice]
        real_length = len(records)
        if real_length > self._config.window_length:
            raise RuntimeError("internal error: token window is too long")
        pad_id = int(SPECIAL_TOKEN_IDS[SpecialToken.PAD])
        output: dict[str, Any] = {
            field_name: torch_module.full(
                (self._config.window_length,),
                pad_id,
                dtype=torch_module.long,
            )
            for field_name in TOKEN_WINDOW_FIELD_NAMES
        }
        attention_mask = torch_module.zeros(
            (self._config.window_length,),
            dtype=torch_module.bool,
        )
        start_offset = (
            self._config.window_length - real_length
            if self._config.padding_side == "left"
            else 0
        )
        for offset, record in enumerate(records):
            target_position = start_offset + offset
            for field_name in TOKEN_FIELDS:
                output[field_name.value][target_position] = _field_value(
                    record,
                    field_name,
                )
            attention_mask[target_position] = True

        output["attention_mask"] = attention_mask
        output["window_start"] = int(index.window_start)
        output["window_end"] = int(index.window_end)
        output["anchor_index"] = int(index.anchor_index)
        return output

    @property
    def config(self) -> TokenWindowConfig:
        """Return the dataset window configuration."""
        return self._config

    @property
    def sequence(self) -> TokenSequence:
        """Return the underlying token sequence."""
        return self._sequence

    @property
    def window_indices(self) -> list[TokenWindowIndex]:
        """Return a defensive copy of deterministic window indices."""
        return list(self._window_indices)


__all__ = [
    "TOKEN_WINDOW_FIELD_NAMES",
    "SplitId",
    "TokenSequenceDataset",
    "TokenWindowConfig",
    "TokenWindowIndex",
    "build_token_window_indices",
]
