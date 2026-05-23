"""Deterministic masking and dataset helpers for self-supervised pretraining.

This module turns the field-wise token windows produced by Phase 11 into
inputs and labels suitable for the Phase-13 self-supervised model wrapper
(:class:`chronoslob.models.ssl.MarketSSLTransformer`). It implements:

* :func:`apply_field_masking` for deterministic BERT-style masking;
* :func:`build_next_field_targets` for one-step look-ahead self-supervised
  targets that never leak past the window;
* :class:`SSLTokenSequenceDataset`, a deterministic dataset that wraps a
  :class:`chronoslob.training.token_datasets.TokenSequenceDataset` and
  emits :class:`MaskedTokenBatch` payloads;
* :func:`collate_ssl_token_windows` for stacking those payloads.

Nothing in this module introduces market labels, supervised targets,
calibration, execution simulation, backtesting or any market-performance
claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from chronoslob.models.ssl import MaskingConfig
from chronoslob.models.tokenisation import (
    SPECIAL_TOKEN_IDS,
    SpecialToken,
    TokenField,
    TokenSequence,
)
from chronoslob.training.token_datasets import (
    TOKEN_WINDOW_FIELD_NAMES,
    TokenSequenceDataset,
    TokenWindowConfig,
)

try:  # pragma: no cover - exercised when torch is unavailable
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    torch = None  # type: ignore[assignment]
    _TorchDataset = object  # type: ignore[assignment,misc]
    _TORCH_AVAILABLE = False


__all__ = [
    "DEFAULT_IGNORE_INDEX",
    "MaskedTokenBatch",
    "MaskingPolicy",
    "SSLTokenSequenceDataset",
    "apply_field_masking",
    "build_next_field_targets",
    "collate_ssl_token_windows",
]


DEFAULT_IGNORE_INDEX: int = -100


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for SSL dataset helpers. Install the 'torch' "
            "optional dependency: pip install -e '.[torch]'"
        )
    return torch


def _validate_non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_field_tuple(
    fields: Sequence[str],
    *,
    name: str,
) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for position, value in enumerate(fields):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name}[{position}] must be a non-empty string")
        token = value.strip()
        if token not in TOKEN_WINDOW_FIELD_NAMES:
            raise ValueError(
                f"{name}[{position}]={token!r} is not a recognised token field; "
                f"expected one of {list(TOKEN_WINDOW_FIELD_NAMES)}"
            )
        if token in seen:
            raise ValueError(f"{name} contains duplicate entry {token!r}")
        cleaned.append(token)
        seen.add(token)
    return tuple(cleaned)


@dataclass(frozen=True)
class MaskingPolicy:
    """How to mask token windows during SSL training.

    The policy bundles the configurable masking probabilities with the
    set of fields to mask and the per-field vocabulary sizes needed for
    random-token replacement.
    """

    fields: tuple[str, ...]
    vocab_sizes: Mapping[str, int]
    config: MaskingConfig = field(default_factory=MaskingConfig)
    ignore_index: int = DEFAULT_IGNORE_INDEX

    def __post_init__(self) -> None:
        if not isinstance(self.config, MaskingConfig):
            raise TypeError("config must be a MaskingConfig instance")
        cleaned = _validate_field_tuple(self.fields, name="fields")
        object.__setattr__(self, "fields", cleaned)
        if not isinstance(self.vocab_sizes, Mapping):
            raise TypeError("vocab_sizes must be a mapping")
        cleaned_sizes: dict[str, int] = {}
        for field_name in cleaned:
            if field_name not in self.vocab_sizes:
                raise ValueError(
                    f"vocab_sizes is missing entry for field {field_name!r}"
                )
            size = int(self.vocab_sizes[field_name])
            if size <= 0:
                raise ValueError(
                    f"vocab_sizes[{field_name!r}] must be positive; got {size}"
                )
            cleaned_sizes[field_name] = size
        object.__setattr__(self, "vocab_sizes", cleaned_sizes)
        if isinstance(self.ignore_index, bool) or not isinstance(
            self.ignore_index, int
        ):
            raise TypeError("ignore_index must be an integer")


@dataclass
class MaskedTokenBatch:
    """One masked SSL training payload."""

    inputs: dict[str, Any]
    masked_labels: dict[str, Any]
    next_labels: dict[str, Any]
    masked_field_names: tuple[str, ...]
    next_field_names: tuple[str, ...]


def _validate_inputs_for_masking(inputs: Mapping[str, Any]) -> tuple[Any, int, int]:
    torch_module = _require_torch()
    if not isinstance(inputs, Mapping):
        raise TypeError("inputs must be a mapping of token fields to tensors")
    if "attention_mask" not in inputs:
        raise KeyError("inputs is missing required 'attention_mask'")
    attention_mask = inputs["attention_mask"]
    if not torch_module.is_tensor(attention_mask):
        raise TypeError("attention_mask must be a torch.Tensor")
    if attention_mask.dtype != torch_module.bool:
        raise TypeError("attention_mask must be a boolean tensor")
    if attention_mask.ndim != 2:
        raise ValueError(
            "attention_mask must be 2D with shape [batch, seq_len]; "
            f"got shape {tuple(attention_mask.shape)}"
        )
    batch_size = int(attention_mask.shape[0])
    seq_len = int(attention_mask.shape[1])
    for field_name in TOKEN_WINDOW_FIELD_NAMES:
        if field_name not in inputs:
            raise KeyError(f"inputs is missing required token field {field_name!r}")
        tensor = inputs[field_name]
        if not torch_module.is_tensor(tensor):
            raise TypeError(f"inputs[{field_name!r}] must be a torch.Tensor")
        if tensor.dtype != torch_module.long:
            raise TypeError(f"inputs[{field_name!r}] must have dtype torch.long")
        if tensor.ndim != 2:
            raise ValueError(
                f"inputs[{field_name!r}] must be 2D [batch, seq_len]; "
                f"got shape {tuple(tensor.shape)}"
            )
        if (
            int(tensor.shape[0]) != batch_size
            or int(tensor.shape[1]) != seq_len
        ):
            raise ValueError(
                f"inputs[{field_name!r}] shape {tuple(tensor.shape)} does not "
                f"match attention_mask shape {(batch_size, seq_len)}"
            )
    return attention_mask, batch_size, seq_len


def _force_one_mask_per_row(
    positions_to_mask: Any,
    attention_mask: Any,
    generator: Any,
) -> Any:
    torch_module = _require_torch()
    batch_size = int(attention_mask.shape[0])
    forced = positions_to_mask.clone()
    has_mask = forced.any(dim=1)
    has_real = attention_mask.any(dim=1)
    for row in range(batch_size):
        if bool(has_real[row].item()) and not bool(has_mask[row].item()):
            real_indices = attention_mask[row].nonzero(as_tuple=False).squeeze(-1)
            if int(real_indices.numel()) == 0:
                continue
            choice = torch_module.randint(
                low=0,
                high=int(real_indices.numel()),
                size=(1,),
                generator=generator,
            )
            target = int(real_indices[int(choice.item())].item())
            forced[row, target] = True
    return forced


def apply_field_masking(
    inputs: Mapping[str, Any],
    policy: MaskingPolicy,
    *,
    generator: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply deterministic BERT-style field masking.

    Returns ``(corrupted_inputs, masked_labels)``. ``corrupted_inputs`` is a
    new mapping with the same keys as ``inputs`` and replaced token IDs at
    selected positions for each masked field. ``masked_labels`` contains one
    ``LongTensor[batch, seq_len]`` per masked field with the original IDs at
    masked positions and ``policy.ignore_index`` everywhere else.

    The input mapping is never mutated. ``[PAD]`` positions are never
    selected for masking; ``attention_mask`` is preserved unchanged. Random
    replacement tokens are sampled from the non-special tail of each field
    vocabulary; if a field vocabulary contains no non-special tokens the
    random replacement step deterministically uses ``[MASK]``.
    """
    torch_module = _require_torch()
    if not isinstance(policy, MaskingPolicy):
        raise TypeError("policy must be a MaskingPolicy instance")
    attention_mask, batch_size, seq_len = _validate_inputs_for_masking(inputs)
    if generator is not None and not isinstance(generator, torch_module.Generator):
        raise TypeError("generator must be a torch.Generator or None")

    pad_id = int(SPECIAL_TOKEN_IDS[SpecialToken.PAD])
    mask_id = int(SPECIAL_TOKEN_IDS[SpecialToken.MASK])
    n_specials = len(SPECIAL_TOKEN_IDS)
    cfg = policy.config

    corrupted: dict[str, Any] = {}
    for key, value in inputs.items():
        if torch_module.is_tensor(value):
            corrupted[key] = value.clone()
        else:
            corrupted[key] = value

    # Padding positions hold the [PAD] id; ensure the contract is intact.
    for field_name in TOKEN_WINDOW_FIELD_NAMES:
        original = inputs[field_name]
        pad_violations = (~attention_mask) & (original != pad_id)
        if bool(pad_violations.any().item()):
            raise ValueError(
                f"inputs[{field_name!r}] contains non-[PAD] tokens at positions "
                "where attention_mask is False; padding positions must hold "
                "the [PAD] token id"
            )

    masked_labels: dict[str, Any] = {}
    for field_name in policy.fields:
        original = inputs[field_name]
        labels = torch_module.full(
            (batch_size, seq_len),
            int(policy.ignore_index),
            dtype=torch_module.long,
        )

        # Sample per-position decisions deterministically.
        selection_rand = torch_module.rand(
            (batch_size, seq_len), generator=generator
        )
        positions_to_mask = (
            selection_rand < float(cfg.mask_probability)
        ) & attention_mask

        if cfg.force_at_least_one_mask:
            positions_to_mask = _force_one_mask_per_row(
                positions_to_mask,
                attention_mask,
                generator=generator,
            )

        labels = torch_module.where(
            positions_to_mask,
            original,
            labels,
        )

        replacement_rand = torch_module.rand(
            (batch_size, seq_len), generator=generator
        )
        mask_cut = float(cfg.mask_token_probability)
        random_cut = mask_cut + float(cfg.random_token_probability)

        do_mask = positions_to_mask & (replacement_rand < mask_cut)
        do_random = (
            positions_to_mask
            & (replacement_rand >= mask_cut)
            & (replacement_rand < random_cut)
        )
        # The remaining slice "keep" is implicit: leave the original in place.

        vocab_size = int(policy.vocab_sizes[field_name])
        if vocab_size > n_specials:
            random_tokens = torch_module.randint(
                low=n_specials,
                high=vocab_size,
                size=(batch_size, seq_len),
                generator=generator,
                dtype=torch_module.long,
            )
        else:
            random_tokens = torch_module.full(
                (batch_size, seq_len),
                mask_id,
                dtype=torch_module.long,
            )

        updated = corrupted[field_name]
        updated = torch_module.where(
            do_mask,
            torch_module.full_like(updated, mask_id),
            updated,
        )
        updated = torch_module.where(do_random, random_tokens, updated)
        corrupted[field_name] = updated
        masked_labels[field_name] = labels

    return corrupted, masked_labels


def build_next_field_targets(
    inputs: Mapping[str, Any],
    fields: Sequence[str],
    *,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> dict[str, Any]:
    """Build one-step next-field targets without leaking past the window.

    For each configured field, the target at position ``t`` is the token ID
    of the field at position ``t + 1`` when both positions are real (i.e.
    ``attention_mask`` is True at both ``t`` and ``t + 1``). All other
    positions, including the final real token and any padding, receive
    ``ignore_index`` so they do not contribute to the loss.
    """
    torch_module = _require_torch()
    attention_mask, batch_size, seq_len = _validate_inputs_for_masking(inputs)
    cleaned_fields = _validate_field_tuple(tuple(fields), name="fields")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError("ignore_index must be an integer")

    targets: dict[str, Any] = {}
    for field_name in cleaned_fields:
        original = inputs[field_name]
        target = torch_module.full(
            (batch_size, seq_len),
            int(ignore_index),
            dtype=torch_module.long,
        )
        if seq_len < 2:
            targets[field_name] = target
            continue
        pair_valid = attention_mask[:, :-1] & attention_mask[:, 1:]
        next_tokens = original[:, 1:]
        target[:, :-1] = torch_module.where(
            pair_valid,
            next_tokens,
            torch_module.full_like(next_tokens, int(ignore_index)),
        )
        targets[field_name] = target
    return targets


def _sample_to_batch(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a per-window sample (1D tensors) into a 1-row batch."""
    torch_module = _require_torch()
    batched: dict[str, Any] = {}
    for key, value in sample.items():
        if torch_module.is_tensor(value) and value.ndim == 1:
            batched[key] = value.unsqueeze(0)
        else:
            batched[key] = value
    return batched


def _batch_to_sample(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a 1-row batched dict back to per-window 1D tensors."""
    torch_module = _require_torch()
    sample: dict[str, Any] = {}
    for key, value in batch.items():
        if torch_module.is_tensor(value) and value.ndim == 2 and int(value.shape[0]) == 1:
            sample[key] = value.squeeze(0)
        else:
            sample[key] = value
    return sample


class SSLTokenSequenceDataset(_TorchDataset):
    """Deterministic SSL dataset over field-wise token windows.

    Wraps a :class:`TokenSequenceDataset` and emits per-window dictionaries
    that contain the corrupted token inputs, the masked-field labels and
    the next-field targets. A separate ``torch.Generator`` is seeded per
    window from ``base_seed + window_index`` so the masking is stable
    regardless of dataloader ordering or batch size.
    """

    def __init__(
        self,
        base_dataset: TokenSequenceDataset,
        policy: MaskingPolicy,
        next_fields: Sequence[str],
        *,
        base_seed: int = 0,
        enable_masking: bool = True,
        enable_next_targets: bool = True,
    ) -> None:
        _require_torch()
        if not isinstance(base_dataset, TokenSequenceDataset):
            raise TypeError(
                "base_dataset must be a chronoslob.training.token_datasets."
                "TokenSequenceDataset"
            )
        if not isinstance(policy, MaskingPolicy):
            raise TypeError("policy must be a MaskingPolicy instance")
        if not isinstance(enable_masking, bool):
            raise TypeError("enable_masking must be a bool")
        if not isinstance(enable_next_targets, bool):
            raise TypeError("enable_next_targets must be a bool")
        if not enable_masking and not enable_next_targets:
            raise ValueError(
                "at least one of enable_masking or enable_next_targets must be True"
            )
        _validate_non_negative_int(base_seed, name="base_seed")
        self._base_dataset = base_dataset
        self._policy = policy
        self._next_fields = _validate_field_tuple(
            tuple(next_fields), name="next_fields"
        )
        self._base_seed = int(base_seed)
        self._enable_masking = enable_masking
        self._enable_next_targets = enable_next_targets

    def __len__(self) -> int:
        """Return the number of token windows in the wrapped dataset."""
        return len(self._base_dataset)

    @property
    def policy(self) -> MaskingPolicy:
        """Return the deterministic masking policy."""
        return self._policy

    @property
    def next_fields(self) -> tuple[str, ...]:
        """Return the fields used for next-token target construction."""
        return self._next_fields

    @property
    def base_seed(self) -> int:
        """Return the seed used to derive per-window generators."""
        return self._base_seed

    def __getitem__(self, item: int) -> dict[str, Any]:
        """Return a corrupted SSL sample for window ``item``."""
        torch_module = _require_torch()
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError("SSL dataset indices must be integers")
        if item < 0:
            item = len(self) + item
        if item < 0 or item >= len(self):
            raise IndexError("SSL dataset index out of range")

        sample = dict(self._base_dataset[item])
        as_batch = _sample_to_batch(sample)

        if self._enable_masking:
            generator = torch_module.Generator()
            generator.manual_seed(self._base_seed + int(item))
            corrupted_batch, masked_labels_batch = apply_field_masking(
                as_batch,
                self._policy,
                generator=generator,
            )
            corrupted_sample = _batch_to_sample(corrupted_batch)
            masked_labels = {
                field_name: tensor.squeeze(0)
                for field_name, tensor in masked_labels_batch.items()
            }
        else:
            corrupted_sample = dict(sample)
            masked_labels = {}

        next_labels: dict[str, Any] = {}
        if self._enable_next_targets:
            next_targets_batch = build_next_field_targets(
                as_batch,
                self._next_fields,
                ignore_index=self._policy.ignore_index,
            )
            next_labels = {
                field_name: tensor.squeeze(0)
                for field_name, tensor in next_targets_batch.items()
            }

        return {
            "inputs": corrupted_sample,
            "masked_labels": masked_labels,
            "next_labels": next_labels,
            "masked_field_names": self._policy.fields if self._enable_masking else (),
            "next_field_names": self._next_fields if self._enable_next_targets else (),
        }


def _stack_field_tensors(
    samples: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
    section: str,
) -> dict[str, Any]:
    torch_module = _require_torch()
    output: dict[str, Any] = {}
    for key in keys:
        tensors: list[Any] = []
        for position, sample in enumerate(samples):
            section_dict = sample.get(section, {})
            if not isinstance(section_dict, Mapping):
                raise TypeError(
                    f"sample[{position}][{section!r}] must be a mapping"
                )
            if key not in section_dict:
                raise KeyError(
                    f"sample[{position}][{section!r}] is missing field {key!r}"
                )
            tensor = section_dict[key]
            if not torch_module.is_tensor(tensor):
                raise TypeError(
                    f"sample[{position}][{section!r}][{key!r}] must be a "
                    "torch.Tensor"
                )
            if tensor.ndim != 1:
                raise ValueError(
                    f"sample[{position}][{section!r}][{key!r}] must be 1D; "
                    f"got shape {tuple(tensor.shape)}"
                )
            tensors.append(tensor)
        output[key] = torch_module.stack(tensors, dim=0)
    return output


def collate_ssl_token_windows(
    samples: Sequence[Mapping[str, Any]],
) -> MaskedTokenBatch:
    """Collate per-window SSL samples into a :class:`MaskedTokenBatch`.

    All samples must share the same set of masked and next-field labels.
    The input fields and attention mask are stacked exactly the way
    :func:`chronoslob.training.token_batching.collate_token_windows` does
    so the result is compatible with the Phase-12 encoder forward.
    """
    torch_module = _require_torch()
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("collate_ssl_token_windows received an empty batch")

    masked_field_names_set: tuple[str, ...] | None = None
    next_field_names_set: tuple[str, ...] | None = None
    for position, sample in enumerate(sample_list):
        if not isinstance(sample, Mapping):
            raise TypeError(f"sample[{position}] must be a mapping")
        for key in ("inputs", "masked_labels", "next_labels"):
            if key not in sample:
                raise KeyError(f"sample[{position}] is missing required key {key!r}")
        masked_field_names = tuple(sample.get("masked_field_names", ()))
        next_field_names = tuple(sample.get("next_field_names", ()))
        if masked_field_names_set is None:
            masked_field_names_set = masked_field_names
        elif masked_field_names_set != masked_field_names:
            raise ValueError(
                "SSL samples disagree on masked_field_names; all samples in a "
                "batch must use the same set of masked fields"
            )
        if next_field_names_set is None:
            next_field_names_set = next_field_names
        elif next_field_names_set != next_field_names:
            raise ValueError(
                "SSL samples disagree on next_field_names; all samples in a "
                "batch must use the same set of next fields"
            )

    if masked_field_names_set is None:
        masked_field_names_set = ()
    if next_field_names_set is None:
        next_field_names_set = ()

    input_keys = list(TOKEN_WINDOW_FIELD_NAMES)
    inputs_batched: dict[str, Any] = {}
    expected_length: int | None = None
    for key in input_keys:
        tensors: list[Any] = []
        for position, sample in enumerate(sample_list):
            inputs_dict = sample["inputs"]
            if not isinstance(inputs_dict, Mapping):
                raise TypeError(
                    f"sample[{position}]['inputs'] must be a mapping"
                )
            if key not in inputs_dict:
                raise KeyError(
                    f"sample[{position}]['inputs'] is missing token field {key!r}"
                )
            tensor = inputs_dict[key]
            if not torch_module.is_tensor(tensor):
                raise TypeError(
                    f"sample[{position}]['inputs'][{key!r}] must be a torch.Tensor"
                )
            if tensor.dtype != torch_module.long:
                raise TypeError(
                    f"sample[{position}]['inputs'][{key!r}] must have dtype "
                    "torch.long"
                )
            if tensor.ndim != 1:
                raise ValueError(
                    f"sample[{position}]['inputs'][{key!r}] must be 1D"
                )
            if expected_length is None:
                expected_length = int(tensor.shape[0])
            elif int(tensor.shape[0]) != expected_length:
                raise ValueError(
                    f"sample[{position}]['inputs'][{key!r}] has inconsistent "
                    "length"
                )
            tensors.append(tensor)
        inputs_batched[key] = torch_module.stack(tensors, dim=0)

    attention_masks: list[Any] = []
    for position, sample in enumerate(sample_list):
        inputs_dict = sample["inputs"]
        if "attention_mask" not in inputs_dict:
            raise KeyError(
                f"sample[{position}]['inputs'] is missing 'attention_mask'"
            )
        mask = inputs_dict["attention_mask"]
        if not torch_module.is_tensor(mask):
            raise TypeError(
                f"sample[{position}]['inputs']['attention_mask'] must be a tensor"
            )
        if mask.dtype != torch_module.bool:
            raise TypeError(
                f"sample[{position}]['inputs']['attention_mask'] must be bool"
            )
        if mask.ndim != 1 or int(mask.shape[0]) != int(expected_length or 0):
            raise ValueError(
                f"sample[{position}]['inputs']['attention_mask'] shape is invalid"
            )
        attention_masks.append(mask)
    inputs_batched["attention_mask"] = torch_module.stack(attention_masks, dim=0)

    masked_labels = (
        _stack_field_tensors(
            sample_list, keys=masked_field_names_set, section="masked_labels"
        )
        if masked_field_names_set
        else {}
    )
    next_labels = (
        _stack_field_tensors(
            sample_list, keys=next_field_names_set, section="next_labels"
        )
        if next_field_names_set
        else {}
    )

    return MaskedTokenBatch(
        inputs=inputs_batched,
        masked_labels=masked_labels,
        next_labels=next_labels,
        masked_field_names=tuple(masked_field_names_set),
        next_field_names=tuple(next_field_names_set),
    )


# Reference the imports we rely on for typing and runtime equivalence checks
# so static analysers do not strip them.
_RESERVED_TOKEN_FIELD = TokenField
_RESERVED_TOKEN_SEQUENCE = TokenSequence
_RESERVED_TOKEN_WINDOW_CONFIG = TokenWindowConfig
