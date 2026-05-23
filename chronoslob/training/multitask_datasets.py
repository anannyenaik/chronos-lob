"""Supervised multi-task datasets for tokenised market microstructure.

Phase 14 wires the Phase-11 field-wise token windows
(:mod:`chronoslob.training.token_datasets`) and an external supervised label
table into a dataset whose samples carry one ``LongTensor`` target per task
plus a boolean mask marking which targets are present. The dataset never
generates labels itself; it consumes a pre-built label table produced by the
existing :mod:`chronoslob.labels` pipeline. This keeps the leakage contract
on the label side, away from the tokeniser.

Label alignment uses the token window end position as the prediction
timestamp. The default behaviour drops windows that have no valid labels for
any task; partially-missing tasks are kept and surfaced through the
``target_mask`` dictionary so the model can ignore them via ``ignore_index``.

This module does not implement supervised training, calibration, confidence
filtering or any market-performance claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from chronoslob.models.tokenisation import (
    TokenisedRecord,
    TokenSequence,
)
from chronoslob.training.token_datasets import (
    TOKEN_WINDOW_FIELD_NAMES,
    TokenSequenceDataset,
    TokenWindowConfig,
    TokenWindowIndex,
    build_token_window_indices,
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
    "DEFAULT_MULTITASK_IGNORE_INDEX",
    "MultiTaskLabelSpec",
    "MultiTaskSampleIndex",
    "MultiTaskTokenDataset",
    "MultiTaskWindowConfig",
    "build_multitask_sample_indices",
    "collate_multitask_token_windows",
]


DEFAULT_MULTITASK_IGNORE_INDEX: int = -100


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for multi-task dataset helpers. Install "
            "the 'torch' optional dependency: pip install -e '.[torch]'"
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
class MultiTaskLabelSpec:
    """How to interpret one column of a supervised label table.

    The spec ties a task name to the per-record integer labels supplied by
    the caller (typically produced by the existing
    :mod:`chronoslob.labels` pipeline). Labels are integer class IDs in
    ``[0, num_classes)``; ``None`` represents a missing label and is
    converted to ``ignore_index`` at collate time.
    """

    name: str
    num_classes: int
    ignore_index: int = DEFAULT_MULTITASK_IGNORE_INDEX

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("MultiTaskLabelSpec.name must be a non-empty string")
        if self.name != self.name.strip():
            raise ValueError(
                "MultiTaskLabelSpec.name must not have leading or trailing "
                "whitespace"
            )
        _validate_positive_int(self.num_classes, name="num_classes")
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2 (binary uses 2 classes)")
        if isinstance(self.ignore_index, bool) or not isinstance(
            self.ignore_index, int
        ):
            raise TypeError("ignore_index must be an integer")
        if 0 <= int(self.ignore_index) < int(self.num_classes):
            raise ValueError(
                "ignore_index must not collide with a valid class index "
                f"(0 <= {self.ignore_index} < {self.num_classes})"
            )


@dataclass(frozen=True)
class MultiTaskWindowConfig:
    """Configuration for multi-task sample-index construction."""

    window_length: int
    stride: int = 1
    drop_incomplete: bool = False
    respect_symbol_boundaries: bool = True
    respect_split_boundaries: bool = True
    drop_all_missing_samples: bool = True

    def __post_init__(self) -> None:
        _validate_positive_int(self.window_length, name="window_length")
        _validate_positive_int(self.stride, name="stride")
        if not isinstance(self.drop_incomplete, bool):
            raise TypeError("drop_incomplete must be a bool")
        if not isinstance(self.respect_symbol_boundaries, bool):
            raise TypeError("respect_symbol_boundaries must be a bool")
        if not isinstance(self.respect_split_boundaries, bool):
            raise TypeError("respect_split_boundaries must be a bool")
        if not isinstance(self.drop_all_missing_samples, bool):
            raise TypeError("drop_all_missing_samples must be a bool")

    def to_token_window_config(self) -> TokenWindowConfig:
        """Return the matching :class:`TokenWindowConfig`."""
        return TokenWindowConfig(
            window_length=self.window_length,
            stride=self.stride,
            drop_incomplete=self.drop_incomplete,
            respect_symbol_boundaries=self.respect_symbol_boundaries,
            respect_split_boundaries=self.respect_split_boundaries,
            padding_side="left",
        )


@dataclass(frozen=True)
class MultiTaskSampleIndex:
    """One supervised multi-task sample index.

    ``window_index`` describes the past-only token window. ``targets`` maps
    task name to integer class ID or ``None`` for missing labels.
    ``target_record_index`` is the underlying tokenised record index used to
    resolve the label (always equal to ``window_end``); it is recorded
    explicitly so downstream tests can verify the alignment contract.
    """

    window_index: TokenWindowIndex
    targets: dict[str, int | None]
    target_record_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.window_index, TokenWindowIndex):
            raise TypeError("window_index must be a TokenWindowIndex")
        if not isinstance(self.targets, Mapping):
            raise TypeError("targets must be a mapping of task name to int or None")
        if int(self.target_record_index) != int(self.window_index.window_end):
            raise ValueError(
                "target_record_index must equal window_end; "
                f"got {self.target_record_index} != {self.window_index.window_end}"
            )

    @property
    def has_any_target(self) -> bool:
        """Return ``True`` if at least one task has a non-missing label."""
        return any(value is not None for value in self.targets.values())


def _resolve_label_for_record(
    record: TokenisedRecord,
    *,
    record_index: int,
    label_table: Mapping[Any, Mapping[str, int | None]] | None,
) -> Mapping[str, int | None]:
    """Resolve labels for ``record`` using the supplied label table.

    The label table maps a lookup key (either the tokenised record position
    or, when available, the source-record position, canonical ``sequence_id``,
    ``(symbol, timestamp)`` pair or exact timestamp) to a mapping from task
    name to integer class ID or ``None``.
    """
    if label_table is None:
        return {}
    candidate_keys: list[Any] = [record_index]
    if record.source_record_index is not None:
        candidate_keys.append(record.source_record_index)
    if record.sequence_id is not None:
        candidate_keys.append(record.sequence_id)
    if record.symbol is not None and record.timestamp is not None:
        candidate_keys.append((record.symbol, record.timestamp))
    if record.timestamp is not None:
        candidate_keys.append(record.timestamp)
    for key in candidate_keys:
        if key in label_table:
            return label_table[key]
    return {}


def _normalise_label_key(value: Any) -> Any:
    """Return a stable key for exact label lookup."""
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    return value


def _label_frame_to_table(
    label_frame: Any,
    label_specs: Sequence[MultiTaskLabelSpec],
    *,
    label_key_column: str | None,
) -> dict[Any, dict[str, int | None]]:
    """Convert a dataframe-like label frame to the lookup-table format.

    The frame must contain one column per task spec. If ``label_key_column``
    is supplied, that column is used as the lookup key. Otherwise a
    ``(symbol, timestamp)`` key is used when both columns are present, an
    exact ``timestamp`` key is used when only that column is present, and
    the dataframe index is used as the final fallback.
    """
    if not hasattr(label_frame, "iterrows") or not hasattr(label_frame, "columns"):
        raise TypeError("label_frame must be a pandas-like DataFrame")
    columns = {str(column) for column in label_frame.columns}
    missing_columns = [spec.name for spec in label_specs if spec.name not in columns]
    if missing_columns:
        raise ValueError(
            "label_frame is missing task label column(s): "
            f"{missing_columns}"
        )
    if label_key_column is not None and label_key_column not in columns:
        raise ValueError(
            f"label_key_column {label_key_column!r} is not present in label_frame"
        )

    table: dict[Any, dict[str, int | None]] = {}
    use_symbol_timestamp = (
        label_key_column is None and {"symbol", "timestamp"}.issubset(columns)
    )
    use_timestamp = label_key_column is None and "timestamp" in columns
    for index, row in label_frame.iterrows():
        if label_key_column is not None:
            key = _normalise_label_key(row[label_key_column])
        elif use_symbol_timestamp:
            key = (
                row["symbol"],
                _normalise_label_key(row["timestamp"]),
            )
        elif use_timestamp:
            key = _normalise_label_key(row["timestamp"])
        else:
            key = _normalise_label_key(index)
        table[key] = {spec.name: row[spec.name] for spec in label_specs}
    return table


def _coerce_label(
    raw: Any,
    *,
    task_name: str,
    spec: MultiTaskLabelSpec,
) -> int | None:
    if raw is None:
        return None
    if hasattr(raw, "item"):
        raw = raw.item()
    if isinstance(raw, float) and raw != raw:
        return None
    if isinstance(raw, bool):
        raw = int(raw)
    if not isinstance(raw, int):
        raise TypeError(
            f"label for task {task_name!r} must be an int or None; "
            f"got {type(raw).__name__}"
        )
    if int(raw) == int(spec.ignore_index):
        return None
    if int(raw) < 0 or int(raw) >= int(spec.num_classes):
        raise ValueError(
            f"label for task {task_name!r}={raw} is outside [0, "
            f"{spec.num_classes})"
        )
    return int(raw)


def build_multitask_sample_indices(
    sequence: TokenSequence,
    config: MultiTaskWindowConfig,
    label_specs: Sequence[MultiTaskLabelSpec],
    *,
    label_table: Mapping[Any, Mapping[str, int | None]] | None = None,
    label_frame: Any | None = None,
    label_key_column: str | None = None,
    split_ids: Sequence[Any] | None = None,
) -> list[MultiTaskSampleIndex]:
    """Build deterministic multi-task sample indices.

    Each sample carries the token-window index, the resolved per-task
    integer labels (or ``None`` for missing labels) and the record index
    used to align labels. Samples whose tasks are all missing labels are
    dropped when ``config.drop_all_missing_samples`` is true.

    Labels must already have been produced by the existing label pipeline;
    this function never generates labels.
    """
    if not isinstance(sequence, TokenSequence):
        raise TypeError("sequence must be a TokenSequence")
    if not isinstance(config, MultiTaskWindowConfig):
        raise TypeError("config must be a MultiTaskWindowConfig")
    cleaned_specs = tuple(label_specs)
    if not cleaned_specs:
        raise ValueError("at least one MultiTaskLabelSpec must be supplied")
    seen_names: set[str] = set()
    for position, spec in enumerate(cleaned_specs):
        if not isinstance(spec, MultiTaskLabelSpec):
            raise TypeError(
                "label_specs must contain MultiTaskLabelSpec instances; "
                f"got {type(spec).__name__} at position {position}"
            )
        if spec.name in seen_names:
            raise ValueError(f"duplicate task name in label_specs: {spec.name!r}")
        seen_names.add(spec.name)
    if label_table is not None and label_frame is not None:
        raise ValueError("pass either label_table or label_frame, not both")
    resolved_label_table: Mapping[Any, Mapping[str, int | None]] | None = label_table
    if label_frame is not None:
        resolved_label_table = _label_frame_to_table(
            label_frame,
            cleaned_specs,
            label_key_column=label_key_column,
        )

    indices = build_token_window_indices(
        sequence,
        config.to_token_window_config(),
        split_ids=split_ids,
    )

    samples: list[MultiTaskSampleIndex] = []
    for window_index in indices:
        record_index = int(window_index.window_end)
        record = sequence.records[record_index]
        raw_labels = _resolve_label_for_record(
            record,
            record_index=record_index,
            label_table=resolved_label_table,
        )
        resolved: dict[str, int | None] = {}
        for spec in cleaned_specs:
            resolved[spec.name] = _coerce_label(
                raw_labels.get(spec.name),
                task_name=spec.name,
                spec=spec,
            )
        sample = MultiTaskSampleIndex(
            window_index=window_index,
            targets=resolved,
            target_record_index=record_index,
        )
        if (
            config.drop_all_missing_samples
            and not sample.has_any_target
        ):
            continue
        samples.append(sample)
    return samples


class MultiTaskTokenDataset(_TorchDataset):
    """Supervised multi-task dataset over field-wise token windows.

    Each sample carries the field-wise token tensors and an attention mask
    produced by :class:`TokenSequenceDataset`, plus two per-task
    dictionaries: ``targets`` holds ``LongTensor[]`` class IDs (``ignore_index``
    when missing) and ``target_mask`` holds ``BoolTensor[]`` markers
    (``True`` when the label is present).
    """

    def __init__(
        self,
        sequence: TokenSequence,
        config: MultiTaskWindowConfig,
        label_specs: Sequence[MultiTaskLabelSpec],
        *,
        label_table: Mapping[Any, Mapping[str, int | None]] | None = None,
        label_frame: Any | None = None,
        label_key_column: str | None = None,
        split_ids: Sequence[Any] | None = None,
    ) -> None:
        _require_torch()
        if not isinstance(sequence, TokenSequence):
            raise TypeError("sequence must be a TokenSequence")
        if not isinstance(config, MultiTaskWindowConfig):
            raise TypeError("config must be a MultiTaskWindowConfig")

        self._sequence = sequence
        self._config = config
        self._label_specs = tuple(label_specs)
        if not self._label_specs:
            raise ValueError("at least one MultiTaskLabelSpec must be supplied")
        self._label_spec_by_name: dict[str, MultiTaskLabelSpec] = {
            spec.name: spec for spec in self._label_specs
        }
        if len(self._label_spec_by_name) != len(self._label_specs):
            raise ValueError("label_specs contains duplicate task names")

        self._samples = tuple(
            build_multitask_sample_indices(
                sequence,
                config,
                self._label_specs,
                label_table=label_table,
                label_frame=label_frame,
                label_key_column=label_key_column,
                split_ids=split_ids,
            )
        )
        self._base_dataset = TokenSequenceDataset(
            sequence,
            config.to_token_window_config(),
            window_indices=[sample.window_index for sample in self._samples],
            split_ids=split_ids,
        )

    def __len__(self) -> int:
        """Return the number of supervised multi-task samples."""
        return len(self._samples)

    @property
    def config(self) -> MultiTaskWindowConfig:
        """Return the multi-task window configuration."""
        return self._config

    @property
    def label_specs(self) -> tuple[MultiTaskLabelSpec, ...]:
        """Return the configured label specifications."""
        return self._label_specs

    @property
    def samples(self) -> tuple[MultiTaskSampleIndex, ...]:
        """Return the resolved multi-task sample indices."""
        return self._samples

    def task_label_counts(self) -> dict[str, int]:
        """Return the number of non-missing labels per task."""
        counts: dict[str, int] = {spec.name: 0 for spec in self._label_specs}
        for sample in self._samples:
            for task_name, value in sample.targets.items():
                if value is not None:
                    counts[task_name] += 1
        return counts

    def __getitem__(self, item: int) -> dict[str, Any]:
        """Return one supervised multi-task sample."""
        torch_module = _require_torch()
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError("multi-task dataset indices must be integers")
        if item < 0:
            item = len(self) + item
        if item < 0 or item >= len(self):
            raise IndexError("multi-task dataset index out of range")

        base_sample = dict(self._base_dataset[item])
        sample_index = self._samples[item]

        targets: dict[str, Any] = {}
        target_mask: dict[str, Any] = {}
        for spec in self._label_specs:
            value = sample_index.targets.get(spec.name)
            if value is None:
                targets[spec.name] = torch_module.tensor(
                    int(spec.ignore_index), dtype=torch_module.long
                )
                target_mask[spec.name] = torch_module.tensor(
                    False, dtype=torch_module.bool
                )
            else:
                targets[spec.name] = torch_module.tensor(
                    int(value), dtype=torch_module.long
                )
                target_mask[spec.name] = torch_module.tensor(
                    True, dtype=torch_module.bool
                )

        base_sample["targets"] = targets
        base_sample["target_mask"] = target_mask
        base_sample["target_record_index"] = int(sample_index.target_record_index)
        return base_sample


def collate_multitask_token_windows(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Stack multi-task token-window samples into a batched dictionary.

    The output mirrors :func:`chronoslob.training.token_batching.collate_token_windows`
    for the token fields and attention mask, and additionally returns two
    per-task dictionaries: ``targets`` (``LongTensor[batch]``) and
    ``target_mask`` (``BoolTensor[batch]``). Missing labels are represented
    by ``False`` mask entries; the underlying target value remains the
    sample's ``ignore_index`` so the model loss can use ``ignore_index``
    directly.
    """
    torch_module = _require_torch()
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("collate_multitask_token_windows received an empty batch")

    expected_length: int | None = None
    field_tensors: dict[str, list[Any]] = {
        field_name: [] for field_name in TOKEN_WINDOW_FIELD_NAMES
    }
    attention_masks: list[Any] = []
    targets_by_task: dict[str, list[Any]] = {}
    target_mask_by_task: dict[str, list[Any]] = {}
    metadata_record_index: list[int] = []
    has_record_index = True
    metadata_window_end: list[int] = []
    has_window_end = True

    for position, sample in enumerate(sample_list):
        if not isinstance(sample, Mapping):
            raise TypeError(f"sample[{position}] must be a mapping")
        if "attention_mask" not in sample:
            raise KeyError(f"sample[{position}] is missing 'attention_mask'")
        mask = sample["attention_mask"]
        if not torch_module.is_tensor(mask) or mask.dtype != torch_module.bool:
            raise TypeError(
                f"sample[{position}]['attention_mask'] must be a bool tensor"
            )
        if mask.ndim != 1:
            raise ValueError(
                f"sample[{position}]['attention_mask'] must be 1D"
            )
        if expected_length is None:
            expected_length = int(mask.shape[0])
        elif int(mask.shape[0]) != expected_length:
            raise ValueError(
                "multi-task samples have mismatched attention_mask lengths"
            )
        attention_masks.append(mask)
        for field_name in TOKEN_WINDOW_FIELD_NAMES:
            if field_name not in sample:
                raise KeyError(
                    f"sample[{position}] is missing token field {field_name!r}"
                )
            tensor = sample[field_name]
            if not torch_module.is_tensor(tensor):
                raise TypeError(
                    f"sample[{position}][{field_name!r}] must be a torch.Tensor"
                )
            if tensor.dtype != torch_module.long:
                raise TypeError(
                    f"sample[{position}][{field_name!r}] must have dtype torch.long"
                )
            if tensor.ndim != 1 or int(tensor.shape[0]) != expected_length:
                raise ValueError(
                    f"sample[{position}][{field_name!r}] shape is invalid"
                )
            field_tensors[field_name].append(tensor)

        if "targets" not in sample or "target_mask" not in sample:
            raise KeyError(
                f"sample[{position}] is missing 'targets' or 'target_mask'"
            )
        sample_targets = sample["targets"]
        sample_masks = sample["target_mask"]
        if not isinstance(sample_targets, Mapping):
            raise TypeError(f"sample[{position}]['targets'] must be a mapping")
        if not isinstance(sample_masks, Mapping):
            raise TypeError(
                f"sample[{position}]['target_mask'] must be a mapping"
            )
        if set(sample_targets.keys()) != set(sample_masks.keys()):
            raise ValueError(
                f"sample[{position}] targets and target_mask have mismatched "
                "task names"
            )
        if position == 0:
            for task_name in sample_targets:
                targets_by_task[task_name] = []
                target_mask_by_task[task_name] = []
        else:
            if set(sample_targets) != set(targets_by_task):
                raise ValueError(
                    f"sample[{position}] task names differ from sample[0]"
                )
        for task_name, target_tensor in sample_targets.items():
            if not torch_module.is_tensor(target_tensor):
                raise TypeError(
                    f"sample[{position}]['targets'][{task_name!r}] must be a "
                    "torch.Tensor"
                )
            if target_tensor.dtype != torch_module.long:
                raise TypeError(
                    f"sample[{position}]['targets'][{task_name!r}] must have "
                    "dtype torch.long"
                )
            if target_tensor.ndim != 0:
                raise ValueError(
                    f"sample[{position}]['targets'][{task_name!r}] must be 0D "
                    f"(scalar); got shape {tuple(target_tensor.shape)}"
                )
            mask_tensor = sample_masks[task_name]
            if not torch_module.is_tensor(mask_tensor):
                raise TypeError(
                    f"sample[{position}]['target_mask'][{task_name!r}] must be "
                    "a torch.Tensor"
                )
            if mask_tensor.dtype != torch_module.bool:
                raise TypeError(
                    f"sample[{position}]['target_mask'][{task_name!r}] must "
                    "have dtype torch.bool"
                )
            if mask_tensor.ndim != 0:
                raise ValueError(
                    f"sample[{position}]['target_mask'][{task_name!r}] must "
                    f"be 0D (scalar); got shape {tuple(mask_tensor.shape)}"
                )
            targets_by_task[task_name].append(target_tensor)
            target_mask_by_task[task_name].append(mask_tensor)

        if has_record_index:
            if "target_record_index" in sample:
                metadata_record_index.append(int(sample["target_record_index"]))
            else:
                has_record_index = False
        if has_window_end:
            if "window_end" in sample:
                metadata_window_end.append(int(sample["window_end"]))
            else:
                has_window_end = False

    batched: dict[str, Any] = {
        field_name: torch_module.stack(tensors, dim=0)
        for field_name, tensors in field_tensors.items()
    }
    batched["attention_mask"] = torch_module.stack(attention_masks, dim=0)
    batched["targets"] = {
        task_name: torch_module.stack(tensors, dim=0)
        for task_name, tensors in targets_by_task.items()
    }
    batched["target_mask"] = {
        task_name: torch_module.stack(tensors, dim=0)
        for task_name, tensors in target_mask_by_task.items()
    }
    if has_record_index and metadata_record_index:
        batched["target_record_index"] = torch_module.tensor(
            metadata_record_index, dtype=torch_module.long
        )
    if has_window_end and metadata_window_end:
        batched["window_end"] = torch_module.tensor(
            metadata_window_end, dtype=torch_module.long
        )
    return batched
