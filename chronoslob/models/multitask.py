"""Supervised multi-task fine-tuning over tokenised market microstructure.

Phase 14 adds supervised multi-task fine-tuning on top of the Phase 11
tokenisation layer (:mod:`chronoslob.models.tokenisation`) and the Phase 12
transformer encoder (:mod:`chronoslob.models.transformer`). It exposes one
shared backbone and one linear classification head per configured task; the
shared pooled representation drives every head.

The supported task categories are mid-price direction, return-quantile,
volatility regime, spread widening, passive fill-proxy and adverse-selection
proxy. All tasks are classification only in this phase. Binary tasks are
treated as two-class classification with cross-entropy.

This module implements only the multi-task model wrapper and loss
computation. It does not implement supervised dataset construction (see
:mod:`chronoslob.training.multitask_datasets`), label generation (see
:mod:`chronoslob.labels`), calibration, confidence filtering, execution
simulation, backtesting or any market-performance claim.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from chronoslob.models.tokenisation import (
    SPECIAL_TOKEN_IDS,
    TOKEN_FIELDS,
    SpecialToken,
)
from chronoslob.models.transformer import (
    TOKEN_WINDOW_FIELD_NAMES,
    MarketTransformerConfig,
    MarketTransformerEncoder,
)

try:  # pragma: no cover - exercised when torch is unavailable
    import torch
    from torch import nn
    from torch.nn import functional as torch_functional

    _TORCH_AVAILABLE = True
    _TORCH_MODULE_BASE: type = nn.Module
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    torch_functional = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    _TORCH_MODULE_BASE = object

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from chronoslob.models.ssl import MarketSSLTransformer

TaskTypeLiteral = Literal["classification"]

__all__ = [
    "DEFAULT_TASK_HEADS",
    "MultiTaskTransformer",
    "MultiTaskTransformerConfig",
    "MultiTaskTransformerOutput",
    "TaskHeadConfig",
    "TaskType",
    "copy_encoder_weights_from_ssl",
    "create_multitask_transformer",
]


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the multi-task transformer. Install "
            "the 'torch' optional dependency: pip install -e '.[torch]'"
        )
    return torch


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_non_negative_float(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _validate_unit_interval(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must satisfy 0 <= {name} <= 1")
    return numeric


class TaskType:
    """Stable identifiers for the supported task types.

    The class intentionally exposes only string constants; Phase 14 supports
    classification heads exclusively. Regression heads can be added later.
    """

    CLASSIFICATION: TaskTypeLiteral = "classification"


SUPPORTED_TASK_TYPES: tuple[str, ...] = (TaskType.CLASSIFICATION,)


@dataclass(frozen=True)
class TaskHeadConfig:
    """Configuration for one supervised classification head."""

    name: str
    task_type: TaskTypeLiteral = TaskType.CLASSIFICATION
    num_classes: int = 2
    loss_weight: float = 1.0
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("task name must be a non-empty string")
        if self.name != self.name.strip():
            raise ValueError("task name must not have leading or trailing whitespace")
        if self.task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError(
                f"task_type must be one of {list(SUPPORTED_TASK_TYPES)}; "
                f"got {self.task_type!r}"
            )
        _validate_positive_int(self.num_classes, name="num_classes")
        if self.num_classes < 2:
            raise ValueError("num_classes must be >= 2 (binary uses 2 classes)")
        _validate_non_negative_float(self.loss_weight, name="loss_weight")
        if isinstance(self.ignore_index, bool) or not isinstance(
            self.ignore_index, int
        ):
            raise TypeError("ignore_index must be an integer")
        if 0 <= int(self.ignore_index) < int(self.num_classes):
            raise ValueError(
                "ignore_index must not collide with a valid class index "
                f"(0 <= {self.ignore_index} < {self.num_classes})"
            )


DEFAULT_TASK_HEADS: tuple[TaskHeadConfig, ...] = (
    TaskHeadConfig(name="direction", num_classes=3, loss_weight=1.0),
    TaskHeadConfig(name="return_quantile", num_classes=5, loss_weight=1.0),
    TaskHeadConfig(name="volatility_regime", num_classes=3, loss_weight=1.0),
    TaskHeadConfig(name="spread_widening", num_classes=2, loss_weight=1.0),
    TaskHeadConfig(name="fill_probability", num_classes=2, loss_weight=1.0),
    TaskHeadConfig(name="adverse_selection", num_classes=2, loss_weight=1.0),
)


def _validate_tasks(tasks: Sequence[TaskHeadConfig]) -> tuple[TaskHeadConfig, ...]:
    cleaned = tuple(tasks)
    if not cleaned:
        raise ValueError("at least one task head must be configured")
    seen: set[str] = set()
    for position, task in enumerate(cleaned):
        if not isinstance(task, TaskHeadConfig):
            raise TypeError(
                "tasks must contain TaskHeadConfig instances; "
                f"got {type(task).__name__} at position {position}"
            )
        if task.name in seen:
            raise ValueError(f"duplicate task name {task.name!r}")
        seen.add(task.name)
    return cleaned


@dataclass(frozen=True)
class MultiTaskTransformerConfig:
    """Configuration for :class:`MultiTaskTransformer`."""

    backbone: MarketTransformerConfig = field(default_factory=MarketTransformerConfig)
    tasks: tuple[TaskHeadConfig, ...] = DEFAULT_TASK_HEADS
    dropout: float = 0.1
    freeze_backbone: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.backbone, MarketTransformerConfig):
            raise TypeError("backbone must be a MarketTransformerConfig instance")
        cleaned_tasks = _validate_tasks(self.tasks)
        object.__setattr__(self, "tasks", cleaned_tasks)
        _validate_unit_interval(self.dropout, name="dropout")
        if not isinstance(self.freeze_backbone, bool):
            raise TypeError("freeze_backbone must be a bool")
        if not any(float(task.loss_weight) > 0.0 for task in cleaned_tasks):
            raise ValueError(
                "at least one task must have a strictly positive loss_weight"
            )

    @property
    def task_names(self) -> tuple[str, ...]:
        """Return the configured task names in declaration order."""
        return tuple(task.name for task in self.tasks)

    def task_by_name(self, name: str) -> TaskHeadConfig:
        """Return the :class:`TaskHeadConfig` for ``name`` or raise ``KeyError``."""
        for task in self.tasks:
            if task.name == name:
                return task
        raise KeyError(f"no task named {name!r}")


@dataclass
class MultiTaskTransformerOutput:
    """Outputs returned by :class:`MultiTaskTransformer`."""

    loss: Any | None
    loss_components: dict[str, Any]
    logits: dict[str, Any]
    pooled: Any | None
    hidden_states: Any | None
    valid_counts: dict[str, int]


def _validate_target_tensor(
    tensor: Any,
    *,
    task_name: str,
    batch_size: int,
) -> None:
    torch_module = _require_torch()
    if not torch_module.is_tensor(tensor):
        raise TypeError(f"target for task {task_name!r} must be a torch.Tensor")
    if tensor.dtype != torch_module.long:
        raise TypeError(
            f"target for task {task_name!r} must have dtype torch.long"
        )
    if tensor.ndim != 1:
        raise ValueError(
            f"target for task {task_name!r} must be 1D [batch]; "
            f"got shape {tuple(tensor.shape)}"
        )
    if int(tensor.shape[0]) != batch_size:
        raise ValueError(
            f"target for task {task_name!r} length {int(tensor.shape[0])} "
            f"does not match batch size {batch_size}"
        )


class MultiTaskTransformer(_TORCH_MODULE_BASE):
    """Supervised multi-task transformer over tokenised market microstructure.

    The model owns one :class:`MarketTransformerEncoder` for the shared
    backbone and one linear classification head per configured task. The
    encoder's classification head is retained for API symmetry but is not
    used by the multi-task forward; the pooled representation drives every
    task head directly.
    """

    def __init__(self, config: MultiTaskTransformerConfig) -> None:
        _require_torch()
        super().__init__()
        if not isinstance(config, MultiTaskTransformerConfig):
            raise TypeError("config must be a MultiTaskTransformerConfig instance")
        self._config = config

        self.encoder = MarketTransformerEncoder(config.backbone)
        model_dim = int(config.backbone.model_dim)
        self.head_dropout = nn.Dropout(p=float(config.dropout))

        heads: dict[str, nn.Module] = {}
        for task in config.tasks:
            heads[task.name] = nn.Linear(model_dim, int(task.num_classes))
        self.task_heads = nn.ModuleDict(heads)

        if config.freeze_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    @property
    def config(self) -> MultiTaskTransformerConfig:
        """Return the validated multi-task configuration."""
        return self._config

    @property
    def task_names(self) -> tuple[str, ...]:
        """Return the configured task names in declaration order."""
        return self._config.task_names

    def n_parameters(self) -> int:
        """Return the total number of parameters in the model."""
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def n_trainable_parameters(self) -> int:
        """Return the number of trainable (``requires_grad``) parameters."""
        return int(
            sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )
        )

    def _encode(
        self,
        inputs: Mapping[str, Any],
        *,
        return_hidden_states: bool,
    ) -> tuple[Any, Any | None]:
        encoder_output = self.encoder(
            inputs,
            return_hidden_states=return_hidden_states,
            return_pooled=True,
        )
        return encoder_output.pooled, encoder_output.hidden_states

    def forward(
        self,
        inputs: Mapping[str, Any],
        *,
        targets: Mapping[str, Any] | None = None,
        target_mask: Mapping[str, Any] | None = None,
        return_hidden_states: bool = False,
        return_pooled: bool = False,
    ) -> MultiTaskTransformerOutput:
        """Run a forward pass and optionally compute a weighted multi-task loss.

        ``inputs`` must contain every required token field as a
        ``LongTensor[batch, seq_len]`` plus a ``BoolTensor[batch, seq_len]``
        attention mask, matching the Phase-12 encoder contract. ``targets``,
        when provided, maps task name to ``LongTensor[batch]`` and may use
        each task's ``ignore_index`` for missing labels. ``target_mask``, if
        provided, maps task name to ``BoolTensor[batch]`` and is treated as
        an additional missing-label mask (False positions are converted to
        ``ignore_index`` before the per-task cross-entropy).
        """
        torch_module = _require_torch()
        if not isinstance(inputs, Mapping):
            raise TypeError("inputs must be a mapping")

        pooled, hidden_states = self._encode(
            inputs,
            return_hidden_states=return_hidden_states,
        )
        batch_size = int(pooled.shape[0])
        pooled_dropped = self.head_dropout(pooled)

        logits: dict[str, Any] = {}
        for task in self._config.tasks:
            head = self.task_heads[task.name]
            logits[task.name] = head(pooled_dropped)

        loss: Any | None = None
        loss_components: dict[str, Any] = {}
        valid_counts: dict[str, int] = {task.name: 0 for task in self._config.tasks}

        if targets is None:
            if target_mask is not None:
                raise ValueError(
                    "target_mask requires targets; pass both or neither"
                )
            return MultiTaskTransformerOutput(
                loss=None,
                loss_components=loss_components,
                logits=logits,
                pooled=pooled if return_pooled else None,
                hidden_states=hidden_states if return_hidden_states else None,
                valid_counts=valid_counts,
            )

        if not isinstance(targets, Mapping):
            raise TypeError("targets must be a mapping or None")
        if target_mask is not None and not isinstance(target_mask, Mapping):
            raise TypeError("target_mask must be a mapping or None")

        active_components: list[tuple[str, Any, float]] = []
        for task in self._config.tasks:
            if task.name not in targets:
                continue
            raw_target = targets[task.name]
            _validate_target_tensor(
                raw_target,
                task_name=task.name,
                batch_size=batch_size,
            )
            adjusted_target = raw_target
            if target_mask is not None and task.name in target_mask:
                mask = target_mask[task.name]
                if not torch_module.is_tensor(mask):
                    raise TypeError(
                        f"target_mask[{task.name!r}] must be a torch.Tensor"
                    )
                if mask.dtype != torch_module.bool:
                    raise TypeError(
                        f"target_mask[{task.name!r}] must have dtype torch.bool"
                    )
                if mask.ndim != 1 or int(mask.shape[0]) != batch_size:
                    raise ValueError(
                        f"target_mask[{task.name!r}] must be 1D and length "
                        f"{batch_size}; got shape {tuple(mask.shape)}"
                    )
                ignore_tensor = torch_module.full_like(
                    raw_target, int(task.ignore_index)
                )
                adjusted_target = torch_module.where(
                    mask, raw_target, ignore_tensor
                )

            valid = int(
                (adjusted_target != int(task.ignore_index)).sum().item()
            )
            valid_counts[task.name] = valid
            if valid == 0:
                continue
            task_loss = torch_functional.cross_entropy(
                logits[task.name],
                adjusted_target,
                ignore_index=int(task.ignore_index),
                reduction="mean",
            )
            loss_components[task.name] = task_loss
            active_components.append(
                (task.name, task_loss, float(task.loss_weight))
            )

        if not active_components:
            raise ValueError(
                "multi-task forward received targets for no task with valid "
                "(non ignore_index) labels; cannot compute a loss"
            )

        loss = torch_module.zeros((), device=pooled.device)
        for _, task_loss, weight in active_components:
            loss = loss + weight * task_loss

        return MultiTaskTransformerOutput(
            loss=loss,
            loss_components=loss_components,
            logits=logits,
            pooled=pooled if return_pooled else None,
            hidden_states=hidden_states if return_hidden_states else None,
            valid_counts=valid_counts,
        )

    def predict_logits(
        self,
        inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return per-task logits in eval mode without computing gradients."""
        torch_module = _require_torch()
        was_training = self.training
        self.eval()
        try:
            with torch_module.no_grad():
                return self.forward(inputs).logits
        finally:
            if was_training:
                self.train()


def create_multitask_transformer(
    config: MultiTaskTransformerConfig,
) -> MultiTaskTransformer:
    """Construct a :class:`MultiTaskTransformer` from a validated config.

    The returned model wraps a Phase-12 transformer encoder backbone with
    one classification head per configured task. It implements supervised
    multi-task fine-tuning only and produces no benchmark or
    market-performance claim. See ``reports/multitask_finetuning.md`` for
    the full design and the explicit limitations.
    """
    return MultiTaskTransformer(config)


def copy_encoder_weights_from_ssl(
    multitask_model: MultiTaskTransformer,
    ssl_model: MarketSSLTransformer,
) -> None:
    """Copy compatible shared-encoder weights from an SSL model.

    Both models wrap the same :class:`MarketTransformerEncoder` class, so the
    encoder ``state_dict`` keys match by construction. If the configured
    backbones disagree on any parameter shape the function raises a clear
    :class:`ValueError` rather than silently overwriting incompatible
    tensors. No checkpoint files are read or written; the copy is in-memory.
    """
    _require_torch()
    # Imported lazily to avoid a circular import at module load time.
    from chronoslob.models.ssl import MarketSSLTransformer

    if not isinstance(multitask_model, MultiTaskTransformer):
        raise TypeError(
            "multitask_model must be a MultiTaskTransformer instance"
        )
    if not isinstance(ssl_model, MarketSSLTransformer):
        raise TypeError("ssl_model must be a MarketSSLTransformer instance")

    target_state = multitask_model.encoder.state_dict()
    source_state = ssl_model.encoder.state_dict()

    missing = sorted(set(target_state) - set(source_state))
    if missing:
        raise ValueError(
            "SSL encoder is missing parameter(s) expected by the multi-task "
            f"encoder: {missing}"
        )
    extra = sorted(set(source_state) - set(target_state))
    if extra:
        raise ValueError(
            "SSL encoder has parameter(s) that do not match the multi-task "
            f"encoder: {extra}"
        )
    for key, target_tensor in target_state.items():
        source_tensor = source_state[key]
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            raise ValueError(
                f"shape mismatch when copying SSL encoder parameter {key!r}: "
                f"source {tuple(source_tensor.shape)} vs target "
                f"{tuple(target_tensor.shape)}"
            )
    multitask_model.encoder.load_state_dict(source_state)


# Reference the special-token IDs the underlying encoder relies on so static
# analysers do not strip the import even though the multi-task wrapper only
# consumes integer IDs at runtime.
_RESERVED_PAD_TOKEN = SpecialToken.PAD
_RESERVED_SPECIAL_TOKEN_IDS = SPECIAL_TOKEN_IDS
_RESERVED_TOKEN_FIELDS = TOKEN_FIELDS
_RESERVED_TOKEN_FIELD_NAMES = TOKEN_WINDOW_FIELD_NAMES
