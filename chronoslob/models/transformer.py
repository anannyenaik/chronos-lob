"""Transformer encoder for field-wise tokenised market microstructure inputs.

This module wires the field-wise categorical token batches produced by Phase
11 (see :mod:`chronoslob.models.tokenisation` and
:mod:`chronoslob.training.token_datasets`) to a small, auditable
``torch.nn.TransformerEncoder`` and a supervised classification head. It is
strictly an architecture and plumbing module: no self-supervised objectives,
no masked event modelling, no next-event prediction, no calibration, no
execution simulation and no benchmark or alpha results are implemented here.

PyTorch is treated as an optional dependency. The model class imports
``torch`` directly and raises a clear ``ImportError`` when the optional
``[torch]`` extra is not installed. Tests should guard imports with
``pytest.importorskip("torch")``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from chronoslob.models.tokenisation import (
    SPECIAL_TOKEN_IDS,
    TOKEN_FIELDS,
    SpecialToken,
)

# Derive the canonical token-field names directly from the Phase 11 enum so
# importing the transformer model does not require importing the training
# package (which would re-import this module and create a circular import).
# This mirrors the definition in
# :mod:`chronoslob.training.token_datasets` (verified by an import-equality
# test) and stays trivially in sync because both come from ``TOKEN_FIELDS``.
TOKEN_WINDOW_FIELD_NAMES: tuple[str, ...] = tuple(
    token_field.value for token_field in TOKEN_FIELDS
)

try:  # pragma: no cover - exercised when torch is unavailable
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
    _TORCH_MODULE_BASE: type = nn.Module
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    _TORCH_MODULE_BASE = object

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    import torch as _torch_typing  # noqa: F401

PoolingMode = Literal["mean", "last", "bos"]
ActivationName = Literal["relu", "gelu"]

__all__ = [
    "MarketTransformerConfig",
    "MarketTransformerEncoder",
    "MarketTransformerOutput",
    "TokenFieldEmbeddingConfig",
    "TransformerPooling",
    "create_market_transformer",
]


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the market transformer encoder. Install "
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


def _validate_unit_interval(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must satisfy 0 <= {name} <= 1")
    return numeric


def _default_vocab_sizes() -> dict[str, int]:
    """Return a minimal placeholder vocabulary covering required token fields.

    The size is the minimum needed to hold the five fixed special tokens. In
    production use, sizes should be supplied from a frozen
    :class:`chronoslob.models.tokenisation.TokenVocabulary`.
    """
    minimum_size = len(SPECIAL_TOKEN_IDS)
    return dict.fromkeys(TOKEN_WINDOW_FIELD_NAMES, minimum_size)


@dataclass(frozen=True)
class TokenFieldEmbeddingConfig:
    """Configuration for the field-wise categorical embedding block."""

    vocab_sizes: dict[str, int]
    field_embedding_dim: int = 16
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.vocab_sizes, Mapping):
            raise TypeError("vocab_sizes must be a mapping of field name to vocab size")
        cleaned: dict[str, int] = {}
        for field_name in TOKEN_WINDOW_FIELD_NAMES:
            if field_name not in self.vocab_sizes:
                raise ValueError(
                    f"vocab_sizes is missing required token field {field_name!r}"
                )
            cleaned[field_name] = _validate_positive_int(
                int(self.vocab_sizes[field_name]),
                name=f"vocab_sizes[{field_name!r}]",
            )
        extra_keys = sorted(set(self.vocab_sizes) - set(TOKEN_WINDOW_FIELD_NAMES))
        if extra_keys:
            raise ValueError(
                "vocab_sizes contains unknown token fields: "
                f"{extra_keys}; expected one of {list(TOKEN_WINDOW_FIELD_NAMES)}"
            )
        _validate_positive_int(self.field_embedding_dim, name="field_embedding_dim")
        _validate_non_negative_int(self.pad_token_id, name="pad_token_id")
        object.__setattr__(self, "vocab_sizes", cleaned)

    @property
    def token_field_names(self) -> tuple[str, ...]:
        """Return the canonical ordered token-field names."""
        return TOKEN_WINDOW_FIELD_NAMES

    @property
    def concatenated_dim(self) -> int:
        """Return the dimensionality of concatenated field embeddings."""
        return self.field_embedding_dim * len(TOKEN_WINDOW_FIELD_NAMES)


@dataclass(frozen=True)
class MarketTransformerConfig:
    """Configuration for :class:`MarketTransformerEncoder`."""

    vocab_sizes: dict[str, int] = field(default_factory=_default_vocab_sizes)
    field_embedding_dim: int = 16
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    feedforward_dim: int = 128
    dropout: float = 0.1
    max_sequence_length: int = 128
    num_classes: int = 3
    pooling: PoolingMode = "mean"
    activation: ActivationName = "gelu"
    use_layer_norm: bool = True
    pad_token_id: int = 0

    def __post_init__(self) -> None:
        # Field-wise embedding configuration is validated by
        # ``TokenFieldEmbeddingConfig`` to keep the validation rules in one
        # place. Building it here also guarantees the vocab_sizes mapping is
        # well-formed before any tensors are allocated.
        TokenFieldEmbeddingConfig(
            vocab_sizes=dict(self.vocab_sizes),
            field_embedding_dim=self.field_embedding_dim,
            pad_token_id=self.pad_token_id,
        )
        _validate_positive_int(self.model_dim, name="model_dim")
        _validate_positive_int(self.num_heads, name="num_heads")
        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                "model_dim must be divisible by num_heads: "
                f"got model_dim={self.model_dim}, num_heads={self.num_heads}"
            )
        _validate_positive_int(self.num_layers, name="num_layers")
        _validate_positive_int(self.feedforward_dim, name="feedforward_dim")
        _validate_unit_interval(self.dropout, name="dropout")
        _validate_positive_int(self.max_sequence_length, name="max_sequence_length")
        _validate_positive_int(self.num_classes, name="num_classes")
        if self.pooling not in {"mean", "last", "bos"}:
            raise ValueError(
                "pooling must be 'mean', 'last' or 'bos'; "
                f"got {self.pooling!r}"
            )
        if self.activation not in {"relu", "gelu"}:
            raise ValueError(
                f"activation must be 'relu' or 'gelu'; got {self.activation!r}"
            )
        if not isinstance(self.use_layer_norm, bool):
            raise TypeError("use_layer_norm must be a bool")
        _validate_non_negative_int(self.pad_token_id, name="pad_token_id")

    @property
    def token_field_names(self) -> tuple[str, ...]:
        """Return the canonical ordered token-field names."""
        return TOKEN_WINDOW_FIELD_NAMES

    @property
    def n_token_fields(self) -> int:
        """Return the number of categorical token fields consumed."""
        return len(TOKEN_WINDOW_FIELD_NAMES)


@dataclass
class MarketTransformerOutput:
    """Outputs returned by :class:`MarketTransformerEncoder`."""

    logits: Any
    pooled: Any | None = None
    hidden_states: Any | None = None


def _resolve_activation(name: ActivationName) -> str:
    if name == "relu":
        return "relu"
    if name == "gelu":
        return "gelu"
    raise ValueError(f"unsupported activation {name!r}")


class TransformerPooling(_TORCH_MODULE_BASE):
    """Mask-aware pooling head used by :class:`MarketTransformerEncoder`.

    Supports mean pooling over real (non-padding) tokens, last-real-token
    pooling and BOS pooling. Fully padded sequences raise a clear error so
    callers can detect invalid batches early.
    """

    def __init__(self, mode: PoolingMode) -> None:
        _require_torch()
        super().__init__()
        if mode not in {"mean", "last", "bos"}:
            raise ValueError(
                f"mode must be 'mean', 'last' or 'bos'; got {mode!r}"
            )
        self._mode = mode

    @property
    def mode(self) -> PoolingMode:
        """Return the configured pooling mode."""
        return self._mode

    def forward(self, hidden_states: Any, attention_mask: Any) -> Any:
        """Return ``[batch, model_dim]`` pooled representations."""
        torch_module = _require_torch()
        if hidden_states.ndim != 3:
            raise ValueError(
                "hidden_states must be 3D with shape "
                "[batch, seq_len, model_dim]; "
                f"got shape {tuple(hidden_states.shape)}"
            )
        if attention_mask.ndim != 2:
            raise ValueError(
                "attention_mask must be 2D with shape [batch, seq_len]; "
                f"got shape {tuple(attention_mask.shape)}"
            )
        if attention_mask.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                "attention_mask shape must match hidden_states batch and "
                "sequence dimensions: "
                f"got attention_mask={tuple(attention_mask.shape)} vs "
                f"hidden_states={tuple(hidden_states.shape[:2])}"
            )
        if attention_mask.dtype != torch_module.bool:
            raise TypeError("attention_mask must be a boolean tensor")

        real_token_counts = attention_mask.sum(dim=1)
        if (real_token_counts == 0).any().item():
            raise ValueError(
                "pooling received at least one fully padded sequence; the "
                "attention_mask must contain at least one True per row"
            )

        if self._mode == "mean":
            mask_float = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
            masked = hidden_states * mask_float
            summed = masked.sum(dim=1)
            denom = real_token_counts.to(hidden_states.dtype).unsqueeze(-1)
            return summed / denom

        if self._mode == "last":
            batch_size = hidden_states.shape[0]
            # ``attention_mask`` has shape [batch, seq_len] with True for real
            # tokens. The last real token sits at the position whose
            # cumulative-True count equals the row's total real-token count.
            indices = real_token_counts.to(torch_module.long) - 1
            row_idx = torch_module.arange(
                batch_size,
                device=hidden_states.device,
            )
            return hidden_states[row_idx, indices, :]

        # bos pooling: take the first real token, which need not be position 0
        # in left-padded windows.
        batch_size = hidden_states.shape[0]
        first_real = attention_mask.to(torch_module.long).argmax(dim=1)
        row_idx = torch_module.arange(
            batch_size,
            device=hidden_states.device,
        )
        return hidden_states[row_idx, first_real, :]


class _LearnedPositionalEmbedding(_TORCH_MODULE_BASE):
    """Trainable absolute positional embedding."""

    def __init__(self, max_sequence_length: int, model_dim: int) -> None:
        _require_torch()
        super().__init__()
        _validate_positive_int(max_sequence_length, name="max_sequence_length")
        _validate_positive_int(model_dim, name="model_dim")
        self._max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(max_sequence_length, model_dim)

    def forward(self, hidden_states: Any) -> Any:
        torch_module = _require_torch()
        seq_len = int(hidden_states.shape[1])
        if seq_len > self._max_sequence_length:
            raise ValueError(
                "sequence length exceeds max_sequence_length: "
                f"{seq_len} > {self._max_sequence_length}"
            )
        positions = torch_module.arange(
            seq_len,
            device=hidden_states.device,
            dtype=torch_module.long,
        )
        return hidden_states + self.embedding(positions).unsqueeze(0)


class MarketTransformerEncoder(_TORCH_MODULE_BASE):
    """Transformer encoder over field-wise tokenised market microstructure.

    The forward pass expects a dictionary of long tensors with shape
    ``[batch, seq_len]`` for each token field listed in
    :data:`chronoslob.training.token_datasets.TOKEN_WINDOW_FIELD_NAMES`, plus
    a boolean ``attention_mask`` of the same shape where ``True`` marks a real
    token. The encoder produces ``[batch, num_classes]`` classification
    logits. It does not implement masked event modelling, next-event
    prediction or any self-supervised objective.
    """

    def __init__(self, config: MarketTransformerConfig) -> None:
        _require_torch()
        super().__init__()
        if not isinstance(config, MarketTransformerConfig):
            raise TypeError("config must be a MarketTransformerConfig instance")
        self._config = config

        embedding_config = TokenFieldEmbeddingConfig(
            vocab_sizes=dict(config.vocab_sizes),
            field_embedding_dim=config.field_embedding_dim,
            pad_token_id=config.pad_token_id,
        )
        self._embedding_config = embedding_config

        self.field_embeddings = nn.ModuleDict(
            {
                field_name: nn.Embedding(
                    embedding_config.vocab_sizes[field_name],
                    embedding_config.field_embedding_dim,
                    padding_idx=embedding_config.pad_token_id,
                )
                for field_name in TOKEN_WINDOW_FIELD_NAMES
            }
        )
        self.input_projection = nn.Linear(
            embedding_config.concatenated_dim,
            config.model_dim,
        )
        self.input_layer_norm: nn.Module | None = (
            nn.LayerNorm(config.model_dim) if config.use_layer_norm else None
        )
        self.positional_embedding = _LearnedPositionalEmbedding(
            config.max_sequence_length,
            config.model_dim,
        )
        self.input_dropout = nn.Dropout(p=float(config.dropout))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=float(config.dropout),
            activation=_resolve_activation(config.activation),
            batch_first=True,
            norm_first=False,
        )
        encoder_norm = (
            nn.LayerNorm(config.model_dim) if config.use_layer_norm else None
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=encoder_norm,
        )
        self.pooling = TransformerPooling(config.pooling)
        self.classifier = nn.Linear(config.model_dim, config.num_classes)

    @property
    def config(self) -> MarketTransformerConfig:
        """Return the validated configuration used to build the encoder."""
        return self._config

    @property
    def embedding_config(self) -> TokenFieldEmbeddingConfig:
        """Return the validated embedding configuration."""
        return self._embedding_config

    def n_parameters(self) -> int:
        """Return the total number of parameters in the model."""
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def _validate_inputs(
        self,
        inputs: Mapping[str, Any],
    ) -> tuple[Any, int, int]:
        torch_module = _require_torch()
        if not isinstance(inputs, Mapping):
            raise TypeError("inputs must be a mapping of token fields to tensors")
        if "attention_mask" not in inputs:
            raise KeyError("inputs is missing required key 'attention_mask'")

        attention_mask = inputs["attention_mask"]
        if not torch_module.is_tensor(attention_mask):
            raise TypeError("inputs['attention_mask'] must be a torch.Tensor")
        if attention_mask.ndim != 2:
            raise ValueError(
                "inputs['attention_mask'] must be 2D with shape "
                "[batch, seq_len]; "
                f"got shape {tuple(attention_mask.shape)}"
            )
        if attention_mask.dtype != torch_module.bool:
            raise TypeError("inputs['attention_mask'] must be a boolean tensor")

        batch_size, seq_len = int(attention_mask.shape[0]), int(attention_mask.shape[1])
        if seq_len > self._config.max_sequence_length:
            raise ValueError(
                "input sequence length exceeds max_sequence_length: "
                f"{seq_len} > {self._config.max_sequence_length}"
            )

        for field_name in TOKEN_WINDOW_FIELD_NAMES:
            if field_name not in inputs:
                raise KeyError(
                    f"inputs is missing required token field {field_name!r}"
                )
            tensor = inputs[field_name]
            if not torch_module.is_tensor(tensor):
                raise TypeError(
                    f"inputs[{field_name!r}] must be a torch.Tensor"
                )
            if tensor.ndim != 2:
                raise ValueError(
                    f"inputs[{field_name!r}] must be 2D with shape "
                    "[batch, seq_len]; "
                    f"got shape {tuple(tensor.shape)}"
                )
            if tensor.dtype != torch_module.long:
                raise TypeError(
                    f"inputs[{field_name!r}] must have dtype torch.long"
                )
            if (
                int(tensor.shape[0]) != batch_size
                or int(tensor.shape[1]) != seq_len
            ):
                raise ValueError(
                    f"inputs[{field_name!r}] shape {tuple(tensor.shape)} does "
                    "not match attention_mask shape "
                    f"{(batch_size, seq_len)}"
                )

        real_token_counts = attention_mask.sum(dim=1)
        if (real_token_counts == 0).any().item():
            raise ValueError(
                "encoder received at least one fully padded sequence; the "
                "attention_mask must contain at least one True per row"
            )

        return attention_mask, batch_size, seq_len

    def _embed_inputs(self, inputs: Mapping[str, Any]) -> Any:
        torch_module = _require_torch()
        field_embeddings = [
            self.field_embeddings[field_name](inputs[field_name])
            for field_name in TOKEN_WINDOW_FIELD_NAMES
        ]
        concatenated = torch_module.cat(field_embeddings, dim=-1)
        projected = self.input_projection(concatenated)
        if self.input_layer_norm is not None:
            projected = self.input_layer_norm(projected)
        projected = self.positional_embedding(projected)
        return self.input_dropout(projected)

    def forward(
        self,
        inputs: Mapping[str, Any],
        *,
        return_hidden_states: bool = False,
        return_pooled: bool = False,
    ) -> MarketTransformerOutput:
        """Run a forward pass over a field-wise token batch.

        ``inputs`` must contain every required token field as a
        ``LongTensor[batch, seq_len]`` plus a ``BoolTensor[batch, seq_len]``
        ``attention_mask`` (True marks real tokens). The Phase-11 attention
        mask convention is converted to PyTorch's ``src_key_padding_mask``
        internally (True marks padding).
        """
        attention_mask, _, _ = self._validate_inputs(inputs)

        embedded = self._embed_inputs(inputs)
        # PyTorch TransformerEncoder expects True = padding in
        # src_key_padding_mask; Phase 11 attention_mask uses True = real.
        key_padding_mask = ~attention_mask
        hidden_states = self.encoder(
            embedded,
            src_key_padding_mask=key_padding_mask,
        )
        pooled = self.pooling(hidden_states, attention_mask)
        logits = self.classifier(pooled)
        return MarketTransformerOutput(
            logits=logits,
            pooled=pooled if return_pooled else None,
            hidden_states=hidden_states if return_hidden_states else None,
        )

    def predict_logits(self, inputs: Mapping[str, Any]) -> Any:
        """Return logits in eval mode without computing gradients."""
        torch_module = _require_torch()
        was_training = self.training
        self.eval()
        try:
            with torch_module.no_grad():
                return self.forward(inputs).logits
        finally:
            if was_training:
                self.train()


def create_market_transformer(
    config: MarketTransformerConfig,
) -> MarketTransformerEncoder:
    """Construct a :class:`MarketTransformerEncoder` from a validated config.

    The resulting model is a small supervised encoder over field-wise
    tokenised market microstructure. It is not pretrained, does not
    implement any self-supervised objective and produces no benchmark or
    alpha claim. See ``reports/transformer_architecture.md`` for the full
    design and the explicit limitations.
    """
    return MarketTransformerEncoder(config)


# The unused ``SpecialToken`` import documents the reserved special token IDs
# the encoder relies on (``pad_token_id`` defaults to ``[PAD]=0``). Reference
# it here so static checkers do not strip the import even though the model
# only consumes integer IDs at runtime.
_RESERVED_SPECIAL_TOKEN = SpecialToken.PAD
_RESERVED_TOKEN_FIELDS = TOKEN_FIELDS
