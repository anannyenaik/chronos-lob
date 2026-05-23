"""Self-supervised pretraining objectives for tokenised market microstructure.

Phase 13 wraps the Phase-12 :class:`MarketTransformerEncoder` with
self-supervised objectives over the field-wise token batches prepared in
Phase 11. The currently supported objectives are:

* Masked field modelling: predict the original token IDs of selected
  fields at positions corrupted by deterministic BERT-style masking.
* Next-field prediction: given the hidden state at position ``t``,
  predict the original token IDs of selected fields at position ``t + 1``.

A contrastive objective is intentionally deferred to a later phase. The
configuration accepts ``enable_contrastive_loss`` for forward compatibility
but rejects an enabled flag in :func:`SSLTransformerConfig.__post_init__`.

This module implements only the SSL model wrapper and loss computation. It
does not implement masking helpers (see :mod:`chronoslob.training.ssl_datasets`),
calibration, supervised multi-task fine-tuning, execution simulation,
backtesting or any market-performance claim.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

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
    import torch as _torch_typing  # noqa: F401

__all__ = [
    "DEFAULT_MASKED_FIELDS",
    "DEFAULT_NEXT_FIELDS",
    "MarketSSLTransformer",
    "MaskingConfig",
    "SSLObjectiveName",
    "SSLTransformerConfig",
    "SSLTransformerOutput",
    "create_ssl_transformer",
]


class SSLObjectiveName(StrEnum):
    """Stable identifiers for the self-supervised objectives."""

    MASKED_FIELD = "masked_field"
    NEXT_FIELD = "next_field"
    CONTRASTIVE = "contrastive"


SSL_OBJECTIVE_NAMES: tuple[str, ...] = tuple(name.value for name in SSLObjectiveName)

DEFAULT_MASKED_FIELDS: tuple[str, ...] = (
    "event_type",
    "side",
    "price_bucket",
    "quantity_bucket",
)
DEFAULT_NEXT_FIELDS: tuple[str, ...] = (
    "event_type",
    "side",
    "price_bucket",
    "quantity_bucket",
)


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the SSL transformer wrapper. Install "
            "the 'torch' optional dependency: pip install -e '.[torch]'"
        )
    return torch


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
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


def _validate_positive_float(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return numeric


def _validate_non_negative_float(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _validate_field_tuple(
    fields: tuple[str, ...],
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
class MaskingConfig:
    """Configuration for deterministic BERT-style field masking.

    The default values follow Devlin et al. (BERT): 15% of valid (non
    padding) positions are selected for masking and within those, 80% are
    replaced with ``[MASK]``, 10% with a random non-special token from the
    same field vocabulary, and 10% are kept unchanged. ``[PAD]`` positions
    are never selected.
    """

    mask_probability: float = 0.15
    mask_token_probability: float = 0.8
    random_token_probability: float = 0.1
    keep_token_probability: float = 0.1
    force_at_least_one_mask: bool = True

    def __post_init__(self) -> None:
        _validate_unit_interval(self.mask_probability, name="mask_probability")
        _validate_unit_interval(
            self.mask_token_probability, name="mask_token_probability"
        )
        _validate_unit_interval(
            self.random_token_probability, name="random_token_probability"
        )
        _validate_unit_interval(
            self.keep_token_probability, name="keep_token_probability"
        )
        total = float(
            self.mask_token_probability
            + self.random_token_probability
            + self.keep_token_probability
        )
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(
                "mask_token_probability, random_token_probability and "
                "keep_token_probability must sum to 1.0; "
                f"got sum={total}"
            )
        if not isinstance(self.force_at_least_one_mask, bool):
            raise TypeError("force_at_least_one_mask must be a bool")


@dataclass
class SSLTransformerOutput:
    """Structured output returned by :class:`MarketSSLTransformer`."""

    loss: Any | None
    loss_components: dict[str, Any]
    masked_logits: dict[str, Any]
    next_logits: dict[str, Any]
    pooled: Any | None
    hidden_states: Any


def _normalise_loss_weights(
    weights: Mapping[str, float],
) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for key, value in weights.items():
        if key not in SSL_OBJECTIVE_NAMES:
            raise ValueError(
                f"loss_weights contains unknown objective {key!r}; "
                f"expected one of {list(SSL_OBJECTIVE_NAMES)}"
            )
        cleaned[key] = _validate_non_negative_float(
            float(value), name=f"loss_weights[{key!r}]"
        )
    for objective in SSL_OBJECTIVE_NAMES:
        cleaned.setdefault(objective, 0.0)
    return cleaned


@dataclass(frozen=True)
class SSLTransformerConfig:
    """Configuration for :class:`MarketSSLTransformer`.

    Wraps a :class:`MarketTransformerConfig` and adds SSL-specific knobs.
    The contrastive objective is currently deferred: setting
    ``enable_contrastive_loss=True`` raises a clear error.
    """

    transformer: MarketTransformerConfig = field(default_factory=MarketTransformerConfig)
    masked_fields: tuple[str, ...] = DEFAULT_MASKED_FIELDS
    next_fields: tuple[str, ...] = DEFAULT_NEXT_FIELDS
    enable_masked_field_loss: bool = True
    enable_next_field_loss: bool = True
    enable_contrastive_loss: bool = False
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    loss_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            SSLObjectiveName.MASKED_FIELD.value: 1.0,
            SSLObjectiveName.NEXT_FIELD.value: 1.0,
            SSLObjectiveName.CONTRASTIVE.value: 0.0,
        }
    )
    contrastive_temperature: float = 0.07
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if not isinstance(self.transformer, MarketTransformerConfig):
            raise TypeError("transformer must be a MarketTransformerConfig instance")
        if not isinstance(self.masking, MaskingConfig):
            raise TypeError("masking must be a MaskingConfig instance")
        for flag_name in (
            "enable_masked_field_loss",
            "enable_next_field_loss",
            "enable_contrastive_loss",
        ):
            if not isinstance(getattr(self, flag_name), bool):
                raise TypeError(f"{flag_name} must be a bool")
        masked = _validate_field_tuple(
            tuple(self.masked_fields), name="masked_fields"
        )
        nxt = _validate_field_tuple(tuple(self.next_fields), name="next_fields")
        object.__setattr__(self, "masked_fields", masked)
        object.__setattr__(self, "next_fields", nxt)
        if self.enable_masked_field_loss and not masked:
            raise ValueError(
                "enable_masked_field_loss=True requires at least one entry in "
                "masked_fields"
            )
        if self.enable_next_field_loss and not nxt:
            raise ValueError(
                "enable_next_field_loss=True requires at least one entry in "
                "next_fields"
            )
        if self.enable_contrastive_loss:
            raise NotImplementedError(
                "The contrastive SSL objective is deferred in Phase 13; set "
                "enable_contrastive_loss=False. See "
                "reports/self_supervised_objectives.md for the rationale."
            )

        weights = _normalise_loss_weights(self.loss_weights)
        object.__setattr__(self, "loss_weights", weights)

        if not (
            self.enable_masked_field_loss
            or self.enable_next_field_loss
            or self.enable_contrastive_loss
        ):
            raise ValueError(
                "at least one SSL objective must be enabled; got all flags False"
            )

        enabled_weight_sum = 0.0
        if self.enable_masked_field_loss:
            enabled_weight_sum += weights[SSLObjectiveName.MASKED_FIELD.value]
        if self.enable_next_field_loss:
            enabled_weight_sum += weights[SSLObjectiveName.NEXT_FIELD.value]
        if self.enable_contrastive_loss:
            enabled_weight_sum += weights[SSLObjectiveName.CONTRASTIVE.value]
        if enabled_weight_sum <= 0.0:
            raise ValueError(
                "at least one enabled SSL objective must have a positive loss "
                "weight"
            )

        for missing_field in self.masked_fields:
            if missing_field not in self.transformer.vocab_sizes:
                raise ValueError(
                    f"masked field {missing_field!r} has no vocabulary size in "
                    "transformer.vocab_sizes"
                )
        for missing_field in self.next_fields:
            if missing_field not in self.transformer.vocab_sizes:
                raise ValueError(
                    f"next field {missing_field!r} has no vocabulary size in "
                    "transformer.vocab_sizes"
                )

        if self.enable_contrastive_loss:  # pragma: no cover - guarded above
            _validate_positive_float(
                float(self.contrastive_temperature), name="contrastive_temperature"
            )
        if not isinstance(self.ignore_index, int) or isinstance(
            self.ignore_index, bool
        ):
            raise TypeError("ignore_index must be an integer")

    def enabled_objectives(self) -> tuple[str, ...]:
        """Return the names of the SSL objectives that are enabled."""
        enabled: list[str] = []
        if self.enable_masked_field_loss:
            enabled.append(SSLObjectiveName.MASKED_FIELD.value)
        if self.enable_next_field_loss:
            enabled.append(SSLObjectiveName.NEXT_FIELD.value)
        if self.enable_contrastive_loss:  # pragma: no cover - guarded above
            enabled.append(SSLObjectiveName.CONTRASTIVE.value)
        return tuple(enabled)


def _check_label_tensor(
    tensor: Any,
    *,
    field_name: str,
    batch_size: int,
    seq_len: int,
) -> None:
    torch_module = _require_torch()
    if not torch_module.is_tensor(tensor):
        raise TypeError(f"label for field {field_name!r} must be a torch.Tensor")
    if tensor.dtype != torch_module.long:
        raise TypeError(
            f"label for field {field_name!r} must have dtype torch.long"
        )
    if tensor.ndim != 2:
        raise ValueError(
            f"label for field {field_name!r} must be 2D [batch, seq_len]; "
            f"got shape {tuple(tensor.shape)}"
        )
    if int(tensor.shape[0]) != batch_size or int(tensor.shape[1]) != seq_len:
        raise ValueError(
            f"label for field {field_name!r} shape {tuple(tensor.shape)} does "
            f"not match inputs shape {(batch_size, seq_len)}"
        )


class MarketSSLTransformer(_TORCH_MODULE_BASE):
    """Self-supervised wrapper around :class:`MarketTransformerEncoder`.

    The wrapper owns one :class:`MarketTransformerEncoder` for the shared
    backbone and one linear head per configured masked or next field. The
    encoder's classification head is retained for API symmetry but is not
    used by the SSL forward.
    """

    def __init__(self, config: SSLTransformerConfig) -> None:
        _require_torch()
        super().__init__()
        if not isinstance(config, SSLTransformerConfig):
            raise TypeError("config must be a SSLTransformerConfig instance")
        self._config = config

        self.encoder = MarketTransformerEncoder(config.transformer)

        model_dim = int(config.transformer.model_dim)
        masked_heads: dict[str, nn.Module] = {}
        if config.enable_masked_field_loss:
            for field_name in config.masked_fields:
                vocab_size = int(config.transformer.vocab_sizes[field_name])
                masked_heads[field_name] = nn.Linear(model_dim, vocab_size)
        self.masked_heads = nn.ModuleDict(masked_heads)

        next_heads: dict[str, nn.Module] = {}
        if config.enable_next_field_loss:
            for field_name in config.next_fields:
                vocab_size = int(config.transformer.vocab_sizes[field_name])
                next_heads[field_name] = nn.Linear(model_dim, vocab_size)
        self.next_heads = nn.ModuleDict(next_heads)

    @property
    def config(self) -> SSLTransformerConfig:
        """Return the validated configuration used to build the SSL model."""
        return self._config

    def n_parameters(self) -> int:
        """Return the total number of trainable parameters in the model."""
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def _encode(
        self,
        inputs: Mapping[str, Any],
        *,
        return_pooled: bool,
    ) -> tuple[Any, Any | None]:
        encoder_output = self.encoder(
            inputs,
            return_hidden_states=True,
            return_pooled=return_pooled,
        )
        return encoder_output.hidden_states, encoder_output.pooled

    def forward(
        self,
        inputs: Mapping[str, Any],
        *,
        masked_labels: Mapping[str, Any] | None = None,
        next_labels: Mapping[str, Any] | None = None,
        return_pooled: bool = False,
    ) -> SSLTransformerOutput:
        """Run a self-supervised forward pass.

        ``inputs`` must contain every required token field plus
        ``attention_mask``, matching the Phase-12 encoder contract. When
        ``masked_labels`` or ``next_labels`` are provided the corresponding
        objective contributes to the combined loss; positions equal to
        ``ignore_index`` are dropped from the loss by
        :func:`torch.nn.functional.cross_entropy`.
        """
        torch_module = _require_torch()
        if not isinstance(inputs, Mapping):
            raise TypeError("inputs must be a mapping")

        hidden_states, pooled = self._encode(inputs, return_pooled=return_pooled)
        batch_size = int(hidden_states.shape[0])
        seq_len = int(hidden_states.shape[1])

        masked_logits: dict[str, Any] = {}
        next_logits: dict[str, Any] = {}
        loss_components: dict[str, Any] = {}

        if self._config.enable_masked_field_loss:
            for field_name in self._config.masked_fields:
                head = self.masked_heads[field_name]
                masked_logits[field_name] = head(hidden_states)

        if self._config.enable_next_field_loss:
            for field_name in self._config.next_fields:
                head = self.next_heads[field_name]
                next_logits[field_name] = head(hidden_states)

        total_loss: Any | None = None
        if masked_labels is not None or next_labels is not None:
            total_loss = torch_module.zeros((), device=hidden_states.device)

            if masked_labels is not None:
                if not isinstance(masked_labels, Mapping):
                    raise TypeError("masked_labels must be a mapping or None")
                if not self._config.enable_masked_field_loss:
                    raise ValueError(
                        "masked_labels was supplied but enable_masked_field_loss "
                        "is False"
                    )
                masked_loss = self._compute_masked_loss(
                    masked_logits=masked_logits,
                    masked_labels=masked_labels,
                    batch_size=batch_size,
                    seq_len=seq_len,
                )
                loss_components[SSLObjectiveName.MASKED_FIELD.value] = masked_loss
                weight = float(
                    self._config.loss_weights[SSLObjectiveName.MASKED_FIELD.value]
                )
                total_loss = total_loss + weight * masked_loss

            if next_labels is not None:
                if not isinstance(next_labels, Mapping):
                    raise TypeError("next_labels must be a mapping or None")
                if not self._config.enable_next_field_loss:
                    raise ValueError(
                        "next_labels was supplied but enable_next_field_loss is "
                        "False"
                    )
                next_loss = self._compute_next_loss(
                    next_logits=next_logits,
                    next_labels=next_labels,
                    batch_size=batch_size,
                    seq_len=seq_len,
                )
                loss_components[SSLObjectiveName.NEXT_FIELD.value] = next_loss
                weight = float(
                    self._config.loss_weights[SSLObjectiveName.NEXT_FIELD.value]
                )
                total_loss = total_loss + weight * next_loss

            if not loss_components:
                raise ValueError(
                    "no enabled SSL objective produced any valid target positions; "
                    "check masking and next-field target construction"
                )

        return SSLTransformerOutput(
            loss=total_loss,
            loss_components=loss_components,
            masked_logits=masked_logits,
            next_logits=next_logits,
            pooled=pooled,
            hidden_states=hidden_states,
        )

    def _compute_masked_loss(
        self,
        *,
        masked_logits: Mapping[str, Any],
        masked_labels: Mapping[str, Any],
        batch_size: int,
        seq_len: int,
    ) -> Any:
        torch_module = _require_torch()
        active_losses: list[Any] = []
        for field_name in self._config.masked_fields:
            if field_name not in masked_labels:
                continue
            labels = masked_labels[field_name]
            _check_label_tensor(
                labels,
                field_name=field_name,
                batch_size=batch_size,
                seq_len=seq_len,
            )
            logits = masked_logits[field_name]
            vocab_size = int(logits.shape[-1])
            flat_logits = logits.reshape(-1, vocab_size)
            flat_labels = labels.reshape(-1)
            valid = (flat_labels != self._config.ignore_index).sum().item()
            if int(valid) == 0:
                continue
            loss = torch_functional.cross_entropy(
                flat_logits,
                flat_labels,
                ignore_index=self._config.ignore_index,
                reduction="mean",
            )
            active_losses.append(loss)
        if not active_losses:
            raise ValueError(
                "masked field objective received no valid (non-ignore_index) "
                "target positions"
            )
        stacked = torch_module.stack(active_losses)
        return stacked.mean()

    def _compute_next_loss(
        self,
        *,
        next_logits: Mapping[str, Any],
        next_labels: Mapping[str, Any],
        batch_size: int,
        seq_len: int,
    ) -> Any:
        torch_module = _require_torch()
        active_losses: list[Any] = []
        for field_name in self._config.next_fields:
            if field_name not in next_labels:
                continue
            labels = next_labels[field_name]
            _check_label_tensor(
                labels,
                field_name=field_name,
                batch_size=batch_size,
                seq_len=seq_len,
            )
            logits = next_logits[field_name]
            vocab_size = int(logits.shape[-1])
            flat_logits = logits.reshape(-1, vocab_size)
            flat_labels = labels.reshape(-1)
            valid = (flat_labels != self._config.ignore_index).sum().item()
            if int(valid) == 0:
                continue
            loss = torch_functional.cross_entropy(
                flat_logits,
                flat_labels,
                ignore_index=self._config.ignore_index,
                reduction="mean",
            )
            active_losses.append(loss)
        if not active_losses:
            raise ValueError(
                "next field objective received no valid (non-ignore_index) "
                "target positions"
            )
        stacked = torch_module.stack(active_losses)
        return stacked.mean()


def create_ssl_transformer(config: SSLTransformerConfig) -> MarketSSLTransformer:
    """Construct a :class:`MarketSSLTransformer` from a validated config.

    The returned wrapper is a self-supervised pretraining module over
    field-wise tokenised market microstructure. It implements masked field
    modelling and next-field prediction only; the contrastive objective is
    deferred to a later phase. The wrapper produces no benchmark or
    market-performance claim. See ``reports/self_supervised_objectives.md``
    for the full design and the explicit limitations.
    """
    return MarketSSLTransformer(config)


# Reference imports that document the reserved special-token IDs the SSL
# wrapper relies on. ``[MASK]`` is the token ID used by the masking helper
# in :mod:`chronoslob.training.ssl_datasets`; ``[PAD]`` defines the
# padding-id contract preserved end to end.
_RESERVED_MASK_TOKEN = SpecialToken.MASK
_RESERVED_PAD_TOKEN = SpecialToken.PAD
_RESERVED_SPECIAL_TOKEN_IDS = SPECIAL_TOKEN_IDS
_RESERVED_TOKEN_FIELDS = TOKEN_FIELDS
