"""Self-supervised pretraining model for normalised FI-2010 matrix windows.

This module implements a self-supervised wrapper over the *same* transformer
encoder used by :class:`chronoslob.models.matrix_transformer.MatrixTransformerClassifier`.
The encoder submodules (``input_projection``, ``position_embedding`` and
``encoder``) are constructed identically so that a pretrained encoder can be
transferred byte-for-byte into a supervised classifier of identical
architecture for fine-tuning.

Two self-supervised objectives are supported over continuous, train-only
standardised feature windows:

* **Masked-field modelling** randomly masks selected ``(position, channel)``
  entries of the input window and reconstructs the original standardised
  values via a per-position linear head (mean squared error over masked
  entries only). The mask probability is configurable.
* **Next-field prediction** predicts the discretised bucket of every feature
  at position ``t + 1`` from the hidden state at position ``t`` (cross-entropy
  over train-only quantile buckets). The final window position has no
  successor and is ignored.

The module implements only the model wrapper and loss computation. Masking,
train-only bucket-edge fitting and dataset assembly live in
:mod:`chronoslob.training.matrix_ssl_datasets`; the pretraining loop and
artefact checkpointing live in
:mod:`chronoslob.training.matrix_ssl_experiment`. Nothing here makes a
market-performance, profitability or benchmark-ranking claim.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

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

from chronoslob.models.matrix_transformer import (
    MatrixTransformerClassifier,
    MatrixTransformerConfig,
)

__all__ = [
    "TRANSFERABLE_ENCODER_PREFIXES",
    "MatrixSSLConfig",
    "MatrixSSLModel",
    "MatrixSSLObjective",
    "MatrixSSLOutput",
    "create_matrix_ssl_model",
    "load_encoder_state_into_classifier",
]

# State-dict key prefixes shared by the SSL model and the supervised
# classifier. Only these are transferred from a pretrained encoder into a
# fine-tuning classifier; the supervised classification head is never seeded
# from self-supervised pretraining.
TRANSFERABLE_ENCODER_PREFIXES: tuple[str, ...] = (
    "input_projection.",
    "position_embedding",
    "encoder.",
)


class MatrixSSLObjective(StrEnum):
    """Stable identifiers for the matrix SSL objectives."""

    MASKED_FIELD = "masked_field"
    NEXT_FIELD = "next_field"


MATRIX_SSL_OBJECTIVE_NAMES: tuple[str, ...] = tuple(
    item.value for item in MatrixSSLObjective
)


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the matrix SSL model. Install the 'torch' "
            "optional dependency: pip install -e '.[torch]'"
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


def _validate_non_negative_float(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if numeric < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _validate_finite_float(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


@dataclass(frozen=True)
class MatrixSSLConfig:
    """Configuration for :class:`MatrixSSLModel`.

    The transformer architecture fields mirror
    :class:`~chronoslob.models.matrix_transformer.MatrixTransformerConfig` so a
    pretrained encoder can be transferred into a supervised classifier with
    identical architecture for fine-tuning.
    """

    input_features: int
    model_dim: int = 16
    num_heads: int = 2
    num_layers: int = 1
    feedforward_dim: int = 32
    dropout: float = 0.0
    max_sequence_length: int = 4
    enable_masked_field: bool = True
    enable_next_field: bool = True
    mask_probability: float = 0.15
    mask_value: float = 0.0
    next_field_bucket_count: int = 3
    masked_loss_weight: float = 1.0
    next_loss_weight: float = 1.0
    ignore_index: int = -100

    def __post_init__(self) -> None:
        _validate_positive_int(self.input_features, name="input_features")
        _validate_positive_int(self.model_dim, name="model_dim")
        _validate_positive_int(self.num_heads, name="num_heads")
        _validate_positive_int(self.num_layers, name="num_layers")
        _validate_positive_int(self.feedforward_dim, name="feedforward_dim")
        _validate_positive_int(
            self.max_sequence_length, name="max_sequence_length"
        )
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if isinstance(self.dropout, bool) or not isinstance(
            self.dropout, (int, float)
        ):
            raise TypeError("dropout must be a float")
        if not (0.0 <= float(self.dropout) < 1.0):
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        for flag_name in ("enable_masked_field", "enable_next_field"):
            if not isinstance(getattr(self, flag_name), bool):
                raise TypeError(f"{flag_name} must be a bool")
        if not (self.enable_masked_field or self.enable_next_field):
            raise ValueError("at least one SSL objective must be enabled")
        _validate_unit_interval(self.mask_probability, name="mask_probability")
        _validate_finite_float(self.mask_value, name="mask_value")
        _validate_positive_int(
            self.next_field_bucket_count, name="next_field_bucket_count"
        )
        if self.next_field_bucket_count < 2:
            raise ValueError("next_field_bucket_count must be >= 2")
        _validate_non_negative_float(
            self.masked_loss_weight, name="masked_loss_weight"
        )
        _validate_non_negative_float(
            self.next_loss_weight, name="next_loss_weight"
        )
        if not isinstance(self.ignore_index, int) or isinstance(
            self.ignore_index, bool
        ):
            raise TypeError("ignore_index must be an integer")
        if self.enable_masked_field and self.mask_probability <= 0.0:
            raise ValueError(
                "mask_probability must be positive when the masked-field "
                "objective is enabled"
            )
        enabled_weight = 0.0
        if self.enable_masked_field:
            enabled_weight += float(self.masked_loss_weight)
        if self.enable_next_field:
            enabled_weight += float(self.next_loss_weight)
        if enabled_weight <= 0.0:
            raise ValueError(
                "at least one enabled SSL objective must have a positive loss "
                "weight"
            )

    def enabled_objectives(self) -> tuple[str, ...]:
        """Return the enabled objective names in stable order."""
        names: list[str] = []
        if self.enable_masked_field:
            names.append(MatrixSSLObjective.MASKED_FIELD.value)
        if self.enable_next_field:
            names.append(MatrixSSLObjective.NEXT_FIELD.value)
        return tuple(names)

    def to_transformer_config(self, *, n_classes: int) -> MatrixTransformerConfig:
        """Build the matching supervised classifier architecture config."""
        return MatrixTransformerConfig(
            input_features=self.input_features,
            n_classes=n_classes,
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            feedforward_dim=self.feedforward_dim,
            dropout=self.dropout,
            max_sequence_length=self.max_sequence_length,
        )


@dataclass
class MatrixSSLOutput:
    """Structured output returned by :class:`MatrixSSLModel`."""

    loss: Any | None
    loss_components: dict[str, Any]
    masked_reconstruction: Any | None
    next_logits: Any | None
    hidden_states: Any


class MatrixSSLModel(_TORCH_MODULE_BASE):
    """Self-supervised transformer over normalised FI-2010 matrix windows.

    The backbone (``input_projection``, ``position_embedding``, ``encoder``) is
    constructed exactly as in
    :class:`~chronoslob.models.matrix_transformer.MatrixTransformerClassifier`,
    so :meth:`encoder_state_dict` yields a state dict that loads cleanly into a
    supervised classifier of identical architecture.
    """

    def __init__(self, config: MatrixSSLConfig) -> None:
        _require_torch()
        super().__init__()
        if not isinstance(config, MatrixSSLConfig):
            raise TypeError("config must be a MatrixSSLConfig instance")
        self._config = config

        # Constructed identically to MatrixTransformerClassifier so that the
        # parameter names and shapes match for encoder transfer.
        self.input_projection = nn.Linear(config.input_features, config.model_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.max_sequence_length, config.model_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        self.masked_head: Any = (
            nn.Linear(config.model_dim, config.input_features)
            if config.enable_masked_field
            else None
        )
        self.next_head: Any = (
            nn.Linear(
                config.model_dim,
                config.input_features * config.next_field_bucket_count,
            )
            if config.enable_next_field
            else None
        )

    @property
    def config(self) -> MatrixSSLConfig:
        """Return the validated SSL configuration."""
        return self._config

    def n_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def encode(self, x: Any) -> Any:
        """Return per-position hidden states ``[batch, window, model_dim]``."""
        torch_module = _require_torch()
        if not torch_module.is_tensor(x):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError(
                "x must be 3D with shape [batch, window, features]; "
                f"got shape {tuple(x.shape)}"
            )
        if int(x.shape[-1]) != self._config.input_features:
            raise ValueError(
                "x feature dimension does not match config.input_features: "
                f"expected {self._config.input_features}, got {int(x.shape[-1])}"
            )
        window_length = int(x.shape[1])
        if window_length <= 0:
            raise ValueError("x window dimension must be positive")
        if window_length > self._config.max_sequence_length:
            raise ValueError(
                "x window dimension exceeds max_sequence_length: "
                f"{window_length} > {self._config.max_sequence_length}"
            )
        hidden = self.input_projection(x)
        hidden = hidden + self.position_embedding[:, :window_length, :]
        return self.encoder(hidden)

    def forward(
        self,
        x: Any,
        *,
        mask: Any | None = None,
        masked_target: Any | None = None,
        next_bucket_labels: Any | None = None,
    ) -> MatrixSSLOutput:
        """Run a self-supervised forward pass.

        ``x`` is the (possibly masked) standardised feature window of shape
        ``[batch, window, features]``. ``mask`` and ``masked_target`` drive the
        masked-field reconstruction loss; ``next_bucket_labels`` drives the
        next-field bucket-prediction loss. Positions equal to ``ignore_index``
        in the labels are dropped from the next-field cross-entropy.
        """
        torch_module = _require_torch()
        hidden = self.encode(x)

        masked_reconstruction: Any | None = None
        next_logits: Any | None = None
        loss_components: dict[str, Any] = {}
        total_loss: Any | None = None

        compute_loss = (
            mask is not None
            or masked_target is not None
            or next_bucket_labels is not None
        )
        if compute_loss:
            total_loss = torch_module.zeros((), device=hidden.device)

        if self._config.enable_masked_field and self.masked_head is not None:
            masked_reconstruction = self.masked_head(hidden)
            if mask is not None or masked_target is not None:
                if mask is None or masked_target is None:
                    raise ValueError(
                        "masked-field objective requires both mask and "
                        "masked_target"
                    )
                masked_loss = self._masked_loss(
                    reconstruction=masked_reconstruction,
                    target=masked_target,
                    mask=mask,
                )
                loss_components[MatrixSSLObjective.MASKED_FIELD.value] = masked_loss
                total_loss = total_loss + (
                    float(self._config.masked_loss_weight) * masked_loss
                )

        if self._config.enable_next_field and self.next_head is not None:
            next_logits = self.next_head(hidden)
            if next_bucket_labels is not None:
                next_loss = self._next_loss(
                    logits=next_logits,
                    labels=next_bucket_labels,
                )
                loss_components[MatrixSSLObjective.NEXT_FIELD.value] = next_loss
                total_loss = total_loss + (
                    float(self._config.next_loss_weight) * next_loss
                )

        if compute_loss and not loss_components:
            raise ValueError(
                "no enabled SSL objective produced a valid target; check the "
                "masking and next-field target construction"
            )

        return MatrixSSLOutput(
            loss=total_loss,
            loss_components=loss_components,
            masked_reconstruction=masked_reconstruction,
            next_logits=next_logits,
            hidden_states=hidden,
        )

    def _masked_loss(self, *, reconstruction: Any, target: Any, mask: Any) -> Any:
        torch_module = _require_torch()
        if not torch_module.is_tensor(target):
            raise TypeError("masked_target must be a torch.Tensor")
        if not torch_module.is_tensor(mask):
            raise TypeError("mask must be a torch.Tensor")
        if reconstruction.shape != target.shape:
            raise ValueError(
                "masked reconstruction and target shapes differ: "
                f"{tuple(reconstruction.shape)} != {tuple(target.shape)}"
            )
        if mask.shape != target.shape:
            raise ValueError(
                "mask and target shapes differ: "
                f"{tuple(mask.shape)} != {tuple(target.shape)}"
            )
        bool_mask = mask.to(torch_module.bool)
        selected = int(bool_mask.sum().item())
        if selected == 0:
            raise ValueError(
                "masked-field objective received no masked entries; ensure the "
                "masking policy forces at least one masked position"
            )
        diff = (reconstruction - target) * bool_mask.to(reconstruction.dtype)
        squared = diff * diff
        return squared.sum() / float(selected)

    def _next_loss(self, *, logits: Any, labels: Any) -> Any:
        torch_module = _require_torch()
        if not torch_module.is_tensor(labels):
            raise TypeError("next_bucket_labels must be a torch.Tensor")
        if labels.dtype != torch_module.long:
            raise TypeError("next_bucket_labels must have dtype torch.long")
        bucket_count = int(self._config.next_field_bucket_count)
        feature_count = int(self._config.input_features)
        batch_size = int(logits.shape[0])
        window_length = int(logits.shape[1])
        if tuple(labels.shape) != (batch_size, window_length, feature_count):
            raise ValueError(
                "next_bucket_labels shape "
                f"{tuple(labels.shape)} does not match expected "
                f"{(batch_size, window_length, feature_count)}"
            )
        reshaped = logits.reshape(
            batch_size, window_length, feature_count, bucket_count
        )
        flat_logits = reshaped.reshape(-1, bucket_count)
        flat_labels = labels.reshape(-1)
        valid = int((flat_labels != self._config.ignore_index).sum().item())
        if valid == 0:
            raise ValueError(
                "next-field objective received no valid (non-ignore_index) "
                "target positions"
            )
        return torch_functional.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=self._config.ignore_index,
            reduction="mean",
        )

    def encoder_state_dict(self) -> dict[str, Any]:
        """Return the transferable encoder parameters as a CPU state dict.

        The returned keys (``input_projection.*``, ``position_embedding`` and
        ``encoder.*``) match the corresponding keys in
        :class:`~chronoslob.models.matrix_transformer.MatrixTransformerClassifier`.
        """
        return {
            str(key): value.detach().cpu().clone()
            for key, value in self.state_dict().items()
            if str(key).startswith(TRANSFERABLE_ENCODER_PREFIXES)
        }


def create_matrix_ssl_model(config: MatrixSSLConfig) -> MatrixSSLModel:
    """Construct a :class:`MatrixSSLModel` from a validated config."""
    return MatrixSSLModel(config)


def load_encoder_state_into_classifier(
    classifier: MatrixTransformerClassifier,
    encoder_state: Mapping[str, Any],
) -> list[str]:
    """Load a pretrained encoder state dict into a supervised classifier.

    Only the shared encoder keys are transferred; the classification head is
    left at its (freshly initialised) values. Returns the sorted list of keys
    that were loaded. Raises ``ValueError`` if a transferable key is missing
    from the classifier or has a mismatched shape, or if no encoder key was
    transferred at all.
    """
    if not isinstance(classifier, MatrixTransformerClassifier):
        raise TypeError("classifier must be a MatrixTransformerClassifier")
    if not isinstance(encoder_state, Mapping):
        raise TypeError("encoder_state must be a mapping")

    target_state = classifier.state_dict()
    loaded: list[str] = []
    for key, value in encoder_state.items():
        key_name = str(key)
        if not key_name.startswith(TRANSFERABLE_ENCODER_PREFIXES):
            continue
        if key_name not in target_state:
            raise ValueError(
                f"pretrained encoder key {key_name!r} is absent from the "
                "fine-tuning classifier; architectures differ"
            )
        if tuple(target_state[key_name].shape) != tuple(value.shape):
            raise ValueError(
                f"pretrained encoder key {key_name!r} shape "
                f"{tuple(value.shape)} does not match classifier shape "
                f"{tuple(target_state[key_name].shape)}"
            )
        target_state[key_name] = value
        loaded.append(key_name)
    if not loaded:
        raise ValueError(
            "no transferable encoder keys were found in the pretrained state; "
            "expected keys prefixed with "
            f"{list(TRANSFERABLE_ENCODER_PREFIXES)}"
        )
    classifier.load_state_dict(target_state, strict=True)
    return sorted(loaded)
