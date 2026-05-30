"""Second-generation matrix SSL objective for FI-2010 windows.

The v2 objective keeps the encoder architecture byte-compatible with
``MatrixTransformerClassifier`` while changing the pretraining signal from
random field reconstruction / next-field prediction to a market-state-aware
multi-task objective:

* structured group masking with reconstruction over masked entries;
* future auxiliary heads for spread widening, volatility, return and imbalance;
* an optional regime contrastive term.

The module is deliberately claim-neutral. It implements the objective and
emits loss components; downstream reports decide what, if anything, the
resulting evidence supports.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from chronoslob.models.matrix_ssl import TRANSFERABLE_ENCODER_PREFIXES
from chronoslob.models.matrix_transformer import (
    MatrixTransformerClassifier,
    MatrixTransformerConfig,
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
    "MATRIX_SSL_V2_FUTURE_TASKS",
    "MATRIX_SSL_V2_LOSS_COMPONENTS",
    "MatrixSSLV2Config",
    "MatrixSSLV2Model",
    "MatrixSSLV2Output",
    "create_matrix_ssl_v2_model",
]

MATRIX_SSL_V2_FUTURE_TASKS: tuple[str, ...] = (
    "future_spread_widening",
    "future_volatility",
    "future_return",
    "future_imbalance",
)
MATRIX_SSL_V2_LOSS_COMPONENTS: tuple[str, ...] = (
    "reconstruction",
    *MATRIX_SSL_V2_FUTURE_TASKS,
    "contrastive",
    "total",
)


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the matrix SSL-v2 model. Install the 'torch' "
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


def _validate_positive_float(value: float, *, name: str) -> float:
    numeric = _validate_non_negative_float(value, name=name)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    return numeric


@dataclass(frozen=True)
class MatrixSSLV2Config:
    """Configuration for the market-state-aware SSL-v2 model."""

    input_features: int
    model_dim: int = 16
    num_heads: int = 2
    num_layers: int = 1
    feedforward_dim: int = 32
    dropout: float = 0.0
    max_sequence_length: int = 50
    enable_structured_reconstruction: bool = True
    enable_future_spread: bool = True
    enable_future_volatility: bool = True
    enable_future_return: bool = True
    enable_future_imbalance: bool = True
    enable_contrastive: bool = False
    mask_probability: float = 0.30
    mask_value: float = 0.0
    future_bucket_count: int = 3
    reconstruction_loss_weight: float = 1.0
    future_spread_loss_weight: float = 0.5
    future_volatility_loss_weight: float = 0.5
    future_return_loss_weight: float = 1.0
    future_imbalance_loss_weight: float = 0.5
    contrastive_loss_weight: float = 0.1
    contrastive_temperature: float = 0.20
    ignore_index: int = -100

    def __post_init__(self) -> None:
        _validate_positive_int(self.input_features, name="input_features")
        _validate_positive_int(self.model_dim, name="model_dim")
        _validate_positive_int(self.num_heads, name="num_heads")
        _validate_positive_int(self.num_layers, name="num_layers")
        _validate_positive_int(self.feedforward_dim, name="feedforward_dim")
        _validate_positive_int(self.max_sequence_length, name="max_sequence_length")
        _validate_positive_int(self.future_bucket_count, name="future_bucket_count")
        if self.future_bucket_count < 2:
            raise ValueError("future_bucket_count must be >= 2")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise TypeError("dropout must be a float")
        if not (0.0 <= float(self.dropout) < 1.0):
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        for flag_name in (
            "enable_structured_reconstruction",
            "enable_future_spread",
            "enable_future_volatility",
            "enable_future_return",
            "enable_future_imbalance",
            "enable_contrastive",
        ):
            if not isinstance(getattr(self, flag_name), bool):
                raise TypeError(f"{flag_name} must be a bool")
        if not self.enabled_objectives():
            raise ValueError("at least one SSL-v2 objective must be enabled")
        _validate_unit_interval(self.mask_probability, name="mask_probability")
        if self.enable_structured_reconstruction and self.mask_probability <= 0.0:
            raise ValueError(
                "mask_probability must be positive when structured reconstruction "
                "is enabled"
            )
        for field_name in (
            "reconstruction_loss_weight",
            "future_spread_loss_weight",
            "future_volatility_loss_weight",
            "future_return_loss_weight",
            "future_imbalance_loss_weight",
            "contrastive_loss_weight",
        ):
            _validate_non_negative_float(getattr(self, field_name), name=field_name)
        _validate_positive_float(
            self.contrastive_temperature, name="contrastive_temperature"
        )
        if isinstance(self.ignore_index, bool) or not isinstance(self.ignore_index, int):
            raise TypeError("ignore_index must be an integer")
        total_weight = 0.0
        for objective in self.enabled_objectives():
            total_weight += self.loss_weight_for(objective)
        if total_weight <= 0.0:
            raise ValueError("at least one enabled SSL-v2 objective needs positive weight")

    def enabled_objectives(self) -> tuple[str, ...]:
        """Return enabled objective names in stable order."""
        names: list[str] = []
        if self.enable_structured_reconstruction:
            names.append("reconstruction")
        if self.enable_future_spread:
            names.append("future_spread_widening")
        if self.enable_future_volatility:
            names.append("future_volatility")
        if self.enable_future_return:
            names.append("future_return")
        if self.enable_future_imbalance:
            names.append("future_imbalance")
        if self.enable_contrastive:
            names.append("contrastive")
        return tuple(names)

    def future_task_classes(self) -> dict[str, int]:
        """Return output class counts for enabled future-state tasks."""
        tasks: dict[str, int] = {}
        if self.enable_future_spread:
            tasks["future_spread_widening"] = 2
        if self.enable_future_volatility:
            tasks["future_volatility"] = int(self.future_bucket_count)
        if self.enable_future_return:
            tasks["future_return"] = int(self.future_bucket_count)
        if self.enable_future_imbalance:
            tasks["future_imbalance"] = int(self.future_bucket_count)
        return tasks

    def loss_weight_for(self, component: str) -> float:
        """Return the configured weight for a loss component."""
        weights = {
            "reconstruction": float(self.reconstruction_loss_weight),
            "future_spread_widening": float(self.future_spread_loss_weight),
            "future_volatility": float(self.future_volatility_loss_weight),
            "future_return": float(self.future_return_loss_weight),
            "future_imbalance": float(self.future_imbalance_loss_weight),
            "contrastive": float(self.contrastive_loss_weight),
        }
        return weights[component]

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
class MatrixSSLV2Output:
    """Structured output returned by :class:`MatrixSSLV2Model`."""

    loss: Any | None
    loss_components: dict[str, Any]
    masked_reconstruction: Any | None
    future_logits: dict[str, Any]
    hidden_states: Any
    pooled_state: Any


class MatrixSSLV2Model(_TORCH_MODULE_BASE):
    """Market-state-aware self-supervised transformer for matrix windows."""

    def __init__(self, config: MatrixSSLV2Config) -> None:
        _require_torch()
        super().__init__()
        if not isinstance(config, MatrixSSLV2Config):
            raise TypeError("config must be a MatrixSSLV2Config instance")
        self._config = config

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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.reconstruction_head: Any = (
            nn.Linear(config.model_dim, config.input_features)
            if config.enable_structured_reconstruction
            else None
        )
        self.future_heads: Any = nn.ModuleDict(
            {
                task_name: nn.Linear(config.model_dim, class_count)
                for task_name, class_count in config.future_task_classes().items()
            }
        )

    @property
    def config(self) -> MatrixSSLV2Config:
        """Return the validated SSL-v2 configuration."""
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
        future_labels: Mapping[str, Any] | None = None,
        contrastive_labels: Any | None = None,
    ) -> MatrixSSLV2Output:
        """Run the SSL-v2 forward pass and optional multi-task loss."""
        torch_module = _require_torch()
        hidden = self.encode(x)
        pooled = hidden.mean(dim=1)
        compute_loss = (
            mask is not None
            or masked_target is not None
            or future_labels is not None
            or contrastive_labels is not None
        )
        total_loss: Any | None = (
            torch_module.zeros((), device=hidden.device) if compute_loss else None
        )
        loss_components: dict[str, Any] = {}

        masked_reconstruction: Any | None = None
        if self.reconstruction_head is not None:
            masked_reconstruction = self.reconstruction_head(hidden)
            if mask is not None or masked_target is not None:
                if mask is None or masked_target is None:
                    raise ValueError(
                        "structured reconstruction requires both mask and masked_target"
                    )
                loss = self._masked_loss(
                    reconstruction=masked_reconstruction,
                    target=masked_target,
                    mask=mask,
                )
                loss_components["reconstruction"] = loss
                total_loss = total_loss + self._config.loss_weight_for("reconstruction") * loss

        future_logits: dict[str, Any] = {}
        for task_name, head in self.future_heads.items():
            logits = head(pooled)
            future_logits[str(task_name)] = logits
            if future_labels is not None and task_name in future_labels:
                loss = self._classification_loss(
                    logits=logits,
                    labels=future_labels[task_name],
                    task_name=str(task_name),
                )
                loss_components[str(task_name)] = loss
                total_loss = total_loss + self._config.loss_weight_for(str(task_name)) * loss

        if self._config.enable_contrastive and contrastive_labels is not None:
            loss = self._contrastive_loss(pooled=pooled, labels=contrastive_labels)
            loss_components["contrastive"] = loss
            total_loss = total_loss + self._config.loss_weight_for("contrastive") * loss

        if compute_loss and not loss_components:
            raise ValueError("no enabled SSL-v2 objective produced a valid target")
        if compute_loss and total_loss is not None:
            loss_components["total"] = total_loss

        return MatrixSSLV2Output(
            loss=total_loss,
            loss_components=loss_components,
            masked_reconstruction=masked_reconstruction,
            future_logits=future_logits,
            hidden_states=hidden,
            pooled_state=pooled,
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
            raise ValueError("structured reconstruction received no masked entries")
        diff = (reconstruction - target) * bool_mask.to(reconstruction.dtype)
        return (diff * diff).sum() / float(selected)

    def _classification_loss(self, *, logits: Any, labels: Any, task_name: str) -> Any:
        torch_module = _require_torch()
        if not torch_module.is_tensor(labels):
            raise TypeError(f"{task_name} labels must be a torch.Tensor")
        if labels.dtype != torch_module.long:
            raise TypeError(f"{task_name} labels must have dtype torch.long")
        if labels.ndim != 1 or int(labels.shape[0]) != int(logits.shape[0]):
            raise ValueError(
                f"{task_name} labels must have shape [batch]; got {tuple(labels.shape)}"
            )
        valid = int((labels != self._config.ignore_index).sum().item())
        if valid == 0:
            raise ValueError(f"{task_name} received no valid labels")
        return torch_functional.cross_entropy(
            logits,
            labels,
            ignore_index=self._config.ignore_index,
            reduction="mean",
        )

    def _contrastive_loss(self, *, pooled: Any, labels: Any) -> Any:
        torch_module = _require_torch()
        if not torch_module.is_tensor(labels):
            raise TypeError("contrastive_labels must be a torch.Tensor")
        if labels.dtype != torch_module.long:
            raise TypeError("contrastive_labels must have dtype torch.long")
        if labels.ndim != 1 or int(labels.shape[0]) != int(pooled.shape[0]):
            raise ValueError(
                "contrastive_labels must have shape [batch]; "
                f"got {tuple(labels.shape)}"
            )
        valid_mask = labels != self._config.ignore_index
        if int(valid_mask.sum().item()) < 2:
            return pooled.sum() * 0.0
        z = torch_functional.normalize(pooled[valid_mask], dim=1)
        regime = labels[valid_mask]
        logits = z @ z.t() / float(self._config.contrastive_temperature)
        eye = torch_module.eye(int(logits.shape[0]), dtype=torch_module.bool, device=logits.device)
        positive_mask = (regime[:, None] == regime[None, :]) & ~eye
        valid_rows = positive_mask.any(dim=1)
        if not bool(valid_rows.any().item()):
            return pooled.sum() * 0.0
        logits = logits.masked_fill(eye, -torch_module.inf)
        log_prob = logits - torch_module.logsumexp(logits, dim=1, keepdim=True)
        per_row = -(log_prob * positive_mask.to(log_prob.dtype)).sum(dim=1)
        per_row = per_row / positive_mask.sum(dim=1).clamp_min(1).to(log_prob.dtype)
        return per_row[valid_rows].mean()

    def encoder_state_dict(self) -> dict[str, Any]:
        """Return transferable encoder parameters as a CPU state dict."""
        return {
            str(key): value.detach().cpu().clone()
            for key, value in self.state_dict().items()
            if str(key).startswith(TRANSFERABLE_ENCODER_PREFIXES)
        }


def create_matrix_ssl_v2_model(config: MatrixSSLV2Config) -> MatrixSSLV2Model:
    """Construct a :class:`MatrixSSLV2Model` from a validated config."""
    return MatrixSSLV2Model(config)


_RESERVED_CLASSIFIER_TYPE = MatrixTransformerClassifier
