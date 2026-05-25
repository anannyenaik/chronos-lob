"""Transformer classifier for normalised FI-2010 matrix windows.

This model consumes numeric feature windows directly. It is intentionally
separate from raw order-book schemas so z-score-normalised FI-2010 values,
including negative quantities, never need to be coerced into
``OrderBookLevel`` or ``OrderBookSnapshot`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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

__all__ = [
    "MatrixTransformerClassifier",
    "MatrixTransformerConfig",
    "create_matrix_transformer",
]


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the matrix transformer model. Install the "
            "'torch' optional dependency: pip install -e '.[torch]'"
        )
    return torch


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_dropout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("dropout must be a float")
    numeric = float(value)
    if numeric < 0.0 or numeric >= 1.0:
        raise ValueError("dropout must satisfy 0 <= dropout < 1")
    return numeric


@dataclass(frozen=True)
class MatrixTransformerConfig:
    """Configuration for the supervised matrix transformer classifier."""

    input_features: int
    n_classes: int
    model_dim: int = 16
    num_heads: int = 2
    num_layers: int = 1
    feedforward_dim: int = 32
    dropout: float = 0.0
    max_sequence_length: int = 4

    def __post_init__(self) -> None:
        _validate_positive_int(self.input_features, name="input_features")
        _validate_positive_int(self.n_classes, name="n_classes")
        if self.n_classes < 2:
            raise ValueError("n_classes must be >= 2 for classification")
        _validate_positive_int(self.model_dim, name="model_dim")
        _validate_positive_int(self.num_heads, name="num_heads")
        _validate_positive_int(self.num_layers, name="num_layers")
        _validate_positive_int(self.feedforward_dim, name="feedforward_dim")
        _validate_positive_int(self.max_sequence_length, name="max_sequence_length")
        _validate_dropout(self.dropout)
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")


class MatrixTransformerClassifier(_TORCH_MODULE_BASE):
    """Small transformer encoder for numeric FI-2010 feature windows."""

    def __init__(self, config: MatrixTransformerConfig) -> None:
        _require_torch()
        super().__init__()
        if not isinstance(config, MatrixTransformerConfig):
            raise TypeError("config must be a MatrixTransformerConfig instance")
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
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )
        self.classifier = nn.Linear(config.model_dim, config.n_classes)

    @property
    def config(self) -> MatrixTransformerConfig:
        """Return the validated model configuration."""
        return self._config

    def forward(self, x: Any) -> Any:
        """Run a forward pass over ``[batch, window, feature]`` tensors."""
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
        encoded = self.encoder(hidden)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)

    def n_parameters(self) -> int:
        """Return the total parameter count."""
        return int(sum(parameter.numel() for parameter in self.parameters()))


def create_matrix_transformer(
    config: MatrixTransformerConfig,
) -> MatrixTransformerClassifier:
    """Construct a supervised normalised-matrix transformer classifier."""
    return MatrixTransformerClassifier(config)
