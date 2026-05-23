"""DeepLOB-style supervised CNN-LSTM baseline.

This module ships a compact, auditable CNN-LSTM classifier inspired by the
DeepLOB architecture for limit order book forecasting. The implementation
is deliberately small so it can be reviewed in isolation and so future
work can match the original paper's architecture in a separate, explicit
phase. Nothing here implements transformers, self-supervised pretraining,
event tokenisation, execution backtests or trading logic.

PyTorch is treated as an optional dependency. The model class imports
``torch`` directly and raises a clear ``ImportError`` when the optional
``[torch]`` extra is not installed. Tests should guard imports with
``pytest.importorskip("torch")``.
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
    "DeepLOBConfig",
    "DeepLOBModel",
    "create_deeplob_model",
]


def _require_torch() -> Any:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the DeepLOB-style model. Install the "
            "'torch' optional dependency: pip install -e '.[torch]'"
        )
    return torch


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class DeepLOBConfig:
    """Configuration for the DeepLOB-style CNN-LSTM baseline.

    The defaults are deliberately small to keep CPU smoke tests fast. They
    do not target benchmark performance — see ``reports/deeplob_baseline.md``
    for why this is a baseline rather than a full DeepLOB replica.
    """

    input_features: int
    n_classes: int
    conv_channels: int = 16
    conv_kernel_size: int = 3
    lstm_hidden_size: int = 32
    lstm_layers: int = 1
    dropout: float = 0.1
    use_batch_norm: bool = True

    def __post_init__(self) -> None:
        """Validate model hyperparameters at construction time."""
        _validate_positive_int(self.input_features, name="input_features")
        if isinstance(self.n_classes, bool) or not isinstance(self.n_classes, int):
            raise TypeError("n_classes must be an integer")
        if self.n_classes < 2:
            raise ValueError("n_classes must be >= 2 for classification")
        _validate_positive_int(self.conv_channels, name="conv_channels")
        _validate_positive_int(self.conv_kernel_size, name="conv_kernel_size")
        _validate_positive_int(self.lstm_hidden_size, name="lstm_hidden_size")
        _validate_positive_int(self.lstm_layers, name="lstm_layers")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise TypeError("dropout must be a float")
        dropout_value = float(self.dropout)
        if dropout_value < 0.0 or dropout_value >= 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if not isinstance(self.use_batch_norm, bool):
            raise TypeError("use_batch_norm must be a bool")


class DeepLOBModel(_TORCH_MODULE_BASE):
    """Compact DeepLOB-style CNN-LSTM classifier.

    The forward pass expects an input tensor of shape
    ``[batch, lookback, n_features]``. Features are treated as channels for
    two 1D convolutional layers, then the resulting representation is fed
    to a small LSTM whose final time-step output is projected to class
    logits. This is *not* an exact reproduction of the DeepLOB paper; see
    :func:`create_deeplob_model` for the full caveat.
    """

    def __init__(self, config: DeepLOBConfig) -> None:
        """Build the CNN-LSTM layers from a validated configuration."""
        _require_torch()
        super().__init__()
        if not isinstance(config, DeepLOBConfig):
            raise TypeError("config must be a DeepLOBConfig instance")
        self._config = config

        padding = config.conv_kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels=config.input_features,
            out_channels=config.conv_channels,
            kernel_size=config.conv_kernel_size,
            padding=padding,
        )
        self.bn1: nn.Module | None = (
            nn.BatchNorm1d(config.conv_channels) if config.use_batch_norm else None
        )
        self.conv2 = nn.Conv1d(
            in_channels=config.conv_channels,
            out_channels=config.conv_channels,
            kernel_size=config.conv_kernel_size,
            padding=padding,
        )
        self.bn2: nn.Module | None = (
            nn.BatchNorm1d(config.conv_channels) if config.use_batch_norm else None
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=float(config.dropout))
        self.lstm = nn.LSTM(
            input_size=config.conv_channels,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=float(config.dropout) if config.lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(config.lstm_hidden_size, config.n_classes)

    @property
    def config(self) -> DeepLOBConfig:
        """Return the validated configuration used to build the model."""
        return self._config

    def forward(self, x: Any) -> Any:
        """Run a forward pass and return ``[batch, n_classes]`` logits."""
        torch_module = _require_torch()
        if not torch_module.is_tensor(x):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError(
                "x must be 3D with shape [batch, lookback, n_features]; "
                f"got shape {tuple(x.shape)}"
            )
        if int(x.shape[-1]) != self._config.input_features:
            raise ValueError(
                "x feature dimension does not match config.input_features: "
                f"expected {self._config.input_features}, got {int(x.shape[-1])}"
            )
        if int(x.shape[1]) <= 0:
            raise ValueError("x lookback dimension must be positive")

        # [batch, lookback, n_features] -> [batch, n_features, lookback]
        conv_input = x.transpose(1, 2)
        hidden = self.conv1(conv_input)
        if self.bn1 is not None:
            hidden = self.bn1(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.conv2(hidden)
        if self.bn2 is not None:
            hidden = self.bn2(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        # [batch, conv_channels, lookback] -> [batch, lookback, conv_channels]
        sequence = hidden.transpose(1, 2)
        lstm_output, _ = self.lstm(sequence)
        final_step = lstm_output[:, -1, :]
        logits = self.classifier(final_step)
        return logits

    def predict_logits(self, x: Any) -> Any:
        """Return logits in eval mode without computing gradients."""
        torch_module = _require_torch()
        was_training = self.training
        self.eval()
        try:
            with torch_module.no_grad():
                return self.forward(x)
        finally:
            if was_training:
                self.train()

    def n_parameters(self) -> int:
        """Return the total number of parameters in the model."""
        return int(sum(parameter.numel() for parameter in self.parameters()))


def create_deeplob_model(config: DeepLOBConfig) -> DeepLOBModel:
    """Construct a :class:`DeepLOBModel` from a validated configuration.

    This is a DeepLOB-*style* baseline, not an exact reproduction of the
    original DeepLOB paper. The convolutional and LSTM block sizes are
    intentionally small so the model can be trained quickly on a CPU for
    smoke testing. A future phase can match the paper's exact channel
    counts, inception modules and 100-row lookback over an FI-2010 input
    layout, and report results against the leakage-safe validation
    protocol used elsewhere in ChronosLOB.
    """
    return DeepLOBModel(config)
