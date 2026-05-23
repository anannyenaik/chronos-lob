"""Tests for the DeepLOB-style supervised CNN-LSTM baseline."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from chronoslob.models.deeplob import (  # noqa: E402
    DeepLOBConfig,
    DeepLOBModel,
    create_deeplob_model,
)


def _config(
    *,
    input_features: int = 4,
    n_classes: int = 3,
    conv_channels: int = 8,
    conv_kernel_size: int = 3,
    lstm_hidden_size: int = 16,
    lstm_layers: int = 1,
    dropout: float = 0.0,
    use_batch_norm: bool = True,
) -> DeepLOBConfig:
    return DeepLOBConfig(
        input_features=input_features,
        n_classes=n_classes,
        conv_channels=conv_channels,
        conv_kernel_size=conv_kernel_size,
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
    )


def test_deeplob_config_rejects_zero_input_features() -> None:
    with pytest.raises(ValueError, match="input_features must be positive"):
        DeepLOBConfig(input_features=0, n_classes=3)


def test_deeplob_config_rejects_single_class() -> None:
    with pytest.raises(ValueError, match="n_classes must be >= 2"):
        DeepLOBConfig(input_features=4, n_classes=1)


def test_deeplob_config_rejects_non_integer_input_features() -> None:
    with pytest.raises(TypeError, match="input_features must be an integer"):
        DeepLOBConfig(input_features=3.5, n_classes=3)  # type: ignore[arg-type]


def test_deeplob_config_rejects_invalid_dropout() -> None:
    with pytest.raises(ValueError, match="dropout"):
        DeepLOBConfig(input_features=4, n_classes=3, dropout=1.0)
    with pytest.raises(ValueError, match="dropout"):
        DeepLOBConfig(input_features=4, n_classes=3, dropout=-0.1)


def test_deeplob_config_rejects_zero_lstm_layers() -> None:
    with pytest.raises(ValueError, match="lstm_layers must be positive"):
        DeepLOBConfig(input_features=4, n_classes=3, lstm_layers=0)


def test_create_deeplob_model_returns_module_instance() -> None:
    model = create_deeplob_model(_config())

    assert isinstance(model, DeepLOBModel)
    assert isinstance(model, torch.nn.Module)


def test_forward_output_shape_is_batch_by_n_classes() -> None:
    config = _config(input_features=5, n_classes=4)
    model = create_deeplob_model(config)

    x = torch.randn(7, 6, 5)
    logits = model(x)

    assert tuple(logits.shape) == (7, 4)


def test_forward_rejects_non_tensor_input() -> None:
    model = create_deeplob_model(_config())

    with pytest.raises(TypeError, match=r"torch\.Tensor"):
        model([[0.0]])  # type: ignore[arg-type]


def test_forward_rejects_non_3d_input() -> None:
    model = create_deeplob_model(_config(input_features=4))

    x = torch.randn(5, 4)
    with pytest.raises(ValueError, match="3D"):
        model(x)


def test_forward_rejects_wrong_feature_count() -> None:
    model = create_deeplob_model(_config(input_features=4))

    x = torch.randn(2, 6, 3)
    with pytest.raises(ValueError, match="input_features"):
        model(x)


def test_n_parameters_is_positive() -> None:
    model = create_deeplob_model(_config())

    assert model.n_parameters() > 0


def test_backward_pass_produces_finite_gradients() -> None:
    config = _config(input_features=4, n_classes=3, dropout=0.0)
    model = create_deeplob_model(config)

    x = torch.randn(4, 5, 4)
    y = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    loss_fn = torch.nn.CrossEntropyLoss()

    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()

    has_grad = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        has_grad = True
        assert torch.isfinite(parameter.grad).all().item()
    assert has_grad


def test_predict_logits_does_not_record_grad() -> None:
    config = _config(input_features=3, n_classes=2)
    model = create_deeplob_model(config)

    x = torch.randn(2, 4, 3)
    logits = model.predict_logits(x)

    assert logits.requires_grad is False
    assert tuple(logits.shape) == (2, 2)


def test_small_lookback_value_is_supported() -> None:
    config = _config(input_features=3, n_classes=2, conv_kernel_size=3)
    model = create_deeplob_model(config)

    x = torch.randn(2, 1, 3)
    logits = model(x)

    assert tuple(logits.shape) == (2, 2)


def test_use_batch_norm_false_path_works() -> None:
    config = _config(use_batch_norm=False)
    model = create_deeplob_model(config)

    x = torch.randn(3, 4, config.input_features)
    logits = model(x)

    assert tuple(logits.shape) == (3, config.n_classes)


def test_multi_layer_lstm_path_works() -> None:
    config = _config(lstm_layers=2, dropout=0.1)
    model = create_deeplob_model(config)

    x = torch.randn(2, 5, config.input_features)
    logits = model(x)

    assert tuple(logits.shape) == (2, config.n_classes)
