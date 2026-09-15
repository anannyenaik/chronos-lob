"""Tests for the supervised market transformer encoder."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from chronoslob.models.transformer import (  # noqa: E402
    MarketTransformerConfig,
    MarketTransformerEncoder,
    MarketTransformerOutput,
    TokenFieldEmbeddingConfig,
    TransformerPooling,
    create_market_transformer,
)
from chronoslob.training.token_datasets import TOKEN_WINDOW_FIELD_NAMES  # noqa: E402

_VOCAB_SIZES = dict.fromkeys(TOKEN_WINDOW_FIELD_NAMES, 10)


def _config(
    *,
    vocab_sizes: dict[str, int] | None = None,
    field_embedding_dim: int = 4,
    model_dim: int = 16,
    num_heads: int = 4,
    num_layers: int = 2,
    feedforward_dim: int = 32,
    dropout: float = 0.0,
    max_sequence_length: int = 8,
    num_classes: int = 3,
    pooling: str = "mean",
    activation: str = "gelu",
    use_layer_norm: bool = True,
    pad_token_id: int = 0,
) -> MarketTransformerConfig:
    return MarketTransformerConfig(
        vocab_sizes=dict(vocab_sizes if vocab_sizes is not None else _VOCAB_SIZES),
        field_embedding_dim=field_embedding_dim,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
        max_sequence_length=max_sequence_length,
        num_classes=num_classes,
        pooling=pooling,  # type: ignore[arg-type]
        activation=activation,  # type: ignore[arg-type]
        use_layer_norm=use_layer_norm,
        pad_token_id=pad_token_id,
    )


def _batch(
    *,
    batch_size: int = 2,
    seq_len: int = 4,
    fill_value: int = 5,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    inputs: dict[str, torch.Tensor] = {
        field_name: torch.full((batch_size, seq_len), fill_value, dtype=torch.long)
        for field_name in TOKEN_WINDOW_FIELD_NAMES
    }
    inputs["attention_mask"] = (
        attention_mask
        if attention_mask is not None
        else torch.ones((batch_size, seq_len), dtype=torch.bool)
    )
    return inputs


def test_config_requires_model_dim_divisible_by_num_heads() -> None:
    with pytest.raises(ValueError, match="model_dim must be divisible by num_heads"):
        _config(model_dim=10, num_heads=4)


def test_config_rejects_non_positive_num_classes() -> None:
    with pytest.raises(ValueError, match="num_classes must be positive"):
        _config(num_classes=0)


def test_config_rejects_invalid_pooling() -> None:
    with pytest.raises(ValueError, match="pooling must be"):
        _config(pooling="weighted")


def test_config_rejects_invalid_activation() -> None:
    with pytest.raises(ValueError, match="activation must be"):
        _config(activation="silu")


def test_config_rejects_dropout_out_of_range() -> None:
    with pytest.raises(ValueError, match="dropout"):
        _config(dropout=1.5)
    with pytest.raises(ValueError, match="dropout"):
        _config(dropout=-0.1)


def test_config_rejects_zero_vocab_size() -> None:
    vocab_sizes = dict(_VOCAB_SIZES)
    vocab_sizes["event_type"] = 0
    with pytest.raises(ValueError, match="vocab_sizes\\['event_type'\\]"):
        _config(vocab_sizes=vocab_sizes)


def test_config_rejects_missing_token_field() -> None:
    vocab_sizes = {key: value for key, value in _VOCAB_SIZES.items() if key != "source"}
    with pytest.raises(ValueError, match="missing required token field 'source'"):
        _config(vocab_sizes=vocab_sizes)


def test_config_rejects_unknown_token_field() -> None:
    vocab_sizes = dict(_VOCAB_SIZES)
    vocab_sizes["mystery"] = 10
    with pytest.raises(ValueError, match="unknown token fields"):
        _config(vocab_sizes=vocab_sizes)


def test_config_rejects_non_positive_max_sequence_length() -> None:
    with pytest.raises(ValueError, match="max_sequence_length must be positive"):
        _config(max_sequence_length=0)


def test_create_market_transformer_returns_module_instance() -> None:
    model = create_market_transformer(_config())
    assert isinstance(model, MarketTransformerEncoder)
    assert isinstance(model, torch.nn.Module)


def test_n_parameters_is_positive() -> None:
    model = create_market_transformer(_config())
    assert model.n_parameters() > 0


def test_forward_returns_logits_with_expected_shape() -> None:
    config = _config(num_classes=5)
    model = create_market_transformer(config)
    inputs = _batch(batch_size=3, seq_len=4, fill_value=2)

    output = model(inputs)

    assert isinstance(output, MarketTransformerOutput)
    assert tuple(output.logits.shape) == (3, 5)
    assert output.pooled is None
    assert output.hidden_states is None


def test_forward_can_return_pooled_and_hidden_states() -> None:
    config = _config(model_dim=16)
    model = create_market_transformer(config)
    inputs = _batch(batch_size=2, seq_len=4, fill_value=3)

    output = model(inputs, return_pooled=True, return_hidden_states=True)

    assert tuple(output.pooled.shape) == (2, 16)
    assert tuple(output.hidden_states.shape) == (2, 4, 16)


def test_forward_supports_all_required_token_fields() -> None:
    model = create_market_transformer(_config())
    inputs = _batch()
    for field_name in TOKEN_WINDOW_FIELD_NAMES:
        assert field_name in inputs
    # Should not raise.
    model(inputs)


def test_forward_missing_required_field_raises_clear_error() -> None:
    model = create_market_transformer(_config())
    inputs = _batch()
    del inputs["source"]
    with pytest.raises(KeyError, match="'source'"):
        model(inputs)


def test_forward_missing_attention_mask_raises_clear_error() -> None:
    model = create_market_transformer(_config())
    inputs = _batch()
    del inputs["attention_mask"]
    with pytest.raises(KeyError, match="'attention_mask'"):
        model(inputs)


def test_forward_mismatched_sequence_length_raises() -> None:
    model = create_market_transformer(_config())
    inputs = _batch(batch_size=2, seq_len=4)
    inputs["side"] = torch.zeros((2, 3), dtype=torch.long)
    with pytest.raises(ValueError, match="does not match attention_mask shape"):
        model(inputs)


def test_forward_attention_mask_shape_mismatch_raises() -> None:
    model = create_market_transformer(_config())
    inputs = _batch(batch_size=2, seq_len=4)
    inputs["attention_mask"] = torch.ones((2, 5), dtype=torch.bool)
    with pytest.raises(ValueError, match="does not match attention_mask shape"):
        model(inputs)


def test_forward_attention_mask_wrong_rank_raises() -> None:
    model = create_market_transformer(_config())
    inputs = _batch()
    inputs["attention_mask"] = torch.ones((2, 4, 1), dtype=torch.bool)
    with pytest.raises(ValueError, match=r"attention_mask.*2D"):
        model(inputs)


def test_forward_attention_mask_wrong_dtype_raises() -> None:
    model = create_market_transformer(_config())
    inputs = _batch()
    inputs["attention_mask"] = torch.ones((2, 4), dtype=torch.long)
    with pytest.raises(TypeError, match="boolean tensor"):
        model(inputs)


def test_forward_categorical_field_wrong_dtype_raises() -> None:
    model = create_market_transformer(_config())
    inputs = _batch()
    inputs["event_type"] = torch.ones((2, 4), dtype=torch.float)
    with pytest.raises(TypeError, match=r"torch\.long"):
        model(inputs)


def test_forward_categorical_field_wrong_rank_raises() -> None:
    model = create_market_transformer(_config())
    inputs = _batch()
    inputs["side"] = torch.zeros((2,), dtype=torch.long)
    with pytest.raises(ValueError, match="must be 2D"):
        model(inputs)


def test_forward_rejects_sequence_above_max_length() -> None:
    config = _config(max_sequence_length=3)
    model = create_market_transformer(config)
    inputs = _batch(batch_size=2, seq_len=5)
    with pytest.raises(ValueError, match="exceeds max_sequence_length"):
        model(inputs)


def test_forward_rejects_fully_padded_sequence() -> None:
    model = create_market_transformer(_config())
    inputs = _batch(batch_size=2, seq_len=4)
    inputs["attention_mask"] = torch.zeros((2, 4), dtype=torch.bool)
    with pytest.raises(ValueError, match="fully padded"):
        model(inputs)


def test_mean_pooling_ignores_padding() -> None:
    pooling = TransformerPooling("mean")
    hidden = torch.tensor(
        [
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [10.0, 10.0],  # padding
                [10.0, 10.0],  # padding
            ]
        ]
    )
    attention_mask = torch.tensor([[True, True, False, False]])
    pooled = pooling(hidden, attention_mask)
    assert torch.allclose(pooled, torch.tensor([[1.5, 1.5]]))


def test_last_pooling_selects_last_real_token() -> None:
    pooling = TransformerPooling("last")
    hidden = torch.tensor(
        [
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [99.0, 99.0],  # padding
            ]
        ]
    )
    attention_mask = torch.tensor([[True, True, True, False]])
    pooled = pooling(hidden, attention_mask)
    assert torch.allclose(pooled, torch.tensor([[3.0, 3.0]]))


def test_bos_pooling_selects_first_real_token_with_left_padding() -> None:
    pooling = TransformerPooling("bos")
    hidden = torch.tensor(
        [
            [
                [99.0, 99.0],  # padding (left-padded)
                [99.0, 99.0],  # padding
                [7.0, 7.0],
                [8.0, 8.0],
            ]
        ]
    )
    attention_mask = torch.tensor([[False, False, True, True]])
    pooled = pooling(hidden, attention_mask)
    assert torch.allclose(pooled, torch.tensor([[7.0, 7.0]]))


def test_pooling_rejects_fully_padded_row() -> None:
    pooling = TransformerPooling("mean")
    hidden = torch.zeros((1, 3, 2))
    attention_mask = torch.zeros((1, 3), dtype=torch.bool)
    with pytest.raises(ValueError, match="fully padded"):
        pooling(hidden, attention_mask)


def test_attention_mask_padding_does_not_affect_pooled_output() -> None:
    torch.manual_seed(0)
    config = _config(dropout=0.0)
    model = create_market_transformer(config)
    model.eval()
    inputs = _batch(batch_size=1, seq_len=3, fill_value=2)
    inputs["attention_mask"] = torch.tensor([[True, True, True]])
    output_real = model(inputs, return_pooled=True)

    # Pad by adding an extra position whose mask is False and whose token IDs
    # differ. With the mask-aware pooling we expect the pooled output to be
    # identical because padding contributes nothing.
    padded_inputs: dict[str, torch.Tensor] = {
        field_name: torch.cat(
            [
                inputs[field_name],
                torch.full((1, 1), 7, dtype=torch.long),
            ],
            dim=1,
        )
        for field_name in TOKEN_WINDOW_FIELD_NAMES
    }
    padded_inputs["attention_mask"] = torch.tensor([[True, True, True, False]])
    output_padded = model(padded_inputs, return_pooled=True)

    # The first three positions are processed identically because their
    # padding mask matches and the encoder is mask-aware. The pooled mean
    # over the first three (real) positions should therefore match exactly.
    assert torch.allclose(output_real.pooled, output_padded.pooled, atol=1e-5)


def test_backward_pass_produces_finite_gradients() -> None:
    config = _config(num_classes=3)
    model = create_market_transformer(config)
    inputs = _batch(batch_size=4, seq_len=4, fill_value=2)
    targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    loss_fn = torch.nn.CrossEntropyLoss()

    logits = model(inputs).logits
    loss = loss_fn(logits, targets)
    loss.backward()

    has_grad = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        has_grad = True
        assert torch.isfinite(parameter.grad).all().item()
    assert has_grad


def test_predict_logits_disables_gradients() -> None:
    config = _config(num_classes=2)
    model = create_market_transformer(config)
    inputs = _batch(batch_size=2, seq_len=3, fill_value=1)
    logits = model.predict_logits(inputs)
    assert logits.requires_grad is False
    assert tuple(logits.shape) == (2, 2)


def test_deterministic_outputs_when_seeded_and_in_eval_mode() -> None:
    config = _config(dropout=0.0)
    inputs = _batch(batch_size=2, seq_len=4, fill_value=3)
    torch.manual_seed(123)
    model_a = create_market_transformer(config)
    model_a.eval()
    torch.manual_seed(123)
    model_b = create_market_transformer(config)
    model_b.eval()
    with torch.no_grad():
        logits_a = model_a(inputs).logits
        logits_b = model_b(inputs).logits
    assert torch.allclose(logits_a, logits_b)


def test_token_field_embedding_config_concatenated_dim() -> None:
    cfg = TokenFieldEmbeddingConfig(
        vocab_sizes=dict(_VOCAB_SIZES),
        field_embedding_dim=4,
    )
    assert cfg.concatenated_dim == 4 * len(TOKEN_WINDOW_FIELD_NAMES)


def test_transformer_field_names_match_phase_11() -> None:
    from chronoslob.models.transformer import (
        TOKEN_WINDOW_FIELD_NAMES as TRANSFORMER_NAMES,
    )

    assert TRANSFORMER_NAMES == TOKEN_WINDOW_FIELD_NAMES
