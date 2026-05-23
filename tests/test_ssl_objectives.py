"""Tests for the SSL transformer wrapper and loss computation."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from chronoslob.models.ssl import (  # noqa: E402
    DEFAULT_MASKED_FIELDS,
    DEFAULT_NEXT_FIELDS,
    MarketSSLTransformer,
    MaskingConfig,
    SSLObjectiveName,
    SSLTransformerConfig,
    SSLTransformerOutput,
    create_ssl_transformer,
)
from chronoslob.models.transformer import MarketTransformerConfig  # noqa: E402
from chronoslob.training.token_datasets import TOKEN_WINDOW_FIELD_NAMES  # noqa: E402

_VOCAB_SIZES = dict.fromkeys(TOKEN_WINDOW_FIELD_NAMES, 12)


def _transformer_config(
    *,
    vocab_sizes: dict[str, int] | None = None,
    field_embedding_dim: int = 4,
    model_dim: int = 16,
    num_heads: int = 4,
    num_layers: int = 1,
    feedforward_dim: int = 32,
    dropout: float = 0.0,
    max_sequence_length: int = 8,
    num_classes: int = 2,
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
        pooling="mean",
        activation="gelu",
        use_layer_norm=True,
    )


def _ssl_config(
    *,
    transformer: MarketTransformerConfig | None = None,
    masked_fields: tuple[str, ...] = DEFAULT_MASKED_FIELDS,
    next_fields: tuple[str, ...] = DEFAULT_NEXT_FIELDS,
    enable_masked_field_loss: bool = True,
    enable_next_field_loss: bool = True,
    enable_contrastive_loss: bool = False,
    masking: MaskingConfig | None = None,
    loss_weights: dict[str, float] | None = None,
) -> SSLTransformerConfig:
    return SSLTransformerConfig(
        transformer=transformer if transformer is not None else _transformer_config(),
        masked_fields=masked_fields,
        next_fields=next_fields,
        enable_masked_field_loss=enable_masked_field_loss,
        enable_next_field_loss=enable_next_field_loss,
        enable_contrastive_loss=enable_contrastive_loss,
        masking=masking if masking is not None else MaskingConfig(),
        loss_weights=(
            dict(loss_weights)
            if loss_weights is not None
            else {
                "masked_field": 1.0,
                "next_field": 1.0,
                "contrastive": 0.0,
            }
        ),
    )


def _make_inputs(
    *,
    batch_size: int = 2,
    seq_len: int = 4,
    fill_value: int = 5,
) -> dict[str, torch.Tensor]:
    inputs: dict[str, torch.Tensor] = {
        field: torch.full((batch_size, seq_len), fill_value, dtype=torch.long)
        for field in TOKEN_WINDOW_FIELD_NAMES
    }
    inputs["attention_mask"] = torch.ones(
        (batch_size, seq_len), dtype=torch.bool
    )
    return inputs


def _label_for_field(
    *,
    batch_size: int = 2,
    seq_len: int = 4,
    ignore_index: int = -100,
    valid_positions: tuple[tuple[int, int, int], ...] = ((0, 1, 3), (1, 2, 7)),
) -> torch.Tensor:
    """Build a label tensor; ``valid_positions`` are ``(row, col, label)``."""
    label = torch.full(
        (batch_size, seq_len), ignore_index, dtype=torch.long
    )
    for row, col, value in valid_positions:
        label[row, col] = value
    return label


def test_default_ssl_config_validates() -> None:
    config = SSLTransformerConfig()
    assert "masked_field" in config.enabled_objectives()
    assert "next_field" in config.enabled_objectives()


def test_config_rejects_all_objectives_disabled() -> None:
    with pytest.raises(ValueError, match="at least one SSL objective"):
        _ssl_config(
            enable_masked_field_loss=False,
            enable_next_field_loss=False,
        )


def test_config_rejects_unknown_masked_field() -> None:
    with pytest.raises(ValueError, match="not a recognised token field"):
        SSLTransformerConfig(
            transformer=_transformer_config(),
            masked_fields=("mystery_field",),
        )


def test_config_rejects_unknown_next_field() -> None:
    with pytest.raises(ValueError, match="not a recognised token field"):
        SSLTransformerConfig(
            transformer=_transformer_config(),
            next_fields=("mystery_field",),
        )


def test_config_rejects_duplicate_masked_fields() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        SSLTransformerConfig(
            transformer=_transformer_config(),
            masked_fields=("event_type", "event_type"),
        )


def test_config_rejects_negative_loss_weight() -> None:
    with pytest.raises(ValueError, match="loss_weights"):
        _ssl_config(
            loss_weights={
                "masked_field": -1.0,
                "next_field": 1.0,
                "contrastive": 0.0,
            }
        )


def test_config_rejects_all_enabled_weights_zero() -> None:
    with pytest.raises(ValueError, match="positive loss weight"):
        _ssl_config(
            loss_weights={
                "masked_field": 0.0,
                "next_field": 0.0,
                "contrastive": 0.0,
            }
        )


def test_config_rejects_unknown_loss_weight_key() -> None:
    with pytest.raises(ValueError, match="unknown objective"):
        _ssl_config(
            loss_weights={
                "masked_field": 1.0,
                "next_field": 1.0,
                "contrastive": 0.0,
                "mystery": 0.5,
            }
        )


def test_contrastive_objective_is_deferred() -> None:
    with pytest.raises(NotImplementedError, match="contrastive"):
        SSLTransformerConfig(
            transformer=_transformer_config(),
            enable_contrastive_loss=True,
        )


def test_masking_probabilities_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match=r"must sum to 1\.0"):
        MaskingConfig(
            mask_probability=0.15,
            mask_token_probability=0.5,
            random_token_probability=0.3,
            keep_token_probability=0.3,
        )


def test_masking_mask_probability_must_be_unit_interval() -> None:
    with pytest.raises(ValueError, match="mask_probability"):
        MaskingConfig(mask_probability=1.5)


def test_create_ssl_transformer_returns_module() -> None:
    model = create_ssl_transformer(_ssl_config())
    assert isinstance(model, MarketSSLTransformer)
    assert isinstance(model, torch.nn.Module)
    assert model.n_parameters() > 0


def test_masked_field_heads_are_created_for_configured_fields() -> None:
    config = _ssl_config(masked_fields=("event_type", "side"))
    model = create_ssl_transformer(config)
    assert set(model.masked_heads.keys()) == {"event_type", "side"}
    assert len(model.next_heads) == len(config.next_fields)


def test_next_field_heads_are_created_for_configured_fields() -> None:
    config = _ssl_config(next_fields=("event_type",))
    model = create_ssl_transformer(config)
    assert set(model.next_heads.keys()) == {"event_type"}


def test_masked_field_loss_disabled_yields_no_masked_heads() -> None:
    config = _ssl_config(enable_masked_field_loss=False)
    model = create_ssl_transformer(config)
    assert len(model.masked_heads) == 0
    assert len(model.next_heads) > 0


def test_forward_returns_hidden_states_and_logits() -> None:
    config = _ssl_config(
        masked_fields=("event_type",),
        next_fields=("event_type",),
    )
    model = create_ssl_transformer(config)
    inputs = _make_inputs(batch_size=2, seq_len=4, fill_value=5)
    output = model(inputs)
    assert isinstance(output, SSLTransformerOutput)
    assert tuple(output.hidden_states.shape) == (2, 4, config.transformer.model_dim)
    assert tuple(output.masked_logits["event_type"].shape) == (
        2,
        4,
        config.transformer.vocab_sizes["event_type"],
    )
    assert tuple(output.next_logits["event_type"].shape) == (
        2,
        4,
        config.transformer.vocab_sizes["event_type"],
    )
    assert output.loss is None


def test_forward_with_labels_returns_finite_loss() -> None:
    config = _ssl_config(
        masked_fields=("event_type",),
        next_fields=("event_type",),
    )
    model = create_ssl_transformer(config)
    inputs = _make_inputs(batch_size=2, seq_len=4, fill_value=5)
    masked_labels = {
        "event_type": _label_for_field(
            batch_size=2,
            seq_len=4,
            valid_positions=((0, 1, 5), (1, 2, 5)),
        )
    }
    next_labels = {
        "event_type": _label_for_field(
            batch_size=2,
            seq_len=4,
            valid_positions=((0, 0, 5), (1, 1, 5), (0, 1, 5)),
        )
    }
    output = model(inputs, masked_labels=masked_labels, next_labels=next_labels)
    assert output.loss is not None
    assert torch.isfinite(output.loss).item()
    assert "masked_field" in output.loss_components
    assert "next_field" in output.loss_components
    assert torch.isfinite(output.loss_components["masked_field"]).item()
    assert torch.isfinite(output.loss_components["next_field"]).item()


def test_loss_ignores_ignore_index_positions() -> None:
    config = _ssl_config(masked_fields=("event_type",), next_fields=("event_type",))
    model = create_ssl_transformer(config)
    inputs = _make_inputs(batch_size=2, seq_len=4, fill_value=5)
    valid_labels = {
        "event_type": _label_for_field(
            batch_size=2,
            seq_len=4,
            valid_positions=((0, 1, 5),),
        )
    }
    next_labels = {
        "event_type": _label_for_field(
            batch_size=2,
            seq_len=4,
            valid_positions=((0, 0, 5),),
        )
    }
    output_valid = model(
        inputs,
        masked_labels=valid_labels,
        next_labels=next_labels,
    )

    all_ignore = {
        "event_type": torch.full((2, 4), -100, dtype=torch.long),
    }
    with pytest.raises(ValueError, match="no valid"):
        model(inputs, masked_labels=all_ignore, next_labels=all_ignore)
    assert torch.isfinite(output_valid.loss).item()


def test_padding_positions_do_not_contribute_to_loss() -> None:
    config = _ssl_config(masked_fields=("event_type",), next_fields=("event_type",))
    model = create_ssl_transformer(config)
    inputs = _make_inputs(batch_size=2, seq_len=4, fill_value=5)
    # left-pad first row
    inputs["attention_mask"][0, 0] = False
    inputs["attention_mask"][0, 1] = False
    for field_name in TOKEN_WINDOW_FIELD_NAMES:
        inputs[field_name][0, 0] = 0
        inputs[field_name][0, 1] = 0

    # Place labels only at real positions; padding positions are -100
    masked_labels = {
        "event_type": _label_for_field(
            batch_size=2,
            seq_len=4,
            valid_positions=((0, 2, 5), (1, 1, 5)),
        )
    }
    next_labels = {
        "event_type": _label_for_field(
            batch_size=2,
            seq_len=4,
            valid_positions=((0, 2, 5), (1, 1, 5)),
        )
    }
    output = model(
        inputs,
        masked_labels=masked_labels,
        next_labels=next_labels,
    )
    assert torch.isfinite(output.loss).item()


def test_backward_pass_produces_gradients() -> None:
    config = _ssl_config(masked_fields=("event_type",), next_fields=("event_type",))
    model = create_ssl_transformer(config)
    inputs = _make_inputs(batch_size=2, seq_len=4, fill_value=5)
    masked_labels = {
        "event_type": _label_for_field(
            batch_size=2,
            seq_len=4,
            valid_positions=((0, 1, 5), (1, 2, 5)),
        )
    }
    next_labels = {
        "event_type": _label_for_field(
            batch_size=2,
            seq_len=4,
            valid_positions=((0, 0, 5), (1, 1, 5)),
        )
    }
    output = model(inputs, masked_labels=masked_labels, next_labels=next_labels)
    output.loss.backward()
    has_grad = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        has_grad = True
        assert torch.isfinite(parameter.grad).all().item()
    assert has_grad


def test_deterministic_outputs_in_eval_mode_with_fixed_seed() -> None:
    config = _ssl_config(masked_fields=("event_type",), next_fields=("event_type",))
    inputs = _make_inputs(batch_size=2, seq_len=4, fill_value=3)
    torch.manual_seed(123)
    model_a = create_ssl_transformer(config)
    model_a.eval()
    torch.manual_seed(123)
    model_b = create_ssl_transformer(config)
    model_b.eval()
    with torch.no_grad():
        out_a = model_a(inputs)
        out_b = model_b(inputs)
    assert torch.allclose(out_a.hidden_states, out_b.hidden_states)
    for key in out_a.masked_logits:
        assert torch.allclose(out_a.masked_logits[key], out_b.masked_logits[key])
    for key in out_a.next_logits:
        assert torch.allclose(out_a.next_logits[key], out_b.next_logits[key])


def test_masked_logits_only_present_when_objective_enabled() -> None:
    config = _ssl_config(enable_masked_field_loss=False)
    model = create_ssl_transformer(config)
    inputs = _make_inputs(batch_size=2, seq_len=4, fill_value=5)
    output = model(inputs)
    assert output.masked_logits == {}
    assert output.next_logits, "next_field heads should still produce logits"


def test_supplying_masked_labels_when_disabled_raises() -> None:
    config = _ssl_config(enable_masked_field_loss=False)
    model = create_ssl_transformer(config)
    inputs = _make_inputs(batch_size=2, seq_len=4, fill_value=5)
    next_labels = {
        field: _label_for_field(
            batch_size=2,
            seq_len=4,
            valid_positions=((0, 0, 5), (1, 1, 5)),
        )
        for field in config.next_fields
    }
    with pytest.raises(ValueError, match="masked_labels"):
        model(
            inputs,
            masked_labels={"event_type": torch.zeros((2, 4), dtype=torch.long)},
            next_labels=next_labels,
        )


def test_ssl_objective_name_enum_values() -> None:
    assert SSLObjectiveName.MASKED_FIELD.value == "masked_field"
    assert SSLObjectiveName.NEXT_FIELD.value == "next_field"
    assert SSLObjectiveName.CONTRASTIVE.value == "contrastive"


def test_n_parameters_positive() -> None:
    model = create_ssl_transformer(_ssl_config())
    assert model.n_parameters() > 0
