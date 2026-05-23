"""Tests for supervised multi-task transformer fine-tuning heads."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from chronoslob.models.multitask import (  # noqa: E402
    MultiTaskTransformer,
    MultiTaskTransformerConfig,
    TaskHeadConfig,
    TaskType,
    copy_encoder_weights_from_ssl,
    create_multitask_transformer,
)
from chronoslob.models.ssl import SSLTransformerConfig, create_ssl_transformer  # noqa: E402
from chronoslob.models.transformer import MarketTransformerConfig  # noqa: E402
from chronoslob.training.token_datasets import TOKEN_WINDOW_FIELD_NAMES  # noqa: E402

_VOCAB_SIZES = dict.fromkeys(TOKEN_WINDOW_FIELD_NAMES, 12)


def _backbone(*, dropout: float = 0.0) -> MarketTransformerConfig:
    return MarketTransformerConfig(
        vocab_sizes=dict(_VOCAB_SIZES),
        field_embedding_dim=4,
        model_dim=16,
        num_heads=4,
        num_layers=1,
        feedforward_dim=32,
        dropout=dropout,
        max_sequence_length=8,
        num_classes=2,
        pooling="mean",
        activation="gelu",
        use_layer_norm=True,
    )


def _tasks(
    *,
    direction_weight: float = 1.0,
    spread_weight: float = 1.0,
) -> tuple[TaskHeadConfig, ...]:
    return (
        TaskHeadConfig(
            name="direction",
            task_type=TaskType.CLASSIFICATION,
            num_classes=3,
            loss_weight=direction_weight,
        ),
        TaskHeadConfig(
            name="spread_widening",
            task_type=TaskType.CLASSIFICATION,
            num_classes=2,
            loss_weight=spread_weight,
        ),
    )


def _config(
    *,
    tasks: tuple[TaskHeadConfig, ...] | None = None,
    dropout: float = 0.0,
    freeze_backbone: bool = False,
) -> MultiTaskTransformerConfig:
    return MultiTaskTransformerConfig(
        backbone=_backbone(dropout=dropout),
        tasks=_tasks() if tasks is None else tasks,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )


def _batch(
    *,
    batch_size: int = 3,
    seq_len: int = 4,
    fill_value: int = 5,
) -> dict[str, torch.Tensor]:
    inputs = {
        field_name: torch.full((batch_size, seq_len), fill_value, dtype=torch.long)
        for field_name in TOKEN_WINDOW_FIELD_NAMES
    }
    inputs["attention_mask"] = torch.ones((batch_size, seq_len), dtype=torch.bool)
    return inputs


def _targets() -> dict[str, torch.Tensor]:
    return {
        "direction": torch.tensor([0, 1, 2], dtype=torch.long),
        "spread_widening": torch.tensor([0, 1, 0], dtype=torch.long),
    }


def test_task_head_config_validation() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        TaskHeadConfig(name="")
    with pytest.raises(ValueError, match="num_classes"):
        TaskHeadConfig(name="bad", num_classes=1)
    with pytest.raises(ValueError, match="loss_weight"):
        TaskHeadConfig(name="bad", loss_weight=-0.1)
    with pytest.raises(ValueError, match="ignore_index"):
        TaskHeadConfig(name="bad", num_classes=2, ignore_index=1)


def test_multitask_config_validation() -> None:
    with pytest.raises(ValueError, match="duplicate task name"):
        _config(tasks=(TaskHeadConfig("x"), TaskHeadConfig("x")))
    with pytest.raises(ValueError, match="at least one task"):
        _config(tasks=())
    with pytest.raises(ValueError, match="positive loss_weight"):
        _config(tasks=(TaskHeadConfig("x", loss_weight=0.0),))
    with pytest.raises(ValueError, match="dropout"):
        _config(dropout=1.5)


def test_model_construction_and_parameter_count() -> None:
    model = create_multitask_transformer(_config())
    assert isinstance(model, MultiTaskTransformer)
    assert isinstance(model, torch.nn.Module)
    assert model.n_parameters() > 0
    assert model.n_trainable_parameters() > 0


def test_forward_returns_logits_for_all_tasks() -> None:
    model = create_multitask_transformer(_config())
    output = model(_batch(batch_size=4))

    assert set(output.logits) == {"direction", "spread_widening"}
    assert tuple(output.logits["direction"].shape) == (4, 3)
    assert tuple(output.logits["spread_widening"].shape) == (4, 2)
    assert output.loss is None


def test_forward_missing_required_token_field_raises() -> None:
    model = create_multitask_transformer(_config())
    inputs = _batch()
    del inputs["source"]

    with pytest.raises(KeyError, match="'source'"):
        model(inputs)


def test_loss_is_finite_and_components_returned() -> None:
    model = create_multitask_transformer(_config())
    output = model(_batch(), targets=_targets())

    assert output.loss is not None
    assert torch.isfinite(output.loss).item()
    assert set(output.loss_components) == {"direction", "spread_widening"}
    assert output.valid_counts == {"direction": 3, "spread_widening": 3}


def test_missing_targets_are_ignored_with_ignore_index_and_mask() -> None:
    model = create_multitask_transformer(_config())
    targets = _targets()
    targets["spread_widening"] = torch.tensor([0, -100, -100], dtype=torch.long)
    target_mask = {
        "direction": torch.tensor([True, True, True], dtype=torch.bool),
        "spread_widening": torch.tensor([True, False, False], dtype=torch.bool),
    }

    output = model(_batch(), targets=targets, target_mask=target_mask)

    assert output.valid_counts["spread_widening"] == 1
    assert "spread_widening" in output.loss_components


def test_task_with_all_missing_targets_is_skipped() -> None:
    model = create_multitask_transformer(_config())
    targets = _targets()
    targets["spread_widening"] = torch.full((3,), -100, dtype=torch.long)

    output = model(_batch(), targets=targets)

    assert output.valid_counts["spread_widening"] == 0
    assert "direction" in output.loss_components
    assert "spread_widening" not in output.loss_components


def test_all_tasks_missing_raises_when_loss_requested() -> None:
    model = create_multitask_transformer(_config())
    targets = {
        "direction": torch.full((3,), -100, dtype=torch.long),
        "spread_widening": torch.full((3,), -100, dtype=torch.long),
    }

    with pytest.raises(ValueError, match="no task with valid"):
        model(_batch(), targets=targets)


def test_loss_weights_affect_combined_loss() -> None:
    torch.manual_seed(7)
    model_a = create_multitask_transformer(
        _config(tasks=_tasks(direction_weight=1.0, spread_weight=1.0))
    )
    model_b = create_multitask_transformer(
        _config(tasks=_tasks(direction_weight=2.0, spread_weight=0.5))
    )
    model_b.load_state_dict(model_a.state_dict())
    inputs = _batch()
    targets = _targets()

    output_a = model_a(inputs, targets=targets)
    output_b = model_b(inputs, targets=targets)

    assert output_a.loss is not None
    assert output_b.loss is not None
    expected_b = (
        2.0 * output_a.loss_components["direction"]
        + 0.5 * output_a.loss_components["spread_widening"]
    )
    assert torch.allclose(output_b.loss, expected_b, atol=1e-6)


def test_freeze_backbone_disables_backbone_gradients() -> None:
    model = create_multitask_transformer(_config(freeze_backbone=True))

    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.task_heads.parameters())


def test_backward_pass_produces_gradients_for_task_heads() -> None:
    model = create_multitask_transformer(_config())
    output = model(_batch(), targets=_targets())
    assert output.loss is not None

    output.loss.backward()

    head_gradients = [
        parameter.grad
        for parameter in model.task_heads.parameters()
        if parameter.requires_grad
    ]
    assert head_gradients
    assert all(gradient is not None for gradient in head_gradients)
    assert all(torch.isfinite(gradient).all().item() for gradient in head_gradients)


def test_eval_outputs_are_deterministic_under_fixed_seed() -> None:
    inputs = _batch()
    torch.manual_seed(123)
    model_a = create_multitask_transformer(_config(dropout=0.0))
    model_a.eval()
    torch.manual_seed(123)
    model_b = create_multitask_transformer(_config(dropout=0.0))
    model_b.eval()

    with torch.no_grad():
        output_a = model_a(inputs)
        output_b = model_b(inputs)

    for task_name in model_a.task_names:
        assert torch.allclose(output_a.logits[task_name], output_b.logits[task_name])


def test_copy_encoder_weights_from_ssl_copies_matching_backbone() -> None:
    multitask_model = create_multitask_transformer(_config())
    ssl_model = create_ssl_transformer(SSLTransformerConfig(transformer=_backbone()))
    with torch.no_grad():
        for parameter in ssl_model.encoder.parameters():
            parameter.add_(0.25)

    copy_encoder_weights_from_ssl(multitask_model, ssl_model)

    for key, value in multitask_model.encoder.state_dict().items():
        assert torch.allclose(value, ssl_model.encoder.state_dict()[key])
