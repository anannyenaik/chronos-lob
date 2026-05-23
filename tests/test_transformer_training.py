"""Tests for the supervised transformer training utilities."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from chronoslob.models.transformer import (  # noqa: E402
    MarketTransformerConfig,
    create_market_transformer,
)
from chronoslob.training.token_datasets import TOKEN_WINDOW_FIELD_NAMES  # noqa: E402
from chronoslob.training.transformer_experiment import (  # noqa: E402
    TransformerEpochResult,
    TransformerTrainingConfig,
    evaluate_transformer_classifier,
    fit_transformer_classifier,
    train_transformer_one_epoch,
)

_VOCAB_SIZES = dict.fromkeys(TOKEN_WINDOW_FIELD_NAMES, 8)


def _build_model(*, num_classes: int = 2, dropout: float = 0.0) -> MarketTransformerConfig:
    config = MarketTransformerConfig(
        vocab_sizes=dict(_VOCAB_SIZES),
        field_embedding_dim=4,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        feedforward_dim=16,
        dropout=dropout,
        max_sequence_length=4,
        num_classes=num_classes,
        pooling="mean",
        activation="gelu",
        use_layer_norm=True,
    )
    return config


def _deterministic_batches(
    *,
    n_batches: int = 2,
    batch_size: int = 4,
    seq_len: int = 4,
    num_classes: int = 2,
    seed: int = 0,
) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    batches: list[dict[str, torch.Tensor]] = []
    for batch_index in range(n_batches):
        batch: dict[str, torch.Tensor] = {}
        for field_name in TOKEN_WINDOW_FIELD_NAMES:
            batch[field_name] = torch.randint(
                low=0,
                high=_VOCAB_SIZES[field_name],
                size=(batch_size, seq_len),
                generator=generator,
                dtype=torch.long,
            )
        batch["attention_mask"] = torch.ones(
            (batch_size, seq_len),
            dtype=torch.bool,
        )
        # Deterministic target derived from the batch index keeps the loss
        # finite and bounded for the smoke-style assertion.
        targets = torch.tensor(
            [(batch_index + position) % num_classes for position in range(batch_size)],
            dtype=torch.long,
        )
        batch["y"] = targets
        batches.append(batch)
    return batches


def _loader(batches: Iterable[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    # ``train_transformer_one_epoch`` and ``evaluate_transformer_classifier``
    # accept any iterable of mapping batches, so a plain list is enough here.
    return list(batches)


def test_train_one_step_produces_finite_loss_on_tiny_batch() -> None:
    torch.manual_seed(0)
    config = _build_model(num_classes=2)
    model = create_market_transformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    batches = _deterministic_batches(n_batches=1, batch_size=4, seq_len=4)
    train_loss = train_transformer_one_epoch(
        model,
        _loader(batches),
        optimizer,
        loss_fn,
        device="cpu",
        gradient_clip_norm=1.0,
    )

    assert torch.tensor(train_loss).isfinite().item()
    assert train_loss > 0.0


def test_fit_transformer_classifier_returns_epoch_results() -> None:
    torch.manual_seed(0)
    config = _build_model(num_classes=3, dropout=0.0)
    model = create_market_transformer(config)
    train_batches = _deterministic_batches(
        n_batches=2,
        batch_size=4,
        seq_len=4,
        num_classes=3,
    )
    validation_batches = _deterministic_batches(
        n_batches=1,
        batch_size=4,
        seq_len=4,
        num_classes=3,
        seed=1,
    )
    training_config = TransformerTrainingConfig(
        epochs=2,
        learning_rate=1e-2,
        seed=42,
        device="cpu",
    )
    history = fit_transformer_classifier(
        model,
        _loader(train_batches),
        _loader(validation_batches),
        training_config,
    )
    assert len(history) == 2
    for result in history:
        assert isinstance(result, TransformerEpochResult)
        assert torch.tensor(result.train_loss).isfinite().item()
        assert result.validation_loss is not None
        assert torch.tensor(result.validation_loss).isfinite().item()


def test_evaluate_transformer_classifier_returns_finite_loss_and_metrics() -> None:
    torch.manual_seed(0)
    config = _build_model(num_classes=2)
    model = create_market_transformer(config)
    batches = _deterministic_batches(
        n_batches=2,
        batch_size=4,
        seq_len=4,
        num_classes=2,
    )
    evaluation = evaluate_transformer_classifier(
        model,
        _loader(batches),
        device="cpu",
    )
    assert "metrics" in evaluation
    assert "loss" in evaluation
    assert torch.tensor(evaluation["loss"]).isfinite().item()
    assert evaluation["metrics"]["n_samples"] == 8
    assert evaluation["notes"]


def test_evaluate_classifier_rejects_empty_dataloader() -> None:
    config = _build_model(num_classes=2)
    model = create_market_transformer(config)
    with pytest.raises(ValueError, match="empty dataloader"):
        evaluate_transformer_classifier(model, _loader([]), device="cpu")


def test_train_one_epoch_rejects_empty_dataloader() -> None:
    config = _build_model(num_classes=2)
    model = create_market_transformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    with pytest.raises(ValueError, match="empty dataloader"):
        train_transformer_one_epoch(
            model,
            _loader([]),
            optimizer,
            loss_fn,
            device="cpu",
        )


def test_train_one_epoch_rejects_batch_without_y() -> None:
    config = _build_model(num_classes=2)
    model = create_market_transformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    batches = _deterministic_batches(n_batches=1, num_classes=2)
    del batches[0]["y"]
    with pytest.raises(KeyError, match="'y'"):
        train_transformer_one_epoch(
            model,
            _loader(batches),
            optimizer,
            loss_fn,
            device="cpu",
        )


def test_cuda_device_validation_when_unavailable() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available; cannot exercise the unavailable path here.")
    config = _build_model()
    model = create_market_transformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    batches = _deterministic_batches(n_batches=1, num_classes=2)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        train_transformer_one_epoch(
            model,
            _loader(batches),
            optimizer,
            loss_fn,
            device="cuda",
        )


def test_fit_does_not_write_checkpoints(tmp_path: Path) -> None:
    config = _build_model()
    model = create_market_transformer(config)
    training_config = TransformerTrainingConfig(epochs=1, device="cpu")
    batches = _deterministic_batches(n_batches=1, num_classes=2)
    initial_contents = sorted(tmp_path.iterdir())
    fit_transformer_classifier(
        model,
        _loader(batches),
        None,
        training_config,
    )
    final_contents = sorted(tmp_path.iterdir())
    assert initial_contents == final_contents


def test_training_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="epochs"):
        TransformerTrainingConfig(epochs=0)
    with pytest.raises(ValueError, match="learning_rate"):
        TransformerTrainingConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="weight_decay"):
        TransformerTrainingConfig(weight_decay=-0.1)
    with pytest.raises(ValueError, match="gradient_clip_norm"):
        TransformerTrainingConfig(gradient_clip_norm=0.0)
    with pytest.raises(ValueError, match="device"):
        TransformerTrainingConfig(device="")
