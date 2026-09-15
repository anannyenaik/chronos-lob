"""Tests for the generic torch classification training utilities."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from chronoslob.training.torch_training import (  # noqa: E402
    TorchEpochResult,
    TorchTrainingConfig,
    evaluate_torch_classifier,
    fit_torch_classifier,
    set_torch_deterministic,
    train_one_epoch,
)


class _TinyClassifier(torch.nn.Module):
    def __init__(self, *, n_features: int = 3, n_classes: int = 2) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(n_features, n_classes)

    def forward(self, x: Any) -> Any:
        # Accept [batch, lookback, n_features] and reduce to [batch, n_features].
        if x.ndim == 3:
            x = x.mean(dim=1)
        return self.linear(x)


def _make_batches(
    *,
    n_batches: int = 3,
    batch_size: int = 4,
    lookback: int = 2,
    n_features: int = 3,
    n_classes: int = 2,
    seed: int = 0,
) -> list[dict[str, Any]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    batches: list[dict[str, Any]] = []
    for _ in range(n_batches):
        x = torch.randn(batch_size, lookback, n_features, generator=generator)
        y = torch.randint(0, n_classes, (batch_size,), generator=generator)
        batches.append(
            {
                "x": x,
                "y": y,
                "target_index": torch.arange(batch_size, dtype=torch.long),
                "window_start": torch.arange(batch_size, dtype=torch.long),
                "window_end": torch.arange(batch_size, dtype=torch.long),
            }
        )
    return batches


class _ListLoader:
    def __init__(self, batches: list[dict[str, Any]]) -> None:
        self._batches = batches

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)

    @property
    def dataset(self) -> list[dict[str, Any]]:
        return self._batches


def test_training_config_validates_epochs() -> None:
    with pytest.raises(ValueError, match="epochs must be positive"):
        TorchTrainingConfig(epochs=0)


def test_training_config_validates_learning_rate() -> None:
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        TorchTrainingConfig(learning_rate=0.0)


def test_training_config_validates_gradient_clip_norm() -> None:
    with pytest.raises(ValueError, match="gradient_clip_norm must be positive"):
        TorchTrainingConfig(gradient_clip_norm=0.0)


def test_training_config_validates_seed() -> None:
    with pytest.raises(ValueError, match="seed must be non-negative"):
        TorchTrainingConfig(seed=-1)


def test_set_torch_deterministic_reseeds_python_random() -> None:
    set_torch_deterministic(123)
    sample_one = torch.randn(4).tolist()
    np_sample_one = np.random.rand(4).tolist()

    set_torch_deterministic(123)
    sample_two = torch.randn(4).tolist()
    np_sample_two = np.random.rand(4).tolist()

    assert sample_one == sample_two
    assert np_sample_one == np_sample_two


def test_train_one_epoch_returns_finite_loss() -> None:
    set_torch_deterministic(0)
    model = _TinyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = _ListLoader(_make_batches(seed=1))

    loss = train_one_epoch(
        model,
        loader,
        optimizer,
        loss_fn,
        device="cpu",
        gradient_clip_norm=1.0,
    )

    assert isinstance(loss, float)
    assert np.isfinite(loss)


def test_train_one_epoch_rejects_empty_dataloader() -> None:
    model = _TinyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = _ListLoader([])

    with pytest.raises(ValueError, match="empty dataloader"):
        train_one_epoch(model, loader, optimizer, loss_fn)


def test_train_one_epoch_rejects_invalid_batch_payload() -> None:
    model = _TinyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    bad_loader = _ListLoader([{"x": torch.randn(2, 3, 3)}])  # type: ignore[list-item]

    with pytest.raises(KeyError, match="'x' and 'y'"):
        train_one_epoch(model, bad_loader, optimizer, loss_fn)


def test_gradient_clipping_path_works() -> None:
    set_torch_deterministic(0)
    model = _TinyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    loader = _ListLoader(_make_batches(seed=2))

    loss = train_one_epoch(
        model,
        loader,
        optimizer,
        loss_fn,
        gradient_clip_norm=0.001,
    )

    assert np.isfinite(loss)


def test_evaluate_torch_classifier_returns_metrics() -> None:
    set_torch_deterministic(0)
    model = _TinyClassifier(n_classes=3)
    loader = _ListLoader(_make_batches(n_classes=3, seed=3))

    result = evaluate_torch_classifier(model, loader, device="cpu")

    assert "metrics" in result
    assert "confusion_matrix" in result
    assert "loss" in result
    assert np.isfinite(result["loss"])
    assert result["metrics"]["n_samples"] > 0
    assert len(result["predictions"]) == result["metrics"]["n_samples"]


def test_evaluate_torch_classifier_rejects_empty_loader() -> None:
    model = _TinyClassifier()
    with pytest.raises(ValueError, match="empty dataloader"):
        evaluate_torch_classifier(model, _ListLoader([]))


def test_fit_torch_classifier_returns_epoch_history() -> None:
    set_torch_deterministic(0)
    model = _TinyClassifier(n_classes=3)
    train_loader = _ListLoader(_make_batches(n_classes=3, seed=4))
    validation_loader = _ListLoader(_make_batches(n_classes=3, seed=5))

    history = fit_torch_classifier(
        model,
        train_loader,
        validation_loader,
        TorchTrainingConfig(epochs=2, learning_rate=1e-2),
    )

    assert len(history) == 2
    for epoch_result in history:
        assert isinstance(epoch_result, TorchEpochResult)
        assert epoch_result.train_loss is not None
        assert np.isfinite(epoch_result.train_loss)
        assert epoch_result.validation_loss is not None
        assert np.isfinite(epoch_result.validation_loss)
        assert epoch_result.validation_metrics is not None


def test_fit_torch_classifier_runs_without_validation_loader() -> None:
    set_torch_deterministic(0)
    model = _TinyClassifier()
    train_loader = _ListLoader(_make_batches(seed=6))

    history = fit_torch_classifier(
        model,
        train_loader,
        None,
        TorchTrainingConfig(epochs=1, learning_rate=1e-2),
    )

    assert len(history) == 1
    assert history[0].validation_loss is None
    assert history[0].validation_metrics is None


def test_cpu_device_path_works() -> None:
    set_torch_deterministic(0)
    model = _TinyClassifier()
    train_loader = _ListLoader(_make_batches(seed=7))

    history = fit_torch_classifier(
        model,
        train_loader,
        None,
        TorchTrainingConfig(epochs=1, device="cpu"),
    )

    assert len(history) == 1


def test_unavailable_cuda_raises() -> None:
    if torch.cuda.is_available():  # pragma: no cover - environment-dependent
        pytest.skip("CUDA is available; cannot test the unavailable branch")

    model = _TinyClassifier()
    loader = _ListLoader(_make_batches(seed=8))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        train_one_epoch(model, loader, optimizer, loss_fn, device="cuda")


def test_unknown_device_raises() -> None:
    model = _TinyClassifier()
    loader = _ListLoader(_make_batches(seed=9))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    with pytest.raises(ValueError, match="unsupported device"):
        train_one_epoch(model, loader, optimizer, loss_fn, device="tpu")
