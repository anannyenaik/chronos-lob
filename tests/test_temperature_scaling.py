"""Tests for temperature scaling utilities."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from chronoslob.models.calibration import (  # noqa: E402
    MultiTaskTemperatureScaler,
    TemperatureScaler,
    negative_log_likelihood,
)


def _fixture() -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.tensor(
        [
            [4.0, 0.2, -1.0],
            [0.1, 3.0, 0.0],
            [-1.0, 0.3, 4.5],
            [2.5, 1.0, -0.5],
            [0.0, 2.8, 0.2],
            [0.2, -0.1, 3.2],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    return logits, targets


def test_default_temperature_is_one() -> None:
    scaler = TemperatureScaler()

    assert scaler.temperature == pytest.approx(1.0)


def test_fitted_temperature_remains_positive() -> None:
    logits, targets = _fixture()
    scaler = TemperatureScaler(max_iterations=25)

    scaler.fit(logits, targets)

    assert scaler.temperature > 0.0


def test_transform_logits_does_not_mutate_input() -> None:
    logits, targets = _fixture()
    scaler = TemperatureScaler(max_iterations=25).fit(logits, targets)
    before = logits.clone()

    transformed = scaler.transform_logits(logits)

    assert torch.allclose(logits, before)
    assert transformed is not logits


def test_predict_proba_rows_sum_to_one() -> None:
    logits, targets = _fixture()
    scaler = TemperatureScaler(max_iterations=25).fit(logits, targets)

    probabilities = scaler.predict_proba(logits)

    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(logits.shape[0]))


def test_fit_ignores_ignore_index() -> None:
    logits, targets = _fixture()
    padded_logits = torch.cat(
        [logits, torch.tensor([[100.0, -100.0, 0.0]], dtype=torch.float32)],
        dim=0,
    )
    padded_targets = torch.cat(
        [targets, torch.tensor([-100], dtype=torch.long)],
        dim=0,
    )

    base = TemperatureScaler(max_iterations=25).fit(logits, targets)
    padded = TemperatureScaler(max_iterations=25).fit(padded_logits, padded_targets)

    assert padded.temperature == pytest.approx(base.temperature)


def test_fit_raises_if_no_valid_targets() -> None:
    logits, _ = _fixture()
    targets = torch.full((logits.shape[0],), -100, dtype=torch.long)

    with pytest.raises(ValueError, match="no valid targets"):
        TemperatureScaler().fit(logits, targets)


def test_serialisation_round_trip() -> None:
    logits, targets = _fixture()
    scaler = TemperatureScaler(max_iterations=25, learning_rate=0.2).fit(
        logits,
        targets,
    )

    restored = TemperatureScaler.from_dict(scaler.to_dict())

    assert restored.temperature == pytest.approx(scaler.temperature)
    assert torch.allclose(
        restored.transform_logits(logits),
        scaler.transform_logits(logits),
    )


def test_temperature_scaling_does_not_increase_nll() -> None:
    logits, targets = _fixture()
    scaler = TemperatureScaler(max_iterations=50)

    before = negative_log_likelihood(logits, targets)
    scaled_logits = scaler.fit(logits, targets).transform_logits(logits)
    after = negative_log_likelihood(scaled_logits, targets)

    assert after.item() <= before.item() + 1e-6


def test_multitask_temperature_scaling_fits_per_task() -> None:
    logits, targets = _fixture()
    task_logits = {
        "direction": logits,
        "spread_widening": logits[:, :2],
    }
    task_targets = {
        "direction": targets,
        "spread_widening": torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.long),
    }

    scaler = MultiTaskTemperatureScaler(
        task_names=("direction", "spread_widening"),
        max_iterations=25,
    ).fit(task_logits, task_targets)
    transformed = scaler.transform_logits(task_logits)
    probabilities = scaler.predict_proba(task_logits)

    assert set(scaler.temperatures) == {"direction", "spread_widening"}
    assert all(value > 0.0 for value in scaler.temperatures.values())
    assert set(transformed) == set(task_logits)
    assert torch.allclose(
        probabilities["direction"].sum(dim=-1),
        torch.ones(logits.shape[0]),
    )
    assert torch.allclose(
        probabilities["spread_widening"].sum(dim=-1),
        torch.ones(logits.shape[0]),
    )
