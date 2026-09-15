"""Tests for fixed-length and variable-length sequence batch collation."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from chronoslob.training.batching import (  # noqa: E402
    collate_fixed_length_batch,
    collate_variable_length_batch,
    pad_variable_length_sequences,
)


def _make_sample(
    lookback: int,
    n_features: int,
    target_index: int,
    label: int = 0,
) -> dict[str, object]:
    x = torch.arange(
        target_index * lookback * n_features,
        target_index * lookback * n_features + lookback * n_features,
        dtype=torch.float32,
    ).reshape(lookback, n_features)
    return {
        "x": x,
        "y": torch.tensor(label, dtype=torch.long),
        "target_index": target_index,
        "window_start": target_index - lookback + 1,
        "window_end": target_index,
    }


def test_fixed_length_collate_stacks_shapes() -> None:
    samples = [
        _make_sample(lookback=3, n_features=4, target_index=2, label=0),
        _make_sample(lookback=3, n_features=4, target_index=3, label=1),
        _make_sample(lookback=3, n_features=4, target_index=4, label=2),
    ]

    batch = collate_fixed_length_batch(samples)

    assert tuple(batch["x"].shape) == (3, 3, 4)
    assert tuple(batch["y"].shape) == (3,)
    assert batch["y"].tolist() == [0, 1, 2]
    assert batch["target_index"].tolist() == [2, 3, 4]
    assert batch["window_start"].tolist() == [0, 1, 2]
    assert batch["window_end"].tolist() == [2, 3, 4]


def test_fixed_length_collate_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty"):
        collate_fixed_length_batch([])


def test_fixed_length_collate_rejects_mismatched_feature_shapes() -> None:
    samples = [
        _make_sample(lookback=3, n_features=4, target_index=2),
        _make_sample(lookback=3, n_features=5, target_index=3),
    ]

    with pytest.raises(ValueError, match="mismatched"):
        collate_fixed_length_batch(samples)


def test_fixed_length_collate_rejects_non_scalar_y() -> None:
    sample = _make_sample(lookback=2, n_features=2, target_index=1)
    sample["y"] = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="scalar"):
        collate_fixed_length_batch([sample])


def test_pad_variable_length_returns_expected_mask() -> None:
    sequences = [
        torch.ones((2, 3), dtype=torch.float32),
        torch.zeros((4, 3), dtype=torch.float32),
        torch.full((3, 3), 5.0, dtype=torch.float32),
    ]

    padded, mask = pad_variable_length_sequences(sequences)

    assert tuple(padded.shape) == (3, 4, 3)
    assert tuple(mask.shape) == (3, 4)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
        [True, True, True, False],
    ]
    assert torch.equal(padded[1], sequences[1])
    assert padded[0, 2:, :].sum().item() == 0.0


def test_pad_variable_length_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no sequences"):
        pad_variable_length_sequences([])


def test_pad_variable_length_rejects_mismatched_feature_dim() -> None:
    with pytest.raises(ValueError, match="feature"):
        pad_variable_length_sequences(
            [
                torch.zeros((2, 3), dtype=torch.float32),
                torch.zeros((2, 4), dtype=torch.float32),
            ]
        )


def test_collate_variable_length_returns_x_mask_y_and_indices() -> None:
    sample_a = {
        "x": torch.ones((2, 3), dtype=torch.float32),
        "y": torch.tensor(1, dtype=torch.long),
        "target_index": 1,
        "window_start": 0,
        "window_end": 1,
    }
    sample_b = {
        "x": torch.zeros((4, 3), dtype=torch.float32),
        "y": torch.tensor(0, dtype=torch.long),
        "target_index": 3,
        "window_start": 0,
        "window_end": 3,
    }

    batch = collate_variable_length_batch([sample_a, sample_b])

    assert tuple(batch["x"].shape) == (2, 4, 3)
    assert tuple(batch["mask"].shape) == (2, 4)
    assert batch["mask"].dtype == torch.bool
    assert batch["mask"].tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]
    assert batch["y"].tolist() == [1, 0]
    assert batch["target_index"].tolist() == [1, 3]
    assert batch["window_start"].tolist() == [0, 0]
    assert batch["window_end"].tolist() == [1, 3]


def test_collate_variable_length_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty"):
        collate_variable_length_batch([])
