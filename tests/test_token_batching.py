"""Tests for token-window batching helpers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from chronoslob.models.tokenisation import SPECIAL_TOKEN_IDS, SpecialToken  # noqa: E402
from chronoslob.training.token_batching import (  # noqa: E402
    collate_token_windows,
    pad_variable_length_token_windows,
)
from chronoslob.training.token_datasets import TOKEN_WINDOW_FIELD_NAMES  # noqa: E402


def _sample(
    length: int,
    value: int,
    *,
    attention_mask: list[bool] | None = None,
) -> dict[str, object]:
    sample: dict[str, object] = {
        field_name: torch.full((length,), value, dtype=torch.long)
        for field_name in TOKEN_WINDOW_FIELD_NAMES
    }
    if attention_mask is None:
        sample["attention_mask"] = torch.ones((length,), dtype=torch.bool)
    else:
        sample["attention_mask"] = torch.tensor(attention_mask, dtype=torch.bool)
    sample["window_start"] = 0
    sample["window_end"] = length - 1
    sample["anchor_index"] = length - 1
    return sample


def test_collate_token_windows_stacks_fixed_length_windows() -> None:
    batch = collate_token_windows([_sample(3, 7), _sample(3, 8)])

    for field_name in TOKEN_WINDOW_FIELD_NAMES:
        assert batch[field_name].shape == (2, 3)
        assert batch[field_name].dtype == torch.long
    assert batch["attention_mask"].shape == (2, 3)
    assert batch["attention_mask"].dtype == torch.bool
    assert batch["window_start"].tolist() == [0, 0]
    assert batch["window_end"].tolist() == [2, 2]
    assert batch["anchor_index"].tolist() == [2, 2]


def test_collate_token_windows_preserves_attention_masks() -> None:
    sample_a = _sample(3, 7, attention_mask=[False, True, True])
    sample_b = _sample(3, 8, attention_mask=[True, True, False])

    batch = collate_token_windows([sample_a, sample_b])

    assert batch["attention_mask"].tolist() == [
        [False, True, True],
        [True, True, False],
    ]


def test_collate_token_windows_batches_all_categorical_fields_consistently() -> None:
    batch = collate_token_windows([_sample(2, 4), _sample(2, 5)])

    expected_shape = batch["event_type"].shape
    for field_name in TOKEN_WINDOW_FIELD_NAMES:
        assert batch[field_name].shape == expected_shape


def test_pad_variable_length_token_windows() -> None:
    pad_id = SPECIAL_TOKEN_IDS[SpecialToken.PAD]
    batch = pad_variable_length_token_windows(
        [
            _sample(2, 7, attention_mask=[True, False]),
            _sample(4, 8, attention_mask=[True, True, True, True]),
        ]
    )

    for field_name in TOKEN_WINDOW_FIELD_NAMES:
        assert batch[field_name].shape == (2, 4)
        assert batch[field_name][0].tolist() == [7, 7, pad_id, pad_id]
        assert batch[field_name][1].tolist() == [8, 8, 8, 8]
    assert batch["attention_mask"].tolist() == [
        [True, False, False, False],
        [True, True, True, True],
    ]


def test_pad_variable_length_token_windows_left_padding() -> None:
    pad_id = SPECIAL_TOKEN_IDS[SpecialToken.PAD]
    batch = pad_variable_length_token_windows(
        [_sample(2, 7), _sample(3, 8)],
        padding_side="left",
    )

    assert batch["event_type"][0].tolist() == [pad_id, 7, 7]
    assert batch["attention_mask"][0].tolist() == [False, True, True]


def test_collate_token_windows_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        collate_token_windows([])
