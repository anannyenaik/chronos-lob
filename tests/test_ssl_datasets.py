"""Tests for SSL masking, next-field targets and dataset helpers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from chronoslob.models.ssl import MaskingConfig  # noqa: E402
from chronoslob.models.tokenisation import (  # noqa: E402
    SPECIAL_TOKEN_IDS,
    SpecialToken,
    TokenisationConfig,
    tokenise_event_log,
)
from chronoslob.training.ssl_datasets import (  # noqa: E402
    DEFAULT_IGNORE_INDEX,
    MaskedTokenBatch,
    MaskingPolicy,
    SSLTokenSequenceDataset,
    apply_field_masking,
    build_next_field_targets,
    collate_ssl_token_windows,
)
from chronoslob.training.token_datasets import (  # noqa: E402
    TOKEN_WINDOW_FIELD_NAMES,
    TokenSequenceDataset,
    TokenWindowConfig,
)
from chronoslob.utils.paths import project_root  # noqa: E402

_FIXTURE = (
    project_root() / "tests" / "fixtures" / "event_logs" / "synthetic_snapshots.jsonl"
)

_VOCAB_SIZES = dict.fromkeys(TOKEN_WINDOW_FIELD_NAMES, 12)

_PAD_ID = int(SPECIAL_TOKEN_IDS[SpecialToken.PAD])
_MASK_ID = int(SPECIAL_TOKEN_IDS[SpecialToken.MASK])


def _make_inputs(
    *,
    batch_size: int = 2,
    seq_len: int = 4,
    fill_value: int = 5,
    left_pad: int = 0,
) -> dict[str, torch.Tensor]:
    inputs: dict[str, torch.Tensor] = {}
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    if left_pad > 0:
        attention_mask[:, :left_pad] = False
    for field in TOKEN_WINDOW_FIELD_NAMES:
        tensor = torch.full((batch_size, seq_len), fill_value, dtype=torch.long)
        if left_pad > 0:
            tensor[:, :left_pad] = _PAD_ID
        inputs[field] = tensor
    inputs["attention_mask"] = attention_mask
    return inputs


def _generator(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_masking_never_selects_padding_positions() -> None:
    inputs = _make_inputs(batch_size=4, seq_len=6, fill_value=5, left_pad=2)
    policy = MaskingPolicy(
        fields=("event_type",),
        vocab_sizes=_VOCAB_SIZES,
        config=MaskingConfig(mask_probability=0.99),
    )
    corrupted, labels = apply_field_masking(inputs, policy, generator=_generator(7))
    # Labels at padded positions must remain ignore_index
    assert (labels["event_type"][:, :2] == DEFAULT_IGNORE_INDEX).all().item()
    # Padded positions should still hold [PAD]
    assert (corrupted["event_type"][:, :2] == _PAD_ID).all().item()


def test_masking_respects_attention_mask() -> None:
    inputs = _make_inputs(batch_size=2, seq_len=4, fill_value=5, left_pad=1)
    policy = MaskingPolicy(
        fields=("side",),
        vocab_sizes=_VOCAB_SIZES,
        config=MaskingConfig(mask_probability=1.0, force_at_least_one_mask=False),
    )
    _, labels = apply_field_masking(inputs, policy, generator=_generator(1))
    # All real positions must receive a label; padded position must be ignored.
    assert (labels["side"][:, 0] == DEFAULT_IGNORE_INDEX).all().item()
    assert (labels["side"][:, 1:] != DEFAULT_IGNORE_INDEX).all().item()


def test_masking_uses_mask_token_for_replacement_when_probability_one() -> None:
    inputs = _make_inputs(batch_size=2, seq_len=4, fill_value=5)
    policy = MaskingPolicy(
        fields=("event_type",),
        vocab_sizes=_VOCAB_SIZES,
        config=MaskingConfig(
            mask_probability=1.0,
            mask_token_probability=1.0,
            random_token_probability=0.0,
            keep_token_probability=0.0,
        ),
    )
    corrupted, labels = apply_field_masking(inputs, policy, generator=_generator(0))
    assert (corrupted["event_type"] == _MASK_ID).all().item()
    # Every position is masked, so every label equals the original value
    assert (labels["event_type"] == 5).all().item()


def test_masking_unmasked_label_positions_are_ignore_index() -> None:
    inputs = _make_inputs(batch_size=3, seq_len=4, fill_value=5)
    policy = MaskingPolicy(
        fields=("event_type",),
        vocab_sizes=_VOCAB_SIZES,
        config=MaskingConfig(
            mask_probability=0.1,
            force_at_least_one_mask=False,
        ),
    )
    _, labels = apply_field_masking(inputs, policy, generator=_generator(2))
    # Every label that isn't ignore_index must equal the original (5)
    valid = labels["event_type"] != DEFAULT_IGNORE_INDEX
    if valid.any().item():
        assert (labels["event_type"][valid] == 5).all().item()


def test_deterministic_masking_with_fixed_seed() -> None:
    inputs = _make_inputs(batch_size=2, seq_len=4, fill_value=5)
    policy = MaskingPolicy(
        fields=("event_type",),
        vocab_sizes=_VOCAB_SIZES,
        config=MaskingConfig(),
    )
    first = apply_field_masking(inputs, policy, generator=_generator(13))
    second = apply_field_masking(inputs, policy, generator=_generator(13))
    assert torch.equal(first[0]["event_type"], second[0]["event_type"])
    assert torch.equal(first[1]["event_type"], second[1]["event_type"])


def test_force_at_least_one_mask_when_no_position_selected() -> None:
    inputs = _make_inputs(batch_size=2, seq_len=4, fill_value=5)
    policy = MaskingPolicy(
        fields=("event_type",),
        vocab_sizes=_VOCAB_SIZES,
        config=MaskingConfig(
            mask_probability=0.0,
            mask_token_probability=1.0,
            random_token_probability=0.0,
            keep_token_probability=0.0,
            force_at_least_one_mask=True,
        ),
    )
    corrupted, labels = apply_field_masking(inputs, policy, generator=_generator(5))
    # At least one position per row is masked
    masked_per_row = (corrupted["event_type"] == _MASK_ID).sum(dim=1)
    assert (masked_per_row >= 1).all().item()
    # And the matching label is set to the original (5)
    has_label_per_row = (labels["event_type"] != DEFAULT_IGNORE_INDEX).any(dim=1)
    assert has_label_per_row.all().item()


def test_apply_field_masking_does_not_mutate_inputs() -> None:
    inputs = _make_inputs(batch_size=2, seq_len=4, fill_value=5)
    snapshot = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    policy = MaskingPolicy(
        fields=("event_type", "side"),
        vocab_sizes=_VOCAB_SIZES,
        config=MaskingConfig(),
    )
    apply_field_masking(inputs, policy, generator=_generator(3))
    for key, original in snapshot.items():
        assert torch.equal(inputs[key], original)


def test_apply_field_masking_rejects_pad_position_with_non_pad_token() -> None:
    inputs = _make_inputs(batch_size=1, seq_len=2, fill_value=5)
    # Mark position 0 as padding but leave the token as non-PAD value.
    inputs["attention_mask"][0, 0] = False
    policy = MaskingPolicy(
        fields=("event_type",),
        vocab_sizes=_VOCAB_SIZES,
        config=MaskingConfig(),
    )
    with pytest.raises(ValueError, match="non-\\[PAD\\] tokens"):
        apply_field_masking(inputs, policy, generator=_generator(0))


def test_build_next_field_targets_shifts_by_one() -> None:
    inputs = _make_inputs(batch_size=1, seq_len=4, fill_value=5)
    inputs["event_type"] = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    targets = build_next_field_targets(
        inputs, ("event_type",), ignore_index=-100
    )
    # Pairs (0,1), (1,2), (2,3) are valid; position 3 has no t+1.
    assert torch.equal(
        targets["event_type"], torch.tensor([[2, 3, 4, -100]], dtype=torch.long)
    )


def test_build_next_field_targets_final_real_token_is_ignored() -> None:
    inputs = _make_inputs(batch_size=1, seq_len=3, fill_value=5)
    inputs["event_type"] = torch.tensor([[7, 8, 9]], dtype=torch.long)
    targets = build_next_field_targets(
        inputs, ("event_type",), ignore_index=-100
    )
    assert targets["event_type"][0, -1].item() == -100


def test_build_next_field_targets_handles_padding_before_data() -> None:
    inputs = _make_inputs(batch_size=1, seq_len=4, fill_value=5, left_pad=2)
    inputs["event_type"] = torch.tensor([[0, 0, 7, 8]], dtype=torch.long)
    targets = build_next_field_targets(
        inputs, ("event_type",), ignore_index=-100
    )
    # Position 0 is padding -> ignored. Position 1 is padding -> ignored.
    # Position 2 is real, position 3 is real, so target[2]=8.
    # Position 3 is the final position -> ignored.
    assert targets["event_type"][0, 0].item() == -100
    assert targets["event_type"][0, 1].item() == -100
    assert targets["event_type"][0, 2].item() == 8
    assert targets["event_type"][0, 3].item() == -100


def test_build_next_field_targets_pair_requires_both_real() -> None:
    inputs = _make_inputs(batch_size=1, seq_len=4, fill_value=5)
    inputs["event_type"] = torch.tensor([[7, 8, 9, 10]], dtype=torch.long)
    # Mark position 2 as padding -> pair (1, 2) and pair (2, 3) become invalid.
    inputs["attention_mask"][0, 2] = False
    inputs["event_type"][0, 2] = _PAD_ID
    targets = build_next_field_targets(
        inputs, ("event_type",), ignore_index=-100
    )
    assert targets["event_type"][0, 0].item() == 8
    assert targets["event_type"][0, 1].item() == -100
    assert targets["event_type"][0, 2].item() == -100
    assert targets["event_type"][0, 3].item() == -100


def _build_dataset() -> TokenSequenceDataset:
    config = TokenisationConfig(max_levels_per_side=2)
    sequence = tokenise_event_log(_FIXTURE, config, symbol="TESTUSDT")
    return TokenSequenceDataset(
        sequence,
        TokenWindowConfig(window_length=4, padding_side="left"),
    )


def test_ssl_dataset_emits_consistent_field_names() -> None:
    base = _build_dataset()
    policy = MaskingPolicy(
        fields=("event_type", "side"),
        vocab_sizes={
            field: base.sequence.field_sizes[field]
            for field in ("event_type", "side")
        },
        config=MaskingConfig(),
    )
    dataset = SSLTokenSequenceDataset(
        base,
        policy,
        next_fields=("event_type",),
        base_seed=0,
    )
    sample = dataset[0]
    assert sample["masked_field_names"] == ("event_type", "side")
    assert sample["next_field_names"] == ("event_type",)
    assert set(sample["inputs"]).issuperset(set(TOKEN_WINDOW_FIELD_NAMES))


def test_ssl_dataset_collation_yields_masked_token_batch() -> None:
    base = _build_dataset()
    policy = MaskingPolicy(
        fields=("event_type",),
        vocab_sizes={"event_type": base.sequence.field_sizes["event_type"]},
        config=MaskingConfig(),
    )
    dataset = SSLTokenSequenceDataset(
        base,
        policy,
        next_fields=("event_type",),
        base_seed=0,
    )
    samples = [dataset[i] for i in range(min(2, len(dataset)))]
    batch = collate_ssl_token_windows(samples)
    assert isinstance(batch, MaskedTokenBatch)
    for field_name in TOKEN_WINDOW_FIELD_NAMES:
        assert batch.inputs[field_name].shape[0] == len(samples)
    assert "event_type" in batch.masked_labels
    assert "event_type" in batch.next_labels


def test_ssl_dataset_collate_rejects_mismatched_field_sets() -> None:
    base = _build_dataset()
    policy_a = MaskingPolicy(
        fields=("event_type",),
        vocab_sizes={"event_type": base.sequence.field_sizes["event_type"]},
        config=MaskingConfig(),
    )
    policy_b = MaskingPolicy(
        fields=("side",),
        vocab_sizes={"side": base.sequence.field_sizes["side"]},
        config=MaskingConfig(),
    )
    dataset_a = SSLTokenSequenceDataset(
        base,
        policy_a,
        next_fields=("event_type",),
        base_seed=0,
    )
    dataset_b = SSLTokenSequenceDataset(
        base,
        policy_b,
        next_fields=("side",),
        base_seed=0,
    )
    with pytest.raises(ValueError, match="masked_field_names"):
        collate_ssl_token_windows([dataset_a[0], dataset_b[0]])


def test_ssl_dataset_deterministic_under_fixed_seed() -> None:
    base = _build_dataset()
    policy = MaskingPolicy(
        fields=("event_type",),
        vocab_sizes={"event_type": base.sequence.field_sizes["event_type"]},
        config=MaskingConfig(),
    )
    dataset_a = SSLTokenSequenceDataset(
        base, policy, next_fields=("event_type",), base_seed=42
    )
    dataset_b = SSLTokenSequenceDataset(
        base, policy, next_fields=("event_type",), base_seed=42
    )
    sample_a = dataset_a[0]
    sample_b = dataset_b[0]
    assert torch.equal(
        sample_a["inputs"]["event_type"], sample_b["inputs"]["event_type"]
    )
    assert torch.equal(
        sample_a["masked_labels"]["event_type"],
        sample_b["masked_labels"]["event_type"],
    )


def test_ssl_dataset_requires_at_least_one_objective() -> None:
    base = _build_dataset()
    policy = MaskingPolicy(
        fields=("event_type",),
        vocab_sizes={"event_type": base.sequence.field_sizes["event_type"]},
        config=MaskingConfig(),
    )
    with pytest.raises(ValueError, match="at least one"):
        SSLTokenSequenceDataset(
            base,
            policy,
            next_fields=("event_type",),
            base_seed=0,
            enable_masking=False,
            enable_next_targets=False,
        )


def test_collate_ssl_token_windows_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        collate_ssl_token_windows([])
