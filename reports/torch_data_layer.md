# PyTorch sequence data layer

This note documents the sequence-window data layer introduced in Phase 7A.
It uses UK English and is intentionally narrow: it covers the indexing
convention, the train-only standardisation surface and the batching
format, and it explicitly states what is **not** yet implemented.

## Sequence-window convention

A supervised sequence sample is described by a `SequenceSampleIndex` with
three integer fields:

- `window_start`: inclusive row index of the first feature row.
- `window_end`: inclusive row index of the last feature row.
- `target_index`: row index that supplies the label.

For past-only supervised windows the dataset enforces
`window_end == target_index`. This invariant is the load-bearing guarantee
that no row after the label timestamp leaks into the feature window. A
sample's feature window is therefore exactly the rows
`[target_index - lookback + 1, ..., target_index]`.

## Target alignment

`SequenceDataset` aligns the user-provided `feature_frame` and
`label_frame` via
`chronoslob.models.preprocessing.align_feature_label_frames`. The join
keys default to `(timestamp, symbol)` and the join validates that the
match is one-to-one. The target value for a sample is read from the
aligned row whose index equals `target_index`. Because the dataset never
reads a row beyond `target_index`, the target and its feature window
always share the same row position in the aligned frame.

## No-look-ahead rule

Three guards combine to keep windows past-only:

1. `SequenceSampleIndex.__post_init__` rejects any sample whose
   `window_end` does not equal `target_index`.
2. `build_sequence_indices` constructs candidate target indices as
   `range(lookback - 1, n_rows)` and computes
   `window_start = target_index - lookback + 1`, so the implementation
   cannot construct a forward-looking window even if a user supplies an
   `allowed_target_indices` set.
3. When `require_contiguous_indices` is true and an allowed set is
   supplied, every row in the window must also belong to that set. This
   stops a validation or test sample from silently consuming training
   rows.

## Split-contained windows

`build_dataloaders_for_split` builds one `SequenceDataset` per partition
using `allowed_target_indices` from `SplitIndices`. By default the helper
also enforces `require_contiguous_indices=True`, so each partition's
samples consume only that partition's rows. This is more conservative
than strictly required: a future configuration option could let a
validation window draw training rows as priming context, but the default
keeps partitions cleanly separated.

## Train-only standardisation

`TorchSequenceStandardiser` is a deliberately small surface for
standardising sequence feature tensors:

- `fit_from_feature_frame(frame, feature_columns)` computes mean and std
  from the supplied training-only frame.
- `fit_from_sequences(x)` computes mean and std from a stacked tensor of
  training feature windows.
- `transform_tensor(x)` applies the fitted statistics to a new tensor and
  returns a copy.
- `fit_transform_tensor(x)` fits and transforms in one call.

Zero standard deviations are replaced with `1.0` so downstream models
see numerically stable inputs without raising. Calling `transform_tensor`
before `fit_*` raises `ValueError`. Inputs are never mutated. The class
deliberately does **not** standardise inside `SequenceDataset` so the
caller can choose whether to apply scaling at all and so the train-only
fitting boundary is explicit.

## Batching format

`collate_fixed_length_batch` stacks samples into a mapping with the
following keys:

| key            | shape                            | dtype          |
|----------------|----------------------------------|----------------|
| `x`            | `[batch, lookback, n_features]`  | dataset dtype  |
| `y`            | `[batch]`                        | `torch.long`   |
| `target_index` | `[batch]`                        | `torch.long`   |
| `window_start` | `[batch]`                        | `torch.long`   |
| `window_end`   | `[batch]`                        | `torch.long`   |

`pad_variable_length_sequences` and `collate_variable_length_batch` are
provided for future event-stream models. They return an extra boolean
`mask` of shape `[batch, max_len]` where `True` marks valid tokens.

## DataLoader defaults

`DataLoaderConfig.shuffle` defaults to `False`. The
`create_sequence_dataloader` helper threads that default through to
`torch.utils.data.DataLoader` and always installs
`collate_fixed_length_batch`. `build_dataloaders_for_split` returns
validation and test loaders with `shuffle=False` and `drop_last=False`
regardless of the supplied loader configuration, so evaluation never
reshuffles temporal order.

## Class mapping

The class-to-index mapping is inferred from training rows only inside
`build_dataloaders_for_split` and is then reused for validation and test
loaders. Unseen validation or test classes raise `ValueError` rather
than being silently re-mapped. The CLI smoke command on the tiny
synthetic fixture passes an explicit full-frame mapping because the
fixture is too small for the training partition to observe every label
class; this is documented inside the smoke command itself.

## What is not yet implemented

- No DeepLOB, CNN, LSTM or transformer architectures.
- No supervised neural training loop.
- No self-supervised pretraining objectives.
- No checkpointing or run output artefacts.
- No GPU-specific paths beyond `torch.utils.data.DataLoader`.
- No execution backtest or PnL evaluation.

These belong in later phases. Phase 7A only delivers the data layer
required by future supervised sequence models.

## Why this phase does not train models

Most leakage bugs in financial machine learning are introduced at the
data layer. Building and reviewing the indexing, alignment and
partition-containment rules in isolation, without a model architecture
to distract from them, keeps the audit surface small and the
guarantees inspectable. Subsequent phases can then add models against a
data layer whose invariants are already tested.
