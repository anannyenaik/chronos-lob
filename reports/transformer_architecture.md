# Transformer Architecture

Phase 12 adds a small supervised transformer encoder over the field-wise
token batches prepared in Phase 11. It is strictly an architecture and
plumbing phase: no self-supervised pretraining, masked event modelling,
next-event prediction, calibration, execution simulation, backtesting or
market-signal claim is implemented here.

## Purpose

The encoder gives later phases a clean, auditable foundation for sequence
modelling over canonical limit order book events and snapshot-derived
pseudo-level tokens. The supervised classification head is intentionally
generic so the same model can be wired to real labels (e.g. mid-price
direction, spread-widening labels) in a later, reproducible experiment.

## Input Contract from Phase 11

The encoder consumes the exact field-wise dictionaries produced by
[chronoslob.training.token_datasets.TokenSequenceDataset](../chronoslob/training/token_datasets.py)
and [chronoslob.training.token_batching.collate_token_windows](../chronoslob/training/token_batching.py).

Required keys per batch are:

```
event_type:        LongTensor[batch, seq_len]
side:              LongTensor[batch, seq_len]
price_bucket:      LongTensor[batch, seq_len]
quantity_bucket:   LongTensor[batch, seq_len]
time_delta_bucket: LongTensor[batch, seq_len]
context_bucket:    LongTensor[batch, seq_len]
source:            LongTensor[batch, seq_len]
attention_mask:    BoolTensor[batch, seq_len]   # True = real token
```

`MarketTransformerEncoder` validates the presence, dtype and shape of
every field and raises clear errors on mismatch. The token-field names
are loaded from `TOKEN_WINDOW_FIELD_NAMES` so the encoder and the
tokeniser cannot drift apart.

## Field-Wise Embedding Design

Each categorical field has its own `nn.Embedding` with `padding_idx`
fixed to the reserved `[PAD] = 0` token ID. The seven field embeddings
are concatenated along the feature dimension and then linearly projected
to `model_dim`. Concatenation plus projection is preferred over summation
because it preserves the contribution of each token field and keeps the
projection layer easy to inspect.

The `TokenFieldEmbeddingConfig` validates that every required token
field has a positive vocabulary size and that no unknown fields are
supplied. Production use should source `vocab_sizes` from the frozen
`TokenVocabulary.field_sizes()` returned by Phase 11 to keep the model
aligned with the train-only vocabulary.

## Positional Embedding Design

Positions are represented with a single trainable absolute embedding
table sized to `max_sequence_length`. Learned absolute positions are
sufficient for fixed-length supervised classification windows and keep
the implementation small. Sinusoidal or relative-position alternatives
are deferred to a later phase if needed.

The positional embedding raises if the input sequence is longer than
`max_sequence_length`. This protects callers from silently truncating
historical context.

## Transformer Encoder

The core is a standard `torch.nn.TransformerEncoder` built from
`TransformerEncoderLayer` blocks with:

* `batch_first=True`
* configurable `num_heads`, `num_layers`, `feedforward_dim`, `dropout`
* `activation` chosen from `relu` or `gelu`
* optional `LayerNorm` after the stack when `use_layer_norm=True`

`model_dim` must be divisible by `num_heads`; this is checked at config
construction time. The encoder is non-autoregressive: no causal mask is
applied because the model is an encoder over a fixed historical window,
not a generator.

## Attention Mask Convention

Phase 11 emits an `attention_mask` where `True` marks a real token.
PyTorch's `TransformerEncoder` expects `src_key_padding_mask` where
`True` marks padding that should be ignored. The encoder converts
between the two conventions internally (`key_padding_mask = ~attention_mask`)
and tests cover the conversion explicitly.

Fully padded sequences (every position is padding) are explicitly
rejected because `nn.MultiheadAttention` produces NaN outputs when no
real tokens are present. The error message states the problem clearly so
that misconfigured collation surfaces early.

## Pooling Choices

`TransformerPooling` supports three mask-aware strategies:

* `mean` (default): average over the non-padding positions only.
* `last`: select the last real token in each row. With left-padding (the
  Phase 11 default), this matches the most recent event in the window.
* `bos`: select the first real token in each row. This is useful only
  when the tokenisation includes a `[BOS]` summary, which is opt-in via
  `TokenisationConfig.include_bos`.

`mean` is the default because it is robust on snapshot-derived windows
where every token contributes information.

## Supervised Classification Head

A single `nn.Linear(model_dim, num_classes)` produces logits over the
configured label space. Cross-entropy is the loss used by the training
utilities. The classification head is intentionally minimal so that
future phases can swap or extend it without touching the encoder body.

## Smoke Training

`run_transformer_smoke_from_event_log` reads a local canonical event
log, tokenises it with Phase 11 defaults, builds fixed-length windows,
generates synthetic labels and trains the encoder for one epoch on CPU.
Both the CLI smoke command (`python -m chronoslob.cli run-transformer-smoke`)
and the in-process API surface a clear `notes` field stating that the
labels are synthetic plumbing only.

## Why Smoke Labels Are Synthetic

The smoke labels are derived from `(sample_index + last_real_token_side_id) mod num_classes`.
This rule is deterministic and trivially separable so the encoder can be
exercised end to end without depending on any real label engine. The
smoke metric reported (loss, optional accuracy) measures only that the
plumbing is wired correctly; it is not a forecast quality measurement.

## Why This Is Not an Alpha or Backtest Result

* No real market labels are used.
* No execution model, slippage, queue position or fill probability is
  modelled.
* No realised PnL, Sharpe ratio or trade list is computed.
* The bundled JSONL fixture is a synthetic engineering example, not
  real venue data.

ChronosLOB keeps prediction and tradability conceptually separate. A
working transformer encoder does not, on its own, imply tradable signal.

## Why SSL Objectives Are Left for Phase 13

Self-supervised pretraining (masked event modelling, masked side
prediction, masked price/quantity bucket prediction, next-event
prediction, contrastive learning, future-state reconstruction) requires
its own design notes, masking utilities, evaluation protocol and report.
Mixing it into Phase 12 would entangle architecture review with
objective review. Phase 13 will introduce SSL objectives on top of the
encoder defined here.

## Limitations and Next Steps

* Positional encoding is learned and absolute. Sinusoidal and relative
  alternatives are deferred.
* Only mean, last and BOS pooling are implemented. Attention pooling,
  CLS-token pooling and weighted pooling are deferred.
* The model does not return attention weights; attribution analysis is
  a later phase.
* CUDA is best-effort. CPU is the primary supported device for tests
  and the smoke runner.
* No checkpoint writing, TensorBoard, Weights & Biases or distributed
  training is implemented.
* No backtest, execution-aware validation or PnL is implemented.
* Smoke labels do not measure forecast skill. Real-label experiments
  belong in a later phase that integrates the existing leakage-safe
  splitters and label engine.
