# Self-Supervised Objectives

Phase 13 adds self-supervised pretraining objectives on top of the
Phase 11 tokenisation layer and the Phase 12 transformer encoder. It is
strictly pretraining infrastructure: it implements no supervised market
labels, no calibration, no execution simulation, no backtesting and no
benchmark or alpha claim. Smoke training paths use synthetic event-log
fixtures and the produced losses are a plumbing check, not market
evidence.

## Purpose

Self-supervised pretraining lets the model learn structural properties
of tokenised market microstructure (which events follow which, how
prices and quantities co-vary, how spread regimes behave) before any
supervised label is introduced. Targets are derived from the token
sequence itself, so the objective is fully reproducible from public
canonical event logs and never leaks future market information into the
training signal.

The objectives in this phase deliberately mirror standard sequence
modelling conventions (masked field modelling, next-step prediction) and
are intentionally agnostic to any trading task. Calibration, execution
viability and tradability are treated as separate downstream concerns
and are not addressed here.

## How Masked Field Modelling Works

For each token window we deterministically select a subset of
non-padding positions in the configured token fields (by default
``event_type``, ``side``, ``price_bucket`` and ``quantity_bucket``).
The selection is BERT-style:

* Each valid (non-padding) position is independently selected with
  probability ``mask_probability`` (default 0.15).
* When ``force_at_least_one_mask=True`` and a row would otherwise have
  no masked position, exactly one valid position is forced to be
  masked. This keeps smoke training stable on very small windows.
* Within selected positions the replacement is sampled from the
  configurable probabilities
  ``(mask_token_probability, random_token_probability, keep_token_probability)``
  which default to ``(0.8, 0.1, 0.1)`` and must sum to one.
* Random replacements are sampled uniformly from the non-special tail
  of the field vocabulary (token IDs ``>= len(SPECIAL_TOKEN_IDS)``). If
  a field vocabulary contains no non-special tokens, the random branch
  deterministically falls back to ``[MASK]``.

The model receives the corrupted inputs and per-field labels containing
the original token IDs at masked positions and ``ignore_index`` everywhere
else. Per-field cross-entropy losses are averaged across fields to form
the masked-field component of the loss.

## How Next-Field Prediction Works

For each configured field, the target at position ``t`` is the original
(unmasked) token ID at position ``t + 1``. The target is
``ignore_index`` for:

* the final real token in the window (no ``t + 1`` exists in the window);
* every padding position;
* every position whose ``t + 1`` neighbour is padding.

Targets are computed from the original inputs before masking is applied,
so the look-ahead objective remains consistent even when the input fed
to the encoder is corrupted. The objective is still self-supervised
because the labels come from the same token sequence, not from any
supervised label engine.

## Contrastive Objective: Deferred

The Phase 13 specification permits an optional contrastive objective.
This phase defers it to keep the implementation small, deterministic and
well tested. The config field ``enable_contrastive_loss`` exists for
forward compatibility but raises ``NotImplementedError`` when set to
``True``. The reasons:

* contrastive InfoNCE requires careful handling of batch sizes below
  two and of view construction, both of which add surface area without
  expanding what we can claim to have validated in Phase 13;
* the masked-field and next-field objectives already exercise every
  encoder code path; a contrastive head can be added in a later phase
  without changing the existing API.

## Handling ``[MASK]``, ``[PAD]`` and ``attention_mask``

* ``[MASK]`` reuses the reserved special-token ID ``4`` from the
  Phase 11 vocabulary; the encoder's field embeddings already accept
  it. Corrupted positions whose replacement is ``[MASK]`` are written
  with this token ID before the batch is fed to the encoder.
* ``[PAD]`` (ID ``0``) is never overwritten and never selected for
  masking. The masking helper raises a clear error if a padded position
  contains a non-``[PAD]`` token, surfacing collation bugs early.
* ``attention_mask`` is preserved unchanged through masking. The
  encoder converts the Phase-11 convention (``True = real``) to
  PyTorch's ``src_key_padding_mask`` convention (``True = padding``)
  internally, as in Phase 12.

Padding positions are excluded from every loss because their targets
are set to ``ignore_index`` and ``torch.nn.functional.cross_entropy``
drops those positions. This ensures padding never contributes gradient
to the model and never inflates the reported loss values.

## Determinism

Masking is deterministic under a fixed seed. The
:class:`SSLTokenSequenceDataset` derives a per-window
``torch.Generator`` seeded from ``base_seed + window_index``, so the
masking is stable regardless of dataloader ordering, batch size or
multi-process workers. The smoke runner reuses the existing
:func:`set_torch_deterministic` helper before model creation and
training so model initialisation, dropout and optimiser steps are also
deterministic on a fixed seed.

## Leakage Controls Between Train, Validation and Test

The SSL objectives never look at any market label, return or future
price. The next-field target is a one-step look-ahead within the same
input window, derived from the already-present token IDs, so it cannot
leak across windows. The token vocabulary fitted in Phase 11 is built
from the training split only when split indices are provided; the
Phase 13 model and masking respect that vocabulary unchanged. Mask
selection probabilities and replacement probabilities are part of the
config and never depend on validation or test data.

## Why Smoke Losses Are Not Market Evidence

The CLI runner ``run-ssl-smoke`` operates on a tiny synthetic event-log
fixture, runs a single epoch with a deliberately small transformer
backbone and emits a payload tagged with an explicit warning. Loss
values from this run only verify that:

1. the SSL wrapper builds correctly with the Phase-11 vocabulary;
2. masked-field and next-field targets are accepted by the loss
   functions;
3. gradients flow back through the wrapped encoder.

They do not measure any market quantity. They cannot be used to compare
models, to support trading decisions, or to claim that the architecture
forecasts prices, returns, volatility, spread or fill outcomes. No
alpha, Sharpe, profitability or execution viability is implied.

## Limitations and Next Steps

* The contrastive objective is deferred; the public API leaves room for
  it without changing the existing call shape.
* Mask sampling currently keeps the same per-window seed across epochs;
  curriculum or per-epoch resampling could be added later if the smoke
  runner is grown into a longer pretraining workflow.
* Random-replacement tokens are sampled uniformly from the non-special
  vocabulary; later phases could weight them by empirical frequency.
* The smoke training path uses synthetic fixtures only; no real
  exchange data is consumed and no network calls are made.

## Out of Scope: Multi-Task Fine-Tuning Is Phase 14

Supervised multi-task fine-tuning on real labels (direction, volatility,
spread, fill, adverse selection) is deliberately left for Master Prompt
14. Phase 13 produces a small, well-tested SSL pretraining wrapper that
Phase 14 can call into without further changes to Phase 11 or Phase 12.
