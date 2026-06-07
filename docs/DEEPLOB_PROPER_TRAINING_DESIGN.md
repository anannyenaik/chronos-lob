# Design note: DeepLOB-style under the proper-training neural protocol

Status: **proposal, not implemented.** This note scopes the change required to
add a DeepLOB-style architecture to the proper-training neural benchmark so a
two-model (matrix transformer + DeepLOB) grid becomes runnable under one matched
protocol. No experiment code is changed by this note. It exists so the approach
can be approved before claim-bearing code is touched.

## Motivation and current state

The broader proper-training neural benchmark currently runs the **matrix
transformer only** (90 cells). The original target scope also names a
DeepLOB-style model (180 cells). Today that is not runnable under proper
training:

- `run-fi2010-neural-proper-training-subset` hard-requires `matrix_transformer`
  (`chronoslob/experiments/fi2010_neural_proper_training.py:350-354`).
- The only DeepLOB entrypoint is the one-epoch full-grid runner, which clamps
  `max_epochs` to 1 (`chronoslob/experiments/fi2010_neural_grid.py:448`) — so it
  cannot produce a longer-trained DeepLOB run.

Crucially, the **protocol already has parity**: `_run_deeplob_style`
(`chronoslob/experiments/neural_adapters.py:580`) and
`run_matrix_transformer_finetune` both call `fit_torch_classifier`
(`chronoslob/training/torch_training.py:451`), which performs validation-only
early stopping and true best-checkpoint restore
(`:561-562`, `model.load_state_dict(best_state)`), with train-only
standardisation and split-confined windows. So this is plumbing, not a new
training regime.

## What must change

The proper-training module treats the experiment axis as `objective`
(supervised / masked_reconstruction / next_field) over a single fixed
architecture. Adding DeepLOB introduces a genuine **model** axis. The change
touches four areas.

### 1. Model selection guard and a model axis
- `run_fi2010_neural_proper_training_subset(...)`: relax the
  `matrix_transformer`-only guard (`:350-354`) to accept a `models` selection of
  `{matrix_transformer, deeplob}`, requiring at least one enabled and matching
  the config's `neural_models`.
- `expand_proper_training_specs(...)`: add a `models` dimension to the spec
  product so each cell carries its architecture. The run-id and `run_dir`
  (`:255`, `:267-274`) must include the model so the matrix-transformer and
  DeepLOB runs never collide on disk. Proposed leaf:
  `fold_{F}/horizon_{H}/seed_{S}/lookback_{L}/{model}/{objective}/`.
  (Note: this changes the run-dir layout. The broader-grid consolidate validator
  and the `.gitignore`/aggregation assume the current
  `.../lookback_{L}/{objective}/` leaf, so they must be updated in lockstep, or
  DeepLOB runs written under a clearly separated subtree.)

### 2. Execution branch
- `_execute_run_spec(...)` currently calls `run_matrix_transformer_finetune`
  unconditionally (`:908`). Branch on the spec's model:
  - `matrix_transformer` -> existing path (unchanged).
  - `deeplob` -> call `_run_deeplob_style(...)` with a `PaperNeuralSettings`
    built from the same config training block (epochs, patience, metric,
    lr, weight decay, gradient clip, dropout, batch size, lookback, device,
    deterministic seed) plus DeepLOB-specific fields
    (`deeplob_conv_channels`, `deeplob_lstm_hidden_size`,
    `deeplob_use_batch_norm`).
  - DeepLOB supports the supervised objective only in this protocol. SSL
    pretraining objectives stay matrix-transformer-only (the SSL encoder is the
    transformer); attempting `deeplob` with an SSL objective must raise a clear
    error rather than silently fall back.

### 3. Reuse signature
- `_expected_reuse_signature(...)` reads `config.neural_models["matrix_transformer"]`
  and emits transformer-only fields (`:797-818`). Make it model-aware: emit the
  DeepLOB hyperparameters when the spec is DeepLOB, and include `model` as a
  signature key so a transformer run is never mistaken for a completed DeepLOB
  run (and vice versa).

### 4. CLI surface
- Add `--models matrix_transformer,deeplob` (default `matrix_transformer` to
  preserve current behaviour) to `run-fi2010-neural-proper-training-subset`,
  parsed like the existing selections, threaded into
  `run_fi2010_neural_proper_training_subset(...)`.
- The DeepLOB config must define a `deeplob` entry under `neural_models` in
  `configs/experiments/fi2010_neural_proper_training.yaml` (conv channels, LSTM
  hidden size, batch-norm flag, dropout).

## Config addition (illustrative)

```yaml
neural_models:
  matrix_transformer:
    enabled: true
    # ... existing ...
  deeplob:
    enabled: true
    architecture: DeepLOB-style conv stack + LSTM head over normalised windows
    conv_channels: [16, 16, 32]
    lstm_hidden_size: 64
    use_batch_norm: true
    dropout: 0.10
```

## Tests and integrity checks (must accompany the change)

- Unit test: a tiny CPU DeepLOB proper-training cell produces `status.txt`,
  `metrics.json`, `predictions.csv`, `curves.csv/json`, `config.json`, with
  `early_stopped`/`best_epoch` populated and best-checkpoint restore exercised.
- Unit test: reuse signature for DeepLOB differs from the transformer signature
  for the same fold/seed/lookback/horizon; re-running a completed DeepLOB cell is
  a skip, and a transformer cell is never matched to a DeepLOB run-dir.
- Unit test: `deeplob` + an SSL objective raises a clear error.
- Leakage check: extend the existing no-lookahead validation to the DeepLOB path
  (the windowing and train-only standardisation are shared, but assert it
  explicitly for DeepLOB rather than relying on the transformer test).
- Run-dir collision test for the new `{model}` leaf.

## Effort and risk

- ~A few hundred lines across `fi2010_neural_proper_training.py` and `cli.py`,
  plus the config entry and the tests above.
- Low algorithmic risk (shared core), but real **claim-integrity** surface: it
  changes the run-dir layout and the experiment harness that backs published
  comparisons. The broader-grid Slurm consolidate validator and `.gitignore`
  must be updated in the same change, or DeepLOB runs isolated to a distinct
  output directory.

## Scope guarantees

- The published `experiments/fi2010_neural_proper_training_subset_v2` tree and
  the SSL-v2 baseline evidence are not touched by this change.
- DeepLOB proper-training would be reported as a distinct result, not merged into
  the matrix-transformer summaries, preserving the existing claim boundaries and
  the one-epoch-vs-proper-training distinction.
- No SSL improvement, profitability, tradability or state-of-the-art claim is
  implied by adding DeepLOB; it broadens the architecture coverage only.
