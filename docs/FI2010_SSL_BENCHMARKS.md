# FI-2010 Self-Supervised Pretraining And Fine-Tuning

This document describes the `ssl_transformer` path added by
`run-fi2010-ssl-neural-benchmark`. It is research software and a leakage-safe
evaluation scaffold. It is not financial advice, not live trading
infrastructure and makes no self-supervised effectiveness claim.

## What The Path Does

For each selected fold, seed and lookback the runner:

1. Builds the official, leakage-safe train/validation/test split and a
   train-only standardised feature matrix.
2. Pretrains a transformer encoder on the official training rows only, using a
   self-supervised objective.
3. Saves the pretrained encoder checkpoint, a config snapshot, a metrics JSON,
   the git commit hash and SHA256 hashes for the key artefacts.
4. Fine-tunes a supervised matrix transformer initialised from the pretrained
   encoder on mid-price direction.
5. Trains a supervised baseline of identical architecture (random
   initialisation) on the same fold, horizon, seed and preprocessing, and
   records both results plus their comparison.

The fine-tuned `ssl_transformer` and the `supervised_transformer` baseline
share one architecture, set of folds, horizons, seeds and preprocessing, so the
comparison is like-for-like.

## Self-Supervised Objectives

- `masked_field`: randomly mask selected feature-channel entries of the input
  window and reconstruct the original standardised values via mean squared
  error over masked entries only. The mask probability is configurable.
- `next_field`: predict the train-only quantile bucket of every feature at the
  next window position from the hidden state at the current position, using
  cross-entropy over the discretised buckets. The final window position has no
  successor and is ignored.
- `both`: enable the masked-field and next-field objectives together.

The `next_field` objective uses a small, robust default bucket count suitable
for CPU smoke tests and is configurable for full FI-2010 training.

## Leakage Policy

- Pretraining consumes official training rows only.
- The validation pretraining loss is computed on a partition carved out of the
  official training rows; it never uses official test rows.
- Feature standardisation and the next-field quantile bucket edges are fitted
  on the training partition only.
- The official split-aware test evaluation is preserved unchanged for both the
  fine-tuned model and the supervised baseline.

## Artefacts

Per run, under `runs/<fold>_seed_<seed>_lb<lookback>/`:

- `pretrain/pretrained_encoder.pt`: the transferable encoder checkpoint.
- `pretrain/pretrain_config.json`: the architecture, objective and split
  snapshot.
- `pretrain/pretrain_metrics.json`: the pretraining loss history and
  train-carved validation pretraining loss.
- `pretrain/artefact_manifest.json`: SHA256 hashes for the encoder, config and
  metrics files plus the git commit hash.
- `ssl_transformer/predictions.csv` and `supervised_transformer/predictions.csv`:
  per-row `label`, `prediction`, class probabilities and confidence.
- `comparison.json`: the per-run side-by-side result.

Top-level: `summary.json`, `run_plan.csv`, `results_by_fold_seed.csv`,
`results_summary.csv`, `ssl_pretraining_summary.csv`, `comparison_summary.csv`
and `model_failures.json`.

## Reporting Boundary

The final report builder refuses to admit any SSL row unless valid SSL
artefacts exist: it requires the SSL summary, results and comparison tables and
at least one pretrained encoder checkpoint whose recorded SHA256 matches the
checkpoint on disk. When those conditions are not met, the report marks the
self-supervised section skipped and no SSL result is claimed.

## Example

```bash
python -m chronoslob.cli run-fi2010-ssl-neural-benchmark \
  --config configs/experiments/fi2010_ssl_smoke.yaml \
  --processed-root data/processed/fi2010_multifold \
  --out experiments/fi2010_ssl \
  --folds fold_1 \
  --seeds 0 \
  --lookbacks 10 \
  --objective both \
  --pretrain-epochs 5 \
  --max-epochs 10 \
  --device cpu
```

Fixture-scale outputs validate the code path only. They are not FI-2010
benchmark evidence and do not demonstrate self-supervised effectiveness.
