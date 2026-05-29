# Proper-Training Neural Subset Plan

Date: 2026-05-29

## Intended run

- Output directory: `experiments/fi2010_neural_proper_training_subset_v2/`
- Folds: 1, 2, 3
- Horizons: 10, 50
- Seeds: 0
- Lookback: 50
- Objectives: supervised, masked_reconstruction, next_field
- SSL pretrain epochs: 5
- Fine-tuning max epochs: 25
- Early-stopping patience: 5
- Early-stopping metric: validation macro-F1
- Model selection: validation only
- Test evaluation: once, after best validation checkpoint restoration
- Device: CPU
- Batch size: 1024

## Evidence classification

This run should be classified as `partial_real`.

It is the fallback credible slice rather than the primary complete-real target.
The primary target is folds 1-5, horizons 10 and 50, seed 0, lookback 50, all
three objectives, max_epochs 25, patience 5. This machine is CPU-only with about
16 GB RAM, and the prepared fold CSVs grow from 127 MB to 353 MB. Running folds
1-3 gives a materially stronger longer-training modelling subset than the old
fold-1/horizon-10/lookback-10/max-2 code-path run while avoiding the largest
folds under the current compute limit.

## Credibility rationale

The planned slice contains 18 matched runs: 3 folds x 2 horizons x 1 seed x 3
objectives. Supervised, masked-reconstruction SSL and next-field SSL share the
same preprocessing, architecture, seed, lookback and validation-only selection
within each fold/horizon cell. The run saves per-run curves, predictions, best
epochs, validation metrics, config snapshots and SHA256 manifests.

If the run cannot complete, the artefacts and report must state the exact
completed scope and remain `partial_real`.

## Actual completed scope

The folds 1-3 fallback was attempted, but CPU runtime made the full fallback
slice too expensive in this pass. The completed v2 scope is:

- Folds: 1
- Horizons: 10, 50
- Seeds: 0
- Lookback: 50
- Objectives: supervised, masked_reconstruction, next_field
- SSL pretrain epochs: 5
- Fine-tuning max epochs: 25
- Early-stopping patience: 5
- Completed runs: 6
- Failed runs: 0
- Missing matched pairs: 0
- Evidence classification: `partial_real`

This remains materially stronger than the previous fold-1/horizon-10/lookback-10
max-2-epoch code-path slice, but it is not complete-real modelling evidence.
