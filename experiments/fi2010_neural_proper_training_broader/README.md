# FI-2010 Broader Proper-Training Neural Benchmark

This directory contains the storage-light retained evidence from the completed
broader proper-training neural benchmark. It is reported separately from the
one-epoch matched full grid.

The one-epoch full grid remains the matched comparison and infrastructure evidence.
This subset is used to assess whether the neural models remain credible under a more realistic training budget with early stopping and validation-only model selection.

## Training Protocol

- max epochs: 25
- early stopping metric: validation_macro_f1 (validation only)
- early stopping patience: 5
- best validation checkpoint is restored before the single official test evaluation
- no model is ever selected on test metrics
- SSL pretraining epochs (SSL objectives only): 10
- SSL pretraining consumes official training rows only

## Scope

- folds: 1, 2, 3, 4, 5
- horizons: 10, 50
- seeds: 0, 1, 2
- lookbacks: 20, 50, 100
- models: matrix_transformer, deeplob_style
- objectives: supervised
- smoke test: no
- evidence level: complete_real
- scope label: broader_proper_training_complete
- completed cells: 180 of 180
- failed cells: 0

## Result Summary

- Across 90 runs per model, the matrix transformer has the stronger overall
  mean macro-F1 (`0.6013` versus `0.5133`) and MCC (`0.4294` versus `0.3356`).
- The matrix transformer also has lower mean Brier score (`0.4318`) and ECE
  (`0.0617`), but its results are substantially more variable.
- Lookback 100 is weak for the matrix transformer, especially at horizon 10.
  DeepLOB-style results are steadier but lower overall.
- Confidence filtering raises retained-sample macro-F1 and MCC while reducing
  active fraction. Coverage falls sharply in some transformer lookback-100
  cells, so active fraction must be reported with filtered metrics.
- The existing classical result is a horizon-10, five-fold, single-seed
  directional reference rather than a matched prediction-level comparison.
  Neural results are mixed by lookback; no broad neural superiority is claimed.

## Hamilton Provenance

The broader proper-training neural benchmark was executed as Slurm jobs on
Durham University Hamilton/NCC HPC. Retained summaries and claim assessments
are committed; large checkpoints, raw predictions and cluster logs are
excluded.

This retained benchmark is post-`v0.2.0` evidence on `main`. `v0.2.0` remains
the published release and does not include it.

See `hamilton_compute_provenance.json` and
`proper_neural_claim_assessment.json`.

## Outputs

- per-run predictions: `runs/**/predictions.csv`
- per-run training curves: `runs/**/curves.csv`, `runs/**/curves.json`
- per-run metrics, best epoch and SHA256 hashes: `runs/**/metrics.json`, `runs/**/sha256_manifest.json`
- root config snapshot and SHA256 inventory: `config_snapshot.json`, `sha256_manifest.json`
- completed-run table: `results_summary.csv`
- training-curve summary (best epoch, early stopping): `training_curves_summary.csv`
- grouped aggregate table: `aggregate_summary.csv`, `aggregate_summary.json`
- matched supervised-vs-SSL deltas: `ssl_comparison.csv`
- failed or reused-existing runs: `failures.csv`
- storage-light grouped and confidence-filtered analysis:
  `per_run_summary.csv`, `fold_summary.csv`, `seed_summary.csv`,
  `lookback_summary.csv`, `model_summary.csv`, `horizon_summary.csv`,
  `confidence_filtered_summary.csv`, `confidence_filtered_aggregate.csv`

## Limitations

- Smoke-test artefacts are code-path checks only and are not empirical evidence.
- A documented partial scope is classified `partial_real`; the report and evidence pack say exactly what was run.
- SSL pretraining is compared under matched conditions; the artefacts do not support a broad SSL improvement claim unless the matched deltas show it.
- The runner reports predictive, calibration and selective-prediction coverage
  diagnostics only; it makes no live-execution or model-ranking claim beyond
  the exact stored scope.
