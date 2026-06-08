# FI-2010 Neural Benchmark Protocol

This protocol defines the supervised neural benchmark infrastructure for
FI-2010. It covers planning, supervised execution and retained metadata,
including the completed broader proper-training benchmark.

## Why This Exists

Earlier neural commands used tiny fixtures, short epochs and smoke-oriented
settings to validate code paths. Those checks were useful for integration, but
they were not enough for benchmark-quality neural evidence.

Benchmark-quality neural runs require:

- official split handling, with test rows held out end-to-end;
- validation carved only from official train rows;
- multiple folds and multiple seeds;
- configurable lookback windows;
- early stopping on validation macro F1 or validation loss;
- recorded parameter count, device, best epoch and training duration;
- lightweight aggregate artefacts rather than full prediction files by default.

## Planned Models

The serious config covers two supervised neural baselines:

- `deeplob_style`
- `matrix_transformer`

This supervised protocol does not itself run SSL. Retained matched SSL evidence
is reported separately in the one-epoch neural full grid, the earlier partial
proper-training SSL subset and the scoped SSL-v2 benchmark.

## Config

Use:

```text
configs/experiments/fi2010_neural_serious.yaml
```

The config declares five official folds, seeds `0,1,2`, horizon `10`,
lookbacks `20,50,100`, benchmark-mode optimisation settings and output roots
under ignored local directories.

## Split Handling

Prepared fold CSVs must preserve the configured `split` column. Rows marked
`test` are used only for final evaluation. Validation rows come from the tail of
official train rows. Feature scaling and label encoding must be fit on train
rows only.

## Seeds And Folds

The run grid is deterministic:

```text
folds x seeds x models x lookbacks
```

The inspection command expands that grid without reading FI-2010 data and
without training.

## Early Stopping

The benchmark config uses validation macro F1 for early stopping. The training
metadata records the configured metric, patience, best epoch, whether stopping
occurred before `max_epochs`, and elapsed training seconds.

## Device And Checkpoints

Device policy is `auto`, `cuda` or `cpu`. `auto` resolves to CUDA when
available and CPU otherwise. Checkpoint writing is disabled by default; when
enabled in a future run, checkpoint files belong under ignored local paths.

## Artefacts

Expected lightweight top-level artefacts are:

```text
summary.json
run_plan.csv
results_by_fold_seed.csv
results_summary.csv
training_summary.csv
model_capacity_summary.csv
model_failures.json
```

Per-run metadata includes `fold_id`, `seed`, `model_name`, `lookback`, `device`,
`parameter_count`, `max_epochs`, `best_epoch`, `early_stopped`,
`training_seconds`, validation metric, test metrics and `status`.

Full prediction files are not written by default.

## Inspect Command

```bash
python -m chronoslob.cli inspect-fi2010-neural-plan \
  --config configs/experiments/fi2010_neural_serious.yaml \
  --folds all \
  --models deeplob_style,matrix_transformer
```

The command prints the run count, folds, seeds, models, lookbacks, device policy,
output roots and whether the config is smoke or benchmark mode. It does not
train models and writes no outputs.

## Smoke Execution Command

```bash
python -m chronoslob.cli run-fi2010-neural-benchmark \
  --config configs/experiments/fi2010_neural_serious.yaml \
  --processed-root data/processed/fi2010 \
  --out experiments/fi2010_multifold_neural \
  --folds fold_1 \
  --models deeplob_style,matrix_transformer \
  --seeds 11 \
  --lookbacks 20 \
  --max-epochs 1 \
  --overwrite
```

This is smoke-level because it selects one fold, one seed, one lookback and a
single epoch. The full configured grid requires an explicit
`--allow-full-benchmark` flag. The operational runbook is
[FI2010_NEURAL_BENCHMARKS.md](FI2010_NEURAL_BENCHMARKS.md).

## Claim Boundary

No neural superiority claim is made. This protocol does not establish
deployment or live-market execution quality. SSL claims must be taken only from
the separately retained matched artefacts and their exact recorded scopes.

## Completed Proper-Training Scope

The broader supervised proper-training benchmark completed all 180 target
cells: folds 1-5, horizons 10/50, seeds 0-2, lookbacks 20/50/100 and the
DeepLOB-style and matrix-transformer model families. It uses validation-only
early stopping and best-checkpoint restore.

The matrix transformer has stronger overall mean predictive and calibration
metrics in the exact retained scope, but substantially higher variability and
weak lookback-100 rows. Results are mixed by model, lookback and horizon, so no
broad neural superiority is claimed. Active fraction is retained only as a
selective-prediction coverage proxy alongside confidence-filtered metrics.

Execution used staged Hamilton/NCC Slurm arrays after a representative timing
gate. Large checkpoints, raw predictions and cluster logs are excluded from the
retained repository evidence.
