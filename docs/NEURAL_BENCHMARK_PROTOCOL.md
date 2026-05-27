# FI-2010 Neural Benchmark Protocol

This protocol defines the supervised neural benchmark infrastructure for
FI-2010. It covers planning, selected supervised execution and metadata; no
full multi-fold neural run is reported here.

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

No SSL result is reported yet, and no self-supervised pretraining path is
enabled by this protocol.

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

No neural superiority claim is made. No profitability, deployment, live
tradability, foundation-model or state-of-the-art claim is made. No SSL result
is reported yet.
