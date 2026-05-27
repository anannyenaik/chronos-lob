# FI-2010 Neural Benchmarks

This page documents the supervised neural benchmark execution layer for
prepared FI-2010 folds. It is an operational runbook, not a final report.

## Purpose

`run-fi2010-neural-benchmark` executes the supervised neural plan from
`configs/experiments/fi2010_neural_serious.yaml` on selected folds, seeds,
models and lookbacks. It writes lightweight aggregate artefacts and per-run
metadata.

## Prerequisite

Prepare the multi-fold CSVs first:

```bash
python -m chronoslob.cli prepare-fi2010-multifold \
  --config configs/experiments/fi2010_multifold.yaml \
  --extracted-root data/raw/fi2010/extracted/BenchmarkDatasets \
  --processed-root data/processed/fi2010 \
  --out runs/fi2010_multifold_prepare \
  --folds all
```

The processed root must contain files such as `fold1_combined.csv`. Raw and
processed FI-2010 files remain local and ignored by git.

## Smoke Versus Benchmark

A smoke run uses a narrow subset and short training, usually one fold, one seed
and `--max-epochs 1`. It checks the execution path and artefact contract.

A benchmark run uses the configured grid of all folds, seeds, models and
lookbacks with the configured epoch budget. The full grid is guarded by
`--allow-full-benchmark` so it is not launched by accident.

## Supported Models

- `deeplob_style`
- `matrix_transformer`

No SSL results are reported yet.

## Split Handling

The runner reads the prepared fold CSV `split` column. Rows marked `train` form
the official train partition; validation is carved from the tail of those rows.
Rows marked `test` are used only for final metrics. Feature standardisation and
label encoding are fit on train rows only.

## Seeds, Folds And Lookbacks

The run plan is:

```text
folds x seeds x models x lookbacks
```

CLI options can select subsets with comma-separated values. Fold IDs may be
written as `fold_1` or `1`.

## Early Stopping

The config uses validation macro F1 by default. Training metadata records the
validation metric name and value, best epoch, whether early stopping occurred,
device, parameter count and elapsed training seconds.

## Artefacts

Top-level artefacts:

```text
summary.json
run_plan.csv
results_by_fold_seed.csv
results_summary.csv
training_summary.csv
model_capacity_summary.csv
model_failures.json
```

Per-run lightweight artefacts are written under `runs/`, including
`result.json` and `model_card.md`. Full predictions and checkpoints are not
written by default.

## Smoke Command

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

## Full Benchmark Example

Example only:

```bash
python -m chronoslob.cli run-fi2010-neural-benchmark \
  --config configs/experiments/fi2010_neural_serious.yaml \
  --processed-root data/processed/fi2010 \
  --out experiments/fi2010_multifold_neural_full \
  --folds all \
  --models all \
  --seeds all \
  --lookbacks all \
  --max-epochs 75 \
  --allow-full-benchmark \
  --overwrite
```

## Reduced-Scope Multi-Fold Run

The full configured grid (90 planned runs at up to 75 epochs each) is not
practical on the available CPU-only hardware, so the committed evidence under
`experiments/fi2010_multifold_neural/` is a reduced-scope run across all five
official NoAuction ZScore folds with a single seed (`0`), single lookback
(`20`) and `max_epochs=25` for each model. All ten planned runs completed
without failures; full predictions and checkpoints are not written.

```bash
python -m chronoslob.cli run-fi2010-neural-benchmark \
  --config configs/experiments/fi2010_neural_serious.yaml \
  --processed-root data/processed/fi2010 \
  --out experiments/fi2010_multifold_neural \
  --folds all \
  --models deeplob_style,matrix_transformer \
  --seeds 0 \
  --lookbacks 20 \
  --max-epochs 25 \
  --overwrite
```

This is a reduced-scope local CPU run, not the full configured grid. Cross-seed
and multi-lookback variance is not reported in this evidence. No neural
superiority claim is made on this single-seed evidence.
