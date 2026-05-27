# FI-2010 Multi-Fold Preparation

This page documents the local-only preparation layer that turns the
official FI-2010 NoAuction ZScore train and test `.txt` matrices into
one combined split-aware CSV per fold, plus a per-fold manifest and a
top-level summary.

The layer is preparation only. It does not download data and does not
train models. The classical benchmark runner that consumes these
prepared folds is documented in
[FI2010_MULTIFOLD_CLASSICAL.md](FI2010_MULTIFOLD_CLASSICAL.md), and the
broader research protocol is documented in
[RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md).

## Acquire FI-2010 Manually

FI-2010 is not shipped with ChronosLOB and is not downloaded by any
command in this repository. Follow the manual acquisition runbook in
[FI2010_DATA_ACQUISITION.md](FI2010_DATA_ACQUISITION.md) to obtain
`BenchmarkDatasets.zip` from the official Fairdata/Etsin landing page,
verify the byte size and SHA-256, and expand the archive locally.

## Expected Local Directory Layout

After extraction the official archive groups all training matrices
under a single `NoAuction_Zscore_Training/` directory and all testing
matrices under `NoAuction_Zscore_Testing/`, with the per-fold suffix
`_CF_{fold}` on the file name. The default templates in
[`configs/experiments/fi2010_multifold.yaml`](../configs/experiments/fi2010_multifold.yaml)
match the published layout:

```text
data/
  raw/
    fi2010/
      BenchmarkDatasets.zip
      extracted/
        BenchmarkDatasets/
          NoAuction/
            1.NoAuction_Zscore/
              NoAuction_Zscore_Training/
                Train_Dst_NoAuction_ZScore_CF_1.txt
                Train_Dst_NoAuction_ZScore_CF_2.txt
                ...
              NoAuction_Zscore_Testing/
                Test_Dst_NoAuction_ZScore_CF_1.txt
                Test_Dst_NoAuction_ZScore_CF_2.txt
                ...
  processed/
    fi2010/
      fold1_combined.csv
      fold2_combined.csv
      ...
```

Both `data/raw/fi2010/` and `data/processed/fi2010/` are listed in
[`.gitignore`](../.gitignore) and are never committed.

If your local copy uses a different layout, set per-fold overrides
under `preparation.fold_overrides` in the YAML config. Each override
takes a `train_path` and `test_path` relative to the extracted dataset
root.

## How Folds Are Discovered

`inspect-fi2010-multifold` walks the configured folds without
converting any data. For each fold it resolves the train and test paths
from `preparation.train_filename_template`,
`preparation.test_filename_template` and any explicit override, and
reports whether each file exists under the supplied `--extracted-root`.

`prepare-fi2010-multifold` runs the same discovery and then converts
each fold's train and test files through the existing official FI-2010
adapter ([`chronoslob/data/fi2010_official.py`](../chronoslob/data/fi2010_official.py)).

## Official Train/Test Split Handling

For each fold the train file is converted with a `train` split label
and the test file with a `test` split label. The two converted CSV
parts are concatenated into one combined CSV with a `split` column
whose values mark each row as either `train` or `test`. Row order
within each partition is preserved as the official file order. The
combined CSV is consumed downstream as the source of truth for
official split semantics, as required by section 5 of the research
protocol.

## Artefacts Generated

Each preparation run writes the following under `--out`:

```text
<out>/
  summary.json                       # top-level summary
  config.yaml                        # snapshot of the executed config
  folds/
    fold_<N>_manifest.json           # per-fold manifest with provenance
```

The combined CSV files are written under `--processed-root` and never
under `--out`, so they are not mixed with the manifests.

The per-fold manifest records:

- the absolute and relative paths of the train and test source files,
- the byte size, SHA-256 and column count of each source file,
- the absolute path, byte size and SHA-256 of the combined CSV,
- the combined row count, column count and per-split row count,
- the configured retained label columns and the target horizon,
- the split column name and the configured train and test values,
- the preparation version and creation timestamp.

The top-level summary records the study name, the config path and
SHA-256, the extracted and processed roots, the requested, prepared
and skipped folds, the per-fold row counts and split counts, the
per-fold source hashes and the git commit at preparation time when one
is available.

## What Stays Gitignored

The following local paths must never be committed:

- `data/raw/fi2010/` (the archive and any extracted `.txt` files),
- `data/processed/fi2010/` (the combined CSV files produced by this
  layer),
- `runs/` (any per-run preparation directories written via `--out`).

The matching gitignore rules already exist in
[`.gitignore`](../.gitignore).

## Exact Commands

Inspect which fold source files are present without converting
anything:

```bash
python -m chronoslob.cli inspect-fi2010-multifold \
  --config configs/experiments/fi2010_multifold.yaml \
  --extracted-root data/raw/fi2010/extracted/BenchmarkDatasets
```

Prepare all configured folds (default `--folds all`):

```bash
python -m chronoslob.cli prepare-fi2010-multifold \
  --config configs/experiments/fi2010_multifold.yaml \
  --extracted-root data/raw/fi2010/extracted/BenchmarkDatasets \
  --processed-root data/processed/fi2010 \
  --out runs/fi2010_multifold_prepare \
  --folds all
```

Prepare a subset of folds:

```bash
python -m chronoslob.cli prepare-fi2010-multifold \
  --config configs/experiments/fi2010_multifold.yaml \
  --extracted-root data/raw/fi2010/extracted/BenchmarkDatasets \
  --processed-root data/processed/fi2010 \
  --out runs/fi2010_multifold_prepare \
  --folds 1,2
```

Pass `--overwrite` to refresh combined CSV files, per-fold manifests
and `summary.json` in place.

## Downstream Classical Runner

After preparation, run `run-fi2010-multifold-classical` to evaluate the
supported classical baselines across prepared fold CSVs and write
aggregate metrics. Neural and self-supervised model execution remains
outside that runner.
