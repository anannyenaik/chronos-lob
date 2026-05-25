# FI-2010 Data Acquisition Runbook

ChronosLOB does not ship FI-2010 data and does not download it
automatically. Real benchmark evidence requires the user to obtain the
official FI-2010 archive from Fairdata/Etsin, place it under an ignored
local directory, verify the file and then run the existing benchmark
preparation commands. This page describes that path step by step.

This document is a runbook. It does not contain benchmark numbers,
plots or statistical claims about FI-2010.

## Official Source

- Dataset title: *Benchmark Dataset for Mid-Price Forecasting of Limit
  Order Book Data with Machine Learning Methods*.
- Authors: Ntakaris, Magris, Kanniainen, Gabbouj, Iosifidis (Tampere
  University of Technology, 2017).
- Fairdata/Etsin landing page:
  <https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649>.
- Associated paper: <https://arxiv.org/abs/1705.03233>.
- Licence at time of writing: Creative Commons Attribution 4.0
  International (CC BY 4.0). Verify the licence on the landing page
  before reuse.

The published archive on the landing page is `BenchmarkDatasets.zip`
(approximately 1.86 GB). It expands into nine cross-validation folds of
training and testing files across three normalisation set-ups
(z-score, min-max and decimal-precision) and an auction/no-auction
split.

## Ignored Local Layout

Use the ignored local layout below. `data/raw/` and `data/processed/`
are listed in [.gitignore](../.gitignore), so any FI-2010 file placed
under them will not be tracked by git.

```text
data/
  raw/
    fi2010/
      BenchmarkDatasets.zip            # original archive (kept locally only)
      <one or more .txt files>         # extracted matrices
  processed/
    fi2010/
      <one or more .csv files>         # produced by convert-fi2010-official
```

If the layout already exists, reuse it. If not, create it before
downloading:

```bash
mkdir -p data/raw/fi2010
mkdir -p data/processed/fi2010
```

```powershell
New-Item -ItemType Directory -Force -Path data/raw/fi2010 | Out-Null
New-Item -ItemType Directory -Force -Path data/processed/fi2010 | Out-Null
```

## Manual Download

The Etsin landing page issues a per-session download token through its
web UI. Programmatic downloads from outside the UI are rejected. The
download must therefore be performed manually:

1. Open the landing page
   <https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649>
   in a browser.
2. Open the *Data* tab. Locate `BenchmarkDatasets.zip` in the file tree.
3. Use the *Download* control next to the file. The page issues a
   short-lived download URL through its own UI.
4. Save the archive to `data/raw/fi2010/BenchmarkDatasets.zip`.
5. Record the source URL and the local download timestamp in a local
   note (for example `data/raw/fi2010/PROVENANCE.txt`). The repository
   does not ship a template provenance file because raw data is not
   committed.

If the official source is unavailable, do not silently fall back to a
third-party mirror. Document the reason, identify the chosen mirror
and treat any results as provisional until the official archive can be
verified.

## Checksum and Size Verification

`BenchmarkDatasets.zip` published at the time of writing had:

- byte size: `1864361899`
- SHA-256: `cea93692a270724fa91e8f124da641db727d757e5e0f0bb85067709e9932f664`

Re-derive these values for any local copy. They are written by the
publisher and may be updated; treat the canonical pair as whichever
pair Fairdata currently advertises for the file.

```bash
stat -c %s data/raw/fi2010/BenchmarkDatasets.zip
sha256sum data/raw/fi2010/BenchmarkDatasets.zip
```

```powershell
(Get-Item data/raw/fi2010/BenchmarkDatasets.zip).Length
Get-FileHash data/raw/fi2010/BenchmarkDatasets.zip -Algorithm SHA256
```

If the byte size or checksum disagrees with the publisher's record,
discard the download and retry from the official page before
continuing.

## Extracting And Selecting A File

The archive expands into several directories. Each individual training
or testing file is a whitespace-separated text matrix with 149 rows
and one column per snapshot. The row layout follows the convention
described on the landing page:

- rows 1-40: 10 order-book levels in the form `P^a V^a P^b V^b` per
  level (ask price, ask volume, bid price, bid volume),
- rows 41-144: 104 hand-crafted features,
- rows 145-149: categorical labels for prediction horizons
  `k = 10, 20, 30, 50, 100` with values `{1, 2, 3}` (up, stationary,
  down).

Pick one normalisation set-up and one fold to start with. The
ChronosLOB benchmark configuration at
[`configs/experiments/fi2010_midprice_h10.yaml`](../configs/experiments/fi2010_midprice_h10.yaml)
targets the standard horizon-10 label (`label_10`). A typical first
selection is the z-score normalisation, no-auction split, fold 1
training and testing files.

```bash
unzip data/raw/fi2010/BenchmarkDatasets.zip -d data/raw/fi2010/
```

```powershell
Expand-Archive -Path data/raw/fi2010/BenchmarkDatasets.zip -DestinationPath data/raw/fi2010/
```

Treat any extracted directory under `data/raw/fi2010/` as raw data. Do
not commit it.

## Local Verification

`verify-fi2010-local` performs a streaming inspection: it does not load
the full matrix into memory, reports the byte size, SHA-256, row and
column counts, label class distribution and any layout issues.

```bash
python -m chronoslob.cli verify-fi2010-local \
  --data-path data/raw/fi2010/<extracted file>.txt
```

A file matching the official layout reports `official layout: True`
with no issues. A different row count or ragged columns is reported as
an issue and the command exits non-zero.

## Conversion To The Loader's Convention

The existing FI-2010 loader at
[`chronoslob/data/fi2010.py`](../chronoslob/data/fi2010.py) consumes a
header-bearing CSV using `bid_price_<level>`, `bid_quantity_<level>`,
`ask_price_<level>`, `ask_quantity_<level>`, `f_001 ... f_104` and
`label_<horizon>` columns. The official `.txt` matrix is converted to
this convention with `convert-fi2010-official`. The converter never
modifies the source file and writes the output under the caller-chosen
path.

Convert a single training file with a `train` split label:

```bash
python -m chronoslob.cli convert-fi2010-official \
  --input data/raw/fi2010/<train file>.txt \
  --output data/processed/fi2010/<train file>.csv \
  --split train
```

Convert the matching testing file with a `test` split label:

```bash
python -m chronoslob.cli convert-fi2010-official \
  --input data/raw/fi2010/<test file>.txt \
  --output data/processed/fi2010/<test file>.csv \
  --split test
```

Concatenate the two CSV files for a single fold to produce a combined
file with a populated `split` column. On POSIX shells:

```bash
head -1 data/processed/fi2010/<train file>.csv \
  > data/processed/fi2010/fold1_combined.csv
tail -n +2 data/processed/fi2010/<train file>.csv \
  >> data/processed/fi2010/fold1_combined.csv
tail -n +2 data/processed/fi2010/<test file>.csv \
  >> data/processed/fi2010/fold1_combined.csv
```

On Windows PowerShell:

```powershell
Get-Content data/processed/fi2010/<train file>.csv -TotalCount 1 |
  Out-File -Encoding utf8 data/processed/fi2010/fold1_combined.csv
Get-Content data/processed/fi2010/<train file>.csv |
  Select-Object -Skip 1 |
  Add-Content -Encoding utf8 data/processed/fi2010/fold1_combined.csv
Get-Content data/processed/fi2010/<test file>.csv |
  Select-Object -Skip 1 |
  Add-Content -Encoding utf8 data/processed/fi2010/fold1_combined.csv
```

Re-run `verify-fi2010-local` is not necessary on the converted CSV;
the existing preparation step will load and validate it.

## Preparing The Benchmark Input

Once a converted CSV file exists, run the existing preparation
command. The command does not train a model.

```bash
python -m chronoslob.cli prepare-fi2010-benchmark \
  --config configs/experiments/fi2010_midprice_h10.yaml \
  --data-path data/processed/fi2010/fold1_combined.csv \
  --out runs/fi2010_midprice_h10_prepare_real
```

`runs/` is ignored by git. The preparation step writes
`preparation_summary.json`, `data_manifest.json`, `label_summary.json`,
`split_summary.json`, `validation_summary.json` and a `config.yaml`
snapshot. See [FI2010_BENCHMARK.md](FI2010_BENCHMARK.md) for the full
artefact contract.

## Distinguishing Smoke Runs From Real Benchmark Runs

The fixture under
[`tests/fixtures/fi2010/tiny_fi2010_like.csv`](../tests/fixtures/fi2010/tiny_fi2010_like.csv)
is a four-row synthetic file used to exercise the loader, validator
and runner. Any output under `runs/` that consumed that fixture is a
smoke run only and is not benchmark evidence.

Real benchmark runs:

- read a file under `data/raw/fi2010/` or `data/processed/fi2010/`,
- have a row count that matches the official FI-2010 file selected,
- store a `data_manifest.json` whose SHA-256 matches the verified
  upstream checksum (for the chosen source file).

If `inspect-paper-experiment` or `inspect-experiment-artifacts` reports
a synthetic fixture flag, that experiment directory is not real
benchmark evidence even if it is named after the FI-2010 task.

## What Not To Commit

Do not commit:

- `BenchmarkDatasets.zip` or any extracted `.txt` files under
  `data/raw/fi2010/`,
- converted CSV files under `data/processed/fi2010/`,
- `predictions.csv` or `predictions.parquet` artefacts that exceed
  GitHub's per-file limits,
- model checkpoints (`.pt`, `.pth`, `.ckpt`, `.joblib`),
- environment-specific paths or credentials.

Lightweight evidence artefacts (`results.json`, `data_manifest.json`,
`calibration_bins.csv`, `execution_sensitivity.csv`,
`confusion_matrix.json`, `runner_summary.json`, `model_card.md` and
the empirical report) may be committed when they are produced from a
real benchmark run and pass `inspect-experiment-artifacts`. Predictions
are usually large and should stay ignored or be replaced by a manifest
that references their checksum.

## Troubleshooting

If `verify-fi2010-local` reports a row count other than 149, the file
is not a single official matrix. Common causes are:

- a copy of the README or licence text was selected by mistake,
- a partial download was placed under `data/raw/fi2010/`,
- a transposed copy was already produced by a different tool.

Re-extract the archive, identify the correct file (typically named
`Train_Dst_NoAuction_ZScore_CF_<fold>.txt` for the training fold and
the matching `Test_...` for the testing fold) and re-run verification.

If `convert-fi2010-official` rejects the file with a label-value
error, the file may be a regression or precision-set variant rather
than the classification labels. ChronosLOB only supports the
categorical labels `{1, 2, 3}` documented on the landing page.

## No Automated Download Logic

ChronosLOB does not add network-aware downloader code into the
`chronoslob` package. The acquisition path documented above is a
local-only utility composed of `verify-fi2010-local` and
`convert-fi2010-official`, both of which operate on user-supplied
local files. This separation is deliberate and matches the broader
project policy of never performing hidden network calls.
