# Reproducibility

ChronosLOB is designed as a local, reproducible research artefact. This page
records the canonical validation and FI-2010 reproduction flow.

## 1. Install

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,torch]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,torch]"
```

## 2. Run Smoke Checks

```bash
python -m chronoslob.cli doctor
python -m chronoslob.cli inspect-release-readiness
python -m chronoslob.cli run-project-audit --strict
python -m chronoslob.cli inspect-fi2010 --path tests/fixtures/fi2010/tiny_fi2010_like.csv
python -m chronoslob.cli run-paper-experiment --config configs/experiments/fi2010_midprice_h10.yaml --data-path tests/fixtures/fi2010/tiny_fi2010_like.csv --out runs/paper_experiment_smoke --models majority,logistic --overwrite --build-plots
```

Fixture outputs are code-path checks only. They are not benchmark evidence.

## 3. Acquire FI-2010 Locally

Follow [FI2010_DATA_ACQUISITION.md](FI2010_DATA_ACQUISITION.md) to download the
official FI-2010 archive from Fairdata/Etsin, keep it under
`data/raw/fi2010/`, verify byte size and SHA-256, and extract only into ignored
local directories.

ChronosLOB does not download FI-2010 automatically and does not commit raw or
processed FI-2010 data.

## 4. Convert Official Files

Convert the selected official train and test `.txt` matrices into the loader's
CSV convention:

```bash
python -m chronoslob.cli convert-fi2010-official \
  --input data/raw/fi2010/<train file>.txt \
  --output data/processed/fi2010/<train file>.csv \
  --split train

python -m chronoslob.cli convert-fi2010-official \
  --input data/raw/fi2010/<test file>.txt \
  --output data/processed/fi2010/<test file>.csv \
  --split test
```

Concatenate the matching train and test CSV files into a single
`fold1_combined.csv` with the `split` column preserved, as shown in the data
acquisition runbook.

## 5. Prepare The Benchmark

```bash
python -m chronoslob.cli prepare-fi2010-benchmark \
  --config configs/experiments/fi2010_midprice_h10.yaml \
  --data-path data/processed/fi2010/fold1_combined.csv \
  --out runs/fi2010_midprice_h10_prepare
```

The preparation step writes provenance, label, split and validation artefacts.
It does not train a model.

## 6. Run The Paper Experiment

```bash
python -m chronoslob.cli run-paper-experiment \
  --config configs/experiments/fi2010_midprice_h10.yaml \
  --data-path data/processed/fi2010/fold1_combined.csv \
  --out experiments/fi2010_midprice_h10 \
  --models majority,logistic,random_forest,gradient_boosting,deeplob_style,transformer \
  --overwrite \
  --build-plots
```

The committed FI-2010 evidence uses official split-aware evaluation from the
combined CSV `split` column. Official test rows are not used for preprocessing,
fitting, validation or model-selection decisions.

## 7. Run Ablations

```bash
python -m chronoslob.cli run-paper-ablations \
  --config configs/experiments/fi2010_midprice_h10.yaml \
  --data-path data/processed/fi2010/fold1_combined.csv \
  --out experiments/fi2010_midprice_h10_ablations \
  --models majority,logistic,deeplob_style,transformer \
  --ablation-set standard \
  --overwrite
```

Skipped ablations are recorded explicitly. `ssl_transformer` remains unsupported
until the paper runner includes traceable train-only pretraining and supervised
fine-tuning.

## 8. Run Systems Benchmarks

```bash
python -m chronoslob.cli run-system-benchmarks \
  --config configs/experiments/fi2010_midprice_h10.yaml \
  --data-path data/processed/fi2010/fold1_combined.csv \
  --out experiments/fi2010_midprice_h10_systems \
  --benchmark-set standard \
  --models majority,logistic \
  --overwrite
```

Systems measurements are local to the recorded environment and input file.

## 9. Build And Inspect Report Artefacts

```bash
python -m chronoslob.cli build-paper-report \
  --experiment experiments/fi2010_midprice_h10 \
  --ablations experiments/fi2010_midprice_h10_ablations \
  --systems experiments/fi2010_midprice_h10_systems \
  --out reports/chronoslob_empirical_report.md \
  --overwrite

python -m chronoslob.cli inspect-paper-experiment \
  --experiment experiments/fi2010_midprice_h10

python -m chronoslob.cli inspect-experiment-artifacts \
  --experiment experiments/fi2010_midprice_h10
```

## Full Local Validation

```bash
python -m pytest
python -m compileall -q chronoslob tests
python -m ruff check .
python -m mypy chronoslob
```

## Reporting Rule

Any reported metric must trace to a versioned config, data source, seed, split
definition, code commit where available and stored output artefacts. Predictive
metrics, calibration metrics and execution-aware sensitivity are reported as
separate evidence streams.
