# Reproducibility

ChronosLOB is designed as a local, reproducible research artefact. This page
records the canonical validation and FI-2010 reproduction flow.

The current public result path is the retained-artefact flow: leakage-safe
FI-2010 benchmark evidence, supervised and SSL comparisons, calibration and
confidence filtering, feature-stability analysis, execution-aware proxy
diagnostics and `build-final-empirical-report`. Synthetic event-level replay and
Binance Spot aggregated L2 replay are supporting engineering evidence.

The empirical study contract for FI-2010 is defined in
[RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md). The multi-fold study config
that the protocol commits to is at
[`configs/experiments/fi2010_multifold.yaml`](../configs/experiments/fi2010_multifold.yaml)
and now includes the executable classical multi-fold layer.

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

For multi-fold preparation, use `prepare-fi2010-multifold` to convert all
configured folds in one pass. See
[FI2010_MULTIFOLD_PROTOCOL.md](FI2010_MULTIFOLD_PROTOCOL.md) for the full
runbook. Example:

```bash
python -m chronoslob.cli prepare-fi2010-multifold \
  --config configs/experiments/fi2010_multifold.yaml \
  --extracted-root data/raw/fi2010/extracted/BenchmarkDatasets \
  --processed-root data/processed/fi2010 \
  --out runs/fi2010_multifold_prepare \
  --folds all
```

Multi-fold preparation produces one combined CSV per fold under the
processed root, plus per-fold manifests and a `summary.json` under `--out`.
It does not train models.

Run the classical multi-fold layer after preparation:

```bash
python -m chronoslob.cli run-fi2010-multifold-classical \
  --config configs/experiments/fi2010_multifold.yaml \
  --processed-root data/processed/fi2010 \
  --out experiments/fi2010_multifold_classical \
  --models majority,logistic,ridge,elastic_net,random_forest,gradient_boosting \
  --folds all \
  --overwrite
```

This writes aggregate classical metrics and proxy diagnostics without writing
full prediction rows by default.

Inspect the supervised neural benchmark plan before launching any long neural
run:

```bash
python -m chronoslob.cli inspect-fi2010-neural-plan \
  --config configs/experiments/fi2010_neural_serious.yaml \
  --folds all \
  --models deeplob_style,matrix_transformer
```

This command expands the planned folds, seeds, models and lookbacks. It does
not train models and writes no outputs. See
[NEURAL_BENCHMARK_PROTOCOL.md](NEURAL_BENCHMARK_PROTOCOL.md).

Run a supervised neural smoke subset only after the prepared fold CSVs exist:

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

This writes lightweight neural artefacts only. Full predictions and
checkpoints are not written by default. See
[FI2010_NEURAL_BENCHMARKS.md](FI2010_NEURAL_BENCHMARKS.md) for the guarded
full-grid example.

A reduced-scope multi-fold neural run is reproducible with the same CLI,
swapping `--folds all --seeds 0 --lookbacks 20 --max-epochs 25` for a CPU
budget; aggregate artefacts under
`experiments/fi2010_multifold_neural/` were produced this way and cover
all five folds for both supervised neural models at a single seed and
lookback. The full configured grid still requires the
`--allow-full-benchmark` flag and is not produced locally.

Generate the statistical uncertainty layer after the multi-fold tables exist:

```bash
python -m chronoslob.cli analyse-fi2010-uncertainty \
  --classical experiments/fi2010_multifold_classical \
  --neural experiments/fi2010_multifold_neural \
  --out experiments/fi2010_uncertainty \
  --baseline gradient_boosting \
  --overwrite
```

This is a diagnostic layer over stored tables; it does not retrain any
model. See [STATISTICAL_UNCERTAINTY.md](STATISTICAL_UNCERTAINTY.md) for
the confidence-interval method, the paired-comparison method and the
limitations of the current evidence.

Run the brutal ablation layer once the prepared folds and the stored
classical and neural tables exist:

```bash
python -m chronoslob.cli run-fi2010-brutal-ablations \
  --config configs/experiments/fi2010_multifold.yaml \
  --neural-config configs/experiments/fi2010_neural_serious.yaml \
  --processed-root data/processed/fi2010 \
  --classical experiments/fi2010_multifold_classical \
  --neural experiments/fi2010_multifold_neural \
  --out experiments/fi2010_brutal_ablations \
  --overwrite
```

The feature-group and horizon families refit a fast linear baseline on
the real folds; the model-class, calibration and execution families
reuse the stored tables; the CPU-expensive neural lookback sweep is
skipped by default and recorded with a reason unless
`--neural-lookbacks` is supplied. Execution numbers are proxy
diagnostics only. See [FI2010_BRUTAL_ABLATIONS.md](FI2010_BRUTAL_ABLATIONS.md).

Build the execution-aware v2 proxy diagnostics over the stored multi-fold
and ablation artefacts:

```bash
python -m chronoslob.cli run-fi2010-execution-v2 \
  --classical experiments/fi2010_multifold_classical \
  --neural experiments/fi2010_multifold_neural \
  --ablations experiments/fi2010_brutal_ablations \
  --out experiments/fi2010_execution_v2 \
  --overwrite
```

This layer retrains nothing and reads no full predictions or checkpoints.
It re-frames the stored execution-sensitivity rows as cost, latency,
confidence, turnover, adverse-selection, fill and statistical-to-execution
degradation proxies. Neural runs ship no stored execution proxy rows, so
their execution-aware diagnostics are recorded as explicit skips. Every
metric is a proxy diagnostic; no profitability or live tradability claim is
made. See [FI2010_EXECUTION_V2.md](FI2010_EXECUTION_V2.md).

Review the external benchmark context layer:

- Public context:
  [FI2010_EXTERNAL_BENCHMARKS.md](FI2010_EXTERNAL_BENCHMARKS.md)
- Structured context:
  [`experiments/fi2010_external_context/`](../experiments/fi2010_external_context/)
- Maintainer note:
  [`reports/external_benchmark_context.md`](../reports/external_benchmark_context.md)

This layer runs no models, writes no predictions or checkpoints and records no
external numeric paper metrics. It explains when qualitative protocol
comparison is meaningful and when direct metric comparison is not.

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

To build the final traceable FI-2010 empirical report from the committed
multi-fold artefacts:

```bash
python -m chronoslob.cli build-final-empirical-report \
  --classical experiments/fi2010_multifold_classical \
  --neural experiments/fi2010_multifold_neural \
  --uncertainty experiments/fi2010_uncertainty \
  --ablations experiments/fi2010_brutal_ablations \
  --execution experiments/fi2010_execution_v2 \
  --execution-v3 experiments/fi2010_execution_v3 \
  --execution-centrepiece reports/execution_centrepiece \
  --external experiments/fi2010_external_context \
  --neural-full-grid experiments/fi2010_neural_full_grid \
  --proper-training experiments/fi2010_neural_proper_training_broader \
  --ssl-v2-analysis reports/ssl_v2_analysis \
  --feature-ablations experiments/fi2010_feature_ablations \
  --feature-ablation-analysis reports/feature_ablation_analysis \
  --evidence-pack reports/evidence_pack \
  --synthetic-lob reports/synthetic_lob_extension \
  --binance-l2 reports/binance_l2_extension \
  --out reports/chronoslob_final_empirical_report.md \
  --overwrite
```

## Hamilton SSL-v2 Refresh

The seed-1 and seed-2 SSL-v2 refresh was run with Slurm on Durham University
Hamilton/NCC HPC. Independent array tasks covered one fold, horizon and seed pair
each, with both supervised and market-state multitask objectives in each task.
The array used a maximum concurrency of four.

The retained aggregate combines the pre-existing seed-0 runs with the Hamilton
seed-1 and seed-2 runs. Reports, summaries and compact run metadata are retained;
large checkpoints, raw predictions and cluster logs are intentionally excluded.
See
[`reports/ssl_v2_analysis/hamilton_compute_provenance.json`](../reports/ssl_v2_analysis/hamilton_compute_provenance.json)
for the recorded environment, Slurm jobs and GPU determinism caveat.

## Full Local Validation

```powershell
python -m pytest -q
python -m ruff check .
python -m mypy chronoslob
python -m chronoslob.cli doctor
python -m chronoslob.cli inspect-release-readiness
python -m chronoslob.cli run-project-audit --strict
git diff --check
```

## Reporting Rule

Any reported metric must trace to a versioned config, data source, seed, split
definition, code commit where available and stored output artefacts. Predictive
metrics, calibration metrics and execution-aware sensitivity are reported as
separate evidence streams.
