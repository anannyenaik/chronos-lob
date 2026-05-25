# CLI Reference

The CLI is a local research-engineering interface. Commands are
read-only unless their help text explicitly says outputs are written,
and no command performs hidden network calls. Commands that take
fixture paths under `tests/fixtures/` operate on small synthetic files.

Run commands with:

```bash
python -m chronoslob.cli <command> [options]
```

## Diagnostics

| Command   | Description                                       |
| --------- | ------------------------------------------------- |
| `version` | Print the installed package version.              |
| `doctor`  | Print Python, package and key-folder checks.      |

## Data Inspection

| Command                                       | Description                                                                  |
| --------------------------------------------- | ---------------------------------------------------------------------------- |
| `inspect-fi2010 --path PATH`                  | Load a local FI-2010-style file and print a data-quality summary.            |
| `inspect-event-log --path PATH`               | Inspect a canonical local JSONL event log.                                   |
| `inspect-binance-replay --snapshot --updates` | Reconstruct a local Binance-style order book offline from supplied JSON.     |
| `inspect-event-tokens --path PATH`            | Tokenise a canonical event log and print vocabulary and window counts.      |

## Features, Labels and Splits

| Command                                              | Description                                                                  |
| ---------------------------------------------------- | ---------------------------------------------------------------------------- |
| `inspect-features-fi2010 --path PATH`                | Build and validate leakage-safe microstructure features locally.             |
| `event-log-to-features --path PATH`                  | Replay a canonical event log into past-only feature rows.                    |
| `inspect-labels-fi2010 --path PATH`                  | Extract FI-2010 labels or build ChronosLOB labels and print counts.          |
| `inspect-split --rows N`                             | Print default temporal train, validation and test split counts.              |
| `init-run --name NAME --phase PHASE --seed --root`   | Create a metadata-only run directory.                                        |

## Models

| Command                                                                | Description                                              |
| ---------------------------------------------------------------------- | -------------------------------------------------------- |
| `inspect-baselines`                                                    | List supported classical baseline model types.           |
| `run-baseline-smoke --path PATH`                                       | Run a deterministic synthetic-fixture baseline check.    |
| `inspect-torch-dataset --path PATH --lookback N`                       | Build a tiny sequence `DataLoader` and print shapes.     |
| `inspect-deeplob`                                                      | Print DeepLOB-style supervised baseline defaults.        |
| `run-deeplob-smoke --path PATH --lookback N --epochs N`                | Run a deterministic synthetic DeepLOB-style check.       |
| `inspect-transformer`                                                  | Print supervised transformer encoder defaults.           |
| `run-transformer-smoke --path PATH`                                    | Run a deterministic synthetic transformer check.         |
| `inspect-ssl`                                                          | Print self-supervised transformer wrapper defaults.      |
| `run-ssl-smoke --path PATH`                                            | Run a tiny synthetic self-supervised objective check.    |
| `inspect-multitask`                                                    | Print multi-task transformer defaults.                   |
| `run-multitask-smoke --path PATH`                                      | Run a tiny synthetic supervised multi-task check.        |

## Calibration, Execution and Analysis

| Command                              | Description                                                       |
| ------------------------------------ | ----------------------------------------------------------------- |
| `inspect-calibration`                | Print calibration and uncertainty utility support.                |
| `run-calibration-smoke`              | Run deterministic synthetic calibration diagnostics.              |
| `inspect-execution-validation`       | Print execution-aware validation support.                         |
| `run-execution-validation-smoke`     | Run deterministic synthetic execution-validation diagnostics.     |
| `inspect-analysis`                   | Print transfer, regime, ablation and sensitivity support.         |
| `run-robustness-analysis-smoke`      | Run deterministic synthetic robustness-analysis diagnostics.      |

## Experiment Artefacts

| Command                                                              | Description                                                        |
| -------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `inspect-experiment-artifacts --experiment PATH`                     | Inspect an experiment directory against the artefact contract.     |
| `prepare-fi2010-benchmark --config PATH --data-path PATH --out PATH` | Prepare a local-only FI-2010 benchmark input (no model run).       |
| `verify-fi2010-local --data-path PATH`                               | Inspect a local FI-2010 ``.txt`` matrix safely without loading it into memory; prints byte size, SHA-256, row and column counts, label class distribution and layout issues. |
| `convert-fi2010-official --input PATH --output PATH [--split train\|test] [--overwrite]` | Convert one official FI-2010 ``.txt`` matrix into a header-bearing CSV file matching the existing loader convention. Operates on user-supplied local files only. |
| `run-paper-experiment --config PATH --data-path PATH --out PATH [--models majority[,logistic,ridge,elastic_net,random_forest,gradient_boosting,deeplob_style,transformer]] [--overwrite] [--build-plots]` | Run the paper benchmark suite and write a validated artefact directory. Phase F additionally emits `calibration_bins.csv` and `execution_sensitivity.csv` when their inputs are available. Phase G adds `--build-plots`, which generates reproducible plots from stored artefacts. |
| `run-paper-ablations --config PATH --data-path PATH --out PATH [--models majority[,logistic,...]] [--ablation-set smoke|standard] [--overwrite] [--build-plots]` | Run controlled paper-experiment ablations and write `ablation_summary.json`, `ablation_results.csv`, `ablation_manifest.json`, child experiment directories for run ablations and explicit skip reports for unsupported ablations. |
| `run-system-benchmarks --config PATH --data-path PATH --out PATH [--benchmark-set smoke|standard] [--models majority[,logistic,...]] [--overwrite]` | Run local systems benchmarks and write `system_benchmark_summary.json`, `system_benchmark_results.csv`, `environment.json`, category reports and a validated child paper experiment for runner timing. |
| `inspect-system-benchmarks --benchmark PATH`                         | Print a concise, read-only summary of a systems benchmark directory. |
| `build-paper-plots --experiment PATH [--overwrite]`                  | Generate paper experiment plots (`plots/reliability_curve.png`, `plots/cost_sensitivity.png`, `plots/confusion_matrix.png`, and `plots/regime_breakdown.png` when genuine regime data is present) from the artefacts stored inside a completed paper experiment directory. |
| `inspect-paper-experiment --experiment PATH`                         | Print a concise, read-only summary of a paper experiment directory (validation status, evidence streams, prediction/calibration/execution row counts, plot inventory, fixture flag). |
| `build-paper-report --experiment PATH [--ablations PATH] [--systems PATH] --out PATH [--overwrite]` | Build a structured empirical report from stored paper experiment, ablation and systems benchmark artefacts. Missing optional artefacts are marked unavailable or skipped. |
| `inspect-paper-report --report PATH`                                 | Inspect a generated empirical report and its companion summary JSON. |

## Audit and Evidence Archive

| Command                                          | Description                                                            |
| ------------------------------------------------ | ---------------------------------------------------------------------- |
| `inspect-release-readiness`                      | Inspect README, documentation structure and wording without writing.   |
| `run-project-audit [--strict] [--root PATH]`     | Run local repository audit checks and print inventory counts.          |
| `build-report-archive [--output PATH] [--strict] [--include-smoke-training]` | Build or update the local evidence archive.    |
| `inspect-report-archive [--output PATH]`         | List expected archive files and whether they are present.              |
