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

## Audit and Evidence Archive

| Command                                          | Description                                                            |
| ------------------------------------------------ | ---------------------------------------------------------------------- |
| `inspect-release-readiness`                      | Inspect README, documentation structure and wording without writing.   |
| `run-project-audit [--strict] [--root PATH]`     | Run local repository audit checks and print inventory counts.          |
| `build-report-archive [--output PATH] [--strict] [--include-smoke-training]` | Build or update the local evidence archive.    |
| `inspect-report-archive [--output PATH]`         | List expected archive files and whether they are present.              |
