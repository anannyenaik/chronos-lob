# ChronosLOB CLI Reference

The CLI is a local research-engineering interface. Commands are read-only unless
their help text explicitly says that outputs are written, and no command performs
hidden network calls.

Run commands with:

```bash
python -m chronoslob.cli <command> [options]
```

## Version And Doctor

- `version`: print the installed package version.
- `doctor`: print Python, package and key-folder checks.

## FI-2010 Inspection

- `inspect-fi2010 --path PATH`: load a local FI-2010-style file and print a
  data-quality summary.

## Feature Inspection

- `inspect-features-fi2010 --path PATH`: build and validate leakage-safe
  microstructure features from a local FI-2010-style file.
- `event-log-to-features --path PATH`: replay a canonical local event log into
  past-only feature rows and print a summary.

## Label Inspection

- `inspect-labels-fi2010 --path PATH`: extract configured FI-2010 labels or
  build ChronosLOB labels and print validation counts.

## Split And Registry

- `inspect-split --rows N`: print default temporal train, validation and test
  split counts.
- `init-run --name NAME --phase PHASE --seed SEED --root RUNS`: create a
  metadata-only run directory.

## Baselines

- `inspect-baselines`: list supported classical baseline model types.
- `run-baseline-smoke --path PATH`: run a tiny synthetic-fixture baseline
  plumbing check. It is not benchmark performance.

## Torch Dataset

- `inspect-torch-dataset --path PATH --lookback N`: build a tiny sequence
  `DataLoader` from a local fixture and print shape and class-mapping details.

## DeepLOB Smoke

- `inspect-deeplob`: print DeepLOB-style supervised baseline defaults.
- `run-deeplob-smoke --path PATH --lookback N --epochs N`: run a deterministic
  synthetic-fixture DeepLOB plumbing check.

## Binance Replay

- `inspect-binance-replay --snapshot PATH --updates PATH`: reconstruct a local
  Binance-style order book from supplied JSON/JSONL files. This is offline only.

## Event Log

- `inspect-event-log --path PATH`: inspect canonical local JSONL event-log
  records and print manifest-style counts.

## Tokenisation

- `inspect-event-tokens --path PATH`: tokenise a canonical event log and print
  vocabulary and window counts.

## Transformer

- `inspect-transformer`: print supervised transformer encoder defaults.
- `run-transformer-smoke --path PATH`: run a deterministic synthetic-label
  transformer plumbing check.

## SSL

- `inspect-ssl`: print self-supervised transformer wrapper defaults.
- `run-ssl-smoke --path PATH`: run a tiny synthetic self-supervised objective
  plumbing check.

## Multi-Task

- `inspect-multitask`: print multi-task transformer defaults.
- `run-multitask-smoke --path PATH`: run a tiny synthetic supervised multi-task
  plumbing check.

## Calibration

- `inspect-calibration`: print calibration and uncertainty utility support.
- `run-calibration-smoke`: run deterministic synthetic calibration diagnostics.

## Execution Validation

- `inspect-execution-validation`: print execution-aware validation support.
- `run-execution-validation-smoke`: run deterministic synthetic
  execution-validation plumbing.

## Analysis

- `inspect-analysis`: print transfer, regime, ablation and sensitivity support.
- `run-robustness-analysis-smoke`: run deterministic synthetic robustness
  analysis plumbing.

## Audit

- `run-project-audit`: run local repository audit checks and print inventory
  counts.
- `run-project-audit --strict`: exit non-zero if warnings or failures are found.
- `run-project-audit --root PATH`: audit an explicit repository root.

## Safety Notes

Smoke commands use bundled synthetic fixtures or deterministic synthetic records.
Their outputs are plumbing checks only. They must not be reported as FI-2010
benchmark results, real venue evidence, alpha evidence or execution performance.
