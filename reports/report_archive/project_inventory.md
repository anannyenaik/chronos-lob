# Project Inventory

Snapshot of repository structure for reproducibility and review.

- Package version: `0.2.0`
- Config files: `31`
- Report files, excluding this generated archive: `49`
- Test files: `113`
- CLI commands: `69`

## Major Package Areas

- `chronoslob.data`
- `chronoslob.book`
- `chronoslob.features`
- `chronoslob.labels`
- `chronoslob.training`
- `chronoslob.models`
- `chronoslob.backtest`
- `chronoslob.analysis`
- `chronoslob.utils`

## Current CLI Commands

- `version`
- `doctor`
- `run-project-audit`
- `inspect-release-readiness`
- `build-report-archive`
- `inspect-report-archive`
- `inspect-experiment-artifacts`
- `inspect-event-log`
- `inspect-event-tokens`
- `event-log-to-features`
- `inspect-fi2010`
- `inspect-features-fi2010`
- `inspect-labels-fi2010`
- `inspect-split`
- `init-run`
- `inspect-baselines`
- `run-baseline-smoke`
- `inspect-torch-dataset`
- `inspect-deeplob`
- `run-deeplob-smoke`
- `inspect-transformer`
- `run-transformer-smoke`
- `inspect-ssl`
- `run-ssl-smoke`
- `inspect-multitask`
- `run-multitask-smoke`
- `inspect-calibration`
- `run-calibration-smoke`
- `inspect-execution-validation`
- `run-execution-validation-smoke`
- `inspect-analysis`
- `run-robustness-analysis-smoke`
- `inspect-binance-replay`
- `prepare-fi2010-benchmark`
- `verify-fi2010-local`
- `convert-fi2010-official`
- `inspect-fi2010-multifold`
- `prepare-fi2010-multifold`
- `run-fi2010-multifold-classical`
- `inspect-fi2010-neural-plan`
- `run-fi2010-neural-benchmark`
- `run-fi2010-ssl-neural-benchmark`
- `run-fi2010-ssl-v2-benchmark`
- `run-fi2010-neural-full-grid`
- `build-fi2010-figures`
- `audit-fi2010-features`
- `run-fi2010-feature-ablations`
- `build-fi2010-ablation-figures`
- `analyse-fi2010-feature-ablations`
- `analyse-fi2010-uncertainty`
- `analyse-fi2010-ssl-results`
- `analyse-fi2010-ssl-v2-results`
- `analyse-fi2010-execution-v3`
- `build-execution-centrepiece`
- `run-fi2010-brutal-ablations`
- `run-fi2010-execution-v2`
- `build-fi2010-execution-v3`
- `run-paper-experiment`
- `run-paper-ablations`
- `run-system-benchmarks`
- `inspect-system-benchmarks`
- `build-paper-plots`
- `inspect-paper-experiment`
- `build-paper-report`
- `build-final-empirical-report`
- `build-evidence-pack`
- `run-synthetic-lob-benchmark`
- `replay-binance-l2-sample`
- `inspect-paper-report`

## Validation Command List

- `python -c "import chronoslob; print(chronoslob.__version__)"`
- `python -m chronoslob.cli doctor`
- `python -m chronoslob.cli inspect-release-readiness`
- `python -m chronoslob.cli run-project-audit --strict`
- `python -m pytest`
- `python -m compileall -q chronoslob tests`
- `python -m ruff check .`
- `python -m mypy chronoslob`
