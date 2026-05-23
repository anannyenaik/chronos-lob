# Project Inventory

This inventory supports later manual report writing. It is not a final technical report and it contains no benchmark result claims.

- Package version: `0.1.0`
- Config files: `24`
- Report files, excluding this generated archive: `22`
- Test files: `73`
- CLI commands: `31`

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
- `build-report-archive`
- `inspect-report-archive`
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

## Validation Command List

- `python -c "import chronoslob; print(chronoslob.__version__)"`
- `python -m chronoslob.cli doctor`
- `python -m chronoslob.cli run-project-audit --strict`
- `python -m pytest`
- `python -m compileall -q chronoslob tests`
- `python -m ruff check .`
- `python -m mypy chronoslob`

## Evidence Boundary

The inventory describes implemented research-engineering artefacts. Real performance claims require separately generated experiment outputs with documented data provenance, configs, seeds and code versions.
