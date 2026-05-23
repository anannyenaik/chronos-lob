# ChronosLOB

[![CI](https://github.com/anannyenaik/chronos-lob/actions/workflows/ci.yml/badge.svg)](https://github.com/anannyenaik/chronos-lob/actions/workflows/ci.yml)

ChronosLOB is a research platform for leakage-safe limit order book
representation learning, market-state forecasting, calibration and
execution-aware validation.

ChronosLOB is research software. It is not financial advice, not live trading
infrastructure and not a production execution system. It makes no deployment or
trading-use claims, and no real benchmark results are claimed unless they are
later added as reproducible experiment artefacts.

## Purpose

The project studies whether self-supervised representations of limit order book
dynamics can improve short-horizon market-state forecasting, and whether those
forecasts remain useful under explicit execution assumptions. It keeps forecast
quality, calibration quality and cost-aware signal quality separate so that
accuracy is not mistaken for tradability.

ChronosLOB emphasises leakage-safe labels, temporal splits, train-only fitting,
calibration, execution assumptions, robustness analysis and reproducible
experiment records.

## Architecture

- Data and schemas: local FI-2010-style loading, canonical event records,
  validation schemas and small synthetic fixtures.
- Book reconstruction: offline Binance-style replay, local order book state and
  event-log-to-feature conversion.
- Features and labels: past-only microstructure features, future-window labels
  and no-look-ahead checks.
- Splits and experiments: temporal, walk-forward and purged or embargoed
  validation helpers plus metadata-only run records.
- Baselines and models: classical baselines, DeepLOB-style plumbing,
  transformer encoders, self-supervised objectives and multi-task heads.
- Calibration: temperature scaling, expected calibration error, abstention and
  confidence filtering utilities.
- Execution-aware validation: configured fees, spread costs, latency, turnover,
  simple risk constraints and adverse-selection summaries.
- Robustness analysis: transfer, regime, ablation and sensitivity summaries for
  supplied experiment records.
- Audit and evidence archive: local release checks, reproducibility checks and a
  technical evidence archive for later manual report writing.

## Installation

Requires Python 3.11 or newer.

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

The `torch` extra is optional for installation, but the full validation suite
uses torch-backed modules.

## Validation

```bash
python -m chronoslob.cli doctor
python -m chronoslob.cli inspect-release-readiness
python -m chronoslob.cli run-project-audit --strict
python -m pytest
python -m compileall -q chronoslob tests
python -m ruff check .
python -m mypy chronoslob
```

## Example Commands

These commands are lightweight and local-only:

```bash
python -c "import chronoslob; print(chronoslob.__version__)"
python -m chronoslob.cli doctor
python -m chronoslob.cli inspect-fi2010 --path tests/fixtures/fi2010/tiny_fi2010_like.csv
python -m chronoslob.cli inspect-event-log --path tests/fixtures/event_logs/synthetic_snapshots.jsonl
python -m chronoslob.cli inspect-transformer
python -m chronoslob.cli inspect-calibration
python -m chronoslob.cli inspect-execution-validation
python -m chronoslob.cli inspect-release-readiness
```

See the [CLI reference](docs/CLI_REFERENCE.md) for the full command inventory.

Synthetic fixture outputs are plumbing checks only. They are not benchmark
results, market evidence, execution evidence or proof of cost-aware signal
quality.

## Data Policy

No real exchange data, private data, licensed data, API keys or credentials are
committed. Files under `tests/fixtures/` are synthetic and intentionally small.
Users must provide any real FI-2010 or public venue data locally, outside
version control, before generating result artefacts.

## Documentation

- [Reproducibility](docs/REPRODUCIBILITY.md)
- [CLI reference](docs/CLI_REFERENCE.md)
- [Project status](docs/PROJECT_STATUS.md)
- [Safety and limitations](docs/SAFETY_AND_LIMITATIONS.md)
- [Report evidence index](docs/REPORT_EVIDENCE_INDEX.md)
- [Roadmap](ROADMAP.md)
- [Contributing](CONTRIBUTING.md)

The technical evidence archive lives in [reports/report_archive](reports/report_archive/).
It supports later manual report writing and does not contain final report prose
or result claims.

## Status

Implemented components include local loaders, schemas, leakage-safe features,
future-window labels, temporal validation helpers, baseline and transformer
plumbing, self-supervised objectives, multi-task training infrastructure,
calibration utilities, execution-aware validation utilities, robustness
summaries, local audit checks and deterministic synthetic fixtures.

Not implemented: live data ingestion, broker integration, order placement,
production queue modelling, production partial-fill modelling, production market
impact modelling, portfolio optimisation, dashboard outputs, committed real
benchmark results or a final technical report.
