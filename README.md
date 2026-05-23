# ChronosLOB

[![CI](https://github.com/anannyenaik/chronos-lob/actions/workflows/ci.yml/badge.svg)](https://github.com/anannyenaik/chronos-lob/actions/workflows/ci.yml)

ChronosLOB is a research platform for limit order book representation
learning, market-state forecasting, calibration and execution-aware
validation.

## Overview

The project treats limit order book data as noisy, high-frequency sequential
data and provides infrastructure for local ingestion, leakage-safe feature
and label construction, temporal validation, classical and neural baselines,
transformer-based representation learning, calibration and uncertainty
analysis, execution-aware validation and robustness analysis.

It is designed so that prediction quality, calibration quality and
cost-aware signal quality remain separate evidence streams. Forecast
accuracy is never used as a proxy for tradability, and execution
assumptions are always explicit.

ChronosLOB is research software. It is not financial advice, not live
trading infrastructure and not a production execution system.

## Why This Exists

- Short-horizon forecast metrics alone do not characterise whether a
  signal is useful under realistic costs and latency.
- Data leakage is easy to introduce in financial sequences through
  shuffled splits, scaler fitting on full series or future-information
  features.
- Calibration and uncertainty are central to any downstream filtering
  or abstention policy.
- Costs, latency and fill assumptions can invert the apparent ordering
  of models.
- Reproducibility requires explicit data provenance, seeds, code
  versions and stored outputs.

## Architecture

| Layer                           | What it contains                                                                |
| ------------------------------- | ------------------------------------------------------------------------------- |
| Data contracts and validation   | FI-2010-style loaders, canonical event records, schemas, synthetic fixtures.    |
| Feature and label engineering   | Past-only microstructure features, future-window labels, no-look-ahead checks.  |
| Temporal validation             | Temporal, walk-forward and purged or embargoed splitters; experiment registry.  |
| Baselines and sequence models   | Classical baselines, DeepLOB-style CNN-LSTM, PyTorch sequence-window datasets.  |
| Representation learning         | Event tokenisation, supervised transformer encoder, self-supervised objectives. |
| Calibration and uncertainty     | Temperature scaling, calibration error, confidence filtering and abstention.    |
| Execution-aware validation      | Configured fees, spread costs, latency, turnover and risk constraints.          |
| Robustness analysis             | Transfer, regime, ablation and sensitivity summaries over experiment records.   |
| Audit and reproducibility       | Local audit checks, release-readiness inspection and evidence archive builder.  |

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

The `torch` extra is optional for installation but required for the full
test suite.

## Quick Validation

```bash
python -m chronoslob.cli doctor
python -m chronoslob.cli inspect-release-readiness
python -m chronoslob.cli run-project-audit --strict
python -m pytest
python -m compileall -q chronoslob tests
python -m ruff check .
python -m mypy chronoslob
```

## Example Local Inspections

```bash
python -c "import chronoslob; print(chronoslob.__version__)"
python -m chronoslob.cli inspect-fi2010 --path tests/fixtures/fi2010/tiny_fi2010_like.csv
python -m chronoslob.cli inspect-event-log --path tests/fixtures/event_logs/synthetic_snapshots.jsonl
python -m chronoslob.cli inspect-transformer
python -m chronoslob.cli inspect-calibration
python -m chronoslob.cli inspect-execution-validation
```

See the [CLI reference](docs/CLI_REFERENCE.md) for the full command
inventory.

## Data Policy

No real exchange data, licensed data or credentials are committed.
Fixtures under `tests/fixtures/` are small synthetic files used only to
exercise the code paths. Users supply any real FI-2010 or public venue
data locally, outside version control.

## Documentation

- [Reproducibility](docs/REPRODUCIBILITY.md)
- [CLI reference](docs/CLI_REFERENCE.md)
- [Project status](docs/PROJECT_STATUS.md)
- [Safety and limitations](docs/SAFETY_AND_LIMITATIONS.md)
- [Roadmap](ROADMAP.md)
- [Contributing](CONTRIBUTING.md)

## Status

The repository contains tested infrastructure for data contracts,
feature and label generation, temporal validation, classical and
DeepLOB-style baselines, transformer modelling, self-supervised
objectives, multi-task fine-tuning, calibration utilities,
execution-aware validation and robustness analysis.

Empirical result tables are intentionally not included until experiments
are run from documented data sources, configs and seeds. Boundaries on
what the project does and does not model are described in
[docs/SAFETY_AND_LIMITATIONS.md](docs/SAFETY_AND_LIMITATIONS.md).
