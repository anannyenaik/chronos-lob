# ChronosLOB

**ChronosLOB: Self-Supervised Market Microstructure Modelling for
Execution-Aware Alpha Discovery**

Recruiter-facing title:

**Self-Supervised Market Microstructure Model with Execution-Aware Alpha
Validation**

ChronosLOB is a research-engineering platform for studying whether
self-supervised representations of limit order book dynamics improve
short-horizon market-state forecasts and whether those forecasts survive
execution-aware validation.

This repository is for research and educational purposes. It does not provide
financial advice and does not claim deployable or profitable trading performance.

## What This Project Is Trying To Prove

ChronosLOB is designed to support careful experiments around:

- whether limit order book representations learned without labels transfer to
  short-horizon forecasting tasks;
- whether forecast quality remains meaningful after calibration, costs and
  execution assumptions are considered;
- where the gap appears between forecast performance and tradable signal quality;
- how leakage-safe labels, temporal splits and reproducible experiment configs can
  make market microstructure research easier to audit.

## What This Project Is Not Claiming

This scaffold does not claim:

- implemented data loaders;
- trained models;
- benchmark results;
- trading performance;
- deployable execution logic;
- profitability or investment usefulness.

Prediction and trading are treated as separate problems. Benchmark accuracy is not
assumed to be tradable alpha.

## Current Status

**Scaffold only.**

The repository currently contains project rules, package structure, utility modules,
configuration conventions, documentation and tests. Model implementation, data
loading, feature engineering, labels and backtesting are planned future phases.

## Planned Architecture

The intended architecture separates data handling, market-state representation,
forecasting, validation and reporting:

- `chronoslob.data`: source adapters, schemas and validation.
- `chronoslob.book`: order book state, event replay and reconstruction.
- `chronoslob.features`: leakage-safe microstructure feature generation.
- `chronoslob.labels`: future-return and market-state labels with leakage tests.
- `chronoslob.models`: model definitions and representation learners.
- `chronoslob.training`: training loops, evaluation and experiment execution.
- `chronoslob.backtest`: simplified execution-aware research simulations.
- `chronoslob.analysis`: calibration, diagnostics and result export helpers.
- `chronoslob.utils`: shared utilities for seeding, logging and paths.

## Expected Data Sources

Planned public or user-provided sources may include:

- FI-2010 limit order book benchmark data;
- public crypto exchange market data for engineering demonstrations;
- local event logs generated from user-configured public data sources.

Crypto microstructure differs from equities and should not be presented as directly
equivalent. Private or licensed data should only be used if the user configures it
explicitly.

## Installation

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Basic Commands

```bash
python -c "import chronoslob; print(chronoslob.__version__)"
python -m chronoslob.cli version
python -m chronoslob.cli doctor
```

With `make` available:

```bash
make install
make smoke
make test
make lint
make typecheck
```

## Testing

```bash
pytest
pytest --cov=chronoslob
```

Initial tests cover package imports, deterministic seeding and path utilities. Future
modules should add behaviour-focused tests and explicit leakage checks.

## Roadmap

The high-level build plan is tracked in `PLANS.md`.

Near-term phases:

1. Repository scaffold and research design.
2. Core schemas and utilities.
3. FI-2010 benchmark loader.
4. Microstructure feature engine.
5. Label generation and leakage tests.

Later phases will add temporal splitters, baselines, PyTorch datasets,
self-supervised transformers, calibration, abstention and execution-aware research
simulation.

## Limitations

See `reports/limitations.md` for the current limitations statement. In short, this
repository currently contains scaffold code only, no model results exist yet and no
trading performance is claimed.

## CV Positioning

Suggested concise CV language:

> Built a reproducible research-engineering platform for limit order book
> representation learning, leakage-safe short-horizon forecasting and
> execution-aware alpha validation.

Use this positioning only with honest detail about the implemented phase and verified
artefacts.
