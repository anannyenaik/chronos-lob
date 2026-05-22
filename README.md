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

- trained models;
- benchmark results;
- replication of any published FI-2010 outcome;
- trading performance;
- deployable execution logic;
- profitability or investment usefulness.

A local FI-2010 loader exists, but it neither downloads benchmark data nor
produces any forecast metrics. Users must obtain and supply the benchmark
locally.

Prediction and trading are treated as separate problems. Benchmark accuracy is not
assumed to be tradable alpha.

## Current Status

**Scaffold, canonical schemas, a local FI-2010 loader, a leakage-safe
microstructure feature engine and a future-window label engine.**

Phases 0 (scaffold), 1 (core schemas), 2 (FI-2010 local loader) and 3
(microstructure feature engine) are complete, and Phase 4 (label generation and
leakage checks) is implemented. The repository contains project rules, package
structure, utility modules, configuration conventions, documentation and tests,
and defines canonical schemas for market events, order book snapshots, feature
rows, label rows and data-quality findings in `chronoslob.data.schemas` with
supporting helpers in `chronoslob.book.events`.
`chronoslob.data.fi2010` adds a configurable, local-file loader for FI-2010-style
benchmark matrices and `chronoslob.data.validation` provides the corresponding
data-quality checks. `chronoslob.features` implements a past-only microstructure
feature engine — mid-price, spread, microprice, depth/queue imbalance, order-flow
imbalance, rolling realised volatility, event intensity and rule-based regime
flags — together with a `FeaturePipelineConfig` that assembles `FeatureRow`
objects and pandas feature frames without ever including label columns. Users
must supply benchmark data locally: no FI-2010 data is downloaded or bundled,
and no benchmark performance is claimed. `chronoslob.labels` now implements
future return, direction, return-quantile, volatility, spread-widening, passive
fill proxy and adverse-selection proxy labels, plus explicit no-look-ahead
checks. Models, training loops and execution backtests remain planned future
phases.

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
python -m chronoslob.cli inspect-fi2010 --path tests/fixtures/fi2010/tiny_fi2010_like.csv
python -m chronoslob.cli inspect-features-fi2010 --path tests/fixtures/fi2010/tiny_fi2010_like.csv
python -m chronoslob.cli inspect-labels-fi2010 --path tests/fixtures/fi2010/tiny_fi2010_like.csv
```

The `inspect-fi2010` command is read-only: it loads a local FI-2010-style file,
runs the validator and prints a short summary. It does not download, train or
write anything. A configurable example sits at `configs/data/fi2010.yaml`.
The `inspect-features-fi2010` command additionally builds the microstructure
feature frame, validates it and prints row/feature counts plus a sample of
feature column names — it is also read-only. See
`reports/feature_engine.md` for the documented feature definitions and
`configs/experiments/feature_audit_fi2010.yaml` for a worked configuration.
The `inspect-labels-fi2010` command extracts configured FI-2010 benchmark labels
or generates ChronosLOB labels from snapshots, validates the label frame and
prints a short read-only summary. See `reports/label_engine.md`,
`reports/leakage_controls.md` and `configs/experiments/label_audit_fi2010.yaml`.

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

Tests cover package imports, deterministic seeding, path utilities, schemas,
FI-2010 loading, feature generation, label generation and explicit leakage
checks. Future modules should keep adding behaviour-focused tests.

## Roadmap

The high-level build plan is tracked in `PLANS.md`.

Near-term phases:

1. Repository scaffold and research design.
2. Core schemas and utilities.
3. FI-2010 benchmark loader.
4. Microstructure feature engine.
5. Temporal splitters and experiment registry.

Later phases will add temporal splitters, baselines, PyTorch datasets,
self-supervised transformers, calibration, abstention and execution-aware research
simulation.

## Limitations

See `reports/limitations.md` for the current limitations statement. In short, this
repository currently contains data, feature and label infrastructure only. No
model results exist yet and no trading performance is claimed.

## CV Positioning

Suggested concise CV language:

> Built a reproducible research-engineering platform for limit order book
> representation learning, leakage-safe short-horizon forecasting and
> execution-aware alpha validation.

Use this positioning only with honest detail about the implemented phase and verified
artefacts.
