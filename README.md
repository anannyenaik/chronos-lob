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
microstructure feature engine, a future-window label engine, safe validation
infrastructure, classical baseline experiment utilities, a PyTorch
sequence-window data layer and a DeepLOB-style supervised CNN-LSTM
baseline, plus offline Binance-style order book reconstruction and
deterministic fixture replay, canonical event-log storage and
replay-to-feature integration.**

Phases 0 (scaffold), 1 (core schemas), 2 (FI-2010 local loader), 3
(microstructure feature engine), 4 (label generation and leakage checks), 5
(temporal splitters and experiment registry skeleton), 6 (classical baseline
interfaces, train-only preprocessing and metrics), 7A (PyTorch
sequence-window data layer) and 7B (DeepLOB-style supervised CNN-LSTM
baseline and minimal neural training smoke loop), and 8 (offline
Binance-style local order book reconstruction), and 9 (event-log storage
and deterministic replay-to-feature integration) are implemented. No
benchmark performance is claimed; transformer and self-supervised work
remain planned future phases. The
repository contains project rules, package
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
checks. `chronoslob.training` adds temporal train/validation/test splitters,
walk-forward folds, purged and embargoed validation helpers for overlapping label
horizons, a train-only quantile binner and a metadata-only experiment registry
skeleton. `chronoslob.models` and `chronoslob.training` also provide
majority-class, logistic-regression, ridge-classifier, elastic-net logistic,
random-forest and gradient-boosting baseline interfaces, train-only
standardisation, classification metrics and a smoke-tested baseline experiment
runner. `chronoslob.training` adds a PyTorch sequence-window data layer
(`SequenceDataset`, `SequenceWindowConfig`, train-only
`TorchSequenceStandardiser`, fixed- and variable-length collation helpers and
a `build_dataloaders_for_split` factory with safe non-shuffling defaults).
`chronoslob.models.deeplob` and `chronoslob.training.torch_training` plus
`chronoslob.training.torch_experiment` now provide a DeepLOB-style
supervised CNN-LSTM baseline, generic torch classification training
utilities and a DeepLOB smoke experiment runner that exercises the full
data layer end-to-end with deterministic CPU tests. The supervised neural
baseline writes no model checkpoints. `chronoslob.data.binance`,
`chronoslob.book.local_order_book`, `chronoslob.book.reconstruction` and
`chronoslob.book.replay` provide offline Binance-style snapshot and
diff-depth parsing, deterministic local replay, update-id gap detection,
stale-event skipping and crossed-book surfacing against synthetic fixtures
only. The CLI exposes a read-only `inspect-binance-replay` command for local
JSON/JSONL files. No live Binance connectivity, WebSockets, REST clients,
downloads, API keys or hidden network calls are implemented. No FI-2010
benchmark performance is reported by the repository.
`chronoslob.data.event_store` and `chronoslob.data.manifests` now provide
canonical local JSONL storage for `BookEvent` and `OrderBookSnapshot` records,
schema-preserving read/write helpers and SHA-256 manifests. `chronoslob.book`
adds event-log replay helpers that extract explicit snapshots, build past-only
feature frames, optionally build future-horizon label frames and run available
no-look-ahead checks. The CLI exposes read-only `inspect-event-log` and
`event-log-to-features` commands against local files. Generic event-level book
reconstruction, transformers, self-supervised representation learning and
execution backtests remain planned future phases.

## Planned Architecture

The intended architecture separates data handling, market-state representation,
forecasting, validation and reporting:

- `chronoslob.data`: source adapters, schemas and validation.
- `chronoslob.book`: order book state, event replay and reconstruction.
- `chronoslob.features`: leakage-safe microstructure feature generation.
- `chronoslob.labels`: future-return and market-state labels with leakage tests.
- `chronoslob.models`: model definitions and representation learners.
- `chronoslob.training`: temporal splitters, validation helpers and experiment
  metadata before future training loops.
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
python -m chronoslob.cli inspect-split --rows 100
python -m chronoslob.cli init-run --name split-audit --phase phase-5 --seed 42 --root runs
python -m chronoslob.cli inspect-baselines
python -m chronoslob.cli run-baseline-smoke --path tests/fixtures/fi2010/tiny_fi2010_like.csv
python -m chronoslob.cli inspect-torch-dataset --path tests/fixtures/fi2010/tiny_fi2010_like.csv --lookback 2
python -m chronoslob.cli inspect-deeplob
python -m chronoslob.cli run-deeplob-smoke --path tests/fixtures/fi2010/tiny_fi2010_like.csv --lookback 2 --epochs 1
python -m chronoslob.cli inspect-binance-replay --snapshot tests/fixtures/binance/synthetic_snapshot.json --updates tests/fixtures/binance/synthetic_diff_updates.jsonl
python -m chronoslob.cli inspect-event-log --path tests/fixtures/event_logs/synthetic_snapshots.jsonl
python -m chronoslob.cli event-log-to-features --path tests/fixtures/event_logs/synthetic_snapshots.jsonl
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
The `inspect-split` command prints default temporal split counts without reading
data. The `init-run` command creates only a run directory and `metadata.json`;
it does not create metrics, checkpoints or model outputs. See
`reports/validation_protocol.md`, `reports/experiment_registry.md` and
`configs/experiments/fi2010_split_audit.yaml`.
The `inspect-baselines` command lists supported classical baseline model types
without training. The `run-baseline-smoke` command runs a tiny synthetic-fixture
pipeline check and prints validation metrics with an explicit warning that the
output is not benchmark performance. It writes nothing unless `--write-outputs`
is passed, in which case outputs go under the gitignored `runs/` tree. See
`reports/baselines.md`, `configs/models/baselines.yaml` and
`configs/experiments/fi2010_baseline_smoke.yaml`.
The `inspect-torch-dataset` command builds a tiny PyTorch sequence
`DataLoader` from the synthetic fixture and prints sample counts, batch
shapes and the train-only class mapping. It is also explicitly labelled as
non-benchmark and writes nothing. See `reports/torch_data_layer.md` and
`configs/experiments/fi2010_torch_dataset_smoke.yaml`. PyTorch is an
optional dependency installed via `pip install -e ".[torch]"`.
The `inspect-deeplob` command prints the DeepLOB-style model defaults
without training. The `run-deeplob-smoke` command runs a tiny synthetic
fixture DeepLOB-style supervised smoke experiment and prints the
parameter count, training history and validation metrics. It writes
nothing, never saves model checkpoints and is explicitly labelled as
non-benchmark performance. See `reports/deeplob_baseline.md`,
`configs/models/deeplob.yaml` and
`configs/experiments/fi2010_deeplob_smoke.yaml`.
The `inspect-binance-replay` command reconstructs a local Binance-style book
from synthetic/local snapshot and diff files, prints update-id and issue counts
and writes nothing. The `inspect-event-log` command validates a canonical
event-log JSONL file and prints manifest-style counts, symbols, timestamp range,
sequence range and a SHA-256 prefix. The `event-log-to-features` command
replays explicit snapshots from a canonical event log into the existing
past-only feature pipeline, validates the frame and writes nothing. See
`reports/order_book_reconstruction.md`, `reports/event_log_storage.md`,
`reports/replay_to_features.md`, `configs/data/event_log.yaml` and
`configs/experiments/event_log_feature_audit.yaml`.

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
FI-2010 loading, feature generation, label generation, explicit leakage checks,
temporal splitters, purged/embargoed validation, train-only fitting and
experiment metadata and classical baseline evaluation. Future modules should
keep adding behaviour-focused tests.

## Roadmap

The high-level build plan is tracked in `PLANS.md`.

Completed phases:

0. Repository scaffold and research design.
1. Core schemas and utilities.
2. FI-2010 benchmark loader.
3. Microstructure feature engine.
4. Label generation and leakage checks.
5. Temporal splitters and experiment registry.
6. Classical baseline interfaces, train-only preprocessing and metrics.
7A. PyTorch sequence-window data layer (datasets, batching, dataloaders).
7B. DeepLOB-style supervised CNN-LSTM baseline and minimal neural
   training smoke loop with train-only standardisation.
8. Offline Binance-style local order book reconstruction.
9. Canonical event-log storage and deterministic replay-to-feature integration.

Later phases will add event tokenisation, self-supervised transformers, calibration,
abstention and execution-aware research simulation. No benchmark
performance is claimed by the DeepLOB-style baseline; it is a
reproducible supervised neural baseline, not a production trading model.

## Limitations

See `reports/limitations.md` for the current limitations statement. In short, this
repository currently contains data, feature, label, split, baseline, supervised
sequence-model, offline reconstruction and event-log replay infrastructure. No
transformer, self-supervised training objective, execution backtest or trading
performance claim exists. Classical and DeepLOB-style utilities can produce
metrics only when a user runs an experiment, and synthetic smoke outputs are not
FI-2010 benchmark results.

## CV Positioning

Suggested concise CV language:

> Built a reproducible research-engineering platform for limit order book
> representation learning, leakage-safe short-horizon forecasting and
> execution-aware alpha validation.

Use this positioning only with honest detail about the implemented phase and verified
artefacts.
