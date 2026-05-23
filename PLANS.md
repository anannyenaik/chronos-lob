# ChronosLOB Build Plan

This plan keeps implementation work staged, auditable and leakage-aware. Each phase
should leave the repository in a tested state.

## Phase 0: Repo scaffold and research design

Goal: Establish the repository foundation, project rules, package skeleton, tooling
and initial documentation.

Files likely to be touched: `AGENTS.md`, `README.md`, `PLANS.md`, `pyproject.toml`,
`Makefile`, `.gitignore`, `chronoslob/`, `tests/`, `configs/`, `reports/`,
`notebooks/`.

Acceptance criteria: Package imports, CLI smoke checks work, documentation clearly
states scope and limitations, no fake results are present.

Tests expected: Import tests, seeding tests, path utility tests.

## Phase 1: Core schemas and utilities (completed)

Goal: Define typed schemas for events, snapshots, trades, quotes, instruments and
time handling.

Files likely to be touched: `chronoslob/data/`, `chronoslob/book/`,
`chronoslob/utils/`, `tests/`.

Acceptance criteria: Schemas validate required fields, timestamp conventions are
documented, invalid records fail loudly.

Tests expected: Schema validation tests, timestamp ordering tests, serialisation
round-trip tests.

Status: `chronoslob.data.schemas` defines `OrderBookLevel`, `OrderBookSnapshot`,
`BookEvent`, `FeatureRow`, `LabelRow`, `DataQualityIssue`, the `Side` and
`EventType` enums and helpers `ensure_utc_datetime`, `is_finite_number` and
`validate_metadata`. `chronoslob.book.events` provides `sort_levels_for_side`,
`validate_book_side_order`, `has_duplicate_prices` and `top_of_book`. Validation
and helper tests live in `tests/test_schemas.py` and `tests/test_events.py`.

## Phase 2: FI-2010 benchmark loader (completed)

Goal: Add a loader for FI-2010 benchmark files without changing labels or splits
implicitly.

Files likely to be touched: `chronoslob/data/`, `configs/data/`, `tests/`.

Acceptance criteria: Loader reads local user-provided files, validates shapes and
documents assumptions.

Tests expected: Fixture-based loading tests, missing-file tests, malformed-input
tests.

Status: `chronoslob.data.fi2010` provides `FI2010Config`, `FI2010Dataset`,
`load_fi2010`, `infer_fi2010_columns` and `build_snapshot_from_row` for reading
local FI-2010-style matrices, with synthetic-timestamp marking on snapshot
conversion. `chronoslob.data.validation` provides `DataValidationResult`,
`validate_numeric_frame` and `validate_fi2010_dataset` and is invoked
automatically by `load_fi2010`. An example config lives in
`configs/data/fi2010.yaml`, a synthetic fixture in
`tests/fixtures/fi2010/tiny_fi2010_like.csv` and the data-quality checks are
documented in `reports/data_quality.md`. A `python -m chronoslob.cli
inspect-fi2010 --path ...` command exposes the loader as a read-only inspection
utility. No FI-2010 data is downloaded or committed and no benchmark
performance is claimed.

## Phase 3: Microstructure feature engine (completed)

Goal: Implement leakage-safe features using only information available at or before
timestamp `t`.

Files likely to be touched: `chronoslob/features/`, `chronoslob/book/`, `tests/`.

Acceptance criteria: Feature definitions are documented, windows are backward-looking
and transforms are explicit.

Tests expected: Feature-value tests on small fixtures, no-look-ahead tests, edge-case
window tests.

Status: `chronoslob.features` ships `microprice`, `imbalance`, `order_flow`,
`volatility`, `regimes` and `pipeline` modules. Single-snapshot features
include mid-price, spread, relative spread, microprice, per-depth bid/ask
depths, depth imbalance, queue imbalance, depth slope and liquidity
concentration. Sequence-level features include a simple top-of-book order
flow imbalance, rolling realised volatility on log returns of the mid-price
and rolling event intensity over a trailing window. `FeaturePipelineConfig`
and `build_feature_frame_from_snapshots`/`build_feature_frame_from_fi2010`
assemble pandas frames whose `synthetic_time` metadata gates time-window
features and whose label columns are never propagated as features.
`validate_feature_frame` checks for missing required columns, non-numeric
features, infinities, NaNs and label-like column names. A
`python -m chronoslob.cli inspect-features-fi2010 --path ...` command
exposes the pipeline as a read-only inspection utility. Documentation lives
in `reports/feature_engine.md` and an example configuration in
`configs/experiments/feature_audit_fi2010.yaml`. No labels, models,
backtests or trading claims are introduced.

## Phase 4: Label generation and leakage tests (completed)

Goal: Implement future-looking labels while guaranteeing labels cannot enter feature
inputs.

Files likely to be touched: `chronoslob/labels/`, `chronoslob/features/`, `tests/`.

Acceptance criteria: Label horizons and alignment rules are configurable and
documented.

Tests expected: Horizon alignment tests, leakage guard tests, boundary-condition
tests.

Status: `chronoslob.labels` now provides future mid-price return, direction,
return-quantile, future realised-volatility, spread-widening, passive-fill proxy
and adverse-selection proxy labels. `LabelPipelineConfig`,
`build_label_rows_from_snapshots`, `build_label_frame_from_snapshots` and
`build_label_frame_from_fi2010` assemble `LabelRow` objects and pandas label
frames while keeping features and labels separate. Existing FI-2010 configured
labels are preserved as benchmark labels with
`label_source = "fi2010_existing_labels"` rather than being presented as
ChronosLOB-generated labels. `chronoslob.labels.leakage` provides explicit
feature/label separation, temporal horizon and no-look-ahead checks.
Documentation lives in `reports/label_engine.md` and
`reports/leakage_controls.md`, with an example audit configuration in
`configs/experiments/label_audit_fi2010.yaml`. No models, baselines, backtests
or result artefacts are introduced.

## Phase 5: Temporal splitters and experiment registry (completed)

Goal: Add temporal train/validation/test splitters and a registry for reproducible
experiment artefacts.

Files likely to be touched: `chronoslob/training/`, `configs/experiments/`,
`chronoslob/utils/`, `tests/`.

Acceptance criteria: Random splits are not the default for financial data, split
metadata is persisted.

Tests expected: Split ordering tests, no-overlap tests, registry metadata tests.

Status: `chronoslob.training` now provides contiguous temporal
train/validation/test splitting, expanding and rolling walk-forward folds,
purged and embargoed training-index filtering for overlapping future label
horizons, timestamp-to-row horizon mapping and a `TrainOnlyQuantileBinner` for
train-only return quantile fitting. `chronoslob.training.experiment`,
`config` and `artifacts` provide a lightweight metadata registry that captures
run name, phase, UTC creation time, seed, git commit when available, config
path, input paths and output path. The CLI exposes `inspect-split` and
metadata-only `init-run` commands. Documentation lives in
`reports/validation_protocol.md` and `reports/experiment_registry.md`, with an
example config in `configs/experiments/fi2010_split_audit.yaml`. No models,
baselines, training loops, backtests, fake metrics or result artefacts are
introduced.

## Phase 6: Classical baselines (completed)

Goal: Implement simple, reproducible forecasting baselines before deep learning.

Files likely to be touched: `chronoslob/models/`, `chronoslob/training/`,
`configs/models/`, `tests/`.

Acceptance criteria: Baselines train from configs and report only reproducible
metrics.

Tests expected: Fit/predict tests, deterministic baseline tests, config validation
tests.

Status: `chronoslob.models` now provides train-only preprocessing helpers,
feature/target matrix containers, strict feature/label alignment and classical
baseline wrappers for majority class, logistic regression, ridge classifier,
elastic-net logistic regression, random forest and gradient boosting.
`chronoslob.training.metrics`, `evaluate` and `baseline_experiment` provide
classification metrics, confusion matrices, temporal baseline execution,
train-fitted standardisation and optional Phase 5 registry output. The CLI
exposes `inspect-baselines` and synthetic-fixture `run-baseline-smoke`; the smoke
command is explicitly labelled as non-benchmark output and writes nothing unless
requested. Documentation lives in `reports/baselines.md`, model defaults in
`configs/models/baselines.yaml` and the synthetic smoke config in
`configs/experiments/fi2010_baseline_smoke.yaml`. No deep learning models,
backtests, fake benchmark metrics or committed run outputs are introduced.

## Phase 7: PyTorch datasets and DeepLOB-style baseline

Goal: Add PyTorch datasets and a DeepLOB-style supervised baseline.

Files likely to be touched: `pyproject.toml`, `chronoslob/models/`,
`chronoslob/training/`, `configs/models/`, `tests/`.

Acceptance criteria: PyTorch dependency is justified, dataset indexing is
leakage-safe and baseline training is configurable.

Tests expected: Dataset slicing tests, shape tests, deterministic small-training
smoke tests.

### Phase 7A: PyTorch sequence data layer (completed)

Status: `chronoslob.training` now provides past-only sequence-window indexing
(`SequenceWindowConfig`, `SequenceSampleIndex`, `build_sequence_indices`),
a PyTorch `SequenceDataset` that aligns feature/label frames and emits
`[lookback, n_features]` windows with scalar long targets, a
`TorchSequenceStandardiser` for explicit train-only feature scaling,
fixed-length and variable-length collation helpers
(`collate_fixed_length_batch`, `collate_variable_length_batch`,
`pad_variable_length_sequences`), a `DataLoaderConfig` with safe
non-shuffling defaults and a `build_dataloaders_for_split` factory that
keeps windows inside their partitions and reuses the train class mapping
for validation and test. PyTorch is declared as an optional `[torch]`
dependency in `pyproject.toml`. The CLI exposes
`inspect-torch-dataset` against the bundled synthetic fixture and writes
nothing. Documentation lives in `reports/torch_data_layer.md` and the
smoke configuration in
`configs/experiments/fi2010_torch_dataset_smoke.yaml`. No neural network
architectures, training loops, checkpoints, backtests or fake benchmark
metrics are introduced.

### Phase 7B: DeepLOB-style CNN-LSTM supervised baseline (completed)

Status: `chronoslob.models.deeplob` ships a compact DeepLOB-style
supervised CNN-LSTM (`DeepLOBConfig`, `DeepLOBModel`,
`create_deeplob_model`) that accepts `[batch, lookback, n_features]`
tensors and emits `[batch, n_classes]` logits.
`chronoslob.training.torch_training` ships generic torch classification
utilities (`TorchTrainingConfig`, `TorchEpochResult`,
`set_torch_deterministic`, `train_one_epoch`,
`evaluate_torch_classifier`, `fit_torch_classifier`) that reuse the
existing classification metrics layer.
`chronoslob.training.torch_experiment` provides
`DeepLOBExperimentConfig`, `run_deeplob_experiment` and
`run_deeplob_smoke_from_fi2010_fixture`; the experiment runner aligns
feature and label frames, validates leakage, builds a temporal split,
fits train-only mean/std on training rows only and reuses the existing
sequence dataloaders. The CLI exposes `inspect-deeplob` and
`run-deeplob-smoke` against the bundled synthetic fixture and writes
nothing. Documentation lives in `reports/deeplob_baseline.md`, model
defaults in `configs/models/deeplob.yaml` and the synthetic smoke
configuration in `configs/experiments/fi2010_deeplob_smoke.yaml`. The
phase ships forward/backward/gradient tests, training-loop tests and
experiment runner tests; the supervised neural baseline writes no
model checkpoints. No transformers, self-supervised learning,
backtests, fake benchmark metrics or committed run outputs are
introduced.

## Phase 8: Binance local order book reconstruction (completed)

Goal: Reconstruct local order book state from public exchange messages for
engineering demonstrations.

Files likely to be touched: `chronoslob/data/`, `chronoslob/book/`, `configs/data/`,
`tests/`.

Acceptance criteria: Reconstruction uses local files or explicit user-configured
sources, sequence gaps fail loudly.

Tests expected: Replay tests, gap-detection tests, snapshot/delta consistency tests.

Status: `chronoslob.data.binance` provides offline Binance-style snapshot and
diff-depth schemas, local JSON/JSONL loaders and conversion to the canonical
`OrderBookSnapshot` schema. `chronoslob.book.local_order_book` manages a
deterministic in-memory book with sorted bid/ask views, zero-quantity deletion,
depth trimming and crossed-book checks. `chronoslob.book.reconstruction` applies
snapshot-plus-diff replay in supplied order, skips stale events, detects update-id
gaps and records crossed-book issues. `chronoslob.book.replay` loads local files
only and returns in-memory reconstruction results without writing outputs. The
CLI exposes `inspect-binance-replay` for local fixtures, and
`configs/data/binance_replay.yaml` plus `reports/order_book_reconstruction.md`
document the offline-only scope. The bundled Binance-style fixtures are
synthetic; no live ingestion, REST/WebSocket clients, downloads, API keys, models,
backtests, PnL or fake benchmark results are introduced. Phase 8 was marked
complete after `python -m pytest` passed with 525 tests.

## Phase 9: Event log storage and deterministic replay (completed)

Goal: Store event logs in an auditable format and replay them deterministically.

Files likely to be touched: `chronoslob/data/`, `chronoslob/book/`, `tests/`.

Acceptance criteria: Stored logs preserve ordering, metadata and source assumptions.

Tests expected: Round-trip storage tests, deterministic replay tests, ordering tests.

Status: `chronoslob.data.event_store` now defines the canonical local JSONL
event-log wrapper for `BookEvent` and `OrderBookSnapshot` records, including
schema versioning, deterministic serialisation, streaming reads, filtering and
explicit sorting. `chronoslob.data.manifests` builds SHA-256 reproducibility
manifests with record counts, symbols, timestamp ranges and sequence-id ranges.
`chronoslob.book.event_replay` extracts explicit snapshots from event logs,
builds past-only feature frames, optionally builds future-horizon label frames,
runs available no-look-ahead checks and writes Phase 8 Binance reconstruction
snapshots as canonical event logs. The CLI exposes read-only
`inspect-event-log` and `event-log-to-features` commands. Documentation lives in
`reports/event_log_storage.md` and `reports/replay_to_features.md`, with
configuration examples in `configs/data/event_log.yaml` and
`configs/experiments/event_log_feature_audit.yaml`. The bundled event-log
fixtures are synthetic. Generic `BookEvent` book reconstruction, tokenisation,
transformers, self-supervised training, execution backtests, PnL and fake results
are not introduced.

## Phase 10: Transformer tokenisation

Goal: Convert market events or book states into token sequences suitable for
transformer models.

Files likely to be touched: `chronoslob/features/`, `chronoslob/models/`,
`configs/models/`, `tests/`.

Acceptance criteria: Tokenisation is deterministic, documented and compatible with
temporal splits.

Tests expected: Token shape tests, vocabulary tests, no-future-context tests.

## Phase 11: Self-supervised transformer

Goal: Implement self-supervised pretraining objectives for market microstructure
representations.

Files likely to be touched: `chronoslob/models/`, `chronoslob/training/`,
`configs/models/`, `configs/experiments/`, `tests/`.

Acceptance criteria: Objectives are clearly defined, runs are reproducible and no
performance claims are made without artefacts.

Tests expected: Forward-pass tests, loss-shape tests, deterministic smoke tests.

## Phase 12: Multi-task fine-tuning

Goal: Fine-tune representations across multiple forecasting labels or horizons.

Files likely to be touched: `chronoslob/models/`, `chronoslob/training/`,
`chronoslob/labels/`, `configs/experiments/`, `tests/`.

Acceptance criteria: Tasks, horizons and label construction are explicit in configs.

Tests expected: Multi-head output tests, label alignment tests, config tests.

## Phase 13: Calibration and abstention

Goal: Add calibration diagnostics and abstention policies for forecast confidence.

Files likely to be touched: `chronoslob/analysis/`, `chronoslob/training/`,
`configs/experiments/`, `tests/`.

Acceptance criteria: Calibration is measured separately from raw accuracy and
abstention rules are transparent.

Tests expected: Calibration metric tests, threshold tests, partition-isolation tests.

## Phase 14: Execution-aware simulator

Goal: Add simplified execution-aware validation for forecast outputs.

Files likely to be touched: `chronoslob/backtest/`, `chronoslob/book/`,
`chronoslob/analysis/`, `configs/experiments/`, `tests/`.

Acceptance criteria: Simulation assumptions are explicit, costs and fills are
configurable and all outputs are labelled as research simulations.

Tests expected: Cost tests, fill-rule tests, no-look-ahead simulation tests.

## Phase 15: Analysis notebooks and result exports

Goal: Add notebooks and reports generated from reproducible experiment outputs.

Files likely to be touched: `notebooks/`, `reports/`, `chronoslob/analysis/`,
`tests/`.

Acceptance criteria: Notebooks do not contain core logic and all result tables trace
to experiment artefacts.

Tests expected: Export helper tests, report-input validation tests.

## Phase 16: Full audit, CI hardening and CV polish

Goal: Harden CI, audit leakage assumptions, polish documentation and prepare a
recruiter-facing project summary.

Files likely to be touched: `.github/`, `README.md`, `AGENTS.md`, `reports/`,
`tests/`, `pyproject.toml`.

Acceptance criteria: CI passes, limitations are current and CV language is accurate
to implemented work.

Tests expected: Full test suite, lint, typecheck and smoke commands.
