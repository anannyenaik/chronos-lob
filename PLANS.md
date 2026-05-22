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

## Phase 1: Core schemas and utilities

Goal: Define typed schemas for events, snapshots, trades, quotes, instruments and
time handling.

Files likely to be touched: `chronoslob/data/`, `chronoslob/book/`,
`chronoslob/utils/`, `tests/`.

Acceptance criteria: Schemas validate required fields, timestamp conventions are
documented, invalid records fail loudly.

Tests expected: Schema validation tests, timestamp ordering tests, serialisation
round-trip tests.

## Phase 2: FI-2010 benchmark loader

Goal: Add a loader for FI-2010 benchmark files without changing labels or splits
implicitly.

Files likely to be touched: `chronoslob/data/`, `configs/data/`, `tests/`.

Acceptance criteria: Loader reads local user-provided files, validates shapes and
documents assumptions.

Tests expected: Fixture-based loading tests, missing-file tests, malformed-input
tests.

## Phase 3: Microstructure feature engine

Goal: Implement leakage-safe features using only information available at or before
timestamp `t`.

Files likely to be touched: `chronoslob/features/`, `chronoslob/book/`, `tests/`.

Acceptance criteria: Feature definitions are documented, windows are backward-looking
and transforms are explicit.

Tests expected: Feature-value tests on small fixtures, no-look-ahead tests, edge-case
window tests.

## Phase 4: Label generation and leakage tests

Goal: Implement future-looking labels while guaranteeing labels cannot enter feature
inputs.

Files likely to be touched: `chronoslob/labels/`, `chronoslob/features/`, `tests/`.

Acceptance criteria: Label horizons and alignment rules are configurable and
documented.

Tests expected: Horizon alignment tests, leakage guard tests, boundary-condition
tests.

## Phase 5: Temporal splitters and experiment registry

Goal: Add temporal train/validation/test splitters and a registry for reproducible
experiment artefacts.

Files likely to be touched: `chronoslob/training/`, `configs/experiments/`,
`chronoslob/utils/`, `tests/`.

Acceptance criteria: Random splits are not the default for financial data, split
metadata is persisted.

Tests expected: Split ordering tests, no-overlap tests, registry metadata tests.

## Phase 6: Classical baselines

Goal: Implement simple, reproducible forecasting baselines before deep learning.

Files likely to be touched: `chronoslob/models/`, `chronoslob/training/`,
`configs/models/`, `tests/`.

Acceptance criteria: Baselines train from configs and report only reproducible
metrics.

Tests expected: Fit/predict tests, deterministic baseline tests, config validation
tests.

## Phase 7: PyTorch datasets and DeepLOB-style baseline

Goal: Add PyTorch datasets and a DeepLOB-style supervised baseline.

Files likely to be touched: `pyproject.toml`, `chronoslob/models/`,
`chronoslob/training/`, `configs/models/`, `tests/`.

Acceptance criteria: PyTorch dependency is justified, dataset indexing is
leakage-safe and baseline training is configurable.

Tests expected: Dataset slicing tests, shape tests, deterministic small-training
smoke tests.

## Phase 8: Binance local order book reconstruction

Goal: Reconstruct local order book state from public exchange messages for
engineering demonstrations.

Files likely to be touched: `chronoslob/data/`, `chronoslob/book/`, `configs/data/`,
`tests/`.

Acceptance criteria: Reconstruction uses local files or explicit user-configured
sources, sequence gaps fail loudly.

Tests expected: Replay tests, gap-detection tests, snapshot/delta consistency tests.

## Phase 9: Event log storage and deterministic replay

Goal: Store event logs in an auditable format and replay them deterministically.

Files likely to be touched: `chronoslob/data/`, `chronoslob/book/`, `tests/`.

Acceptance criteria: Stored logs preserve ordering, metadata and source assumptions.

Tests expected: Round-trip storage tests, deterministic replay tests, ordering tests.

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
