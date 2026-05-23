# ChronosLOB

[![CI](https://github.com/anannyenaik/chronos-lob/actions/workflows/ci.yml/badge.svg)](https://github.com/anannyenaik/chronos-lob/actions/workflows/ci.yml)

**ChronosLOB: Self-Supervised Market Microstructure Modelling for
Execution-Aware Alpha Discovery**

ChronosLOB is a research-engineering platform for limit order book
representation learning, short-horizon market-state forecasting and
execution-aware validation. It is built to study the gap between forecast
quality and cost-adjusted signal quality under explicit leakage, calibration and
execution assumptions.

This is not a trading bot, live trading system, financial advice or a source of
deployable trading claims.

## Architecture Summary

ChronosLOB is organised as a reproducible experiment artefact:

- data and book layers for local FI-2010-style loading, offline Binance-style
  reconstruction, canonical event logs and replay;
- feature and label layers for leakage-safe microstructure features,
  future-window labels and no-look-ahead checks;
- split and experiment layers for temporal, walk-forward and purged or
  embargoed validation protocols;
- model layers for classical baselines, DeepLOB-style supervised plumbing,
  transformer encoders, self-supervised objectives and multi-task heads;
- calibration, execution-aware validation and robustness-analysis layers for
  evaluating forecast quality separately from simplified signal-quality
  assumptions;
- audit, CI and report evidence tooling for reproducible public review.

## Current Status

Implemented through Phase 19: report evidence archive and GitHub polish support.
The repository contains tested infrastructure, synthetic fixtures and smoke
commands. It does not contain real benchmark results, real venue data, fake
plots, notebook outputs or final technical report prose.

See `docs/PROJECT_STATUS.md` for implemented and not implemented scope.

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

## Quickstart

```bash
python -c "import chronoslob; print(chronoslob.__version__)"
python -m chronoslob.cli doctor
python -m chronoslob.cli run-project-audit --strict
python -m chronoslob.cli build-report-archive
python -m chronoslob.cli inspect-report-archive
```

Core lightweight smoke commands:

```bash
python -m chronoslob.cli inspect-fi2010 --path tests/fixtures/fi2010/tiny_fi2010_like.csv
python -m chronoslob.cli inspect-features-fi2010 --path tests/fixtures/fi2010/tiny_fi2010_like.csv
python -m chronoslob.cli inspect-labels-fi2010 --path tests/fixtures/fi2010/tiny_fi2010_like.csv
python -m chronoslob.cli inspect-event-log --path tests/fixtures/event_logs/synthetic_snapshots.jsonl
python -m chronoslob.cli inspect-event-tokens --path tests/fixtures/event_logs/synthetic_snapshots.jsonl
python -m chronoslob.cli inspect-transformer
python -m chronoslob.cli inspect-ssl
python -m chronoslob.cli inspect-multitask
python -m chronoslob.cli inspect-calibration
python -m chronoslob.cli inspect-execution-validation
python -m chronoslob.cli inspect-analysis
```

Synthetic fixture outputs are plumbing checks only. They are not benchmark
results, market evidence, execution evidence or proof of signal quality.

## Validation

```bash
python -m chronoslob.cli doctor
python -m chronoslob.cli run-project-audit --strict
python -m pytest
python -m compileall -q chronoslob tests
python -m ruff check .
python -m mypy chronoslob
```

## Data Policy

No real exchange data, private data, licensed data, API keys or credentials are
committed. Files under `tests/fixtures/` are synthetic and intentionally small.
Users must provide real FI-2010 or public venue data locally, outside version
control, before generating result artefacts.

## Documentation

- `docs/REPRODUCIBILITY.md`: installation, validation and deterministic smoke
  guidance.
- `docs/CLI_REFERENCE.md`: CLI command inventory.
- `docs/PROJECT_STATUS.md`: implemented and not implemented functionality.
- `docs/SAFETY_AND_LIMITATIONS.md`: safety boundaries and modelling caveats.
- `docs/REPORT_EVIDENCE_INDEX.md`: map from final-report sections to repo
  evidence.
- `docs/REPORT_WRITING_GUIDE.md`: practical guide for writing the final report
  manually.
- `docs/GITHUB_POLISH_CHECKLIST.md`: public-facing release checklist.
- `reports/report_archive/`: generated evidence archive for report writing.
- `reports/limitations.md`: current limitations statement.

The build plan is tracked in `PLANS.md`.
