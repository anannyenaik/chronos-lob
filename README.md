# ChronosLOB

**ChronosLOB: Self-Supervised Market Microstructure Modelling for
Execution-Aware Alpha Discovery**

ChronosLOB is a research-engineering platform for limit order book
representation learning, short-horizon market-state forecasting and
execution-aware validation. It is designed to study the gap between forecast
quality and cost-adjusted signal quality under explicit leakage, calibration and
execution assumptions.

This repository is for research and education. It is not financial advice, not a
live trading system and not a source of deployable trading claims.

## Scope

ChronosLOB separates prediction from tradability. Benchmark accuracy, calibration
quality and simplified execution-validation metrics are different questions and
must be reported separately.

The repository does not claim:

- real FI-2010 benchmark results;
- live exchange connectivity;
- broker integration;
- production execution logic;
- profitability or investment usefulness.

## Current Status

Implemented through Phase 17:

- schemas, event types and data-quality validation;
- local FI-2010-style loading and inspection;
- leakage-safe microstructure features;
- future-window labels and no-look-ahead checks;
- temporal, walk-forward and purged or embargoed splitters;
- experiment registry skeleton;
- classical baselines and train-only preprocessing;
- PyTorch sequence-window datasets;
- DeepLOB-style supervised CNN-LSTM baseline;
- offline Binance-style order book reconstruction;
- canonical JSONL event-log storage and replay integration;
- deterministic event tokenisation and transformer input preparation;
- supervised transformer encoder architecture;
- self-supervised objectives;
- multi-task fine-tuning;
- calibration and uncertainty analysis;
- execution-aware validation;
- transfer, regime, ablation and sensitivity analysis.

Phase 18 adds audit and CI hardening infrastructure. See
`docs/PROJECT_STATUS.md` for details.

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

The `torch` extra is optional for package installation, but the full test suite
uses torch-backed modules.

## Basic Commands

```bash
python -c "import chronoslob; print(chronoslob.__version__)"
python -m chronoslob.cli doctor
python -m chronoslob.cli run-project-audit
python -m chronoslob.cli inspect-fi2010 --path tests/fixtures/fi2010/tiny_fi2010_like.csv
python -m chronoslob.cli inspect-event-log --path tests/fixtures/event_logs/synthetic_snapshots.jsonl
python -m chronoslob.cli run-calibration-smoke
python -m chronoslob.cli run-execution-validation-smoke
python -m chronoslob.cli run-robustness-analysis-smoke
```

See `docs/CLI_REFERENCE.md` for the full command inventory.

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
committed. The files under `tests/fixtures/` are synthetic and intentionally
small. Users must provide real FI-2010 or public venue data locally, outside
version control.

Synthetic smoke outputs are plumbing checks only. They are not benchmark
results, market evidence, execution evidence or proof of signal quality.

## Documentation

- `docs/REPRODUCIBILITY.md`: installation, validation and deterministic smoke
  guidance.
- `docs/CLI_REFERENCE.md`: CLI command inventory.
- `docs/PROJECT_STATUS.md`: implemented and not implemented functionality.
- `docs/SAFETY_AND_LIMITATIONS.md`: safety boundaries and modelling caveats.
- `reports/limitations.md`: current limitations statement.

The build plan is tracked in `PLANS.md`.
