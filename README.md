# ChronosLOB

**A research-engineering platform for leakage-safe limit order book representation learning, calibrated forecasting and execution-aware validation.**

[![CI](https://github.com/anannyenaik/chronos-lob/actions/workflows/ci.yml/badge.svg)](https://github.com/anannyenaik/chronos-lob/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

ChronosLOB provides a reproducible framework for studying market microstructure as noisy sequential data: data validation, feature and label construction, temporal splits, classical and neural baselines, transformer-based representation learning, uncertainty analysis, execution-aware validation and robustness analysis.

## Why ChronosLOB

Most public order-book research collapses distinct questions into one number:
is the forecast accurate, is the forecast trustworthy, and does the forecast
remain useful after explicit execution assumptions? ChronosLOB keeps these as
separate evidence streams.

The platform is built around four commitments:

- Leakage-safe feature construction, labels, splitters and train-only fitting.
- Calibration and uncertainty analysis as first-class evaluation concerns.
- Explicit cost, latency, turnover, fill and risk assumptions.
- Reproducible experiment records with configs, seeds, code versions and
  stored outputs.

Forecast accuracy is never treated as proof of tradability. Synthetic fixture
outputs are plumbing checks only and are not benchmark results, market evidence
or execution evidence.

## Architecture

ChronosLOB is organised as a layered research stack. Each layer has clear
contracts, tests and CLI entry points so components can be inspected or replaced
without breaking leakage boundaries.

```text
audit and reproducibility
robustness analysis
execution-aware validation
calibration and uncertainty
representation learning
baselines and sequence models
temporal validation
feature and label engineering
data contracts and validation
```

| Layer | Responsibility | Key modules |
| --- | --- | --- |
| Data contracts | FI-2010-style loading, event records, schemas, fixtures. | [data](chronoslob/data/), [book](chronoslob/book/) |
| Features | Past-only microstructure features and regime indicators. | [features](chronoslob/features/) |
| Labels | Future-window labels and leakage guards. | [labels](chronoslob/labels/) |
| Temporal validation | Temporal, walk-forward, purged and embargoed splitters. | [splitters](chronoslob/training/splitters.py) |
| Baselines | Classical baselines, DeepLOB-style plumbing and datasets. | [models](chronoslob/models/) |
| Representation learning | Tokenisation, transformer encoder, SSL and multi-task heads. | [training](chronoslob/training/) |
| Calibration | Temperature scaling, calibration error and abstention. | [calibration](chronoslob/models/calibration.py) |
| Execution validation | Fees, spread costs, latency, turnover and risk constraints. | [backtest](chronoslob/backtest/) |
| Robustness analysis | Transfer, regime, ablation and sensitivity summaries. | [analysis](chronoslob/analysis/) |
| Audit | Release-readiness checks and technical evidence archive. | [utils](chronoslob/utils/) |

## Highlights

- Predictive quality, calibration quality and cost-aware signal quality are
  reported separately.
- Past-only features, future-window labels and train-only preprocessing are
  covered by dedicated leakage tests.
- Canonical event logs support deterministic tokenisation and transformer
  inputs.
- Self-supervised objectives and multi-task fine-tuning are implemented as
  tested infrastructure.
- Execution-aware validation keeps fees, spread, latency, turnover and risk
  assumptions explicit.
- Local audit tooling checks public release wording, required files, synthetic
  labelling and unsupported claims.

## Installation

ChronosLOB targets Python 3.11 or newer.

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

The `torch` extra is optional for installation, but it is required for the full
test suite and neural model smoke paths.

## Quick Start

Run the core local validation path:

```bash
python -m chronoslob.cli doctor
python -m chronoslob.cli inspect-release-readiness
python -m chronoslob.cli run-project-audit --strict
python -m pytest
python -m compileall -q chronoslob tests
python -m ruff check .
python -m mypy chronoslob
```

Inspect local fixtures and model plumbing:

```bash
python -c "import chronoslob; print(chronoslob.__version__)"
python -m chronoslob.cli inspect-fi2010 --path tests/fixtures/fi2010/tiny_fi2010_like.csv
python -m chronoslob.cli inspect-event-log --path tests/fixtures/event_logs/synthetic_snapshots.jsonl
python -m chronoslob.cli inspect-transformer
python -m chronoslob.cli inspect-calibration
python -m chronoslob.cli inspect-execution-validation
```

Run synthetic smoke paths:

```bash
python -m chronoslob.cli run-baseline-smoke --path tests/fixtures/fi2010/tiny_fi2010_like.csv
python -m chronoslob.cli run-transformer-smoke --path tests/fixtures/event_logs/synthetic_snapshots.jsonl
python -m chronoslob.cli run-ssl-smoke --path tests/fixtures/event_logs/synthetic_snapshots.jsonl
python -m chronoslob.cli run-multitask-smoke --path tests/fixtures/event_logs/synthetic_snapshots.jsonl
python -m chronoslob.cli run-calibration-smoke
python -m chronoslob.cli run-execution-validation-smoke
python -m chronoslob.cli run-robustness-analysis-smoke
```

The full command inventory lives in the [CLI reference](docs/CLI_REFERENCE.md).

## Repository Layout

```text
chronoslob/
  data/       FI-2010 loader, event store, schemas and validators
  book/       Local order book, replay and reconstruction
  features/   Microprice, imbalance, order flow, volatility and regimes
  labels/     Mid-price, spread, volatility, fill and leakage checks
  models/     Baselines, DeepLOB, transformer, SSL, multi-task, calibration
  training/   Splitters, datasets, dataloaders, experiments and metrics
  backtest/   Costs, execution, latency, turnover, risk and validation
  analysis/   Transfer, regimes, ablations, sensitivity and summaries
  utils/      Seeding, paths, logging, audit and archive utilities

configs/      YAML configs for data, models and experiments
docs/         CLI, reproducibility, status, evidence and safety docs
reports/      Per-component technical reports and evidence archive
tests/        Deterministic tests and synthetic fixtures
```

## Data Policy

No real exchange data, licensed data, private data, API keys or credentials are
committed. Fixtures under [tests/fixtures](tests/fixtures/) are tiny synthetic
files used only to exercise code paths. Users supply any real FI-2010 or public
venue data locally, outside version control.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the data provenance
and validation contract.

## Documentation

| Document | Purpose |
| --- | --- |
| [CLI reference](docs/CLI_REFERENCE.md) | Commands and options. |
| [Reproducibility](docs/REPRODUCIBILITY.md) | Validation, data provenance and evidence archives. |
| [Project status](docs/PROJECT_STATUS.md) | Implemented scope and future work. |
| [Safety and limitations](docs/SAFETY_AND_LIMITATIONS.md) | Scope boundaries and modelling caveats. |
| [Experiment evidence index](docs/EXPERIMENT_EVIDENCE_INDEX.md) | Experiment artefact registry. |
| [Roadmap](ROADMAP.md) | Completed work, planned work and out-of-scope items. |
| [Contributing](CONTRIBUTING.md) | Development workflow and contribution standards. |

Per-component reports live under [reports](reports/).

## Engineering Standards

| Concern | Enforced by |
| --- | --- |
| Style | `ruff` |
| Types | `mypy` |
| Tests | `pytest` |
| Determinism | central seeding utilities and manifests |
| Leakage controls | no-look-ahead and train-only fitting tests |
| Release hygiene | `inspect-release-readiness` and `run-project-audit --strict` |

## Roadmap

ChronosLOB's infrastructure layer is implemented and tested. The active
research workstream is empirical: running documented experiments on locally
provided datasets with provenance-tracked data, temporal splits, seeds and
stored outputs. Predictive, calibration and execution-aware streams should be
reported as separate evidence, not as a single headline number.

See [ROADMAP.md](ROADMAP.md) for full detail.

## License

Released under the [MIT License](LICENSE).

## Citation

If ChronosLOB supports your research, please cite the repository:

```bibtex
@software{chronoslob,
  title  = {ChronosLOB: Leakage-safe representation learning and
            execution-aware validation for limit order books},
  author = {{ChronosLOB contributors}},
  year   = {2026},
  url    = {https://github.com/anannyenaik/chronos-lob}
}
```
