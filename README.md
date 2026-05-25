# ChronosLOB

**Research software for leakage-safe limit order book representation learning,
calibrated forecasting and execution-aware sensitivity analysis.**

[![CI](https://github.com/anannyenaik/chronos-lob/actions/workflows/ci.yml/badge.svg)](https://github.com/anannyenaik/chronos-lob/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](http://mypy-lang.org/)

ChronosLOB is a research platform for limit order book representation learning,
market-state forecasting, calibration and execution-aware validation. It turns
locally supplied limit order book benchmark files into auditable experiment
artefacts: configs, data manifests, split summaries, model metrics,
calibration tables, execution-aware sensitivity rows, ablations, plots and
local systems measurements.

ChronosLOB is research and engineering infrastructure. It is not financial
advice and it is not live trading infrastructure.

## Current Evidence

The repository now includes a real FI-2010 fold-1 evidence set built from the
official NoAuction ZScore train/test files after local verification and
conversion. The run uses official split-aware evaluation from the combined
CSV `split` column, with validation carved only from official train rows.

The committed FI-2010 fold-1 run includes majority, logistic, random forest,
gradient boosting, DeepLOB-style and normalised-matrix transformer baselines.
In the current artefacts, gradient boosting is the strongest model by macro-F1,
while the transformer path runs through the normalised FI-2010 matrix
representation. `ssl_transformer` is not supported by the paper runner and is
not reported as a model result.

Supported evidence streams:

- predictive metrics from `results.json`
- calibration evidence from `calibration_bins.csv`
- execution-aware sensitivity from `execution_sensitivity.csv`
- controlled ablations under `experiments/fi2010_midprice_h10_ablations/`
- local systems measurements under `experiments/fi2010_midprice_h10_systems/`

Primary references:

- [FI-2010 benchmark preparation](docs/FI2010_BENCHMARK.md)
- [Paper experiment runner](docs/PAPER_EXPERIMENTS.md)
- [Paper ablations](docs/PAPER_ABLATIONS.md)
- [Systems benchmarks](docs/SYSTEM_BENCHMARKS.md)
- [Empirical artefact report](reports/chronoslob_empirical_report.md)
- [FI-2010 model card](experiments/fi2010_midprice_h10/model_card.md)
- [Experiment evidence index](docs/EXPERIMENT_EVIDENCE_INDEX.md)

The numbers are run-specific benchmark artefacts. Forecast quality, calibration
quality and execution-aware sensitivity are reported separately and should not
be collapsed into a trading or deployment claim.

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

The `torch` extra is required for the full test suite and neural smoke paths.

## Quick Start

Run the local health and release checks:

```bash
python -m chronoslob.cli doctor
python -m chronoslob.cli inspect-release-readiness
python -m chronoslob.cli run-project-audit --strict
```

Exercise the tiny bundled fixture:

```bash
python -m chronoslob.cli inspect-fi2010 \
  --path tests/fixtures/fi2010/tiny_fi2010_like.csv

python -m chronoslob.cli run-paper-experiment \
  --config configs/experiments/fi2010_midprice_h10.yaml \
  --data-path tests/fixtures/fi2010/tiny_fi2010_like.csv \
  --out runs/paper_experiment_smoke \
  --models majority,logistic \
  --overwrite \
  --build-plots
```

Fixture outputs validate code paths only. They are not FI-2010 benchmark
evidence.

Run the full local validation suite:

```bash
python -m pytest
python -m compileall -q chronoslob tests
python -m ruff check .
python -m mypy chronoslob
```

The full command inventory lives in the [CLI reference](docs/CLI_REFERENCE.md).

## Real FI-2010 Reproduction

Raw FI-2010 data is not committed and is never downloaded automatically. To
reproduce the real evidence, download the official archive locally, keep it
under `data/raw/fi2010/`, convert the selected official `.txt` files into a
loader-ready CSV under `data/processed/fi2010/`, then run the paper experiment,
ablation suite, systems benchmarks and report builder.

The acquisition and conversion sequence is documented in
[FI2010_DATA_ACQUISITION.md](docs/FI2010_DATA_ACQUISITION.md), and the
end-to-end command flow is in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Repository Layout

```text
chronoslob/  data, features, labels, models, training, backtest and analysis
configs/     YAML configs for data, models and experiments
docs/        CLI, reproducibility, benchmark, evidence and safety docs
reports/     Technical reports and generated empirical artefact report
experiments/ Stored FI-2010 evidence artefacts
tests/       Deterministic tests and tiny synthetic fixtures
```

## Documentation

| Document | Purpose |
| --- | --- |
| [CLI reference](docs/CLI_REFERENCE.md) | Commands and options. |
| [Reproducibility](docs/REPRODUCIBILITY.md) | Local validation and real-data reproduction flow. |
| [Project status](docs/PROJECT_STATUS.md) | Implemented scope and current limitations. |
| [Safety and limitations](docs/SAFETY_AND_LIMITATIONS.md) | Canonical scope boundary. |
| [Roadmap](ROADMAP.md) | Completed milestone and future work. |
| [Contributing](CONTRIBUTING.md) | Development workflow and contribution standards. |

## Data Policy

No real exchange data, licensed data, private data, API keys or credentials are
committed. Tiny files under [tests/fixtures](tests/fixtures/) are synthetic and
exist only to exercise deterministic code paths.

## Licence

Released under the [MIT Licence](LICENSE).

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
