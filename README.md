# ChronosLOB

**ChronosLOB is a research platform for limit order book representation
learning, market-state forecasting, calibration and execution-aware
validation.**

It is research software, not financial advice and not live trading
infrastructure.

## Result Snapshot

| Field | Current evidence |
| --- | --- |
| Dataset | FI-2010 NoAuction Z-score, folds 1-5 |
| Split protocol | Official train/test split, validation carved from train |
| Best classical | `gradient_boosting`, test macro-F1 `0.4654 ± 0.0039` |
| Reduced-scope neural | `matrix_transformer`, test macro-F1 `0.7337 ± 0.0280` |
| Neural caveat | Single-seed, lookback 20, reduced-scope supervised neural evidence |
| Execution | Proxy diagnostics only, not a backtest |
| SSL | No SSL result claimed |
| External comparison | Protocol context only, no ranking claim |

## Final Report Command

```bash
python -m chronoslob.cli build-final-empirical-report \
  --classical experiments/fi2010_multifold_classical \
  --neural experiments/fi2010_multifold_neural \
  --uncertainty experiments/fi2010_uncertainty \
  --ablations experiments/fi2010_brutal_ablations \
  --execution experiments/fi2010_execution_v2 \
  --external experiments/fi2010_external_context \
  --out reports/chronoslob_final_empirical_report.md \
  --overwrite
```

## Main Reproduction Path

Follow [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). After local
FI-2010 acquisition and conversion, the main path is
`prepare-fi2010-multifold` -> `run-fi2010-multifold-classical` ->
`run-fi2010-neural-benchmark` with the reduced-scope settings ->
`analyse-fi2010-uncertainty` -> `run-fi2010-brutal-ablations` ->
`run-fi2010-execution-v2` -> review `experiments/fi2010_external_context/` ->
`build-final-empirical-report`.

Raw and processed FI-2010 files stay in ignored local directories under
`data/raw/fi2010/` and `data/processed/fi2010/`.

## What This Proves

- Reproducible multi-fold FI-2010 evaluation.
- Leakage-safe split handling.
- Calibrated forecasting diagnostics.
- Execution-aware proxy stress testing.
- Uncertainty, ablations and traceable artefacts.

## What This Does Not Prove

- Live tradability.
- Profitability.
- Production execution quality.
- State-of-the-art ranking.
- SSL effectiveness.
- Generalisation to other markets without further tests.

## Evidence Map

| Evidence | Path |
| --- | --- |
| Final empirical report | [reports/chronoslob_final_empirical_report.md](reports/chronoslob_final_empirical_report.md) |
| Multi-fold classical results | [experiments/fi2010_multifold_classical/](experiments/fi2010_multifold_classical/) |
| Reduced-scope neural results | [experiments/fi2010_multifold_neural/](experiments/fi2010_multifold_neural/) |
| Statistical uncertainty | [experiments/fi2010_uncertainty/](experiments/fi2010_uncertainty/) |
| Brutal ablations | [experiments/fi2010_brutal_ablations/](experiments/fi2010_brutal_ablations/) |
| Execution-aware proxy diagnostics | [experiments/fi2010_execution_v2/](experiments/fi2010_execution_v2/) |
| External benchmark context artefacts | [experiments/fi2010_external_context/](experiments/fi2010_external_context/) |
| External benchmark context doc | [docs/FI2010_EXTERNAL_BENCHMARKS.md](docs/FI2010_EXTERNAL_BENCHMARKS.md) |
| Final report builder doc | [docs/FINAL_EMPIRICAL_REPORT.md](docs/FINAL_EMPIRICAL_REPORT.md) |

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

## Validation

```bash
python -m chronoslob.cli doctor
python -m chronoslob.cli inspect-release-readiness
python -m chronoslob.cli run-project-audit --strict
python -m pytest
python -m compileall -q chronoslob tests
python -m ruff check .
python -m mypy chronoslob
```

Fixture outputs validate code paths only. They are not FI-2010 benchmark
evidence.

## Repository Layout

```text
chronoslob/  data, features, labels, models, training, backtest and analysis
configs/     YAML configs for data, models and experiments
docs/        CLI, reproducibility, benchmark, evidence and safety docs
reports/     Technical reports and generated empirical artefact reports
experiments/ Stored FI-2010 evidence artefacts
tests/       Deterministic tests and tiny synthetic fixtures
```

## Documentation

| Document | Purpose |
| --- | --- |
| [CLI reference](docs/CLI_REFERENCE.md) | Commands and options. |
| [Reproducibility](docs/REPRODUCIBILITY.md) | Local validation and real-data reproduction flow. |
| [Final empirical report](docs/FINAL_EMPIRICAL_REPORT.md) | Report inputs, command and claim boundaries. |
| [Experiment evidence index](docs/EXPERIMENT_EVIDENCE_INDEX.md) | Map from claims to artefacts and tests. |
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
