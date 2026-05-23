# Reproducibility

ChronosLOB is designed as a reproducible experiment artefact for market
microstructure research engineering. This page records the local validation path
used by contributors.

## Python And Installation

Use Python 3.11 or newer. The current CI target is Python 3.11.

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
uses torch-backed dataset, model and smoke tests. Install `.[dev,torch]` when
reproducing CI locally.

## Local Validation Commands

```bash
python -c "import chronoslob; print(chronoslob.__version__)"
python -m chronoslob.cli doctor
python -m chronoslob.cli run-project-audit
python -m chronoslob.cli run-project-audit --strict
python -m chronoslob.cli build-report-archive
python -m chronoslob.cli inspect-report-archive
python -m pytest
python -m compileall -q chronoslob tests
python -m ruff check .
python -m mypy chronoslob
```

`make` targets exist for convenience, but `make` may be unavailable on Windows
developer machines. The Python commands above are the canonical cross-platform
validation commands.

## Data Policy

No real exchange data, private data, licensed data, API keys or credentials are
committed to this repository. The committed data under `tests/fixtures/` is
synthetic and deliberately tiny. Users must provide any real FI-2010 or public
venue data locally, outside version control.

Configs may contain example paths such as `data/raw/...`; those are documented
placeholders and are not expected to exist in a fresh clone.

## Determinism

Smoke commands and tests use explicit seeds where randomness is involved. Torch
smoke paths run on CPU by default and use deterministic settings where practical.
Financial time-series splitting is temporal by default; random splits are not
used for core experiments.

## Smoke Commands

Useful local smoke commands include:

```bash
python -m chronoslob.cli inspect-fi2010 --path tests/fixtures/fi2010/tiny_fi2010_like.csv
python -m chronoslob.cli run-baseline-smoke --path tests/fixtures/fi2010/tiny_fi2010_like.csv
python -m chronoslob.cli run-deeplob-smoke --path tests/fixtures/fi2010/tiny_fi2010_like.csv --lookback 2 --epochs 1
python -m chronoslob.cli inspect-event-log --path tests/fixtures/event_logs/synthetic_snapshots.jsonl
python -m chronoslob.cli run-transformer-smoke --path tests/fixtures/event_logs/synthetic_snapshots.jsonl
python -m chronoslob.cli run-ssl-smoke --path tests/fixtures/event_logs/synthetic_snapshots.jsonl
python -m chronoslob.cli run-multitask-smoke --path tests/fixtures/event_logs/synthetic_snapshots.jsonl
python -m chronoslob.cli run-calibration-smoke
python -m chronoslob.cli run-execution-validation-smoke
python -m chronoslob.cli run-robustness-analysis-smoke
```

Synthetic smoke outputs show that plumbing executes, shapes align and summaries
are produced. They are not benchmark results, market evidence, execution
evidence or proof of cost-adjusted signal quality.

## Report Evidence Archive

The report archive can be rebuilt locally with:

```bash
python -m chronoslob.cli build-report-archive
python -m chronoslob.cli inspect-report-archive
```

The default archive build captures lightweight inspect and audit outputs only.
Use `--include-smoke-training` only when synthetic smoke-training outputs are
explicitly needed for documentation. The archive supports manual report writing;
it is not the final report.

## Reporting Rule

Do not create fake benchmark tables, fake plots or manually invented results.
Any future reported metric must trace to a config, code version, data version,
seed and stored experiment output.
