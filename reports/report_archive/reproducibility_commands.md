# Reproducibility Commands

These Python commands are canonical because `make` may be unavailable on Windows. Run them from the repository root.

## Install

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,torch]"
```

## Core Checks

```bash
python -c "import chronoslob; print(chronoslob.__version__)"
python -m chronoslob.cli doctor
python -m chronoslob.cli run-project-audit --strict
python -m pytest
python -m compileall -q chronoslob tests
python -m ruff check .
python -m mypy chronoslob
```

## Build The Report Evidence Archive

```bash
python -m chronoslob.cli build-report-archive
python -m chronoslob.cli inspect-report-archive
```

## Lightweight CLI Smoke Commands

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

## Warning Caveats

- Synthetic fixture outputs are not market evidence.
- Torch and scikit-learn may emit upstream warnings in tests; record exact warnings rather than hiding them.
- Real benchmark reporting requires separately generated artefacts.
