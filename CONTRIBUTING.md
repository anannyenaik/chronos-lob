# Contributing

ChronosLOB is maintained as a research-engineering project. Contributions should
keep the codebase reproducible, leakage-aware and honest about its limitations.

## Local Setup

Use Python 3.11 or newer:

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

## Validation Commands

Run these before submitting changes:

```bash
python -m chronoslob.cli doctor
python -m chronoslob.cli inspect-release-readiness
python -m chronoslob.cli run-project-audit --strict
python -m pytest
python -m compileall -q chronoslob tests
python -m ruff check .
python -m mypy chronoslob
```

## Coding Standards

- Prefer explicit names and small, reviewable modules.
- Use type hints where practical.
- Use `pathlib.Path` for filesystem paths.
- Avoid runtime side effects at import time.
- Avoid hidden network calls.
- Keep dependency additions conservative and justified.
- Keep core logic in the package rather than notebooks.

## Data Policy

- Do not commit real exchange data, licensed data, private data, credentials,
  API keys or personal paths.
- Keep committed fixtures synthetic, small and clearly labelled.
- Document any future data assumptions, preprocessing steps and provenance.
- Fit transforms only on the training partition unless a test documents another
  leakage-safe design.

## Test Expectations

- Add tests for new modules and bug fixes.
- Prefer deterministic tests over stochastic checks.
- Add leakage tests for feature, label, split or dataset logic.
- Do not use random splits for financial time-series experiments unless the
  fixture explicitly requires it.
- Do not hide data-pipeline failures.

## Claims And Results

- Do not add fake results, invented metrics, manually fabricated plots or
  placeholder performance tables.
- Do not present synthetic smoke outputs as market evidence.
- Do not claim investment usefulness, production readiness or live execution
  capability.
- Any future result claim must trace to a reproducible config, data source,
  seed, code version and stored output artefact.

## Pull Request Checklist

- The validation commands above pass locally.
- New behaviour has focused tests.
- Documentation distinguishes implemented functionality from future work.
- Synthetic outputs and limitations remain clearly labelled.
- No secrets, private data, large generated files or notebook outputs are added.
- Forecast quality, calibration quality and execution-aware validation remain
  reported as separate evidence types.
