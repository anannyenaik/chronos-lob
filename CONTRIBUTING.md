# Contributing

ChronosLOB is a research-engineering project. Contributions should keep
the codebase reproducible, leakage-aware and honest about its scope.

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

## Coding Style

- Prefer small, reviewable modules with explicit names.
- Use type hints throughout and `pathlib.Path` for filesystem paths.
- Avoid import-time side effects and hidden network calls.
- Keep dependencies conservative; justify any new addition.
- Keep core logic in the package, not in notebooks.

## Tests

- Add tests for new modules and bug fixes.
- Prefer deterministic checks; seed any randomness explicitly.
- Add leakage tests for feature, label, split or dataset logic.
- Do not use random splits for financial time-series experiments.

## Data Policy

- Do not commit real exchange data, licensed data, credentials, API
  keys or personal paths.
- Keep committed fixtures synthetic, small and clearly labelled.
- Document data provenance, preprocessing and any leakage-safe design
  choices.
- Fit transforms on the training partition only.

## Documentation

- Keep public documentation concise and technical.
- Distinguish implemented functionality from planned work.
- Any reported metric must trace to a config, data source, seed, code
  version and stored output artefact.

## Pull Request Checklist

- The validation commands above pass locally.
- New behaviour has focused tests.
- Documentation reflects the new scope.
- No secrets, real venue data, large generated files or notebook
  outputs are added.
- Predictive, calibration and execution-aware validation evidence
  remain reported as separate streams.
