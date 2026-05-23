# Public Release Readiness

This report records the public-release cleanup for ChronosLOB. It keeps the
repository focused on research engineering, reproducibility and honest
limitations.

## What Changed

- Public documentation was rewritten around leakage-safe limit order book
  representation learning, market-state forecasting, calibration and
  execution-aware validation.
- Internal maintainer-instruction material was replaced with normal human
  contribution guidance.
- The internal planning document was replaced with a professional roadmap.
- External-positioning language was removed from public documentation.
- The report archive was reframed as a technical evidence archive for later
  manual report writing.
- Local audit checks now include public wording, README structure and
  release-readiness checks.

## Why The Cleanup Matters

Public documentation should look like maintained research software, not an
internal process record. The repository should expose technical scope, commands,
tests, limitations and reproducibility boundaries without preserving internal
workflow history.

External-positioning language was removed because it can distract from the
technical artefact. ChronosLOB should be reviewed on its engineering quality,
auditability, leakage controls and documented limitations.

## Manual GitHub Metadata

Repository metadata is not changed by local files. The recommended GitHub
description is:

> Research platform for limit order book representation learning, calibration and execution-aware validation.

Repository topics should also be reviewed manually in the GitHub UI.

## Public-Release Checks

Run:

```bash
python -m chronoslob.cli inspect-release-readiness
python -m chronoslob.cli run-project-audit --strict
python -m chronoslob.cli build-report-archive
python -m chronoslob.cli inspect-report-archive
python -m pytest
python -m compileall -q chronoslob tests
python -m ruff check .
python -m mypy chronoslob
python -m chronoslob.cli doctor
```

## Still Not Claimed

ChronosLOB still does not claim real benchmark results, production execution
readiness, investment usefulness, live order placement, real venue performance,
portfolio optimisation or market impact realism. Synthetic fixture outputs
remain plumbing checks only.

## Technical Report Status

The final technical report is intentionally not included. It should be written
manually after reproducible experiment outputs exist and after all claims can be
traced to configs, data provenance, seeds, code versions and stored artefacts.
