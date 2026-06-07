# ChronosLOB v0.2.0 Alpha Release Notes

Date: 2026-06-07

## Headline

ChronosLOB v0.2.0 presents a claim-audited public evidence release centred on
one result: forecasting metrics and trading-signal diagnostics can diverge.

## What Changed

- Aligned the README, roadmap, project status, safety document, evidence pack
  and generated final report around one conservative public narrative.
- Reorganised the final empirical report into an evidence-led public report
  rather than an experiment-by-experiment dump.
- Clarified artefact completeness and freshness through `complete_real`,
  `partial_real`, `archived_valid`, `optional_missing`,
  `obsolete_superseded` and `unknown_staleness`.
- Added the SSL-v2 seed-1 and seed-2 refresh run with Slurm on Durham University
  Hamilton/NCC HPC.
- Updated release metadata from `0.1.0` to `0.2.0` alpha.

## Evidence Included

- Leakage-safe FI-2010 classical benchmark evidence.
- Matched supervised and SSL comparison evidence.
- Scoped mean SSL-v2 predictive and calibration improvement for the exact stored
  folds 1-5, horizons 10/50, seeds 0-2, lookback-50 slice.
- Calibration and confidence-filtering diagnostics.
- Scoped FI-2010 snapshot-feature stability analysis.
- Offline execution-aware proxy diagnostics and the execution centrepiece.
- Controlled synthetic event-level replay.
- Binance Spot aggregated L2 replay engineering evidence, with the committed
  sample explicitly fixture-backed.

## Limitations

- SSL-v2 results are mixed by seed and horizon; broad SSL improvement is
  unsupported.
- Execution evidence is offline proxy diagnostics only.
- FI-2010 does not expose true event-level order flow or queue position.
- Feature ablations are not causal feature importance.
- Synthetic results do not establish real-market generalisation.
- Binance replay does not establish equity-market or predictive success.

## Deferred Work

- A broader proper-training neural benchmark.
- Broader non-linear feature-stability coverage.
- The manual paper.

## Reproducibility

```powershell
python -m pytest -q
python -m ruff check .
python -m mypy chronoslob
python -m chronoslob.cli doctor
python -m chronoslob.cli inspect-release-readiness
python -m chronoslob.cli run-project-audit --strict
git diff --check
```
