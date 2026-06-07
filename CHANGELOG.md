# Changelog

All notable public-release changes are documented here.

## [0.2.0] - 2026-06-07

### Changed

- Aligned the public documentation around the forecasting-versus-signal-quality
  gap and explicit claim boundaries.
- Reorganised the generated final empirical report around evidence, limitations,
  reproducibility and deferred work.
- Clarified retained artefact statuses and scoped SSL-v2 interpretation.
- Added the completed SSL-v2 seed-1 and seed-2 refresh run with Slurm on Durham
  University Hamilton/NCC HPC.
- Updated package maturity to alpha.

### Evidence Boundaries

- SSL-v2 mean predictive and calibration improvements are limited to the exact
  stored folds 1-5, horizons 10/50, seeds 0-2, lookback-50 scope. Results are
  mixed by seed and horizon; broad SSL improvement remains unsupported.
- Execution outputs remain offline proxy diagnostics.
- Synthetic and Binance L2 replay evidence remain separately scoped.

## [0.1.0] - 2026-05-29

- Initial public research-platform release.
