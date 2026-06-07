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
- Added a dedicated execution-proxy validity statement.
- Updated package maturity to alpha.

### Evidence Boundaries

The SSL-v2 benchmark is complete for the stored FI-2010 scope: folds 1–5,
horizons 10/50, seeds 0–2 and lookback 50. Across 30 matched comparison cells,
SSL-v2 has positive mean deltas for macro-F1, MCC, ECE and Brier, supporting
scoped predictive and calibration improvement for this exact retained scope.
The evidence is mixed by seed and horizon, including negative mean macro-F1
deltas for seed 1 and horizon 50, so broad SSL improvement remains unsupported.

The seed-1 and seed-2 SSL-v2 refresh was executed as independent Slurm array
jobs on Durham University Hamilton/NCC HPC. Retained summaries, provenance and
claim assessments are committed; large checkpoints, raw predictions and cluster
logs are intentionally excluded. GPU determinism warnings are documented, and
bitwise reproducibility is not claimed.

The one-epoch neural full grid is matched comparison evidence, not a
performance-maximising neural benchmark. The proper-training neural subset
remains partial, and a broader proper-training neural benchmark across folds,
seeds, lookbacks and model families is deferred.

- Execution outputs remain offline proxy diagnostics.
- Synthetic and Binance L2 replay evidence remain separately scoped.

## [0.1.0] - 2026-05-29

- Initial public research-platform release.
