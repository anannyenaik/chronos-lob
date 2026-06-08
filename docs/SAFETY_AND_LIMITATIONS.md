# Safety and Limitations

ChronosLOB is research software for limit order book modelling. It is
not financial advice, not live trading infrastructure and not an
execution system for live markets. This document is the single, canonical
statement of scope. Other documents stay focused on technical content
and refer here for boundaries.

## Research Scope

The platform tests whether market-microstructure forecasts remain meaningful
under leakage-safe validation, calibration checks, feature-stability analysis,
event-level replay and execution-aware proxy diagnostics. Forecast quality,
calibration quality and signal-quality proxies are reported as separate
evidence streams.

## Data Limitations

- The repository ships no real exchange data, no licensed data and no
  credentials. Users provide any real FI-2010 or public venue data
  locally.
- Public limit order book datasets may have restricted coverage,
  simplified message semantics, survivorship effects, preprocessing
  choices or unclear timestamp conventions. Any experiment should
  document its exact data source and preprocessing.
- Fixtures under `tests/fixtures/` are small synthetic files. They
  exercise code paths and are not market evidence.
- Crypto-style reconstruction examples should not be treated as
  evidence for equity-market behaviour.
- Binance Spot L2 replay uses aggregated diff-depth updates. These updates do
  not expose individual orders, true trades, true cancellations, queue position
  or market impact; fixture runs are engineering checks rather than exchange
  data evidence.

## Modelling Limitations

- Benchmark metrics are run-specific and depend on the recorded dataset,
  split, config, seed and local environment.
- Implemented self-supervised objectives are evaluated only within their stored
  scopes. SSL-v1 does not support broad predictive or calibration improvement.
  SSL-v2 supports scoped predictive and calibration improvement only for its
  exact retained scope; broad SSL improvement remains unsupported.
- Forecast accuracy, calibration error and confidence-filtering
  diagnostics do not in themselves characterise tradability.

The SSL-v2 benchmark is complete for the stored FI-2010 scope: folds 1–5,
horizons 10/50, seeds 0–2 and lookback 50. Across 30 matched comparison cells,
SSL-v2 has positive mean deltas for macro-F1, MCC, ECE and Brier, supporting
scoped predictive and calibration improvement for this exact retained scope.
The evidence is mixed by seed and horizon, including negative mean macro-F1
deltas for seed 1 and horizon 50, so broad SSL improvement remains unsupported.

The one-epoch neural full grid is matched comparison evidence, not a
performance-maximising neural benchmark. The broader proper-training benchmark
completed its exact 180-cell supervised scope, but results are mixed by model,
lookback and horizon. The matrix-transformer lookback-100 rows are weak, and no
broad neural superiority is claimed.

## Execution-Validation Limitations

The execution-aware validation layer provides deterministic offline proxy
diagnostics. It supports configured fees, spread costs, row-step latency,
turnover, passive fill proxies, adverse-selection labels and simple risk
constraints.

It does not report realised returns or model live trading, broker or exchange
integration, venue-specific queue priority, live partial fills, queue dynamics,
market impact or portfolio optimisation.
Any future execution-aware result must state these assumptions.
See [EXECUTION_PROXY_VALIDITY.md](EXECUTION_PROXY_VALIDITY.md) for the dedicated
validity statement.

## Compute Provenance Boundary

The seed-1 and seed-2 SSL-v2 refresh was executed as independent Slurm array
jobs on Durham University Hamilton/NCC HPC. Retained summaries, provenance and
claim assessments are committed; large checkpoints, raw predictions and cluster
logs are intentionally excluded. GPU determinism warnings are documented, and
bitwise reproducibility is not claimed.

The broader proper-training neural benchmark was also executed as staged Slurm
jobs on Durham University Hamilton/NCC HPC. Retained storage-light summaries,
provenance and claim assessment are committed; large checkpoints, raw
predictions and cluster logs are excluded. GPU bitwise reproducibility is not
claimed.

This benchmark is post-`v0.2.0` work on `main`. `v0.2.0` remains the published
release and does not include it. No later release is implied by the retained
evidence.

## Reporting Discipline

Any reported metric must trace to a versioned config, data source,
seed, code commit and stored output artefact. Predictive, calibration
and execution-aware validation outputs are reported as separate
evidence types rather than collapsed into a single score.
Older generating commits do not by themselves invalidate retained summaries;
hash mismatches, changed retained content or newer required inputs do.
