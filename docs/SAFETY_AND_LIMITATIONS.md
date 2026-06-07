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
  scopes. The current SSL-v1 and SSL-v2 evidence does not support a broad SSL
  improvement or SSL calibration-improvement claim.
- Forecast accuracy, calibration error and confidence-filtering
  diagnostics do not in themselves characterise tradability.

## Execution-Validation Limitations

The execution-aware validation layer provides deterministic offline proxy
diagnostics. It supports configured fees, spread costs, row-step latency,
turnover, passive fill proxies, adverse-selection labels and simple risk
constraints.

It does not report realised returns or model live trading, broker or exchange
integration, venue-specific queue priority, live partial fills, queue dynamics,
market impact or portfolio optimisation.
Any future execution-aware result must state these assumptions.

## Reporting Discipline

Any reported metric must trace to a versioned config, data source,
seed, code commit and stored output artefact. Predictive, calibration
and execution-aware validation outputs are reported as separate
evidence types rather than collapsed into a single score.
Older generating commits do not by themselves invalidate retained summaries;
hash mismatches, changed retained content or newer required inputs do.
