# Safety and Limitations

ChronosLOB is research software for limit order book modelling. It is
not financial advice, not live trading infrastructure and not a
production execution system. This document is the single, canonical
statement of scope. Other documents stay focused on technical content
and refer here for boundaries.

## Research Scope

The platform studies whether self-supervised representations of order
book dynamics can improve short-horizon market-state forecasting, and
whether those forecasts remain useful under explicit execution
assumptions. Forecast quality, calibration quality and cost-aware
signal quality are reported as separate evidence streams.

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

## Modelling Limitations

- Implemented baselines, transformer encoders and self-supervised
  objectives are infrastructure. No benchmark performance is claimed.
- Forecast accuracy, calibration error and confidence-filtering
  diagnostics do not in themselves characterise tradability.

## Execution-Validation Limitations

The execution-aware validation layer is a deterministic research
simulation. It supports configured fees, spread costs, row-step
latency, turnover, passive fill proxies, adverse-selection labels and
simple risk constraints.

It does not model live trading, broker or exchange integration,
venue-specific queue priority, production-grade partial fills,
production queue dynamics, market impact or portfolio optimisation.
Any future execution-aware result must state these assumptions.

## Reporting Discipline

Any reported metric must trace to a versioned config, data source,
seed, code commit and stored output artefact. Predictive, calibration
and execution-aware validation outputs are reported as separate
evidence types rather than collapsed into a single score.
