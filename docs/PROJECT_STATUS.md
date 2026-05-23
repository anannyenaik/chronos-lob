# Project Status

Current package version: `0.1.0`.

ChronosLOB is a research platform for leakage-safe limit order book
representation learning, short-horizon market-state forecasting, calibration and
execution-aware validation. It is not live trading infrastructure and it does
not claim investment usefulness or production execution capability.

## Implemented

- Local loaders, schemas, validators and synthetic fixtures for reproducible
  engineering tests.
- Past-only feature generation and explicit future-horizon labels.
- Leakage checks for feature and label alignment.
- Temporal validation helpers, purged or embargoed splitters and train-only
  transforms.
- Classical, DeepLOB-style and transformer model plumbing.
- Self-supervised and multi-task training infrastructure.
- Calibration and confidence-filtering diagnostics.
- Simplified execution-aware validation utilities.
- Robustness analysis summaries for supplied experiment records.
- CLI smoke commands, configs, reports and tests for the implemented modules.
- Local audit, release-readiness and CI hardening utilities.
- Technical evidence archive generation for later manual report writing.

## Not Implemented

- Real benchmark result generation or committed benchmark artefacts.
- Live market data ingestion, broker integration or order placement.
- Production queue-position, partial-fill or market impact models.
- Portfolio optimisation or deployable execution logic.
- Dashboards, notebook outputs or final technical report prose.
- Real result tables, benchmark plots or report-ready findings.

## Current Limitations

The repository is a research infrastructure artefact. Smoke outputs and report
archive CLI captures are synthetic plumbing checks only where labelled.
FI-2010 data must be supplied locally by the user. Public crypto-style fixtures
are not evidence for equity-market behaviour. Simplified execution validation
does not model all venue mechanics, queue dynamics, latency, partial fills or
market impact.

## Next Work

The next substantive work is manual experiment reporting from reproducible
outputs. That work should use documented data provenance, temporal splits,
train-only transforms, seeds, code versions and output paths. No performance
claim should be added without a corresponding reproducible artefact.
