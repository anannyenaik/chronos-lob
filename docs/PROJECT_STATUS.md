# Project Status

Current package version: `0.1.0`.

ChronosLOB is a research-engineering platform for leakage-safe limit order book
representation learning, short-horizon market-state forecasting and
execution-aware validation. It is not a live trading system and it does not
claim deployable alpha.

## Completed Through Phase 19

- Phase 0: repository scaffold, tooling, documentation conventions and project
  safety rules.
- Phase 1: typed market event, order book, feature, label and data-quality
  schemas.
- Phase 2: local FI-2010-style loader and validation helpers.
- Phase 3: leakage-safe microstructure feature engine.
- Phase 4: future-window label engine and no-look-ahead leakage checks.
- Phase 5: temporal, walk-forward and purged or embargoed splitters plus a
  metadata-only experiment registry skeleton.
- Phase 6: classical baseline interfaces, train-only preprocessing and metrics.
- Phase 7: PyTorch sequence-window data layer.
- Phase 8: DeepLOB-style supervised CNN-LSTM baseline and deterministic smoke
  training utilities.
- Phase 9: offline Binance-style local order book reconstruction.
- Phase 10: canonical JSONL event-log storage and deterministic replay into
  feature and label frames.
- Phase 11: deterministic event tokenisation and transformer input preparation.
- Phase 12: supervised transformer encoder architecture and smoke training path.
- Phase 13: self-supervised masked-field and next-field objectives.
- Phase 14: multi-task fine-tuning infrastructure.
- Phase 15: calibration, uncertainty and abstention diagnostics.
- Phase 16: deterministic execution-aware validation infrastructure with
  aggressive, passive and hybrid modes, explicit fees, spread costs, latency,
  turnover, risk constraints and adverse-selection tracking.
- Phase 17: transfer, regime, ablation and sensitivity analysis utilities with
  structured summaries and synthetic smoke tooling.
- Phase 18: local project audit utilities, `run-project-audit`, GitHub Actions
  CI, CLI reference documentation, reproducibility notes, project status
  documentation and safety/limitations documentation.
- Phase 19: report evidence archive utilities, `build-report-archive`,
  report-writing guide, evidence index, GitHub polish checklist, Mermaid
  diagrams and archive inventory files.

## Implemented

- Local loaders, schemas, validators and fixtures for reproducible engineering
  tests.
- Past-only feature generation and explicit future-horizon labels.
- Leakage checks for feature and label alignment.
- Temporal validation helpers and train-only transforms.
- Classical, DeepLOB-style and transformer model plumbing.
- Self-supervised and multi-task training infrastructure.
- Calibration and confidence filtering diagnostics.
- Simplified execution-aware validation utilities.
- Robustness analysis summaries for supplied experiment records.
- CLI smoke commands, configs, reports and tests for the implemented modules.
- Local audit and CI hardening utilities.
- Report evidence archive generation for manual final-report writing.

## Not Implemented

- Real benchmark result generation or committed benchmark artefacts.
- Live market data ingestion, broker integration or order placement.
- Production queue-position, partial-fill or market impact models.
- Portfolio optimisation or deployable execution logic.
- Dashboards, notebook outputs, final technical report or CV packaging.
- Real report results or final report prose.

## Current Limitations

The repository is a research infrastructure artefact. Smoke outputs and report
archive CLI captures are synthetic plumbing checks only where labelled.
FI-2010 data must be supplied locally by the user. Public crypto-style fixtures
are not evidence for equity-market behaviour. Simplified execution validation
does not model all venue mechanics, queue dynamics, latency, partial fills or
market impact.

## Next Phases

The next planned work is final public release readiness and CV packaging
support. Those phases should summarise implemented and verified artefacts only,
without adding fake results or unsupported trading claims.
