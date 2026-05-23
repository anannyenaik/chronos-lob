# Safety And Limitations

ChronosLOB is research infrastructure for market microstructure modelling. It is
not financial advice, not live trading infrastructure and not a production
execution platform.

## Claim Boundaries

- No investment usefulness or live execution claim is made.
- No real benchmark result is claimed unless it is later generated from a
  reproducible experiment artefact.
- Synthetic smoke outputs are plumbing checks only.
- Report archive CLI captures are report-writing references only; synthetic
  captures are not market evidence.
- Prediction quality, uncertainty quality and execution-aware validation are
  reported separately.

## Data Caveats

FI-2010 is a useful public benchmark, but users must obtain it locally and verify
the exact mirror, preprocessing and label conventions they use. The repository
does not bundle FI-2010 data and does not claim published benchmark replication.

Crypto and Binance-style data can be useful for engineering demonstrations, but
crypto market microstructure should not be treated as directly equivalent to
equity-market behaviour. The bundled Binance-style files are synthetic fixtures,
not real venue data.

## Execution Caveats

Execution-aware validation in this repository is a simplified research
simulation. It can account for configured fees, spread costs, latency, turnover,
risk constraints, passive fill proxies and adverse-selection labels, but it is
not a production execution simulator.

The project does not implement:

- live trading;
- broker or exchange integration;
- production queue modelling;
- production partial-fill modelling;
- venue-specific matching rules;
- market impact modelling;
- portfolio optimisation.

All future reports should state these assumptions clearly and should avoid
treating forecast accuracy as cost-adjusted signal quality.

## Evidence Archive Caveat

The technical evidence archive organises repository evidence for manual report
writing. It does not create the final report, benchmark results, real data
outputs or production-readiness evidence. Any future result claim must cite
reproducible experiment inputs and outputs rather than the archive itself.
