# Safety And Limitations

ChronosLOB is research infrastructure for market microstructure modelling. It is
not financial advice, not a live trading system and not a deployable execution
platform.

## Claim Boundaries

- No deployable alpha claim is made.
- No profitability, investment usefulness or live execution claim is made.
- No real benchmark result is claimed unless it is later generated from a
  reproducible experiment artefact.
- Synthetic smoke outputs are plumbing checks only.
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
simulation. It can account for configured fees, spread costs, latency,
turnover, risk constraints, passive fill proxies and adverse-selection labels,
but it is not a production execution simulator.

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
