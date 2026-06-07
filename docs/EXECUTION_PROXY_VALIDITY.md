# Execution Proxy Validity

## Purpose

ChronosLOB includes execution-aware proxy diagnostics because forecasting
metrics alone are insufficient to characterise signal quality. A forecast can
score well on macro-F1, MCC or calibration while becoming less useful after
confidence filtering, frequent signal changes, simple cost assumptions or
latency. The proxy layer tests that divergence without presenting an offline
diagnostic as a trading result.

## What The Proxy Can Support

The retained diagnostics can support descriptive comparisons of:

- sensitivity to confidence thresholds;
- active fraction;
- signal-change turnover proxy;
- configured cost sensitivity;
- row-step latency sensitivity;
- adverse-selection proxy;
- forecast-quality metrics versus signal-quality diagnostics.

These outputs show how stored predictions respond to explicit assumptions. They
are conservative signal-quality diagnostics rather than realised execution
outcomes.

## What The Proxy Cannot Support

The execution-aware proxy diagnostics cannot support claims about:

- PnL or realised profitability;
- tradability;
- live execution quality;
- production execution simulation;
- venue-specific queue priority;
- market impact;
- true fill modelling.

Cost-adjusted proxy values are not realised returns. Passive-fill assumptions
are documented modelling choices, not observed fills.

## Why FI-2010 Limits Execution Realism

FI-2010 contains normalised order-book snapshots rather than a complete
event-level exchange feed. It does not provide:

- true order identifiers;
- individual-order queue position;
- venue-level matching-engine state;
- real order submissions from a strategy;
- market-impact feedback;
- a true fill, cancel and trade event stream.

These omissions prevent reconstruction of queue priority, order fate and
venue-specific execution outcomes. Snapshot-derived order-flow fields therefore
remain labelled proxies, not true event-level order-flow imbalance.

## Data Needed For Stronger Execution Evidence

A stronger event-level execution study would require:

- order-add, cancel, modify and trade events;
- order IDs or reconstructable queue state;
- precise timestamps;
- trade-aggressor information where available;
- the applicable fee schedule;
- a documented latency model;
- venue-specific matching rules;
- explicit market-impact assumptions or a simulation framework.

Even with those inputs, realised profitability and live execution quality would
remain separate empirical claims requiring separate evidence.

## Conclusion

The execution centrepiece is a conservative signal-quality diagnostic. It is
deliberately not a trading claim.
