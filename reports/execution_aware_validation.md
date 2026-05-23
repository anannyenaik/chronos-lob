# Execution-Aware Validation

This report documents the simplified execution-aware validation layer for
ChronosLOB. The goal is to measure whether prediction-like signals retain
cost-adjusted signal quality after explicit execution assumptions are applied.
It is not live execution infrastructure, not a production backtest and not a
deployment claim.

## Why Forecasting Metrics Are Not Enough

Short-horizon market-state forecasting metrics measure statistical prediction
quality. They do not include spread costs, fees, latency, missed fills, turnover
or inventory limits. A forecast can be accurate or well calibrated while still
being unusable after execution constraints are applied. ChronosLOB therefore
treats prediction and tradability as separate research questions.

## Validation Goal

The validation layer consumes prediction-like signals and market-state-like
rows. It produces simulation metrics such as coverage, fill rate, adverse
selection rate, turnover, gross simulated PnL, total simulated cost and net
simulated PnL. These are execution-validation diagnostics only.

## Execution Modes

Aggressive taking crosses the spread at the best ask for buys or best bid for
sells. The model assumes an immediate fill after the configured latency if a
market state is still available. Costs include fixed fees, proportional fees and
either a half-spread or full-spread convention.

Passive posting places at the near touch: best bid for buys or best ask for
sells. Fill is controlled by a supplied fill-probability proxy. Queue position,
partial fills and venue priority are not modelled. Passive posting does not pay
spread crossing cost by default, but can include explicit adverse-selection or
slippage assumptions.

Hybrid mode takes aggressively only above a high confidence threshold. It posts
passively for moderate confidence only when the fill-probability proxy passes
the configured threshold. It abstains otherwise.

## Costs, Latency and Turnover

Spread costs and fees are decomposed so assumptions are auditable. The cost
model supports fixed fee per trade, proportional basis-point fees, aggressive
half-spread or full-spread conventions and passive adverse-selection cost terms.

Latency is represented in row or event steps. If the latency-adjusted state is
beyond the available market-state sequence, the signal is marked unexecutable.
A deterministic latency-sensitivity grid reports how coverage, fills and net
simulated PnL change as latency assumptions vary.

Turnover is tracked by absolute quantity, with notional turnover available for
risk checks. Position paths are deterministic and are used for inventory
constraints.

## Risk Constraints

The risk layer supports inventory caps, maximum trade count, maximum turnover
and an optional simulated drawdown cap. These are simple validation constraints,
not portfolio optimisation or production risk controls. Abstention reasons are
stored explicitly so blocked signals can be audited.

## Confidence Filtering

Confidence thresholds interact with execution validation through coverage. The
threshold sweep reports coverage, fill rate, simulated costs and net simulated
PnL across thresholds. Confidence filtering remains an uncertainty diagnostic;
it is not a trading strategy and does not prove tradability.

## Adverse Selection and Fill Assumptions

Passive fills can carry an adverse-selection label supplied by the market-state
row. The validation summary reports the adverse-selection rate among filled
orders. This is a proxy check only. It does not model queue depletion, hidden
liquidity, exchange-specific matching, partial fills or market impact.

## Why This Is Not a Real Backtest

The implementation is deterministic simulation infrastructure. It does not
download data, place orders, connect to brokers or exchanges, model production
latency, estimate market impact, reconstruct queue position or verify venue
rules. See [docs/SAFETY_AND_LIMITATIONS.md](../docs/SAFETY_AND_LIMITATIONS.md)
for the scope statement.

## Future Work

Future work can connect this layer to reproducible experiment artefacts,
realistic temporal evaluation runs, transfer and regime-shift analysis, ablation
studies, richer passive-fill labels and more careful venue-specific assumptions.
Any such work should continue to distinguish forecasting quality from
execution-aware cost-adjusted signal quality.
