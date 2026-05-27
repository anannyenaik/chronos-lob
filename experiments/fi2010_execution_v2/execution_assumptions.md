# FI-2010 Execution-Aware Evaluation v2 - Assumptions

Every number produced by this layer is a simplified proxy diagnostic.
The assumptions below are deliberately explicit so the gap between a
forecasting metric and a tradability proxy is never read as a result.

## What this is not

- This is not a backtest.
- This is not a live-trading simulation.
- There is no market impact model.
- There is no queue-position ground truth.
- Fills are approximate or unavailable depending on the input artefacts.

## What the proxies assume

- Costs are scenario assumptions expressed in basis points, applied as a
  fixed per-trade deduction; they are not measured exchange fees.
- Latency is row-step latency (the realised forward return is read a fixed
  number of rows later); it is not exchange or network latency.
- The fill model is `full_fill_at_mid_no_queue`: every eligible directional
  signal is assumed filled at the mid price with no queue position.
- The return proxy is a forward mid-price change in basis points, inherited
  from the stored execution-sensitivity rows.

## How to read the output

- The metrics are useful for stress-testing signal fragility, not for
  proving tradability.
- A model can hold a respectable forecasting metric while its net proxy
  signal shrinks or turns negative once cost and latency are applied.

## Boundaries

- These are simplified proxy diagnostics, not a backtest.
- This is not a live-trading simulation and models no market impact.
- No profitability or live tradability claim is made.
- No foundation-model, leading-benchmark or self-supervised result is claimed.
- Neural superiority over the classical baseline is not asserted.
