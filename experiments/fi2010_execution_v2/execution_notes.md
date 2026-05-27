# FI-2010 Execution-Aware Evaluation v2 - Notes

These notes summarise the stored proxy diagnostics. They describe
fragility, not tradability, and make no profitability claim.

## Most sensitive to cost

- The cost proxy is a fixed per-trade deduction, so the absolute net reduction from cost is the same across models; the model whose gross proxy return is smallest crosses into the weakest net proxy first.
- At the highest stored cost (5 bps, reference threshold and latency) `random_forest` has the lowest mean net signal return proxy (+4.8244 bps).

## Most sensitive to latency

- At the highest stored latency (1 steps) `gradient_boosting` shows the largest adverse-selection proxy (+2.4313 bps of gross signal lost versus latency 0).

## Confidence thresholding

- The sharpest stored coverage trade-off is `majority` at threshold 0.7: coverage changes by -1.0000 and the hit-rate proxy by -0.5894 versus the most permissive threshold.
- Raising the confidence threshold lowers coverage (fewer eligible predictions) and generally raises the hit-rate proxy; `confidence_threshold_summary.csv` records both deltas per model.

## Where the net proxy signal degrades most

- `gradient_boosting` shows the largest gap between its base gross proxy return and its stressed net proxy return (+7.4313 bps); see `degradation_summary.csv`.
- Neural runs report a forecasting metric but ship no execution proxy rows, so their execution-aware side is skipped, not assumed; the gap between their forecasting metric and tradability is therefore unquantified here.

## What cannot be concluded

- Nothing here demonstrates profitability or live tradability.
- Without a market-impact model, queue ground truth and a realistic fill
  model, the net proxy return cannot be read as an achievable return.
- Cross-model net comparisons are conditioned on identical scenario
  assumptions and a shared, simplified return proxy.
