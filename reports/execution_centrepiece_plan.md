# Execution Centrepiece Plan

## Objective

Create a reviewer-facing execution centrepiece that makes the
forecasting-versus-signal-quality gap visible without adding a new trading,
backtest or profitability claim. The centrepiece will join retained predictive
summaries, calibration metrics and execution-v3 proxy diagnostics so the report
shows why macro-F1 or accuracy must be read alongside confidence filtering,
active fraction, turnover proxy, cost-adjusted proxy, latency sensitivity and
adverse-selection proxy diagnostics.

## Retained Inputs Used

- `reports/execution_v3_analysis/summary.json`
- `reports/execution_v3_analysis/confidence_filtering_summary.csv`
- `reports/execution_v3_analysis/turnover_proxy_summary.csv`
- `reports/execution_v3_analysis/cost_sensitivity_summary.csv`
- `reports/execution_v3_analysis/latency_sensitivity_summary.csv`
- `reports/execution_v3_analysis/fill_assumption_summary.csv`
- `reports/execution_v3_analysis/adverse_selection_proxy_summary.csv`
- `reports/execution_v3_analysis/skipped_regime_diagnostics.json`
- `reports/execution_v3_analysis/execution_claim_assessment.json`
- `experiments/fi2010_neural_full_grid/aggregate_summary.csv`
- `experiments/fi2010_neural_full_grid/summary.json`
- `experiments/fi2010_execution_v3/summary.json`
- `experiments/fi2010_execution_v3/execution_v3_manifest.json`

## What Can Be Analysed

- Raw predictive metrics available from retained full-grid aggregate summaries:
  macro-F1, accuracy, MCC, ECE, Brier score and NLL by objective and horizon.
- Confidence filtering from retained execution-v3 analysis tables: retained
  fraction, active fraction, retained macro-F1, directional hit rate and
  cost-adjusted proxy by confidence threshold.
- Turnover proxy by objective, horizon and confidence threshold.
- Cost sensitivity across retained fee-bps and spread-multiplier assumptions.
- Latency sensitivity across retained row-step lags.
- Fill-assumption sensitivity across retained proxy fill modes.
- Adverse-selection proxy by retained confidence bucket and fill assumption.
- Explicit skip state for regime diagnostics where retained artefacts lack
  regime or snapshot market-context columns.

## What Cannot Be Analysed Without Regeneration

- Deleted raw neural-grid prediction arrays are not available and are not
  required for this centrepiece.
- Confidence-filtered ECE is unavailable because the retained execution-v3
  summaries include thresholded macro-F1 and accuracy but not thresholded
  calibration bins or thresholded ECE.
- True realised returns, queue position, market impact, broker fills and venue
  mechanics cannot be inferred from the retained proxy tables.
- Supported regime diagnostics cannot be built without regenerating prediction
  rows with explicit regime labels or snapshot market-context columns.
- True event-level OFI on FI-2010 cannot be derived from retained snapshot
  matrices.

## Chosen Figures

- `forecasting_vs_signal_quality.png`: one four-panel centrepiece figure.
  Panels show active fraction, cost-adjusted proxy and turnover proxy versus
  confidence threshold, plus adverse-selection proxy by confidence bucket.
  A text annotation records the retained macro-F1 and ECE range from the
  full-grid aggregate summary.

## Chosen Tables

- `forecasting_vs_signal_quality.csv`: long table joining retained predictive
  and calibration metrics to confidence-threshold proxy diagnostics.
- `confidence_threshold_tradeoff.csv`: threshold-level retained fraction, active
  fraction, macro-F1, turnover proxy and cost-adjusted proxy.
- `metric_to_proxy_gap.csv`: compact reviewer table by objective and horizon,
  with selected thresholds 0.50, 0.70 and 0.85 and explicit unavailable fields.
- `latency_cost_gap.csv`: representative cost and latency sensitivity by
  objective and horizon.
- `adverse_selection_by_confidence.csv`: adverse-selection proxy by objective,
  horizon, confidence bucket and fill assumption.

## Claim Boundaries

Use only the following interpretation:

- execution-aware proxy diagnostic
- cost-adjusted proxy
- signal-quality proxy
- confidence filtering
- active fraction
- turnover proxy
- latency sensitivity
- adverse-selection proxy
- forecasting-versus-signal-quality gap
- offline diagnostic

The centrepiece must not claim PnL, profitability, live trading, tradable alpha,
real execution realism, market impact modelling, queue-position modelling,
production execution simulation or real-market predictive success.

## Interpretation Plan

1. Start from retained macro-F1 and ECE to show what forecasting metrics say.
2. Add confidence filtering to show how the active fraction shrinks.
3. Add turnover proxy to expose signal churn under the same thresholds.
4. Add cost-adjusted proxy to show why gross predictive metrics are incomplete.
5. Add latency sensitivity and adverse-selection proxy as separate stress
   diagnostics, not as trading-performance claims.
6. End with the limitation that the centrepiece does not establish
   profitability or tradability; it shows why forecast metrics must be read
   alongside execution-aware proxy diagnostics.
