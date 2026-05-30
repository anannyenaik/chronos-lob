# FI-2010 Execution Validation V3

Execution-v3 is an offline execution-aware proxy diagnostic for stored FI-2010
neural full-grid predictions. It can also be pointed explicitly at
microstructure feature-ablation prediction artefacts. It measures how
directional signal quality changes under confidence filtering, costs, row-step
latency and proxy fill assumptions.
It is not a live trading system, broker integration, profitability claim or
realistic execution simulator.

The current canonical execution-v3 artefact is `complete_real` and is built from
the completed neural full grid. It uses `unit_payoff` and `unit_proxy` modes, so
reported cost-adjusted values are cost-adjusted proxies, not PnL.

## Command

```bash
python -m chronoslob.cli build-fi2010-execution-v3 \
  --neural-full-grid experiments/fi2010_neural_full_grid \
  --feature-ablations experiments/fi2010_feature_ablations \
  --out experiments/fi2010_execution_v3 \
  --models all \
  --horizons all \
  --folds all \
  --seeds all \
  --confidence-thresholds 0.33,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95 \
  --fee-bps 0,1,2,5,10 \
  --spread-multipliers 0,0.5,1.0,2.0 \
  --latency-steps 0,1,2,5,10 \
  --fill-assumptions aggressive_crossing,passive_optimistic,passive_conservative,abstain_only \
  --overwrite \
  --strict
```

Smoke-test grids require `--allow-smoke-test`. Smoke outputs are marked as
smoke-test-only and are not empirical evidence.

## Inputs

Required inputs are stored full-grid prediction artefacts and
`results_summary.csv`, unless `--feature-ablations` is supplied. In that mode
the builder reads `runs/**/predictions.csv` from the feature-ablation directory
and preserves model, ablation-mode and feature-group metadata. The builder also
records hashes for `summary.json`,
`aggregate_summary.csv`, `aggregate_summary.json`, `ssl_comparison.csv` and
`label_mapping_audit.json` when they are available.

Prediction rows must provide `y_true`, `y_pred`, named FI-2010 probabilities and
confidence. Metadata is inherited from the result row when needed:
`fold`, `horizon`, `seed`, `lookback`, `model_family` and
`pretraining_objective`.

Strict mode refuses ambiguous probability columns such as `prob_0`, `prob_1`
and `prob_2`. The canonical FI-2010 mapping is always:

- `1 = up`
- `2 = stationary`
- `3 = down`

Optional market-context fields may include `mid_price`, `spread`,
top-of-book depth, `imbalance`, signed future return or move columns and regime
labels. Missing optional fields skip only the diagnostics that require them.
The builder does not invent market context.

## Directional Proxy

Predicted `up` is a long signal, predicted `down` is a short signal and
predicted `stationary` is no trade. Stationary predictions remain part of
classification metrics but are excluded from trade-specific metrics.

When a signed future return or move column is present, execution-v3 uses
`realised_return` payoff mode. Otherwise it uses `unit_payoff` mode:

- correct directional trade: `+1`
- incorrect directional trade: `-1`
- stationary or no trade: `0`

Every summary records the selected payoff mode.

## Cost Modes

If spread fields are unavailable, costs use `unit_proxy` mode. Fee bps are
converted to payoff units and spread multipliers apply to a small abstract unit
cost. If spread fields are available, costs use `spread_proxy` mode and combine
fee bps with the configured spread multiplier. Costs are subtracted from active
or filled directional trades only.

Cost-adjusted values are cost-adjusted proxies. They are not PnL and do not
support profitability claims.

## Outputs

- `confidence_threshold_summary.csv`: per model, objective, horizon, fold, seed,
  lookback and threshold retained fraction, active fraction, retained
  classification metrics, directional hit rate, gross directional proxy,
  cost-adjusted proxy, turnover proxy and retained class mix.
- `confidence_threshold_aggregate.csv`: model/objective/horizon aggregates over
  threshold rows.
- `cost_sensitivity_summary.csv`: gross proxy, cost-adjusted proxy, degradation
  percentage, active trade count, active fraction, average cost and cost mode for
  each fee and spread-multiplier setting.
- `latency_sensitivity_summary.csv`: row-step latency diagnostics. Outcomes are
  shifted only within the same fold, partition and run group; dropped samples are
  recorded.
- `fill_assumption_summary.csv`: filled counts and proxy outcomes under
  `aggressive_crossing`, `passive_optimistic`, `passive_conservative` and
  `abstain_only`.
- `adverse_selection_summary.csv`: adverse-selection proxy by model, objective,
  horizon, confidence bucket and fill assumption.
- `regime_execution_summary.csv`: execution metrics by explicit regime label
  when regime labels are present. Missing regimes are recorded as skipped rows.
- `skipped_diagnostics.json`: skipped diagnostics and reasons.
- `summary.json`: compact execution-v3 status.
- `execution_v3_manifest.json`: input artefact paths, input hashes, output
  files, output hashes, label-mapping audit status, payoff mode, cost mode,
  fill modes, latency modes, smoke status, strict-mode status and skipped
  diagnostics.

## Latency Handling

Latency is a row-step diagnostic. For latency `L`, the signal at row `t` is
evaluated against the outcome at row `t + L` inside the same fold, partition and
run group. The shift never crosses fold or partition boundaries. If a latency
step drops all rows, the row is marked skipped.

## Fill Assumptions

- `aggressive_crossing`: every active directional signal is assumed filled and
  receives the highest cost multiplier.
- `passive_optimistic`: active signals require the configured confidence
  threshold and favourable spread/liquidity proxies when available. Without
  those proxies, the mode records a confidence-only fallback.
- `passive_conservative`: active signals require higher confidence and more
  favourable spread/liquidity proxies. Missing market context is recorded as a
  conservative confidence-only fallback.
- `abstain_only`: no active trades are filled. This is a sanity-check baseline.

## Adverse Selection

If future move or post-fill movement data is available, a long is adversely
selected when the future movement is negative and a short is adversely selected
when the future movement is positive. Otherwise execution-v3 uses
`label_proxy`: a long is adverse when `y_true` is down, and a short is adverse
when `y_true` is up.

## Supported Claims

With real, non-smoke artefacts, the repository can support this claim:

> ChronosLOB evaluates FI-2010 neural and SSL predictions using offline
> execution-aware proxy diagnostics, showing how confidence filtering, costs,
> latency and fill assumptions affect directional signal quality under
> leakage-safe evaluation.

## Unsupported Claims

Execution-v3 does not support claims about profitability, tradability,
broker routing, live trading, market impact, queue position, exchange fills,
production execution quality or a live-market backtest.

## Evidence-Pack Use

`build-evidence-pack` reads `summary.json` and `execution_v3_manifest.json` to
separate smoke diagnostics from real proxy artefacts, verify recorded hashes
where possible and keep release wording limited to offline execution-aware proxy
diagnostics. Stale or unknown-staleness execution-v3 rows are not treated as
clean empirical support.

## Richer Analysis (`analyse-fi2010-execution-v3`)

`analyse-fi2010-execution-v3` is a second-stage, reviewer-facing summariser. It
consumes only the retained lightweight execution-v3 output tables above and
writes a richer analysis to `reports/execution_v3_analysis/`. It never opens the
deleted raw per-run prediction arrays; `raw_predictions_required` is always
`false`, and if the upstream tables are missing it fails with a clear
missing-input message rather than silently requiring deleted data.

```bash
python -m chronoslob.cli analyse-fi2010-execution-v3 \
  --execution-v3 experiments/fi2010_execution_v3 \
  --out reports/execution_v3_analysis \
  --overwrite
```

It answers one practical question: do apparently better forecasts still look
useful after confidence filtering, simple cost assumptions, latency shifts,
turnover pressure and adverse-selection proxies? Outputs:

- `execution_v3_analysis.md`: the narrative report.
- `confidence_filtering_summary.csv`: mean retained fraction, active fraction,
  abstention fraction, accuracy, macro-F1, directional hit rate and cost-adjusted
  proxy by objective, horizon and confidence threshold.
- `turnover_proxy_summary.csv`: mean signal-change-rate turnover proxy, active
  fraction and a turnover-adjusted cost proxy by threshold.
- `latency_sensitivity_summary.csv`: mean cost-adjusted proxy degradation versus
  latency 0 and directional hit rate by horizon, objective and row-step lag.
- `cost_sensitivity_summary.csv`: mean gross proxy, cost-adjusted proxy and
  degradation percentage across the fee and spread-multiplier grid.
- `fill_assumption_summary.csv`: mean fill fraction, hit rate and cost-adjusted
  proxy by proxy fill mode.
- `adverse_selection_proxy_summary.csv`: filled-weighted adverse-selection proxy
  rate by objective, horizon, confidence bucket and fill assumption.
- `skipped_regime_diagnostics.json`: explicit, non-silent skip. Regime
  diagnostics are skipped because the retained tables and the underlying FI-2010
  prediction artefacts carry no regime labels or snapshot market-context columns;
  the exact fields needed to build supported snapshot-derived proxy regimes are
  recorded as future work. Regimes are not invented.
- `execution_claim_assessment.json`, `summary.json` and `figure_manifest.json`.

Optional figures cover active fraction, cost-adjusted proxy and turnover proxy
versus confidence threshold, latency degradation by horizon, the
adverse-selection proxy by confidence bucket and fill-assumption sensitivity.

All outputs remain offline execution-aware proxy diagnostics. They are not PnL,
not live-trading evidence and not a production execution simulator.

## Execution Centrepiece (`build-execution-centrepiece`)

`build-execution-centrepiece` is the compact reviewer-facing layer over the
retained execution-v3 analysis tables. It joins retained full-grid predictive
and calibration summaries with confidence filtering, active fraction, turnover
proxy, cost-adjusted proxy, latency sensitivity and adverse-selection proxy
diagnostics. Deleted raw prediction arrays are not required.

```bash
python -m chronoslob.cli build-execution-centrepiece \
  --execution-analysis reports/execution_v3_analysis \
  --execution-v3 experiments/fi2010_execution_v3 \
  --neural-full-grid experiments/fi2010_neural_full_grid \
  --out reports/execution_centrepiece \
  --overwrite
```

The centrepiece writes `execution_centrepiece.md`, compact CSVs, a claim
assessment, `centrepiece_summary.json`, `figure_manifest.json` and the central
`forecasting_vs_signal_quality.png` figure. It makes the
forecasting-versus-signal-quality gap visible while keeping the same offline
diagnostic boundary: no PnL, no live trading and no tradability claim.
