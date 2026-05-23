# Transfer, Regime and Ablation Analysis

This report documents the Phase 17 analysis layer added under
`chronoslob/analysis/`. The layer organises and summarises supplied
experiment-result records. It does not train models, does not run
backtests and does not generate evidence by itself.

## Purpose

Earlier phases produced predictive metrics (accuracy, macro-F1, MCC,
NLL), calibration metrics (Brier score, expected calibration error) and
execution-aware validation metrics (coverage, fill rate, simulated net
PnL, turnover, adverse-selection rate, max drawdown). To turn those
metrics into a research story rather than a single headline number, we
must be able to ask:

- Does the model transfer between symbols, periods or regimes?
- How do metrics change across volatility, spread, liquidity, confidence
  and latency regimes?
- Which features, objectives, task heads or execution settings actually
  matter when removed?
- How sensitive is the system to the confidence threshold, latency, fee
  bps, spread multiplier, turnover cap or inventory cap?

The analysis layer answers these questions over structured records. It
keeps predictive and execution metrics clearly separated so that a
calibration improvement is never confused with a tradability claim.

## Why robustness analysis matters for microstructure

Market microstructure models tend to be optimised on a narrow slice of
data. A single accuracy number on one symbol, one period and one
execution assumption is rarely informative. Real research-grade work
requires evidence that the result is not an accident of a particular
regime, sample or hyperparameter. The analysis layer makes those checks
first-class artefacts rather than ad-hoc notebook cells.

## Regime analysis design

`chronoslob/analysis/regimes.py` provides explicit regime kinds:

- volatility: low / medium / high
- spread: tight / normal / wide
- liquidity: thin / normal / deep
- confidence: low / medium / high / very_high
- latency: zero / low / medium / high

Regime labels can be assigned in three ways:

1. **Explicit labels already on the record** — preferred when upstream
   experiments emit a regime label directly.
2. **Threshold-based assignment** — uses explicit thresholds supplied via
   config or function arguments. Thresholds are not derived from
   evaluation data inside assignment functions.
3. **Fitted boundaries** — implemented in a separate
   `fit_regime_boundaries` function that the caller must drive with
   training or calibration values only. This separation prevents
   accidental data snooping on evaluation data.

`summarise_by_regime` groups records by regime label and reports count,
mean, minimum and maximum for a single metric. Synthetic flags survive
aggregation so synthetic records cannot be silently treated as evidence.

## Transfer matrix design

`chronoslob/analysis/transfer.py` represents transfer-style records with
`TransferResult(train_scope, eval_scope, metric_name, metric_value, ...)`.

`build_transfer_matrix` produces a deterministic matrix indexed by
`(train_scope, eval_scope)` for a single metric. Missing cells are left
as `None` rather than imputed. `compare_in_domain_vs_out_of_domain`
summarises diagonal versus off-diagonal cells.

Transfer comparisons support:

- train symbol A, evaluate symbol B
- train period A, evaluate period B
- train regime A, evaluate regime B
- pretrain source, fine-tune target, evaluate target

Phase 17 does not train models. It strictly organises supplied result
records.

## Ablation comparison design

`chronoslob/analysis/ablations.py` defines `AblationSpec`,
`AblationResult` and `AblationComparison`. `compare_against_baseline`
computes absolute and relative deltas against a named baseline for a
single metric. Metric direction (`higher_is_better` or
`lower_is_better`) is encoded explicitly. `rank_ablations` orders the
comparisons from largest improvement to largest regression under the
declared direction. Relative deltas are returned as `None` when the
baseline is zero or non-finite to avoid silent division surprises.

Ablation categories supported:

- feature
- token_field
- model_component
- objective
- task_head
- execution_setting

Example ablation names referenced by the smoke config:

- `no_order_flow_features`
- `no_depth_imbalance`
- `no_ssl_pretraining`
- `no_calibration`
- `no_confidence_filtering`
- `no_latency`
- `aggressive_only`
- `passive_only`

## Sensitivity curve design

`chronoslob/analysis/sensitivity.py` defines
`SensitivityParameter`, `SensitivityPoint` and `SensitivityCurve`.
`build_sensitivity_curve` orders the supplied points by parameter value
and refuses duplicate parameter values. `summarise_sensitivity_curve`
selects the best point under the declared metric direction.
`compare_sensitivity_curves` compares curves that share the same metric
and direction.

Supported parameters:

- confidence_threshold
- latency_steps
- fee_bps
- spread_multiplier
- turnover_cap
- inventory_cap
- mask_probability
- temperature

This is the natural home for the confidence-threshold sweep from
Phase 15 calibration and the latency-sensitivity grid from Phase 16
execution validation.

## Keeping predictive, calibration and execution metrics separate

The summary layer registers each supported metric with an explicit
direction:

- predictive: accuracy, macro_f1, mcc, nll, brier_score, ece
- execution: coverage, fill_rate, simulated_net_pnl, total_cost,
  turnover, adverse_selection_rate, max_drawdown, latency_steps

Predictive and execution names do not overlap. The layer refuses to
accept records containing forbidden combined-score fields such as
`combined_score`, `magic_score`, `alpha_score`, `tradability_score` or
`sharpe`. This is deliberate. Mixing predictive quality and execution
quality into a single number hides the gap between statistical
predictability and execution-aware signal quality, which is the
project's core research question.

## Why no combined magic score is used

A model can have a strong accuracy or macro-F1 score while losing money
after spread, fee and latency costs. It can also clear a generous
calibration threshold while still failing under realistic adverse
selection. Aggregating these into one score makes it impossible to see
which property failed. The analysis layer therefore reports per-metric
summaries only and leaves the interpretation to a human reader.

## Why synthetic smoke records are not evidence

The smoke runner in `chronoslob/analysis/summary.py` builds
deterministic synthetic analysis records, transfer records, ablation
records and sensitivity points. Every record carries
`is_synthetic=True`. The CLI command
`run-robustness-analysis-smoke` and the YAML config
`configs/experiments/robustness_analysis_smoke.yaml` both label the run
as plumbing. The synthetic records are not market evidence, alpha
evidence, tradability evidence or live performance.

The smoke layer exists only to verify that the analysis machinery runs
end to end and that future real-record inputs will be summarised
correctly.

## How this will later support the final report

When Phase 18 hardens CI and the final paper/CV phase begins, real
upstream experiments will produce real `AnalysisRecord`,
`TransferResult`, `AblationResult` and `SensitivityPoint` inputs. The
analysis layer will then turn those records into transfer matrices,
regime breakdowns, ablation tables and sensitivity curves without any
new modelling code. The same layer will support the final report's
robustness section.

## Limitations and next steps

- Synthetic smoke records are not evidence.
- Regime thresholds are explicit by default; data-derived boundaries
  require an explicit `fit_regime_boundaries` call with training or
  calibration data only.
- Sensitivity comparisons assume the analyst has already produced the
  underlying sweep results elsewhere.
- There is no combined magic score. The analysis layer reports
  per-metric summaries only.
- No new model architectures, live data ingestion, broker integration,
  notebook outputs, dashboards or fake benchmark tables are added.

Phase 18 will harden CI, full audit and reproducibility around the
combined Phase 16 and Phase 17 infrastructure. No live trading,
profitability claim or alpha claim is introduced by Phase 17.
