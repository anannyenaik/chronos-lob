# FI-2010 Execution-Aware Evaluation v2

## Purpose

This layer makes the forecasting-versus-tradability gap explicit. It does
not introduce a new model and it does not retrain anything. It re-frames
the lightweight artefacts already produced by the multi-fold runners and
the brutal ablation layer as a focused set of execution-aware proxy
diagnostics, so the distance between a statistical metric and a stressed
execution proxy is visible per model.

Every execution number here is a simplified proxy under stated
assumptions. It is not a backtest, it is not a live-trading simulation,
and it makes no profitability or live tradability claim.

## Inputs

The command consumes stored, lightweight artefacts only:

- `experiments/fi2010_multifold_classical/` - per-fold
  `folds/fold_<N>/execution_sensitivity.csv` rows are the numeric source;
  `results_by_fold.csv` supplies the held-out test macro-F1 used in the
  degradation summary. When per-fold files are absent the aggregate
  `execution_summary.csv` is used with a synthetic `aggregate` fold id.
- `experiments/fi2010_multifold_neural/` - `results_by_fold_seed.csv`
  supplies the neural held-out test macro-F1. Neural runs ship no stored
  execution proxy rows, so their execution-aware diagnostics are skipped.
- `experiments/fi2010_brutal_ablations/` - recorded as a cross-reference
  input in `summary.json`; the layer does not depend on it.

No raw data, processed CSV files, full prediction rows or checkpoints are
read or required.

## Command

```bash
python -m chronoslob.cli run-fi2010-execution-v2 \
  --classical experiments/fi2010_multifold_classical \
  --neural experiments/fi2010_multifold_neural \
  --ablations experiments/fi2010_brutal_ablations \
  --out experiments/fi2010_execution_v2 \
  --overwrite
```

Optional filters narrow the loaded rows before any aggregation:

- `--models gradient_boosting,logistic` - restrict to specific models.
- `--cost-bps 0,1,5` - keep specific cost scenarios.
- `--latency-steps 0,1` - keep specific latency steps.
- `--confidence-thresholds 0,0.6` - keep specific thresholds.

## Artefacts

All artefacts are written under `--out` and are small enough to keep:

- `summary.json` - inputs and their hashes, scenario grid, filters,
  counts, diagnostics produced and skipped, and claim boundaries.
- `execution_v2_results.csv` - the per-scenario long table; one row per
  model, fold, threshold, cost and latency, plus one skipped row per
  neural model.
- `cost_latency_surface.csv` - mean and spread of the proxy metrics across
  folds for each cost and latency cell.
- `confidence_threshold_summary.csv` - coverage and hit-rate proxy per
  threshold at the reference cost and latency, with deltas versus the most
  permissive threshold.
- `turnover_summary.csv` - turnover proxy, trade-count proxy and coverage
  per threshold.
- `adverse_selection_summary.csv` - the latency-induced signal-decay proxy
  per model, or an explicit skip where it cannot be computed.
- `fill_assumption_summary.csv` - the eligible-to-fill share per model
  under the stated fill assumption, or an explicit skip.
- `degradation_summary.csv` - the headline gap: a statistical metric next
  to a stressed execution-aware proxy metric per model.
- `skipped_diagnostics.json` - every skipped diagnostic with its reason.
- `execution_assumptions.md` - the full assumption list.
- `execution_notes.md` - a concise interpretation.

## Proxy metrics

Every metric carries a `_proxy` name or sits under an artefact labelled as
a proxy diagnostic:

- `coverage` - eligible predictions relative to the most permissive
  threshold for the same model, fold and latency.
- `trade_count_proxy` and `turnover_proxy` - directional trade activity.
- `gross_signal_return_proxy` and `net_signal_return_proxy` - mean
  direction-signed forward mid-price change in basis points, before and
  after the per-trade cost deduction.
- `hit_rate_proxy` - share of eligible predictions that are correct.
- `adverse_selection_proxy` - gross signal lost relative to the reference
  latency, a proxy for signal decay when acting later.
- `fill_assumption_proxy` - trade-count proxy divided by eligible
  predictions, the share assumed filled under `full_fill_at_mid_no_queue`.
- The degradation columns contrast `test_macro_f1` with the base gross
  proxy and the stressed net proxy.

## Skipped diagnostics

Nothing is silently dropped. A diagnostic that needs unavailable data is
written as a skipped row with a populated `skip_reason` and is also listed
in `skipped_diagnostics.json`. The recurring cases are:

- neural models, which ship no stored execution proxy rows, so their
  adverse-selection, fill-assumption and degradation execution side are
  skipped;
- the adverse-selection proxy when only a single latency step is stored;
- the fill-assumption proxy when the stored rows lack a trade-count
  column.

## Limitations

- There is no market impact model and no queue-position ground truth.
- Costs are scenario assumptions, not measured exchange fees.
- Latency is row-step latency, not exchange or network latency.
- The net proxy return cannot be read as an achievable return.
- Cross-model comparisons are conditioned on identical scenario
  assumptions and a shared, simplified return proxy.

## Claim boundaries

- These are simplified proxy diagnostics, not a backtest.
- This is not a live-trading simulation.
- No profitability or live tradability claim is made.
- No foundation-model, leading-benchmark or self-supervised result is
  claimed.
- Neural superiority over the classical baseline is not asserted.
