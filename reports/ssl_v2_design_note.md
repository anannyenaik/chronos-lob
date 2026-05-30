# SSL-v2 Design Note

## Motivation

The first-generation FI-2010 SSL objectives were implemented and analysed under
matched supervised-vs-SSL settings. The completed evidence does not support a
broad SSL improvement claim, and it does not support a calibration-improvement
claim. The proper-training subset left a narrower observation: fold 1, horizon
50 masked reconstruction improved predictive metrics, but calibration worsened
and the scope remained partial_real.

SSL-v2 is added because that failure pattern is informative. Random field
reconstruction can spend capacity rebuilding normalised columns that are easy
to interpolate but weakly related to the downstream mid-price direction task.
Next-field prediction is also local: it predicts one-step bucketed feature
movement rather than the future market state used by horizon-specific labels.

## Objective

The implemented SSL-v2 objective is `market_state_multitask`.

It reuses the same matrix-transformer encoder architecture as the supervised
classifier so the encoder can be transferred into downstream fine-tuning with
matching state-dict keys.

The objective combines:

- structured masked reconstruction over coherent feature groups:
  `price_depth`, `imbalance`, `spread_microprice` and `temporal_context`;
- future spread-widening classification;
- future volatility-bucket classification;
- future return-bucket classification;
- future imbalance-bucket classification;
- an optional regime contrastive term, disabled by default.

Structured masking masks groups over a contiguous temporal span rather than
independent random fields. Future labels are computed strictly after the window
end. Quantile edges for auxiliary buckets are fitted only on train-current /
train-future pairs.

## Leakage Controls

- Pretraining optimisation uses windows built from the training partition.
- A window is admitted only if every input row and its future target row are in
  the same partition.
- Future auxiliary labels use row `t + horizon`, never row `t`.
- Auxiliary quantile edges are fitted from training rows only.
- Validation pretraining losses, when present, are diagnostics and not used for
  test selection.
- Downstream fine-tuning keeps validation-only model selection and restores the
  best validation checkpoint before the single official test evaluation.

## Expected Benefits

SSL-v2 should be more aligned with downstream prediction because its auxiliary
heads ask the encoder to model future spread, volatility, return and imbalance
states rather than reconstruct arbitrary fields. If it helps, the most credible
signal would be matched improvements in macro-F1 and MCC under the same
fine-tuning budget.

## Risks

- Auxiliary proxies may still be too crude for the downstream label.
- Future-state heads can improve representation geometry while worsening
  calibration.
- A single fold or horizon cannot support broad SSL claims.
- Contrastive learning can add instability if regime labels are sparse, so it
  remains optional.

## Run Scope

Preferred credible scope:

- folds: 1-3, with 1-5 as a larger extension if feasible;
- horizons: 10 and 50;
- seed: 0;
- lookback: 50;
- objectives: supervised, masked_reconstruction, market_state_multitask;
- SSL-v2 pretrain epochs: 5;
- fine-tune max epochs: 25;
- patience: 5.

Minimum useful scope if compute is limiting:

- fold: 1;
- horizons: 10 and 50;
- seed: 0;
- lookback: 50;
- objectives: supervised and market_state_multitask, with
  masked_reconstruction imported or rerun if feasible.

Any reduced scope is classified `partial_real` and described exactly.

## Claim Boundaries

Allowed claims are limited to implementation, scoped evaluation, and exact
metric deltas from stored artefacts. Calibration improvement requires ECE and
Brier support. Broad SSL improvement remains unsupported unless broad matched
evidence changes that conclusion.

This design does not claim profitability, PnL, live trading, tradable alpha,
state-of-the-art performance, a foundation model, market-wide generalisation,
equity-market generalisation, event-level FI-2010 OFI, or queue-position
modelling.

## Compute and Storage Budget

The default runner writes compact summaries, per-run metrics, training curves,
pretraining loss components, config snapshots and SHA256 manifests. Predictions
are retained in per-run directories when produced by the fine-tuning path; the
benchmark root summary and analysis can be used without loading large
prediction files.
