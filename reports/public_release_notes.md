# Public Release Notes

Date: 2026-05-29

## Current Contents

ChronosLOB contains a leakage-safe FI-2010 research platform with classical
benchmarks, supervised and self-supervised transformer comparison
infrastructure, calibration diagnostics, execution-aware proxy diagnostics,
feature ablations, generated figures and an evidence pack with claim auditing.
The public story now centres on the execution-aware gap between forecast quality
and trading-signal quality.

## Retained Evidence

- Classical FI-2010 benchmark artefacts: `archived_valid`.
- Neural full grid: `archived_valid`, 135/135 matched one-epoch runs across
  folds 1-5, horizons 10/20/50, seeds 0-2 and supervised/masked/next-field
  objectives.
- Execution-v3: `archived_valid`, offline cost-adjusted proxy diagnostics from
  stored neural-grid predictions.
- Execution-v3 analysis: `complete_real`, using retained proxy tables.
- Execution centrepiece: `archived_valid`, with the
  forecasting-versus-signal-quality figure and retained proxy diagnostics.
- SSL-v2: `complete_real` for folds 1-5, horizons 10/50, seed 0 and lookback 50.
- Final empirical report and evidence pack: regenerated from stored artefacts.

## Partial Evidence

- Proper-training neural subset: `partial_real`, currently fold 1, horizons
  10/50, seed 0 and lookback 50.
- Feature ablations: `partial_real`, with logistic/ridge folds 1-5, horizons
  10/20/50 and seeds 0-2, plus a small gradient-boosting slice.
- Figure outputs: real and traceable, with unsupported regime plots skipped
  because regime labels are unavailable.
- Project-audit archive: `unknown_staleness`.

## What Did Not Work

- SSL pretraining did not improve the completed matched full-grid comparison.
- Broad SSL improvement remains unsupported.
- SSL-v2 calibration improvement remains unsupported; the predictive-metric
  improvement is scoped to the exact stored seed-0 SSL-v2 slice.
- The standalone SSL runner artefact is superseded by the matched full grid and
  SSL-v2 benchmark.
- Regime execution plots are skipped because the required regime labels are not
  present in prediction artefacts.

## Intentionally Not Claimed

- No live trading.
- No profitability or PnL.
- No tradable alpha.
- No SOTA or foundation-model status.
- No production execution simulator.
- No true event-level OFI, cancellation imbalance, trade imbalance or
  queue-position modelling from FI-2010 snapshots.

## Future Work

- Broaden SSL-v2 beyond seed 0; seeds 1 and 2 are deferred.
- Broaden non-linear feature-ablation coverage beyond the current small slice.
- Refresh the project-audit archive to clear `unknown_staleness`.
- Write the manual paper only after deciding its scope separately from generated
  reports.
