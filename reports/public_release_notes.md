# Public Release Notes

Date: 2026-05-29

## Current Contents

ChronosLOB contains a leakage-safe FI-2010 research platform with classical
benchmarks, supervised and self-supervised transformer comparison
infrastructure, calibration diagnostics, execution-aware proxy diagnostics,
feature ablations, generated figures and an evidence pack with claim auditing.

## Complete Evidence

- Classical FI-2010 benchmark artefacts: `complete_real`.
- Neural full grid: `complete_real`, 135/135 matched one-epoch runs across
  folds 1-5, horizons 10/20/50, seeds 0-2 and supervised/masked/next-field
  objectives.
- Execution-v3: `complete_real`, offline cost-adjusted proxy diagnostics from
  stored neural-grid predictions.
- Final empirical report and evidence pack: regenerated from stored artefacts.

## Partial Evidence

- Feature ablations: `partial_real`, currently folds 1-5, horizon 10, seeds 0-2
  and logistic/ridge models.
- Figure outputs: real and traceable, with unsupported regime plots skipped
  because regime labels are unavailable.
- Project-audit archive: `unknown_staleness`.

## What Did Not Work

- SSL pretraining did not improve the completed matched full-grid comparison.
- The standalone SSL runner artefact is missing.
- Feature ablations beyond horizon 10 and beyond logistic/ridge remain
  unfinished because the wider scope was too expensive for this pass.
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

- Broaden feature ablations to horizons 20/50 and slower model families.
- Produce standalone SSL-runner evidence only if it adds useful, non-duplicative
  support beyond the matched full grid.
- Refresh the project-audit archive to clear `unknown_staleness`.
- Write the manual paper only after deciding its scope separately from generated
  reports.
