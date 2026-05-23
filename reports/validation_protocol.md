# Validation Protocol

Phase 5 adds the split and registry infrastructure needed before any modelling
phase begins. The purpose is to make short-horizon market-state forecasting
experiments auditable under temporal ordering and future-window labels.

## Temporal Splits

`chronoslob.training.splitters.temporal_train_validation_test_split` builds
contiguous ordered train, validation and test partitions. Train rows occur first,
validation rows second and test rows last. The splitter uses row order only and
does not inspect feature or label values.

Random splits are dangerous for financial time series because they can mix market
states across regimes and allow future information to influence earlier training
decisions. Temporal splitting is therefore the default validation assumption.

## Walk-Forward Validation

`walk_forward_splits` creates deterministic walk-forward folds. Expanding folds
start training at row zero and grow the train window each fold. Rolling folds use
a fixed train window. Validation follows training, and an optional test block
follows validation. Incomplete final folds are skipped.

Walk-forward validation is useful for checking whether signal quality is stable
across time rather than concentrated in one favourable period.

## Purging And Embargo

Financial labels often summarise future windows. If a training label at row `t`
uses data through `t + h`, it can overlap a later validation block. Purging
removes training rows whose label horizon intersects the evaluation block.

`apply_row_embargo` then removes training rows within a symmetric row window
around the evaluation block. The current implementation supports row-based
embargoes only; time-based embargoes can be added later once the convention is
needed and tested.

## Label Horizon Mapping

`label_horizon_end_indices_from_rows` and
`make_label_horizon_end_indices_from_frame` convert `horizon_end` timestamps into
row-index horizon ends. This lets purging operate on generated `LabelRow`
objects or label frames without using label values.

## Train-Only Fitting

`TrainOnlyQuantileBinner` is a small guardrail for return-quantile labels and
similar preprocessing artefacts. It fits bin edges on training values and then
applies those fixed edges to validation or test values. Future normalisation,
calibration and threshold selection must follow the same principle: fit on the
training partition only, then transform later partitions.

## Implemented Now

- Contiguous temporal train/validation/test splitting.
- Expanding and rolling walk-forward folds.
- Purged and embargoed train-index filtering for overlapping label horizons.
- Timestamp-to-row horizon mapping for `LabelRow` objects and label frames.
- Train-only quantile bin fitting.
- Tests for split boundaries, overlap removal, embargo and metadata behaviour.

## Requirements For Future Model Phases

Future models, baselines and evaluation layers must use these split definitions
or provide an equally explicit leakage-safe alternative. They must not fit
normalisation statistics, quantile bins, calibration parameters or decision
thresholds on validation/test data. Any execution-aware validation remains a
simplified research simulation until costs, queue position, latency, partial
fills and venue rules are explicitly modelled.
