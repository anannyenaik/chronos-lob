# ChronosLOB Leakage Controls

Phase 4 adds explicit leakage checks in `chronoslob.labels.leakage`. These
checks are designed to make feature and label artefacts auditable before any
model phase exists.

## Feature And Label Separation

`FeatureRow` and `LabelRow` remain separate schemas. Feature frames should contain
only information available at or before timestamp `t`. Label frames may describe
future outcomes, but those values must not appear in feature columns.

`assert_feature_label_separation` checks:

- label value columns do not appear in the feature frame;
- feature-like value columns are reported if they appear in label value columns;
- obvious label prefixes are rejected in feature columns: `label`, `y_`,
  `future_` and `target`.

The default shared columns are `timestamp`, `symbol` and `split`.

## Timestamp Alignment

`assert_temporal_label_alignment` checks `LabelRow` objects for:

- timezone-aware `timestamp`, `horizon_start` and `horizon_end`;
- `horizon_start >= timestamp`;
- `horizon_end > horizon_start`.

`assert_no_future_feature_timestamps` checks `FeatureRow` objects for:

- timezone-aware feature timestamps;
- `horizon_origin_timestamp <= timestamp` whenever an origin is present.

`validate_no_lookahead` combines separation checks with any available horizon
checks on label rows and tabular frames.

## Explicit Horizon Bounds

Generated label rows record the future window through `horizon_start` and
`horizon_end`. When multiple horizons are included in one `LabelRow`, the end
timestamp is the maximum requested horizon or adverse-selection evaluation
horizon used in that row.

This makes the forecasting target auditable, but it does not by itself solve
overlap between adjacent labels.

## What Is Guaranteed Now

The current tests prove that:

- generated feature frames do not contain future label columns;
- generated label frames do not contain feature columns such as `mid_price` or
  `spread`;
- feature generation is not mutated by label generation;
- generated label horizons are explicit and timezone-aware;
- obvious label-like feature columns fail leakage checks.

## What Later Phases Must Still Handle

Phase 5 must add temporal splitters, purged or embargoed validation and an
experiment registry skeleton. Those splitters must prevent leakage through:

- overlapping label windows across train, validation and test partitions;
- fitting quantile bins or normalisation statistics on validation/test data;
- reuse of future outcomes in calibration, threshold selection or model inputs;
- accidental random splits in financial time-series experiments.

The current checks are necessary guardrails, not a full validation protocol.
