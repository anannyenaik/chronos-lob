# Data Quality

This note documents the data-quality checks implemented during Phase 2 of
ChronosLOB. It is deliberately conservative: validation here is about
catching malformed inputs early, not about claiming benchmark performance.

## Checks implemented

The generic `validate_numeric_frame` helper raises issues for:

- empty dataframes (no rows);
- duplicate column names;
- absent required columns;
- non-numeric required columns;
- NaN values in numeric columns (unless `allow_nan=True`);
- positive or negative infinity in numeric columns.

The FI-2010-specific `validate_fi2010_dataset` helper additionally checks:

- presence of every configured feature column;
- presence of every configured label column;
- numeric dtype and finiteness of feature columns;
- finiteness of label columns when they are numeric;
- presence and absence-of-missing-values for the split column when one is
  configured;
- parseability of the timestamp column when one is configured;
- pairing consistency across each LOB level — bid price, bid quantity, ask
  price and ask quantity must be all present or all absent at a given
  level, with `bid_size_i` / `ask_size_i` accepted as aliases for the
  quantity columns when the default prefixes are configured.

`load_fi2010` always runs `validate_fi2010_dataset` and raises a
`DataValidationError` if any error-severity issues are found. Callers that
want to surface warnings without raising can invoke the validator directly.

## What is not yet checked

Phase 2 does not verify:

- temporal ordering across rows (deferred to feature/label phases);
- the calibration or distribution of label classes;
- statistical agreement with any published FI-2010 mirror;
- look-ahead leakage between label horizons and features (this becomes
  testable only after the feature and label phases land);
- exchange-specific microstructure constraints such as tick-size or
  lot-size grids.

## How FI-2010 validation differs from raw order book event validation

FI-2010 is a normalised, snapshot-style benchmark matrix. It is not raw
order book event data. The validator therefore deliberately does **not**:

- attempt to reconstruct order books from event streams;
- enforce sequence-id continuity;
- check that bid prices are strictly less than ask prices, because
  normalisation can place benchmark values on an arbitrary scale;
- assume that exchange timestamps are present.

Raw order book event validation will be added in later phases (Binance
reconstruction, event log storage and deterministic replay) and will live
behind its own validator with checks for sequence gaps, monotonic
timestamps and crossed books.

## Why no benchmark performance is claimed here

This phase implements only loading and validation. It does not train any
model, generate any forecast, run any backtest or publish any metric. Any
benchmark or performance numbers must come from a reproducible experiment
artefact in a later phase, with explicit configuration, code version and
random seed. None such exist yet.
