# Event Tokenisation

Phase 11 adds deterministic event tokenisation and token-window preparation for
future transformer work. It converts canonical `OrderBookSnapshot` and
`BookEvent` records into field-wise categorical IDs, while keeping prediction
targets, trainable embeddings, transformer architecture and self-supervised
objectives out of scope.

## Field-Wise Token Design

Every tokenised record carries separate integer IDs for:

- event type
- side
- relative price bucket
- quantity bucket
- time-delta bucket
- spread/context bucket
- source

Special token IDs are fixed in every field vocabulary:

- `[PAD] = 0`
- `[UNK] = 1`
- `[BOS] = 2`
- `[EOS] = 3`
- `[MASK] = 4`

The field-wise design avoids flattening market-state information into a single
opaque token too early. Future transformer phases can decide whether to embed
fields separately, combine channels, or add positional encodings.

## Snapshot-Derived Tokens

`OrderBookSnapshot` records are converted deterministically:

1. Emit one snapshot summary token.
2. Emit up to `max_levels_per_side` bid pseudo-level tokens from best to deeper
   levels.
3. Emit up to `max_levels_per_side` ask pseudo-level tokens from best to deeper
   levels.

The `snapshot_level` tokens are derived transformer input representations. They
are not raw exchange-native add, cancel, modify or trade events.

By default, snapshot level prices are bucketed relative to the snapshot mid-price,
which is symmetric across bid and ask sides. The config also supports
`best_same_side` and `best_opposite_side` references.

## BookEvent Tokenisation Limits

`BookEvent` records are tokenised directly from the fields present on the event.
The implementation does not reconstruct a full order book from generic events.
For event records, context/spread buckets remain missing unless the current event
metadata already contains a safe spread value known at or before the event
timestamp.

## Bucket Choices

Price buckets are fixed relative basis-point buckets. Quantity buckets are fixed
log-style buckets with missing, zero, very small, small, medium, large and very
large labels. Time-delta buckets are computed from the previous record timestamp
within the same symbol sequence and use fixed thresholds from milliseconds to
more than ten seconds. Spread/context buckets are fixed and can use either
absolute spread values or spread in basis points relative to the mid-price.

No quantile bucket fitting is implemented in this phase. This keeps validation
and test tokenisation independent of future data.

## Split Safety

The default bucket boundaries are static. Source vocabulary fitting is optional
and, when used through `fit_tokenisation_state`, should be fitted on training
records only. If split indices are supplied, only `SplitIndices.train` rows are
used to fit source tokens. Validation and test tokenisation use the frozen
vocabulary; unseen source values map to `[UNK]`, while missing source metadata
maps to `unknown_source`.

Token-window indexing preserves temporal order and, by default, stops windows
before crossing symbol or supplied split-id boundaries. Padding uses `[PAD]` and
the boolean attention mask marks real tokens.

## No-Look-Ahead Constraints

Tokenisation uses only information available on the record at timestamp `t`:

- snapshot mid-price and spread are computed from the same snapshot;
- event fields are tokenised directly;
- event context is not inferred from future snapshots;
- labels are not read or used;
- vocabulary and fitted state are frozen before validation/test tokenisation.

## Out of Scope

This phase deliberately does not implement:

- transformer encoders or attention layers;
- trainable embeddings;
- masked event modelling;
- next-event prediction;
- supervised transformer classifiers;
- calibration;
- execution simulation;
- backtesting;
- fake metrics or result artefacts.

## Known Limitations

The buckets are intentionally conservative defaults and may need domain-specific
calibration in a later reproducible experiment. Generic `BookEvent` records do
not carry reconstructed book context unless upstream metadata provides it
safely. The bundled JSONL fixtures are synthetic engineering examples, not
evidence about real venue behaviour or tradability.
