# Event Log Storage

Phase 9 adds a canonical local JSONL event-log layer for auditable market
microstructure research artefacts.

Each line is one JSON object with:

- `record_type`: `book_event` or `order_book_snapshot`;
- `schema_version`: currently `1.0`;
- `payload`: the JSON form of the existing `BookEvent` or
  `OrderBookSnapshot` schema.

The helpers in `chronoslob.data.event_store` preserve schema identity by
round-tripping through the existing Pydantic schemas. They reject unsupported
record types, unsupported schema versions, empty payloads, malformed JSON and
empty writes. Reads preserve file order. The separate
`sort_event_log_records` helper intentionally returns a sorted copy by
timestamp and sequence id where available.

`chronoslob.data.manifests` builds reproducibility manifests from local logs.
The manifest records record counts, record-type counts, sorted symbols,
timestamp range, sequence-id range, a SHA-256 content hash and a timezone-aware
UTC creation timestamp. Manifest metadata accepts only simple scalar values so
it stays predictable in JSON.

Generic event-level book reconstruction from arbitrary `BookEvent` records is
not implemented in this phase. Phase 8 Binance-style reconstruction should
first emit canonical `OrderBookSnapshot` objects, which can then be written as
event-log snapshot records.

No real venue data is committed. The event-log fixtures under
`tests/fixtures/event_logs/` are synthetic and exist only for storage and replay
tests.
