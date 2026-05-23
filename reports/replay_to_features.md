# Replay To Features

Phase 9 connects canonical event logs to the existing feature and label
pipelines without adding a modelling layer.

Replay starts by extracting explicit `OrderBookSnapshot` records from an event
log. Mixed logs may contain `BookEvent` records, but Phase 9 does not infer a
full book from generic events. If a log contains no explicit snapshots, replay
raises a clear error.

`replay_event_log_to_feature_frame` sends ordered snapshots into the Phase 3
feature pipeline. These features remain past-only because each feature row uses
the current snapshot and, where needed, prior snapshots only.

`replay_event_log_to_label_frame` sends the same snapshots into the Phase 4
label pipeline. Labels keep explicit `horizon_start` and `horizon_end` columns
so future-window assumptions remain auditable.

`replay_event_log_to_feature_label_frames` builds both frames and runs the
available no-look-ahead checks. A failed leakage check raises instead of
silently returning questionable data.

The Phase 8 bridge, `write_binance_reconstruction_to_event_log`, writes only
reconstructed snapshots to the canonical event-log format. Reconstruction
issues are not embedded in the log itself; they can be summarised separately in
manifest metadata or reports in a later phase.

This phase performs no model training, transformer tokenisation,
self-supervised learning, execution backtesting or PnL calculation. The next
research step is event tokenisation and transformer input preparation from
auditable replay artefacts.
