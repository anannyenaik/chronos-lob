"""Replay canonical event logs into snapshots, features and labels."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from chronoslob.book.reconstruction import ReconstructionResult
from chronoslob.data.event_store import (
    EventLogObject,
    sort_event_log_records,
    write_event_log_jsonl,
)
from chronoslob.data.schemas import BookEvent, OrderBookSnapshot
from chronoslob.features.pipeline import (
    FeaturePipelineConfig,
    build_feature_frame_from_snapshots,
)
from chronoslob.labels.leakage import validate_no_lookahead
from chronoslob.labels.pipeline import (
    LabelPipelineConfig,
    build_label_frame_from_snapshots,
)


def snapshots_from_event_log_records(
    records: Sequence[BookEvent | OrderBookSnapshot],
    *,
    sort_records: bool = False,
) -> list[OrderBookSnapshot]:
    """Extract explicit snapshots from canonical event-log records.

    Generic :class:`BookEvent` reconstruction is intentionally not
    implemented in Phase 9. Binance-style replay should first reconstruct
    snapshots via Phase 8, then persist those snapshots as event-log
    records.
    """
    ordered: Sequence[EventLogObject] = (
        sort_event_log_records(records) if sort_records else records
    )
    snapshots = [
        record for record in ordered if isinstance(record, OrderBookSnapshot)
    ]
    if not snapshots:
        raise ValueError(
            "event log contains no explicit order book snapshots; generic "
            "BookEvent reconstruction is not implemented in Phase 9"
        )
    return snapshots


def replay_event_log_to_feature_frame(
    records: Sequence[BookEvent | OrderBookSnapshot],
    *,
    feature_config: FeaturePipelineConfig | None = None,
    sort_records: bool = False,
) -> pd.DataFrame:
    """Replay explicit event-log snapshots into a past-only feature frame."""
    snapshots = snapshots_from_event_log_records(
        records, sort_records=sort_records
    )
    return build_feature_frame_from_snapshots(
        snapshots,
        config=feature_config,
    )


def replay_event_log_to_label_frame(
    records: Sequence[BookEvent | OrderBookSnapshot],
    *,
    label_config: LabelPipelineConfig | None = None,
    sort_records: bool = False,
) -> pd.DataFrame:
    """Replay explicit event-log snapshots into a future-horizon label frame."""
    snapshots = snapshots_from_event_log_records(
        records, sort_records=sort_records
    )
    return build_label_frame_from_snapshots(
        snapshots,
        config=label_config,
    )


def replay_event_log_to_feature_label_frames(
    records: Sequence[BookEvent | OrderBookSnapshot],
    *,
    feature_config: FeaturePipelineConfig | None = None,
    label_config: LabelPipelineConfig | None = None,
    sort_records: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay event-log snapshots into feature and label frames, then audit leakage."""
    snapshots = snapshots_from_event_log_records(
        records, sort_records=sort_records
    )
    feature_frame = build_feature_frame_from_snapshots(
        snapshots,
        config=feature_config,
    )
    label_frame = build_label_frame_from_snapshots(
        snapshots,
        config=label_config,
    )
    leakage = validate_no_lookahead(feature_frame, label_frame)
    if not leakage.ok:
        summary = leakage.summary()
        raise ValueError(
            "event-log replay feature/label leakage validation failed: "
            f"{summary}"
        )
    return feature_frame, label_frame


def write_binance_reconstruction_to_event_log(
    result: ReconstructionResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write reconstructed Binance-style snapshots as a canonical event log."""
    if not isinstance(result, ReconstructionResult):
        raise TypeError("result must be a ReconstructionResult")
    if not result.snapshots:
        raise ValueError("reconstruction result contains no snapshots to write")
    return write_event_log_jsonl(path, result.snapshots, overwrite=overwrite)


__all__ = [
    "replay_event_log_to_feature_frame",
    "replay_event_log_to_feature_label_frames",
    "replay_event_log_to_label_frame",
    "snapshots_from_event_log_records",
    "write_binance_reconstruction_to_event_log",
]
