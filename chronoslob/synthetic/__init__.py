"""Synthetic event-level limit-order-book extension for ChronosLOB.

This package adds a small, storage-light, *synthetic* event-level pipeline that
complements the FI-2010 snapshot evidence. It generates deterministic
order-book event streams with known regimes, replays them into snapshots,
computes genuine event-level features (order-flow, cancellation and trade
imbalance, event intensity), derives future-window labels and runs small
baseline diagnostics.

Everything produced here is synthetic. It is a controlled stress-test
environment, not real-market evidence. It does not establish profitability,
tradability or real-market generalisation, and it does not change any FI-2010
limitation: FI-2010 snapshots still expose only snapshot proxies, never true
event-level order flow.
"""

from __future__ import annotations

from chronoslob.synthetic.events import (
    REGIME_LIBRARY,
    RegimeSpec,
    SyntheticEventConfig,
    SyntheticGenerationResult,
    default_regime_plan,
    generate_synthetic_events,
)
from chronoslob.synthetic.features import (
    EVENT_FEATURE_COLUMNS,
    build_event_feature_frame,
)
from chronoslob.synthetic.labels import (
    LABEL_COLUMNS,
    build_label_frame,
    validate_no_lookahead_frames,
)
from chronoslob.synthetic.replay import (
    ReplayQualityReport,
    ReplayResult,
    replay_events_to_snapshots,
)

__all__ = [
    "EVENT_FEATURE_COLUMNS",
    "LABEL_COLUMNS",
    "REGIME_LIBRARY",
    "RegimeSpec",
    "ReplayQualityReport",
    "ReplayResult",
    "SyntheticEventConfig",
    "SyntheticGenerationResult",
    "build_event_feature_frame",
    "build_label_frame",
    "default_regime_plan",
    "generate_synthetic_events",
    "replay_events_to_snapshots",
    "validate_no_lookahead_frames",
]
