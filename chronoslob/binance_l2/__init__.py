"""Aggregated Binance Spot L2 replay for ChronosLOB.

This package provides a compact, offline pipeline that ingests a Binance Spot
L2 depth snapshot plus a diff-depth update stream, reconstructs the local order
book deterministically with snapshot-plus-diff update-id logic, validates update
continuity and book invariants, computes the supported event-level features and
writes a compact result summary. Binance diff-depth updates contain aggregated
level changes rather than individual order events and are evaluated separately
from FI-2010.
"""

from __future__ import annotations

from chronoslob.binance_l2.features import (
    BINANCE_FEATURE_COLUMNS,
    UNSUPPORTED_FEATURES,
    build_binance_feature_frame,
    build_update_continuity_frame,
)
from chronoslob.binance_l2.pipeline import (
    BINANCE_L2_BUILDER_VERSION,
    BinanceL2Config,
    BinanceL2Result,
    default_fixture_paths,
    run_binance_l2_pipeline,
)
from chronoslob.binance_l2.quality import (
    BinanceReplayQualityReport,
    build_replay_quality_report,
)

__all__ = [
    "BINANCE_FEATURE_COLUMNS",
    "BINANCE_L2_BUILDER_VERSION",
    "UNSUPPORTED_FEATURES",
    "BinanceL2Config",
    "BinanceL2Result",
    "BinanceReplayQualityReport",
    "build_binance_feature_frame",
    "build_replay_quality_report",
    "build_update_continuity_frame",
    "default_fixture_paths",
    "run_binance_l2_pipeline",
]
