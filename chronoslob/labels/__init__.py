"""Future market-state labels and leakage-control utilities."""

from chronoslob.labels.adverse_selection import (
    compute_adverse_selection_after_fill_proxy,
    compute_adverse_selection_proxy_series,
)
from chronoslob.labels.fill_probability import (
    compute_passive_fill_proxy,
    compute_passive_fill_proxy_series,
)
from chronoslob.labels.leakage import (
    LeakageCheckResult,
    assert_feature_label_separation,
    assert_no_future_feature_timestamps,
    assert_temporal_label_alignment,
    validate_no_lookahead,
)
from chronoslob.labels.midprice import (
    classify_direction,
    compute_direction_labels,
    compute_future_return,
    compute_future_returns,
    compute_return_quantile_labels,
)
from chronoslob.labels.pipeline import (
    LabelPipelineConfig,
    build_label_frame_from_fi2010,
    build_label_frame_from_snapshots,
    build_label_rows_from_snapshots,
    validate_label_frame,
)
from chronoslob.labels.spread import (
    compute_future_spread_change,
    compute_spread_widening_label,
    compute_spread_widening_labels,
)
from chronoslob.labels.volatility import (
    classify_volatility_labels,
    compute_future_realised_volatility,
    compute_future_volatility_series,
)

__all__ = [
    "LabelPipelineConfig",
    "LeakageCheckResult",
    "assert_feature_label_separation",
    "assert_no_future_feature_timestamps",
    "assert_temporal_label_alignment",
    "build_label_frame_from_fi2010",
    "build_label_frame_from_snapshots",
    "build_label_rows_from_snapshots",
    "classify_direction",
    "classify_volatility_labels",
    "compute_adverse_selection_after_fill_proxy",
    "compute_adverse_selection_proxy_series",
    "compute_direction_labels",
    "compute_future_realised_volatility",
    "compute_future_return",
    "compute_future_returns",
    "compute_future_spread_change",
    "compute_future_volatility_series",
    "compute_passive_fill_proxy",
    "compute_passive_fill_proxy_series",
    "compute_return_quantile_labels",
    "compute_spread_widening_label",
    "compute_spread_widening_labels",
    "validate_label_frame",
    "validate_no_lookahead",
]
