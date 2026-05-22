"""Import smoke tests for the initial package scaffold."""

from __future__ import annotations

import importlib


def test_chronoslob_imports() -> None:
    package = importlib.import_module("chronoslob")

    assert package.__version__


def test_key_subpackages_import() -> None:
    modules = [
        "chronoslob.data",
        "chronoslob.book",
        "chronoslob.features",
        "chronoslob.labels",
        "chronoslob.models",
        "chronoslob.training",
        "chronoslob.backtest",
        "chronoslob.analysis",
        "chronoslob.utils",
    ]

    for module in modules:
        assert importlib.import_module(module)


def test_cli_module_imports() -> None:
    assert importlib.import_module("chronoslob.cli")


def test_schema_modules_import() -> None:
    schemas = importlib.import_module("chronoslob.data.schemas")
    events = importlib.import_module("chronoslob.book.events")

    for name in (
        "OrderBookLevel",
        "OrderBookSnapshot",
        "BookEvent",
        "FeatureRow",
        "LabelRow",
        "DataQualityIssue",
        "Side",
        "EventType",
    ):
        assert hasattr(schemas, name), f"chronoslob.data.schemas is missing {name}"

    for name in (
        "sort_levels_for_side",
        "validate_book_side_order",
        "has_duplicate_prices",
        "top_of_book",
    ):
        assert hasattr(events, name), f"chronoslob.book.events is missing {name}"


def test_fi2010_modules_import() -> None:
    fi2010 = importlib.import_module("chronoslob.data.fi2010")
    validation = importlib.import_module("chronoslob.data.validation")

    for name in (
        "FI2010Config",
        "FI2010Dataset",
        "load_fi2010",
        "infer_fi2010_columns",
        "build_snapshot_from_row",
    ):
        assert hasattr(fi2010, name), f"chronoslob.data.fi2010 is missing {name}"

    for name in (
        "DataValidationResult",
        "DataValidationError",
        "validate_numeric_frame",
        "validate_fi2010_dataset",
    ):
        assert hasattr(validation, name), (
            f"chronoslob.data.validation is missing {name}"
        )


def test_feature_modules_import() -> None:
    microprice = importlib.import_module("chronoslob.features.microprice")
    imbalance = importlib.import_module("chronoslob.features.imbalance")
    order_flow = importlib.import_module("chronoslob.features.order_flow")
    volatility = importlib.import_module("chronoslob.features.volatility")
    regimes = importlib.import_module("chronoslob.features.regimes")
    pipeline = importlib.import_module("chronoslob.features.pipeline")
    features = importlib.import_module("chronoslob.features")

    for name in (
        "compute_mid_price",
        "compute_spread",
        "compute_relative_spread",
        "compute_microprice",
        "compute_snapshot_price_features",
    ):
        assert hasattr(microprice, name), (
            f"chronoslob.features.microprice is missing {name}"
        )

    for name in (
        "compute_depth",
        "compute_depth_imbalance",
        "compute_queue_imbalance",
        "compute_level_imbalances",
        "compute_depth_slope",
        "compute_liquidity_concentration",
    ):
        assert hasattr(imbalance, name), (
            f"chronoslob.features.imbalance is missing {name}"
        )

    for name in (
        "compute_order_flow_imbalance_from_snapshots",
        "compute_order_flow_imbalance_series",
        "compute_trade_imbalance_from_events",
    ):
        assert hasattr(order_flow, name), (
            f"chronoslob.features.order_flow is missing {name}"
        )

    for name in (
        "compute_log_returns",
        "compute_realised_volatility",
        "compute_rolling_realised_volatility",
        "compute_event_intensity",
        "compute_rolling_event_intensity",
    ):
        assert hasattr(volatility, name), (
            f"chronoslob.features.volatility is missing {name}"
        )

    for name in (
        "RegimeThresholds",
        "classify_spread_regime",
        "classify_volatility_regime",
        "classify_liquidity_regime",
        "classify_imbalance_regime",
        "compute_regime_thresholds_from_frame",
    ):
        assert hasattr(regimes, name), (
            f"chronoslob.features.regimes is missing {name}"
        )

    for name in (
        "FeaturePipelineConfig",
        "build_features_from_snapshot",
        "build_feature_frame_from_snapshots",
        "build_feature_frame_from_fi2010",
        "validate_feature_frame",
    ):
        assert hasattr(pipeline, name), (
            f"chronoslob.features.pipeline is missing {name}"
        )
        assert hasattr(features, name), (
            f"chronoslob.features is missing re-exported {name}"
        )


def test_label_modules_import() -> None:
    midprice = importlib.import_module("chronoslob.labels.midprice")
    volatility = importlib.import_module("chronoslob.labels.volatility")
    spread = importlib.import_module("chronoslob.labels.spread")
    fill_probability = importlib.import_module("chronoslob.labels.fill_probability")
    adverse_selection = importlib.import_module("chronoslob.labels.adverse_selection")
    leakage = importlib.import_module("chronoslob.labels.leakage")
    pipeline = importlib.import_module("chronoslob.labels.pipeline")
    labels = importlib.import_module("chronoslob.labels")

    for name in (
        "compute_future_return",
        "compute_future_returns",
        "classify_direction",
        "compute_direction_labels",
        "compute_return_quantile_labels",
    ):
        assert hasattr(midprice, name), f"chronoslob.labels.midprice is missing {name}"

    for name in (
        "compute_future_realised_volatility",
        "compute_future_volatility_series",
        "classify_volatility_labels",
    ):
        assert hasattr(volatility, name), (
            f"chronoslob.labels.volatility is missing {name}"
        )

    for name in (
        "compute_future_spread_change",
        "compute_spread_widening_label",
        "compute_spread_widening_labels",
    ):
        assert hasattr(spread, name), f"chronoslob.labels.spread is missing {name}"

    for name in ("compute_passive_fill_proxy", "compute_passive_fill_proxy_series"):
        assert hasattr(fill_probability, name), (
            f"chronoslob.labels.fill_probability is missing {name}"
        )

    for name in (
        "compute_adverse_selection_after_fill_proxy",
        "compute_adverse_selection_proxy_series",
    ):
        assert hasattr(adverse_selection, name), (
            f"chronoslob.labels.adverse_selection is missing {name}"
        )

    for name in (
        "LeakageCheckResult",
        "assert_feature_label_separation",
        "assert_temporal_label_alignment",
        "assert_no_future_feature_timestamps",
        "validate_no_lookahead",
    ):
        assert hasattr(leakage, name), f"chronoslob.labels.leakage is missing {name}"

    for name in (
        "LabelPipelineConfig",
        "build_label_rows_from_snapshots",
        "build_label_frame_from_snapshots",
        "build_label_frame_from_fi2010",
        "validate_label_frame",
    ):
        assert hasattr(pipeline, name), f"chronoslob.labels.pipeline is missing {name}"
        assert hasattr(labels, name), (
            f"chronoslob.labels is missing re-exported {name}"
        )
