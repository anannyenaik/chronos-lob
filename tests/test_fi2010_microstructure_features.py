from __future__ import annotations

import math

import pandas as pd
import pytest

from chronoslob.features.microstructure_fi2010 import build_microstructure_features
from chronoslob.features.registry import (
    FeatureRegistryError,
    feature_groups_for_columns,
    group_names,
    unsupported_group_names,
)


def _frame() -> pd.DataFrame:
    rows = []
    for idx in range(6):
        row: dict[str, object] = {
            "timestamp": f"2024-01-02T09:30:0{idx}+00:00",
            "split": "train" if idx < 3 else "test",
            "label_10": 1 if idx < 2 else (2 if idx < 4 else 3),
        }
        for level in range(1, 6):
            row[f"bid_price_{level}"] = 100.0 - level - idx * 0.1
            row[f"ask_price_{level}"] = 102.0 + level + idx * 0.1
            row[f"bid_quantity_{level}"] = 10.0 + level + idx
            row[f"ask_quantity_{level}"] = 8.0 + level
        rows.append(row)
    return pd.DataFrame(rows)


def test_registry_contains_expected_and_unsupported_groups() -> None:
    names = set(group_names())
    assert {
        "price_levels",
        "spread",
        "microprice",
        "snapshot_order_flow_proxy",
        "volatility_proxy",
    } <= names
    assert {"true_order_flow_imbalance", "queue_position"} <= set(unsupported_group_names())


def test_strict_registry_fails_unknown_or_empty_group() -> None:
    with pytest.raises(FeatureRegistryError):
        feature_groups_for_columns(["bid_price_1"], ["not_a_group"], strict=True)
    with pytest.raises(FeatureRegistryError):
        feature_groups_for_columns(["bid_price_1"], ["microprice"], strict=True)


def test_microstructure_formulas_and_label_exclusion() -> None:
    frame = _frame()
    result = build_microstructure_features(
        frame,
        feature_groups=[
            "spread",
            "midprice",
            "microprice",
            "top_of_book_imbalance",
            "depth_imbalance",
            "liquidity_concentration",
        ],
        strict=True,
    )
    first = result.features.iloc[0]
    bid = frame.loc[0, "bid_price_1"]
    ask = frame.loc[0, "ask_price_1"]
    bid_size = frame.loc[0, "bid_quantity_1"]
    ask_size = frame.loc[0, "ask_quantity_1"]
    assert first["spread"] == pytest.approx(ask - bid)
    assert first["relative_spread"] == pytest.approx((ask - bid) / ((ask + bid) / 2.0))
    assert first["midprice"] == pytest.approx((ask + bid) / 2.0)
    assert first["microprice"] == pytest.approx(
        (ask * bid_size + bid * ask_size) / (bid_size + ask_size)
    )
    assert first["top_of_book_imbalance"] == pytest.approx(
        (bid_size - ask_size) / (bid_size + ask_size)
    )
    bid_5 = sum(float(frame.loc[0, f"bid_quantity_{level}"]) for level in range(1, 6))
    ask_5 = sum(float(frame.loc[0, f"ask_quantity_{level}"]) for level in range(1, 6))
    assert first["depth_imbalance_l5"] == pytest.approx((bid_5 - ask_5) / (bid_5 + ask_5))
    total = bid_5 + ask_5
    top1 = bid_size + ask_size
    assert first["liquidity_concentration_top1"] == pytest.approx(top1 / total)
    assert "label_10" not in result.features.columns


def test_snapshot_delta_proxy_resets_at_partition_boundary() -> None:
    frame = _frame()
    result = build_microstructure_features(
        frame,
        feature_groups=["snapshot_order_flow_proxy"],
        partition_columns=["split"],
        strict=True,
    )
    delta = result.features["snapshot_delta_bid_price_1"].tolist()
    assert delta[0] == 0.0
    assert delta[3] == 0.0
    assert delta[1] != 0.0


def test_rolling_volatility_uses_past_rows_only() -> None:
    frame = _frame()
    base = build_microstructure_features(
        frame,
        feature_groups=["volatility_proxy"],
        partition_columns=["split"],
        volatility_window=3,
        strict=True,
    ).features
    changed = frame.copy()
    changed.loc[5, "bid_price_1"] += 1000.0
    changed.loc[5, "ask_price_1"] += 1000.0
    mutated = build_microstructure_features(
        changed,
        feature_groups=["volatility_proxy"],
        partition_columns=["split"],
        volatility_window=3,
        strict=True,
    ).features
    assert base.loc[:4, "volatility_proxy"].tolist() == pytest.approx(
        mutated.loc[:4, "volatility_proxy"].tolist()
    )
    assert math.isfinite(float(mutated.loc[5, "volatility_proxy"]))


def test_feature_rows_align_with_labels() -> None:
    frame = _frame()
    result = build_microstructure_features(frame, feature_groups=["spread"], strict=True)
    assert result.features["row_id"].tolist() == list(frame.index)
    assert len(result.features) == len(frame["label_10"])
