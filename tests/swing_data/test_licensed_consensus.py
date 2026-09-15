import pandas as pd

from swing_data.licensed_consensus import (
    composite_targets,
    load_ifis_normalized,
    load_quick_normalized,
)


def test_ifis_normalized_loader(tmp_path):
    path = tmp_path / "ifis.csv"
    pd.DataFrame([
        {
            "stock_code": "4506",
            "target_low": 1500,
            "target_mean": 2100,
            "target_high": 2700,
            "analyst_count": 7,
            "base_date": "2026-09-14",
            "source_url": "licensed-ifis",
        }
    ]).to_csv(path, index=False)
    frame = load_ifis_normalized(path)
    row = frame.iloc[0]
    assert row["stock_code"] == "4506"
    assert row["ifis_target_low"] == 1500
    assert row["ifis_target_mean"] == 2100
    assert row["ifis_target_high"] == 2700
    assert row["ifis_analyst_count"] == 7


def test_quick_broker_rows_are_aggregated(tmp_path):
    path = tmp_path / "quick.csv"
    pd.DataFrame([
        {"stock_code": "4506", "target_price": 1600, "broker_name": "A", "updated_at": "2026-09-10", "source_url": "quick"},
        {"stock_code": "4506", "target_price": 2200, "broker_name": "B", "updated_at": "2026-09-11", "source_url": "quick"},
        {"stock_code": "4506", "target_price": 2800, "broker_name": "C", "updated_at": "2026-09-12", "source_url": "quick"},
    ]).to_csv(path, index=False)
    frame = load_quick_normalized(path)
    row = frame.iloc[0]
    assert row["quick_target_low"] == 1600
    assert row["quick_target_mean"] == 2200
    assert row["quick_target_high"] == 2800
    assert row["quick_analyst_count"] == 3


def test_three_source_composite_is_equal_weight_mean():
    result = composite_targets({
        "yahoo_target_low": 1500,
        "yahoo_target_mean": 2000,
        "yahoo_target_high": 2500,
        "ifis_target_low": 1600,
        "ifis_target_mean": 2100,
        "ifis_target_high": 2600,
        "quick_target_low": 1700,
        "quick_target_mean": 2200,
        "quick_target_high": 2700,
    })
    assert result["target_low"] == 1600
    assert result["target_mean"] == 2100
    assert result["target_high"] == 2600
    assert result["composite_source_count"] == 3
    assert result["composite_quality"] == "three_source"


def test_quick_missing_still_uses_yahoo_and_ifis():
    result = composite_targets({
        "yahoo_target_low": 1500,
        "yahoo_target_mean": 2000,
        "yahoo_target_high": 2500,
        "ifis_target_low": 1700,
        "ifis_target_mean": 2200,
        "ifis_target_high": 2700,
        "quick_target_low": None,
        "quick_target_mean": None,
        "quick_target_high": None,
    })
    assert result["target_low"] == 1600
    assert result["target_mean"] == 2100
    assert result["target_high"] == 2600
    assert result["composite_source_count"] == 2
    assert result["composite_quality"] == "two_source"


def test_one_source_does_not_create_composite():
    result = composite_targets({
        "yahoo_target_low": 1500,
        "yahoo_target_mean": 2000,
        "yahoo_target_high": 2500,
    })
    assert result["target_low"] is None
    assert result["target_mean"] is None
    assert result["target_high"] is None
    assert result["composite_source_count"] == 1
    assert result["composite_quality"] == "insufficient_sources"
