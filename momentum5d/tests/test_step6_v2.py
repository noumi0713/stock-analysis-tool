# ruff: noqa: E501
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_step6_v2.py"
SPEC = importlib.util.spec_from_file_location("build_step6_v2", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_validation_auc_uses_formation_direction() -> None:
    assert MODULE.oriented(0.40, "high") == 0.40
    assert MODULE.oriented(0.70, "low") == pytest.approx(0.30)


def test_filter_rejects_future_and_absolute_levels() -> None:
    dtype = pd.Series([1.0]).dtype
    assert MODULE.base_exclusion("forward_return_40d", dtype) == "future_or_target_information"
    assert MODULE.base_exclusion("Close", dtype) == "absolute_price_volume_or_trading_value_level"
    assert MODULE.base_exclusion("volume_mean_20d", dtype) == "volume_or_trading_value_average_level"
    assert MODULE.base_exclusion("return_20d", dtype) is None


def test_formation_quantiles_are_fixed_for_validation() -> None:
    definitions = dict(MODULE.quantile_definitions(pd.Series(range(1, 101), dtype="float64")))
    thresholds = definitions["coarse_30_70"]
    assert thresholds == [30.7, 70.3]
    assert MODULE.assign_bins(pd.Series([-100.0, 50.0, 1000.0]), thresholds).tolist() == [1, 2, 3]


def test_validation_is_not_reoriented() -> None:
    rows = []
    for period, year, event_value, control_value in (
        ("formation", 2023, 1.0, 2.0), ("validation", 2024, 3.0, 2.0),
    ):
        for index in range(20):
            event_id = f"{period}-{index}"
            rows.extend([
                {"event_id": event_id, "period": period, "pair_year": year, "outcome": 1, "x": event_value},
                {"event_id": event_id, "period": period, "pair_year": year, "outcome": 0, "x": control_value},
            ])
    metrics, annual = MODULE.metrics_for_bucket(pd.DataFrame(rows), ["x"], MODULE.MAIN_BUCKET)
    result = metrics.iloc[0]
    assert result.formation_direction == "low"
    assert result.formation_raw_auc == 0.0
    assert result.validation_raw_auc == 1.0
    assert result.validation_fixed_direction_auc == 0.0
    validation = annual[annual.period == "validation"].iloc[0]
    assert validation.formation_direction == "low"
    assert validation.fixed_direction_auc == 0.0
