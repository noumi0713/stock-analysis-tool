from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from app.yahoo.sakata_backtest import (
    SUPPORT_BASE_METHODS,
    SakataBacktestConfig,
    _add_support_features,
    _apply_support_stop,
    _attach_outcomes,
)


def test_confirmed_swing_low_is_not_available_before_confirmation() -> None:
    lows = [10.0, 9.0, 8.0, 9.0, 10.0, 11.0]
    frame = pd.DataFrame(
        {
            "date": [date(2026, 1, 1) + timedelta(days=index) for index in range(6)],
            "ticker": ["1111.T"] * 6,
            "low": lows,
            "high": [value + 2.0 for value in lows],
            "close": [value + 1.0 for value in lows],
            "volume": [100.0] * 6,
        }
    )

    result = _add_support_features(frame)

    assert pd.isna(result.loc[3, "support_confirmed_swing_low"])
    assert result.loc[4, "support_confirmed_swing_low"] == 8.0


def test_anchored_vwap_uses_only_information_available_before_session() -> None:
    frame = pd.DataFrame(
        {
            "date": [date(2026, 1, 1) + timedelta(days=index) for index in range(4)],
            "ticker": ["1111.T"] * 4,
            "open": [10.0, 20.0, 30.0, 40.0],
            "high": [10.0, 20.0, 30.0, 40.0],
            "low": [10.0, 20.0, 30.0, 40.0],
            "close": [10.0, 20.0, 30.0, 40.0],
            "volume": [100.0, 300.0, 500.0, 700.0],
        }
    )
    features = _add_support_features(frame)

    result = _attach_outcomes(features, SakataBacktestConfig(horizon_days=2))

    assert result.loc[0, "support_anchored_vwap_1"] == 10.0
    assert result.loc[0, "support_anchored_vwap_2"] == pytest.approx(17.5)


def test_same_day_support_and_target_hit_exits_at_stop_first() -> None:
    config = SakataBacktestConfig(horizon_days=1, support_atr_buffer=0.5)
    trade = pd.DataFrame(
        {
            "date": [date(2026, 4, 1)],
            "ticker": ["1111.T"],
            "entry_price": [100.0],
            "exit_price": [105.0],
            "exit_date": [date(2026, 4, 2)],
            "target_hit_day": [1],
            "target_hit": [True],
            "gross_return": [0.05],
            "net_return": [0.048],
            "trade_win": [True],
        }
    )
    evaluated_data: dict[str, list[object]] = {
        "date": [date(2026, 4, 1)],
        "ticker": ["1111.T"],
        "atr_14": [2.0],
        "future_date_1": [date(2026, 4, 2)],
        "future_open_1": [100.0],
        "future_high_1": [106.0],
        "future_low_1": [88.0],
        "future_close_1": [102.0],
    }
    for method in SUPPORT_BASE_METHODS:
        evaluated_data[f"support_{method}_1"] = [90.0]
    evaluated = pd.DataFrame(evaluated_data)

    result = _apply_support_stop(
        trade,
        evaluated,
        support_method="inflow_day_low",
        config=config,
    )

    assert result.loc[0, "initial_stop_price"] == 89.0
    assert result.loc[0, "exit_price"] == 89.0
    assert bool(result.loc[0, "stop_hit"])
    assert not bool(result.loc[0, "target_hit"])
    assert bool(result.loc[0, "ambiguous_both_hit"])
    assert result.loc[0, "net_return"] == pytest.approx(-0.112)


def test_nearest_valid_support_selects_highest_line_below_entry() -> None:
    config = SakataBacktestConfig(horizon_days=1, support_atr_buffer=0.5)
    trade = pd.DataFrame(
        {
            "date": [date(2026, 4, 1)],
            "ticker": ["1111.T"],
            "entry_price": [100.0],
            "exit_price": [101.0],
            "exit_date": [date(2026, 4, 2)],
            "target_hit_day": [pd.NA],
            "target_hit": [False],
            "gross_return": [0.01],
            "net_return": [0.008],
            "trade_win": [True],
        }
    )
    evaluated_data: dict[str, list[object]] = {
        "date": [date(2026, 4, 1)],
        "ticker": ["1111.T"],
        "atr_14": [2.0],
        "future_date_1": [date(2026, 4, 2)],
        "future_open_1": [100.0],
        "future_high_1": [102.0],
        "future_low_1": [99.0],
        "future_close_1": [101.0],
    }
    for method in SUPPORT_BASE_METHODS:
        evaluated_data[f"support_{method}_1"] = [90.0]
    evaluated_data["support_ma_5_1"] = [98.0]
    evaluated_data["support_anchored_vwap_1"] = [101.0]
    evaluated = pd.DataFrame(evaluated_data)

    result = _apply_support_stop(
        trade,
        evaluated,
        support_method="nearest_valid",
        config=config,
    )

    assert result.loc[0, "support_method"] == "ma_5"
    assert result.loc[0, "initial_support_price"] == 98.0
    assert result.loc[0, "initial_stop_price"] == 97.0
