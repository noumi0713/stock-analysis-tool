from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.build_step10_v2 import EXIT_NAMES, make_exit_definitions, simulate_one


def series(start: str = "2023-01-02", periods: int = 100) -> dict[str, np.ndarray]:
    dates = pd.bdate_range(start, periods=periods).to_numpy(dtype="datetime64[ns]")
    return {
        "date": dates,
        "adj_open": np.full(periods, 100.0),
        "adj_high": np.full(periods, 101.0),
        "adj_low": np.full(periods, 99.0),
        "adj_close": np.full(periods, 100.0),
        "adj_atr14": np.full(periods, 1.0),
        "volume": np.full(periods, 1000.0),
        "price_values_loaded": np.ones(periods, dtype=bool),
    }


def entry(**changes):
    values = {
        "sample_id": "s1",
        "event_id": "e1",
        "Ticker": "1000.T",
        "period": "formation",
        "observation_type": "event",
        "outcome": 1,
        "anchor_date": pd.Timestamp("2023-01-30"),
        "entry_definition": "anchor_close",
        "entry_date": pd.Timestamp("2023-01-30"),
        "entry_session_position": 20,
        "entry_price": 100.0,
        "entry_status": "hypothetical_fill",
        "filled": True,
        "entry_rule_reason": "hypothetical_fill",
    }
    values.update(changes)
    return SimpleNamespace(**values)


def test_exit_definitions_are_fixed() -> None:
    definitions = make_exit_definitions()
    assert tuple(definitions.exit_definition) == EXIT_NAMES
    assert definitions.fixed_before_formation_review.all()
    assert not definitions.validation_used_to_change_rule.any()
    assert set(definitions.round_trip_cost_rate) == {0.004}


def test_time_exit_uses_ticker_row_and_fixed_cost() -> None:
    prices = series()
    prices["adj_close"][40] = 110.0
    prices["adj_high"][40] = 110.5
    result = simulate_one(entry(), "time_20_close", prices)
    assert result["exit_evaluation_available"]
    assert result["holding_sessions"] == 20
    assert np.isclose(result["gross_return"], 0.10)
    assert np.isclose(result["net_return"], 0.096)
    assert 0 < result["profit_capture_ratio"] <= 1


def test_profit_capture_is_missing_for_a_losing_trade() -> None:
    prices = series()
    prices["adj_close"][40] = 90.0
    result = simulate_one(entry(), "time_20_close", prices)
    assert result["gross_return"] < 0
    assert np.isnan(result["profit_capture_ratio"])


def test_close_below_ma20_uses_only_trailing_adjusted_closes() -> None:
    prices = series()
    prices["adj_close"][21] = 90.0
    prices["adj_high"][21] = 91.0
    prices["adj_low"][21] = 89.0
    result = simulate_one(entry(), "close_below_ma20", prices)
    assert result["exit_reason"] == "first_close_below_ma20"
    assert result["exit_session_position"] == 21
    assert result["exit_price"] == 90.0


def test_atr_gap_uses_open() -> None:
    prices = series()
    prices["adj_open"][21] = 96.0
    prices["adj_high"][21] = 98.0
    prices["adj_low"][21] = 95.0
    result = simulate_one(entry(), "atr3_trailing", prices)
    assert result["exit_reason"] == "atr_stop_gap_at_open"
    assert result["exit_price"] == 96.0
    assert not result["intraday_sequence_ambiguous"]


def test_atr_same_day_unknown_order_uses_low_first_conservative_order() -> None:
    prices = series()
    prices["adj_open"][21] = 100.0
    prices["adj_high"][21] = 110.0
    prices["adj_low"][21] = 100.0
    result = simulate_one(entry(), "atr3_trailing", prices)
    assert result["intraday_sequence_ambiguous"]
    assert result["same_day_conservative_assumption_used"]
    assert result["exit_session_position"] == 22
    assert result["exit_reason"] == "atr_stop_gap_at_open"
    assert result["exit_price"] == 100.0


def test_open_entry_mfe_includes_entry_day_but_close_entry_does_not() -> None:
    prices = series()
    prices["adj_high"][20] = 110.0
    close_result = simulate_one(entry(), "time_20_close", prices)
    open_result = simulate_one(
        entry(entry_definition="next_session_open"), "time_20_close", prices
    )
    assert np.isclose(close_result["mfe"], 0.01)
    assert np.isclose(open_result["mfe"], 0.10)


def test_period_boundary_is_excluded_before_values_are_used() -> None:
    prices = series("2023-10-02", 130)
    result = simulate_one(
        entry(
            anchor_date=pd.Timestamp(prices["date"][50]),
            entry_date=pd.Timestamp(prices["date"][50]),
            entry_session_position=50,
        ),
        "time_60_close",
        prices,
    )
    assert not result["exit_evaluation_available"]
    assert result["exclusion_reason"] == "exit_window_crosses_period_boundary"
    assert np.isnan(result["exit_price"])


def test_missing_ma_input_is_not_imputed() -> None:
    prices = series()
    prices["adj_close"][19] = np.nan
    result = simulate_one(entry(), "close_below_ma20", prices)
    assert not result["exit_evaluation_available"]
    assert result["exclusion_reason"] == "ma20_window_missing_or_invalid"
