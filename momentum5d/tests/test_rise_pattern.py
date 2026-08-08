from __future__ import annotations

import pandas as pd
import pytest

from app.yahoo.rise_pattern import (
    RisePatternConfig,
    _attach_trade_outcomes,
    _stopped_trade_outcome,
)


def _bars(
    *,
    opens: list[float],
    highs: list[float],
    lows: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame([opens]),
        pd.DataFrame([highs]),
        pd.DataFrame([lows]),
    )


def test_same_day_stop_and_target_uses_stop_first() -> None:
    future_open, future_high, future_low = _bars(
        opens=[100.0],
        highs=[106.0],
        lows=[96.0],
    )

    result = _stopped_trade_outcome(
        pd.Series([100.0]),
        future_open,
        future_high,
        future_low,
        pd.Series([102.0]),
        pd.Series([True]),
        pd.Series([-0.03]),
    )

    assert result["gross_return"].iloc[0] == pytest.approx(-0.03)
    assert bool(result["stop_hit"].iloc[0])
    assert not bool(result["target_hit"].iloc[0])
    assert bool(result["ambiguous_both_hit"].iloc[0])


def test_gap_through_stop_exits_at_open() -> None:
    future_open, future_high, future_low = _bars(
        opens=[95.0],
        highs=[98.0],
        lows=[94.0],
    )

    result = _stopped_trade_outcome(
        pd.Series([100.0]),
        future_open,
        future_high,
        future_low,
        pd.Series([97.0]),
        pd.Series([True]),
        pd.Series([-0.03]),
    )

    assert result["gross_return"].iloc[0] == pytest.approx(-0.05)
    assert bool(result["stop_hit"].iloc[0])


def test_target_hit_before_later_stop_keeps_profit() -> None:
    future_open, future_high, future_low = _bars(
        opens=[100.0, 101.0],
        highs=[106.0, 102.0],
        lows=[99.0, 95.0],
    )

    result = _stopped_trade_outcome(
        pd.Series([100.0]),
        future_open,
        future_high,
        future_low,
        pd.Series([96.0]),
        pd.Series([True]),
        pd.Series([-0.03]),
    )

    assert result["gross_return"].iloc[0] == pytest.approx(0.05)
    assert bool(result["target_hit"].iloc[0])
    assert not bool(result["stop_hit"].iloc[0])


def test_trade_outcomes_include_full_stop_grid() -> None:
    features = pd.DataFrame(
        {
            "ticker": ["1111.T"] * 3,
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 106.0, 102.0],
            "low": [99.0, 96.0, 94.0],
            "close": [100.0, 101.0, 98.0],
            "adjusted_close": [100.0, 101.0, 98.0],
            "atr_14": [4.0, 4.0, 4.0],
        }
    )
    config = RisePatternConfig(horizon_days=2)

    result = _attach_trade_outcomes(features, config)

    for key in ("fixed_3pct", "fixed_4pct", "fixed_5pct"):
        assert f"rise_trade_{key}_net_return" in result
    for key in ("atr_0_5x", "atr_1x", "atr_1_5x", "atr_2x"):
        assert f"rise_trade_{key}_net_return" in result
