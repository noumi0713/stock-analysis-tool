from __future__ import annotations

import pandas as pd
import pytest

from app.yahoo.rise_pattern import (
    RisePatternConfig,
    _attach_trade_outcomes,
    _exit_trade_outcome,
    _score_selective_day,
    _select_selective_trades,
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
    assert "rise_trade_entry_gap_return" in result
    assert "rise_trade_future_min_return" in result
    assert "rise_trade_down_5pct" in result
    assert "rise_trade_down_8pct" in result


def test_time_exit_uses_requested_day_close() -> None:
    result = _exit_trade_outcome(
        pd.Series([100.0]),
        pd.DataFrame([[102.0, 104.0, 106.0]]),
        pd.DataFrame([[101.0, 103.0, 105.0]]),
        pd.Series([True]),
        target_return=0.05,
        exit_day=2,
    )

    assert result["gross_return"].iloc[0] == pytest.approx(0.03)
    assert not bool(result["target_hit"].iloc[0])
    assert bool(result["early_exit"].iloc[0])


def test_follow_through_exit_uses_close_after_target_check() -> None:
    result = _exit_trade_outcome(
        pd.Series([100.0, 100.0]),
        pd.DataFrame([[103.0, 104.0], [106.0, 102.0]]),
        pd.DataFrame([[99.0, 98.0], [99.0, 101.0]]),
        pd.Series([True, True]),
        target_return=0.05,
        exit_day=2,
        follow_through_day=1,
        minimum_close_return=0.0,
    )

    assert result["gross_return"].iloc[0] == pytest.approx(-0.01)
    assert bool(result["early_exit"].iloc[0])
    assert result["gross_return"].iloc[1] == pytest.approx(0.05)
    assert bool(result["target_hit"].iloc[1])
    assert not bool(result["early_exit"].iloc[1])


def test_trade_outcomes_include_exit_grid() -> None:
    features = pd.DataFrame(
        {
            "ticker": ["1111.T"] * 4,
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 98.0, 97.0, 96.0],
            "close": [100.0, 99.0, 101.0, 102.0],
            "adjusted_close": [100.0, 99.0, 101.0, 102.0],
            "atr_14": [4.0, 4.0, 4.0, 4.0],
        }
    )
    config = RisePatternConfig(horizon_days=3, exit_holding_days=(2,))

    result = _attach_trade_outcomes(features, config)

    for key in ("target_3pct", "target_4pct", "time_2d"):
        assert f"rise_trade_exit_{key}_net_return" in result
    for key in (
        "day1_min_0pct",
        "day1_min_1pct",
        "day1_min_2pct",
        "day2_min_0pct",
        "day2_min_1pct",
        "day2_min_2pct",
    ):
        assert f"rise_trade_exit_{key}_net_return" in result


def test_selective_model_uses_prior_live_outcomes_by_shape() -> None:
    samples = 50
    training = pd.DataFrame(
        {
            "trade_outcome_available": [True] * samples,
            "rise_trade_entry_gap_return": [0.0] * samples,
            "_rise_shape": ["sharp_selloff"] * samples,
            "_rise_feature_signature": ["11111"] * samples,
            "_rise_quality_score": [5] * samples,
            "rise_trade_target_hit": [True] * 40 + [False] * 10,
            "rise_trade_down_5pct": [False] * 47 + [True] * 3,
            "rise_trade_down_8pct": [False] * 49 + [True],
            "rise_trade_net_return": [0.048] * 40 + [-0.02] * 10,
        }
    )
    day = training.iloc[:1].drop(
        columns=[
            "trade_outcome_available",
            "rise_trade_entry_gap_return",
            "rise_trade_target_hit",
            "rise_trade_down_5pct",
            "rise_trade_down_8pct",
            "rise_trade_net_return",
        ]
    )
    config = RisePatternConfig(
        selective_min_samples=40,
        selective_prior_strength=10.0,
    )

    result = _score_selective_day(day, training, config)

    assert result["selective_70_probability"].iloc[0] == pytest.approx(0.80)
    assert result["selective_70_down_5pct_probability"].iloc[0] == pytest.approx(
        0.06
    )
    assert result["selective_70_expected_net_return"].iloc[0] > 0
    assert result["selective_70_samples"].iloc[0] == samples


def test_selective_selection_allows_zero_to_three_trades_per_day() -> None:
    scored = pd.DataFrame(
        {
            "date": ["2026-08-01"] * 5 + ["2026-08-02"],
            "rise_trade_entry_gap_return": [0.0] * 6,
            "selective_70_probability": [0.80, 0.78, 0.76, 0.74, 0.72, 0.60],
            "selective_70_down_5pct_probability": [0.10] * 6,
            "selective_70_down_8pct_probability": [0.05] * 6,
            "selective_70_expected_net_return": [0.02, 0.018, 0.016, 0.014, 0.012, 0.01],
            "selective_70_samples": [100] * 6,
        }
    )
    config = RisePatternConfig(selective_top_n=3)

    result = _select_selective_trades(
        scored,
        config,
        probability_threshold=0.70,
        top_n=3,
    )

    assert len(result) == 3
    assert result["date"].nunique() == 1
