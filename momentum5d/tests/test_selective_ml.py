from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.yahoo.rise_pattern import (
    RisePatternConfig,
    calculate_ten_day_limit_order_study,
)
from app.yahoo.selective_ml import (
    evaluate_frozen_ml_strategy,
    score_latest_strong_shape_candidates,
    score_latest_ten_day_candidates,
    tune_and_select_ml_strategy,
)


def test_latest_strong_shape_signal_uses_frozen_rule_and_top_one() -> None:
    dates = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(542)]
    shapes = ("sharp_selloff", "capitulation_reversal", "rounded_base")
    rows = []
    for index, value in enumerate(dates[:540]):
        target_hit = index % 4 != 0
        rows.append(
            {
                "date": value,
                "ticker": f"{1000 + index}.T",
                "_rise_shape": shapes[index % len(shapes)],
                "rise_pattern_live_bottom": True,
                "turnover_value": 300_000_000.0,
                "rsi_14": 45.0,
                "return_1d": 0.0,
                "trade_outcome_available": True,
                "rise_trade_entry_gap_return": 0.0,
                "rise_trade_target_hit": target_hit,
                "rise_trade_down_5pct": index % 10 == 0,
                "rise_trade_down_8pct": index % 20 == 0,
                "rise_trade_net_return": 0.048 if target_hit else -0.02,
            }
        )
    for rank in range(2):
        rows.append(
            {
                "date": dates[-1],
                "ticker": f"900{rank}.T",
                "_rise_shape": ("rounded_base", "sharp_selloff")[rank],
                "rise_pattern_live_bottom": True,
                "turnover_value": 400_000_000.0 - rank * 50_000_000,
                "rsi_14": 40.0,
                "return_1d": 0.0,
                "trade_outcome_available": False,
                "rise_trade_entry_gap_return": float("nan"),
                "rise_trade_target_hit": False,
                "rise_trade_down_5pct": False,
                "rise_trade_down_8pct": False,
                "rise_trade_net_return": float("nan"),
            }
        )

    result = score_latest_strong_shape_candidates(pd.DataFrame(rows))

    assert len(result) == 2
    assert int(result["ml_sharp_signal"].sum()) == 1
    assert result.loc[result["ml_sharp_signal"], "ml_sharp_rank"].iloc[0] == 1
    assert result["ml_sharp_model_samples"].max() == 180
    assert result.loc[result["ml_sharp_signal"], "ml_sharp_probability"].iloc[0] >= 0.55
    assert result["ml_sharp_entry_rule"].str.contains(r"\+3%").all()


def test_latest_ten_day_signal_uses_development_parameters_and_top_one() -> None:
    dates = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(542)]
    shapes = ("sharp_selloff", "capitulation_reversal", "rounded_base")
    rows = []
    for index, value in enumerate(dates[:540]):
        target_hit = index % 4 != 0
        rows.append(
            {
                "date": value,
                "ticker": f"{1000 + index}.T",
                "_rise_shape": shapes[index % len(shapes)],
                "rise_pattern_live_bottom": True,
                "turnover_value": 300_000_000.0,
                "rsi_14": 45.0,
                "return_1d": 0.0,
                "_rise_market_breadth_5d": 0.60,
                "_rise_market_median_return_20d": 0.01,
                "trade_outcome_available": True,
                "rise_trade_entry_gap_return": 0.0,
                "rise_trade_target_hit": target_hit,
                "rise_trade_down_5pct": index % 10 == 0,
                "rise_trade_down_8pct": index % 20 == 0,
                "rise_trade_net_return": 0.048 if target_hit else -0.02,
            }
        )
    for rank in range(2):
        rows.append(
            {
                "date": dates[-1],
                "ticker": f"900{rank}.T",
                "_rise_shape": ("rounded_base", "sharp_selloff")[rank],
                "rise_pattern_live_bottom": True,
                "turnover_value": 400_000_000.0,
                "rsi_14": 40.0,
                "return_1d": 0.0,
                "_rise_market_breadth_5d": 0.60,
                "_rise_market_median_return_20d": 0.01,
                "trade_outcome_available": False,
                "rise_trade_entry_gap_return": float("nan"),
                "rise_trade_target_hit": False,
                "rise_trade_down_5pct": False,
                "rise_trade_down_8pct": False,
                "rise_trade_net_return": float("nan"),
            }
        )
    parameters = {
        "allowed_shapes": [],
        "model": "hist_gradient_boosting",
        "probability_threshold": 0.55,
        "max_gap_up": 0.02,
        "max_down_5pct_probability": 0.50,
        "max_down_8pct_probability": 0.30,
        "min_expected_net_return": -0.01,
        "min_market_breadth_5d": None,
        "min_market_median_return_20d": None,
    }

    result = score_latest_ten_day_candidates(pd.DataFrame(rows), parameters)

    assert len(result) == 2
    assert int(result["ml_ten_day_signal"].sum()) == 1
    assert result.loc[result["ml_ten_day_signal"], "ml_ten_day_rank"].iloc[0] == 1
    assert result.loc[result["ml_ten_day_signal"], "ml_ten_day_probability"].iloc[0] >= 0.55
    assert result["ml_ten_day_entry_rule"].str.contains(r"\+2%").all()


def test_tuning_freezes_development_rule_for_validation() -> None:
    dates = [date(2026, 1, 1) + timedelta(days=offset) for offset in range(120)]
    rows = []
    for day_index, value in enumerate(dates):
        for rank in range(2):
            favorable = rank == 0
            target_hit = favorable and day_index % 10 < 7
            net_return = 0.048 if target_hit else -0.02
            rows.append(
                {
                    "date": value,
                    "_rise_shape": "sharp_selloff" if favorable else "rounded_base",
                    "_rise_market_breadth_5d": 0.62,
                    "_rise_market_median_return_20d": 0.01,
                    "rise_trade_entry_gap_return": 0.0,
                    "rise_trade_target_hit": target_hit,
                    "rise_trade_net_return": net_return,
                    "ml_logistic_target_probability": 0.72 if favorable else 0.42,
                    "ml_logistic_down_5pct_probability": 0.10 if favorable else 0.40,
                    "ml_logistic_down_8pct_probability": 0.05 if favorable else 0.20,
                    "ml_logistic_expected_net_return": 0.02 if favorable else -0.01,
                    "ml_hist_gradient_boosting_target_probability": (0.70 if favorable else 0.40),
                    "ml_hist_gradient_boosting_down_5pct_probability": (
                        0.12 if favorable else 0.42
                    ),
                    "ml_hist_gradient_boosting_down_8pct_probability": (
                        0.06 if favorable else 0.22
                    ),
                    "ml_hist_gradient_boosting_expected_net_return": (
                        0.018 if favorable else -0.012
                    ),
                }
            )
    scored = pd.DataFrame(rows)

    validation_trades, diagnostics = tune_and_select_ml_strategy(
        scored,
        dates,
        minimum_development_signals=40,
    )

    assert diagnostics["status"] == "completed"
    assert diagnostics["validation_goal_met"]
    assert diagnostics["chosen_parameters"]["top_n_per_day"] >= 1
    assert diagnostics["chosen_parameters"]["shape_profile"] in {
        "all_strong_shapes",
        "sharp_selloff",
    }
    assert len(diagnostics["development_folds"]) == 3
    assert all(fold["selected_signals"] >= 5 for fold in diagnostics["development_folds"])
    assert len(diagnostics["validation_folds"]) == 3
    assert diagnostics["sharp_selloff_candidate"]["validation"]["selected_signals"] == 60
    assert (
        diagnostics["sharp_selloff_candidate"]["latest_period_confirmation"]["selected_signals"]
        == 20
    )
    assert diagnostics["sharp_selloff_candidate"]["historical_60pct_candidate_met"]
    assert len(validation_trades) == 60
    assert validation_trades["rise_trade_target_hit"].mean() >= 0.60


def test_tuning_can_lock_capitulation_reversal_only() -> None:
    dates = [date(2026, 1, 1) + timedelta(days=offset) for offset in range(80)]
    rows = []
    for day_index, value in enumerate(dates):
        for shape in ("capitulation_reversal", "sharp_selloff"):
            favorable = shape == "capitulation_reversal"
            target_hit = favorable and day_index % 10 < 7
            net_return = 0.048 if target_hit else -0.02
            rows.append(
                {
                    "date": value,
                    "_rise_shape": shape,
                    "_rise_market_breadth_5d": 0.62,
                    "_rise_market_median_return_20d": 0.01,
                    "rise_trade_entry_gap_return": 0.0,
                    "rise_trade_target_hit": target_hit,
                    "rise_trade_net_return": net_return,
                    "ml_logistic_target_probability": 0.72 if favorable else 0.42,
                    "ml_logistic_down_5pct_probability": 0.10 if favorable else 0.40,
                    "ml_logistic_down_8pct_probability": 0.05 if favorable else 0.20,
                    "ml_logistic_expected_net_return": 0.02 if favorable else -0.01,
                    "ml_hist_gradient_boosting_target_probability": (
                        0.70 if favorable else 0.40
                    ),
                    "ml_hist_gradient_boosting_down_5pct_probability": (
                        0.12 if favorable else 0.42
                    ),
                    "ml_hist_gradient_boosting_down_8pct_probability": (
                        0.06 if favorable else 0.22
                    ),
                    "ml_hist_gradient_boosting_expected_net_return": (
                        0.018 if favorable else -0.012
                    ),
                }
            )

    validation_trades, diagnostics = tune_and_select_ml_strategy(
        pd.DataFrame(rows),
        dates,
        minimum_development_signals=20,
        probability_thresholds=(0.60,),
        gap_limits=(0.03,),
        top_n_options=(1,),
        allowed_shape_profiles=("capitulation_reversal",),
    )

    assert diagnostics["status"] == "completed"
    assert diagnostics["allowed_shape_profiles"] == ["capitulation_reversal"]
    assert diagnostics["chosen_parameters"]["shape_profile"] == "capitulation_reversal"
    assert diagnostics["chosen_parameters"]["allowed_shapes"] == [
        "capitulation_reversal"
    ]
    assert set(validation_trades["_rise_shape"]) == {"capitulation_reversal"}
    assert diagnostics["validation"]["median_trade_net_return"] is not None
    assert diagnostics["validation"]["worst_trade_net_return"] is not None


def test_frozen_strategy_evaluates_explicit_dates_without_retuning() -> None:
    dates = [date(2026, 4, 1) + timedelta(days=offset) for offset in range(6)]
    rows = []
    for value in dates:
        for rank in range(2):
            selected = rank == 0
            rows.append(
                {
                    "date": value,
                    "_rise_shape": "capitulation_reversal",
                    "_rise_market_breadth_5d": 0.60,
                    "_rise_market_median_return_20d": 0.01,
                    "rise_trade_entry_gap_return": 0.0,
                    "rise_trade_target_hit": selected,
                    "rise_trade_net_return": 0.048 if selected else -0.02,
                    "ml_logistic_target_probability": 0.70 if selected else 0.40,
                    "ml_logistic_down_5pct_probability": 0.10,
                    "ml_logistic_down_8pct_probability": 0.05,
                    "ml_logistic_expected_net_return": 0.02,
                }
            )
    parameters = {
        "allowed_shapes": ["capitulation_reversal"],
        "model": "logistic",
        "probability_threshold": 0.55,
        "max_gap_up": 0.03,
        "max_down_5pct_probability": 0.50,
        "max_down_8pct_probability": 0.30,
        "min_expected_net_return": -0.01,
        "top_n_per_day": 1,
    }

    trades, diagnostics = evaluate_frozen_ml_strategy(
        pd.DataFrame(rows),
        dates[1:5],
        parameters,
    )

    assert len(trades) == 4
    assert set(trades["date"]) == set(dates[1:5])
    assert diagnostics["validation"]["target_hit_rate"] == 1.0
    assert len(diagnostics["validation_folds"]) == 3


def test_limit_order_study_uses_conservative_intraday_fill_rule() -> None:
    dates = [date(2026, 6, 1) + timedelta(days=offset) for offset in range(12)]
    rows = []
    closes = [100.0] * 12
    for index, value in enumerate(dates):
        rows.append(
            {
                "ticker": "1000.T",
                "date": value,
                "open": 101.0 if index == 1 else closes[index],
                "high": 106.0 if index == 1 else closes[index],
                "low": 99.0 if index == 1 else closes[index],
                "close": closes[index],
                "adjusted_close": closes[index],
            }
        )
    validation = pd.DataFrame([{"ticker": "1000.T", "date": dates[0]}])

    result = calculate_ten_day_limit_order_study(
        pd.DataFrame(rows),
        validation,
        RisePatternConfig(horizon_days=10),
        minimum_turnover=150_000_000.0,
    )

    flat_limit = result["levels"][0]
    assert flat_limit["filled_orders"] == 1
    assert flat_limit["open_fill_rate"] == 0.0
    assert flat_limit["target_hit_rate"] == 0.0
    assert flat_limit["mean_filled_trade_net_return"] == -0.002
