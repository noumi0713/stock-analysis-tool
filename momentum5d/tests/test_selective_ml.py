from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.yahoo.selective_ml import tune_and_select_ml_strategy


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
    assert len(diagnostics["development_folds"]) == 3
    assert all(fold["selected_signals"] >= 5 for fold in diagnostics["development_folds"])
    assert len(diagnostics["validation_folds"]) == 3
    assert len(validation_trades) == 60
    assert validation_trades["rise_trade_target_hit"].mean() >= 0.60
