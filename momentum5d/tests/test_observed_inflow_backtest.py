from __future__ import annotations

from datetime import date

import pandas as pd

from app.yahoo.sakata_backtest import SakataBacktestConfig, _select_candidates


def test_observed_inflow_strategy_selects_only_confirmed_signal() -> None:
    common = {
        "date": date(2026, 4, 1),
        "entry_date": date(2026, 4, 2),
        "exit_date": date(2026, 4, 8),
        "trade_available": True,
        "entry_gap": 0.005,
        "turnover_value": 200_000_000.0,
        "return_1d": 0.025,
        "return_5d": 0.06,
        "return_20d": 0.08,
        "intraday_return": 0.02,
        "rsi_14": 64.0,
        "volume_ratio_5_20": 1.30,
        "up_volume_share_10d": 0.62,
        "sakata_pattern": "該当なし",
        "sakata_score": 0.30,
        "legacy_setup_score": 0.40,
        "retail_flow_score": 0.70,
        "retail_discovery_score": 0.70,
        "retail_understanding_proxy_score": 0.60,
        "retail_expectation_score": 0.70,
        "retail_safety_score": 0.65,
        "retail_action_score": 0.75,
        "retail_overheat_penalty": 0.10,
        "retail_loss_anxiety_penalty": 0.05,
        "volume_ratio_1_20": 1.80,
        "turnover_ratio_1_20": 1.95,
        "observed_volume_ratio_rank": 0.90,
        "observed_turnover_ratio_rank": 0.92,
        "observed_price_confirmation_score": 0.75,
        "observed_inflow_score": 0.84,
        "sector_17_name": "電機・精密",
        "sector_17_trend_score": 0.70,
        "entry_price": 100.0,
        "exit_price": 105.0,
        "target_hit_day": 3,
        "target_hit": True,
        "gross_return": 0.05,
        "net_return": 0.048,
        "trade_win": True,
        "max_favorable_excursion": 0.06,
        "max_adverse_excursion": -0.02,
    }
    frame = pd.DataFrame(
        [
            {
                **common,
                "ticker": "1111.T",
                "code": "11110",
                "observed_inflow_confirmed": True,
            },
            {
                **common,
                "ticker": "2222.T",
                "code": "22220",
                "observed_inflow_score": 0.95,
                "observed_inflow_confirmed": False,
            },
        ]
    )

    selected = _select_candidates(
        frame,
        config=SakataBacktestConfig(top_n=10),
        start=date(2026, 4, 1),
        end=date(2026, 4, 1),
        strategy="observed_inflow",
    )

    assert selected["ticker"].tolist() == ["1111.T"]
    assert selected.iloc[0]["ranking_score"] == 0.84
    assert bool(selected.iloc[0]["observed_inflow_confirmed"])
    assert selected.iloc[0]["return_1d"] == 0.025
