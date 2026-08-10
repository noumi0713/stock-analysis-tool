from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.yahoo.demand_supply import (
    DemandSupplyConfig,
    _simulate_outcomes,
    add_supply_pressure_features,
)


def test_supply_pressure_requires_heavy_bearish_turnover() -> None:
    frame = pd.DataFrame(
        {
            "observed_volume_ratio_rank": [0.95, 0.95],
            "observed_turnover_ratio_rank": [0.95, 0.95],
            "observed_volume_intensity_score": [0.90, 0.90],
            "observed_turnover_intensity_score": [0.90, 0.90],
            "observed_inflow_score": [0.20, 0.75],
            "return_1d": [-0.04, 0.04],
            "intraday_return": [-0.03, 0.03],
            "up_volume_share_10d": [0.30, 0.70],
        }
    )

    result = add_supply_pressure_features(frame)

    assert result.loc[0, "observed_supply_pressure_score"] > 0.75
    assert bool(result.loc[0, "observed_supply_pressure_confirmed"])
    assert not bool(result.loc[1, "observed_supply_pressure_confirmed"])
    assert result.loc[0, "observed_demand_supply_balance"] < 0


def test_supply_signal_exits_at_next_open_without_lookahead() -> None:
    rows = 8
    dates = [date(2026, 1, 1) + timedelta(days=offset) for offset in range(rows)]
    frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["TEST.T"] * rows,
            "open": [100, 100, 98, 100, 100, 100, 100, 100],
            "high": [101, 102, 105, 101, 101, 101, 101, 101],
            "low": [99, 99, 98, 99, 99, 99, 99, 99],
            "close": [100, 101, 104, 100, 100, 100, 100, 100],
            "adjusted_close": [100, 101, 104, 100, 100, 100, 100, 100],
            "observed_supply_pressure_score": [0.0, 0.80, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "observed_volume_ratio_rank": [0.0, 0.90, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "observed_turnover_ratio_rank": [0.0, 0.90, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "return_1d": [0.0, -0.02, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0],
            "intraday_return": [0.0, -0.01, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
            "up_volume_share_10d": [0.5, 0.40, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5],
        }
    )
    config = DemandSupplyConfig()

    fixed = _simulate_outcomes(frame, config, supply_threshold=None)
    supply = _simulate_outcomes(
        frame,
        config,
        supply_threshold=0.55,
        use_fixed_stop=False,
    )

    assert np.isclose(fixed.loc[0, "net_return"], 0.038)
    assert bool(fixed.loc[0, "target_hit"])
    assert np.isclose(supply.loc[0, "net_return"], -0.022)
    assert bool(supply.loc[0, "supply_exit"])
    assert supply.loc[0, "holding_days"] == 1
