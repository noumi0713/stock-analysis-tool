from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.live_backtest import EXECUTION_ENGINE_ID, simulate_portfolio
from app.live_strategy import load_frozen_strategy


def _trade(ticker: str, rank: int, *, net_return: float) -> dict:
    entry = 1_000.0
    return {
        "signal_type": "first_pullback",
        "signal_date": date(2026, 1, 2),
        "entry_date": date(2026, 1, 5),
        "exit_date": date(2026, 1, 6),
        "ticker": ticker,
        "candidate_rank": rank,
        "signal_close": entry,
        "entry_price": entry,
        "exit_price": entry * (1 + net_return),
        "gross_return": net_return,
        "net_return": net_return - 0.002,
        "exit_reason": "time",
        "holding_sessions": 2,
    }


def test_portfolio_obeys_maximum_new_positions_per_day() -> None:
    paths = pd.DataFrame(
        [
            _trade("1111.T", 1, net_return=0.05),
            _trade("2222.T", 2, net_return=-0.02),
            _trade("3333.T", 3, net_return=0.10),
        ]
    )
    prices = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "date": day,
                "open": 1_000.0,
                "high": 1_000.0,
                "low": 1_000.0,
                "close": 1_000.0,
                "adjusted_close": 1_000.0,
                "volume": 1_000,
                "stock_splits": 0.0,
            }
            for ticker in ("1111.T", "2222.T", "3333.T")
            for day in ("2026-01-05", "2026-01-06")
        ]
    )

    trades, _, summary = simulate_portfolio(paths, prices)

    assert trades["ticker"].tolist() == ["1111.T", "2222.T"]
    assert summary["completed_trades"] == 2
    assert summary["execution_engine_id"] == EXECUTION_ENGINE_ID
    assert summary["ending_equity_yen"] < 2_100_000


def test_same_rank_order_is_deterministic() -> None:
    paths = pd.DataFrame(
        [_trade("2222.T", 1, net_return=0.0), _trade("1111.T", 1, net_return=0.0)]
    )
    prices = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "date": day,
                "open": 1_000.0,
                "high": 1_000.0,
                "low": 1_000.0,
                "close": 1_000.0,
                "adjusted_close": 1_000.0,
                "volume": 1_000,
                "stock_splits": 0.0,
            }
            for ticker in ("1111.T", "2222.T")
            for day in ("2026-01-05", "2026-01-06")
        ]
    )

    trades, _, _ = simulate_portfolio(paths, prices)

    assert trades["ticker"].tolist() == ["1111.T", "2222.T"]
    assert trades["net_profit_yen"].sum() == pytest.approx(-2_400.0, abs=20.0)


def test_backtest_rejects_runtime_strategy_parameter_change() -> None:
    strategy = load_frozen_strategy()
    strategy["signals"]["first_pullback"]["execution"]["holding_days"] = 99

    with pytest.raises(ValueError, match="differs from the frozen"):
        simulate_portfolio(pd.DataFrame(), pd.DataFrame(), strategy=strategy)
