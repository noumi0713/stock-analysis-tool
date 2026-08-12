from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.yahoo.perfect_order import (
    PerfectOrderBacktestConfig,
    backtest_perfect_order_pullbacks,
)


def _perfect_order_features(*, stop_on_entry_day: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = date(2026, 1, 1)
    closes = [100.0 + 0.5 * index for index in range(130)]
    closes[85:91] = [141.0, 140.5, 140.0, 142.0, 146.0, 147.0]
    for index, close in enumerate(closes):
        open_ = close
        high = close + 1.0
        low = close - 1.0
        rsi = 60.0
        if index == 88:
            open_ = 140.0
            high = 143.0
            low = 135.0 if stop_on_entry_day else 139.0
            close = 142.0
        elif index == 89:
            open_ = 142.0
            high = 147.0
            low = 141.0
            close = 146.0
            rsi = 75.0
        elif index == 90:
            open_ = 147.0
            high = 148.0
            low = 146.0
            close = 147.0
            rsi = 75.0
        rows.append(
            {
                "date": start + timedelta(days=index),
                "ticker": "1111.T",
                "code": "1111",
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "adjusted_close": close,
                "volume": 2_000_000.0,
                "turnover_value": close * 2_000_000.0,
                "rsi_14": rsi,
                "volume_ratio_1_20": 1.3,
            }
        )
    frame = pd.DataFrame(rows)
    frame["return_1d"] = frame.groupby("ticker")["close"].pct_change()
    return frame


def _config() -> PerfectOrderBacktestConfig:
    return PerfectOrderBacktestConfig(
        minimum_calibration_trades=1,
        calibration_fraction=0.9,
    )


def test_three_day_pullback_exits_next_open_after_overheat() -> None:
    report = backtest_perfect_order_pullbacks(
        _perfect_order_features(),
        _config(),
    )

    result = report["strategies"]["three_day_pullback__rsi_70"]["all"]
    assert result["trades"] >= 1
    assert result["overheat_exit_rate"] > 0
    assert result["five_pct_reach_rate"] > 0
    example = next(item for item in report["selected_trade_examples"] if item["ticker"] == "1111.T")
    assert example["exit_day"] >= 3
    assert report["selected_strategy"] is not None
    risk_filter = report["risk_filter_backtest"]
    assert risk_filter["filter"]["maximum_allowed_flags"] == 1
    assert len(risk_filter["filter"]["conditions"]) == 7
    assert risk_filter["filtered_selected_strategy"] is not None


def test_entry_day_stop_takes_priority_over_later_overheat() -> None:
    report = backtest_perfect_order_pullbacks(
        _perfect_order_features(stop_on_entry_day=True),
        _config(),
    )

    result = report["strategies"]["three_day_pullback__rsi_70"]["all"]
    assert result["trades"] >= 1
    assert result["stop_hit_rate"] > 0
    assert result["mean_net_return"] < 0


def test_empty_input_returns_empty_study() -> None:
    report = backtest_perfect_order_pullbacks(pd.DataFrame(), _config())

    assert report["strategies"] == {}
    assert report["selected_strategy"] is None
    assert report["selected_validation"] is None
    assert report["error"] == "評価可能なデータがありません"
