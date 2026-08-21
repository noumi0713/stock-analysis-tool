from __future__ import annotations

from datetime import date

import pandas as pd

from scripts.export_close_signals import MAX_SIGNALS, select_latest_signals


def _candidate(ticker: str, volume_ratio: float, *, signal_date: date) -> dict:
    return {
        "ticker": ticker,
        "date": signal_date,
        "RSI": 30.0,
        "return_1d": -0.01,
        "return_5d": -0.05,
        "volume_ratio_1_20": volume_ratio,
        "trading_value": 500_000_000.0,
        "ATR": 0.05,
        "_close": 900.0,
        "_open": 880.0,
        "ma25": 1_000.0,
    }


def test_select_latest_signals_keeps_volume_ratio_top_three() -> None:
    signal_date = date(2026, 8, 21)
    frame = pd.DataFrame(
        [
            _candidate("1001.T", 1.6, signal_date=signal_date),
            _candidate("1002.T", 2.1, signal_date=signal_date),
            _candidate("1003.T", 1.8, signal_date=signal_date),
            _candidate("1004.T", 3.0, signal_date=signal_date),
        ]
    )

    selected = select_latest_signals(frame)

    assert len(selected) == MAX_SIGNALS
    assert selected["ticker"].tolist() == ["1004.T", "1002.T", "1003.T"]


def test_select_latest_signals_never_reuses_previous_day() -> None:
    frame = pd.DataFrame(
        [
            _candidate("1001.T", 2.0, signal_date=date(2026, 8, 20)),
            {
                **_candidate("1002.T", 2.0, signal_date=date(2026, 8, 21)),
                "RSI": 50.0,
            },
        ]
    )

    selected = select_latest_signals(frame)

    assert selected.empty
