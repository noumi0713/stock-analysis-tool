from __future__ import annotations

from datetime import date

import pandas as pd

from scripts.export_close_signals import MAX_SIGNALS, build_payload, select_latest_signals


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


def test_select_latest_signals_rejects_retired_condition_ranges() -> None:
    signal_date = date(2026, 8, 24)
    frame = pd.DataFrame(
        [
            _candidate("1001.T", 2.0, signal_date=signal_date),
            {
                **_candidate("1002.T", 2.0, signal_date=signal_date),
                "return_5d": -0.14,
            },
            {
                **_candidate("1003.T", 2.0, signal_date=signal_date),
                "return_5d": 0.0,
            },
            {
                **_candidate("1004.T", 2.0, signal_date=signal_date),
                "ATR": 0.10,
            },
        ]
    )

    selected = select_latest_signals(frame)

    assert selected["ticker"].tolist() == ["1001.T"]


def test_payload_publishes_stable_score_rules() -> None:
    rows = []
    for day in pd.date_range("2026-07-01", periods=40, freq="B"):
        close = 1_000.0
        rows.append(
            {
                "ticker": "1001.T",
                "date": day.date(),
                "open": close - 5,
                "high": close + 10,
                "low": close - 10,
                "close": close,
                "adjusted_close": close,
                "volume": 500_000,
            }
        )
    prices = pd.DataFrame(rows)
    payload = build_payload(prices, generated_at="2026-08-24T08:00:00+00:00")
    conditions = payload["signal_model"]["conditions"]

    assert payload["signal_model"]["key"] == "rsi14_stable_score_10d_v1"
    assert conditions["return_5d_min"] == -0.12
    assert conditions["return_5d_max"] == -0.05
    assert conditions["atr_14_pct_min"] == 0.005
    assert conditions["atr_14_pct_max"] == 0.08
    assert conditions["take_profit_pct"] == 0.235
    assert conditions["stop_loss_pct"] == -0.22
