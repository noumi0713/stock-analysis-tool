from __future__ import annotations

from datetime import date

import pandas as pd

from scripts.export_close_signals import (
    MAX_NEAR_MISSES,
    MAX_SIGNALS,
    build_payload,
    select_latest_near_misses,
    select_latest_signals,
)


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


def test_select_latest_near_misses_keeps_exactly_seven_conditions() -> None:
    signal_date = date(2026, 8, 25)
    valid = _candidate("1001.T", 2.0, signal_date=signal_date)
    frame = pd.DataFrame(
        [
            valid,
            {**_candidate("1002.T", 2.0, signal_date=signal_date), "return_1d": 0.001},
            {**_candidate("1003.T", 1.4, signal_date=signal_date)},
            {**_candidate("1004.T", 2.0, signal_date=signal_date), "RSI": 36.0},
            {
                **_candidate("1005.T", 2.0, signal_date=signal_date),
                "return_1d": 0.01,
            },
            {
                **_candidate("1006.T", 2.0, signal_date=signal_date),
                "trading_value": 250_000_000.0,
            },
            {
                **_candidate("1007.T", 2.0, signal_date=signal_date),
                "_close": 1_100.0,
            },
            {
                **_candidate("1008.T", 1.4, signal_date=signal_date),
                "RSI": 36.0,
            },
        ]
    )

    selected = select_latest_near_misses(frame)

    assert len(selected) == MAX_NEAR_MISSES
    assert selected["ticker"].tolist() == [
        "1002.T",
        "1003.T",
        "1004.T",
        "1006.T",
        "1005.T",
    ]
    assert selected["_failed_condition"].tolist() == [
        "return_1d",
        "volume_ratio_1_20",
        "rsi",
        "trading_value",
        "return_1d",
    ]


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
    assert payload["near_miss_count"] == 0
    assert payload["near_misses"] == []


def test_payload_publishes_near_miss_reason(monkeypatch) -> None:
    signal_date = date(2026, 8, 25)
    indicators = pd.DataFrame(
        [_candidate("6707.T", 1.126, signal_date=signal_date)]
    )
    monkeypatch.setattr(
        "scripts.export_close_signals.calculate_indicators",
        lambda _prices: indicators,
    )

    payload = build_payload(pd.DataFrame(), names={"6707": "サンケン電気"})

    assert payload["signal_count"] == 0
    assert payload["near_miss_count"] == 1
    candidate = payload["near_misses"][0]
    assert candidate["code"] == "6707"
    assert candidate["name"] == "サンケン電気"
    assert candidate["passed_conditions"] == 7
    assert candidate["failed_condition"] == {
        "key": "volume_ratio_1_20",
        "label": "出来高比",
        "actual_value": 1.126,
        "actual_label": "1.13倍",
        "required_label": "1.5倍以上",
    }
