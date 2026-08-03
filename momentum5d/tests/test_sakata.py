from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.yahoo.sakata import add_sakata_features


def _frame(candles: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    start = date(2026, 1, 5)
    return pd.DataFrame(
        [
            {
                "date": start + timedelta(days=index),
                "ticker": "1111.T",
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
            }
            for index, (open_, high, low, close) in enumerate(candles)
        ]
    )


def test_detects_morning_star_as_sakata_buy_signal() -> None:
    candles = [
        (112, 113, 110, 111),
        (111, 112, 108, 109),
        (109, 110, 106, 107),
        (107, 108, 104, 105),
        (105, 106, 102, 103),
        (103, 104, 100, 101),
        (101, 102, 98, 99),
        (100, 101, 94, 95),
        (94.5, 95.0, 93.5, 94.3),
        (94.5, 100.5, 94.0, 100.0),
    ]

    latest = add_sakata_features(_frame(candles)).iloc[-1]

    assert bool(latest["sakata_morning_star"])
    assert bool(latest["sakata_buy_signal"])
    assert latest["sakata_pattern"] == "三川明けの明星"


def test_detects_rising_three_methods_and_bearish_counterpart() -> None:
    rising = [
        (100, 111, 99, 110),
        (108, 109, 105, 106),
        (107, 108, 103, 105),
        (106, 108, 104, 107),
        (107, 113, 106, 112),
    ]
    falling = [
        (110, 111, 99, 100),
        (102, 105, 101, 104),
        (103, 107, 102, 105),
        (104, 106, 102, 103),
        (103, 104, 97, 98),
    ]

    bullish = add_sakata_features(_frame(rising)).iloc[-1]
    bearish = add_sakata_features(_frame(falling)).iloc[-1]

    assert bool(bullish["sakata_rising_three_methods"])
    assert bullish["sakata_score"] == 1.0
    assert bool(bearish["sakata_falling_three_methods"])
    assert bearish["sakata_score"] == 0.0


def test_sakata_signal_does_not_change_when_future_candles_are_appended() -> None:
    candles = [(100, 101, 99, 100)] * 9 + [
        (100, 103, 99.8, 102),
        (101, 105, 100.8, 104),
        (103, 107, 102.8, 106),
    ]
    base = add_sakata_features(_frame(candles))
    extended = add_sakata_features(
        _frame(candles + [(80, 85, 75, 82), (120, 125, 115, 123)])
    )

    columns = [column for column in base if column.startswith("sakata_")]
    pd.testing.assert_frame_equal(
        base[columns].reset_index(drop=True),
        extended.iloc[: len(base)][columns].reset_index(drop=True),
    )
