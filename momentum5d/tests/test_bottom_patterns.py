from __future__ import annotations

import pandas as pd

from app.yahoo.bottom_patterns import analyze_bottom_patterns


def _ticker_rows(ticker: str, code: str, *, succeeds: bool) -> list[dict[str, object]]:
    dates = pd.date_range("2025-01-02", periods=50, freq="B")
    rows: list[dict[str, object]] = []
    for index, day in enumerate(dates):
        close = 100.0 + index * 0.05
        low = close - 1.0
        high = close + 1.0
        open_ = close - 0.2
        if index == 30:
            open_, high, low, close = 91.0, 92.0, 88.0, 89.0
        elif 31 <= index <= 35 and not succeeds:
            open_, high, low, close = 90.0, 91.5, 89.5, 90.5
        elif index in (28, 32):
            open_, high, low, close = 96.0, 97.0, 94.0, 95.0
        elif index in (29, 31):
            open_, high, low, close = 94.0, 95.0, 92.0, 93.0
        rows.append(
            {
                "date": day.date(),
                "ticker": ticker,
                "code": code,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "adjusted_close": close,
                "return_5d": -0.08 if index == 30 else 0.0,
                "return_20d": -0.12 if index == 30 else 0.0,
                "volume_ratio_5_20": 1.0,
                "volatility_10d": 0.03,
                "range_width_10d": 0.12,
            }
        )
    return rows


def test_bottom_pattern_study_counts_successes_and_similar_failures() -> None:
    rows = [
        *_ticker_rows("1111.T", "11110", succeeds=True),
        *_ticker_rows("2222.T", "22220", succeeds=False),
    ]

    summary, events = analyze_bottom_patterns(pd.DataFrame(rows))

    studied = events.loc[events["shape"] == "sharp_selloff"]
    assert len(studied) == 2
    assert studied["target_5pct"].sum() == 1
    assert summary["events"] == 2
    assert summary["successes"] == 1
    assert summary["failures"] == 1
    assert summary["overall_success_rate"] == 0.5
    ranking = next(row for row in summary["rankings"] if row["shape"] == "sharp_selloff")
    assert ranking["samples"] == 2
    assert ranking["successes"] == 1
    assert ranking["failures"] == 1
    assert ranking["success_rate"] == 0.5
    assert 0 <= ranking["wilson_95_low"] < ranking["wilson_95_high"] <= 1
    assert ranking["success_examples"]
    assert ranking["failure_examples"]
    assert len(ranking["success_examples"][0]["pre_shape"]) == 20
