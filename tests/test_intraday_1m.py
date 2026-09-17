import pandas as pd

from swing_data.intraday_1m import (
    filter_session,
    morning_low_summary,
    normalize_symbol,
    parse_tickers,
)


def test_normalize_japan_symbols():
    assert normalize_symbol("7203") == "7203.T"
    assert normalize_symbol("130a") == "130A.T"
    assert normalize_symbol("7203.T") == "7203.T"
    assert normalize_symbol("AAPL") == "AAPL"


def test_parse_tickers_deduplicates_and_splits():
    parsed = parse_tickers(["7203, 6758", "7203 130A"])
    assert parsed == [
        ("7203", "7203.T"),
        ("6758", "6758.T"),
        ("130A", "130A.T"),
    ]


def _sample_frame():
    idx = pd.DatetimeIndex(
        [
            "2026-09-17 08:59:00+09:00",
            "2026-09-17 09:00:00+09:00",
            "2026-09-17 09:01:00+09:00",
            "2026-09-17 09:02:00+09:00",
            "2026-09-17 11:30:00+09:00",
            "2026-09-17 12:00:00+09:00",
            "2026-09-17 12:30:00+09:00",
            "2026-09-17 15:30:00+09:00",
            "2026-09-17 15:31:00+09:00",
        ]
    )
    return pd.DataFrame(
        {
            "Open": [100, 100, 99, 98, 101, 101, 102, 103, 104],
            "High": [101, 101, 100, 99, 102, 102, 103, 104, 105],
            "Low": [99, 99, 98, 97, 100, 100, 101, 102, 103],
            "Close": [100, 99, 98, 98, 101, 101, 102, 103, 104],
            "Volume": [10] * 9,
        },
        index=idx,
    )


def test_filter_morning_session():
    result = filter_session(_sample_frame(), "morning")
    assert result.index[0].strftime("%H:%M") == "09:00"
    assert result.index[-1].strftime("%H:%M") == "11:30"
    assert len(result) == 4


def test_filter_full_session_excludes_lunch_and_outside_hours():
    result = filter_session(_sample_frame(), "full")
    times = set(result.index.strftime("%H:%M"))
    assert "08:59" not in times
    assert "12:00" not in times
    assert "15:31" not in times
    assert "12:30" in times
    assert "15:30" in times


def test_morning_low_summary_reports_minutes_from_open():
    morning = filter_session(_sample_frame(), "morning")
    summary = morning_low_summary(morning, "7203.T")
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["MorningLow"] == 97.0
    assert row["MorningLowTime"] == "09:02"
    assert row["MinutesFromOpen"] == 2
