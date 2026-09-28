from datetime import datetime
import sqlite3
from zoneinfo import ZoneInfo

import pandas as pd

from swing_data import intraday_5m_db, intraday_5m_quality


ASOF = datetime(2026, 9, 27, 8, 30, tzinfo=ZoneInfo("Asia/Tokyo"))


def bars(date: str, price: float = 100):
    times = list(pd.date_range(f"{date} 09:00", f"{date} 11:25", freq="5min"))
    times += list(pd.date_range(f"{date} 12:30", f"{date} 15:25", freq="5min"))
    rows = [{"Open": price, "High": price + 1, "Low": price - 1,
             "Close": price, "Volume": 1000} for _ in times]
    rows[7]["Low"] = price - 2
    return pd.DataFrame(rows, index=pd.DatetimeIndex(times, tz="Asia/Tokyo"))


def make_db(path):
    with sqlite3.connect(path) as db:
        db.executescript(intraday_5m_db.SCHEMA)
        for date in ("2026-09-24", "2026-09-25"):
            intraday_5m_db.store_day(db, "9984.T", date, bars(date))


def kinds(result):
    return {x["type"] for x in result["issues"]}


def test_complete_db_passes_and_skips_weekend(tmp_path):
    path = tmp_path / "bars.sqlite"
    make_db(path)
    report = intraday_5m_quality.audit(path, ["9984.T"], ASOF, lookback_sessions=2)
    assert report["expected_latest_session"] == "2026-09-25"
    assert report["status"] == "PASS"
    assert report["issues"] == []


def test_closing_auction_at_1530_without_1525_is_complete(tmp_path):
    path = tmp_path / "bars.sqlite"
    make_db(path)
    with sqlite3.connect(path) as db:
        for date in ("2026-09-24", "2026-09-25"):
            day = bars(date)
            day = day.drop(day.index[-1])
            auction = day.iloc[-1].copy()
            day.loc[pd.Timestamp(f"{date} 15:30", tz="Asia/Tokyo")] = auction
            intraday_5m_db.store_day(db, "9984.T", date, day)
    report = intraday_5m_quality.audit(path, ["9984.T"], ASOF, lookback_sessions=2)
    assert report["status"] == "PASS"


def test_missing_bar_and_summary_mismatch_block_publication(tmp_path):
    path = tmp_path / "bars.sqlite"
    make_db(path)
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM bars WHERE ticker=? AND trading_date=? AND bar_time=?",
                   ("9984.T", "2026-09-25", "09:35"))
    report = intraday_5m_quality.audit(path, ["9984.T"], ASOF, lookback_sessions=2)
    assert report["status"] == "FAIL"
    assert {"missing_bars", "bar_count_mismatch", "summary_mismatch"} <= kinds(report)


def test_stale_ticker_blocks_publication(tmp_path):
    path = tmp_path / "bars.sqlite"
    make_db(path)
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM bars WHERE trading_date=?", ("2026-09-25",))
        db.execute("DELETE FROM daily_lows WHERE trading_date=?", ("2026-09-25",))
    report = intraday_5m_quality.audit(path, ["9984.T"], ASOF, lookback_sessions=2)
    assert report["status"] == "FAIL"
    assert {"missing_session", "stale"} <= kinds(report)


def test_large_gap_is_flagged_without_inventing_split(tmp_path):
    path = tmp_path / "bars.sqlite"
    make_db(path)
    with sqlite3.connect(path) as db:
        intraday_5m_db.store_day(db, "9984.T", "2026-09-25", bars("2026-09-25", 150))
    report = intraday_5m_quality.audit(path, ["9984.T"], ASOF, lookback_sessions=2)
    assert report["status"] == "WARN"
    assert "possible_corporate_action" in kinds(report)
