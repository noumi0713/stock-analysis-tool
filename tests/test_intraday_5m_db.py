from datetime import date, timedelta
import sqlite3

import pandas as pd

from swing_data import intraday_5m_db as subject


def sample_bars(day: date, low: float = 90.0):
    times = list(pd.date_range(f"{day} 09:00", f"{day} 11:25", freq="5min"))
    times += list(pd.date_range(f"{day} 12:30", f"{day} 15:25", freq="5min"))
    rows = [{"Open": 100.0, "High": 101.0, "Low": 99.0,
             "Close": 100.0, "Volume": 1000} for _ in times]
    rows[7]["Low"] = low
    return pd.DataFrame(rows, index=pd.DatetimeIndex(times, tz="Asia/Tokyo"))


def test_collect_replaces_existing_day_and_keeps_history(tmp_path, monkeypatch):
    yesterday = date.today() - timedelta(days=1)
    previous = yesterday - timedelta(days=1)
    first = pd.concat([sample_bars(previous), sample_bars(yesterday)])
    monkeypatch.setattr(subject, "fetch", lambda *args, **kwargs: first)
    target = tmp_path / "test.sqlite"
    report = subject.collect(target, ["9984.T"], 7)
    assert report["total_days"] == 2
    assert report["total_bars"] == len(first)
    assert report["integrity"] == "ok"

    monkeypatch.setattr(subject, "fetch", lambda *args, **kwargs: sample_bars(yesterday, low=85))
    report = subject.collect(target, ["9984.T"], 7)
    assert report["total_days"] == 2
    assert report["total_bars"] == len(first)
    with sqlite3.connect(target) as db:
        assert db.execute("SELECT low,first_low_time FROM daily_lows WHERE trading_date=?",
                          (yesterday.isoformat(),)).fetchone() == (85, "09:35")


def test_incomplete_and_future_day_are_skipped():
    today = date.today()
    full = sample_bars(today)
    assert list(subject.normalized_days(full, today.isoformat(), False)) == []
    partial = full.iloc[:40]
    assert list(subject.normalized_days(partial, (today + timedelta(days=1)).isoformat(),
                                        False)) == []
