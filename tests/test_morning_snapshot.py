from datetime import datetime, timedelta
import json
import sqlite3
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from swing_data import intraday_5m_db, morning_snapshot

JST = ZoneInfo("Asia/Tokyo")
NY = ZoneInfo("America/New_York")
START = datetime(2026, 9, 28, 8, 5, tzinfo=JST)


def source(symbol, interval):
    if interval == "5m":
        index = pd.DatetimeIndex([START - timedelta(minutes=5), START + timedelta(minutes=5)])
        return pd.DataFrame({"Close": [147.0, 999.0]}, index=index)
    index = pd.DatetimeIndex([datetime(2026, 9, 25, tzinfo=NY),
                              datetime(2026, 9, 28, tzinfo=NY)])
    return pd.DataFrame({"Close": [100.0, 999.0]}, index=index)


def domestic_files(tmp_path):
    database = tmp_path / "bars.sqlite"
    quality = tmp_path / "quality.json"
    with sqlite3.connect(database) as db:
        db.executescript(intraday_5m_db.SCHEMA)
        db.execute("INSERT INTO daily_lows VALUES (?,?,?,?,?,?,?,?,?)",
                   ("9984.T", "2026-09-25", 90, "09:35", 110, "10:10", 100, 105, 66))
        db.execute("INSERT INTO bars VALUES (?,?,?,?,?,?,?,?,?)",
                   ("9984.T", "2026-09-25", "09:00", 100, 110, 90, 105, 1000, START.isoformat()))
    quality.write_text(json.dumps({"status": "PASS", "expected_latest_session": "2026-09-25"}))
    return database, quality


def test_frozen_snapshot_excludes_future_values_and_keeps_provenance(tmp_path):
    db, quality = domestic_files(tmp_path)
    snapshot = morning_snapshot.build_snapshot(
        START, fetch=source, db_path=db, quality_path=quality,
        tickers=["9984.T"], clock=lambda: START,
    )
    assert snapshot["status"] == "READY"
    assert snapshot["series"]["sp500"]["session_date"] == "2026-09-25"
    assert snapshot["series"]["sp500"]["value"] == 100
    assert snapshot["series"]["usd_jpy"]["value"] == 147
    assert snapshot["domestic"]["tickers"]["9984.T"]["volume"] == 1000
    assert len(snapshot["domestic"]["database_sha256"]) == 64


def test_cutoff_discards_late_completion(tmp_path):
    db, quality = domestic_files(tmp_path)
    late = START.replace(hour=8, minute=16)
    with pytest.raises(ValueError, match="finished after"):
        morning_snapshot.build_snapshot(
            START, fetch=source, db_path=db, quality_path=quality,
            tickers=["9984.T"], clock=lambda: late,
        )
    with pytest.raises(ValueError, match="started after"):
        morning_snapshot.build_snapshot(late)


def test_missing_verified_domestic_db_is_incomplete(tmp_path):
    snapshot = morning_snapshot.build_snapshot(
        START, fetch=source, db_path=tmp_path / "absent.sqlite",
        quality_path=tmp_path / "absent.json", tickers=["9984.T"], clock=lambda: START,
    )
    assert snapshot["status"] == "INCOMPLETE"
    assert "domestic" in snapshot["errors"]


def test_tokyo_holiday_is_skipped():
    holiday = datetime(2026, 9, 23, 8, 5, tzinfo=JST)
    assert morning_snapshot.build_snapshot(holiday)["status"] == "SKIP"
