from datetime import datetime
import hashlib
import json
import sqlite3
from zoneinfo import ZoneInfo

import pandas as pd

from swing_data import intraday_5m_db, intraday_5m_quality, intraday_selector

ASOF = datetime(2026, 9, 28, 8, 10, tzinfo=ZoneInfo("Asia/Tokyo"))


def fixture_db(path, n=60, negative_validation=False):
    dates = pd.bdate_range(end="2026-09-25", periods=n).strftime("%Y-%m-%d")
    with sqlite3.connect(path) as db:
        db.executescript(intraday_5m_db.SCHEMA)
        for i, date in enumerate(dates):
            slope = -0.05 if negative_validation and i >= max(7, int(n * .70)) else 0.05
            bars = []
            for j, clock in enumerate(intraday_5m_quality.EXPECTED_TIMES):
                price = 100 + j * slope
                bars.append(("9984.T", date, clock, price, price + .1,
                             price - .1, price, 30000, "2026-09-25T16:45:00+09:00"))
            db.executemany("INSERT INTO bars VALUES (?,?,?,?,?,?,?,?,?)", bars)
            db.execute("INSERT INTO daily_lows VALUES (?,?,?,?,?,?,?,?,?)",
                       ("9984.T", date, 99, "09:00", 104, "15:25", 100, 103, len(bars)))
    return dates


def snapshot_for(db):
    return {"trading_date": "2026-09-28", "prior_tse_session": "2026-09-25",
            "cutoff_jst": "2026-09-28T08:15:00+09:00",
            "completed_at_jst": "2026-09-28T08:06:00+09:00", "status": "READY",
            "domestic": {"database_sha256": hashlib.sha256(db.read_bytes()).hexdigest(),
                         "session_date": "2026-09-25",
                         "tickers": {"9984.T": {"date": "2026-09-25", "staleness_sessions": 0,
                                                  "close": 100, "volume": 2_000_000}}}}


def run(snapshot, db, now=ASOF):
    raw = json.dumps(snapshot).encode()
    return intraday_selector.select(snapshot, raw, db, ["9984.T"], now)


def test_positive_holdout_generates_sized_timed_pick(tmp_path):
    path = tmp_path / "bars.sqlite"
    fixture_db(path)
    result = run(snapshot_for(path), path)
    assert result["status"] == "PROVISIONAL_PICKS"
    pick = result["recommendations"][0]
    assert pick["shares"] % 100 == 0
    assert pick["reserved_cash_jpy"] <= 1_000_000
    assert pick["validation_net_lower_bound_pct"] > 0
    assert pick["buy_time_jst"] < pick["sell_time_jst"]
    assert result["diagnostics"]["9984.T"]["validation_days"] >= 3


def test_out_of_sample_loss_or_small_sample_abstains(tmp_path):
    path = tmp_path / "bars.sqlite"
    fixture_db(path, negative_validation=True)
    assert run(snapshot_for(path), path)["status"] == "NO_TRADE"
    smaller = tmp_path / "small.sqlite"
    fixture_db(smaller, n=9)
    result = run(snapshot_for(smaller), smaller)
    assert result["status"] == "NO_TRADE"
    assert "9 complete" in result["diagnostics"]["9984.T"]


def test_ten_sessions_are_enough_to_fit(tmp_path):
    path = tmp_path / "ten.sqlite"
    fixture_db(path, n=10)
    result = run(snapshot_for(path), path)
    assert result["status"] == "PROVISIONAL_PICKS"
    assert result["diagnostics"]["9984.T"]["train_days"] == 7
    assert result["diagnostics"]["9984.T"]["validation_days"] == 3


def test_one_missing_prior_session_halves_ticker_cap(tmp_path):
    path = tmp_path / "stale.sqlite"
    fixture_db(path, n=10)
    snapshot = snapshot_for(path)
    snapshot["domestic"]["quality_status"] = "FAIL"
    snapshot["domestic"]["tickers"]["9984.T"]["staleness_sessions"] = 1
    result = run(snapshot, path)
    assert result["status"] == "PROVISIONAL_PICKS"
    assert result["source_quality_status"] == "FAIL"
    assert result["recommendations"][0]["reserved_cash_jpy"] <= 500_000
    assert result["recommendations"][0]["staleness_sessions"] == 1


def test_future_rows_cannot_enter_training_and_input_hash_is_required(tmp_path):
    path = tmp_path / "bars.sqlite"
    fixture_db(path)
    with sqlite3.connect(path) as db:
        original = intraday_selector.valid_sessions(db, "9984.T", "2026-09-28")
        db.execute("INSERT INTO daily_lows VALUES (?,?,?,?,?,?,?,?,?)",
                   ("9984.T", "2026-09-28", 1, "09:00", 1000, "15:20", 100, 1000, 66))
        later = intraday_selector.valid_sessions(db, "9984.T", "2026-09-28")
    assert original == later
    snapshot = snapshot_for(path)
    snapshot["domestic"]["database_sha256"] = "wrong"
    result = run(snapshot, path)
    assert result["status"] == "NO_TRADE"
    assert "does not match" in result["reason"]


def test_late_or_incomplete_snapshot_abstains(tmp_path):
    path = tmp_path / "bars.sqlite"
    fixture_db(path)
    snapshot = snapshot_for(path)
    snapshot["status"] = "INCOMPLETE"
    assert run(snapshot, path)["status"] == "NO_TRADE"
    snapshot["status"] = "READY"
    result = run(snapshot, path, ASOF.replace(hour=8, minute=16))
    assert result["status"] == "NO_TRADE"
    assert "deadline" in result["reason"]
