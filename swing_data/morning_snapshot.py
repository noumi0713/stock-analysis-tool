"""Freeze the information actually available before 08:15 JST."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
import yfinance as yf

from .intraday_5m_db import read_tickers
from .intraday_5m_quality import EXPECTED_TIMES

JST = ZoneInfo("Asia/Tokyo")
NY = ZoneInfo("America/New_York")
CUTOFF = time(8, 15)
US_SERIES = {"sp500": "^GSPC", "nasdaq": "^IXIC", "semiconductor": "^SOX",
             "vix": "^VIX"}


def prior_session(now: datetime) -> str | None:
    local = now.astimezone(JST)
    calendar = xcals.get_calendar("XTKS")
    if not calendar.is_session(local.date()):
        return None
    return calendar.previous_session(local.date()).strftime("%Y-%m-%d")


def fetch_history(symbol: str, interval: str) -> pd.DataFrame:
    return yf.Ticker(symbol).history(
        period="7d" if interval == "1d" else "2d", interval=interval,
        auto_adjust=False, prepost=False, actions=False, repair=False,
        raise_errors=True,
    )


def us_daily(frame: pd.DataFrame, now: datetime) -> dict:
    if frame.empty:
        raise ValueError("no rows")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        raise ValueError("source daily index has no timezone")
    eligible = []
    for stamp, row in frame.iterrows():
        session = stamp.tz_convert(NY).date()
        close = datetime.combine(session, time(16, 15), NY)
        # Conservative publication buffer. Do not treat the midnight date label
        # of a daily candle as its actual availability time.
        available = close + timedelta(minutes=60)
        if available <= now:
            eligible.append((session, row, available))
    if not eligible:
        raise ValueError("no completed US session available before cutoff")
    session, row, available = eligible[-1]
    if (now.astimezone(NY).date() - session).days > 4:
        raise ValueError("US daily value is stale")
    return {"value": float(row["Close"]), "session_date": session.isoformat(),
            "available_after_jst": available.astimezone(JST).isoformat()}


def fx_latest(frame: pd.DataFrame, now: datetime) -> dict:
    if frame.empty:
        raise ValueError("no rows")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        raise ValueError("source FX index has no timezone")
    eligible = frame.loc[index <= pd.Timestamp(now)]
    if eligible.empty:
        raise ValueError("no FX observation at or before cutoff")
    stamp = eligible.index[-1].to_pydatetime()
    if now - stamp > timedelta(minutes=30):
        raise ValueError("FX observation is over 30 minutes old")
    return {"value": float(eligible.iloc[-1]["Close"]),
            "observed_at_jst": stamp.astimezone(JST).isoformat()}


def wilder_average(values: list[float], period: int = 14) -> float | None:
    if len(values) < period:
        return None
    average = sum(values[:period]) / period
    for value in values[period:]:
        average = (average * (period - 1) + value) / period
    return average


def daily_indicators(rows: list[tuple]) -> dict:
    closes = [float(r[4]) for r in rows]
    changes = [b - a for a, b in zip(closes, closes[1:])]
    gain = wilder_average([max(x, 0) for x in changes])
    loss = wilder_average([max(-x, 0) for x in changes])
    rsi = None if gain is None else (
        50.0 if gain == 0 and loss == 0 else
        100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
    )
    tr = [max(float(row[2]) - float(row[3]),
              abs(float(row[2]) - float(rows[i - 1][4])) if i else 0,
              abs(float(row[3]) - float(rows[i - 1][4])) if i else 0)
          for i, row in enumerate(rows)]
    return {
        "previous_return_pct": (closes[-1] / closes[-2] - 1) * 100 if len(closes) >= 2 else None,
        "five_day_return_pct": (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 else None,
        "ma25_deviation_pct": (closes[-1] / (sum(closes[-25:]) / 25) - 1) * 100
        if len(closes) >= 25 else None,
        "rsi14": rsi,
        "atr14": wilder_average(tr),
    }


def domestic_context(db_path: Path, quality_path: Path, tickers: list[str],
                     expected_date: str) -> dict:
    if not db_path.is_file() or not quality_path.is_file():
        raise ValueError("5-minute DB or its quality report is missing")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if (quality.get("status") not in ("PASS", "WARN", "FAIL") or
            quality.get("expected_latest_session") != expected_date):
        raise ValueError("5-minute DB quality report is absent or for the wrong session")
    if any(x.get("severity") == "FAIL" and x.get("ticker") == "ALL"
           for x in quality.get("issues", [])):
        raise ValueError("5-minute DB has a global integrity/audit failure")
    blocked_days = {(x.get("ticker"), x.get("date"))
                    for x in quality.get("issues", []) if x.get("severity") == "FAIL"}
    calendar = xcals.get_calendar("XTKS")
    result = {"database_sha256": hashlib.sha256(db_path.read_bytes()).hexdigest(),
              "quality_sha256": hashlib.sha256(quality_path.read_bytes()).hexdigest(),
              "quality_status": quality["status"], "session_date": expected_date,
              "tickers": {}, "excluded_tickers": {}}
    with sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True) as db:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("5-minute DB integrity check failed")
        for ticker in tickers:
            rows = db.execute(
                "SELECT trading_date,open,high,low,close FROM daily_lows "
                "WHERE ticker=? AND trading_date<=? ORDER BY trading_date DESC LIMIT 250",
                (ticker, expected_date),
            ).fetchall()
            if not rows:
                result["excluded_tickers"][ticker] = "No usable historical session"
                continue
            latest = rows[0][0]
            age = len(calendar.sessions_in_range(latest, expected_date)) - 1
            if age > 1 or age < 0:
                result["excluded_tickers"][ticker] = f"Latest session {latest} is {age} trading days stale"
                continue
            if (ticker, latest) in blocked_days:
                result["excluded_tickers"][ticker] = f"Session {latest} failed quality audit"
                continue
            bars = db.execute(
                "SELECT bar_time,open,close,volume FROM bars WHERE ticker=? AND trading_date=? "
                "ORDER BY bar_time", (ticker, latest)
            ).fetchall()
            clocks = [bar[0] for bar in bars]
            if (not set(EXPECTED_TIMES).issubset(clocks) or
                    len(clocks) != len(set(clocks)) or len(bars) != db.execute(
                        "SELECT bar_count FROM daily_lows WHERE ticker=? AND trading_date=?",
                        (ticker, latest)).fetchone()[0] or
                    bars[0][1] != rows[0][1] or bars[-1][2] != rows[0][4]):
                result["excluded_tickers"][ticker] = "Prior session bars are incomplete or inconsistent"
                continue
            rows.reverse()
            current = rows[-1]
            vol = sum(bar[3] for bar in bars)
            result["tickers"][ticker] = {
                "date": latest, "staleness_sessions": age, "open": float(current[1]),
                "high": float(current[2]), "low": float(current[3]),
                "close": float(current[4]), "volume": vol,
                **daily_indicators(rows),
            }
    return result


def build_snapshot(now: datetime, fetch=fetch_history, db_path: Path | None = None,
                   quality_path: Path | None = None, tickers: list[str] | None = None,
                   clock=lambda: datetime.now(JST)) -> dict:
    if now.tzinfo is None:
        raise ValueError("start time needs a timezone")
    local = now.astimezone(JST)
    if local.time() > CUTOFF:
        raise ValueError("started after the 08:15 JST cutoff")
    previous = prior_session(local)
    if previous is None:
        return {"status": "SKIP", "reason": "not a Tokyo trading day"}

    snapshot = {"schema_version": 1, "trading_date": local.date().isoformat(),
                "cutoff_jst": f"{local.date()}T08:15:00+09:00",
                "started_at_jst": local.isoformat(), "prior_tse_session": previous,
                "series": {}, "domestic": None, "errors": {}, "warnings": {}}
    for name, symbol in US_SERIES.items():
        try:
            value = us_daily(fetch(symbol, "1d"), clock())
            value.update({"provider": "Yahoo Finance via yfinance", "symbol": symbol,
                          "fetched_at_jst": clock().astimezone(JST).isoformat()})
            snapshot["series"][name] = value
        except Exception as exc:
            snapshot["warnings"][name] = str(exc)
    try:
        value = fx_latest(fetch("JPY=X", "5m"), clock())
        value.update({"provider": "Yahoo Finance via yfinance", "symbol": "JPY=X",
                      "fetched_at_jst": clock().astimezone(JST).isoformat()})
        snapshot["series"]["usd_jpy"] = value
    except Exception as exc:
        snapshot["warnings"]["usd_jpy"] = str(exc)

    try:
        if db_path is None or quality_path is None or tickers is None:
            raise ValueError("domestic inputs not configured")
        snapshot["domestic"] = domestic_context(db_path, quality_path, tickers, previous)
    except Exception as exc:
        snapshot["errors"]["domestic"] = str(exc)

    finished = clock().astimezone(JST)
    if finished.date() != local.date() or finished.time() > CUTOFF:
        raise ValueError("snapshot finished after the 08:15 JST cutoff; discard all values")
    snapshot["completed_at_jst"] = finished.isoformat()
    snapshot["status"] = "READY" if not snapshot["errors"] else "INCOMPLETE"
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--tickers-file", type=Path, default=Path("swing_data/config/intraday_5m_tickers.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    snapshot = build_snapshot(datetime.now(JST), db_path=args.db, quality_path=args.quality,
                              tickers=read_tickers(args.tickers_file))
    if snapshot["status"] == "SKIP":
        print(json.dumps(snapshot, ensure_ascii=False))
        return 0
    date = snapshot["trading_date"]
    path = args.output_dir / date[:4] / date[5:7] / f"{date}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(snapshot, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print(f"{path}: {snapshot['status']} ({len(snapshot['errors'])} source errors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
