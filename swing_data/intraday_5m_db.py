"""Incrementally collect Tokyo-listed 5-minute bars into a durable SQLite snapshot."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from .intraday_1m import normalize_symbol

JST = ZoneInfo("Asia/Tokyo")
SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    ticker TEXT NOT NULL,
    trading_date TEXT NOT NULL,
    bar_time TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    collected_at TEXT NOT NULL,
    PRIMARY KEY (ticker, trading_date, bar_time)
);
CREATE TABLE IF NOT EXISTS daily_lows (
    ticker TEXT NOT NULL,
    trading_date TEXT NOT NULL,
    low REAL NOT NULL,
    first_low_time TEXT NOT NULL,
    high REAL NOT NULL,
    first_high_time TEXT NOT NULL,
    open REAL NOT NULL,
    close REAL NOT NULL,
    bar_count INTEGER NOT NULL,
    PRIMARY KEY (ticker, trading_date)
);
"""


def read_tickers(path: Path) -> list[str]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not values:
        raise ValueError("ticker configuration must be a nonempty JSON array")
    tickers = [normalize_symbol(x) for x in values]
    if len(tickers) != len(set(tickers)) or any(not x.endswith(".T") for x in tickers):
        raise ValueError("tickers must be unique Tokyo-listed symbols")
    return tickers


def fetch(symbol: str, lookback_days: int, attempts: int = 3) -> pd.DataFrame:
    # Yahoo restricts intraday history to approximately 60 calendar days.
    today = datetime.now(JST).date()
    start = today - timedelta(days=lookback_days - 1)
    last_error = None
    for attempt in range(attempts):
        try:
            frame = yf.Ticker(symbol).history(
                start=start.isoformat(), end=(today + timedelta(days=1)).isoformat(),
                interval="5m", auto_adjust=False, prepost=False,
                actions=False, repair=False, raise_errors=True,
            )
            if frame.empty:
                raise ValueError("no 5-minute bars returned")
            return frame
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{symbol}: {last_error}")


def normalized_days(frame: pd.DataFrame, today: str, include_today: bool):
    frame = frame.copy()
    index = pd.DatetimeIndex(frame.index)
    frame.index = index.tz_localize(JST) if index.tz is None else index.tz_convert(JST)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.between_time("09:00", "15:30", inclusive="both")
    frame = frame[~((frame.index.time >= pd.Timestamp("11:30").time()) &
                    (frame.index.time < pd.Timestamp("12:30").time()))]
    for date, group in frame.groupby(frame.index.strftime("%Y-%m-%d")):
        if date >= today and not include_today:
            continue
        times = group.index.strftime("%H:%M")
        # Reject partial trading sessions. The closing auction may appear as
        # 15:25 or 15:30 depending on the data source.
        if "09:00" not in times or not any(t >= "15:20" for t in times) or len(group) < 50:
            continue
        numeric = group[["Open", "High", "Low", "Close", "Volume"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if numeric.isna().any().any():
            continue
        if (numeric[["Open", "High", "Low", "Close"]] <= 0).any().any():
            continue
        if (numeric["Volume"] < 0).any():
            continue
        if (numeric["High"] < numeric[["Open", "Low", "Close"]].max(axis=1)).any():
            continue
        if (numeric["Low"] > numeric[["Open", "High", "Close"]].min(axis=1)).any():
            continue
        yield date, numeric


def store_day(db: sqlite3.Connection, ticker: str, date: str, group: pd.DataFrame) -> None:
    now = datetime.now(JST).isoformat()
    # A refreshed day replaces earlier partial/corrected source bars atomically.
    with db:
        db.execute("DELETE FROM bars WHERE ticker=? AND trading_date=?", (ticker, date))
        db.executemany(
            "INSERT INTO bars VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (ticker, date, timestamp.strftime("%H:%M"),
                 float(row.Open), float(row.High), float(row.Low), float(row.Close),
                 int(row.Volume), now)
                for timestamp, row in group.iterrows()
            ],
        )
        low = group["Low"].min()
        high = group["High"].max()
        db.execute(
            "INSERT OR REPLACE INTO daily_lows VALUES (?,?,?,?,?,?,?,?,?)",
            (ticker, date, float(low), group.index[group["Low"] == low][0].strftime("%H:%M"),
             float(high), group.index[group["High"] == high][0].strftime("%H:%M"),
             float(group.iloc[0]["Open"]), float(group.iloc[-1]["Close"]), len(group)),
        )


def collect(db_path: Path, tickers: list[str], lookback_days: int,
            include_today: bool = False) -> dict:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"generated_at_jst": datetime.now(JST).isoformat(), "source": "yfinance",
              "interval": "5m", "auto_adjust": False, "lookback_calendar_days": lookback_days,
              "results": []}
    # A manually dispatched run before the close must not treat today's partial
    # 15:20 bar as a completed session.
    include_today = include_today and datetime.now(JST).time() >= datetime.strptime(
        "15:40", "%H:%M"
    ).time()
    with sqlite3.connect(db_path) as db:
        db.executescript(SCHEMA)
        for ticker in tickers:
            item = {"ticker": ticker, "days_written": 0, "rows_written": 0}
            try:
                data = fetch(ticker, lookback_days)
                for date, group in normalized_days(
                    data, datetime.now(JST).date().isoformat(), include_today
                ):
                    store_day(db, ticker, date, group)
                    item["days_written"] += 1
                    item["rows_written"] += len(group)
            except Exception as exc:
                item["error"] = str(exc)
            report["results"].append(item)
        report["total_days"] = db.execute("SELECT COUNT(*) FROM daily_lows").fetchone()[0]
        report["total_bars"] = db.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
        report["integrity"] = db.execute("PRAGMA integrity_check").fetchone()[0]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers-file", type=Path, default=Path("swing_data/config/intraday_5m_tickers.json"))
    parser.add_argument("--db", type=Path, default=Path("intraday_5m.sqlite"))
    parser.add_argument("--report", type=Path, default=Path("intraday_5m_report.json"))
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--include-today", action="store_true", help="Only use after the Tokyo close")
    args = parser.parse_args()
    if not 1 <= args.lookback_days <= 59:
        parser.error("--lookback-days must be between 1 and 59")
    report = collect(args.db, read_tickers(args.tickers_file), args.lookback_days,
                     args.include_today)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["integrity"] == "ok" and any(
        r["days_written"] for r in report["results"]
    ) else 1


if __name__ == "__main__":
    sys.exit(main())
