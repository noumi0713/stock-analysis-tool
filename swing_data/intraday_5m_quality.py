"""Audit completeness and internal consistency of the incremental 5-minute DB."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals

from .intraday_5m_db import read_tickers

JST = ZoneInfo("Asia/Tokyo")
EXPECTED_TIMES = tuple(
    f"{h:02d}:{m:02d}"
    for h, start, end in ((9, 0, 60), (10, 0, 60), (11, 0, 30),
                           (12, 30, 60), (13, 0, 60), (14, 0, 60), (15, 0, 30))
    for m in range(start, end, 5)
)


def expected_sessions(asof: datetime, count: int) -> list[str]:
    if asof.tzinfo is None:
        raise ValueError("asof must include a timezone")
    local = asof.astimezone(JST)
    end = local.date()
    if local.time() < time(15, 40):
        end -= timedelta(days=1)
    calendar = xcals.get_calendar("XTKS")
    start = end - timedelta(days=max(30, count * 4))
    return [date.strftime("%Y-%m-%d") for date in calendar.sessions_in_range(start, end)][-count:]


def issue(kind: str, ticker: str, date: str | None, detail: str, severity: str) -> dict:
    return {"type": kind, "severity": severity, "ticker": ticker,
            "date": date, "detail": detail}


def audit(db_path: Path, tickers: list[str], asof: datetime,
          lookback_sessions: int = 10) -> dict:
    sessions = expected_sessions(asof, lookback_sessions)
    report = {"generated_at_jst": asof.astimezone(JST).isoformat(),
              "expected_latest_session": sessions[-1] if sessions else None,
              "lookback_sessions": lookback_sessions, "status": "PASS",
              "issues": [], "tickers": {}}
    if not db_path.is_file():
        report["issues"].append(issue("database_missing", "ALL", None, str(db_path), "FAIL"))
        report["status"] = "FAIL"
        report["issue_counts"] = {"FAIL": 1, "WARN": 0}
        return report

    with sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True) as db:
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            report["issues"].append(issue("integrity", "ALL", None, integrity, "FAIL"))
        for ticker in tickers:
            try:
                daily = db.execute(
                    "SELECT trading_date,low,first_low_time,high,first_high_time,open,close,bar_count "
                    "FROM daily_lows WHERE ticker=? ORDER BY trading_date", (ticker,)
                ).fetchall()
                dates = {row[0]: row for row in daily}
                available = sorted(dates)
                report["tickers"][ticker] = {"first_date": available[0] if available else None,
                                              "last_date": available[-1] if available else None,
                                              "stored_days": len(available)}
                if not available:
                    report["issues"].append(issue("no_data", ticker, None,
                                                  "No completed session stored", "FAIL"))
                    continue
                for date in sessions:
                    if date < available[0]:
                        continue  # IPOs need not have earlier rows.
                    if date not in dates:
                        report["issues"].append(issue("missing_session", ticker, date,
                                                      "Expected trading date absent", "FAIL"))
                if sessions and sessions[-1] not in dates:
                    report["issues"].append(issue("stale", ticker, sessions[-1],
                                                  "Latest completed session absent", "FAIL"))

                previous_close = None
                for date, low, first_low, high, first_high, day_open, day_close, n in daily:
                    if date not in sessions:
                        previous_close = day_close
                        continue
                    bars = db.execute(
                        "SELECT bar_time,open,high,low,close,volume FROM bars "
                        "WHERE ticker=? AND trading_date=? ORDER BY bar_time", (ticker, date)
                    ).fetchall()
                    times = [bar[0] for bar in bars]
                    missing = sorted(set(EXPECTED_TIMES) - set(times))
                    unexpected = sorted(set(times) - set(EXPECTED_TIMES) - {"15:30"})
                    if missing:
                        report["issues"].append(issue("missing_bars", ticker, date,
                                                      f"{len(missing)} slots: {','.join(missing)}", "FAIL"))
                    if unexpected or len(times) != len(set(times)):
                        report["issues"].append(issue("unexpected_or_duplicate_bars", ticker, date,
                                                      str(unexpected), "FAIL"))
                    if len(bars) != n:
                        report["issues"].append(issue("bar_count_mismatch", ticker, date,
                                                      f"stored={n} actual={len(bars)}", "FAIL"))
                    if not bars:
                        continue
                    if any(v <= 0 for bar in bars for v in bar[1:5]) or any(bar[5] < 0 for bar in bars):
                        report["issues"].append(issue("invalid_ohlcv", ticker, date,
                                                      "Nonpositive price or negative volume", "FAIL"))
                    if any(bar[2] < max(bar[1], bar[3], bar[4]) or
                           bar[3] > min(bar[1], bar[2], bar[4]) for bar in bars):
                        report["issues"].append(issue("invalid_ohlcv", ticker, date,
                                                      "OHLC relationships violated", "FAIL"))
                    computed_low = min(bar[3] for bar in bars)
                    computed_high = max(bar[2] for bar in bars)
                    low_time = next(bar[0] for bar in bars if bar[3] == computed_low)
                    high_time = next(bar[0] for bar in bars if bar[2] == computed_high)
                    if (computed_low, low_time, computed_high, high_time,
                        bars[0][1], bars[-1][4]) != (low, first_low, high, first_high,
                                                      day_open, day_close):
                        report["issues"].append(issue("summary_mismatch", ticker, date,
                                                      "daily_lows differs from bars", "FAIL"))
                    if previous_close and previous_close > 0:
                        # A large overnight gap may be a split, a corporate action,
                        # a limit move or a vendor error. Do not auto-adjust it.
                        if day_open / previous_close >= 1.35 or day_open / previous_close <= 0.65:
                            report["issues"].append(issue("possible_corporate_action", ticker, date,
                                                          f"open / previous close={day_open / previous_close:.3f}",
                                                          "WARN"))
                    previous_close = day_close
            except sqlite3.DatabaseError as exc:
                report["issues"].append(issue("database_error", ticker, None, str(exc), "FAIL"))

    report["status"] = "FAIL" if any(x["severity"] == "FAIL" for x in report["issues"]) else (
        "WARN" if report["issues"] else "PASS"
    )
    report["issue_counts"] = {level: sum(x["severity"] == level for x in report["issues"])
                              for level in ("FAIL", "WARN")}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--tickers-file", type=Path, default=Path("swing_data/config/intraday_5m_tickers.json"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--lookback-sessions", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.lookback_sessions <= 60:
        parser.error("--lookback-sessions must be between 1 and 60")
    try:
        result = audit(args.db, read_tickers(args.tickers_file), datetime.now(JST),
                       args.lookback_sessions)
    except (sqlite3.DatabaseError, ValueError) as exc:
        result = {"generated_at_jst": datetime.now(JST).isoformat(), "status": "FAIL",
                  "issues": [issue("audit_error", "ALL", None, str(exc), "FAIL")],
                  "issue_counts": {"FAIL": 1, "WARN": 0}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "issues"}, ensure_ascii=False))
    return 0 if result["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
