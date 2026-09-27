"""Conservative, point-in-time baseline for a fixed-time cash-only contest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .intraday_5m_db import read_tickers
from .intraday_5m_quality import EXPECTED_TIMES

JST = ZoneInfo("Asia/Tokyo")
ENTRY_TIMES = ("09:05", "09:30", "10:00", "10:30", "12:30", "13:30", "14:30")
EXIT_TIMES = ("11:25", "14:30", "15:20")
PAIRS = tuple((buy, sell) for buy in ENTRY_TIMES for sell in EXIT_TIMES if buy < sell)
CAPITAL = 2_000_000
MAX_PER_TICKER = 1_000_000
MAX_POSITIONS = 3
COST_PCT = 0.20  # round trip: fees and an assumed allowance for market-order slippage
MIN_TRAIN = 35
MIN_VALIDATION = 15


def valid_sessions(db: sqlite3.Connection, ticker: str, cutoff: str,
                   limit: int = 120) -> list[tuple[str, dict[str, float]]]:
    dates = [x[0] for x in db.execute(
        "SELECT trading_date FROM daily_lows WHERE ticker=? AND trading_date<? "
        "ORDER BY trading_date DESC LIMIT ?", (ticker, cutoff, limit)
    )][::-1]
    if not dates:
        return []
    rows = db.execute(
        "SELECT trading_date,bar_time,open FROM bars WHERE ticker=? "
        "AND trading_date>=? AND trading_date<? ORDER BY trading_date,bar_time",
        (ticker, dates[0], cutoff),
    )
    grouped: dict[str, dict[str, float]] = {date: {} for date in dates}
    counts: dict[str, int] = {date: 0 for date in dates}
    for date, clock, opening in rows:
        if date in grouped:
            grouped[date][clock] = opening
            counts[date] += 1
    needed = set(EXPECTED_TIMES)
    return [(date, grouped[date]) for date in dates
            if needed.issubset(grouped[date]) and
            counts[date] == len(grouped[date]) and
            all(isinstance(price, (int, float)) and math.isfinite(price) and price > 0
                for price in grouped[date].values())]


def returns_for_pair(days: list[tuple[str, dict[str, float]]],
                     pair: tuple[str, str]) -> list[float]:
    buy, sell = pair
    return [((bars[sell] / bars[buy] - 1) * 100 - COST_PCT) for _, bars in days]


def fit_one(days: list[tuple[str, dict[str, float]]]) -> dict | None:
    if len(days) < MIN_TRAIN + MIN_VALIDATION:
        return None
    split = max(MIN_TRAIN, int(len(days) * 0.75))
    if len(days) - split < MIN_VALIDATION:
        split = len(days) - MIN_VALIDATION
    training, validation = days[:split], days[split:]
    if len(training) < MIN_TRAIN or len(validation) < MIN_VALIDATION:
        return None
    # Fix the pair using training dates only; validation dates are untouched
    # until exactly one pair has been chosen for this ticker.
    pair = max(PAIRS, key=lambda p: (
        statistics.mean(returns_for_pair(training, p)) * len(training) / (len(training) + 20),
        p,
    ))
    train_net = returns_for_pair(training, pair)
    validation_net = returns_for_pair(validation, pair)
    shrunk_mean = statistics.mean(train_net) * len(training) / (len(training) + 20)
    val_mean = statistics.mean(validation_net)
    val_std = statistics.stdev(validation_net)
    lower_bound = val_mean - 1.28 * val_std / math.sqrt(len(validation_net))
    return {
        "buy_time": pair[0], "sell_time": pair[1],
        "train_days": len(training), "validation_days": len(validation),
        "first_train_date": training[0][0], "last_train_date": training[-1][0],
        "first_validation_date": validation[0][0],
        "last_validation_date": validation[-1][0],
        "train_shrunk_net_pct": shrunk_mean,
        "validation_net_mean_pct": val_mean,
        "validation_net_lower_bound_pct": lower_bound,
        "validation_win_rate": sum(x > 0 for x in validation_net) / len(validation_net),
        "score_pct": min(shrunk_mean, lower_bound),
    }


def select(snapshot: dict, snapshot_bytes: bytes, db_path: Path,
           allowed_tickers: list[str], now: datetime) -> dict:
    local = now.astimezone(JST)
    output = {"schema_version": 1, "trade_date": snapshot.get("trading_date"),
              "generated_at_jst": local.isoformat(),
              "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
              "method": "fixed-pair train/validation baseline v1",
              "capital_jpy": CAPITAL, "round_trip_cost_pct": COST_PCT,
              "recommendations": [], "diagnostics": {}, "status": "NO_TRADE"}
    cutoff = snapshot.get("cutoff_jst")
    if not cutoff or local > datetime.fromisoformat(cutoff).astimezone(JST):
        output["reason"] = "Selection missed the 08:15 JST deadline"
        return output
    if snapshot.get("status") != "READY":
        output["reason"] = "Morning input snapshot is incomplete"
        return output
    if datetime.fromisoformat(snapshot["completed_at_jst"]) > datetime.fromisoformat(cutoff):
        output["reason"] = "Morning snapshot completed after the deadline"
        return output
    domestic = snapshot.get("domestic") or {}
    if not db_path.is_file() or hashlib.sha256(db_path.read_bytes()).hexdigest() != domestic.get("database_sha256"):
        output["reason"] = "Historical database does not match the frozen morning snapshot"
        return output
    if domestic.get("session_date") != snapshot.get("prior_tse_session"):
        output["reason"] = "Prior session is not verified"
        return output

    candidates = []
    with sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True) as db:
        for ticker in allowed_tickers:
            context = domestic.get("tickers", {}).get(ticker)
            if context is None:
                output["diagnostics"][ticker] = "Missing prior-session context"
                continue
            if (context.get("volume") or 0) * context["close"] < 150_000_000:
                output["diagnostics"][ticker] = "Prior turnover below JPY 150 million"
                continue
            days = valid_sessions(db, ticker, snapshot["trading_date"])
            fitted = fit_one(days)
            if fitted is None:
                output["diagnostics"][ticker] = f"Only {len(days)} complete historical sessions; need 50"
                continue
            output["diagnostics"][ticker] = fitted
            if fitted["score_pct"] > 0:
                candidates.append((ticker, fitted, context))

    candidates.sort(key=lambda x: (-x[1]["score_pct"], x[0]))
    remaining = CAPITAL
    for ticker, fitted, context in candidates:
        if len(output["recommendations"]) >= MAX_POSITIONS:
            break
        # A 20% overnight gap reserve prevents using the full cash balance
        # against yesterday's price. An even larger actual gap may still reject.
        reference_price = context["close"] * 1.20
        shares = math.floor(min(MAX_PER_TICKER, remaining) / (reference_price * 100)) * 100
        if shares < 100:
            output["diagnostics"][ticker] = "Positive score, but no affordable 100-share lot"
            continue
        reserved = shares * reference_price
        remaining -= reserved
        output["recommendations"].append({
            "ticker": ticker, "shares": shares,
            "buy_time_jst": fitted["buy_time"], "sell_time_jst": fitted["sell_time"],
            "sizing_reference_jpy": reference_price,
            "reserved_cash_jpy": reserved,
            "validation_net_lower_bound_pct": fitted["validation_net_lower_bound_pct"],
            "score_pct": fitted["score_pct"],
        })
    if output["recommendations"]:
        output["status"] = "PROVISIONAL_PICKS"
    else:
        output["reason"] = "No candidate passed the sample, cost and validation gates"
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--tickers-file", type=Path,
                        default=Path("swing_data/config/contest_equity_tickers.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.snapshot.read_bytes()
    snapshot = json.loads(raw)
    result = select(snapshot, raw, args.db, read_tickers(args.tickers_file), datetime.now(JST))
    date = result["trade_date"]
    if not date:
        parser.error("snapshot has no trade date")
    path = args.output_dir / date[:4] / date[5:7] / f"{date}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    print(f"{path}: {result['status']} ({len(result['recommendations'])} picks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
