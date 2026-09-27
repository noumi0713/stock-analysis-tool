"""Estimate fixed-time market-order fills without claiming tape-level certainty."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from pathlib import Path

from .intraday_5m_quality import EXPECTED_TIMES

CAPITAL_JPY = 2_000_000
MAX_PER_TICKER_JPY = 1_000_000
MAX_POSITIONS = 3
LOT_SIZE = 100
FEE_RATE = 0.0005  # 0.05% each side
SLIPPAGE_RATE = 0.0005  # an additional 0.05% each side
MAX_BAR_PARTICIPATION = 0.10
LAST_CONTINUOUS_BAR = "15:20"


def get_bars(db: sqlite3.Connection, ticker: str, date: str) -> dict[str, dict]:
    rows = db.execute(
        "SELECT bar_time,open,high,low,close,volume FROM bars "
        "WHERE ticker=? AND trading_date=? ORDER BY bar_time", (ticker, date)
    ).fetchall()
    return {clock: {"open": opening, "high": high, "low": low,
                    "close": close, "volume": volume}
            for clock, opening, high, low, close, volume in rows}


def _blocking_window(clock: str, windows: list[dict]) -> dict | None:
    return next((item for item in windows if item["start"] <= clock < item["end"]), None)


def resolve_order(bars: dict[str, dict], requested: str, side: str, shares: int,
                  state: dict | None = None) -> dict:
    """Only an explicit event can justify carrying an order past a missing bar."""
    state = state or {}
    windows = state.get("untradable", [])
    for window in windows:
        if (window.get("reason") not in {"special_quote", "halt", "other"} or
            window.get("start") not in EXPECTED_TIMES or
            window.get("end") not in (*EXPECTED_TIMES, "15:25", "15:30") or
            window["start"] >= window["end"]):
            return {"status": "INVALID_MARKET_EVENT"}
    blocked = _blocking_window(requested, windows)
    if blocked and any(blocked["start"] <= clock < blocked["end"] for clock in bars):
        return {"status": "MARKET_EVENT_CONFLICT", "requested_time": requested}
    if requested not in bars and blocked is None:
        return {"status": "UNVERIFIABLE_MISSING_BAR", "requested_time": requested}
    if requested in bars and blocked is not None:
        return {"status": "MARKET_EVENT_CONFLICT", "requested_time": requested}

    eligible = [clock for clock in sorted(bars) if requested <= clock <= LAST_CONTINUOUS_BAR]
    if blocked:
        eligible = [clock for clock in eligible if clock >= blocked["end"]]
    for clock in eligible:
        if _blocking_window(clock, windows):
            continue
        bar = bars[clock]
        prices = (bar["open"], bar["high"], bar["low"], bar["close"])
        if (not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0
                    for x in prices) or
            bar["high"] < max(bar["open"], bar["low"], bar["close"]) or
            bar["low"] > min(bar["open"], bar["high"], bar["close"])):
            return {"status": "INVALID_BAR", "requested_time": requested, "bar_time": clock}
        if not isinstance(bar["volume"], (int, float)) or bar["volume"] <= 0:
            return {"status": "UNVERIFIABLE_VOLUME", "bar_time": clock}
        limit = state.get("limit_up" if side == "BUY" else "limit_down")
        if limit is not None and (bar["open"] >= limit if side == "BUY" else bar["open"] <= limit):
            return {"status": "UNVERIFIABLE_LIMIT_QUEUE", "bar_time": clock}
        if shares > bar["volume"] * MAX_BAR_PARTICIPATION:
            return {"status": "UNVERIFIABLE_VOLUME", "bar_time": clock}
        return {"status": "ESTIMATED_DELAYED" if clock != requested else "ESTIMATED",
                "requested_time": requested, "bar_time": clock, "bar": bar,
                "delay_reason": blocked.get("reason") if blocked else None}
    return {"status": "NO_PRINT_AFTER_EVENT" if blocked else "UNVERIFIABLE_MISSING_BAR",
            "requested_time": requested}


def settle(picks: dict, picks_bytes: bytes, snapshot: dict, snapshot_bytes: bytes,
           db_path: Path, market_events: dict | None = None) -> dict:
    date = picks.get("trade_date")
    result = {"schema_version": 1, "trade_date": date,
              "picks_sha256": hashlib.sha256(picks_bytes).hexdigest(),
              "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
              "initial_cash_jpy": CAPITAL_JPY, "fill_basis": "5-minute OHLCV proxy",
              "assumed_fee_each_side_pct": FEE_RATE * 100,
              "assumed_slippage_each_side_pct": SLIPPAGE_RATE * 100,
              "trades": [], "status": "UNSETTLED", "proxy_total_pnl_jpy": None}
    if (picks.get("snapshot_sha256") != result["snapshot_sha256"] or
        date != snapshot.get("trading_date")):
        result["reason"] = "Picks and frozen snapshot do not match"
        return result
    if market_events and market_events.get("trading_date") != date:
        result["reason"] = "Market-event date does not match"
        return result
    if picks.get("status") == "NO_TRADE" and not picks.get("recommendations"):
        result.update({"status": "NO_TRADE", "proxy_total_pnl_jpy": 0,
                       "proxy_final_assets_jpy": CAPITAL_JPY})
        return result
    if picks.get("status") != "PROVISIONAL_PICKS":
        result["reason"] = "Picks are not an approved provisional record"
        return result
    orders = picks.get("recommendations", [])
    if len(orders) > MAX_POSITIONS or len({x.get("ticker") for x in orders}) != len(orders):
        result["reason"] = "Position count or ticker uniqueness violated"
        return result
    if not db_path.is_file():
        result["reason"] = "5-minute database missing"
        return result

    remaining = CAPITAL_JPY
    unsettled = False
    total_pnl = 0.0
    states = (market_events or {}).get("tickers", {})
    with sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True) as db:
        for order in sorted(orders, key=lambda x: (x["buy_time_jst"], x["ticker"])):
            ticker, shares = order["ticker"], order["shares"]
            buy_at, sell_at = order["buy_time_jst"], order["sell_time_jst"]
            trade = {"ticker": ticker, "shares": shares,
                     "requested_buy_time": buy_at, "requested_sell_time": sell_at}
            if (not isinstance(shares, int) or shares <= 0 or shares % LOT_SIZE or
                buy_at not in EXPECTED_TIMES or sell_at not in EXPECTED_TIMES or
                buy_at >= sell_at or sell_at > LAST_CONTINUOUS_BAR):
                trade.update(status="INVALID_ORDER")
                unsettled = True
                result["trades"].append(trade)
                continue
            bars = get_bars(db, ticker, date)
            state = states.get(ticker, {})
            buy = resolve_order(bars, buy_at, "BUY", shares, state)
            trade["buy_resolution"] = {k: v for k, v in buy.items() if k != "bar"}
            if not buy["status"].startswith("ESTIMATED"):
                trade["status"] = "NO_CONFIRMED_ENTRY"
                unsettled = True
                result["trades"].append(trade)
                continue
            if buy["bar_time"] >= sell_at:
                trade["status"] = "ENTRY_AFTER_PLANNED_EXIT"
                result["trades"].append(trade)
                continue
            buy_price = buy["bar"]["open"] * (1 + SLIPPAGE_RATE)
            buy_cost = buy_price * shares * (1 + FEE_RATE)
            if buy_cost > remaining or buy_cost > MAX_PER_TICKER_JPY:
                trade["status"] = "REJECTED_CASH_LIMIT"
                result["trades"].append(trade)
                continue
            remaining -= buy_cost
            trade["proxy_buy_price_jpy"] = buy_price
            trade["proxy_buy_cost_jpy"] = buy_cost
            sell = resolve_order(bars, sell_at, "SELL", shares, state)
            trade["sell_resolution"] = {k: v for k, v in sell.items() if k != "bar"}
            if not sell["status"].startswith("ESTIMATED"):
                trade["status"] = "OPEN_UNSETTLED"
                unsettled = True
                result["trades"].append(trade)
                continue
            sell_price = sell["bar"]["open"] * (1 - SLIPPAGE_RATE)
            proceeds = sell_price * shares * (1 - FEE_RATE)
            pnl = proceeds - buy_cost
            worst_buy = buy["bar"]["high"] * (1 + SLIPPAGE_RATE) * shares * (1 + FEE_RATE)
            worst_sell = sell["bar"]["low"] * (1 - SLIPPAGE_RATE) * shares * (1 - FEE_RATE)
            trade.update(status="ESTIMATED_ROUND_TRIP", proxy_sell_price_jpy=sell_price,
                         proxy_pnl_jpy=pnl, pessimistic_bar_pnl_jpy=worst_sell - worst_buy)
            total_pnl += pnl
            result["trades"].append(trade)

    if not unsettled:
        result.update(status="ESTIMATED_COMPLETE", proxy_total_pnl_jpy=total_pnl,
                      proxy_final_assets_jpy=CAPITAL_JPY + total_pnl)
    else:
        result["reason"] = "At least one order or open position has no defensible fill estimate"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--picks", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--market-events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    picks_bytes, snapshot_bytes = args.picks.read_bytes(), args.snapshot.read_bytes()
    events = json.loads(args.market_events.read_text(encoding="utf-8")) if args.market_events else None
    report = settle(json.loads(picks_bytes), picks_bytes, json.loads(snapshot_bytes),
                    snapshot_bytes, args.db, events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    print(f"{args.output}: {report['status']}")
    return 0 if report["status"] in {"NO_TRADE", "ESTIMATED_COMPLETE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
