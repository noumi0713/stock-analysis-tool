from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import test_psychology_11_roundtrips as base

COST_ONE_WAY = 0.001
MAX_HOLD = 10


def mark_simple_entries(g: pd.DataFrame) -> pd.DataFrame:
    g = base.enrich(g)
    c = g["close"]
    high_part = g["participation"].between(1.2, 2.0, inclusive="both")
    trend = (
        (g["ma25"].notna() & (c > g["ma25"]) & (g["ma25_slope5"] > 0))
        | ((g["history_n"] < 60) & g["ma10"].notna() & (c > g["ma10"]) & (g["ma10_slope5"] > 0))
    )
    g["blue_simple"] = (
        g["quality"] & g["liquid"] & high_part
        & g["sentiment"].between(60, 80) & trend
    )
    g["contra_simple"] = (
        g["quality"] & g["liquid"] & high_part
        & (g["sentiment"] <= 20)
        & (c > c.shift(1)) & (g["day_pos"] >= 0.60)
    )
    return g


def simulate(g: pd.DataFrame, mode: str) -> list[dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    pos: dict[str, Any] | None = None
    i = 0
    while i < len(g) - 1:
        r = g.iloc[i]
        if pos is None:
            sig = "BLUE" if bool(r.blue_simple) else ("CONTRARIAN" if bool(r.contra_simple) else None)
            if sig and (mode == "COMBINED" or mode == sig):
                er = g.iloc[i + 1]
                pos = {
                    "signal": sig,
                    "signal_date": r.date,
                    "entry_i": i + 1,
                    "entry_date": er.date,
                    "entry_price": float(er.open),
                    "entry_sentiment": float(r.sentiment),
                    "entry_participation": float(r.participation),
                }
                i += 1
        else:
            hold = i - pos["entry_i"] + 1
            exit_reason = None
            if r.sentiment >= 80:
                exit_reason = "EUPHORIA"
            elif pos["signal"] == "BLUE" and (r.sentiment < 50 or (pd.notna(r.ma25) and r.close < r.ma25)):
                exit_reason = "MOMENTUM_LOST"
            elif pos["signal"] == "CONTRARIAN" and r.sentiment >= 60:
                exit_reason = "SENTIMENT_NORMALIZED"
            elif hold >= MAX_HOLD:
                exit_reason = "MAX_10D"

            if exit_reason:
                xr = g.iloc[i + 1]
                ep, xp = pos["entry_price"], float(xr.open)
                path = g.iloc[pos["entry_i"]: i + 2]
                trades.append({
                    **pos,
                    "exit_signal_date": r.date,
                    "exit_date": xr.date,
                    "exit_price": xp,
                    "exit_reason": exit_reason,
                    "hold_days": int((i + 1) - pos["entry_i"]),
                    "gross_return": xp / ep - 1,
                    "net_return": (xp * (1 - COST_ONE_WAY)) / (ep * (1 + COST_ONE_WAY)) - 1,
                    "mfe": float(path.high.max() / ep - 1),
                    "mae": float(path.low.min() / ep - 1),
                })
                pos = None
                i += 1
        i += 1
    return trades


def pct(x):
    return "-" if x is None else f"{100*x:.2f}%"


def num(x):
    return "-" if x is None else f"{x:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    a = ap.parse_args()

    raw, m = base.load_prices(a.data_dir)
    rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    for code, g0 in raw.items():
        g = mark_simple_entries(g0)
        counts[code] = {
            "blue_signal_days": int(g.blue_simple.sum()),
            "contra_signal_days": int(g.contra_simple.sum()),
        }
        for mode in ["BLUE", "CONTRARIAN", "COMBINED"]:
            for tr in simulate(g, mode):
                tr["code"] = code
                tr["name"] = str(g0.stock_name.iloc[0])
                tr["mode"] = mode
                rows.append(tr)

    t = pd.DataFrame(rows)
    for col in ["signal_date", "entry_date", "exit_signal_date", "exit_date"]:
        if col in t.columns:
            t[col] = pd.to_datetime(t[col]).dt.strftime("%Y-%m-%d")

    overall = {}
    by_code = {}
    for mode in ["BLUE", "CONTRARIAN", "COMBINED"]:
        overall[mode] = base.summarize(t[t.mode == mode].copy()) if not t.empty else {"n": 0}
    for code in sorted(raw):
        by_code[code] = {}
        for mode in ["BLUE", "CONTRARIAN", "COMBINED"]:
            by_code[code][mode] = base.summarize(t[(t.code == code) & (t.mode == mode)].copy()) if not t.empty else {"n": 0}

    out = {
        "meta": {
            "data_start": m["dates"][0], "data_end": m["dates"][-1],
            "participation": "1.2x-2.0x current turnover / 20d median",
            "cost_one_way": COST_ONE_WAY, "max_hold": MAX_HOLD,
            "note": "Exploratory direct state-machine after strict rule produced zero trades; no grid search used."
        },
        "signal_counts": counts,
        "overall": overall,
        "by_code": by_code,
        "trades": t.to_dict("records") if not t.empty else [],
    }
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 11 stocks direct psychology-state round-trip test", "",
        f"- Data: {m['dates'][0]} to {m['dates'][-1]}",
        "- Entry: next-day open", "- Exit: next-day open", "- Cost: 0.10% each side",
        "- Participation: 1.2x–2.0x current turnover / 20d median", "- No grid search", "",
        "## Rules", "",
        "- BLUE: psychology 60–80 + participation 1.2–2.0x + rising trend.",
        "- CONTRARIAN: psychology <=20 + participation 1.2–2.0x + bullish reversal day.",
        "- BLUE exit: psychology >=80, psychology <50 / MA25 break, or 10 days.",
        "- CONTRARIAN exit: psychology >=60 or 10 days.", "",
        "## Overall", "",
        "| Mode | N | Mean net | Median | Win | PF | Compound* | Max DD* | Avg MFE | Avg MAE | Avg hold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ["BLUE", "CONTRARIAN", "COMBINED"]:
        s = overall[mode]
        lines.append(f"| {mode} | {s.get('n',0)} | {pct(s.get('mean_net'))} | {pct(s.get('median_net'))} | {pct(s.get('win_rate'))} | {num(s.get('pf'))} | {pct(s.get('compound'))} | {pct(s.get('max_dd'))} | {pct(s.get('avg_mfe'))} | {pct(s.get('avg_mae'))} | {num(s.get('avg_hold'))} |")
    lines += ["", "## By code (COMBINED)", "", "| Code | Signal days B/C | Trades | Mean net | Win | PF | Compound | Max DD |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for code in sorted(raw):
        s = by_code[code]["COMBINED"]
        cc = counts[code]
        lines.append(f"| {code} | {cc['blue_signal_days']}/{cc['contra_signal_days']} | {s.get('n',0)} | {pct(s.get('mean_net'))} | {pct(s.get('win_rate'))} | {num(s.get('pf'))} | {pct(s.get('compound'))} | {pct(s.get('max_dd'))} |")
    lines += ["", "* Compound/Max DD are sequential-trade statistics, not a portfolio capital-allocation simulation.", "", "This is an in-sample teaching-stock audit; the 11 stocks were selected after observing their charts."]
    a.output_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
