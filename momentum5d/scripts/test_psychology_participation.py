from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_crowd_psychology_v2 import (
    TRAINING_CODES, HORIZON, MAX_ABS_DAILY_RETURN,
    load_prices, rsi14, scale, trimmed_mean, sentiment_bucket,
)


def participation_bucket(turnover_med20: float) -> str:
    if turnover_med20 < 2_000_000_000:
        return "P1 0.5-2B"
    if turnover_med20 < 10_000_000_000:
        return "P2 2-10B"
    if turnover_med20 < 50_000_000_000:
        return "P3 10-50B"
    return "P4 50B+"


def activity_bucket(activity_ratio: float) -> str:
    if activity_ratio < 0.8:
        return "A1 <0.8x"
    if activity_ratio < 1.2:
        return "A2 0.8-1.2x"
    if activity_ratio < 2.0:
        return "A3 1.2-2.0x"
    return "A4 2.0x+"


def stats(rows: pd.DataFrame) -> dict:
    rows = rows.loc[rows.fwd10.notna()].copy()
    if rows.empty:
        return {"n": 0}
    return {
        "n": int(len(rows)),
        "trim10": trimmed_mean(rows.fwd10),
        "median10": float(rows.fwd10.median()),
        "win": float((rows.fwd10 > 0).mean()),
        "plus5": float((rows.mfe10 >= 0.05).mean()),
        "minus5": float((rows.mae10 <= -0.05).mean()),
    }


def run(data_dir: Path) -> dict:
    prices, manifest = load_prices(data_dir)
    meta = manifest.get("stocks", {})
    out_frames = []

    for ticker, g in prices.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True).copy()
        if len(g) < 25:
            continue
        c, h, l, v = g.close, g.high, g.low, g.volume
        turn = c * v
        g["ret1"] = c.pct_change()
        g["ret5"] = c.pct_change(5)
        g["ret10"] = c.pct_change(10)
        g["ret20"] = c.pct_change(20)
        g["ma10"] = c.rolling(10, min_periods=10).mean()
        g["ma25"] = c.rolling(25, min_periods=25).mean()
        g["rsi14"] = rsi14(c)
        g["high10"] = h.rolling(10, min_periods=10).max()
        g["high20"] = h.rolling(20, min_periods=15).max()
        g["vol_avg20"] = v.rolling(20, min_periods=10).mean().shift(1)
        g["vol_ratio"] = v / g.vol_avg20.replace(0.0, np.nan)
        g["turnover_med20"] = turn.rolling(20, min_periods=10).median()
        g["turnover_today"] = turn
        g["activity_ratio"] = g.turnover_today / g.turnover_med20.replace(0.0, np.nan)

        ma_base = g.ma25.where(g.ma25.notna(), g.ma10)
        ret_base = g.ret20.where(g.ret20.notna(), g.ret10)
        high_base = g.high20.where(g.high20.notna(), g.high10)
        g["ma_gap"] = c / ma_base - 1.0
        signed_vol = (g.vol_ratio - 1.0) * np.sign(g.ret5.fillna(0.0))
        g["sentiment"] = (
            0.40 * g.rsi14
            + 0.25 * scale(ret_base, -0.20, 0.30)
            + 0.20 * scale(g.ma_gap, -0.10, 0.15)
            + 0.10 * scale(c / high_base, 0.75, 1.00)
            + 0.05 * scale(signed_vol, -1.5, 1.5)
        ).clip(0, 100)

        bad = (
            (g.open <= 0) | (h <= 0) | (l <= 0) | (c <= 0) | (v < 0)
            | (h < g[["open", "close"]].max(axis=1))
            | (l > g[["open", "close"]].min(axis=1))
            | (g.ret1.abs() > MAX_ABS_DAILY_RETURN)
        ).fillna(False)
        bad_past = bad.rolling(75, min_periods=1).max().astype(bool)
        bad_future = pd.concat([bad.shift(-i).fillna(False) for i in range(1, HORIZON + 1)], axis=1).any(axis=1)
        quality = ~bad_past & ~bad_future

        fh = pd.concat([h.shift(-i) for i in range(1, HORIZON + 1)], axis=1).max(axis=1)
        fl = pd.concat([l.shift(-i) for i in range(1, HORIZON + 1)], axis=1).min(axis=1)
        g["fwd10"] = c.shift(-HORIZON) / c - 1.0
        g["mfe10"] = fh / c - 1.0
        g["mae10"] = fl / c - 1.0

        code = str(meta.get(ticker, {}).get("code") or ticker.removesuffix(".T"))[:4]
        if code in TRAINING_CODES:
            continue
        eligible = (
            quality & g.sentiment.notna() & g.fwd10.notna()
            & (g.turnover_med20 >= 500_000_000)
        )
        x = g.loc[eligible, ["date", "sentiment", "fwd10", "mfe10", "mae10", "turnover_med20", "activity_ratio"]].copy()
        x["ticker"] = ticker
        x["p_bucket"] = x.turnover_med20.map(participation_bucket)
        x["a_bucket"] = x.activity_ratio.map(activity_bucket)
        x["s_bucket"] = x.sentiment.map(sentiment_bucket)
        out_frames.append(x)

    all_rows = pd.concat(out_frames, ignore_index=True)

    result = {"meta": {
        "data_start": manifest["meta"]["startDate"],
        "data_end": manifest["meta"]["endDate"],
        "rows": int(len(all_rows)),
        "note": "Participant count is unavailable in OHLCV; 20d median yen turnover is the persistent-participation proxy and today's turnover/20d median is the activity-surge proxy.",
    }, "persistent_participation": {}, "activity_surge": {}}

    for key, group_col in [("persistent_participation", "p_bucket"), ("activity_surge", "a_bucket")]:
        for bucket, b in all_rows.groupby(group_col, sort=True):
            corr = b[["sentiment", "fwd10"]].corr(method="spearman").iloc[0, 1]
            sentiment_stats = {sb: stats(rows) for sb, rows in b.groupby("s_bucket", sort=True)}
            pess = sentiment_stats.get("0-20 極端な悲観", {}).get("trim10")
            eup = sentiment_stats.get("80-100 熱狂", {}).get("trim10")
            spread = None if pess is None or eup is None else float(pess - eup)
            result[key][bucket] = {
                "n": int(len(b)),
                "spearman_sentiment_vs_fwd10": None if pd.isna(corr) else float(corr),
                "pessimism_minus_euphoria_trim10": spread,
                "sentiment": sentiment_stats,
            }

    # Combined: persistent high participation x current activity surge.
    all_rows["combined"] = all_rows.p_bucket.astype(str) + " | " + all_rows.a_bucket.astype(str)
    result["combined"] = {}
    for bucket, b in all_rows.groupby("combined", sort=True):
        if len(b) < 500:
            continue
        corr = b[["sentiment", "fwd10"]].corr(method="spearman").iloc[0, 1]
        sstats = {sb: stats(rows) for sb, rows in b.groupby("s_bucket", sort=True)}
        pess = sstats.get("0-20 極端な悲観", {}).get("trim10")
        eup = sstats.get("80-100 熱狂", {}).get("trim10")
        result["combined"][bucket] = {
            "n": int(len(b)),
            "spearman": None if pd.isna(corr) else float(corr),
            "spread": None if pess is None or eup is None else float(pess - eup),
        }
    return result


def fmt(v):
    return "-" if v is None else f"{v*100:.2f}%"


def render_md(r: dict) -> str:
    lines = [
        "# Psychology index × participation test", "",
        f"- Data: {r['meta']['data_start']} to {r['meta']['data_end']}",
        f"- Holdout rows: {r['meta']['rows']:,}",
        "- 11 teaching stocks excluded", "- No threshold optimization", "",
        "## Persistent participation: 20d median turnover", "",
        "| Bucket | N | Spearman(sentiment, 10d) | Extreme pessimism - euphoria | Pessimism trim10 | Euphoria trim10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k, v in r["persistent_participation"].items():
        ss = v["sentiment"]
        p = ss.get("0-20 極端な悲観", {}).get("trim10")
        e = ss.get("80-100 熱狂", {}).get("trim10")
        lines.append(f"| {k} | {v['n']:,} | {v['spearman_sentiment_vs_fwd10']:.4f} | {fmt(v['pessimism_minus_euphoria_trim10'])} | {fmt(p)} | {fmt(e)} |")
    lines += ["", "## Current activity surge: today's turnover / 20d median", "",
        "| Bucket | N | Spearman(sentiment, 10d) | Extreme pessimism - euphoria | Pessimism trim10 | Euphoria trim10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k, v in r["activity_surge"].items():
        ss = v["sentiment"]
        p = ss.get("0-20 極端な悲観", {}).get("trim10")
        e = ss.get("80-100 熱狂", {}).get("trim10")
        lines.append(f"| {k} | {v['n']:,} | {v['spearman_sentiment_vs_fwd10']:.4f} | {fmt(v['pessimism_minus_euphoria_trim10'])} | {fmt(p)} | {fmt(e)} |")
    lines += ["", "Interpretation: a more negative Spearman and a larger pessimism-minus-euphoria spread mean the psychology index is more discriminating in that participation regime.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    a = ap.parse_args()
    r = run(a.data_dir)
    a.output_json.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    a.output_md.write_text(render_md(r), encoding="utf-8")
    print(render_md(r))


if __name__ == "__main__":
    main()
