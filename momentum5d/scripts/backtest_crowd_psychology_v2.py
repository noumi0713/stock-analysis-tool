from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRAINING_CODES = {
    "285A", "6976", "593A", "278A", "5801", "5803", "6857", "6920", "9348", "7013", "4592"
}
LIQUIDITY_YEN = 500_000_000.0
HORIZON = 10
COOLDOWN = 5
MAX_ABS_DAILY_RETURN = 0.45  # conservative corporate-action / bad-bar guard


def scale(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return ((s - lo) / (hi - lo) * 100.0).clip(0.0, 100.0)


def rsi14(close: pd.Series) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-d.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.where(loss.ne(0.0), 100.0)


def dedupe(mask: pd.Series, cooldown: int = COOLDOWN) -> np.ndarray:
    x = mask.fillna(False).to_numpy(dtype=bool)
    y = np.zeros(len(x), dtype=bool)
    last = -9999
    for i, v in enumerate(x):
        if v and i - last >= cooldown:
            y[i] = True
            last = i
    return y


def period_label(d: pd.Timestamp) -> str:
    if d < pd.Timestamp("2024-08-21"):
        return "Y1_2023-08-21_2024-08-20"
    if d < pd.Timestamp("2025-08-21"):
        return "Y2_2024-08-21_2025-08-20"
    return "Y3_2025-08-21_2026-08-21"


def sentiment_bucket(v: float) -> str:
    if v < 20:
        return "0-20 極端な悲観"
    if v < 40:
        return "20-40 悲観"
    if v < 60:
        return "40-60 中立"
    if v < 80:
        return "60-80 楽観"
    return "80-100 熱狂"


def load_prices(data_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    dates = pd.to_datetime(manifest["dates"])
    frames: list[pd.DataFrame] = []
    for shard in manifest["shards"]:
        payload = json.loads((data_dir / shard["path"]).read_text(encoding="utf-8"))
        for ticker, bars in payload["bars"].items():
            if not bars:
                continue
            a = np.asarray(bars, dtype=float)
            frames.append(pd.DataFrame({
                "ticker": ticker,
                "date": dates[a[:, 0].astype(int)],
                "open": a[:, 1], "high": a[:, 2], "low": a[:, 3],
                "close": a[:, 4], "volume": a[:, 5],
            }))
    out = pd.concat(frames, ignore_index=True)
    out.sort_values(["ticker", "date"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out, manifest


def trimmed_mean(s: pd.Series, p: float = 0.01) -> float | None:
    s = s.dropna()
    if s.empty:
        return None
    if len(s) < 50:
        return float(s.mean())
    lo, hi = s.quantile([p, 1.0 - p])
    t = s[(s >= lo) & (s <= hi)]
    return float(t.mean()) if not t.empty else None


def metric_summary(rows: pd.DataFrame, baseline_trimmed: float | None = None) -> dict[str, Any]:
    rows = rows[rows["fwd10"].notna()].copy()
    if rows.empty:
        return {k: None for k in [
            "mean_fwd10", "trimmed_mean_fwd10", "median_fwd10", "win_rate",
            "plus5_rate", "minus5_rate", "avg_mfe10", "avg_mae10", "edge_vs_baseline_trimmed"
        ]} | {"count": 0}
    tm = trimmed_mean(rows["fwd10"])
    return {
        "count": int(len(rows)),
        "mean_fwd10": float(rows["fwd10"].mean()),
        "trimmed_mean_fwd10": tm,
        "median_fwd10": float(rows["fwd10"].median()),
        "win_rate": float((rows["fwd10"] > 0).mean()),
        "plus5_rate": float((rows["mfe10"] >= 0.05).mean()),
        "minus5_rate": float((rows["mae10"] <= -0.05).mean()),
        "avg_mfe10": float(rows["mfe10"].mean()),
        "avg_mae10": float(rows["mae10"].mean()),
        "edge_vs_baseline_trimmed": None if baseline_trimmed is None or tm is None else float(tm - baseline_trimmed),
    }


def pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.2f}%"


def run(data_dir: Path) -> dict[str, Any]:
    prices, manifest = load_prices(data_dir)
    stock_meta = manifest.get("stocks", {})
    signals: list[dict[str, Any]] = []
    baselines: list[pd.DataFrame] = []
    quality_excluded = 0
    quality_candidate = 0

    for ticker, g in prices.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True).copy()
        if len(g) < 20:
            continue

        c, h, l, v = g["close"], g["high"], g["low"], g["volume"]
        turn = c * v
        g["history_n"] = np.arange(1, len(g) + 1)
        g["ret1"] = c.pct_change()
        g["ret5"] = c.pct_change(5)
        g["ret10"] = c.pct_change(10)
        g["ret20"] = c.pct_change(20)
        g["ma10"] = c.rolling(10, min_periods=10).mean()
        g["ma25"] = c.rolling(25, min_periods=25).mean()
        g["ma75"] = c.rolling(75, min_periods=75).mean()
        g["ma10_slope5"] = g["ma10"] / g["ma10"].shift(5) - 1
        g["ma25_slope5"] = g["ma25"] / g["ma25"].shift(5) - 1
        g["rsi14"] = rsi14(c)
        g["high10"] = h.rolling(10, min_periods=10).max()
        g["high20"] = h.rolling(20, min_periods=15).max()
        g["prev_high20"] = h.rolling(20, min_periods=15).max().shift(1)
        g["vol_avg20"] = v.rolling(20, min_periods=10).mean().shift(1)
        g["vol_ratio"] = v / g["vol_avg20"].replace(0.0, np.nan)
        g["turnover_med20"] = turn.rolling(20, min_periods=10).median()
        g["day_pos"] = (c - l) / (h - l).replace(0.0, np.nan)

        ma_base = g["ma25"].where(g["ma25"].notna(), g["ma10"])
        ret_base = g["ret20"].where(g["ret20"].notna(), g["ret10"])
        high_base = g["high20"].where(g["high20"].notna(), g["high10"])
        g["ma_gap"] = c / ma_base - 1
        g["drawdown_high"] = c / high_base - 1
        signed_vol = (g["vol_ratio"] - 1.0) * np.sign(g["ret5"].fillna(0.0))
        g["sentiment"] = (
            0.40 * g["rsi14"]
            + 0.25 * scale(ret_base, -0.20, 0.30)
            + 0.20 * scale(g["ma_gap"], -0.10, 0.15)
            + 0.10 * scale(c / high_base, 0.75, 1.00)
            + 0.05 * scale(signed_vol, -1.5, 1.5)
        ).clip(0, 100)

        # Data-quality gate. We conservatively reject windows around impossible/suspicious
        # one-day adjusted moves and OHLC inconsistencies before evaluating expectancy.
        bad = (
            (g["open"] <= 0) | (h <= 0) | (l <= 0) | (c <= 0) | (v < 0)
            | (h < g[["open", "close"]].max(axis=1))
            | (l > g[["open", "close"]].min(axis=1))
            | (g["ret1"].abs() > MAX_ABS_DAILY_RETURN)
        ).fillna(False)
        bad_past = bad.rolling(75, min_periods=1).max().astype(bool)
        bad_future = pd.concat([bad.shift(-i).fillna(False) for i in range(1, HORIZON + 1)], axis=1).any(axis=1)
        quality = ~bad_past & ~bad_future

        future_high = pd.concat([h.shift(-i) for i in range(1, HORIZON + 1)], axis=1).max(axis=1)
        future_low = pd.concat([l.shift(-i) for i in range(1, HORIZON + 1)], axis=1).min(axis=1)
        g["fwd10"] = c.shift(-HORIZON) / c - 1
        g["mfe10"] = future_high / c - 1
        g["mae10"] = future_low / c - 1
        g.loc[g["fwd10"].isna(), ["mfe10", "mae10"]] = np.nan
        g["period"] = g["date"].map(period_label)

        liquid_raw = g["turnover_med20"] >= LIQUIDITY_YEN
        possible = liquid_raw & g["sentiment"].notna() & g["fwd10"].notna()
        quality_candidate += int(possible.sum())
        quality_excluded += int((possible & ~quality).sum())
        tradable = liquid_raw & quality

        long_trend = (
            (g["ma25"].notna() & g["ma75"].notna() & (c > g["ma25"]) & (g["ma25"] > g["ma75"]) & (g["ma25_slope5"] > 0))
            | (g["ma25"].notna() & g["ma75"].isna() & (c > g["ma25"]) & (g["ma25_slope5"] > 0))
        )
        ipo_trend = (
            (g["history_n"] >= 15) & (g["history_n"] < 60) & g["ma10"].notna()
            & (c > g["ma10"]) & (g["ma10_slope5"] > 0) & (g["ret10"] > 0.05)
        )
        breakout = (c >= g["prev_high20"] * 0.98) & (g["ret5"] > 0) & (g["vol_ratio"] >= 1.0)
        pullback = (
            g["drawdown_high"].between(-0.10, -0.02) & (c > c.shift(1))
            & (g["day_pos"] >= 0.60) & (g["vol_ratio"] >= 0.70)
        )
        ipo_breakout = ipo_trend & (c / g["high10"] >= 0.94) & (g["vol_ratio"] >= 1.0)

        blue = (
            tradable & (long_trend | ipo_trend) & g["sentiment"].between(60, 80)
            & g["rsi14"].between(52, 72) & g["ma_gap"].between(-0.01, 0.12)
            & (breakout | pullback | ipo_breakout)
        )
        contrarian = (
            tradable & (g["sentiment"] <= 35) & (g["rsi14"] <= 38)
            & ((g["ret5"] <= -0.08) | (g["drawdown_high"] <= -0.15))
            & (g["vol_ratio"] >= 1.30) & (c > g["open"]) & (c > c.shift(1))
            & (g["day_pos"] >= 0.65)
        )
        euphoria = (
            tradable & (g["sentiment"] >= 80) & (g["rsi14"] >= 72)
            & ((g["ma_gap"] >= 0.12) | (g["ret10"] >= 0.15))
            & ((g["vol_ratio"] >= 1.30) | (g["ret5"] >= 0.10))
            & (g["drawdown_high"] >= -0.04)
        )

        code = str(stock_meta.get(ticker, {}).get("code") or ticker.removesuffix(".T"))[:4]
        training = code in TRAINING_CODES
        eligible = tradable & g["sentiment"].notna() & g["fwd10"].notna()
        b = g.loc[eligible, ["ticker", "date", "fwd10", "mfe10", "mae10", "sentiment", "period"]].copy()
        b["is_training"] = training
        b["bucket"] = b["sentiment"].map(sentiment_bucket)
        baselines.append(b)

        for name, mask in [("BLUE_FOLLOW", blue), ("RED_CONTRARIAN_BUY", contrarian), ("RED_EUPHORIA_AVOID", euphoria)]:
            for i in np.flatnonzero(dedupe(mask)):
                r = g.iloc[i]
                signals.append({
                    "ticker": ticker, "code": code,
                    "name": str(stock_meta.get(ticker, {}).get("name") or ticker),
                    "date": r["date"].strftime("%Y-%m-%d"), "period": r["period"],
                    "signal": name, "is_training": training, "close": float(r["close"]),
                    "sentiment": float(r["sentiment"]) if pd.notna(r["sentiment"]) else None,
                    "rsi14": float(r["rsi14"]) if pd.notna(r["rsi14"]) else None,
                    "ma_gap": float(r["ma_gap"]) if pd.notna(r["ma_gap"]) else None,
                    "ret10": float(r["ret10"]) if pd.notna(r["ret10"]) else None,
                    "vol_ratio": float(r["vol_ratio"]) if pd.notna(r["vol_ratio"]) else None,
                    "fwd10": float(r["fwd10"]) if pd.notna(r["fwd10"]) else None,
                    "mfe10": float(r["mfe10"]) if pd.notna(r["mfe10"]) else None,
                    "mae10": float(r["mae10"]) if pd.notna(r["mae10"]) else None,
                })

    base = pd.concat(baselines, ignore_index=True)
    sig = pd.DataFrame(signals)
    if sig.empty:
        sig = pd.DataFrame(columns=["signal", "is_training", "period", "code", "fwd10", "mfe10", "mae10"])
    hold = base[~base["is_training"]].copy()
    base_tm = trimmed_mean(hold["fwd10"])

    result: dict[str, Any] = {
        "meta": {
            "data_start": manifest["meta"]["startDate"], "data_end": manifest["meta"]["endDate"],
            "stock_count": manifest["meta"]["stockCount"], "date_count": manifest["meta"]["dateCount"],
            "training_codes": sorted(TRAINING_CODES), "horizon": HORIZON,
            "liquidity_filter_yen": LIQUIDITY_YEN, "cooldown": COOLDOWN,
            "max_abs_daily_return_quality_guard": MAX_ABS_DAILY_RETURN,
            "quality_candidate_rows": quality_candidate, "quality_excluded_rows": quality_excluded,
            "optimization": "none; fixed initial thresholds",
        },
        "rules": {
            "sentiment": "40% RSI14 + 25% return + 20% MA gap + 10% distance to recent high + 5% signed volume acceleration",
            "blue": "60-80 sentiment + positive trend + RSI52-72 + not stretched + breakout/first pullback",
            "contrarian": "<=35 sentiment + RSI<=38 + selloff + volume capitulation + same-day reversal confirmation",
            "euphoria": ">=80 sentiment + RSI>=72 + stretch + price/volume acceleration near recent high",
        },
        "holdout": {}, "sentiment_buckets": {}, "periods": {}, "training_audit": {},
    }
    result["holdout"]["BASELINE"] = metric_summary(hold)
    for name in ["BLUE_FOLLOW", "RED_CONTRARIAN_BUY", "RED_EUPHORIA_AVOID"]:
        result["holdout"][name] = metric_summary(sig[(sig["signal"] == name) & (~sig["is_training"])], base_tm)

    for bucket in ["0-20 極端な悲観", "20-40 悲観", "40-60 中立", "60-80 楽観", "80-100 熱狂"]:
        result["sentiment_buckets"][bucket] = metric_summary(hold[hold["bucket"] == bucket], base_tm)

    for p in sorted(hold["period"].unique()):
        pb = hold[hold["period"] == p]
        ptm = trimmed_mean(pb["fwd10"])
        result["periods"][p] = {"BASELINE": metric_summary(pb)}
        for name in ["BLUE_FOLLOW", "RED_CONTRARIAN_BUY", "RED_EUPHORIA_AVOID"]:
            result["periods"][p][name] = metric_summary(sig[(sig["signal"] == name) & (~sig["is_training"]) & (sig["period"] == p)], ptm)

    ts = sig[sig["is_training"]].copy()
    for code in sorted(TRAINING_CODES):
        r = ts[ts["code"] == code].sort_values("date")
        result["training_audit"][code] = {
            "signals": int(len(r)), "full_outcomes": int(r["fwd10"].notna().sum()) if not r.empty else 0,
            "by_signal": {name: metric_summary(r[r["signal"] == name], base_tm) for name in ["BLUE_FOLLOW", "RED_CONTRARIAN_BUY", "RED_EUPHORIA_AVOID"]},
            "latest_events": r.tail(8).to_dict(orient="records") if not r.empty else [],
        }
    return result


def render(r: dict[str, Any]) -> str:
    m = r["meta"]
    out = [
        "# Crowd Psychology 3Y Backtest v2", "",
        f"- Data: {m['data_start']} to {m['data_end']}",
        f"- Universe: all TSE, {m['stock_count']} stocks / {m['date_count']} market days",
        "- Primary test excludes the 11 teaching stocks", "- No grid search / no parameter optimization",
        f"- Liquidity: 20d median turnover >= {m['liquidity_filter_yen']/1e8:.1f}億円",
        f"- Data-quality guard: reject windows around abs 1d adjusted return > {m['max_abs_daily_return_quality_guard']*100:.0f}% or invalid OHLC",
        f"- Quality rows excluded: {m['quality_excluded_rows']} / {m['quality_candidate_rows']}", "",
        "## Holdout results", "",
        "| Class | N | Mean10d | TrimMean10d | Median10d | Win | +5% MFE | -5% MAE | Avg MFE | Avg MAE | Edge(trim) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in ["BASELINE", "BLUE_FOLLOW", "RED_CONTRARIAN_BUY", "RED_EUPHORIA_AVOID"]:
        s = r["holdout"][k]
        out.append(f"| {k} | {s['count']} | {pct(s['mean_fwd10'])} | {pct(s['trimmed_mean_fwd10'])} | {pct(s['median_fwd10'])} | {pct(s['win_rate'])} | {pct(s['plus5_rate'])} | {pct(s['minus5_rate'])} | {pct(s['avg_mfe10'])} | {pct(s['avg_mae10'])} | {pct(s['edge_vs_baseline_trimmed'])} |")

    out += ["", "## Sentiment buckets", "", "| Sentiment | N | TrimMean10d | Median | Win | +5% MFE | -5% MAE | Edge(trim) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, s in r["sentiment_buckets"].items():
        out.append(f"| {k} | {s['count']} | {pct(s['trimmed_mean_fwd10'])} | {pct(s['median_fwd10'])} | {pct(s['win_rate'])} | {pct(s['plus5_rate'])} | {pct(s['minus5_rate'])} | {pct(s['edge_vs_baseline_trimmed'])} |")

    out += ["", "## Period stability", ""]
    for p, d in r["periods"].items():
        out += [f"### {p}", "", "| Class | N | TrimMean10d | Median | Win | +5% MFE | -5% MAE | Edge(trim) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for k in ["BASELINE", "BLUE_FOLLOW", "RED_CONTRARIAN_BUY", "RED_EUPHORIA_AVOID"]:
            s = d[k]
            out.append(f"| {k} | {s['count']} | {pct(s['trimmed_mean_fwd10'])} | {pct(s['median_fwd10'])} | {pct(s['win_rate'])} | {pct(s['plus5_rate'])} | {pct(s['minus5_rate'])} | {pct(s['edge_vs_baseline_trimmed'])} |")
        out.append("")

    out += ["## Training-stock audit", "", "| Code | Signals | Full 10d outcomes |", "|---|---:|---:|"]
    for code, a in r["training_audit"].items():
        out.append(f"| {code} | {a['signals']} | {a['full_outcomes']} |")
    out += ["", "## Caveats", "", "- The 11 examples were selected after observing distinctive charts; selection bias is real.", "- This is a technical proxy for optimism/pessimism, not direct board/SNS text sentiment.", "- Stored 3y data ends 2026-08-21; September examples are outside this run.", "- This is research evidence, not clean time-OOS proof. A frozen rule needs a subsequent untouched forward/OOS test."]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("dashboard-data/technical-backtest-3y"))
    ap.add_argument("--output-json", type=Path, default=Path("crowd-psychology-backtest-v2.json"))
    ap.add_argument("--output-md", type=Path, default=Path("crowd-psychology-backtest-v2.md"))
    a = ap.parse_args()
    result = run(a.data_dir)
    a.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render(result)
    a.output_md.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
