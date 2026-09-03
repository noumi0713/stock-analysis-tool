from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dashboard-data" / "technical-backtest-3y"
OUT_DIR = ROOT / "research" / "results" / "volume_top100_overheat_3y"
HORIZONS = (5, 10, 20)
MIN_PRICE = 100.0
MIN_TURNOVER20 = 120_000_000.0
TOP_N = 100
MAX_MAIN_SIGNALS_PER_DAY = 5


def mean(xs):
    return statistics.fmean(xs) if xs else None


def median(xs):
    return statistics.median(xs) if xs else None


def pct(x):
    return None if x is None else round(x * 100.0, 4)


def safe_div(a, b, default=0.0):
    return a / b if b and math.isfinite(b) else default


def summarize(vals):
    xs = [float(x) for x in vals if x is not None and math.isfinite(float(x))]
    if not xs:
        return {"n": 0, "mean_pct": None, "median_pct": None, "win_rate_pct": None}
    return {
        "n": len(xs),
        "mean_pct": pct(mean(xs)),
        "median_pct": pct(median(xs)),
        "win_rate_pct": round(sum(x > 0 for x in xs) / len(xs) * 100.0, 2),
    }


def load_bars():
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    merged = defaultdict(list)
    for shard in manifest["shards"]:
        payload = json.loads((DATA_DIR / shard["path"]).read_text(encoding="utf-8"))
        for ticker, bars in payload["bars"].items():
            merged[ticker].extend(bars)
    all_bars = {}
    for ticker, bars in merged.items():
        # Monthly shards can overlap. Keep one bar per global date index.
        by_idx = {int(b[0]): b for b in bars}
        all_bars[ticker] = [by_idx[k] for k in sorted(by_idx)]
    return manifest, all_bars


def avg_turnover(bars, p, n=20):
    start = max(0, p - n + 1)
    vals = [float(x[4]) * float(x[5]) for x in bars[start : p + 1] if float(x[4]) > 0 and float(x[5]) >= 0]
    return mean(vals) or 0.0


def sma_close(bars, p, n):
    if p + 1 < n:
        return None
    return mean([float(x[4]) for x in bars[p - n + 1 : p + 1]])


def rsi14(bars, p):
    if p < 14:
        return None
    gains, losses = [], []
    for j in range(p - 13, p + 1):
        d = float(bars[j][4]) - float(bars[j - 1][4])
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag = mean(gains) or 0.0
    al = mean(losses) or 0.0
    if al == 0:
        return 100.0 if ag > 0 else 50.0
    rs = ag / al
    return 100.0 - 100.0 / (1.0 + rs)


def atr14(bars, p):
    if p < 14:
        return None
    trs = []
    for j in range(p - 13, p + 1):
        hi, lo = float(bars[j][2]), float(bars[j][3])
        prev = float(bars[j - 1][4])
        trs.append(max(hi - lo, abs(hi - prev), abs(lo - prev)))
    return mean(trs)


def overheat_features(bars, p):
    if p < 20:
        return None
    b = bars[p]
    prev = bars[p - 1]
    close = float(b[4])
    open_ = float(b[1])
    high = float(b[2])
    low = float(b[3])
    prev_close = float(prev[4])
    vol = float(b[5])
    prev_vol = float(prev[5])
    if close <= 0 or prev_close <= 0 or vol < 0 or prev_vol <= 0:
        return None

    turnover20 = avg_turnover(bars, p, 20)
    if close < MIN_PRICE or turnover20 < MIN_TURNOVER20:
        return None

    vol_change = vol / prev_vol - 1.0
    ret1 = close / prev_close - 1.0
    ret5 = close / float(bars[p - 5][4]) - 1.0
    ret20 = close / float(bars[p - 20][4]) - 1.0
    gap = open_ / prev_close - 1.0 if open_ > 0 else 0.0
    ma20 = sma_close(bars, p, 20) or close
    ma_dev20 = close / ma20 - 1.0 if ma20 > 0 else 0.0
    rsi = rsi14(bars, p)
    atr = atr14(bars, p) or 0.0
    range_atr = (high - low) / atr if atr > 0 else 0.0
    high20 = max(float(x[2]) for x in bars[p - 19 : p + 1])
    dist_high20 = high20 / close - 1.0 if close > 0 else 9.0
    close_location = (close - low) / (high - low) if high > low else 0.5

    # 100-point overheat risk. Price extension carries most of the weight.
    score = 0
    # 1-day price shock: max 15
    if ret1 >= 0.03: score += 5
    if ret1 >= 0.05: score += 5
    if ret1 >= 0.08: score += 5
    # 5-day extension: max 20
    if ret5 >= 0.05: score += 5
    if ret5 >= 0.10: score += 5
    if ret5 >= 0.15: score += 5
    if ret5 >= 0.22: score += 5
    # 20-day extension: max 15
    if ret20 >= 0.12: score += 5
    if ret20 >= 0.25: score += 5
    if ret20 >= 0.40: score += 5
    # RSI: max 15
    if rsi is not None and rsi >= 65: score += 5
    if rsi is not None and rsi >= 72: score += 5
    if rsi is not None and rsi >= 80: score += 5
    # 20-day MA extension: max 15
    if ma_dev20 >= 0.05: score += 5
    if ma_dev20 >= 0.10: score += 5
    if ma_dev20 >= 0.15: score += 5
    # Gap and range expansion: max 10
    if gap >= 0.04: score += 5
    if range_atr >= 1.8: score += 5
    # Extreme volume shock can itself indicate climax: max 10
    if vol_change >= 3.0: score += 5
    if vol_change >= 6.0: score += 5

    # Demand confirmation is deliberately secondary to overheat.
    confirm = 0
    if ret1 > 0: confirm += 6
    if 0.55 <= close_location: confirm += 6
    if close > ma20: confirm += 6
    if dist_high20 <= 0.08: confirm += 6
    if gap < 0.04: confirm += 3
    if range_atr < 1.8: confirm += 3

    return {
        "idx": int(b[0]),
        "close": close,
        "volume_change": vol_change,
        "turnover20": turnover20,
        "ret1": ret1,
        "ret5": ret5,
        "ret20": ret20,
        "gap": gap,
        "rsi14": rsi,
        "ma_dev20": ma_dev20,
        "range_atr": range_atr,
        "dist_high20": dist_high20,
        "close_location": close_location,
        "overheat": score,
        "confirm": confirm,
        "signal_score": max(0, min(100, round((100 - score) * 0.70 + confirm, 2))),
    }


def evaluate(ticker, bars, p, feat, dates, strategy, volume_rank):
    if p + max(HORIZONS) >= len(bars):
        return None
    entry = bars[p + 1]
    entry_price = float(entry[1])
    if entry_price <= 0:
        return None
    row = {
        "strategy": strategy,
        "ticker": ticker,
        "signal_date": dates[int(bars[p][0])],
        "entry_date": dates[int(entry[0])],
        "entry_open": round(entry_price, 4),
        "volume_rank": volume_rank,
        "volume_change_pct": pct(feat["volume_change"]),
        "overheat_score": feat["overheat"],
        "confirmation_score": feat["confirm"],
        "signal_score": feat["signal_score"],
        "rsi14": round(feat["rsi14"], 2) if feat["rsi14"] is not None else None,
        "return_1d_pct": pct(feat["ret1"]),
        "return_5d_pct_at_signal": pct(feat["ret5"]),
        "return_20d_pct_at_signal": pct(feat["ret20"]),
        "ma20_deviation_pct": pct(feat["ma_dev20"]),
        "gap_pct": pct(feat["gap"]),
        "range_atr": round(feat["range_atr"], 3),
        "close_location": round(feat["close_location"], 3),
        "turnover20_yen": round(feat["turnover20"], 2),
    }
    for h in HORIZONS:
        exit_close = float(bars[p + h][4])
        window = bars[p + 1 : p + h + 1]
        r = exit_close / entry_price - 1.0
        mfe = max(float(x[2]) for x in window) / entry_price - 1.0
        mae = min(float(x[3]) for x in window) / entry_price - 1.0
        row[f"return_{h}d_pct"] = pct(r)
        row[f"mfe_{h}d_pct"] = pct(mfe)
        row[f"mae_{h}d_pct"] = pct(mae)
        row[f"hit_plus5_{h}d"] = int(mfe >= 0.05)
        row[f"hit_minus5_{h}d"] = int(mae <= -0.05)
    return row


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest, all_bars = load_bars()
    dates = manifest["dates"]

    # Map global date index -> ticker/local position/features.
    daily = defaultdict(list)
    for ticker, bars in all_bars.items():
        if len(bars) < 45:
            continue
        for p in range(20, len(bars) - max(HORIZONS)):
            feat = overheat_features(bars, p)
            if feat is not None and feat["volume_change"] > 0:
                daily[feat["idx"]].append((ticker, p, feat, bars))

    events = []
    bucket_returns = defaultdict(lambda: defaultdict(list))
    strategy_returns = defaultdict(lambda: defaultdict(list))
    yearly_main = defaultdict(list)
    day_counts = []

    for idx in sorted(daily):
        ranked = sorted(daily[idx], key=lambda x: x[2]["volume_change"], reverse=True)[:TOP_N]
        if not ranked:
            continue
        day_counts.append(len(ranked))

        # Main signal: lowest overheat first, confirmation used only after overheat.
        main_candidates = []
        for rank, (ticker, p, feat, bars) in enumerate(ranked, start=1):
            bucket = "0-19" if feat["overheat"] < 20 else "20-39" if feat["overheat"] < 40 else "40-59" if feat["overheat"] < 60 else "60+"
            ev_all = evaluate(ticker, bars, p, feat, dates, "top100_all", rank)
            if ev_all:
                events.append(ev_all)
                for h in HORIZONS:
                    r = ev_all[f"return_{h}d_pct"] / 100.0
                    bucket_returns[bucket][h].append(r)
                    strategy_returns["top100_all"][h].append(r)

            for threshold in (20, 30, 40):
                if feat["overheat"] <= threshold:
                    ev = evaluate(ticker, bars, p, feat, dates, f"overheat_le_{threshold}", rank)
                    if ev:
                        events.append(ev)
                        for h in HORIZONS:
                            strategy_returns[f"overheat_le_{threshold}"][h].append(ev[f"return_{h}d_pct"] / 100.0)

            # Main detection rule: overheat <=30 and only mild/positive price response.
            if feat["overheat"] <= 30 and 0.0 < feat["ret1"] <= 0.05 and feat["close_location"] >= 0.55 and feat["confirm"] >= 18:
                main_candidates.append((feat["overheat"], -feat["confirm"], rank, ticker, p, feat, bars))

        main_candidates.sort()
        for _, _, rank, ticker, p, feat, bars in main_candidates[:MAX_MAIN_SIGNALS_PER_DAY]:
            ev = evaluate(ticker, bars, p, feat, dates, "main_top5", rank)
            if not ev:
                continue
            events.append(ev)
            for h in HORIZONS:
                strategy_returns["main_top5"][h].append(ev[f"return_{h}d_pct"] / 100.0)
            yearly_main[ev["signal_date"][:4]].append(ev["return_10d_pct"] / 100.0)

    events.sort(key=lambda x: (x["signal_date"], x["strategy"], x["volume_rank"], x["ticker"]))
    if events:
        with (OUT_DIR / "signals.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(events[0].keys()))
            w.writeheader()
            w.writerows(events)

    summary = {
        "meta": {
            "data_start": dates[0] if dates else None,
            "data_end": dates[-1] if dates else None,
            "stock_count": len(all_bars),
            "top_n_daily_volume_change": TOP_N,
            "volume_change_definition": "today_volume / previous_trading_day_volume - 1",
            "signal_timing": "rank and features known at signal-day close; entry next trading day open",
            "min_price_yen": MIN_PRICE,
            "min_average_turnover20_yen": MIN_TURNOVER20,
            "main_rule": "daily volume-change top100; overheat<=30; 0<ret1<=5%; close_location>=0.55; confirmation>=18; choose up to 5 lowest-overheat candidates",
            "overheat_priority": "70% of signal score is inverse overheat risk; demand confirmation is secondary",
            "average_ranked_candidates_per_day": round(mean(day_counts) or 0.0, 2),
        },
        "strategies": {
            s: {str(h): summarize(vals[h]) for h in HORIZONS}
            for s, vals in sorted(strategy_returns.items())
        },
        "overheat_buckets_within_top100": {
            b: {str(h): summarize(vals[h]) for h in HORIZONS}
            for b, vals in sorted(bucket_returns.items())
        },
        "yearly_main_10d": {y: summarize(v) for y, v in sorted(yearly_main.items())},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
