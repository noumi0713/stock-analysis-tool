from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dashboard-data" / "technical-backtest-3y"
THEME_FILE = ROOT / "dashboard-data" / "theme_members.csv"
OUT_DIR = ROOT / "research" / "results" / "demand_acceleration_3y"
HORIZONS = (5, 10, 20, 40)
COOLDOWN = 20
MIN_PRICE = 100.0
MIN_TURNOVER20 = 120_000_000.0


def mean(xs):
    return statistics.fmean(xs) if xs else None


def median(xs):
    return statistics.median(xs) if xs else None


def pct(x):
    return None if x is None else round(x * 100.0, 4)


def safe_div(a, b, default=0.0):
    return a / b if b and math.isfinite(b) else default


def summarize(xs):
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not vals:
        return {"n": 0, "mean_pct": None, "median_pct": None, "win_rate_pct": None}
    return {
        "n": len(vals),
        "mean_pct": pct(mean(vals)),
        "median_pct": pct(median(vals)),
        "win_rate_pct": round(sum(x > 0 for x in vals) / len(vals) * 100.0, 2),
    }


def load_memberships():
    ticker_themes = defaultdict(list)
    with THEME_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            ticker = str(row.get("yahoo_ticker") or "").strip()
            theme = str(row.get("theme_name") or "").strip()
            if ticker and theme and theme not in ticker_themes[ticker]:
                ticker_themes[ticker].append(theme)
    return dict(ticker_themes)


def load_bars():
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    all_bars = {}
    for shard in manifest["shards"]:
        payload = json.loads((DATA_DIR / shard["path"]).read_text(encoding="utf-8"))
        for ticker, bars in payload["bars"].items():
            all_bars[ticker] = bars
    return manifest, all_bars


def precompute_context(all_bars, ticker_themes):
    market_sum = defaultdict(float)
    market_count = defaultdict(int)
    theme_sum = defaultdict(lambda: defaultdict(float))
    theme_count = defaultdict(lambda: defaultdict(int))

    for ticker, bars in all_bars.items():
        themes = ticker_themes.get(ticker)
        if not themes:
            continue
        for p in range(5, len(bars)):
            c0 = float(bars[p - 5][4])
            c1 = float(bars[p][4])
            if c0 <= 0 or c1 <= 0:
                continue
            r5 = c1 / c0 - 1.0
            idx = int(bars[p][0])
            market_sum[idx] += r5
            market_count[idx] += 1
            for theme in themes:
                theme_sum[idx][theme] += r5
                theme_count[idx][theme] += 1

    market_mean = {idx: market_sum[idx] / market_count[idx] for idx in market_sum if market_count[idx]}
    theme_pct = {}
    for idx, sums in theme_sum.items():
        vals = []
        for theme, total in sums.items():
            n = theme_count[idx][theme]
            if n >= 3:
                vals.append((total / n, theme))
        vals.sort()
        n = len(vals)
        ranks = {}
        for rank, (_, theme) in enumerate(vals):
            ranks[theme] = rank / (n - 1) if n > 1 else 0.5
        theme_pct[idx] = ranks
    return market_mean, theme_pct


def avg_bar_value(bars, start, end):
    vals = [float(b[4]) * float(b[5]) for b in bars[start:end] if float(b[4]) > 0 and float(b[5]) >= 0]
    return mean(vals) or 0.0


def avg_volume(bars, start, end):
    vals = [float(b[5]) for b in bars[start:end] if float(b[5]) >= 0]
    return mean(vals) or 0.0


def feature_row(ticker, bars, p, themes, market_mean, theme_pct):
    if p < 24:
        return None
    b = bars[p]
    idx = int(b[0])
    close = float(b[4])
    if close < MIN_PRICE:
        return None

    turnover20 = avg_bar_value(bars, p - 19, p + 1)
    if turnover20 < MIN_TURNOVER20:
        return None

    recent5_value = avg_bar_value(bars, p - 4, p + 1)
    prior15_value = avg_bar_value(bars, p - 19, p - 4)
    turnover_ratio = safe_div(recent5_value, prior15_value)

    vol3 = avg_volume(bars, p - 2, p + 1)
    vol20 = avg_volume(bars, p - 19, p + 1)
    volume_ratio = safe_div(vol3, vol20)

    up_vols, down_vols = [], []
    for j in range(p - 9, p + 1):
        if j <= 0:
            continue
        v = float(bars[j][5])
        if float(bars[j][4]) >= float(bars[j - 1][4]):
            up_vols.append(v)
        else:
            down_vols.append(v)
    uv_ratio = safe_div(mean(up_vols) or 0.0, mean(down_vols) or 0.0, default=2.0 if up_vols and not down_vols else 0.0)

    cls = []
    for x in bars[p - 4 : p + 1]:
        hi, lo, c = float(x[2]), float(x[3]), float(x[4])
        cls.append((c - lo) / (hi - lo) if hi > lo else 0.5)
    close_location5 = mean(cls) or 0.5

    ret1 = close / float(bars[p - 1][4]) - 1.0
    ret5 = close / float(bars[p - 5][4]) - 1.0
    ret20 = close / float(bars[p - 20][4]) - 1.0
    rs5 = ret5 - market_mean.get(idx, 0.0)

    ranks = theme_pct.get(idx, {})
    best_theme_pct = max((ranks.get(t, 0.0) for t in themes), default=0.0)

    high20 = max(float(x[2]) for x in bars[p - 19 : p + 1])
    high_distance = high20 / close - 1.0 if close > 0 else 9.0
    ma20 = mean([float(x[4]) for x in bars[p - 19 : p + 1]]) or close
    ma20_prev = mean([float(x[4]) for x in bars[p - 24 : p - 4]]) or ma20

    score = 0
    if turnover_ratio >= 1.10:
        score += 7
    if turnover_ratio >= 1.30:
        score += 8
    if volume_ratio >= 1.15:
        score += 10
    if uv_ratio >= 1.10:
        score += 10
    if close_location5 >= 0.60:
        score += 10
    if 0.005 <= ret5 <= 0.08:
        score += 10
    if rs5 > 0:
        score += 7
    if rs5 > 0.02:
        score += 8
    if best_theme_pct >= 0.70:
        score += 5
    if best_theme_pct >= 0.85:
        score += 5
    if high_distance <= 0.05:
        score += 10
    if close > ma20 and ma20 > ma20_prev:
        score += 10

    penalty = 0
    if ret5 > 0.12:
        penalty += 20
    if ret20 > 0.25:
        penalty += 20
    if volume_ratio > 2.50:
        penalty += 15
    if turnover_ratio > 3.00:
        penalty += 15
    if ret1 > 0.07:
        penalty += 15

    return {
        "idx": idx,
        "score": score,
        "penalty": penalty,
        "turnover20": turnover20,
        "turnover_ratio": turnover_ratio,
        "volume_ratio": volume_ratio,
        "up_down_volume_ratio": uv_ratio,
        "close_location5": close_location5,
        "ret5": ret5,
        "ret20": ret20,
        "rs5": rs5,
        "theme_pct": best_theme_pct,
        "high_distance": high_distance,
        "close": close,
    }


def evaluate_event(ticker, bars, p, feat, prev_score, dates, strategy):
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
        "score": feat["score"],
        "score_5d_ago": prev_score,
        "score_delta_5d": feat["score"] - prev_score,
        "overheat_penalty": feat["penalty"],
        "turnover20_yen": round(feat["turnover20"], 2),
        "turnover_ratio_5v15": round(feat["turnover_ratio"], 4),
        "volume_ratio_3v20": round(feat["volume_ratio"], 4),
        "up_down_volume_ratio_10d": round(feat["up_down_volume_ratio"], 4),
        "close_location_5d": round(feat["close_location5"], 4),
        "return_5d_pct": pct(feat["ret5"]),
        "rs_5d_pct": pct(feat["rs5"]),
        "best_theme_percentile": round(feat["theme_pct"], 4),
        "distance_from_20d_high_pct": pct(feat["high_distance"]),
    }
    for h in HORIZONS:
        exit_close = float(bars[p + h][4])
        window = bars[p + 1 : p + h + 1]
        highs = [float(x[2]) for x in window]
        lows = [float(x[3]) for x in window]
        r = exit_close / entry_price - 1.0
        mfe = max(highs) / entry_price - 1.0
        mae = min(lows) / entry_price - 1.0
        row[f"return_{h}d_pct"] = pct(r)
        row[f"mfe_{h}d_pct"] = pct(mfe)
        row[f"mae_{h}d_pct"] = pct(mae)
        row[f"hit_plus5_{h}d"] = int(mfe >= 0.05)
        row[f"hit_plus10_{h}d"] = int(mfe >= 0.10)
        row[f"hit_minus5_{h}d"] = int(mae <= -0.05)
        row[f"hit_minus8_{h}d"] = int(mae <= -0.08)
    return row


def add_aggregate(agg, event):
    for h in HORIZONS:
        agg[(event["strategy"], h, "return")].append(event[f"return_{h}d_pct"] / 100.0)
        agg[(event["strategy"], h, "mfe")].append(event[f"mfe_{h}d_pct"] / 100.0)
        agg[(event["strategy"], h, "mae")].append(event[f"mae_{h}d_pct"] / 100.0)
        for tag in ("plus5", "plus10", "minus5", "minus8"):
            agg[(event["strategy"], h, tag)].append(event[f"hit_{tag}_{h}d"])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ticker_themes = load_memberships()
    manifest, all_bars = load_bars()
    dates = manifest["dates"]
    market_mean, theme_pct = precompute_context(all_bars, ticker_themes)

    events = []
    agg = defaultdict(list)
    yearly = defaultdict(lambda: defaultdict(list))
    baseline = defaultdict(list)
    eligible_tickers = 0

    for ticker, bars in all_bars.items():
        themes = ticker_themes.get(ticker)
        if not themes or len(bars) < 70:
            continue
        eligible_tickers += 1
        feats = [None] * len(bars)
        for p in range(24, len(bars)):
            feats[p] = feature_row(ticker, bars, p, themes, market_mean, theme_pct)

        # Broad liquid-universe baseline, sampled every 20 bars to reduce serial duplication.
        for p in range(24, len(bars) - max(HORIZONS), COOLDOWN):
            feat = feats[p]
            if not feat:
                continue
            entry_price = float(bars[p + 1][1])
            if entry_price <= 0:
                continue
            for h in HORIZONS:
                baseline[h].append(float(bars[p + h][4]) / entry_price - 1.0)

        last_signal = defaultdict(lambda: -10_000)
        for p in range(29, len(bars) - max(HORIZONS)):
            feat = feats[p]
            prev = feats[p - 5]
            if not feat or not prev:
                continue
            delta = feat["score"] - prev["score"]
            definitions = {
                "static_score60": feat["score"] >= 60 and feat["penalty"] <= 20,
                "static_score70": feat["score"] >= 70 and feat["penalty"] <= 20,
                "acceleration_raw": feat["score"] >= 60 and prev["score"] <= 50 and delta >= 15,
                "acceleration_main": feat["score"] >= 60 and prev["score"] <= 50 and delta >= 15 and feat["penalty"] <= 20,
            }
            for strategy, ok in definitions.items():
                if not ok or p - last_signal[strategy] < COOLDOWN:
                    continue
                event = evaluate_event(ticker, bars, p, feat, prev["score"], dates, strategy)
                if not event:
                    continue
                events.append(event)
                add_aggregate(agg, event)
                if strategy == "acceleration_main":
                    yearly[event["signal_date"][:4]]["return_20d"].append(event["return_20d_pct"] / 100.0)
                    score_bucket = "60-69" if feat["score"] < 70 else "70-79" if feat["score"] < 80 else "80+"
                    yearly["score_bucket"][score_bucket].append(event["return_20d_pct"] / 100.0)
                    delta_bucket = "15-24" if delta < 25 else "25+"
                    yearly["delta_bucket"][delta_bucket].append(event["return_20d_pct"] / 100.0)
                last_signal[strategy] = p

    events.sort(key=lambda x: (x["signal_date"], x["strategy"], -x["score"], x["ticker"]))
    if events:
        with (OUT_DIR / "signals.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(events[0].keys()))
            w.writeheader()
            w.writerows(events)

    performance_rows = []
    summary = {
        "meta": {
            "data_start": manifest["meta"]["startDate"],
            "data_end": manifest["meta"]["endDate"],
            "stock_count": manifest["meta"]["stockCount"],
            "eligible_theme_mapped_tickers": eligible_tickers,
            "theme_membership_mode": "current_124_theme_snapshot_applied_historically",
            "quality_fundamental_filter": "NOT_APPLIED: point-in-time historical fundamentals are not stored; applying current fundamentals would create look-ahead bias",
            "signal_timing": "features known at signal-day close; entry at next trading day's open",
            "cooldown_trading_days_per_ticker_per_strategy": COOLDOWN,
            "min_price_yen": MIN_PRICE,
            "min_average_turnover20_yen": MIN_TURNOVER20,
            "main_definition": "score>=60, score_5d_ago<=50, score_delta>=15, overheat_penalty<=20",
            "score_definition": "100-point fixed demand score using turnover acceleration, volume acceleration, up/down volume asymmetry, close location, moderate 5d return, cross-sectional relative strength, 124-theme strength percentile, 20d-high proximity, and rising 20d trend",
        },
        "baseline": {str(h): summarize(vals) for h, vals in baseline.items()},
        "strategies": {},
        "yearly_main_20d": {year: summarize(groups["return_20d"]) for year, groups in yearly.items() if year.isdigit()},
        "main_20d_score_buckets": {k: summarize(v) for k, v in yearly["score_bucket"].items()},
        "main_20d_delta_buckets": {k: summarize(v) for k, v in yearly["delta_bucket"].items()},
    }

    strategies = sorted({e["strategy"] for e in events})
    for strategy in strategies:
        summary["strategies"][strategy] = {}
        for h in HORIZONS:
            rets = agg[(strategy, h, "return")]
            mfes = agg[(strategy, h, "mfe")]
            maes = agg[(strategy, h, "mae")]
            rec = summarize(rets)
            rec.update({
                "mfe_mean_pct": pct(mean(mfes)) if mfes else None,
                "mfe_median_pct": pct(median(mfes)) if mfes else None,
                "mae_mean_pct": pct(mean(maes)) if maes else None,
                "mae_median_pct": pct(median(maes)) if maes else None,
                "plus5_hit_rate_pct": round(mean(agg[(strategy, h, "plus5")]) * 100.0, 2) if agg[(strategy, h, "plus5")] else None,
                "plus10_hit_rate_pct": round(mean(agg[(strategy, h, "plus10")]) * 100.0, 2) if agg[(strategy, h, "plus10")] else None,
                "minus5_hit_rate_pct": round(mean(agg[(strategy, h, "minus5")]) * 100.0, 2) if agg[(strategy, h, "minus5")] else None,
                "minus8_hit_rate_pct": round(mean(agg[(strategy, h, "minus8")]) * 100.0, 2) if agg[(strategy, h, "minus8")] else None,
            })
            summary["strategies"][strategy][str(h)] = rec
            performance_rows.append({"strategy": strategy, "horizon_days": h, **rec})

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if performance_rows:
        with (OUT_DIR / "performance_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(performance_rows[0].keys()))
            w.writeheader()
            w.writerows(performance_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
