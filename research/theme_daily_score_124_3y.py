from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MIN_VALID_MEMBERS = 5
TOP_GROUPS = (5, 10)
HORIZONS = (5, 10, 20)
WEIGHTS = {
    "price_strength": 0.25,
    "turnover_inflow": 0.25,
    "breadth": 0.20,
    "relative_strength": 0.15,
    "persistence": 0.15,
}


def load_memberships(path: Path):
    theme_members: dict[str, set[str]] = defaultdict(set)
    theme_clusters: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            ticker = str(row.get("yahoo_ticker") or "").strip()
            theme = str(row.get("theme_name") or "").strip()
            cluster = str(row.get("cluster") or "").strip()
            if ticker and theme:
                theme_members[theme].add(ticker)
                theme_clusters[theme] = cluster
    if len(theme_members) != 124:
        raise ValueError(f"Expected 124 themes, got {len(theme_members)}")
    return {k: set(v) for k, v in theme_members.items()}, theme_clusters


def load_market(data_dir: Path, wanted: set[str]):
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    dates = pd.DatetimeIndex(pd.to_datetime(manifest["dates"]))
    tickers = sorted(wanted)
    tcol = {t: j for j, t in enumerate(tickers)}
    close = np.full((len(dates), len(tickers)), np.nan, dtype=np.float64)
    volume = np.full_like(close, np.nan)
    for shard in manifest["shards"]:
        payload = json.loads((data_dir / shard["path"]).read_text(encoding="utf-8"))
        for ticker, bars in payload["bars"].items():
            j = tcol.get(ticker)
            if j is None or not bars:
                continue
            a = np.asarray(bars, dtype=float)
            idx = a[:, 0].astype(int)
            close[idx, j] = a[:, 4]
            volume[idx, j] = a[:, 5]
    return manifest, dates, tickers, close, volume


def pct_change(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    prev = a[:-k]
    cur = a[k:]
    ok = np.isfinite(prev) & np.isfinite(cur) & (prev > 0)
    block = np.full_like(cur, np.nan)
    block[ok] = cur[ok] / prev[ok] - 1.0
    out[k:] = block
    return out


def forward_return(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan)
    cur = a[:-k]
    fut = a[k:]
    ok = np.isfinite(cur) & np.isfinite(fut) & (cur > 0)
    block = np.full_like(cur, np.nan)
    block[ok] = fut[ok] / cur[ok] - 1.0
    out[:-k] = block
    return out


def theme_median(matrix: np.ndarray, member_idx: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    d = matrix.shape[0]
    t = len(member_idx)
    med = np.full((d, t), np.nan, dtype=np.float64)
    cnt = np.zeros((d, t), dtype=np.int32)
    for j, idx in enumerate(member_idx):
        if len(idx) == 0:
            continue
        x = matrix[:, idx]
        cnt[:, j] = np.isfinite(x).sum(axis=1)
        with np.errstate(all="ignore"):
            med[:, j] = np.nanmedian(x, axis=1)
        med[cnt[:, j] == 0, j] = np.nan
    return med, cnt


def theme_fraction(mask_value: np.ndarray, valid: np.ndarray, member_idx: list[np.ndarray]) -> np.ndarray:
    d = mask_value.shape[0]
    out = np.full((d, len(member_idx)), np.nan, dtype=np.float64)
    for j, idx in enumerate(member_idx):
        if len(idx) == 0:
            continue
        v = valid[:, idx]
        n = v.sum(axis=1)
        num = (mask_value[:, idx] & v).sum(axis=1)
        ok = n > 0
        out[ok, j] = num[ok] / n[ok]
    return out


def xsec_percentile(raw: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    x = np.where(eligible, raw, np.nan)
    return pd.DataFrame(x).rank(axis=1, method="average", pct=True).to_numpy() * 100.0


def stat_summary(values: np.ndarray, baseline_mean: float | None = None) -> dict:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"n": 0, "mean_pct": None, "median_pct": None, "positive_rate_pct": None,
                "plus3_rate_pct": None, "plus5_rate_pct": None, "minus3_rate_pct": None,
                "minus5_rate_pct": None, "edge_vs_baseline_mean_pct": None}
    mean_v = float(vals.mean())
    return {
        "n": int(len(vals)),
        "mean_pct": round(mean_v * 100, 4),
        "median_pct": round(float(np.median(vals)) * 100, 4),
        "positive_rate_pct": round(float((vals > 0).mean()) * 100, 2),
        "plus3_rate_pct": round(float((vals >= 0.03).mean()) * 100, 2),
        "plus5_rate_pct": round(float((vals >= 0.05).mean()) * 100, 2),
        "minus3_rate_pct": round(float((vals <= -0.03).mean()) * 100, 2),
        "minus5_rate_pct": round(float((vals <= -0.05).mean()) * 100, 2),
        "edge_vs_baseline_mean_pct": None if baseline_mean is None else round((mean_v - baseline_mean) * 100, 4),
    }


def fmt(v, digits=2):
    return "-" if v is None else f"{v:.{digits}f}"


def run(data_dir: Path, memberships: Path, output_json: Path, output_md: Path, output_csv: Path):
    theme_members, theme_clusters = load_memberships(memberships)
    themes = sorted(theme_members)
    wanted = set().union(*theme_members.values())
    manifest, dates, tickers, close, volume = load_market(data_dir, wanted)
    tcol = {t: j for j, t in enumerate(tickers)}
    member_idx = [np.array([tcol[x] for x in sorted(theme_members[theme]) if x in tcol], dtype=int) for theme in themes]

    ret1 = pct_change(close, 1)
    ret5 = pct_change(close, 5)
    ret10 = pct_change(close, 10)
    ma25 = pd.DataFrame(close).rolling(25, min_periods=20).mean().to_numpy()

    th_ret1, cnt1 = theme_median(ret1, member_idx)
    th_ret5, cnt5 = theme_median(ret5, member_idx)
    th_ret10, cnt10 = theme_median(ret10, member_idx)

    # Price strength: recent momentum, robust to one-stock spikes via constituent medians.
    price_raw = 0.60 * th_ret5 + 0.40 * th_ret10

    # Turnover inflow: current aggregate yen turnover versus its own prior-20-day median.
    # A stock belonging to multiple themes is divided by its number of memberships before aggregation.
    membership_count = np.zeros(len(tickers), dtype=float)
    M = np.zeros((len(tickers), len(themes)), dtype=float)
    for j, idx in enumerate(member_idx):
        M[idx, j] = 1.0
        membership_count[idx] += 1.0
    membership_count[membership_count <= 0] = 1.0
    turnover = close * volume
    adjusted_turnover = np.nan_to_num(turnover / membership_count[None, :], nan=0.0, posinf=0.0, neginf=0.0)
    th_turnover = adjusted_turnover @ M
    prior20_turn_med = pd.DataFrame(th_turnover).shift(1).rolling(20, min_periods=10).median().to_numpy()
    turnover_raw = np.where(prior20_turn_med > 0, th_turnover / prior20_turn_med - 1.0, np.nan)

    # Breadth: half 5-day advancers, half members above their 25-day average.
    pos5 = theme_fraction(ret5 > 0, np.isfinite(ret5), member_idx)
    above25 = theme_fraction(close > ma25, np.isfinite(close) & np.isfinite(ma25), member_idx)
    breadth_raw = 0.50 * pos5 + 0.50 * above25

    # Relative strength against the cross-sectional median stock return.
    with np.errstate(all="ignore"):
        market_ret5 = np.nanmedian(ret5, axis=1)
    relative_raw = th_ret5 - market_ret5[:, None]

    # Persistence: fraction of positive theme median daily returns over the last 5 sessions.
    positive_day = np.where(np.isfinite(th_ret1), (th_ret1 > 0).astype(float), np.nan)
    persistence_raw = pd.DataFrame(positive_day).rolling(5, min_periods=3).mean().to_numpy()

    eligible = (
        (cnt5 >= MIN_VALID_MEMBERS) & (cnt10 >= MIN_VALID_MEMBERS)
        & np.isfinite(price_raw) & np.isfinite(turnover_raw)
        & np.isfinite(breadth_raw) & np.isfinite(relative_raw) & np.isfinite(persistence_raw)
    )

    factors = {
        "price_strength": price_raw,
        "turnover_inflow": turnover_raw,
        "breadth": breadth_raw,
        "relative_strength": relative_raw,
        "persistence": persistence_raw,
    }
    factor_pct = {name: xsec_percentile(raw, eligible) for name, raw in factors.items()}
    score = np.zeros_like(price_raw)
    score[:] = np.nan
    combined = sum(WEIGHTS[name] * factor_pct[name] for name in WEIGHTS)
    score[eligible] = combined[eligible]
    rank = pd.DataFrame(score).rank(axis=1, method="first", ascending=False).to_numpy()

    fwd = {}
    fwd_counts = {}
    for h in HORIZONS:
        stock_fwd = forward_return(close, h)
        fwd[h], fwd_counts[h] = theme_median(stock_fwd, member_idx)
        fwd[h][fwd_counts[h] < MIN_VALID_MEMBERS] = np.nan

    baseline = {}
    baseline_means = {}
    for h in HORIZONS:
        vals = fwd[h][eligible & np.isfinite(fwd[h])]
        s = stat_summary(vals)
        baseline[h] = s
        baseline_means[h] = None if s["mean_pct"] is None else s["mean_pct"] / 100.0

    results = {"daily_top": {}, "entry_events": {}, "yearly_entry_events": {}}
    entry_rows: list[dict] = []

    for topn in TOP_GROUPS:
        label = f"top{topn}"
        in_top = eligible & (rank <= topn)
        prev = np.vstack([np.zeros((1, in_top.shape[1]), dtype=bool), in_top[:-1]])
        entry = in_top & ~prev
        results["daily_top"][label] = {}
        results["entry_events"][label] = {}
        results["yearly_entry_events"][label] = {}

        for h in HORIZONS:
            results["daily_top"][label][str(h)] = stat_summary(fwd[h][in_top & np.isfinite(fwd[h])], baseline_means[h])
            results["entry_events"][label][str(h)] = stat_summary(fwd[h][entry & np.isfinite(fwd[h])], baseline_means[h])

        years = sorted(set(dates.year))
        for year in years:
            dmask = np.asarray(dates.year == year)[:, None]
            results["yearly_entry_events"][label][str(year)] = {
                str(h): stat_summary(fwd[h][entry & dmask & np.isfinite(fwd[h])], baseline_means[h])
                for h in HORIZONS
            }

        ii, jj = np.where(entry)
        for i, j in zip(ii, jj):
            row = {
                "date": dates[i].strftime("%Y-%m-%d"),
                "group": label,
                "rank": int(rank[i, j]),
                "theme": themes[j],
                "cluster": theme_clusters.get(themes[j], ""),
                "score": round(float(score[i, j]), 4),
                "price_strength_pctile": round(float(factor_pct["price_strength"][i, j]), 4),
                "turnover_inflow_pctile": round(float(factor_pct["turnover_inflow"][i, j]), 4),
                "breadth_pctile": round(float(factor_pct["breadth"][i, j]), 4),
                "relative_strength_pctile": round(float(factor_pct["relative_strength"][i, j]), 4),
                "persistence_pctile": round(float(factor_pct["persistence"][i, j]), 4),
            }
            for h in HORIZONS:
                row[f"fwd{h}_pct"] = None if not np.isfinite(fwd[h][i, j]) else round(float(fwd[h][i, j]) * 100, 4)
            entry_rows.append(row)

    # Theme-level diagnostics based on top10 entry events; descriptive only, not used to set rules.
    theme_diag = []
    for theme in themes:
        vals = [r for r in entry_rows if r["group"] == "top10" and r["theme"] == theme and r["fwd10_pct"] is not None]
        if not vals:
            continue
        arr = np.array([r["fwd10_pct"] / 100.0 for r in vals], dtype=float)
        s = stat_summary(arr, baseline_means[10])
        theme_diag.append({"theme": theme, "cluster": theme_clusters.get(theme, ""), **s})
    theme_diag.sort(key=lambda x: (-999 if x["mean_pct"] is None else x["mean_pct"]), reverse=True)

    meta = {
        "data_start": manifest["meta"]["startDate"],
        "data_end": manifest["meta"]["endDate"],
        "trading_days": int(len(dates)),
        "stock_count_source": manifest["meta"]["stockCount"],
        "membership_tickers": int(len(tickers)),
        "theme_count": int(len(themes)),
        "theme_membership_rows_unique": int(sum(len(v) for v in theme_members.values())),
        "theme_membership_mode": "current_snapshot_applied_historically",
        "min_valid_members": MIN_VALID_MEMBERS,
        "weights": WEIGHTS,
        "score_method": "daily cross-sectional percentile per factor, weighted to 0-100; all inputs point-in-time",
        "price_strength": "60% median constituent 5d return + 40% median constituent 10d return",
        "turnover_inflow": "aggregate close*volume / prior-20d median; stock turnover divided by number of theme memberships",
        "breadth": "50% fraction positive 5d + 50% fraction above 25d moving average",
        "relative_strength": "theme median 5d return minus cross-sectional median stock 5d return",
        "persistence": "fraction positive theme-median daily returns in trailing 5 sessions",
        "forward_return": "median constituent close-to-close forward return at 5/10/20 trading days",
        "entry_event": "first day theme is in top-N after not being in same top-N on prior trading day",
        "optimization": "none; factor weights fixed before outcome calculation",
    }
    out = {
        "meta": meta,
        "baseline_all_eligible_theme_days": {str(k): v for k, v in baseline.items()},
        **results,
        "top10_theme_diagnostics_10d": theme_diag,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if entry_rows:
        pd.DataFrame(entry_rows).sort_values(["date", "group", "rank", "theme"]).to_csv(output_csv, index=False, encoding="utf-8-sig")

    lines = [
        "# 124-theme daily score 3-year backtest", "",
        f"- Data: {meta['data_start']} to {meta['data_end']} ({meta['trading_days']} trading days)",
        f"- Themes: {meta['theme_count']}; unique membership rows: {meta['theme_membership_rows_unique']}",
        "- Membership: current 124-theme snapshot applied historically (static-membership diagnostic)",
        "- Score weights fixed before outcomes: price 25 / turnover inflow 25 / breadth 20 / relative strength 15 / persistence 15", "",
        "## Entry-event results (primary)", "",
        "| Group | Horizon | N | Mean | Median | Positive | +5% | -5% | Edge vs baseline mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for topn in TOP_GROUPS:
        label = f"top{topn}"
        for h in HORIZONS:
            s = results["entry_events"][label][str(h)]
            lines.append(f"| {label} | {h}d | {s['n']} | {fmt(s['mean_pct'])}% | {fmt(s['median_pct'])}% | {fmt(s['positive_rate_pct'])}% | {fmt(s['plus5_rate_pct'])}% | {fmt(s['minus5_rate_pct'])}% | {fmt(s['edge_vs_baseline_mean_pct'])}pt |")

    lines += ["", "## Baseline: all eligible theme-days", "", "| Horizon | N | Mean | Median | Positive |", "|---|---:|---:|---:|---:|"]
    for h in HORIZONS:
        s = baseline[h]
        lines.append(f"| {h}d | {s['n']} | {fmt(s['mean_pct'])}% | {fmt(s['median_pct'])}% | {fmt(s['positive_rate_pct'])}% |")

    lines += ["", "## Top10 entry events by signal year", "", "| Year | Horizon | N | Mean | Median | Positive | Edge |", "|---|---:|---:|---:|---:|---:|---:|"]
    for year, yr in results["yearly_entry_events"]["top10"].items():
        for h in HORIZONS:
            s = yr[str(h)]
            lines.append(f"| {year} | {h}d | {s['n']} | {fmt(s['mean_pct'])}% | {fmt(s['median_pct'])}% | {fmt(s['positive_rate_pct'])}% | {fmt(s['edge_vs_baseline_mean_pct'])}pt |")

    eligible_diag = [x for x in theme_diag if x["n"] >= 5]
    lines += ["", "## Theme diagnostics: top10 entry events, 10d (n>=5; descriptive)", "", "| Theme | N | Mean | Median | Positive | Edge |", "|---|---:|---:|---:|---:|---:|"]
    for s in eligible_diag[:12]:
        lines.append(f"| {s['theme']} | {s['n']} | {fmt(s['mean_pct'])}% | {fmt(s['median_pct'])}% | {fmt(s['positive_rate_pct'])}% | {fmt(s['edge_vs_baseline_mean_pct'])}pt |")

    lines += ["", "## Guardrails", "",
              "- Score uses only information available on or before each ranking date; forward returns are used only for evaluation.",
              "- The 124-theme membership file is a current snapshot applied to the full 3 years, so this is not a point-in-time constituent backtest.",
              "- Repeated DAILY_TOP observations overlap; ENTRY_EVENT is the primary 'entered top themes' diagnostic.",
              "- Theme forward return is the median member return, not a tradable portfolio return; transaction costs and capital allocation are not modeled."]
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("dashboard-data/technical-backtest-3y"))
    p.add_argument("--memberships", type=Path, default=Path("research/data/theme_members_124.csv"))
    p.add_argument("--output-json", type=Path, default=Path("research-output/theme-daily-score-124-3y.json"))
    p.add_argument("--output-md", type=Path, default=Path("research-output/theme-daily-score-124-3y.md"))
    p.add_argument("--output-csv", type=Path, default=Path("research-output/theme-daily-score-124-3y-entry-events.csv"))
    a = p.parse_args()
    run(a.data_dir, a.memberships, a.output_json, a.output_md, a.output_csv)


if __name__ == "__main__":
    main()
