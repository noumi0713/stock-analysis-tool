from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dashboard-data" / "technical-backtest-3y"
THEME_FILE = ROOT / "dashboard-data" / "theme_members.csv"
OUT_DIR = ROOT / "research" / "results" / "theme_weekly_rotation_3y"
MIN_VALID_MEMBERS = 3
TOP_N = 15


def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def mean(values):
    return statistics.fmean(values) if values else None


def median(values):
    return statistics.median(values) if values else None


def pct(v):
    return None if v is None else round(v * 100.0, 4)


def summary(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean_pct": None, "median_pct": None, "positive_rate_pct": None}
    return {
        "n": len(vals),
        "mean_pct": pct(mean(vals)),
        "median_pct": pct(median(vals)),
        "positive_rate_pct": round(sum(v > 0 for v in vals) / len(vals) * 100.0, 2),
    }


def load_memberships():
    ticker_themes = defaultdict(list)
    theme_members = defaultdict(set)
    with THEME_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            ticker = str(row.get("yahoo_ticker") or "").strip()
            theme = str(row.get("theme_name") or "").strip()
            if not ticker or not theme:
                continue
            if theme not in ticker_themes[ticker]:
                ticker_themes[ticker].append(theme)
            theme_members[theme].add(ticker)
    return dict(ticker_themes), {k: set(v) for k, v in theme_members.items()}


def load_stock_weeks():
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    dates = [date.fromisoformat(x) for x in manifest["dates"]]
    stock_weeks = defaultdict(lambda: defaultdict(lambda: {"last_date": None, "last_close": None, "volume": 0.0}))
    for shard in manifest["shards"]:
        payload = json.loads((DATA_DIR / shard["path"]).read_text(encoding="utf-8"))
        for ticker, bars in payload["bars"].items():
            for bar in bars:
                d = dates[int(bar[0])]
                w = week_start(d)
                rec = stock_weeks[ticker][w]
                rec["volume"] += float(bar[5])
                if rec["last_date"] is None or d > rec["last_date"]:
                    rec["last_date"] = d
                    rec["last_close"] = float(bar[4])
    return manifest, stock_weeks


def build_theme_weeks(ticker_themes, theme_members, stock_weeks):
    theme_volume = defaultdict(float)
    theme_returns = defaultdict(list)
    theme_breadth = defaultdict(list)

    for ticker, weeks in stock_weeks.items():
        themes = ticker_themes.get(ticker)
        if not themes:
            continue
        divisor = float(len(themes))
        ordered = sorted(weeks.items())
        prev_w = None
        prev_close = None
        for w, rec in ordered:
            for theme in themes:
                theme_volume[(theme, w)] += rec["volume"] / divisor
            if prev_w is not None and (w - prev_w).days == 7 and prev_close and rec["last_close"]:
                r = rec["last_close"] / prev_close - 1.0
                for theme in themes:
                    theme_returns[(theme, w)].append(r)
                    theme_breadth[(theme, w)].append(1.0 if r > 0 else 0.0)
            prev_w = w
            prev_close = rec["last_close"]

    rows = []
    all_weeks = sorted({w for _, w in set(theme_volume) | set(theme_returns)})
    themes = sorted(theme_members)
    by_key = {}
    for theme in themes:
        for w in all_weeks:
            rets = theme_returns.get((theme, w), [])
            vol = theme_volume.get((theme, w), 0.0)
            if len(rets) < MIN_VALID_MEMBERS or vol <= 0:
                continue
            row = {
                "theme": theme,
                "week_start": w,
                "week_end": w + timedelta(days=4),
                "member_count": len(theme_members[theme]),
                "valid_return_count": len(rets),
                "weekly_volume": vol,
                "weekly_return_median": median(rets),
                "weekly_return_mean": mean(rets),
                "breadth_positive": mean(theme_breadth[(theme, w)]),
            }
            by_key[(theme, w)] = row
            rows.append(row)

    for row in rows:
        theme = row["theme"]
        w = row["week_start"]
        prev = by_key.get((theme, w - timedelta(days=7)))
        nxt = by_key.get((theme, w + timedelta(days=7)))
        row["volume_change"] = (
            row["weekly_volume"] / prev["weekly_volume"] - 1.0
            if prev and prev["weekly_volume"] > 0 else None
        )
        row["next_week_return"] = nxt["weekly_return_median"] if nxt else None
    return rows, by_key


def rank_weekly(rows):
    by_week = defaultdict(list)
    for row in rows:
        by_week[row["week_start"]].append(row)

    ranking_rows = []
    bucket_observations = defaultdict(list)
    yearly = defaultdict(lambda: defaultdict(list))
    spread_by_week = defaultdict(list)
    combo_observations = defaultdict(list)

    for w, items in sorted(by_week.items()):
        vol_items = [r for r in items if r["volume_change"] is not None and r["next_week_return"] is not None]
        ret_items = [r for r in items if r["weekly_return_median"] is not None and r["next_week_return"] is not None]
        if len(vol_items) < TOP_N * 2 or len(ret_items) < TOP_N * 2:
            continue

        vol_sorted = sorted(vol_items, key=lambda r: (r["volume_change"], r["theme"]), reverse=True)
        ret_sorted = sorted(ret_items, key=lambda r: (r["weekly_return_median"], r["theme"]), reverse=True)
        vol_top, vol_bottom = vol_sorted[:TOP_N], vol_sorted[-TOP_N:]
        ret_top, ret_bottom = ret_sorted[:TOP_N], ret_sorted[-TOP_N:]

        groups = {
            "volume_top15": vol_top,
            "volume_bottom15": vol_bottom,
            "return_top15": ret_top,
            "return_bottom15": ret_bottom,
        }
        for label, group in groups.items():
            ordered = group if "top" in label else list(reversed(group))
            for rank, r in enumerate(ordered, 1):
                ranking_rows.append({
                    "week_start": r["week_start"].isoformat(),
                    "week_end": r["week_end"].isoformat(),
                    "ranking": label,
                    "rank": rank,
                    "theme": r["theme"],
                    "volume_change_pct": pct(r["volume_change"]),
                    "weekly_return_median_pct": pct(r["weekly_return_median"]),
                    "weekly_return_mean_pct": pct(r["weekly_return_mean"]),
                    "breadth_positive_pct": pct(r["breadth_positive"]),
                    "weekly_volume": round(r["weekly_volume"], 2),
                    "member_count": r["member_count"],
                    "valid_return_count": r["valid_return_count"],
                    "next_week_return_pct": pct(r["next_week_return"]),
                })
                bucket_observations[label].append(r["next_week_return"])
                yearly[str(w.year)][label].append(r["next_week_return"])

        spread_by_week["volume_top_minus_bottom"].append(
            mean([r["next_week_return"] for r in vol_top]) - mean([r["next_week_return"] for r in vol_bottom])
        )
        spread_by_week["return_top_minus_bottom"].append(
            mean([r["next_week_return"] for r in ret_top]) - mean([r["next_week_return"] for r in ret_bottom])
        )

        sets = {
            "volume_top15": {r["theme"]: r for r in vol_top},
            "volume_bottom15": {r["theme"]: r for r in vol_bottom},
            "return_top15": {r["theme"]: r for r in ret_top},
            "return_bottom15": {r["theme"]: r for r in ret_bottom},
        }
        combos = {
            "volume_top15_and_return_top15": set(sets["volume_top15"]) & set(sets["return_top15"]),
            "volume_top15_and_return_bottom15": set(sets["volume_top15"]) & set(sets["return_bottom15"]),
            "volume_bottom15_and_return_top15": set(sets["volume_bottom15"]) & set(sets["return_top15"]),
            "volume_bottom15_and_return_bottom15": set(sets["volume_bottom15"]) & set(sets["return_bottom15"]),
        }
        item_map = {r["theme"]: r for r in items}
        for label, names in combos.items():
            for name in names:
                combo_observations[label].append(item_map[name]["next_week_return"])

    return ranking_rows, bucket_observations, yearly, spread_by_week, combo_observations


def position_buckets(ranking_rows):
    result = defaultdict(list)
    for row in ranking_rows:
        if not row["ranking"].endswith("top15"):
            continue
        rank = int(row["rank"])
        bucket = "1-5" if rank <= 5 else "6-10" if rank <= 10 else "11-15"
        result[f'{row["ranking"]}_rank_{bucket}'].append(row["next_week_return_pct"] / 100.0)
    return {k: summary(v) for k, v in sorted(result.items())}


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ticker_themes, theme_members = load_memberships()
    manifest, stock_weeks = load_stock_weeks()
    rows, _ = build_theme_weeks(ticker_themes, theme_members, stock_weeks)
    ranking_rows, buckets, yearly, spreads, combos = rank_weekly(rows)

    baseline = summary([r["next_week_return"] for r in rows if r["next_week_return"] is not None])
    summary_json = {
        "meta": {
            "data_start": manifest["meta"]["startDate"],
            "data_end": manifest["meta"]["endDate"],
            "stock_count": manifest["meta"]["stockCount"],
            "theme_count": len(theme_members),
            "theme_membership_rows": sum(len(v) for v in theme_members.values()),
            "theme_membership_mode": "current_snapshot_applied_historically",
            "min_valid_members": MIN_VALID_MEMBERS,
            "top_n": TOP_N,
            "volume_definition": "sum of split-adjusted weekly share volume, divided by each stock's number of theme memberships; ranked by week-over-week percentage change",
            "return_definition": "median constituent close-to-close weekly return; constituent return is not divided by theme-membership count",
            "signal_timing": "ranking known after current week close; evaluated on next week's theme median return",
        },
        "baseline_next_week": baseline,
        "buckets": {k: summary(v) for k, v in sorted(buckets.items())},
        "weekly_spreads": {k: summary(v) for k, v in sorted(spreads.items())},
        "combinations": {k: summary(v) for k, v in sorted(combos.items())},
        "rank_position_buckets": position_buckets(ranking_rows),
        "yearly": {
            year: {label: summary(vals) for label, vals in sorted(groups.items())}
            for year, groups in sorted(yearly.items())
        },
        "ranking_week_count": len({r["week_start"] for r in ranking_rows}),
        "ranking_row_count": len(ranking_rows),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(OUT_DIR / "weekly_rankings.csv", ranking_rows)

    compact_rows = []
    for label, vals in sorted(buckets.items()):
        s = summary(vals)
        compact_rows.append({"group": label, **s})
    for label, vals in sorted(combos.items()):
        s = summary(vals)
        compact_rows.append({"group": label, **s})
    for label, vals in sorted(spreads.items()):
        s = summary(vals)
        compact_rows.append({"group": label, **s})
    write_csv(OUT_DIR / "performance_summary.csv", compact_rows)

    print(json.dumps(summary_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
