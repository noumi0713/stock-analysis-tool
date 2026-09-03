from __future__ import annotations

import csv
import json
from collections import defaultdict

import volume_top100_overheat_3y as base

COOLDOWN = 10
OUT = base.OUT_DIR / "final_summary.json"


def main():
    base.OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest, all_bars = base.load_bars()
    dates = manifest["dates"]
    daily = defaultdict(list)

    for ticker, bars in all_bars.items():
        if len(bars) < 45:
            continue
        for p in range(20, len(bars) - max(base.HORIZONS)):
            feat = base.overheat_features(bars, p)
            if feat is not None and feat["volume_change"] > 0:
                daily[feat["idx"]].append((ticker, p, feat, bars))

    definitions = {
        "oh20_confirm18": lambda f: f["overheat"] < 20 and 0 < f["ret1"] <= 0.05 and f["close_location"] >= 0.55 and f["confirm"] >= 18,
        "oh30_confirm18": lambda f: f["overheat"] <= 30 and 0 < f["ret1"] <= 0.05 and f["close_location"] >= 0.55 and f["confirm"] >= 18,
        "oh20_confirm12": lambda f: f["overheat"] < 20 and 0 < f["ret1"] <= 0.05 and f["close_location"] >= 0.55 and f["confirm"] >= 12,
    }

    returns = defaultdict(lambda: defaultdict(list))
    yearly = defaultdict(lambda: defaultdict(list))
    counts = defaultdict(int)
    last_signal_idx = {name: {} for name in definitions}
    rows = []

    for idx in sorted(daily):
        ranked = sorted(daily[idx], key=lambda x: x[2]["volume_change"], reverse=True)[:base.TOP_N]
        for name, rule in definitions.items():
            candidates = []
            for rank, (ticker, p, feat, bars) in enumerate(ranked, start=1):
                if not rule(feat):
                    continue
                last = last_signal_idx[name].get(ticker, -10_000)
                if idx - last < COOLDOWN:
                    continue
                candidates.append((feat["overheat"], -feat["confirm"], rank, ticker, p, feat, bars))
            candidates.sort()
            for _, _, rank, ticker, p, feat, bars in candidates[:base.MAX_MAIN_SIGNALS_PER_DAY]:
                ev = base.evaluate(ticker, bars, p, feat, dates, name, rank)
                if not ev:
                    continue
                rows.append(ev)
                counts[name] += 1
                last_signal_idx[name][ticker] = idx
                for h in base.HORIZONS:
                    returns[name][h].append(ev[f"return_{h}d_pct"] / 100.0)
                yearly[name][ev["signal_date"][:4]].append(ev["return_10d_pct"] / 100.0)

    if rows:
        rows.sort(key=lambda r: (r["signal_date"], r["strategy"], r["volume_rank"], r["ticker"]))
        with (base.OUT_DIR / "final_signals.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    result = {
        "meta": {
            "cooldown_trading_days_per_ticker": COOLDOWN,
            "max_signals_per_day": base.MAX_MAIN_SIGNALS_PER_DAY,
            "ranking": "daily volume-change top100; sort eligible candidates by lowest overheat, then confirmation, then volume rank",
            "recommended_logic_if_supported": "overheat<20 hard gate; confirmation secondary",
        },
        "strategies": {
            name: {str(h): base.summarize(returns[name][h]) for h in base.HORIZONS}
            for name in definitions
        },
        "yearly_10d": {
            name: {year: base.summarize(vals) for year, vals in sorted(yearly[name].items())}
            for name in definitions
        },
        "signal_counts": dict(counts),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
