from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import test_psychology_11_roundtrips as base
import test_psychology_11_simple_state as simple

BUCKETS = {
    "ANY": lambda p: p.notna(),
    "A1_LT_0.8": lambda p: p < 0.8,
    "A2_0.8_1.2": lambda p: (p >= 0.8) & (p < 1.2),
    "A3_1.2_2.0": lambda p: (p >= 1.2) & (p <= 2.0),
    "A4_GE_2.0": lambda p: p > 2.0,
}


def prepare(g0: pd.DataFrame, bucket: str) -> pd.DataFrame:
    g = base.enrich(g0)
    c = g.close
    trend = (
        (g.ma25.notna() & (c > g.ma25) & (g.ma25_slope5 > 0))
        | ((g.history_n < 60) & g.ma10.notna() & (c > g.ma10) & (g.ma10_slope5 > 0))
    )
    part = BUCKETS[bucket](g.participation)
    g["blue_simple"] = g.quality & g.liquid & part & g.sentiment.between(60, 80) & trend
    g["contra_simple"] = g.quality & g.liquid & part & (g.sentiment <= 20) & (c > c.shift(1)) & (g.day_pos >= 0.60)
    return g


def pct(x): return "-" if x is None else f"{100*x:.2f}%"
def num(x): return "-" if x is None else f"{x:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    a = ap.parse_args()
    raw, m = base.load_prices(a.data_dir)

    results: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        bucket_rows: list[dict[str, Any]] = []
        signal_days = 0
        for code, g0 in raw.items():
            g = prepare(g0, bucket)
            signal_days += int((g.blue_simple | g.contra_simple).sum())
            for tr in simple.simulate(g, "COMBINED"):
                tr["code"] = code
                tr["bucket"] = bucket
                bucket_rows.append(tr)
                all_rows.append(tr)
        t = pd.DataFrame(bucket_rows)
        s = base.summarize(t) if not t.empty else {"n": 0}
        s["signal_days"] = signal_days
        results[bucket] = s

    out = {
        "meta": {"data_start": m["dates"][0], "data_end": m["dates"][-1], "note": "Same direct psychology rule; only participation bucket changes. No optimization."},
        "results": results,
        "trades": all_rows,
    }
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# 11 stocks participation-bucket round-trip comparison", "",
        f"- Data: {m['dates'][0]} to {m['dates'][-1]}",
        "- Same direct psychology state rule; only participation bucket changes", "- No threshold optimization", "",
        "| Participation | Signal days | Trades | Mean net | Median | Win | PF | Compound* | Max DD* | Avg MFE | Avg MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"ANY":"Any", "A1_LT_0.8":"<0.8x", "A2_0.8_1.2":"0.8-1.2x", "A3_1.2_2.0":"1.2-2.0x", "A4_GE_2.0":">2.0x"}
    for bucket, s in results.items():
        lines.append(f"| {labels[bucket]} | {s.get('signal_days',0)} | {s.get('n',0)} | {pct(s.get('mean_net'))} | {pct(s.get('median_net'))} | {pct(s.get('win_rate'))} | {num(s.get('pf'))} | {pct(s.get('compound'))} | {pct(s.get('max_dd'))} | {pct(s.get('avg_mfe'))} | {pct(s.get('avg_mae'))} |")
    lines += ["", "* Compound/Max DD are sequential-trade statistics, not a capital-allocation portfolio simulation.", "", "This is an in-sample teaching-stock diagnostic. Do not select a bucket from these results without subsequent OOS validation."]
    a.output_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
