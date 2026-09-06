from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import test_psychology_11_roundtrips as base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    a = ap.parse_args()

    raw, manifest = base.load_prices(a.data_dir)
    rows = []
    for code, g0 in sorted(raw.items()):
        g = base.enrich(g0)
        c = g["close"]
        trend = (
            (g["ma25"].notna() & (c > g["ma25"]) & (g["ma25_slope5"] > 0))
            | ((g["history_n"] < 60) & g["ma10"].notna() & (c > g["ma10"]) & (g["ma10_slope5"] > 0))
        )
        bullish_reversal = (c > c.shift(1)) & (g["day_pos"] >= 0.60)
        sent_blue = g["sentiment"].between(60, 80)
        sent_contra = g["sentiment"] <= 20
        part_12_20 = g["participation"].between(1.2, 2.0, inclusive="both")

        rows.append({
            "code": code,
            "name": str(g0.stock_name.iloc[0]),
            "rows": int(len(g)),
            "sent_nonnull": int(g.sentiment.notna().sum()),
            "sent_min": None if g.sentiment.dropna().empty else float(g.sentiment.min()),
            "sent_max": None if g.sentiment.dropna().empty else float(g.sentiment.max()),
            "quality_true": int(g.quality.sum()),
            "liquid_true": int(g.liquid.sum()),
            "quality_liquid": int((g.quality & g.liquid).sum()),
            "trend_true": int(trend.sum()),
            "sent_60_80": int(sent_blue.sum()),
            "sent_le20": int(sent_contra.sum()),
            "bullish_reversal": int(bullish_reversal.sum()),
            "part_nonnull": int(g.participation.notna().sum()),
            "part_1.2_2.0": int(part_12_20.sum()),
            "blue_pre_quality": int((sent_blue & trend).sum()),
            "blue_after_quality": int((sent_blue & trend & g.quality).sum()),
            "blue_after_liquid": int((sent_blue & trend & g.quality & g.liquid).sum()),
            "blue_after_part": int((sent_blue & trend & g.quality & g.liquid & part_12_20).sum()),
            "contra_pre_quality": int((sent_contra & bullish_reversal).sum()),
            "contra_after_quality": int((sent_contra & bullish_reversal & g.quality).sum()),
            "contra_after_liquid": int((sent_contra & bullish_reversal & g.quality & g.liquid).sum()),
            "contra_after_part": int((sent_contra & bullish_reversal & g.quality & g.liquid & part_12_20).sum()),
            "base_blue_entry": int(g.blue_entry.sum()),
            "base_contra_entry": int(g.contra_entry.sum()),
            "base_euphoria": int(g.euphoria.sum()),
            "turnover_med20_max": None if g.turnover_med20.dropna().empty else float(g.turnover_med20.max()),
            "participation_max": None if g.participation.dropna().empty else float(g.participation.max()),
        })

    df = pd.DataFrame(rows)
    out = {
        "meta": {"data_start": manifest["dates"][0], "data_end": manifest["dates"][-1]},
        "rows": rows,
    }
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = [
        "code", "rows", "sent_nonnull", "quality_true", "liquid_true", "quality_liquid",
        "trend_true", "sent_60_80", "sent_le20", "part_1.2_2.0",
        "blue_pre_quality", "blue_after_quality", "blue_after_liquid", "blue_after_part",
        "contra_pre_quality", "contra_after_quality", "contra_after_liquid", "contra_after_part",
        "base_blue_entry", "base_contra_entry", "base_euphoria",
    ]
    lines = [
        "# Psychology 11 gate diagnostics", "",
        f"Data: {manifest['dates'][0]} to {manifest['dates'][-1]}", "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|",
    ]
    for _, r in df[cols].iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
    lines += ["", "## Sentiment ranges", "", "| Code | Min | Max | Max 20d median turnover | Max participation |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['code']} | {r['sent_min']} | {r['sent_max']} | {r['turnover_med20_max']} | {r['participation_max']} |")
    a.output_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
