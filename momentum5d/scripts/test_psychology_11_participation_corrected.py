from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import test_psychology_11_roundtrips as base
import test_psychology_11_simple_state as simple
import test_psychology_11_participation_buckets as buckets
import test_psychology_11_roundtrips_corrected as corrected


def pct(v): return "-" if v is None else f"{100*v:.2f}%"
def num(v): return "-" if v is None else f"{v:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    a = ap.parse_args()

    raw, manifest = corrected.load_prices_corrected(a.data_dir)
    all_rows: list[dict[str, Any]] = []
    result: dict[str, Any] = {}

    for bucket in buckets.BUCKETS:
        rows: list[dict[str, Any]] = []
        sigdays = 0
        for code, g0 in sorted(raw.items()):
            g = buckets.prepare(g0, bucket)
            sigdays += int((g.blue_simple | g.contra_simple).sum())
            for tr in simple.simulate(g, "COMBINED"):
                tr["code"] = code
                tr["bucket"] = bucket
                rows.append(tr)
                all_rows.append(tr)
        t = pd.DataFrame(rows)
        s = base.summarize(t) if not t.empty else {"n":0}
        s["signal_days"] = sigdays
        result[bucket] = s

    # Stability of the chosen 1.2-2.0x bucket by signal date year and signal type.
    chosen_rows: list[dict[str, Any]] = []
    for code, g0 in sorted(raw.items()):
        g = buckets.prepare(g0, "A3_1.2_2.0")
        for tr in simple.simulate(g, "COMBINED"):
            tr["code"] = code
            chosen_rows.append(tr)
    chosen = pd.DataFrame(chosen_rows)
    yearly = {}
    by_signal = {}
    by_exit = {}
    if not chosen.empty:
        chosen["year"] = pd.to_datetime(chosen["signal_date"]).dt.year.astype(str)
        for year, t in chosen.groupby("year"):
            yearly[year] = base.summarize(t.copy())
        for sig, t in chosen.groupby("signal"):
            by_signal[sig] = base.summarize(t.copy())
        for reason, t in chosen.groupby("exit_reason"):
            by_exit[reason] = base.summarize(t.copy())

    out = {
        "meta":{"data_start":manifest["dates"][0],"data_end":manifest["dates"][-1],"note":"Corrected full-shard loader; same direct psychology rule; only participation bucket changes; no threshold optimization."},
        "buckets": result,
        "chosen_1.2_2.0_yearly": yearly,
        "chosen_1.2_2.0_by_signal": by_signal,
        "chosen_1.2_2.0_by_exit": by_exit,
    }
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    labels={"ANY":"Any", "A1_LT_0.8":"<0.8x", "A2_0.8_1.2":"0.8-1.2x", "A3_1.2_2.0":"1.2-2.0x", "A4_GE_2.0":">2.0x"}
    lines=["# Corrected 11-stock participation comparison","",f"- Data: {manifest['dates'][0]} to {manifest['dates'][-1]}","- Same direct psychology entry/exit logic; only participation bucket changes","- Corrected loader concatenates all monthly shards","- No parameter optimization","","## Participation buckets","","| Participation | Signal days | Trades | Mean net | Median | Win | PF | Avg MFE | Avg MAE | Avg hold |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k,s in result.items():
        lines.append(f"| {labels[k]} | {s.get('signal_days',0)} | {s.get('n',0)} | {pct(s.get('mean_net'))} | {pct(s.get('median_net'))} | {pct(s.get('win_rate'))} | {num(s.get('pf'))} | {pct(s.get('avg_mfe'))} | {pct(s.get('avg_mae'))} | {num(s.get('avg_hold'))} |")
    lines += ["", "## 1.2-2.0x stability by year", "", "| Year | Trades | Mean net | Median | Win | PF | Avg MFE | Avg MAE |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for y,s in sorted(yearly.items()):
        lines.append(f"| {y} | {s.get('n',0)} | {pct(s.get('mean_net'))} | {pct(s.get('median_net'))} | {pct(s.get('win_rate'))} | {num(s.get('pf'))} | {pct(s.get('avg_mfe'))} | {pct(s.get('avg_mae'))} |")
    lines += ["", "## 1.2-2.0x by signal", "", "| Signal | Trades | Mean net | Median | Win | PF | Avg MFE | Avg MAE |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for sig,s in by_signal.items():
        lines.append(f"| {sig} | {s.get('n',0)} | {pct(s.get('mean_net'))} | {pct(s.get('median_net'))} | {pct(s.get('win_rate'))} | {num(s.get('pf'))} | {pct(s.get('avg_mfe'))} | {pct(s.get('avg_mae'))} |")
    lines += ["", "## 1.2-2.0x by exit reason", "", "| Exit | Trades | Mean net | Median | Win | PF |", "|---|---:|---:|---:|---:|---:|"]
    for reason,s in by_exit.items():
        lines.append(f"| {reason} | {s.get('n',0)} | {pct(s.get('mean_net'))} | {pct(s.get('median_net'))} | {pct(s.get('win_rate'))} | {num(s.get('pf'))} |")
    lines += ["", "This is an in-sample teaching-stock diagnostic. Bucket selection must be validated on untouched stocks/time periods before production use."]
    a.output_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
