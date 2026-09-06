from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import test_psychology_11_roundtrips as strict
import test_psychology_11_simple_state as direct

CODES = {"285A", "6976", "593A", "278A", "5801", "5803", "6857", "6920", "9348", "7013", "4592"}


def load_prices_corrected(data_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    dates = pd.to_datetime(manifest["dates"])
    meta = manifest.get("stocks", {})
    chunks: dict[str, list[pd.DataFrame]] = defaultdict(list)

    for shard in manifest["shards"]:
        payload = json.loads((data_dir / shard["path"]).read_text(encoding="utf-8"))
        for ticker, bars in payload["bars"].items():
            code = str(meta.get(ticker, {}).get("code") or ticker.removesuffix(".T"))[:4]
            if code not in CODES or not bars:
                continue
            a = np.asarray(bars, dtype=float)
            g = pd.DataFrame({
                "date": dates[a[:, 0].astype(int)],
                "open": a[:, 1],
                "high": a[:, 2],
                "low": a[:, 3],
                "close": a[:, 4],
                "volume": a[:, 5],
            })
            g["code"] = code
            g["ticker"] = ticker
            g["stock_name"] = str(meta.get(ticker, {}).get("name") or ticker)
            chunks[code].append(g)

    out: dict[str, pd.DataFrame] = {}
    for code, frames in chunks.items():
        g = pd.concat(frames, ignore_index=True)
        g = g.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        out[code] = g
    return out, manifest


def serialize_dates(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    for col in ["signal_date", "entry_date", "exit_signal_date", "exit_date"]:
        if col in t.columns:
            t[col] = pd.to_datetime(t[col]).dt.strftime("%Y-%m-%d")
    return t


def pct(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:.2f}%"


def num(v: float | None, digits: int = 2) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def run_strategy(raw: dict[str, pd.DataFrame], strategy: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    trades: list[dict[str, Any]] = []
    signal_counts: dict[str, dict[str, int]] = {}

    for code, g0 in sorted(raw.items()):
        if strategy == "STRICT":
            g = strict.enrich(g0)
            blue_col, contra_col = "blue_entry", "contra_entry"
            sim = strict.simulate
        else:
            g = direct.mark_simple_entries(g0)
            blue_col, contra_col = "blue_simple", "contra_simple"
            sim = direct.simulate

        signal_counts[code] = {
            "rows": int(len(g)),
            "blue_days": int(g[blue_col].sum()),
            "contra_days": int(g[contra_col].sum()),
            "euphoria_days": int(g["euphoria"].sum()),
            "sentiment_nonnull": int(g["sentiment"].notna().sum()),
            "participation_1.2_2.0_days": int(g["participation"].between(1.2, 2.0, inclusive="both").sum()),
        }

        for tr in sim(g, "COMBINED"):
            tr["code"] = code
            tr["name"] = str(g0.stock_name.iloc[0])
            tr["strategy"] = strategy
            trades.append(tr)

    t = pd.DataFrame(trades)
    return serialize_dates(t), signal_counts


def summarize_by_code(t: pd.DataFrame, raw: dict[str, pd.DataFrame]) -> dict[str, Any]:
    out = {}
    for code in sorted(raw):
        mt = t[t["code"] == code].copy() if not t.empty else pd.DataFrame()
        out[code] = strict.summarize(mt)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    a = ap.parse_args()

    raw, manifest = load_prices_corrected(a.data_dir)
    strict_t, strict_counts = run_strategy(raw, "STRICT")
    direct_t, direct_counts = run_strategy(raw, "DIRECT")

    result = {
        "meta": {
            "data_start": manifest["dates"][0],
            "data_end": manifest["dates"][-1],
            "codes": sorted(raw),
            "loaded_rows": {code: int(len(g)) for code, g in sorted(raw.items())},
            "entry_execution": "next trading day open",
            "exit_execution": "next trading day open",
            "cost_one_way": 0.001,
            "max_hold": 10,
            "participation": "1.2x-2.0x current turnover / 20d median",
            "loader_fix": "concatenate all monthly shards per ticker instead of overwriting with final shard",
        },
        "STRICT": {
            "overall": strict.summarize(strict_t),
            "by_code": summarize_by_code(strict_t, raw),
            "signal_counts": strict_counts,
            "trades": strict_t.to_dict("records") if not strict_t.empty else [],
        },
        "DIRECT": {
            "overall": strict.summarize(direct_t),
            "by_code": summarize_by_code(direct_t, raw),
            "signal_counts": direct_counts,
            "trades": direct_t.to_dict("records") if not direct_t.empty else [],
        },
    }

    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Corrected 11-stock psychology round-trip test", "",
        f"- Data: {manifest['dates'][0]} to {manifest['dates'][-1]}",
        "- Fixed loader: all monthly shards concatenated per ticker",
        "- Entry/exit execution: next trading day open",
        "- Cost: 0.10% each side",
        "- Max hold: 10 trading days",
        "- Participation high-confidence zone: 1.2x–2.0x current turnover / 20d median",
        "- No grid-search/parameter optimization in this run", "",
        "## Loaded rows", "",
        "| Code | Rows |", "|---|---:|",
    ]
    for code, g in sorted(raw.items()):
        lines.append(f"| {code} | {len(g)} |")

    lines += ["", "## Overall comparison", "",
              "| Strategy | Trades | Mean net | Median | Win | PF | Compound* | Max DD* | Avg MFE | Avg MAE | Avg hold |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for strategy, t in [("STRICT", strict_t), ("DIRECT", direct_t)]:
        s = result[strategy]["overall"]
        lines.append(
            f"| {strategy} | {s.get('n',0)} | {pct(s.get('mean_net'))} | {pct(s.get('median_net'))} | "
            f"{pct(s.get('win_rate'))} | {num(s.get('pf'))} | {pct(s.get('compound'))} | {pct(s.get('max_dd'))} | "
            f"{pct(s.get('avg_mfe'))} | {pct(s.get('avg_mae'))} | {num(s.get('avg_hold'),1)} |"
        )

    lines += ["", "## DIRECT by code", "",
              "| Code | Blue days | Contra days | Euphoria days | Trades | Mean net | Win | PF | Compound* | Max DD* |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for code in sorted(raw):
        c = direct_counts[code]
        s = result["DIRECT"]["by_code"][code]
        lines.append(
            f"| {code} | {c['blue_days']} | {c['contra_days']} | {c['euphoria_days']} | {s.get('n',0)} | "
            f"{pct(s.get('mean_net'))} | {pct(s.get('win_rate'))} | {num(s.get('pf'))} | {pct(s.get('compound'))} | {pct(s.get('max_dd'))} |"
        )

    lines += ["", "## DIRECT trades", "",
              "| Code | Signal | Signal date | Entry | Exit | Exit reason | Hold | Net | MFE | MAE | Sentiment | Participation |",
              "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    if direct_t.empty:
        lines.append("| - | - | - | - | - | - | - | - | - | - | - | - |")
    else:
        for _, r in direct_t.sort_values(["entry_date", "code"]).iterrows():
            lines.append(
                f"| {r['code']} | {r['signal']} | {r['signal_date']} | {r['entry_date']} @ {r['entry_price']:.2f} | "
                f"{r['exit_date']} @ {r['exit_price']:.2f} | {r['exit_reason']} | {int(r['hold_days'])} | "
                f"{pct(float(r['net_return']))} | {pct(float(r['mfe']))} | {pct(float(r['mae']))} | "
                f"{float(r['entry_sentiment']):.1f} | {float(r['entry_participation']):.2f}x |"
            )

    lines += ["", "* Compound/Max DD are sequential-trade diagnostics, not a capital-allocation portfolio simulation.",
              "", "These 11 stocks were selected after observing their charts. This is an in-sample teaching-stock audit, not OOS proof."]

    a.output_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
