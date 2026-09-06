from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import test_psychology_11_simple_state as direct
import compare_psychology_turnover_rank_rise as prev

LOOKBACK_DAYS = 10


def simulate_variant(enriched: dict[str, pd.DataFrame], variant: str, scope: str) -> tuple[pd.DataFrame, dict[str, int]]:
    trades: list[dict[str, Any]] = []
    counts = {"stocks": 0, "blue_signal_days": 0, "contra_signal_days": 0, "combined_signal_days": 0}
    for ticker, g0 in enriched.items():
        code = str(g0["code"].iloc[0])
        teaching = code in prev.TEACHING_CODES
        if scope == "OOS" and teaching:
            continue
        if scope == "TEACHING" and not teaching:
            continue
        counts["stocks"] += 1
        g = g0
        if variant == "PRE10_RISE":
            g = g0.copy()
            g["blue_simple"] = g["blue_simple"] & g["rank_rapid_pre10"]
            g["contra_simple"] = g["contra_simple"] & g["rank_rapid_pre10"]
        counts["blue_signal_days"] += int(g["blue_simple"].sum())
        counts["contra_signal_days"] += int(g["contra_simple"].sum())
        counts["combined_signal_days"] += int((g["blue_simple"] | g["contra_simple"]).sum())
        for tr in prev.fast_simulate(g):
            tr["ticker"] = ticker
            tr["code"] = code
            tr["name"] = str(g["stock_name"].iloc[0])
            tr["variant"] = variant
            tr["scope"] = scope
            trades.append(tr)
    return pd.DataFrame(trades), counts


def summarize(t: pd.DataFrame) -> dict[str, Any]:
    return prev.extra_summary(t)


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100*x:.2f}%"


def num(x: float | None, digits: int = 2) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    a = ap.parse_args()

    raw, manifest = prev.load_all_corrected(a.data_dir)
    rank_pct, rank_rise, rapid = prev.build_market_rank(raw, manifest)

    enriched: dict[str, pd.DataFrame] = {}
    for ticker, g0 in raw.items():
        g = prev.attach_rank(direct.mark_simple_entries(g0), ticker, rank_pct, rank_rise, rapid)
        # Signal day is before next-day-open entry. Include signal day + previous 9 trading days.
        g["rank_rapid_pre10"] = g["rank_rapid"].rolling(LOOKBACK_DAYS, min_periods=1).max().astype(bool)
        enriched[ticker] = g

    result: dict[str, Any] = {
        "meta": {
            "data_start": manifest["dates"][0],
            "data_end": manifest["dates"][-1],
            "loaded_stocks": len(raw),
            "teaching_codes_excluded_from_oos": sorted(prev.TEACHING_CODES),
            "baseline": "existing DIRECT psychology rule, no market-rank filter",
            "pre10_rule": f"at least one turnover-rank rapid-rise day in signal day or prior {LOOKBACK_DAYS-1} trading days",
            "rapid_rise_definition": f"daily TSE turnover percentile >= {prev.RANK_TOP_PCT:.0f} and >= {prev.RANK_RISE_PTS:.0f} percentile points above prior-20-day median",
            "entry_exit": "same DIRECT state machine; next-day open; 0.10% one-way cost; max hold 10 days",
            "optimization": "none; same frozen rapid-rise thresholds as prior test",
        },
        "scopes": {},
    }

    tables: dict[tuple[str, str], pd.DataFrame] = {}
    for scope in ["OOS", "TEACHING"]:
        result["scopes"][scope] = {}
        for variant in ["NO_RANK", "PRE10_RISE"]:
            t, counts = simulate_variant(enriched, variant, scope)
            tables[(scope, variant)] = t
            result["scopes"][scope][variant] = {
                "counts": counts,
                "overall": summarize(t),
                "by_signal": {sig: summarize(t[t["signal"] == sig].copy()) if not t.empty else {"n": 0} for sig in ["BLUE", "CONTRARIAN"]},
                "by_year": {
                    str(y): summarize(t[pd.to_datetime(t["entry_date"]).dt.year == y].copy())
                    for y in sorted(pd.to_datetime(t["entry_date"]).dt.year.unique())
                } if not t.empty else {},
            }

    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Psychology: no rank vs prior-10-day turnover-rank rise", "",
        f"- Data: {manifest['dates'][0]} to {manifest['dates'][-1]}",
        f"- Universe loaded: {len(raw)} stocks",
        "- Primary comparison: OOS, excluding the 11 teaching stocks",
        "- NO_RANK: existing DIRECT psychology rule unchanged",
        f"- PRE10_RISE: at least one rapid-rise day in signal day + previous {LOOKBACK_DAYS-1} trading days",
        f"- Rapid-rise definition frozen: top {100-prev.RANK_TOP_PCT:.0f}% turnover percentile AND +{prev.RANK_RISE_PTS:.0f}pt vs prior-20-day median",
        "- Same entry/exit logic; next-day open; 0.10% each side; max hold 10 days",
        "- No threshold optimization", "",
    ]

    for scope in ["OOS", "TEACHING"]:
        lines += [f"## {scope}", "",
                  "| Variant | Stocks | Signal days | Trades | Mean net | Median | Win | PF | +5% | -5% | -10% | Avg MFE | Avg MAE | Avg hold |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for variant in ["NO_RANK", "PRE10_RISE"]:
            r = result["scopes"][scope][variant]
            s, c = r["overall"], r["counts"]
            lines.append(
                f"| {variant} | {c['stocks']} | {c['combined_signal_days']} | {s.get('n',0)} | {pct(s.get('mean_net'))} | {pct(s.get('median_net'))} | "
                f"{pct(s.get('win_rate'))} | {num(s.get('pf'))} | {pct(s.get('hit_5'))} | {pct(s.get('loss_5'))} | {pct(s.get('loss_10'))} | "
                f"{pct(s.get('avg_mfe'))} | {pct(s.get('avg_mae'))} | {num(s.get('avg_hold'),1)} |"
            )
        if scope == "OOS":
            lines += ["", "### OOS by year", "", "| Year | Variant | Trades | Mean net | Median | Win | PF | -10% |", "|---|---|---:|---:|---:|---:|---:|---:|"]
            years = sorted(set(result["scopes"][scope]["NO_RANK"]["by_year"]) | set(result["scopes"][scope]["PRE10_RISE"]["by_year"]))
            for y in years:
                for variant in ["NO_RANK", "PRE10_RISE"]:
                    s = result["scopes"][scope][variant]["by_year"].get(y, {"n": 0})
                    lines.append(f"| {y} | {variant} | {s.get('n',0)} | {pct(s.get('mean_net'))} | {pct(s.get('median_net'))} | {pct(s.get('win_rate'))} | {num(s.get('pf'))} | {pct(s.get('loss_10'))} |")

    lines += ["", "## Guardrails", "",
              "- OOS is primary; teaching-stock results are secondary diagnostics only.",
              "- Rank history is an entry filter only; exits are identical.",
              "- The 10-day window was specified before seeing this test outcome.",
              "- Trade-level metrics are not a simultaneous-capital portfolio simulation."]

    a.output_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
