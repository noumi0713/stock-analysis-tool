from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import test_psychology_11_roundtrips as base
import test_psychology_11_simple_state as direct

TEACHING_CODES = {"285A", "6976", "593A", "278A", "5801", "5803", "6857", "6920", "9348", "7013", "4592"}
RANK_TOP_PCT = 80.0
RANK_RISE_PTS = 20.0


def load_all_corrected(data_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    dates = pd.to_datetime(manifest["dates"])
    meta = manifest.get("stocks", {})
    chunks: dict[str, list[pd.DataFrame]] = defaultdict(list)

    for shard in manifest["shards"]:
        payload = json.loads((data_dir / shard["path"]).read_text(encoding="utf-8"))
        for ticker, bars in payload["bars"].items():
            if not bars:
                continue
            a = np.asarray(bars, dtype=float)
            code = str(meta.get(ticker, {}).get("code") or ticker.removesuffix(".T"))[:4]
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
            chunks[ticker].append(g)

    out: dict[str, pd.DataFrame] = {}
    for ticker, frames in chunks.items():
        g = pd.concat(frames, ignore_index=True)
        g = g.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
        out[ticker] = g
    return out, manifest


def build_market_rank(raw: dict[str, pd.DataFrame], manifest: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(manifest["dates"])
    tickers = sorted(raw)
    col = {t: i for i, t in enumerate(tickers)}
    row = {d: i for i, d in enumerate(dates)}
    mat = np.full((len(dates), len(tickers)), np.nan, dtype=np.float64)

    for ticker, g in raw.items():
        j = col[ticker]
        turnover = (g["close"].to_numpy(dtype=float) * g["volume"].to_numpy(dtype=float))
        for d, v in zip(pd.to_datetime(g["date"]), turnover):
            i = row.get(d)
            if i is not None and np.isfinite(v) and v >= 0:
                mat[i, j] = v

    turn = pd.DataFrame(mat, index=dates, columns=tickers)
    # 100 = the most actively traded end of the market cross-section.
    rank_pct = turn.rank(axis=1, method="average", pct=True) * 100.0
    prior20_med = rank_pct.shift(1).rolling(20, min_periods=10).median()
    rank_rise = rank_pct - prior20_med
    rapid = (rank_pct >= RANK_TOP_PCT) & (rank_rise >= RANK_RISE_PTS)
    return rank_pct, rank_rise, rapid


def attach_rank(g: pd.DataFrame, ticker: str, rank_pct: pd.DataFrame, rank_rise: pd.DataFrame, rapid: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    d = pd.to_datetime(g["date"])
    g["market_turnover_pct"] = rank_pct[ticker].reindex(d).to_numpy()
    g["market_rank_rise_pts"] = rank_rise[ticker].reindex(d).to_numpy()
    g["rank_rapid"] = rapid[ticker].reindex(d).fillna(False).to_numpy(dtype=bool)
    return g


def extra_summary(t: pd.DataFrame) -> dict[str, Any]:
    s = base.summarize(t) if not t.empty else {"n": 0}
    if t.empty:
        s.update({"hit_5": None, "loss_5": None, "loss_10": None})
        return s
    s.update({
        "hit_5": float((t["net_return"] >= 0.05).mean()),
        "loss_5": float((t["net_return"] <= -0.05).mean()),
        "loss_10": float((t["net_return"] <= -0.10).mean()),
    })
    # Compound/max_dd from base are not portfolio statistics when trades overlap across stocks.
    s.pop("compound", None)
    s.pop("max_dd", None)
    return s


def simulate_variant(enriched: dict[str, pd.DataFrame], variant: str, scope: str) -> tuple[pd.DataFrame, dict[str, int]]:
    trades: list[dict[str, Any]] = []
    counts = {"stocks": 0, "blue_signal_days": 0, "contra_signal_days": 0, "combined_signal_days": 0}

    for ticker, g0 in enriched.items():
        code = str(g0["code"].iloc[0])
        teaching = code in TEACHING_CODES
        if scope == "OOS" and teaching:
            continue
        if scope == "TEACHING" and not teaching:
            continue
        counts["stocks"] += 1

        g = g0.copy()
        if variant == "RANK_RISE":
            g["blue_simple"] = g["blue_simple"] & g["rank_rapid"]
            g["contra_simple"] = g["contra_simple"] & g["rank_rapid"]

        counts["blue_signal_days"] += int(g["blue_simple"].sum())
        counts["contra_signal_days"] += int(g["contra_simple"].sum())
        counts["combined_signal_days"] += int((g["blue_simple"] | g["contra_simple"]).sum())

        for tr in direct.simulate(g, "COMBINED"):
            tr["ticker"] = ticker
            tr["code"] = code
            tr["name"] = str(g["stock_name"].iloc[0])
            tr["variant"] = variant
            tr["scope"] = scope
            sig_date = pd.Timestamp(tr["signal_date"])
            rr = g.loc[pd.to_datetime(g["date"]) == sig_date]
            if not rr.empty:
                tr["market_turnover_pct"] = float(rr["market_turnover_pct"].iloc[0]) if pd.notna(rr["market_turnover_pct"].iloc[0]) else None
                tr["market_rank_rise_pts"] = float(rr["market_rank_rise_pts"].iloc[0]) if pd.notna(rr["market_rank_rise_pts"].iloc[0]) else None
            trades.append(tr)

    t = pd.DataFrame(trades)
    return t, counts


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

    raw, manifest = load_all_corrected(a.data_dir)
    rank_pct, rank_rise, rapid = build_market_rank(raw, manifest)

    enriched: dict[str, pd.DataFrame] = {}
    for ticker, g0 in raw.items():
        g = direct.mark_simple_entries(g0)
        enriched[ticker] = attach_rank(g, ticker, rank_pct, rank_rise, rapid)

    result: dict[str, Any] = {
        "meta": {
            "data_start": manifest["dates"][0],
            "data_end": manifest["dates"][-1],
            "loaded_stocks": len(raw),
            "teaching_codes_excluded_from_oos": sorted(TEACHING_CODES),
            "baseline": "existing DIRECT psychology rule; no market turnover rank filter",
            "rank_rise_rule": f"current TSE turnover percentile >= {RANK_TOP_PCT:.0f} and >= {RANK_RISE_PTS:.0f} percentile points above prior-20-day median",
            "rank_definition": "cross-sectional daily close*volume percentile; 100 = highest turnover",
            "entry_exit": "same DIRECT state machine; next-day open execution; 0.10% one-way cost; max hold 10 trading days",
            "optimization": "none; thresholds frozen before this comparison",
        },
        "scopes": {},
    }

    all_tables: dict[tuple[str, str], pd.DataFrame] = {}
    for scope in ["OOS", "TEACHING"]:
        result["scopes"][scope] = {}
        for variant in ["NO_RANK", "RANK_RISE"]:
            t, counts = simulate_variant(enriched, variant, scope)
            all_tables[(scope, variant)] = t
            result["scopes"][scope][variant] = {
                "counts": counts,
                "overall": extra_summary(t),
                "by_signal": {
                    sig: extra_summary(t[t["signal"] == sig].copy()) if not t.empty else {"n": 0}
                    for sig in ["BLUE", "CONTRARIAN"]
                },
                "by_year": {
                    str(y): extra_summary(t[pd.to_datetime(t["entry_date"]).dt.year == y].copy())
                    for y in sorted(pd.to_datetime(t["entry_date"]).dt.year.unique())
                } if not t.empty else {},
            }

    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Psychology: no market rank vs turnover-rank rapid rise", "",
        f"- Data: {manifest['dates'][0]} to {manifest['dates'][-1]}",
        f"- Universe loaded: {len(raw)} stocks",
        "- Primary result is OOS: 11 teaching stocks excluded",
        "- NO_RANK: existing DIRECT psychology rule unchanged",
        f"- RANK_RISE: current turnover percentile top {100-RANK_TOP_PCT:.0f}% AND +{RANK_RISE_PTS:.0f}pt or more vs prior-20-day median",
        "- Same entry/exit logic, next-day open, 0.10% cost each side, max 10 days",
        "- No threshold optimization/grid search", "",
    ]

    for scope in ["OOS", "TEACHING"]:
        lines += [f"## {scope}", "",
                  "| Variant | Stocks | Signal days | Trades | Mean net | Median | Win | PF | +5% | -5% | -10% | Avg MFE | Avg MAE | Avg hold |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for variant in ["NO_RANK", "RANK_RISE"]:
            r = result["scopes"][scope][variant]
            s, c = r["overall"], r["counts"]
            lines.append(
                f"| {variant} | {c['stocks']} | {c['combined_signal_days']} | {s.get('n',0)} | {pct(s.get('mean_net'))} | {pct(s.get('median_net'))} | "
                f"{pct(s.get('win_rate'))} | {num(s.get('pf'))} | {pct(s.get('hit_5'))} | {pct(s.get('loss_5'))} | {pct(s.get('loss_10'))} | "
                f"{pct(s.get('avg_mfe'))} | {pct(s.get('avg_mae'))} | {num(s.get('avg_hold'),1)} |"
            )
        if scope == "OOS":
            lines += ["", "### OOS by year", "",
                      "| Year | Variant | Trades | Mean net | Median | Win | PF | -10% |",
                      "|---|---|---:|---:|---:|---:|---:|---:|"]
            years = sorted(set(result["scopes"][scope]["NO_RANK"]["by_year"]) | set(result["scopes"][scope]["RANK_RISE"]["by_year"]))
            for y in years:
                for variant in ["NO_RANK", "RANK_RISE"]:
                    s = result["scopes"][scope][variant]["by_year"].get(y, {"n": 0})
                    lines.append(f"| {y} | {variant} | {s.get('n',0)} | {pct(s.get('mean_net'))} | {pct(s.get('median_net'))} | {pct(s.get('win_rate'))} | {num(s.get('pf'))} | {pct(s.get('loss_10'))} |")

    lines += ["", "## Interpretation guardrails", "",
              "- OOS is the primary comparison because the 11 teaching stocks were selected after chart observation.",
              "- Market-rank rapid rise is an entry filter only; exits are identical between variants.",
              "- This test does not optimize the top-percentile or rank-rise thresholds after seeing outcomes.",
              "- Trade-level PF/mean are not a simultaneous-capital portfolio simulation."]

    a.output_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
