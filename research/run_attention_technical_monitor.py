from __future__ import annotations

import argparse
import json
from pathlib import Path

from first_pullback_watch_engine_v2 import new_state, update_one
from run_ifis_first_pullback_monitor import fetch_bars, discovery_index, sma, rsi, atr


def f(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def enrich_v3(rows: list[dict]) -> list[dict]:
    out, highs, lows, closes, vols = [], [], [], [], []
    for row in rows:
        highs.append(f(row["high"])); lows.append(f(row["low"])); closes.append(f(row["close"])); vols.append(f(row["volume"]))
        c = closes[-1]; ma5 = sma(closes, 5); ma25 = sma(closes, 25); ma75 = sma(closes, 75)
        rr = rsi(closes); aa = atr(highs, lows, closes)
        ret1 = c / closes[-2] - 1 if len(closes) >= 2 and closes[-2] else 0.0
        ret5 = c / closes[-6] - 1 if len(closes) >= 6 and closes[-6] else 0.0
        ret10 = c / closes[-11] - 1 if len(closes) >= 11 and closes[-11] else 0.0
        ret20 = c / closes[-21] - 1 if len(closes) >= 21 and closes[-21] else 0.0
        ma25d = c / ma25 - 1 if ma25 else 0.0
        avg20 = sum(vols[-20:]) / min(20, len(vols)) if vols else 0.0; vr = vols[-1] / avg20 if avg20 else 0.0
        high20 = max(highs[-20:]) if highs else c; distance20 = c / high20 - 1 if high20 else 0.0
        o, h, l = f(row.get("open"), c), f(row.get("high"), c), f(row.get("low"), c); rng = h - l
        upper_wick = (h - max(o, c)) / rng if rng > 0 else 0.0; close_location = (c - l) / rng if rng > 0 else 0.5
        score = 0.0
        if rr is not None:
            score += max(0.0, min(30.0, (rr - 55.0) * 1.5))
        score += max(0.0, min(25.0, ret5 * 125.0)); score += max(0.0, min(20.0, ret20 * 50.0)); score += max(0.0, min(15.0, ma25d * 100.0))
        if vr > 2.0:
            score += min(10.0, (vr - 2.0) * 4.0)
        x = dict(row)
        x.update({
            "ma5": ma5, "ma25": ma25 if ma25 is not None else c, "ma75": ma75,
            "rsi14": rr if rr is not None else 50.0, "atr14": aa if aa is not None else 0.0,
            "atr14_pct": (aa / c * 100.0) if aa is not None and c else 0.0,
            "return_1d_pct": ret1 * 100.0, "return_5d_pct": ret5 * 100.0, "return_10d_pct": ret10 * 100.0, "return_20d_pct": ret20 * 100.0,
            "ma25_deviation_pct": ma25d * 100.0, "volume_ratio20": vr, "upper_wick_ratio": upper_wick,
            "close_location": close_location, "distance_from_20d_high_pct": distance20 * 100.0,
            "overheat_score": max(0.0, min(100.0, score)),
        })
        out.append(x)
    return out


def compact_metrics(row: dict, float_shares: float | None = None) -> dict:
    volume = f(row.get("volume")); turnover = (volume / float_shares * 100.0) if float_shares and float_shares > 0 else None
    keys = ["rsi14","ma5","ma25","ma75","atr14","atr14_pct","return_1d_pct","return_5d_pct","return_10d_pct","return_20d_pct","ma25_deviation_pct","volume_ratio20","upper_wick_ratio","close_location","distance_from_20d_high_pct","overheat_score"]
    out = {k: (round(f(row.get(k)), 6) if row.get(k) is not None else None) for k in keys}
    out["float_turnover_pct"] = round(turnover, 6) if turnover is not None else None
    out["float_turnover_evaluable"] = turnover is not None
    return out


def monitor_candidate(candidate: dict) -> dict:
    bars = enrich_v3(fetch_bars(candidate["ticker"])); idx = discovery_index(bars, candidate["snapshot_at"])
    resolved_symbol = bars[0].get("resolved_symbol", candidate["ticker"]) if bars else candidate["ticker"]
    try:
        float_shares = float(candidate.get("float_shares")) if candidate.get("float_shares") not in (None, "") else None
    except Exception:
        float_shares = None
    base = {
        "ifis_rank": candidate["ifis_rank"], "stock_code": candidate["stock_code"], "ticker": candidate["ticker"], "resolved_symbol": resolved_symbol,
        "company_name": candidate["company_name"], "snapshot_at": candidate["snapshot_at"], "best_theme": candidate["best_theme"],
        "theme_relevance_score": candidate["theme_relevance_score"], "candidate_class": candidate.get("candidate_class", ""),
    }
    if idx is None:
        return {**base, "ok": False, "technical_status": "NO_DISCOVERY_BAR", "error": "no completed market bar existed at IFIS snapshot"}
    if idx < 24:
        return {**base, "ok": False, "technical_status": "INSUFFICIENT_HISTORY", "discovery_market_date": bars[idx]["date"], "available_prior_bars": idx + 1, "error": "fewer than 25 bars available at discovery"}
    dbar = bars[idx]
    discovery = {"ticker": resolved_symbol, "stock_code": candidate["stock_code"], "name": candidate["company_name"], "snapshot_at": candidate["snapshot_at"], "discovery_price": dbar["close"], "theme_relevance_score": candidate["theme_relevance_score"], "theme_relevance_band": candidate.get("theme_relevance_band"), "theme_relevance_confidence": candidate.get("theme_relevance_confidence"), "candidate_class": candidate.get("candidate_class", ""), "ifis_rank": candidate["ifis_rank"], "best_theme": candidate["best_theme"]}
    state = new_state(discovery, dbar["date"], dbar)
    for row in bars[idx + 1:]:
        state = update_one(state, row, row["date"])
        if state.get("terminal"):
            break
    latest_bar = bars[-1]
    return {**base, "ok": True, "technical_status": state["status"], "terminal": state["terminal"],
        "discovery_market_date": dbar["date"], "discovery_price": round(dbar["close"], 4), "discovery_metrics": compact_metrics(dbar, float_shares),
        "latest_market_date": latest_bar["date"], "latest_close": round(latest_bar["close"], 4), "latest_metrics": compact_metrics(latest_bar, float_shares),
        "last_updated": state.get("last_updated") or dbar["date"], "last_reasons": state.get("last_reasons", ["no post-discovery completed bar yet"]),
        "max_gain_from_discovery_pct": round(float(state.get("max_gain_from_discovery", 0.0)) * 100.0, 4),
        "deepest_pullback_from_peak_pct": round(float(state.get("deepest_pullback_from_peak", 0.0)) * 100.0, 4), "history": state.get("history", [])}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--watch", required=True); ap.add_argument("--out", required=True); args = ap.parse_args()
    watch = json.loads(Path(args.watch).read_text(encoding="utf-8")); items = []
    for c in watch.get("primary_watch", []):
        try:
            items.append(monitor_candidate(c))
        except Exception as e:
            items.append({"ifis_rank": c.get("ifis_rank"), "stock_code": c.get("stock_code"), "ticker": c.get("ticker"), "company_name": c.get("company_name"), "ok": False, "technical_status": "FETCH_ERROR", "error": str(e)})
    payload = {"status": "complete", "stage": "ENTRY_TIMING_TECHNICAL", "source": "Yahoo Finance chart endpoint", "price_basis": "adjusted close factor applied to OHLC", "point_in_time_rule": "Discovery metrics use only completed daily bars available at the IFIS snapshot; snapshots before 16:00 JST use the prior market day.", "metrics": ["volume_ratio20","float_turnover_pct(optional)","RSI14","MA5/25/75","MA25 deviation","ATR14%","1/5/10/20-day returns","upper wick ratio","distance from 20-day high","overheat score"], "excluded_inputs": ["Minkabu"], "actionability": "RESEARCH_ONLY", "candidate_count": len(items), "ok_count": sum(1 for x in items if x.get("ok")), "error_count": sum(1 for x in items if not x.get("ok")), "items": sorted(items, key=lambda x: int(x.get("ifis_rank") or 999999))}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "candidate_count": payload["candidate_count"], "ok_count": payload["ok_count"], "error_count": payload["error_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
