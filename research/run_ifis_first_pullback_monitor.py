from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from first_pullback_watch_engine_v2 import new_state, update_one

UA = "Mozilla/5.0"
TOKYO = ZoneInfo("Asia/Tokyo")


def sma(xs: list[float], n: int) -> float | None:
    return sum(xs[-n:]) / n if len(xs) >= n else None


def rsi(xs: list[float], n: int = 14) -> float | None:
    if len(xs) < n + 1:
        return None
    gains, losses = [], []
    for i in range(len(xs) - n, len(xs)):
        d = xs[i] - xs[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag = sum(gains) / n
    al = sum(losses) / n
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


def atr(highs: list[float], lows: list[float], closes: list[float], n: int = 14) -> float | None:
    if len(closes) < n + 1:
        return None
    vals = []
    for i in range(len(closes) - n, len(closes)):
        vals.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    return sum(vals) / n


def symbol_candidates(ticker: str) -> list[str]:
    raw = str(ticker or "").strip().upper()
    code = raw.split(".", 1)[0]
    candidates = [raw] if raw else []
    for suffix in (".T", ".S", ".N", ".F"):
        symbol = f"{code}{suffix}"
        if symbol not in candidates:
            candidates.append(symbol)
    return candidates


def fetch_bars(ticker: str) -> list[dict]:
    last_error: Exception | None = None
    for symbol in symbol_candidates(ticker):
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            + urllib.parse.quote(symbol)
            + "?range=1y&interval=1d&events=div%2Csplits&includeAdjustedClose=true"
        )
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = json.load(r)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_error = e
            continue

        chart = raw.get("chart", {})
        if chart.get("error") or not chart.get("result"):
            last_error = ValueError(str(chart.get("error") or "missing chart result"))
            continue

        result = chart["result"][0]
        ts = result.get("timestamp", [])
        q = result["indicators"]["quote"][0]
        adj = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose", [])
        rows = []
        for i, stamp in enumerate(ts):
            try:
                o = q["open"][i]; h = q["high"][i]; l = q["low"][i]; c = q["close"][i]
                v = q.get("volume", [0] * len(ts))[i] or 0
            except (IndexError, TypeError):
                continue
            if any(x is None for x in (o, h, l, c)):
                continue
            ac = adj[i] if i < len(adj) and adj[i] is not None else c
            factor = ac / c if c else 1.0
            d = datetime.fromtimestamp(stamp, timezone.utc).astimezone(TOKYO).date().isoformat()
            rows.append({
                "date": d,
                "open": float(o) * factor,
                "high": float(h) * factor,
                "low": float(l) * factor,
                "close": float(c) * factor,
                "volume": float(v),
                "resolved_symbol": symbol,
            })
        if rows:
            return rows
        last_error = ValueError(f"no usable bars for {symbol}")

    raise ValueError(f"Yahoo symbol resolution failed for {ticker}: {last_error}")


def enrich(rows: list[dict]) -> list[dict]:
    out = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    vols: list[float] = []
    for row in rows:
        opens.append(row["open"]); highs.append(row["high"]); lows.append(row["low"])
        closes.append(row["close"]); vols.append(row["volume"])
        c = closes[-1]
        ma25 = sma(closes, 25)
        ma75 = sma(closes, 75)
        rr = rsi(closes)
        aa = atr(highs, lows, closes)
        ret5 = c / closes[-6] - 1 if len(closes) >= 6 else 0.0
        ret20 = c / closes[-21] - 1 if len(closes) >= 21 else 0.0
        ma25d = c / ma25 - 1 if ma25 else 0.0
        avg20 = sum(vols[-20:]) / min(20, len(vols)) if vols else 0.0
        vr = vols[-1] / avg20 if avg20 else 0.0
        score = 0.0
        if rr is not None:
            score += max(0.0, min(30.0, (rr - 55.0) * 1.5))
        score += max(0.0, min(25.0, ret5 * 125.0))
        score += max(0.0, min(20.0, ret20 * 50.0))
        score += max(0.0, min(15.0, ma25d * 100.0))
        if vr > 2.0:
            score += min(10.0, (vr - 2.0) * 4.0)
        x = dict(row)
        x.update({
            "rsi14": rr if rr is not None else 50.0,
            "ma25": ma25 if ma25 is not None else c,
            "ma75": ma75,
            "atr14": aa if aa is not None else 0.0,
            "overheat_score": max(0.0, min(100.0, score)),
        })
        out.append(x)
    return out


def discovery_index(rows: list[dict], snapshot_at: str) -> int | None:
    snap = datetime.fromisoformat(snapshot_at)
    if snap.tzinfo is None:
        snap = snap.replace(tzinfo=TOKYO)
    local = snap.astimezone(TOKYO)
    same_day_close_known = local.timetz().replace(tzinfo=None) >= time(16, 0)
    eligible = []
    for i, row in enumerate(rows):
        d = datetime.fromisoformat(row["date"]).date()
        if d < local.date() or (same_day_close_known and d == local.date()):
            eligible.append(i)
    return eligible[-1] if eligible else None


def monitor_candidate(candidate: dict) -> dict:
    bars = enrich(fetch_bars(candidate["ticker"]))
    idx = discovery_index(bars, candidate["snapshot_at"])
    resolved_symbol = bars[0].get("resolved_symbol", candidate["ticker"]) if bars else candidate["ticker"]
    base = {
        "ifis_rank": candidate["ifis_rank"],
        "stock_code": candidate["stock_code"],
        "ticker": candidate["ticker"],
        "resolved_symbol": resolved_symbol,
        "company_name": candidate["company_name"],
        "snapshot_at": candidate["snapshot_at"],
        "best_theme": candidate["best_theme"],
        "theme_relevance_score": candidate["theme_relevance_score"],
        "candidate_class": candidate["candidate_class"],
    }
    if idx is None:
        return {**base, "ok": False, "technical_status": "NO_DISCOVERY_BAR", "error": "no completed market bar existed at IFIS snapshot"}
    if idx < 24:
        return {
            **base,
            "ok": False,
            "technical_status": "INSUFFICIENT_HISTORY",
            "discovery_market_date": bars[idx]["date"],
            "available_prior_bars": idx + 1,
            "error": "fewer than 25 bars available at discovery",
        }

    dbar = bars[idx]
    discovery = {
        "ticker": resolved_symbol,
        "stock_code": candidate["stock_code"],
        "name": candidate["company_name"],
        "snapshot_at": candidate["snapshot_at"],
        "discovery_price": dbar["close"],
        "theme_relevance_score": candidate["theme_relevance_score"],
        "theme_relevance_band": candidate.get("theme_relevance_band"),
        "theme_relevance_confidence": candidate.get("theme_relevance_confidence"),
        "candidate_class": candidate["candidate_class"],
        "ifis_rank": candidate["ifis_rank"],
        "best_theme": candidate["best_theme"],
    }
    state = new_state(discovery, dbar["date"], dbar)
    for row in bars[idx + 1:]:
        state = update_one(state, row, row["date"])
        if state.get("terminal"):
            break

    latest_bar = bars[-1]
    return {
        **base,
        "ok": True,
        "technical_status": state["status"],
        "terminal": state["terminal"],
        "discovery_market_date": dbar["date"],
        "discovery_price": round(dbar["close"], 4),
        "latest_market_date": latest_bar["date"],
        "latest_close": round(latest_bar["close"], 4),
        "last_updated": state.get("last_updated") or dbar["date"],
        "last_reasons": state.get("last_reasons", ["no post-discovery completed bar yet"]),
        "max_gain_from_discovery_pct": round(float(state.get("max_gain_from_discovery", 0.0)) * 100.0, 4),
        "deepest_pullback_from_peak_pct": round(float(state.get("deepest_pullback_from_peak", 0.0)) * 100.0, 4),
        "history": state.get("history", []),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    watch = json.loads(Path(args.watch).read_text(encoding="utf-8"))
    items = []
    for c in watch.get("primary_watch", []):
        try:
            items.append(monitor_candidate(c))
        except Exception as e:
            items.append({
                "ifis_rank": c.get("ifis_rank"),
                "stock_code": c.get("stock_code"),
                "ticker": c.get("ticker"),
                "company_name": c.get("company_name"),
                "ok": False,
                "technical_status": "FETCH_ERROR",
                "error": str(e),
            })

    payload = {
        "status": "complete",
        "source": "Yahoo Finance chart endpoint",
        "price_basis": "adjusted close factor applied to OHLC",
        "symbol_rule": "Try requested symbol, then Japanese Yahoo suffixes .T/.S/.N/.F for regional listings.",
        "point_in_time_rule": "Discovery price uses only a completed market daily bar available at the IFIS snapshot. Snapshots before 16:00 JST use the prior market day.",
        "actionability": "RESEARCH_ONLY_PENDING_CATALYST_REVIEW",
        "candidate_count": len(items),
        "ok_count": sum(1 for x in items if x.get("ok")),
        "error_count": sum(1 for x in items if not x.get("ok")),
        "items": sorted(items, key=lambda x: int(x.get("ifis_rank") or 999999)),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "candidate_count": payload["candidate_count"],
        "ok_count": payload["ok_count"],
        "error_count": payload["error_count"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
