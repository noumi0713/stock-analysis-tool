from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Research-only prototype. Thresholds are deliberately coarse and must be validated
# before any production use.

STATUS_BUY_NOW = "BUY_NOW"
STATUS_WAIT = "WAIT_FIRST_PULLBACK"
STATUS_FORMING = "PULLBACK_FORMING"
STATUS_SIGNAL = "FIRST_PULLBACK_SIGNAL"
STATUS_RAN_AWAY = "EXCLUDE_RAN_AWAY"
STATUS_INVALID = "INVALIDATED"
TERMINAL = {STATUS_SIGNAL, STATUS_RAN_AWAY, STATUS_INVALID}

# Coarse prototype thresholds; not production parameters.
BUY_MAX_OVERHEAT = 20.0
BUY_MAX_RSI = 65.0
RUNAWAY_GAIN = 0.10
RUNAWAY_MIN_OVERHEAT = 40.0
PULLBACK_MIN = 0.03
PULLBACK_MAX = 0.12
INVALID_MAX_DRAWDOWN = 0.15
MA25_BREAK = 0.97
SIGNAL_MAX_OVERHEAT = 30.0
SIGNAL_RSI_MIN = 45.0
SIGNAL_RSI_MAX = 68.0


def f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def close_location(row: dict[str, Any]) -> float:
    hi, lo, close = f(row.get("high")), f(row.get("low")), f(row.get("close"))
    return (close - lo) / (hi - lo) if hi > lo else 0.5


def pct(a: float, b: float) -> float:
    return a / b - 1.0 if b else 0.0


@dataclass
class Decision:
    status: str
    reasons: list[str]


def validate_new_candidate(c: dict[str, Any]) -> None:
    required = ("ticker", "name", "discovery_price", "minkabu_relevance", "kabutan_member", "ifis_rank")
    missing = [k for k in required if c.get(k) in (None, "")]
    if missing:
        raise ValueError(f"new candidate missing fields: {', '.join(missing)}")
    if not bool(c.get("kabutan_member")):
        raise ValueError("candidate must be a Kabutan theme member in this prototype")


def new_state(c: dict[str, Any], as_of: str, technical: dict[str, Any]) -> dict[str, Any]:
    validate_new_candidate(c)
    discovery_price = f(c["discovery_price"])
    first_high = max(discovery_price, f(technical.get("high"), discovery_price))
    return {
        "ticker": str(c["ticker"]),
        "name": str(c["name"]),
        "discovered_date": str(c.get("discovered_date") or as_of),
        "discovery_price": discovery_price,
        "catalyst_date": c.get("catalyst_date"),
        "catalyst_reason": c.get("catalyst_reason"),
        "theme": c.get("theme"),
        "minkabu_relevance": f(c.get("minkabu_relevance")),
        "kabutan_member": True,
        "ifis_rank_at_discovery": int(c.get("ifis_rank") or 999),
        "ifis_access_change_at_discovery_pct": f(c.get("ifis_access_change_pct")),
        "reference_high": first_high,
        "max_gain_from_discovery": max(0.0, pct(first_high, discovery_price)),
        "deepest_pullback_from_peak": 0.0,
        "qualifying_pullback_seen": False,
        "status": STATUS_WAIT,
        "terminal": False,
        "history": [],
    }


def decide(state: dict[str, Any], row: dict[str, Any], prev: dict[str, Any] | None) -> Decision:
    close = f(row.get("close"))
    high = f(row.get("high"), close)
    volume = f(row.get("volume"))
    rsi = f(row.get("rsi14"), 50.0)
    ma25 = f(row.get("ma25"), close)
    overheat = f(row.get("overheat_score"))
    material_valid = bool(row.get("material_valid", True))
    theme_active = bool(row.get("theme_active", True))

    discovery = f(state["discovery_price"])
    prior_peak = max(f(state.get("reference_high"), discovery), high)
    pullback = max(0.0, 1.0 - close / prior_peak) if prior_peak else 0.0
    gain = pct(close, discovery)
    cl = close_location(row)

    # 1) Hard invalidation first.
    if not material_valid:
        return Decision(STATUS_INVALID, ["material/catalyst invalidated"])
    if not theme_active and gain < 0:
        return Decision(STATUS_INVALID, ["theme weakened while price is below discovery price"])
    if close < discovery * (1.0 - INVALID_MAX_DRAWDOWN):
        return Decision(STATUS_INVALID, ["price fell 15%+ below discovery price"])
    if ma25 > 0 and close < ma25 * MA25_BREAK:
        return Decision(STATUS_INVALID, ["close is materially below rising-trend reference MA25"])

    # 2) If a proper pullback has already formed, look for the first re-acceleration.
    if bool(state.get("qualifying_pullback_seen")):
        if prev:
            prev_close = f(prev.get("close"), close)
            prev_high = f(prev.get("high"), prev_close)
            prev_vol = f(prev.get("volume"), volume)
            rebound = close > prev_close and close >= prev_high
            volume_reaccel = volume >= prev_vol * 1.05 if prev_vol > 0 else True
            sane = overheat <= SIGNAL_MAX_OVERHEAT and SIGNAL_RSI_MIN <= rsi <= SIGNAL_RSI_MAX
            if rebound and volume_reaccel and sane and cl >= 0.60:
                return Decision(
                    STATUS_SIGNAL,
                    ["first pullback already formed", "rebound confirmed above prior high", "volume re-accelerated", "overheat remains acceptable"],
                )
        return Decision(STATUS_FORMING, ["proper first pullback has formed; waiting for rebound confirmation"])

    # 3) Detect a normal first pullback. Volume should contract versus discovery-day volume if available.
    discovery_vol = f(state.get("discovery_volume"))
    volume_contract = True if discovery_vol <= 0 else volume <= discovery_vol * 0.85
    if PULLBACK_MIN <= pullback <= PULLBACK_MAX and volume_contract and close >= ma25:
        return Decision(
            STATUS_FORMING,
            [f"pullback depth {pullback:.1%} is in prototype range", "volume contracted", "price remains above MA25"],
        )

    # 4) Re-evaluate whether it became buyable without a price pullback (time correction).
    buyable = (
        overheat <= BUY_MAX_OVERHEAT
        and rsi <= BUY_MAX_RSI
        and close >= ma25
        and cl >= 0.55
        and gain <= 0.07
    )
    if buyable:
        return Decision(
            STATUS_BUY_NOW,
            ["overheat cooled to low zone", "RSI is not overheated", "trend remains intact", "price has not run too far from discovery"],
        )

    # 5) Explicitly stop chasing if no first pullback arrived and price ran away.
    if gain >= RUNAWAY_GAIN and pullback < PULLBACK_MIN and overheat >= RUNAWAY_MIN_OVERHEAT:
        return Decision(
            STATUS_RAN_AWAY,
            ["price rose 10%+ from discovery", "no qualifying 3% pullback occurred", "overheat increased; chasing is prohibited"],
        )

    return Decision(STATUS_WAIT, ["still not buyable and no valid first pullback yet"])


def update_one(state: dict[str, Any], row: dict[str, Any], as_of: str) -> dict[str, Any]:
    if state.get("terminal"):
        return state

    history = state.setdefault("history", [])
    prev = history[-1] if history else None
    if not history:
        state["discovery_volume"] = f(row.get("volume"))

    high = f(row.get("high"), f(row.get("close")))
    close = f(row.get("close"))
    prior_peak = max(f(state.get("reference_high"), state["discovery_price"]), high)
    pullback = max(0.0, 1.0 - close / prior_peak) if prior_peak else 0.0

    decision = decide(state, row, prev)

    state["reference_high"] = prior_peak
    state["max_gain_from_discovery"] = max(
        f(state.get("max_gain_from_discovery")), pct(prior_peak, f(state["discovery_price"]))
    )
    state["deepest_pullback_from_peak"] = max(f(state.get("deepest_pullback_from_peak")), pullback)

    if decision.status == STATUS_FORMING and PULLBACK_MIN <= pullback <= PULLBACK_MAX:
        state["qualifying_pullback_seen"] = True

    snapshot = {
        "date": as_of,
        "open": f(row.get("open"), close),
        "high": high,
        "low": f(row.get("low"), close),
        "close": close,
        "volume": f(row.get("volume")),
        "rsi14": f(row.get("rsi14"), 50.0),
        "ma25": f(row.get("ma25"), close),
        "atr14": f(row.get("atr14")),
        "overheat_score": f(row.get("overheat_score")),
        "ifis_rank": row.get("ifis_rank"),
        "ifis_access_change_pct": row.get("ifis_access_change_pct"),
        "minkabu_theme_rank": row.get("minkabu_theme_rank"),
        "material_valid": bool(row.get("material_valid", True)),
        "theme_active": bool(row.get("theme_active", True)),
        "pullback_from_reference_high_pct": round(pullback * 100.0, 4),
        "gain_from_discovery_pct": round(pct(close, f(state["discovery_price"])) * 100.0, 4),
        "decision": decision.status,
        "reasons": decision.reasons,
    }
    history.append(snapshot)
    state["status"] = decision.status
    state["terminal"] = decision.status in TERMINAL
    state["last_updated"] = as_of
    state["last_reasons"] = decision.reasons
    return state


def process(payload: dict[str, Any], state_payload: dict[str, Any]) -> dict[str, Any]:
    as_of = str(payload["as_of"])
    states = {str(x["ticker"]): x for x in state_payload.get("watchlist", [])}

    for item in payload.get("candidates", []):
        ticker = str(item["ticker"])
        technical = dict(item.get("technical") or {})
        # The candidate may be newly discovered or an existing watch item.
        if ticker not in states:
            discovery = dict(item.get("discovery") or {})
            discovery.setdefault("ticker", ticker)
            discovery.setdefault("name", item.get("name", ticker))
            states[ticker] = new_state(discovery, as_of, technical)
        states[ticker] = update_one(states[ticker], technical, as_of)

    watchlist = sorted(states.values(), key=lambda x: (bool(x.get("terminal")), x.get("status", ""), x["ticker"]))
    return {"as_of": as_of, "watchlist": watchlist}


def self_test() -> None:
    state: dict[str, Any] = {"watchlist": []}
    days = [
        {
            "as_of": "2026-09-01",
            "candidates": [{
                "ticker": "9999.T", "name": "Prototype Co",
                "discovery": {
                    "ticker": "9999.T", "name": "Prototype Co", "discovery_price": 1000,
                    "minkabu_relevance": 90, "kabutan_member": True, "ifis_rank": 8,
                    "theme": "AI", "catalyst_reason": "sample catalyst"
                },
                "technical": {"open": 1030, "high": 1080, "low": 1020, "close": 1070, "volume": 1000000, "rsi14": 74, "ma25": 980, "overheat_score": 46}
            }]
        },
        {
            "as_of": "2026-09-02",
            "candidates": [{"ticker": "9999.T", "technical": {"open": 1050, "high": 1060, "low": 1015, "close": 1025, "volume": 650000, "rsi14": 61, "ma25": 985, "overheat_score": 22}}]
        },
        {
            "as_of": "2026-09-03",
            "candidates": [{"ticker": "9999.T", "technical": {"open": 1030, "high": 1068, "low": 1028, "close": 1065, "volume": 720000, "rsi14": 64, "ma25": 990, "overheat_score": 25}}]
        },
    ]
    for p in days:
        state = process(p, state)
    item = state["watchlist"][0]
    assert item["status"] == STATUS_SIGNAL, item
    assert item["terminal"] is True
    print("self-test: PASS")


def main() -> None:
    ap = argparse.ArgumentParser(description="Research prototype: persistent first-pullback watch state machine")
    ap.add_argument("--input", type=Path, help="daily snapshot JSON")
    ap.add_argument("--state", type=Path, default=Path("research/state/first_pullback_watch.json"))
    ap.add_argument("--output", type=Path, help="optional separate output JSON; defaults to --state")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.input:
        ap.error("--input is required unless --self-test is used")

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    state_payload = {"watchlist": []}
    if args.state.exists():
        state_payload = json.loads(args.state.read_text(encoding="utf-8"))

    result = process(payload, state_payload)
    target = args.output or args.state
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"as_of": result["as_of"], "count": len(result["watchlist"]), "output": str(target)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
