from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Research-only engine. This preserves the existing prototype state logic while
# replacing the retired Minkabu input with the completed 124-theme relevance master.

STATUS_BUY_NOW = "BUY_NOW"
STATUS_WAIT = "WAIT_FIRST_PULLBACK"
STATUS_FORMING = "PULLBACK_FORMING"
STATUS_SIGNAL = "FIRST_PULLBACK_SIGNAL"
STATUS_RAN_AWAY = "EXCLUDE_RAN_AWAY"
STATUS_INVALID = "INVALIDATED"
TERMINAL = {STATUS_SIGNAL, STATUS_RAN_AWAY, STATUS_INVALID}

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


def pct(a: float, b: float) -> float:
    return a / b - 1.0 if b else 0.0


def close_location(row: dict[str, Any]) -> float:
    hi, lo, close = f(row.get("high")), f(row.get("low")), f(row.get("close"))
    return (close - lo) / (hi - lo) if hi > lo else 0.5


@dataclass
class Decision:
    status: str
    reasons: list[str]


def validate_discovery(c: dict[str, Any]) -> None:
    required = ("ticker", "name", "discovery_price", "theme_relevance_score", "ifis_rank", "best_theme")
    missing = [k for k in required if c.get(k) in (None, "")]
    if missing:
        raise ValueError(f"discovery missing fields: {', '.join(missing)}")


def new_state(c: dict[str, Any], discovery_date: str, discovery_bar: dict[str, Any]) -> dict[str, Any]:
    validate_discovery(c)
    discovery_price = f(c["discovery_price"])
    first_high = max(discovery_price, f(discovery_bar.get("high"), discovery_price))
    return {
        "ticker": str(c["ticker"]),
        "stock_code": str(c.get("stock_code") or str(c["ticker"]).replace(".T", "")),
        "name": str(c["name"]),
        "snapshot_at": c.get("snapshot_at"),
        "discovered_date": discovery_date,
        "discovery_price": discovery_price,
        "discovery_volume": f(discovery_bar.get("volume")),
        "theme": c.get("best_theme"),
        "theme_relevance_score": f(c.get("theme_relevance_score")),
        "theme_relevance_band": c.get("theme_relevance_band"),
        "theme_relevance_confidence": c.get("theme_relevance_confidence"),
        "candidate_class": c.get("candidate_class"),
        "ifis_rank_at_discovery": int(c.get("ifis_rank") or 999999),
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

    if not material_valid:
        return Decision(STATUS_INVALID, ["material/catalyst invalidated"])
    if not theme_active and gain < 0:
        return Decision(STATUS_INVALID, ["theme weakened while price is below discovery price"])
    if close < discovery * (1.0 - INVALID_MAX_DRAWDOWN):
        return Decision(STATUS_INVALID, ["price fell 15%+ below discovery price"])
    if ma25 > 0 and close < ma25 * MA25_BREAK:
        return Decision(STATUS_INVALID, ["close is materially below MA25"])

    if bool(state.get("qualifying_pullback_seen")):
        if prev:
            prev_close = f(prev.get("close"), close)
            prev_high = f(prev.get("high"), prev_close)
            prev_vol = f(prev.get("volume"), volume)
            rebound = close > prev_close and close >= prev_high
            volume_reaccel = volume >= prev_vol * 1.05 if prev_vol > 0 else True
            sane = overheat <= SIGNAL_MAX_OVERHEAT and SIGNAL_RSI_MIN <= rsi <= SIGNAL_RSI_MAX
            if rebound and volume_reaccel and sane and cl >= 0.60:
                return Decision(STATUS_SIGNAL, [
                    "first pullback already formed",
                    "rebound confirmed above prior high",
                    "volume re-accelerated",
                    "overheat remains acceptable",
                ])
        return Decision(STATUS_FORMING, ["proper first pullback has formed; waiting for rebound confirmation"])

    discovery_vol = f(state.get("discovery_volume"))
    volume_contract = True if discovery_vol <= 0 else volume <= discovery_vol * 0.85
    if PULLBACK_MIN <= pullback <= PULLBACK_MAX and volume_contract and close >= ma25:
        return Decision(STATUS_FORMING, [
            f"pullback depth {pullback:.1%} is in prototype range",
            "volume contracted",
            "price remains above MA25",
        ])

    buyable = (
        overheat <= BUY_MAX_OVERHEAT
        and rsi <= BUY_MAX_RSI
        and close >= ma25
        and cl >= 0.55
        and gain <= 0.07
    )
    if buyable:
        return Decision(STATUS_BUY_NOW, [
            "overheat cooled to low zone",
            "RSI is not overheated",
            "trend remains intact",
            "price has not run too far from discovery",
        ])

    if gain >= RUNAWAY_GAIN and pullback < PULLBACK_MIN and overheat >= RUNAWAY_MIN_OVERHEAT:
        return Decision(STATUS_RAN_AWAY, [
            "price rose 10%+ from discovery",
            "no qualifying 3% pullback occurred",
            "overheat increased; chasing is prohibited",
        ])

    return Decision(STATUS_WAIT, ["still not buyable and no valid first pullback yet"])


def update_one(state: dict[str, Any], row: dict[str, Any], as_of: str) -> dict[str, Any]:
    if state.get("terminal"):
        return state
    history = state.setdefault("history", [])
    prev = history[-1] if history else None
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
