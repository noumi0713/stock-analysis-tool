from first_pullback_watch_engine_v2 import (
    STATUS_FORMING,
    STATUS_SIGNAL,
    new_state,
    update_one,
)


def discovery():
    return {
        "ticker": "9999.T",
        "stock_code": "9999",
        "name": "Prototype Co",
        "discovery_price": 1000,
        "theme_relevance_score": 92,
        "theme_relevance_band": "主力テーマ",
        "theme_relevance_confidence": "A",
        "candidate_class": "CORE",
        "ifis_rank": 3,
        "best_theme": "AI",
        "snapshot_at": "2026-09-04T01:20:00+09:00",
    }


def test_state_uses_theme_relevance_not_minkabu():
    s = new_state(discovery(), "2026-09-03", {"high": 1010, "volume": 1000})
    assert s["theme_relevance_score"] == 92
    assert "minkabu_relevance" not in s


def test_pullback_then_reacceleration_signal():
    s = new_state(discovery(), "2026-09-03", {"high": 1080, "volume": 1000})
    s = update_one(s, {
        "open": 1040, "high": 1050, "low": 1015, "close": 1025, "volume": 700,
        "rsi14": 60, "ma25": 990, "atr14": 35, "overheat_score": 22,
    }, "2026-09-04")
    assert s["status"] == STATUS_FORMING
    s = update_one(s, {
        "open": 1030, "high": 1068, "low": 1028, "close": 1068, "volume": 800,
        "rsi14": 64, "ma25": 995, "atr14": 36, "overheat_score": 25,
    }, "2026-09-07")
    assert s["status"] == STATUS_SIGNAL
