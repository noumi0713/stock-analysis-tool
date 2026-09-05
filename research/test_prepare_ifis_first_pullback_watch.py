from prepare_ifis_first_pullback_watch import build_watch, normalize_ticker


def row(rank, code, cls, score):
    return {
        "ifis_rank": str(rank),
        "stock_code": code,
        "company_name": f"C{code}",
        "snapshot_at": "2026-09-04T01:20:00+09:00",
        "candidate_class": cls,
        "best_theme": "AI",
        "best_relevance_score": str(score),
        "best_band": "主力テーマ" if score >= 80 else "有力関連" if score >= 60 else "補助関連",
        "best_confidence": "A",
        "theme_count": "1",
        "all_themes_sorted": f"AI:{score}",
        "mapped_to_master": "true",
    }


def test_ticker_supports_alphanumeric_tse_codes():
    assert normalize_ticker("603A") == "603A.T"
    assert normalize_ticker("6654") == "6654.T"


def test_core_and_strong_become_primary_support_is_reference():
    payload = build_watch([
        row(3, "5039", "CORE", 97),
        row(2, "7066", "STRONG", 76),
        row(9, "7864", "SUPPORT", 51),
        row(11, "9999", "NOISE", 20),
    ])
    assert [x["stock_code"] for x in payload["primary_watch"]] == ["7066", "5039"]
    assert [x["stock_code"] for x in payload["reference_watch"]] == ["7864"]
    assert [x["stock_code"] for x in payload["rejected"]] == ["9999"]
    assert payload["excluded_inputs"] == ["Minkabu"]


def test_ifis_rank_is_preserved_as_primary_order():
    payload = build_watch([
        row(8, "8746", "CORE", 90),
        row(1, "6654", "CORE", 93),
        row(6, "603A", "CORE", 97),
    ])
    assert [x["ifis_rank"] for x in payload["primary_watch"]] == [1, 6, 8]
