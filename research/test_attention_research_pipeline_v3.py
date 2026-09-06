from attention_discovery_stage import build_discovery
from material_attention_classifier import build_material_stage, classify_review
from final_attention_entry_classifier import combine


def test_attention_discovery_requires_ifis_and_relevant_theme_membership():
    ifis = [
        {"ifis_rank":"1","stock_code":"603A","company_name":"Alpha","snapshot_at":"2026-09-04T01:20:00+09:00"},
        {"ifis_rank":"2","stock_code":"1234","company_name":"Beta","snapshot_at":"2026-09-04T01:20:00+09:00"},
    ]
    master = [
        {"stock_code":"603A","company_name":"Alpha","theme_name":"再生可能エネルギー","status":"finalized","final_relevance_score":"97","final_band":"主力テーマ","final_confidence":"A"},
        {"stock_code":"1234","company_name":"Beta","theme_name":"クラウド","status":"finalized","final_relevance_score":"39","final_band":"ノイズ候補","final_confidence":"B"},
    ]
    candidates, themes, summary = build_discovery(ifis, master, top_n=30)
    assert [x["stock_code"] for x in candidates] == ["603A"]
    assert candidates[0]["buy_decision"] == "NOT_EVALUATED"
    assert candidates[0]["theme_relevance_score"] == 97
    assert len(themes) == 2
    assert summary["discovered_count"] == 1
    assert summary["stage"] == "IFIS_THEME_DISCOVERY_ONLY"


def test_attention_discovery_does_not_create_second_attention_score():
    ifis = [{"ifis_rank":"1","stock_code":"1111","company_name":"A","snapshot_at":"2026-09-04T01:20:00+09:00","attention_score":"999"}]
    master = [{"stock_code":"1111","company_name":"A","theme_name":"防災","status":"finalized","final_relevance_score":"96","final_band":"主力テーマ","final_confidence":"A"}]
    candidates, _, _ = build_discovery(ifis, master)
    assert "trading_score" not in candidates[0]
    assert "buy_score" not in candidates[0]
    assert "minkabu_attention_type" not in candidates[0]


def test_sub40_theme_noise_is_logged_but_not_discovered():
    ifis = [{"ifis_rank":"1","stock_code":"2222","company_name":"Noise","snapshot_at":"2026-09-04T01:20:00+09:00"}]
    master = [{"stock_code":"2222","company_name":"Noise","theme_name":"AI","status":"finalized","final_relevance_score":"39","final_band":"ノイズ候補","final_confidence":"B"}]
    candidates, themes, summary = build_discovery(ifis, master)
    assert candidates == []
    assert len(themes) == 1 and themes[0]["eligible_by_relevance"] is False
    assert summary["decision_counts"]["NOISE_ONLY"] == 1


def test_material_stage_positive_negative_and_lookahead():
    snap = "2026-09-04T01:20:00+09:00"
    strong, _ = classify_review({"catalyst_gate":"PASS","catalyst_direction":"POSITIVE","catalyst_status":"VALID","catalyst_type":"ORDER_ADOPTION","catalyst_date":"2026-09-03T15:00:00+09:00"}, snap)
    negative, _ = classify_review({"catalyst_gate":"FAIL","catalyst_direction":"NEGATIVE","catalyst_status":"VALID","catalyst_type":"FINANCING","catalyst_date":"2026-09-03T15:00:00+09:00"}, snap)
    lookahead, _ = classify_review({"catalyst_gate":"PASS","catalyst_direction":"POSITIVE","catalyst_status":"VALID","catalyst_type":"ORDER_ADOPTION","catalyst_date":"2026-09-04"}, snap)
    assert strong == "STRONG" and negative == "NEGATIVE" and lookahead == "LOOKAHEAD_REJECT"


def test_material_stage_runs_before_technical_and_preserves_semantic_review():
    discovery = [{"ifis_rank":"1","stock_code":"1111","company_name":"A","snapshot_at":"2026-09-04T01:20:00+09:00","best_theme":"AI","theme_relevance_score":"95","attention_stage_status":"DISCOVERED"}]
    reviews = [{"stock_code":"1111","catalyst_gate":"PASS","catalyst_direction":"POSITIVE","catalyst_status":"VALID","catalyst_type":"PARTNERSHIP_TECH","catalyst_date":"2026-09-02","confidence":"A","catalyst_summary":"specific partnership","source_urls":"https://example.com"}]
    rows, summary = build_material_stage(discovery, reviews)
    assert rows[0]["material_class"] == "STRONG"
    assert rows[0]["material_continuity"] == "STRUCTURAL_OR_SPECIFIC"
    assert summary["stage"] == "MATERIAL_BEFORE_TECHNICAL"


def test_final_three_classes_and_timing_first_ranking():
    discovery = [
        {"ifis_rank":"10","stock_code":"A001","company_name":"A","best_theme":"AI","attention_stage_status":"DISCOVERED","theme_relevance_score":"95"},
        {"ifis_rank":"1","stock_code":"B001","company_name":"B","best_theme":"AI","attention_stage_status":"DISCOVERED","theme_relevance_score":"95"},
        {"ifis_rank":"2","stock_code":"C001","company_name":"C","best_theme":"AI","attention_stage_status":"DISCOVERED","theme_relevance_score":"95"},
    ]
    materials = [
        {"stock_code":"A001","review_status":"REVIEWED","material_class":"STRONG","material_continuity":"STRUCTURAL_OR_SPECIFIC"},
        {"stock_code":"B001","review_status":"REVIEWED","material_class":"STRONG","material_continuity":"STRUCTURAL_OR_SPECIFIC"},
        {"stock_code":"C001","review_status":"REVIEWED","material_class":"NEGATIVE","material_continuity":"NONE"},
    ]
    technical = {"items":[
        {"stock_code":"A001","technical_status":"BUY_NOW","latest_market_date":"2026-09-04","latest_close":100,"discovery_metrics":{"return_1d_pct":1,"return_5d_pct":3,"ma25_deviation_pct":2,"rsi14":58,"volume_ratio20":1.5},"latest_metrics":{"rsi14":60,"ma25_deviation_pct":3,"return_5d_pct":4,"return_10d_pct":5,"return_20d_pct":8,"volume_ratio20":1.6,"upper_wick_ratio":0.1,"overheat_score":10,"atr14_pct":3,"float_turnover_pct":18}},
        {"stock_code":"B001","technical_status":"WAIT_FIRST_PULLBACK","latest_market_date":"2026-09-04","latest_close":130,"discovery_metrics":{"return_1d_pct":8,"return_5d_pct":15,"ma25_deviation_pct":13,"rsi14":76,"volume_ratio20":5.5},"latest_metrics":{"rsi14":72,"ma25_deviation_pct":12,"return_5d_pct":15,"return_10d_pct":18,"return_20d_pct":22,"volume_ratio20":3,"upper_wick_ratio":0.2,"overheat_score":45,"atr14_pct":4,"float_turnover_pct":35}},
        {"stock_code":"C001","technical_status":"BUY_NOW","latest_market_date":"2026-09-04","latest_close":90,"discovery_metrics":{"return_1d_pct":0,"return_5d_pct":0,"ma25_deviation_pct":0,"rsi14":50,"volume_ratio20":2},"latest_metrics":{"rsi14":50,"ma25_deviation_pct":0,"return_5d_pct":0,"return_10d_pct":0,"return_20d_pct":0,"volume_ratio20":2,"upper_wick_ratio":0.1,"overheat_score":0,"atr14_pct":2,"float_turnover_pct":10}},
    ]}
    rows = combine(discovery, materials, technical)
    by_code = {x["stock_code"]:x for x in rows}
    assert by_code["A001"]["entry_class"] == "BUY_NOW"
    assert by_code["B001"]["entry_class"] == "WAIT_FIRST_PULLBACK"
    assert by_code["B001"]["attention_price_sequence_proxy"] == "PRICE_MOVE_PRECEDED_ATTENTION_OR_LATE"
    assert by_code["C001"]["entry_class"] == "OVERHEAT_SKIP"
    assert rows[0]["stock_code"] == "A001"
    assert set(x["entry_class"] for x in rows) == {"BUY_NOW","WAIT_FIRST_PULLBACK","OVERHEAT_SKIP"}


def test_buy_now_requires_float_turnover_evidence_but_not_minkabu():
    discovery = [{"ifis_rank":"1","stock_code":"D001","company_name":"D","best_theme":"AI","attention_stage_status":"DISCOVERED","theme_relevance_score":"95"}]
    materials = [{"stock_code":"D001","review_status":"REVIEWED","material_class":"STRONG","material_continuity":"STRUCTURAL_OR_SPECIFIC"}]
    technical = {"items":[{"stock_code":"D001","technical_status":"BUY_NOW","discovery_metrics":{"return_1d_pct":1,"return_5d_pct":2,"ma25_deviation_pct":2,"rsi14":58,"volume_ratio20":1.5},"latest_metrics":{"rsi14":60,"ma25_deviation_pct":3,"return_5d_pct":4,"return_10d_pct":5,"return_20d_pct":8,"volume_ratio20":1.6,"upper_wick_ratio":0.1,"overheat_score":10,"atr14_pct":3,"float_turnover_pct":None}}]}
    rows = combine(discovery, materials, technical)
    assert rows[0]["entry_class"] == "WAIT_FIRST_PULLBACK"
    assert "float turnover must be evaluable" in rows[0]["entry_reasons"]
    assert "theme price trend" not in rows[0]["entry_reasons"]
