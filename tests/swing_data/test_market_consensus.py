from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from swing_data.market_consensus import (
    classify, collect, inspect, parse_snapshot, scenario_prices,
)

JST = ZoneInfo("Asia/Tokyo")


def payload():
    return {
        "quoteSummary": {
            "result": [{
                "price": {
                    "symbol": "7203.T",
                    "regularMarketPrice": {"raw": 100},
                    "regularMarketTime": {"raw": 1789369200},
                },
                "financialData": {
                    "currentPrice": {"raw": 100},
                    "targetLowPrice": {"raw": 105},
                    "targetMeanPrice": {"raw": 120},
                    "targetHighPrice": {"raw": 140},
                    "recommendationKey": "buy",
                    "recommendationMean": {"raw": 2.0},
                    "numberOfAnalystOpinions": {"raw": 8},
                },
                "earningsTrend": {
                    "trend": [
                        {
                            "period": "0y", "endDate": "2027-03-31",
                            "earningsEstimate": {"numberOfAnalysts": {"raw": 8}},
                            "epsTrend": {"current": {"raw": 11}, "30daysAgo": {"raw": 10}},
                        },
                        {
                            "period": "+1y", "endDate": "2028-03-31",
                            "earningsEstimate": {"numberOfAnalysts": {"raw": 7}},
                            "epsTrend": {"current": {"raw": 13}, "30daysAgo": {"raw": 12}},
                        },
                    ]
                },
            }]
        }
    }


def snapshot():
    row = parse_snapshot(payload(), "7203")
    # Keep this unit test independent of the literal epoch used above.
    row["provider_price_date"] = "2026-09-14"
    return row


def test_parse_and_bullish_classification():
    row = snapshot()
    local = {"date": "2026-09-14", "close": 100, "atr14": 4, "recent_split": False}
    price, issues, insufficient = inspect(row, local, "2026-09-14")
    assert price == 100 and issues == [] and insufficient == []
    assert classify(row, price, issues, insufficient) == ("強気", 3)
    assert scenario_prices(row, price, 4, True) == (
        105.0, 120.0, 140.0, "market_consensus_low_mean_high"
    )


def test_missing_consensus_is_not_neutral_and_gets_labelled_prices():
    row = snapshot()
    row["target_low"] = row["target_mean"] = row["target_high"] = None
    row["analyst_count"] = 1
    local = {"date": "2026-09-14", "close": 100, "atr14": 4, "recent_split": False}
    price, issues, insufficient = inspect(row, local, "2026-09-14")
    assert not issues and "target_prices_missing" in insufficient
    assert classify(row, price, issues, insufficient) == ("判定不能", None)
    assert scenario_prices(row, price, 4, False) == (
        92, 100, 108, "atr14_plus_minus_2atr; not analyst targets"
    )


def test_invalid_targets_and_recent_split_are_unclassifiable():
    row = snapshot()
    row["target_low"], row["target_mean"], row["target_high"] = 130, 120, 110
    local = {"date": "2026-09-14", "close": 100, "atr14": 4, "recent_split": True}
    price, issues, insufficient = inspect(row, local, "2026-09-14")
    assert "target_price_order_invalid" in issues
    assert "recent_stock_split" in issues
    assert classify(row, price, issues, insufficient)[0] == "判定不能"


def write_inputs(target):
    (target / "stocks").mkdir()
    pd.DataFrame([{
        "date": "2026-09-14", "rank": 1, "stock_code": "7203",
        "stock_name": "トヨタ自動車(株)", "market": "東証PRM", "price": 100,
        "source_updated_at": "2026-09-14T09:00:00+09:00",
        "collected_at": "2026-09-14T09:05:00+09:00",
    }]).to_csv(target / "bbs_ranking_latest.csv", index=False)
    (target / "bbs_ranking_status.json").write_text(
        '{"status":"success","ranking_date":"2026-09-14"}', encoding="utf-8"
    )
    (target / "manifest.json").write_text(
        '{"expected_equity_date":"2026-09-14"}', encoding="utf-8"
    )
    prices = []
    for i in range(20):
        prices.append({
            "date": f"2026-08-{25+i:02d}" if i < 7 else f"2026-09-{i-6:02d}",
            "close": 100, "high": 102, "low": 98,
            "adj_close": 100, "adj_high": 102, "adj_low": 98,
            "stock_splits": 0,
        })
    prices[-1]["date"] = "2026-09-14"
    pd.DataFrame(prices).to_csv(target / "stocks" / "7203.csv", index=False)


def test_collect_writes_latest_history_status_and_deduplicates(tmp_path):
    write_inputs(tmp_path)
    now = datetime(2026, 9, 14, 17, 0, tzinfo=JST)
    result = collect(tmp_path, now=now, fetcher=lambda code: snapshot(), max_workers=1)
    assert result["status"] == "success"
    assert result["target_count"] == 1
    latest = pd.read_csv(tmp_path / "market_consensus_latest.csv")
    assert latest.loc[0, "classification"] == "強気"
    assert latest.loc[0, "bear_price"] == 105
    assert latest.loc[0, "next_year_eps_30d"] == 12
    assert round(latest.loc[0, "next_year_eps_change_pct"], 2) == 8.33
    collect(tmp_path, now=now, fetcher=lambda code: snapshot(), max_workers=1)
    history = pd.read_csv(tmp_path / "market_consensus_history.csv")
    assert len(history) == 1


def test_collect_uses_completed_equity_session_after_midnight(tmp_path):
    write_inputs(tmp_path)
    now = datetime(2026, 9, 15, 0, 30, tzinfo=JST)
    result = collect(tmp_path, now=now, fetcher=lambda code: snapshot(), max_workers=1)
    assert result["status"] == "success"
    assert result["ranking_date"] == "2026-09-14"


def test_collect_refuses_previous_day_ranking(tmp_path):
    (tmp_path / "manifest.json").write_text(
        '{"expected_equity_date":"2026-09-14"}', encoding="utf-8"
    )
    (tmp_path / "bbs_ranking_status.json").write_text(
        '{"status":"success","ranking_date":"2026-09-13"}', encoding="utf-8"
    )
    now = datetime(2026, 9, 14, 17, 0, tzinfo=JST)
    try:
        collect(tmp_path, now=now, fetcher=lambda code: snapshot(), max_workers=1)
        assert False, "must fail"
    except RuntimeError as exc:
        assert "当日の掲示板ランキング取得失敗" in str(exc)
