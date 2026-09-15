from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from swing_data.weekly_consensus import COLUMNS, collect_shard, merge_shards

JST = ZoneInfo("Asia/Tokyo")


def snapshot(code):
    return {
        "provider_price": 100.0,
        "provider_price_date": "2026-09-11",
        "target_low": 90.0,
        "target_mean": 120.0,
        "target_high": 150.0,
        "recommendation_key": "buy",
        "recommendation_mean": 2.0,
        "analyst_count": 5.0,
        "current_year": {"end_date": "2027-03-31", "analyst_count": 5, "eps": 10.0, "eps_30d": 9.0},
        "next_year": {"end_date": "2028-03-31", "analyst_count": 4, "eps": 12.0, "eps_30d": 11.0},
        "source_url": f"https://finance.yahoo.com/quote/{code}.T/analysis/",
    }


def write_universe(path):
    pd.DataFrame([
        {"stock_code": "4506", "company_name": "住友ファーマ", "market": "プライム", "sector17": "医薬品", "sector33": "医薬品"},
        {"stock_code": "7203", "company_name": "トヨタ自動車", "market": "プライム", "sector17": "輸送用機器", "sector33": "輸送用機器"},
    ]).to_csv(path, index=False)


def test_sharded_collection_and_week_over_week_merge(tmp_path):
    universe = tmp_path / "universe.csv"
    write_universe(universe)
    shards = tmp_path / "shards"
    shards.mkdir()
    now = datetime(2026, 9, 12, 10, 0, tzinfo=JST)
    for shard in range(2):
        collect_shard(
            universe, shards / f"shard-{shard}.csv",
            shard_index=shard, shard_count=2, max_workers=1,
            fetcher=snapshot, now=now,
        )

    target = tmp_path / "public"
    history = target / "weekly_consensus"
    history.mkdir(parents=True)
    old = pd.concat([
        pd.read_csv(shards / "shard-0.csv", dtype={"stock_code": str}),
        pd.read_csv(shards / "shard-1.csv", dtype={"stock_code": str}),
    ], ignore_index=True)
    old["snapshot_date"] = "2026-09-05"
    old["target_mean"] = 100.0
    old["current_year_eps"] = 8.0
    old["next_year_eps"] = 10.0
    old[COLUMNS].to_csv(history / "2026-09-05.csv", index=False)

    result = merge_shards(universe, shards, target, now=now)
    assert result["status"] == "success"
    assert result["universe_count"] == 2
    assert result["yahoo_success_count"] == 2
    assert result["insufficient_source_count"] == 2

    latest = pd.read_csv(target / "weekly_consensus_latest.csv", dtype={"stock_code": str})
    assert set(latest["stock_code"]) == {"4506", "7203"}
    assert latest["target_mean_wow_pct"].round(2).tolist() == [20.0, 20.0]
    assert latest["current_year_eps_wow_pct"].round(2).tolist() == [25.0, 25.0]
    assert latest["next_year_eps_wow_pct"].round(2).tolist() == [20.0, 20.0]
    assert (latest["data_quality"] == "complete_targets").all()
    assert (latest["composite_quality"] == "insufficient_sources").all()
    assert latest["weak_reference_price"].isna().all()
    assert latest["normal_reference_price"].isna().all()
    assert latest["strong_reference_price"].isna().all()
