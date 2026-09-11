from datetime import datetime
import json
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from swing_data.bbs_ranking import HISTORY_COLUMNS, build_trends, collect, parse_ranking_page


def page(items, *, page=1, total=2):
    state = {"mainRankingList":{"results":items,"paging":{"page":page,"totalPage":1,"totalSize":total}}}
    return f'<script>window.__PRELOADED_STATE__ = {json.dumps(state, ensure_ascii=False)}</script>'


def item(rank, code, name, market, price, updated="2026/09/11 07:12"):
    return {"rank":str(rank), "stockCode":code, "stockName":name, "marketName":market,
            "savePrice":price, "rankingResult":{"bbsContents":{"updateDateTime":updated}}}


def test_parse_preloaded_ranking_state():
    rows, paging = parse_ranking_page(page([
        item(1, "285A", "キオクシア", "東証PRM", "54,030"),
        item(2, "9984", "ソフトバンクＧ", "東証PRM", "6,540"),
    ]))
    assert paging["totalSize"] == 2
    assert rows[0]["stock_code"] == "285A" and rows[0]["price"] == 54030
    assert rows[1]["source_updated_at"] == "2026-09-11T07:12:00+09:00"


def test_rank_changes_new_entries_exits_and_streaks():
    rows = [
        ["2026-09-09",3,"1111","A","東証PRM",100,"2026-09-09T07:12:00+09:00","x"],
        ["2026-09-09",1,"2222","B","東証PRM",200,"2026-09-09T07:12:00+09:00","x"],
        ["2026-09-10",2,"1111","A","東証PRM",101,"2026-09-10T07:12:00+09:00","x"],
        ["2026-09-10",1,"3333","C","東証GRT",300,"2026-09-10T07:12:00+09:00","x"],
        ["2026-09-11",1,"1111","A","東証PRM",102,"2026-09-11T07:12:00+09:00","x"],
        ["2026-09-11",3,"3333","C","東証GRT",290,"2026-09-11T07:12:00+09:00","x"],
    ]
    trends, exits = build_trends(pd.DataFrame(rows, columns=HISTORY_COLUMNS))
    a = trends.set_index("stock_code").loc["1111"]
    c = trends.set_index("stock_code").loc["3333"]
    assert a.previous_rank == 2 and a.rank_change == 1 and a.consecutive_days == 3 and a.best_rank == 1
    assert c.previous_rank == 1 and c.rank_change == -2 and c.consecutive_days == 2
    assert exits[(exits.date == "2026-09-10") & (exits.stock_code == "2222")].shape[0] == 1


def test_missing_day_is_not_treated_as_previous_day_or_continuous_streak():
    rows = [
        ["2026-09-09",1,"1111","A","東証PRM",100,"x","x"],
        ["2026-09-11",2,"1111","A","東証PRM",101,"x","x"],
    ]
    trends, exits = build_trends(pd.DataFrame(rows, columns=HISTORY_COLUMNS))
    latest = trends.iloc[0]
    assert pd.isna(latest.previous_rank) and pd.isna(latest.rank_change)
    assert latest.consecutive_days == 1 and "2026-09-10:未取得" in latest.rank_history_3d
    assert exits.empty


def test_failed_current_fetch_removes_latest_but_preserves_history(tmp_path):
    pd.DataFrame([["2026-09-10",1,"1111","A","東証PRM",100,"x","x"]], columns=HISTORY_COLUMNS).to_csv(
        tmp_path / "bbs_ranking_history.csv", index=False)
    (tmp_path / "bbs_ranking_latest.csv").write_text("old", encoding="utf-8")
    def fail(**kwargs):
        raise RuntimeError("provider unavailable")
    with pytest.raises(RuntimeError):
        collect(tmp_path, now=datetime(2026,9,11,7,20,tzinfo=ZoneInfo("Asia/Tokyo")), fetcher=fail)
    assert not (tmp_path / "bbs_ranking_latest.csv").exists()
    assert (tmp_path / "bbs_ranking_history.csv").exists()
    status = json.loads((tmp_path / "bbs_ranking_status.json").read_text())
    assert status["status"] == "failed" and status["used_previous_day"] is False
