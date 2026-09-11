from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from swing_data.collector import (ROOT, WINDOW, PRICE_COLUMNS, atomic_json, collect, extract_ticker,
    fetch_group, load_themes, normalize, session_window, theme_daily, update_lock)
from swing_data.publish import publish


def history(sessions):
    return pd.DataFrame({"Open":100., "High":103., "Low":99., "Close":102., "Volume":1000.,
                         "Adj Close":51., "Dividends":0., "Stock Splits":0.}, index=pd.to_datetime(sessions))


def test_completed_sessions_holidays_and_intraday():
    before = session_window(datetime(2026, 9, 9, 14, tzinfo=ZoneInfo("Asia/Tokyo")))
    after = session_window(datetime(2026, 9, 9, 18, tzinfo=ZoneInfo("Asia/Tokyo")))
    assert len(before) == len(after) == WINDOW
    assert before[-1] == "2026-09-08"
    assert after[-1] == "2026-09-09"
    holiday = session_window(datetime(2026, 9, 21, 18, tzinfo=ZoneInfo("Asia/Tokyo")))
    assert holiday[-1] == "2026-09-18"
    assert session_window(datetime(2026, 9, 9, 18, tzinfo=ZoneInfo("Asia/Tokyo")), "XNYS")[-1] == "2026-09-08"


def test_exact_window_and_adjustment_do_not_mix():
    dates = session_window(datetime(2026, 9, 9, 18, tzinfo=ZoneInfo("Asia/Tokyo")))
    raw = history(dates + ["2026-09-10"])
    data, status = normalize(raw, "7203.T", dates)
    assert status["status"] == "ready"
    assert len(data) == 120 and data.date.max() == dates[-1]
    assert data.close.iloc[0] == 102 and data.adj_close.iloc[0] == 51
    assert data.adj_open.iloc[0] == 50 and data.volume.iloc[0] == 1000


@pytest.mark.parametrize("issue,expected", [("gap","insufficient_history"),("old","stale"),("bad","invalid_rows"),("duplicate","invalid_rows"),("volume","zero_volume")])
def test_bad_data_is_not_ready(issue, expected):
    dates = session_window(datetime(2026, 9, 9, 18, tzinfo=ZoneInfo("Asia/Tokyo")))
    raw = history(dates)
    if issue == "gap": raw = raw.drop(raw.index[20])
    if issue == "old": raw = raw.iloc[:-1]
    if issue == "bad": raw.loc[raw.index[0], "High"] = 1
    if issue == "duplicate": raw = pd.concat([raw, raw.iloc[:1]])
    if issue == "volume": raw.loc[raw.index[0], "Volume"] = 0
    data, status = normalize(raw, "7203.T", dates)
    assert status["status"] == expected
    assert len(data) <= 120


def test_missing_ticker_retried_not_silently_dropped():
    dates = session_window(datetime(2026, 9, 9, 18, tzinfo=ZoneInfo("Asia/Tokyo")))
    calls = []
    def download(tickers, **kwargs):
        calls.append(tickers)
        return pd.concat({"7203.T":history(dates)}, axis=1) if len(calls) == 1 else pd.DataFrame()
    data, reports = fetch_group(["7203.T", "9984.T"], dates, downloader=download, sleeper=lambda _:None)
    assert reports["7203.T"]["status"] == "ready"
    assert reports["9984.T"]["status"] == "fetch_failed"
    assert calls == [["7203.T","9984.T"],["9984.T"],["9984.T"]]


def test_multiindex_both_yahoo_layouts():
    raw = history(["2026-09-09"])
    result = pd.concat({"7203.T":raw}, axis=1)
    pd.testing.assert_frame_equal(extract_ticker(result, "7203.T"), raw)
    pd.testing.assert_frame_equal(extract_ticker(result.swaplevel(axis=1), "7203.T"), raw)


def test_existing_taxonomy_and_allocations():
    members = pd.read_csv(ROOT / "config/theme_members.csv", dtype=str)
    universe = pd.DataFrame({"stock_code":members.stock_code.unique()})
    loaded, provenance = load_themes(ROOT / "config", universe)
    assert loaded.theme_name.nunique() == 124
    assert loaded.cluster.nunique() == 13
    assert loaded.groupby("stock_code").membership_weight.sum().round(10).eq(1).all()
    assert "final_confidence" in loaded and "relevance_source_url" in loaded
    assert len(provenance["sources"]) == 3


def test_duplicate_membership_does_not_double_count_turnover():
    dates = session_window(datetime(2026, 9, 9, 18, tzinfo=ZoneInfo("Asia/Tokyo")))
    prices, _ = normalize(history(dates), "7203.T", dates)
    members = pd.DataFrame({"stock_code":["7203","7203"], "theme_name":["A","B"], "cluster":["C","C"], "membership_weight":[.5,.5]})
    output = theme_daily(prices, members)
    assert output[output.date == dates[-1]].allocated_turnover_proxy.sum() == 102000
    assert output[output.date == dates[0]].return_pct.isna().all()


def test_failed_update_keeps_previous_data_and_releases_lock(tmp_path):
    target = tmp_path / "public"
    target.mkdir()
    atomic_json(target / "manifest.json", {"run_id":"previous"})
    source = tmp_path / "runtime"
    atomic_json(source / "status.json", {"state":"failed", "run_id":"new", "error":"provider error"})
    publish(source, target)
    assert json.loads((target / "manifest.json").read_text())["run_id"] == "previous"
    assert json.loads((target / "update_status.json").read_text())["state"] == "failed"
    with update_lock(source):
        with pytest.raises(RuntimeError):
            with update_lock(source): pass
    with update_lock(source): pass


def test_collection_to_site_handoff_and_manifest_hashes(tmp_path):
    import hashlib
    import zipfile
    now = datetime(2026, 9, 9, 18, tzinfo=ZoneInfo("Asia/Tokyo"))
    def download(tickers, start, end, **kwargs):
        dates = pd.bdate_range(start, pd.Timestamp(end) - pd.Timedelta(days=1))
        return pd.concat({ticker:history(dates) for ticker in tickers}, axis=1)
    source, target = tmp_path / "runtime", tmp_path / "public"
    def market_fetcher(item, sessions, **kwargs):
        from swing_data.market_sources import normalize_market
        raw = history(sessions)
        frame, report = normalize_market(raw, item, sessions)
        return frame, report, raw
    meta = collect(source, tickers=["7203", "9984"], now=now, downloader=download, market_fetcher=market_fetcher)
    assert meta["ready_count"] == 2 and meta["quality"] == "PARTIAL"
    assert meta["universe"]["scope"] == "selected_subset"
    publish(source, target)
    result = json.loads((target / "stocks.json").read_text())
    assert len(result["stocks"]) == 2
    assert len(pd.read_csv(target / "stocks/7203.csv")) == 120
    index = (target / "index.md").read_text()
    assert "stock_status.csv" in index and "bbs_ranking_status.json" in index
    with zipfile.ZipFile(target / "chatgpt_120d.zip") as archive:
        hashes = json.loads(archive.read("sha256.json"))
        for filename, expected in hashes.items():
            assert hashlib.sha256(archive.read(filename)).hexdigest() == expected
