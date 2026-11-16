from datetime import datetime
import json
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from swing_data.bollinger_screen import build_watchlist, evaluate
from swing_data.collector import atomic_json, session_window

NOW = datetime(2026, 11, 16, 9, tzinfo=ZoneInfo("Asia/Tokyo"))
SESSIONS = session_window(NOW)


def history(ticker="7203.T", close=None):
    close = np.asarray(close if close is not None else [90., 110.] * 60)
    f = pd.DataFrame({"date": SESSIONS, "ticker": ticker, "open": close,
                      "high": close + 1, "low": close - 1, "close": close, "volume": 1000.})
    for col in ["open", "high", "low", "close"]:
        f["adj_" + col] = f[col]
    return f


def bundle(path, prices=None, scope="tse_domestic_common"):
    path.mkdir(exist_ok=True)
    f = history() if prices is None else prices
    f.to_csv(path / "equities_120d.csv", index=False)
    universe = pd.DataFrame({"ticker": ["7203.T", "9984.T"], "stock_code": ["7203", "9984"],
                             "company_name": ["トヨタ", "ソフトバンクグループ"]})
    universe.to_csv(path / "universe.csv", index=False)
    pd.DataFrame({"stock_code": ["7203"], "theme_name": ["自動車"]}).to_csv(path / "theme_members.csv", index=False)
    atomic_json(path / "manifest.json", {"run_id": "current", "expected_equity_date": SESSIONS[-1],
                "target_count": 2, "ready_count": 1,
                "universe": {"scope": scope, "status": "fetched"}, "stock_status_counts": {"ready": 1, "stale": 1}})
    atomic_json(path / "update_status.json", {"run_id": "current", "state": "partial"})


def test_exact_lower_band_touch_includes_equality_and_distinguishes_close():
    f = history()
    f.loc[119, ["low", "adj_low"]] = 80
    result = evaluate(f, SESSIONS)
    assert result["ma20"] == pytest.approx(100)
    assert result["lower_band"] == pytest.approx(80)
    assert result["close_sigma"] == pytest.approx(1)
    assert result["matched"] and result["new_touch"]
    assert not result["close_below_band"]
    f.loc[119, ["low", "adj_low"]] = 80.00001
    assert not evaluate(f, SESSIONS)["matched"]


def test_population_std_and_close_below_by_hand_calculation():
    f = history(close=[100.] * 119 + [80.])
    result = evaluate(f, SESSIONS)
    assert result["ma20"] == pytest.approx(99)
    assert result["lower_band"] == pytest.approx(99 - 2 * np.sqrt(19))
    assert result["close_sigma"] == pytest.approx(-np.sqrt(19))
    assert result["matched"] and result["close_below_band"] and result["new_touch"]


def test_previous_day_uses_its_own_band_and_flat_series_is_not_a_drop():
    f = history()
    f.loc[[118, 119], ["low", "adj_low"]] = 70
    assert evaluate(f, SESSIONS)["matched"]
    assert not evaluate(f, SESSIONS)["new_touch"]
    result = evaluate(history(close=[100.] * 120), SESSIONS)
    assert not result["matched"] and result["close_sigma"] is None


def test_split_adjusted_comparison_returns_current_yen_units():
    f = history(close=[100.] * 119 + [80.])
    expected = evaluate(f, SESSIONS)
    # Raw historical prices in an older share scale must never enter the rolling band.
    f.loc[:59, ["open", "high", "low", "close"]] *= 2
    for col in ["adj_open", "adj_high", "adj_low", "adj_close"]:
        f[col] *= .5
    result = evaluate(f, SESSIONS)
    for key in ["lower_band", "ma20", "close_sigma", "matched", "close_below_band"]:
        assert result[key] == pytest.approx(expected[key])


@pytest.mark.parametrize("issue", ["missing", "duplicate", "stale", "future", "nan", "ohlc", "zero_volume"])
def test_bad_or_unfinished_history_is_rejected(issue):
    f = history()
    if issue == "missing": f = f.drop(50)
    if issue == "duplicate": f.loc[50, "date"] = f.date.iloc[49]
    if issue == "stale": f = f.iloc[:-1]
    if issue == "future": f.loc[119, "date"] = "2026-11-16"
    if issue == "nan": f.loc[50, "adj_close"] = np.nan
    if issue == "ohlc": f.loc[50, "adj_low"] = 999
    if issue == "zero_volume": f.loc[50, "volume"] = 0
    with pytest.raises(ValueError): evaluate(f, SESSIONS)


def test_bundle_writes_readable_outputs_and_partial_coverage(tmp_path):
    f = history(close=[100.] * 119 + [80.])
    bundle(tmp_path, f)
    status = build_watchlist(tmp_path, now=NOW)
    assert status["status"] == "partial"
    assert status["target_count"] == 2 and status["evaluated_count"] == 1
    assert status["excluded_count"] == 1 and status["matched_count"] == 1
    data = json.loads((tmp_path / "bollinger_minus2_latest.json").read_text())
    assert data["matches"][0]["stock_code"] == "7203"
    assert "トヨタ" in (tmp_path / "bollinger_minus2.md").read_text()
    assert len(pd.read_csv(tmp_path / "bollinger_minus2_latest.csv")) == 1


@pytest.mark.parametrize("failure", ["stale", "different_run", "failed"])
def test_failed_update_removes_old_latest_matches(tmp_path, failure):
    bundle(tmp_path, history(close=[100.] * 119 + [80.]))
    assert build_watchlist(tmp_path, now=NOW)["matched_count"] == 1
    if failure == "stale":
        path = tmp_path / "manifest.json"
        m = json.loads(path.read_text()); m["expected_equity_date"] = "2026-11-02"
        atomic_json(path, m)
    else:
        atomic_json(tmp_path / "update_status.json", {"run_id": "other" if failure == "different_run" else "current",
                                                     "state": "failed" if failure == "failed" else "partial"})
    status = build_watchlist(tmp_path, now=NOW)
    assert status["status"] == "not_ready" and status["matched_count"] == 0
    assert json.loads((tmp_path / "bollinger_minus2_latest.json").read_text())["matches"] == []
    assert pd.read_csv(tmp_path / "bollinger_minus2_latest.csv").empty
    assert "該当銘柄はありません" not in (tmp_path / "bollinger_minus2.md").read_text()


def test_empty_valid_result_and_selected_subset_are_explicit(tmp_path):
    prices = pd.concat([history(), history("9984.T")], ignore_index=True)
    bundle(tmp_path, prices)
    status = build_watchlist(tmp_path, now=NOW)
    assert status["status"] == "success" and status["matched_count"] == 0
    assert "該当銘柄はありません" in (tmp_path / "bollinger_minus2.md").read_text()
    bundle(tmp_path, prices, scope="selected_subset")
    assert build_watchlist(tmp_path, now=NOW)["status"] == "partial"


def test_failed_publish_generates_failure_page(tmp_path):
    from swing_data.publish import publish
    target, source = tmp_path / "public", tmp_path / "runtime"
    bundle(target)
    atomic_json(source / "status.json", {"run_id": "failed-new-run", "state": "failed"})
    publish(source, target)
    assert json.loads((target / "bollinger_minus2_status.json").read_text())["status"] == "not_ready"
