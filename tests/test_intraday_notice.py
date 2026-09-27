import json

from swing_data import intraday_notice


def test_publish_frozen_pick_and_deduplicate(tmp_path, monkeypatch):
    path = tmp_path / "picks.json"
    path.write_text(json.dumps({"trade_date": "2026-09-28", "status": "PROVISIONAL_PICKS",
                                "recommendations": [{"ticker": "9984.T", "shares": 100,
                                                     "buy_time_jst": "09:30",
                                                     "sell_time_jst": "15:20"}]}))
    monkeypatch.setenv("GITHUB_REPOSITORY", "noumi0713/stock-analysis-tool")
    calls = []

    def fake_api(method, endpoint, payload=None):
        calls.append((method, endpoint, payload))
        if method == "GET":
            return [] if len(calls) == 1 else [{"title": "Intraday forecast 2026-09-28"}]
        return {"number": 1}

    monkeypatch.setattr(intraday_notice, "api", fake_api)
    assert intraday_notice.publish("2026-09-28", path) == "POSTED_READY"
    assert "9984.T 100株" in calls[1][2]["body"]
    assert calls[1][2]["assignees"] == ["noumi0713"]
    assert intraday_notice.publish("2026-09-28", path) == "ALREADY_POSTED"
    assert len(calls) == 3


def test_missing_forecast_and_holiday(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "noumi0713/stock-analysis-tool")
    calls = []

    def fake_api(method, endpoint, payload=None):
        calls.append((method, payload))
        return [] if method == "GET" else {"number": 2}

    monkeypatch.setattr(intraday_notice, "api", fake_api)
    assert intraday_notice.publish("2026-09-28", None) == "POSTED_MISSING"
    assert "保存済みの予想を確認できません" in calls[1][1]["body"]
    assert intraday_notice.publish("2026-09-27", None) == "SKIP_NON_SESSION"
    assert len(calls) == 2
