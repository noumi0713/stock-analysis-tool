import hashlib
import json

from swing_data import intraday_notice


def test_publish_frozen_pick_and_deduplicate(tmp_path, monkeypatch):
    path = tmp_path / "picks.json"
    snapshot = tmp_path / "morning.json"
    snapshot.write_text(json.dumps({"trading_date": "2026-09-28", "status": "READY",
                                    "cutoff_jst": "2026-09-28T08:15:00+09:00",
                                    "completed_at_jst": "2026-09-28T08:06:00+09:00"}))
    path.write_text(json.dumps({"trade_date": "2026-09-28", "status": "PROVISIONAL_PICKS",
                                "generated_at_jst": "2026-09-28T08:10:00+09:00",
                                "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
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
    assert intraday_notice.publish("2026-09-28", path, snapshot) == "POSTED_READY"
    assert "9984.T 100株" in calls[1][2]["body"]
    assert calls[1][2]["assignees"] == ["noumi0713"]
    assert intraday_notice.publish("2026-09-28", path, snapshot) == "ALREADY_POSTED"
    assert len(calls) == 3


def test_incomplete_snapshot_cannot_be_notified_as_trade(tmp_path, monkeypatch):
    snapshot = tmp_path / "morning.json"
    snapshot.write_text(json.dumps({"trading_date": "2026-09-28", "status": "INCOMPLETE",
                                    "cutoff_jst": "2026-09-28T08:15:00+09:00",
                                    "completed_at_jst": "2026-09-28T08:06:00+09:00"}))
    picks = tmp_path / "picks.json"
    picks.write_text(json.dumps({"trade_date": "2026-09-28", "status": "NO_TRADE",
                                 "generated_at_jst": "2026-09-28T08:10:00+09:00",
                                 "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                                 "recommendations": []}))
    monkeypatch.setenv("GITHUB_REPOSITORY", "noumi0713/stock-analysis-tool")
    posted = []
    monkeypatch.setattr(intraday_notice, "api", lambda method, endpoint, payload=None:
                        [] if method == "GET" else posted.append(payload))
    assert intraday_notice.publish("2026-09-28", picks, snapshot) == "POSTED_MISSING"
    assert "保存済みの予想を確認できません" in posted[0]["body"]


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
