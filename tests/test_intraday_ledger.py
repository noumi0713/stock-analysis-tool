from datetime import datetime
import hashlib
import json

import pytest

from swing_data import intraday_ledger
from test_intraday_execution import DATE, order, sample_db


NOW = datetime(2026, 9, 28, 16, 0, tzinfo=intraday_ledger.JST)


def inputs(day=DATE, orders=None, status="PROVISIONAL_PICKS"):
    snapshot = {"trading_date": day, "cutoff_jst": f"{day}T08:15:00+09:00",
                "completed_at_jst": f"{day}T08:06:00+09:00", "status": "READY"}
    snapshot_bytes = json.dumps(snapshot).encode()
    picks = {"trade_date": day, "generated_at_jst": f"{day}T08:10:00+09:00",
             "status": status, "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
             "recommendations": [order()] if orders is None and status != "NO_TRADE" else (orders or [])}
    return json.dumps(picks).encode(), snapshot_bytes


def test_append_estimated_result_and_prevent_replacement(tmp_path):
    db = tmp_path / "bars.sqlite"
    sample_db(db)
    archive = tmp_path / "ledger"
    picks, snapshot = inputs()
    path, summary = intraday_ledger.append_day(archive, picks, snapshot, db, now=NOW)
    assert path.name == "rev-0001.json"
    assert summary["recorded_sessions"] == 1
    assert summary["estimated_trade_days"] == 1
    assert summary["performance"]["proxy_total_pnl_jpy"] > 0
    assert summary["performance"]["profit_factor"] is None
    assert json.loads(path.read_text())["picks_sha256"] == hashlib.sha256(picks).hexdigest()
    with pytest.raises(FileExistsError):
        intraday_ledger.append_day(archive, picks, snapshot, db, now=NOW)


def test_unsettled_revision_preserves_old_result(tmp_path):
    db = tmp_path / "bars.sqlite"
    sample_db(db, missing=("09:30", "09:35", "09:40", "09:45"))
    archive = tmp_path / "ledger"
    picks, snapshot = inputs()
    first, summary = intraday_ledger.append_day(archive, picks, snapshot, db, now=NOW)
    assert summary["performance"] is None
    assert summary["unsettled_dates"] == [DATE]
    events = json.dumps({"trading_date": DATE, "tickers": {"9984.T": {
        "untradable": [{"start": "09:30", "end": "09:50", "reason": "special_quote"}]
    }}}).encode()
    with pytest.raises(ValueError, match="evidence"):
        intraday_ledger.append_day(archive, picks, snapshot, db,
                                   amend_unsettled=True, now=NOW)
    second, summary = intraday_ledger.append_day(archive, picks, snapshot, db, events,
                                                  amend_unsettled=True, now=NOW)
    assert first.exists() and second.name == "rev-0002.json"
    assert summary["performance"] is not None
    assert json.loads(second.read_text())["previous_record_sha256"] == json.loads(first.read_text())["record_sha256"]
    with pytest.raises(FileExistsError):
        intraday_ledger.append_day(archive, picks, snapshot, db, events,
                                   amend_unsettled=True, now=NOW)


def test_no_trade_and_cutoff_guards(tmp_path):
    db = tmp_path / "bars.sqlite"
    archive = tmp_path / "ledger"
    picks, snapshot = inputs(status="NO_TRADE")
    with pytest.raises(ValueError, match="15:30"):
        intraday_ledger.append_day(archive, picks, snapshot, db,
                                   now=NOW.replace(hour=14))
    path, summary = intraday_ledger.append_day(archive, picks, snapshot, db, now=NOW)
    assert path.exists()
    assert summary["no_trade_days"] == 1
    assert summary["performance"]["proxy_total_pnl_jpy"] == 0
    late = json.loads(picks)
    late["generated_at_jst"] = f"{DATE}T08:16:00+09:00"
    with pytest.raises(ValueError, match="cutoff"):
        intraday_ledger.append_day(tmp_path / "late", json.dumps(late).encode(), snapshot, db, now=NOW)


def test_broken_morning_input_is_not_counted_as_zero_pnl(tmp_path):
    db = tmp_path / "bars.sqlite"
    picks, snapshot_raw = inputs(status="NO_TRADE")
    snapshot = json.loads(snapshot_raw)
    snapshot["status"] = "INCOMPLETE"
    snapshot_raw = json.dumps(snapshot).encode()
    pick = json.loads(picks)
    pick["snapshot_sha256"] = hashlib.sha256(snapshot_raw).hexdigest()
    _, summary = intraday_ledger.append_day(tmp_path / "ledger", json.dumps(pick).encode(),
                                             snapshot_raw, db, now=NOW)
    assert summary["unsettled_dates"] == [DATE]
    assert summary["performance"] is None


def test_missing_session_blocks_continuous_metrics_and_hash_tampering(tmp_path):
    db = tmp_path / "bars.sqlite"
    sample_db(db)
    archive = tmp_path / "ledger"
    picks, snapshot = inputs("2026-09-25", status="NO_TRADE")
    intraday_ledger.append_day(archive, picks, snapshot, db, now=NOW)
    picks, snapshot = inputs("2026-09-29", status="NO_TRADE")
    path, summary = intraday_ledger.append_day(archive, picks, snapshot, db,
                                                now=NOW.replace(day=29))
    assert summary["missing_tse_sessions"] == ["2026-09-28"]
    assert summary["performance"] is None
    record = json.loads(path.read_text())
    record["picks"]["status"] = "PROVISIONAL_PICKS"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="hash mismatch"):
        intraday_ledger.summarize(archive)
