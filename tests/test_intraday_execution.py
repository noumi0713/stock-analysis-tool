import hashlib
import json
import sqlite3

from swing_data import intraday_5m_db, intraday_5m_quality, intraday_execution

DATE = "2026-09-28"


def sample_db(path, tickers=("9984.T",), missing=()):
    with sqlite3.connect(path) as db:
        db.executescript(intraday_5m_db.SCHEMA)
        for ticker in tickers:
            for clock in intraday_5m_quality.EXPECTED_TIMES:
                if clock in missing:
                    continue
                price = 102 if clock == "15:20" else 100
                db.execute("INSERT INTO bars VALUES (?,?,?,?,?,?,?,?,?)",
                           (ticker, DATE, clock, price, price + 1, price - 1,
                            price, 100000, "2026-09-28T16:45:00+09:00"))


def sample_records(orders):
    snapshot = {"trading_date": DATE, "cutoff_jst": f"{DATE}T08:15:00+09:00"}
    snapshot_bytes = json.dumps(snapshot).encode()
    picks = {"trade_date": DATE, "status": "PROVISIONAL_PICKS",
             "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
             "recommendations": orders}
    picks_bytes = json.dumps(picks).encode()
    return picks, picks_bytes, snapshot, snapshot_bytes


def order(ticker="9984.T", shares=100, buy="09:30", sell="15:20"):
    return {"ticker": ticker, "shares": shares,
            "buy_time_jst": buy, "sell_time_jst": sell}


def run(db, orders, events=None):
    picks, raw_picks, snapshot, raw_snapshot = sample_records(orders)
    return intraday_execution.settle(picks, raw_picks, snapshot, raw_snapshot, db, events)


def test_regular_bars_create_estimates_with_costs_and_bounds(tmp_path):
    db = tmp_path / "bars.sqlite"
    sample_db(db)
    report = run(db, [order()])
    trade = report["trades"][0]
    assert report["status"] == "ESTIMATED_COMPLETE"
    assert trade["status"] == "ESTIMATED_ROUND_TRIP"
    assert trade["proxy_pnl_jpy"] > 0
    assert trade["pessimistic_bar_pnl_jpy"] < trade["proxy_pnl_jpy"]
    assert report["proxy_final_assets_jpy"] == 2_000_000 + trade["proxy_pnl_jpy"]


def test_absent_bar_without_event_does_not_invent_fill(tmp_path):
    db = tmp_path / "bars.sqlite"
    sample_db(db, missing=("09:30",))
    report = run(db, [order()])
    assert report["status"] == "UNSETTLED"
    assert report["proxy_total_pnl_jpy"] is None
    assert report["trades"][0]["buy_resolution"]["status"] == "UNVERIFIABLE_MISSING_BAR"


def test_documented_special_quote_defers_to_first_print(tmp_path):
    db = tmp_path / "bars.sqlite"
    sample_db(db, missing=("09:30", "09:35", "09:40", "09:45"))
    events = {"trading_date": DATE, "tickers": {"9984.T": {
        "untradable": [{"start": "09:30", "end": "09:50", "reason": "special_quote"}]
    }}}
    report = run(db, [order()], events)
    assert report["status"] == "ESTIMATED_COMPLETE"
    assert report["trades"][0]["buy_resolution"]["bar_time"] == "09:50"
    assert report["trades"][0]["buy_resolution"]["status"] == "ESTIMATED_DELAYED"


def test_limit_queue_and_event_conflict_are_unverifiable(tmp_path):
    db = tmp_path / "bars.sqlite"
    sample_db(db)
    events = {"trading_date": DATE, "tickers": {"9984.T": {"limit_up": 100}}}
    assert run(db, [order()], events)["trades"][0]["buy_resolution"]["status"] == "UNVERIFIABLE_LIMIT_QUEUE"
    events["tickers"]["9984.T"] = {"untradable": [
        {"start": "09:30", "end": "09:50", "reason": "halt"}
    ]}
    assert run(db, [order()], events)["trades"][0]["buy_resolution"]["status"] == "MARKET_EVENT_CONFLICT"


def test_missing_exit_leaves_open_exposure_without_final_assets(tmp_path):
    db = tmp_path / "bars.sqlite"
    sample_db(db, missing=("15:20",))
    report = run(db, [order()])
    assert report["status"] == "UNSETTLED"
    assert report["trades"][0]["status"] == "OPEN_UNSETTLED"
    assert "proxy_final_assets_jpy" not in report


def test_cash_is_not_reused_after_earlier_sale(tmp_path):
    db = tmp_path / "bars.sqlite"
    sample_db(db, tickers=("9984.T", "7936.T", "8306.T"))
    report = run(db, [order("9984.T", 7000, "09:05", "11:25"),
                      order("7936.T", 7000, "12:30", "14:30"),
                      order("8306.T", 7000, "13:30", "15:20")])
    assert report["status"] == "ESTIMATED_COMPLETE"
    assert [t["status"] for t in report["trades"]] == [
        "ESTIMATED_ROUND_TRIP", "ESTIMATED_ROUND_TRIP", "REJECTED_CASH_LIMIT"
    ]


def test_snapshot_identity_required_and_no_trade_is_zero(tmp_path):
    db = tmp_path / "bars.sqlite"
    sample_db(db)
    picks, raw_picks, snapshot, raw_snapshot = sample_records([order()])
    picks["snapshot_sha256"] = "bad"
    report = intraday_execution.settle(picks, raw_picks, snapshot, raw_snapshot, db)
    assert report["status"] == "UNSETTLED"
    picks["snapshot_sha256"] = hashlib.sha256(raw_snapshot).hexdigest()
    picks["status"] = "NO_TRADE"
    picks["recommendations"] = []
    report = intraday_execution.settle(picks, json.dumps(picks).encode(), snapshot, raw_snapshot, db)
    assert report["status"] == "NO_TRADE"
    assert report["proxy_total_pnl_jpy"] == 0
