"""Append point-in-time forecasts and estimated outcomes to a forward-test ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals

from .intraday_execution import CAPITAL_JPY, settle

JST = ZoneInfo("Asia/Tokyo")
TARGET_SESSIONS = 20
DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}\Z")


def encoded(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def day_dir(archive: Path, trade_date: str) -> Path:
    if not DATE_PATTERN.fullmatch(trade_date):
        raise ValueError("Expected YYYY-MM-DD trade date")
    date.fromisoformat(trade_date)
    return archive / "days" / trade_date[:4] / trade_date[5:7] / trade_date


def _load_revisions(folder: Path) -> list[dict]:
    files = sorted(folder.glob("rev-*.json"))
    revisions = []
    previous = None
    for index, path in enumerate(files, 1):
        if path.name != f"rev-{index:04d}.json":
            raise ValueError(f"Nonconsecutive revision: {path}")
        item = json.loads(path.read_bytes())
        checksum = item.pop("record_sha256", None)
        if checksum != digest(encoded(item)):
            raise ValueError(f"Record hash mismatch: {path}")
        if item.get("revision") != index or item.get("previous_record_sha256") != previous:
            raise ValueError(f"Broken revision chain: {path}")
        if item.get("trade_date") != folder.name:
            raise ValueError(f"Date/path mismatch: {path}")
        if (json.loads(item["picks_raw_utf8"]) != item["picks"] or
            digest(item["picks_raw_utf8"].encode("utf-8")) != item["picks_sha256"] or
            json.loads(item["execution_raw_utf8"]) != item["execution"] or
            digest(item["execution_raw_utf8"].encode("utf-8")) != item["execution_sha256"] or
            (item["market_events_raw_utf8"] is not None and
             (json.loads(item["market_events_raw_utf8"]) != item["market_events"] or
              digest(item["market_events_raw_utf8"].encode("utf-8")) != item["market_events_sha256"])) or
            (item["market_events_raw_utf8"] is None and item["market_events"] is not None) or
            item["execution"]["picks_sha256"] != item["picks_sha256"] or
            item["execution"]["snapshot_sha256"] != item["snapshot_sha256"]):
            raise ValueError(f"Input/result hash mismatch: {path}")
        item["record_sha256"] = checksum
        revisions.append(item)
        previous = checksum
    return revisions


def summarize(archive: Path) -> dict:
    latest = []
    seen = set()
    for folder in sorted((archive / "days").glob("*/*/*")):
        if not folder.is_dir():
            continue
        trade_date = folder.name
        if day_dir(archive, trade_date) != folder or trade_date in seen:
            raise ValueError(f"Invalid date directory: {folder}")
        revisions = _load_revisions(folder)
        if not revisions:
            raise ValueError(f"Empty date directory: {folder}")
        seen.add(trade_date)
        latest.append(revisions[-1])
    latest.sort(key=lambda x: x["trade_date"])
    season = latest[:TARGET_SESSIONS]
    missing = []
    if season:
        calendar = xcals.get_calendar("XTKS")
        sessions = calendar.sessions_in_range(season[0]["trade_date"], season[-1]["trade_date"])
        missing = [str(session.date()) for session in sessions if str(session.date()) not in seen]
    unresolved = [x["trade_date"] for x in season if x["execution"]["status"] == "UNSETTLED"]
    rows = []
    for item in season:
        report = item["execution"]
        rows.append({"trade_date": item["trade_date"], "revision": item["revision"],
                     "record_sha256": item["record_sha256"],
                     "forecast_status": item["picks"]["status"],
                     "outcome_status": report["status"],
                     "recommended_count": len(item["picks"].get("recommendations", [])),
                     "estimated_filled_count": sum(t["status"] == "ESTIMATED_ROUND_TRIP"
                                                   for t in report["trades"]),
                     "proxy_pnl_jpy": report["proxy_total_pnl_jpy"]})
    complete = not unresolved and not missing
    performance = None
    if complete:
        pnls = [float(row["proxy_pnl_jpy"]) for row in rows]
        wins = sum(pnl > 0 for pnl in pnls if pnl != 0)
        losses = sum(pnl < 0 for pnl in pnls if pnl != 0)
        positive = sum(pnl for pnl in pnls if pnl > 0)
        negative = -sum(pnl for pnl in pnls if pnl < 0)
        equity = peak = float(CAPITAL_JPY)
        max_drawdown = 0.0
        for pnl in pnls:
            equity += pnl
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
        performance = {"proxy_total_pnl_jpy": sum(pnls),
                       "proxy_final_assets_jpy": equity,
                       "trade_day_win_rate": wins / (wins + losses) if wins + losses else None,
                       "profit_factor": positive / negative if negative else None,
                       "max_drawdown_pct": max_drawdown * 100,
                       "winning_days": wins, "losing_days": losses}
    return {"schema_version": 1, "basis": "estimated 5-minute OHLCV fills; not broker results",
            "target_sessions": TARGET_SESSIONS, "recorded_sessions": len(season),
            "extra_records_outside_first_20": max(0, len(latest) - TARGET_SESSIONS),
            "missing_tse_sessions": missing, "unsettled_dates": unresolved,
            "no_trade_days": sum(row["forecast_status"] == "NO_TRADE" for row in rows),
            "estimated_trade_days": sum(row["estimated_filled_count"] > 0 for row in rows),
            "records": rows, "performance": performance}


def append_day(archive: Path, picks_bytes: bytes, snapshot_bytes: bytes, db_path: Path,
               events_bytes: bytes | None = None, *, amend_unsettled: bool = False,
               now: datetime | None = None) -> tuple[Path, dict]:
    now = (now or datetime.now(JST)).astimezone(JST)
    picks, snapshot = json.loads(picks_bytes), json.loads(snapshot_bytes)
    trade_date = picks.get("trade_date")
    folder = day_dir(archive, trade_date)
    if snapshot.get("trading_date") != trade_date:
        raise ValueError("Snapshot date differs from picks")
    if now < datetime.combine(date.fromisoformat(trade_date), time(15, 30), JST):
        raise ValueError("Cannot record results before 15:30 JST on trade date")
    cutoff = datetime.fromisoformat(snapshot["cutoff_jst"])
    issued = datetime.fromisoformat(picks["generated_at_jst"])
    completed = datetime.fromisoformat(snapshot["completed_at_jst"])
    if (cutoff.utcoffset() is None or issued.utcoffset() is None or completed.utcoffset() is None or
        cutoff.astimezone(JST).date().isoformat() != trade_date or
        cutoff.astimezone(JST).time() != time(8, 15) or
        issued > cutoff or completed > cutoff):
        raise ValueError("Forecast or morning snapshot missed the 08:15 JST cutoff")
    revisions = _load_revisions(folder) if folder.exists() else []
    if revisions:
        if not amend_unsettled or revisions[-1]["execution"]["status"] != "UNSETTLED":
            raise FileExistsError("Daily record exists; only an unsettled day may be amended")
        if not events_bytes:
            raise ValueError("An unsettled amendment requires market-event evidence")
        if (revisions[0]["picks_sha256"] != digest(picks_bytes) or
            revisions[0]["snapshot_sha256"] != digest(snapshot_bytes)):
            raise ValueError("Amendment cannot change frozen morning inputs")
    elif amend_unsettled:
        raise ValueError("No unsettled record to amend")
    events = json.loads(events_bytes) if events_bytes else None
    report = settle(picks, picks_bytes, snapshot, snapshot_bytes, db_path, events)
    if snapshot.get("status") != "READY":
        # A failed data feed is not a successful strategic abstention. Keep the
        # day's audit record, but block continuous performance calculations.
        report.update(status="UNSETTLED", proxy_total_pnl_jpy=None,
                      reason="Morning input snapshot was not READY")
        report.pop("proxy_final_assets_jpy", None)
    if report["status"] not in {"NO_TRADE", "ESTIMATED_COMPLETE", "UNSETTLED"}:
        raise ValueError("Unknown execution status")
    if report["status"] == "UNSETTLED" and not report.get("reason"):
        raise ValueError("Unsettled result has no reason")
    record = {"schema_version": 1, "trade_date": trade_date,
              "recorded_at_jst": now.isoformat(), "revision": len(revisions) + 1,
              "previous_record_sha256": revisions[-1]["record_sha256"] if revisions else None,
              "snapshot_sha256": digest(snapshot_bytes), "picks_sha256": digest(picks_bytes),
              "execution_sha256": digest(encoded(report)),
              "market_events_sha256": digest(events_bytes) if events_bytes else None,
              "market_events_raw_utf8": events_bytes.decode("utf-8") if events_bytes else None,
              "market_events": events, "picks": picks, "execution": report}
    # JSON re-encoding of archived picks must preserve the exact source hash. The
    # source is generated by this project with indentation and newline, so store
    # its raw bytes separately for validation instead of assuming canonical JSON.
    record["picks_raw_utf8"] = picks_bytes.decode("utf-8")
    record["execution_raw_utf8"] = encoded(report).decode("utf-8")
    record["record_sha256"] = digest(encoded(record))
    path = folder / f"rev-{len(revisions) + 1:04d}.json"
    folder.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n")
    summary = summarize(archive)
    (archive / "summary.json").write_bytes(json.dumps(summary, ensure_ascii=False, indent=2,
                                              allow_nan=False).encode("utf-8") + b"\n")
    return path, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--picks", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--market-events", type=Path)
    parser.add_argument("--amend-unsettled", action="store_true")
    args = parser.parse_args()
    path, summary = append_day(
        args.archive, args.picks.read_bytes(), args.snapshot.read_bytes(), args.db,
        args.market_events.read_bytes() if args.market_events else None,
        amend_unsettled=args.amend_unsettled)
    print(f"{path}: {summary['records'][-1]['outcome_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
