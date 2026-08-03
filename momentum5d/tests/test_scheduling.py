from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from app.yahoo.scheduling import should_run, update_is_complete


def test_matching_complete_session_is_skipped() -> None:
    payload = {
        "update": {
            "status": "complete",
            "session": "morning",
            "market_date": "2026-07-31",
        }
    }

    assert update_is_complete(
        payload,
        session="morning",
        market_date=date(2026, 7, 31),
    )
    assert not update_is_complete(
        payload,
        session="close",
        market_date=date(2026, 7, 31),
    )


def test_missing_or_stale_dashboard_runs_update(tmp_path: Path) -> None:
    dashboard = tmp_path / "latest.json"
    assert should_run(
        dashboard,
        session="morning",
        market_date=date(2026, 7, 31),
    )


def test_old_schema_is_rebuilt_even_when_session_is_complete(tmp_path: Path) -> None:
    dashboard = tmp_path / "latest.json"
    update = {
        "status": "complete",
        "session": "close",
        "market_date": "2026-07-31",
    }
    dashboard.write_text(
        json.dumps({"schema_version": 7, "update": update}),
        encoding="utf-8",
    )
    assert should_run(
        dashboard,
        session="close",
        market_date=date(2026, 7, 31),
    )

    dashboard.write_text(
        json.dumps({"schema_version": 8, "update": update}),
        encoding="utf-8",
    )
    assert not should_run(
        dashboard,
        session="close",
        market_date=date(2026, 7, 31),
    )

    dashboard.write_text(
        json.dumps(
            {
                "update": {
                    "status": "complete",
                    "session": "morning",
                    "market_date": "2026-07-30",
                }
            }
        ),
        encoding="utf-8",
    )
    assert should_run(
        dashboard,
        session="morning",
        market_date=date(2026, 7, 31),
    )
