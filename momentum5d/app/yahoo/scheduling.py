from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")
DASHBOARD_SCHEMA_VERSION = 12


def update_is_complete(
    payload: dict[str, Any],
    *,
    session: str,
    market_date: date,
) -> bool:
    update = payload.get("update")
    if not isinstance(update, dict):
        return False
    return (
        update.get("status") == "complete"
        and update.get("session") == session
        and update.get("market_date") == market_date.isoformat()
    )


def should_run(
    dashboard_path: Path,
    *,
    session: str,
    market_date: date,
) -> bool:
    if not dashboard_path.exists():
        return True
    try:
        payload = json.loads(dashboard_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    if int(payload.get("schema_version", 0)) < DASHBOARD_SCHEMA_VERSION:
        return True
    return not update_is_complete(payload, session=session, market_date=market_date)


def main() -> int:
    parser = argparse.ArgumentParser(description="同一セッションの重複更新を抑止")
    parser.add_argument("--session", choices=["morning", "close"], required=True)
    parser.add_argument("--dashboard", type=Path, required=True)
    parser.add_argument("--market-date", type=date.fromisoformat, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="完了済みセッションでも再取得・再分析する",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        default=Path(os.environ["GITHUB_OUTPUT"]) if os.getenv("GITHUB_OUTPUT") else None,
    )
    args = parser.parse_args()
    market_date = args.market_date or datetime.now(JST).date()

    # ソースコードだけを公開する push では重い市場データ取得・バックテストを回さない。
    # 市場更新は定時 schedule または workflow_dispatch のときだけ実行する。
    is_code_push = os.getenv("GITHUB_EVENT_NAME") == "push"
    run = False if is_code_push else (
        args.force
        or should_run(
            args.dashboard,
            session=args.session,
            market_date=market_date,
        )
    )
    values = {
        "session": args.session,
        "market_date": market_date.isoformat(),
        "should_run": str(run).lower(),
    }
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as output:
            for key, value in values.items():
                output.write(f"{key}={value}\n")
    for key, value in values.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
