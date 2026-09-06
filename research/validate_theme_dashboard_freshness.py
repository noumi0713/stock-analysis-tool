from __future__ import annotations

import json
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals

SNAPSHOT = Path("theme-dashboard/data/chatgpt_snapshot.json")
STATUS = Path("theme-dashboard/data/update_status.json")
JST = ZoneInfo("Asia/Tokyo")


def expected_market_date(now: datetime) -> str:
    cal = xcals.get_calendar("XTKS")
    # Before the post-close refresh window, the latest completed session is the
    # previous session. After 16:00 JST, today's session is expected if today is
    # a TSE session; holidays/weekends resolve to the previous session.
    anchor = now.date() if now.time() >= time(16, 0) else now.date() - timedelta(days=1)
    session = cal.date_to_session(anchor, direction="previous")
    return session.strftime("%Y-%m-%d")


def main() -> int:
    now = datetime.now(JST)
    if not SNAPSHOT.exists():
        print("snapshot missing")
        return 2
    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    actual = str(data.get("market_date") or "")
    expected = expected_market_date(now)
    if actual == expected:
        print(json.dumps({"fresh": True, "market_date": actual, "expected": expected}, ensure_ascii=False))
        return 0

    status = {
        "attempted_at": now.isoformat(timespec="seconds"),
        "result": "STALE_MARKET_DATE",
        "market_date": actual or None,
        "expected_market_date": expected,
        "message": "Latest downloaded market date is older than the latest completed TSE session; preserve previous certified snapshot and retry later.",
    }
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 2


if __name__ == "__main__":
    sys.exit(main())
