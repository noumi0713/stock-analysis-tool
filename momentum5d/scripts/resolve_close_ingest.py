from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")
CLOSE_DATA_READY_AT = time(15, 40)


@dataclass(frozen=True)
class CloseIngestPlan:
    as_of: date
    use_intraday_close: bool


def resolve_close_ingest_plan(now: datetime | None = None) -> CloseIngestPlan:
    """Choose a completed Japanese market date even when Actions runs late."""

    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    current_jst = current.astimezone(JST)
    if current_jst.time() >= CLOSE_DATA_READY_AT:
        return CloseIngestPlan(
            as_of=current_jst.date(),
            use_intraday_close=True,
        )
    return CloseIngestPlan(
        as_of=current_jst.date() - timedelta(days=1),
        use_intraday_close=False,
    )


def main() -> None:
    plan = resolve_close_ingest_plan()
    print(f"as_of={plan.as_of.isoformat()}")
    print(f"use_intraday_close={str(plan.use_intraday_close).lower()}")


if __name__ == "__main__":
    main()
