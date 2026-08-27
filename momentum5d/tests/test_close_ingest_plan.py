from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from scripts.resolve_close_ingest import resolve_close_ingest_plan


@pytest.mark.parametrize(
    ("now", "expected_date", "expected_intraday"),
    [
        (
            datetime(2026, 8, 27, 6, 39, tzinfo=UTC),
            date(2026, 8, 26),
            False,
        ),
        (
            datetime(2026, 8, 27, 6, 40, tzinfo=UTC),
            date(2026, 8, 27),
            True,
        ),
        (
            datetime(2026, 8, 27, 18, 21, tzinfo=UTC),
            date(2026, 8, 27),
            False,
        ),
    ],
)
def test_resolve_close_ingest_plan(
    now: datetime,
    expected_date: date,
    expected_intraday: bool,
) -> None:
    plan = resolve_close_ingest_plan(now)

    assert plan.as_of == expected_date
    assert plan.use_intraday_close is expected_intraday


def test_resolve_close_ingest_plan_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        resolve_close_ingest_plan(datetime(2026, 8, 27, 15, 40))
