from datetime import timezone

from collect_ifis_candidate_news import categories, parse_gdelt_date, parse_snapshot


def test_keyword_categories_are_metadata_only():
    assert "EARNINGS" in categories("業績予想を上方修正")
    assert "ORDER_ADOPTION" in categories("大型案件を受注")
    assert "PARTNERSHIP_MA" in categories("資本業務提携を発表")
    assert categories("会社紹介") == ["OTHER"]


def test_snapshot_and_gdelt_dates_are_timezone_aware():
    snap = parse_snapshot("2026-09-04T01:20:00+09:00")
    seen = parse_gdelt_date("20260903T150000Z")
    assert seen is not None
    assert seen.tzinfo == timezone.utc
    assert seen <= snap.astimezone(timezone.utc)
