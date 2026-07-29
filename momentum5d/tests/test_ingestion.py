from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import date
from typing import Any

from app.api.client import APIError, ApiPage
from app.config import Settings
from app.ingestion.pipeline import DatasetSpec, IngestionPipeline


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], str | None, int]] = []

    def iter_pages(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        *,
        pagination_key: str | None = None,
        start_page: int = 0,
    ) -> Iterator[ApiPage]:
        query = dict(params or {})
        self.calls.append((path, query, pagination_key, start_page))
        day = query.get("date") or query.get("from")
        if path == "/markets/calendar":
            yield ApiPage(
                records=[
                    {"Date": "2026-01-05", "HolDiv": "1"},
                    {"Date": "2026-01-06", "HolDiv": "1"},
                    {"Date": "2026-01-07", "HolDiv": "0"},
                ],
                next_key=None,
                page_number=start_page,
            )
        elif path == "/equities/master":
            yield ApiPage(
                records=[
                    {
                        "Date": day,
                        "Code": "13010",
                        "CoName": "テスト",
                        "CoNameEn": "Test",
                        "S17": "1",
                        "S17Nm": "食品",
                        "S33": "50",
                        "S33Nm": "水産",
                        "ScaleCat": "-",
                        "Mkt": "0111",
                        "MktNm": "プライム",
                        "Mrgn": "2",
                        "MrgnNm": "貸借",
                    },
                    {
                        "Date": day,
                        "Code": "13020",
                        "CoName": "対象外",
                        "CoNameEn": "Excluded",
                        "S17": "1",
                        "S17Nm": "食品",
                        "S33": "50",
                        "S33Nm": "水産",
                        "ScaleCat": "-",
                        "Mkt": "0112",
                        "MktNm": "スタンダード",
                        "Mrgn": "2",
                        "MrgnNm": "貸借",
                    },
                ],
                next_key=None,
                page_number=start_page,
            )
        elif path == "/equities/bars/daily":
            yield ApiPage(
                records=[
                    {
                        "Date": day,
                        "Code": "13010",
                        "O": 100,
                        "H": 110,
                        "L": 90,
                        "C": 105,
                        "UL": "0",
                        "LL": "0",
                        "Vo": 1000,
                        "Va": 100000,
                        "AdjFactor": 1.0,
                        "AdjO": 100,
                        "AdjH": 110,
                        "AdjL": 90,
                        "AdjC": 105,
                        "AdjVo": 1000,
                    },
                    {
                        "Date": day,
                        "Code": "13020",
                        "O": 200,
                        "H": 210,
                        "L": 190,
                        "C": 205,
                        "UL": "0",
                        "LL": "0",
                        "Vo": 2000,
                        "Va": 400000,
                        "AdjFactor": 1.0,
                        "AdjO": 200,
                        "AdjH": 210,
                        "AdjL": 190,
                        "AdjC": 205,
                        "AdjVo": 2000,
                    },
                ],
                next_key=None,
                page_number=start_page,
            )
        elif path == "/indices/bars/daily/topix":
            raise APIError("plan does not include TOPIX", status_code=403)
        elif path == "/indices/bars/daily":
            yield ApiPage(
                records=[
                    {
                        "Date": day,
                        "Code": "0000",
                        "O": 1000,
                        "H": 1010,
                        "L": 990,
                        "C": 1005,
                    }
                ],
                next_key=None,
                page_number=start_page,
            )
        else:
            raise AssertionError(path)


def test_full_ingestion_uses_official_calendar_and_skips_unlicensed_optional(
    settings: Settings,
) -> None:
    fake = FakeClient()
    pipeline = IngestionPipeline(settings, fake)  # type: ignore[arg-type]

    summary = pipeline.ingest(date(2026, 1, 5), date(2026, 1, 7))

    assert summary["trading_days"] == 2
    assert summary["unavailable_datasets"] == ["topix_daily"]
    equities = pipeline.parquet.read_processed("equities_daily")
    masters = pipeline.parquet.read_processed("listed_master")
    assert len(equities) == 2
    assert len(masters) == 2
    assert set(equities["code"]) == {"13010"}
    assert set(masters["market_code"]) == {"0111"}
    assert pipeline.parquet.max_date("equities_daily") == date(2026, 1, 6)
    topix_calls = [call for call in fake.calls if "topix" in call[0]]
    assert len(topix_calls) == 1


def test_prime_filter_is_applied_to_raw_and_processed(settings: Settings) -> None:
    pipeline = IngestionPipeline(settings, FakeClient())  # type: ignore[arg-type]

    summary = pipeline.ingest(
        date(2026, 1, 5),
        date(2026, 1, 5),
        datasets=("equities_daily",),
    )

    assert summary["datasets"] == ["listed_master", "equities_daily"]
    raw_master = pipeline.parquet.read_raw_unit("listed_master", "2026-01-05")
    raw_equities = pipeline.parquet.read_raw_unit("equities_daily", "2026-01-05")
    assert set(raw_master["Mkt"]) == {"0111"}
    assert set(raw_equities["Code"]) == {"13010"}
    assert set(pipeline.parquet.read_processed("listed_master")["code"]) == {"13010"}
    assert set(pipeline.parquet.read_processed("equities_daily")["code"]) == {"13010"}


def test_unit_resume_starts_from_saved_cursor(settings: Settings) -> None:
    fake = FakeClient()
    pipeline = IngestionPipeline(settings, fake)  # type: ignore[arg-type]
    master = DatasetSpec("listed_master", "/equities/master", True)
    pipeline._ingest_unit(
        master,
        "2026-01-05",
        {"date": "2026-01-05"},
        allow_empty=False,
    )
    fake.calls.clear()
    spec = DatasetSpec("equities_daily", "/equities/bars/daily", True)
    pipeline.checkpoints.begin(
        spec.name,
        "2026-01-05",
        endpoint=spec.endpoint,
        params={"date": "2026-01-05"},
    )
    pipeline.parquet.write_raw_page(
        spec.name,
        "2026-01-05",
        endpoint=spec.endpoint,
        page_number=0,
        records=[],
    )
    pipeline.checkpoints.page_saved(
        spec.name,
        "2026-01-05",
        next_pagination_key="resume-key",
        next_page=1,
        added_rows=0,
    )

    pipeline._ingest_unit(
        spec,
        "2026-01-05",
        {"date": "2026-01-05"},
        allow_empty=False,
    )

    assert fake.calls[0][2:] == ("resume-key", 1)
    checkpoint = pipeline.checkpoints.get(spec.name, "2026-01-05")
    assert checkpoint["status"] == "complete"


def test_update_can_resume_earlier_transient_failure(settings: Settings) -> None:
    pipeline = IngestionPipeline(settings, FakeClient())  # type: ignore[arg-type]
    pipeline.checkpoints.begin(
        "indices_daily",
        "2026-01-05",
        endpoint="/indices/bars/daily",
        params={"date": "2026-01-05"},
    )
    pipeline.checkpoints.fail(
        "indices_daily",
        "2026-01-05",
        "temporary network failure",
    )
    pipeline.checkpoints.begin(
        "topix_daily",
        "2026-01-04",
        endpoint="/indices/bars/daily/topix",
        params={"from": "2026-01-04", "to": "2026-01-04"},
    )
    pipeline.checkpoints.skip_unavailable(
        "topix_daily",
        "2026-01-04",
        "plan does not include TOPIX",
    )

    assert pipeline._earliest_pending_date() == date(2026, 1, 5)
