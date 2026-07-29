from __future__ import annotations

from app.config import Settings
from app.quality.checks import QualityValidator
from app.storage.parquet import ParquetStore


def equity(
    day: str,
    code: str,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 1000,
    factor: float = 1.0,
    adjusted_close: float | None = None,
) -> dict[str, object]:
    adjusted = close if adjusted_close is None else adjusted_close
    return {
        "Date": day,
        "Code": code,
        "O": open_,
        "H": high,
        "L": low,
        "C": close,
        "UL": "0",
        "LL": "0",
        "Vo": volume,
        "Va": 100000,
        "AdjFactor": factor,
        "AdjO": adjusted,
        "AdjH": adjusted,
        "AdjL": adjusted,
        "AdjC": adjusted,
        "AdjVo": volume,
    }


def seed_reference_data(
    store: ParquetStore, days: list[str], codes_by_day: dict[str, list[str]]
) -> None:
    store.write_raw_page(
        "trading_calendar",
        "range",
        endpoint="/markets/calendar",
        page_number=0,
        records=[{"Date": day, "HolDiv": "1"} for day in days],
    )
    store.process_raw_unit("trading_calendar", "range")

    records: list[dict[str, object]] = []
    for day, codes in codes_by_day.items():
        for code in codes:
            records.append(
                {
                    "Date": day,
                    "Code": code,
                    "CoName": code,
                    "CoNameEn": code,
                    "S17": "1",
                    "S17Nm": "食品",
                    "S33": "50",
                    "S33Nm": "水産",
                    "ScaleCat": "-",
                    "Mkt": "0111",
                    "MktNm": "プライム",
                    "Mrgn": "2",
                    "MrgnNm": "貸借",
                }
            )
    store.write_raw_page(
        "listed_master",
        "history",
        endpoint="/equities/master",
        page_number=0,
        records=records,
    )
    store.process_raw_unit("listed_master", "history")


def test_quality_detects_ohlc_zero_volume_and_abnormal_return(
    settings: Settings,
) -> None:
    store = ParquetStore(settings.raw_dir, settings.processed_dir)
    days = ["2026-01-05", "2026-01-06"]
    seed_reference_data(store, days, {day: ["13010"] for day in days})
    records = [
        equity(
            days[0],
            "13010",
            open_=100,
            high=110,
            low=90,
            close=100,
            volume=0,
        ),
        equity(
            days[1],
            "13010",
            open_=120,
            high=110,
            low=100,
            close=200,
            adjusted_close=200,
        ),
    ]
    store.write_raw_page(
        "equities_daily",
        "prices",
        endpoint="/equities/bars/daily",
        page_number=0,
        records=records,
    )
    store.process_raw_unit("equities_daily", "prices")

    report = QualityValidator(settings, parquet=store).run()
    checks = set(report.issues["check_name"])

    assert "zero_volume" in checks
    assert "unadjusted_ohlc_inconsistent" in checks
    assert "abnormal_adjusted_return" in checks
    assert report.has_errors
    assert report.issues_path.exists()
    assert report.summary_path.exists()


def test_quality_checks_split_consistency_and_historical_membership(
    settings: Settings,
) -> None:
    store = ParquetStore(settings.raw_dir, settings.processed_dir)
    days = ["2026-01-05", "2026-01-06"]
    # 99990は2日目の過去時点マスターに存在しない。
    seed_reference_data(
        store,
        days,
        {
            days[0]: ["13010", "99990"],
            days[1]: ["13010"],
        },
    )
    records = [
        equity(days[0], "13010", open_=100, high=105, low=95, close=100),
        equity(
            days[1],
            "13010",
            open_=80,
            high=85,
            low=75,
            close=80,
            factor=0.5,
            adjusted_close=100,
        ),
        equity(days[1], "99990", open_=100, high=105, low=95, close=100),
    ]
    store.write_raw_page(
        "equities_daily",
        "prices",
        endpoint="/equities/bars/daily",
        page_number=0,
        records=records,
    )
    store.process_raw_unit("equities_daily", "prices")

    report = QualityValidator(settings, parquet=store).run()
    checks = set(report.issues["check_name"])

    assert "stock_split_event" in checks
    assert "stock_split_raw_price_mismatch" in checks
    assert "price_without_historical_membership" in checks


def test_quality_handles_missing_adjusted_ohlc_without_crashing(
    settings: Settings,
) -> None:
    store = ParquetStore(settings.raw_dir, settings.processed_dir)
    day = "2026-01-05"
    seed_reference_data(store, [day], {day: ["13010"]})
    record = equity(day, "13010", open_=100, high=105, low=95, close=100)
    record["AdjO"] = None
    record["AdjH"] = None
    record["AdjL"] = None
    record["AdjC"] = None
    store.write_raw_page(
        "equities_daily",
        "prices",
        endpoint="/equities/bars/daily",
        page_number=0,
        records=[record],
    )
    store.process_raw_unit("equities_daily", "prices")

    report = QualityValidator(settings, parquet=store).run()

    assert "adjusted_ohlc_missing" in set(report.issues["check_name"])
