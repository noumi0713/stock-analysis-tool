from __future__ import annotations

from datetime import date

import pandas as pd

from app.point_in_time_universe import (
    build_point_in_time_universe,
    filter_prices_by_point_in_time_universe,
    parse_delisting_table,
    parse_new_listing_table,
)


def test_parses_paired_jpx_new_listing_rows() -> None:
    table = pd.DataFrame(
        [
            ["2025/04/01 （2025/03/01）", "新会社", "123A"],
            ["2025/04/01 （2025/03/01）", "新会社", "グロース"],
        ]
    )

    result = parse_new_listing_table(table)

    assert result.iloc[0]["ticker"] == "123A.T"
    assert result.iloc[0]["valid_from"] == pd.Timestamp("2025-04-01")


def test_builds_current_and_delisted_point_in_time_intervals() -> None:
    current = pd.DataFrame(
        [{"ticker": "1111.T", "company_name": "現上場会社"}]
    )
    listings = pd.DataFrame(
        [
            {
                "ticker": "1111.T",
                "company_name": "現上場会社",
                "market": "プライム",
                "valid_from": pd.Timestamp("2024-01-10"),
            }
        ]
    )
    delisted_table = pd.DataFrame(
        [
            {
                "上場廃止日": "2025/06/30",
                "銘柄名": "旧上場会社",
                "コード": "2222",
                "市場区分": "スタンダード",
            }
        ]
    )
    delistings = parse_delisting_table(delisted_table)

    history = build_point_in_time_universe(
        current,
        listings,
        delistings,
        start=date(2023, 8, 1),
        as_of=date(2026, 8, 31),
    )

    current_row = history.loc[history["ticker"].eq("1111.T")].iloc[0]
    old_row = history.loc[history["ticker"].eq("2222.T")].iloc[0]
    assert current_row["valid_from"] == pd.Timestamp("2024-01-10")
    assert pd.isna(current_row["valid_to"])
    assert old_row["valid_from"] == pd.Timestamp("2023-08-01")
    assert old_row["valid_to"] == pd.Timestamp("2025-06-30")


def test_filters_prices_outside_listing_interval() -> None:
    prices = pd.DataFrame(
        {
            "ticker": ["1111.T", "1111.T", "1111.T"],
            "date": ["2024-12-31", "2025-01-01", "2025-12-31"],
            "close": [100, 101, 102],
        }
    )
    history = pd.DataFrame(
        {
            "ticker": ["1111.T"],
            "valid_from": ["2025-01-01"],
            "valid_to": ["2025-06-30"],
            "ticker_reused": [False],
        }
    )

    result = filter_prices_by_point_in_time_universe(prices, history)

    assert result["date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-01-01"]
