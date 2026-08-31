import json

import pandas as pd

from scripts.export_technical_backtest_shards import export_shards


def test_exports_all_tickers_into_monthly_three_year_shards(tmp_path) -> None:
    source = {
        "latest_date": "2026-08-20",
        "generated_at": "2026-08-20T00:00:00Z",
        "stocks": [{"ticker": "1111.T", "company_name": "一社"}],
    }
    prices = pd.DataFrame(
        [
            {
                "date": "2023-08-19",
                "ticker": "1111.T",
                "open": 90,
                "high": 91,
                "low": 89,
                "close": 90,
                "adjusted_close": 90,
                "volume": 10,
            },
            {
                "date": "2023-08-21",
                "ticker": "1111.T",
                "open": 100,
                "high": 110,
                "low": 95,
                "close": 105,
                "adjusted_close": 105,
                "volume": 1_000,
            },
            {
                "date": "2026-08-20",
                "ticker": "2222.T",
                "open": 200,
                "high": 210,
                "low": 190,
                "close": 205,
                "adjusted_close": 205,
                "volume": 2_000,
            },
        ]
    )

    manifest = export_shards(source, prices, tmp_path)

    assert manifest["meta"]["dataScope"] == "all-tse"
    assert manifest["meta"]["stockCount"] == 2
    assert manifest["meta"]["startDate"] == "2023-08-21"
    assert [item["path"] for item in manifest["shards"]] == [
        "2023-08.json",
        "2026-08.json",
    ]
    first = json.loads((tmp_path / "2023-08.json").read_text())
    assert first["bars"]["1111.T"][0] == [0, 100.0, 110.0, 95.0, 105.0, 1000]


def test_export_adjusts_volume_to_match_adjusted_price_basis(tmp_path) -> None:
    source = {
        "latest_date": "2026-08-28",
        "generated_at": "2026-08-28T08:00:00Z",
        "stocks": [{"ticker": "9279.T", "company_name": "ギフトHD"}],
    }
    prices = pd.DataFrame(
        [
            {
                "date": "2026-08-27",
                "ticker": "9279.T",
                "open": 5_000,
                "high": 5_200,
                "low": 4_900,
                "close": 5_120,
                "adjusted_close": 2_560,
                "volume": 100_000,
            },
            {
                "date": "2026-08-28",
                "ticker": "9279.T",
                "open": 2_484,
                "high": 2_530,
                "low": 2_435,
                "close": 2_475,
                "adjusted_close": 2_475,
                "volume": 200_000,
            },
        ]
    )

    export_shards(source, prices, tmp_path)

    shard = json.loads((tmp_path / "2026-08.json").read_text())
    assert shard["bars"]["9279.T"][0][-1] == 200_000
    assert shard["bars"]["9279.T"][1][-1] == 200_000


def test_export_applies_point_in_time_universe(tmp_path) -> None:
    source = {
        "latest_date": "2026-08-20",
        "generated_at": "2026-08-20T00:00:00Z",
        "stocks": [{"ticker": "1111.T", "company_name": "一社"}],
    }
    prices = pd.DataFrame(
        [
            {
                "date": day,
                "ticker": "1111.T",
                "open": 100,
                "high": 101,
                "low": 99,
                "close": 100,
                "adjusted_close": 100,
                "volume": 1_000,
            }
            for day in ("2024-12-31", "2025-01-02", "2025-07-01")
        ]
    )
    history = pd.DataFrame(
        {
            "ticker": ["1111.T"],
            "valid_from": ["2025-01-01"],
            "valid_to": ["2025-06-30"],
            "ticker_reused": [False],
        }
    )

    manifest = export_shards(source, prices, tmp_path, universe_history=history)

    assert manifest["meta"]["universeMode"] == "jpx_point_in_time"
    shard = json.loads((tmp_path / "2025-01.json").read_text())
    assert len(shard["bars"]["1111.T"]) == 1
