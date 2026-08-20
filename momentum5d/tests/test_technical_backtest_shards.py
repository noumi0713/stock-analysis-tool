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
