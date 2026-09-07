import json
import runpy
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_market_history.py"


def test_discover_universe_prefers_sector_record(tmp_path: Path) -> None:
    dashboard = tmp_path / "dashboard-data"
    config = tmp_path / "momentum5d" / "config"
    dashboard.mkdir(parents=True)
    config.mkdir(parents=True)
    (config / "prime_tickers.txt").write_text("7203.T\n", encoding="utf-8")
    (dashboard / "latest.json").write_text(
        json.dumps(
            {
                "duplicate_without_sector": {"ticker": "7203.T"},
                "stocks": [
                    {
                        "ticker": "7203.T",
                        "company_name": "Toyota",
                        "sector_17_name": "自動車・輸送機",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    module = runpy.run_path(str(SCRIPT))
    universe, _ = module["discover_universe"](tmp_path)
    row = universe.loc[universe["ticker"] == "7203.T"].iloc[0]
    assert row["name"] == "Toyota"
    assert row["sector"] == "自動車・輸送機"
