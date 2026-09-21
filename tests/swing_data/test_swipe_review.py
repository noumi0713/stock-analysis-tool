from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from swing_data.swipe_review import POPULATION_TYPE, build_swipe_review


def _write_stock(path: Path, code: str, latest_volume: float, *, latest_date: str = "2026-09-18") -> None:
    days = ["2026-09-11", "2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", latest_date]
    rows = []
    for i, date in enumerate(days):
        rows.append(
            {
                "date": date,
                "ticker": f"{code}.T",
                "open": 100 + i,
                "high": 102 + i,
                "low": 99 + i,
                "close": 101 + i,
                "volume": latest_volume if i == 5 else 1000 + i,
                "adj_close": 101 + i,
                "adj_open": 100 + i,
                "adj_high": 102 + i,
                "adj_low": 99 + i,
                "dividends": 0,
                "stock_splits": 0,
            }
        )
    pd.DataFrame(rows).to_csv(path / f"{code}.csv", index=False)


def _write_universe(path: Path, rows: list[tuple[str, str]]) -> None:
    pd.DataFrame(
        [{"stock_code": code, "ticker": f"{code}.T", "company_name": name, "sector17": "テスト"} for code, name in rows]
    ).to_csv(path / "universe.csv", index=False)


def test_build_swipe_review_ranks_latest_volume_descending(tmp_path: Path):
    (tmp_path / "stocks").mkdir()
    _write_universe(tmp_path, [("1111", "A"), ("2222", "B"), ("3333", "C")])
    _write_stock(tmp_path / "stocks", "1111", 3_000)
    _write_stock(tmp_path / "stocks", "2222", 9_000)
    _write_stock(tmp_path / "stocks", "3333", 6_000)

    status = build_swipe_review(tmp_path, price_date="2026-09-18")
    assert status["status"] == "success"
    assert status["population_type"] == POPULATION_TYPE
    assert status["count"] == 3

    payload = json.loads((tmp_path / "swipe_review_universe.json").read_text(encoding="utf-8"))
    assert [x["stock_code"] for x in payload["items"]] == ["2222", "3333", "1111"]
    assert [x["volume_rank"] for x in payload["items"]] == [1, 2, 3]
    assert payload["items"][0]["ranking_volume"] == 9_000
    assert "five_day_return_pct" not in payload["items"][0]
    assert status["technical_indicators_persisted"] is False


def test_build_swipe_review_limits_population_to_100(tmp_path: Path):
    (tmp_path / "stocks").mkdir()
    members = [(f"{code:04d}", f"Company {code}") for code in range(1000, 1101)]
    _write_universe(tmp_path, members)
    for index, (code, _) in enumerate(members):
        _write_stock(tmp_path / "stocks", code, 10_000 + index)

    status = build_swipe_review(tmp_path, price_date="2026-09-18")
    payload = json.loads((tmp_path / "swipe_review_universe.json").read_text(encoding="utf-8"))
    assert status["count"] == 100
    assert len(payload["items"]) == 100
    assert payload["items"][0]["stock_code"] == "1100"
    assert payload["items"][-1]["stock_code"] == "1001"


def test_build_swipe_review_excludes_stale_or_missing_latest_data(tmp_path: Path):
    (tmp_path / "stocks").mkdir()
    _write_universe(tmp_path, [("1111", "Current"), ("2222", "Stale"), ("3333", "Missing")])
    _write_stock(tmp_path / "stocks", "1111", 5_000)
    _write_stock(tmp_path / "stocks", "2222", 99_000, latest_date="2026-09-17")

    status = build_swipe_review(tmp_path, price_date="2026-09-18")
    payload = json.loads((tmp_path / "swipe_review_universe.json").read_text(encoding="utf-8"))
    assert status["status"] == "success"
    assert [x["stock_code"] for x in payload["items"]] == ["1111"]


def test_build_swipe_review_requires_price_date(tmp_path: Path):
    status = build_swipe_review(tmp_path)
    assert status["status"] == "not_ready"
    payload = json.loads((tmp_path / "swipe_review_universe.json").read_text(encoding="utf-8"))
    assert payload["items"] == []
