from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from swing_data.swipe_review import build_swipe_review


def _write_stock(path: Path, code: str, start: float = 100.0) -> None:
    rows = []
    for i in range(6):
        rows.append(
            {
                "date": f"2026-09-{10+i:02d}",
                "ticker": f"{code}.T",
                "open": start + i,
                "high": start + i + 2,
                "low": start + i - 1,
                "close": start + i + 1,
                "volume": 1000 + i,
                "adj_close": start + i + 1,
                "adj_open": start + i,
                "adj_high": start + i + 2,
                "adj_low": start + i - 1,
                "dividends": 0,
                "stock_splits": 0,
            }
        )
    pd.DataFrame(rows).to_csv(path / f"{code}.csv", index=False)


def test_build_swipe_review_success(tmp_path: Path):
    (tmp_path / "stocks").mkdir()
    (tmp_path / "bbs_ranking_status.json").write_text(
        json.dumps({"status": "success", "ranking_date": "2026-09-18"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "date": "2026-09-18",
                "rank": 1,
                "stock_code": "1111",
                "stock_name": "A",
                "market": "東証PRM",
                "price": 105,
            },
            {
                "date": "2026-09-18",
                "rank": 2,
                "stock_code": "2222",
                "stock_name": "B",
                "market": "東証GRT",
                "price": 205,
            },
        ]
    ).to_csv(tmp_path / "bbs_ranking_latest.csv", index=False)
    _write_stock(tmp_path / "stocks", "1111")
    _write_stock(tmp_path / "stocks", "2222", 200)

    status = build_swipe_review(tmp_path, price_date="2026-09-18")
    assert status["status"] == "success"
    assert status["count"] == 2
    assert status["priced_count"] == 2

    payload = json.loads((tmp_path / "swipe_review_universe.json").read_text(encoding="utf-8"))
    assert [x["stock_code"] for x in payload["items"]] == ["1111", "2222"]
    assert len(payload["items"][0]["recent_prices"]) == 6
    assert "five_day_return_pct" not in payload["items"][0]
    assert status["technical_indicators_persisted"] is False


def test_build_swipe_review_rejects_stale_latest(tmp_path: Path):
    (tmp_path / "stocks").mkdir()
    (tmp_path / "bbs_ranking_status.json").write_text(
        json.dumps({"status": "success", "ranking_date": "2026-09-18"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "date": "2026-09-17",
                "rank": 1,
                "stock_code": "1111",
                "stock_name": "A",
                "market": "東証PRM",
                "price": 105,
            }
        ]
    ).to_csv(tmp_path / "bbs_ranking_latest.csv", index=False)

    status = build_swipe_review(tmp_path, price_date="2026-09-18")
    assert status["status"] == "not_ready"
    payload = json.loads((tmp_path / "swipe_review_universe.json").read_text(encoding="utf-8"))
    assert payload["items"] == []


def test_build_swipe_review_keeps_missing_price_member(tmp_path: Path):
    (tmp_path / "stocks").mkdir()
    (tmp_path / "bbs_ranking_status.json").write_text(
        json.dumps({"status": "success", "ranking_date": "2026-09-18"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "date": "2026-09-18",
                "rank": 1,
                "stock_code": "3333",
                "stock_name": "Missing",
                "market": "東証STD",
                "price": 300,
            }
        ]
    ).to_csv(tmp_path / "bbs_ranking_latest.csv", index=False)

    status = build_swipe_review(tmp_path, price_date="2026-09-18")
    assert status["count"] == 1
    assert status["priced_count"] == 0
    payload = json.loads((tmp_path / "swipe_review_universe.json").read_text(encoding="utf-8"))
    assert payload["items"][0]["stock_code"] == "3333"
    assert payload["items"][0]["price_data_issue"] == "stock_csv_missing"
