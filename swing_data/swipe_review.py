"""Build the swipe-review input from the published 120-session snapshot.

This module persists only raw/recent price observations needed by the UI.
It deliberately does not persist technical indicators or derived returns.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from swing_data.collector import atomic_json

DECISION_PUBLIC = (
    "https://raw.githubusercontent.com/noumi0713/stock-analysis-tool/"
    "swipe-decisions/swipe_review"
)


def _status_payload(
    *,
    status: str,
    ranking_date: str | None,
    price_date: str | None,
    count: int,
    priced_count: int,
    note: str,
) -> dict[str, Any]:
    date_part = ranking_date or "YYYY-MM-DD"
    return {
        "status": status,
        "ranking_date": ranking_date,
        "price_date": price_date,
        "population": "Yahoo掲示板投稿ランキング当日1〜100位",
        "population_limit": 100,
        "display_sort": "画面で直近6営業日の調整後終値から5営業日騰落率を計算し降順",
        "technical_indicators_persisted": False,
        "count": count,
        "priced_count": priced_count,
        "decision_url": f"{DECISION_PUBLIC}/data/{date_part}.json",
        "recommendation_url": f"{DECISION_PUBLIC}/recommendations/{date_part}.json",
        "note": note,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _recent_prices(stock_file: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not stock_file.exists():
        return [], "stock_csv_missing"
    try:
        frame = pd.read_csv(stock_file)
    except Exception:
        return [], "stock_csv_unreadable"

    required = {"date", "adj_close", "close", "volume"}
    if not required.issubset(frame.columns):
        return [], "stock_csv_columns_missing"

    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    frame["adj_close"] = pd.to_numeric(frame["adj_close"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    valid = frame[
        frame["adj_close"].notna()
        & frame["close"].notna()
        & frame["volume"].notna()
        & (frame["adj_close"] > 0)
        & (frame["close"] > 0)
        & (frame["volume"] >= 0)
    ]
    if valid.empty:
        return [], "no_valid_price_rows"

    recent = valid.tail(6)
    rows = [
        {
            "date": str(row.date),
            "adj_close": float(row.adj_close),
            "close": float(row.close),
            "volume": float(row.volume),
        }
        for row in recent.itertuples(index=False)
    ]
    if len(rows) < 6:
        return rows, "fewer_than_6_sessions"
    return rows, None


def build_swipe_review(target: Path, price_date: str | None = None) -> dict[str, Any]:
    """Create swipe_review_universe.json and swipe_review_status.json in target."""
    target = Path(target)
    ranking_status = _read_json(target / "bbs_ranking_status.json")
    ranking_date = ranking_status.get("ranking_date")

    if ranking_status.get("status") != "success":
        status = _status_payload(
            status="not_ready",
            ranking_date=ranking_date,
            price_date=price_date,
            count=0,
            priced_count=0,
            note="当日の掲示板ランキング取得が成功していないため、母集団を作成しません。",
        )
        atomic_json(target / "swipe_review_status.json", status)
        atomic_json(
            target / "swipe_review_universe.json",
            {"status": "not_ready", "ranking_date": ranking_date, "items": []},
        )
        return status

    ranking_file = target / "bbs_ranking_latest.csv"
    if not ranking_file.exists():
        status = _status_payload(
            status="not_ready",
            ranking_date=ranking_date,
            price_date=price_date,
            count=0,
            priced_count=0,
            note="bbs_ranking_latest.csv がありません。前日データでは代用しません。",
        )
        atomic_json(target / "swipe_review_status.json", status)
        atomic_json(
            target / "swipe_review_universe.json",
            {"status": "not_ready", "ranking_date": ranking_date, "items": []},
        )
        return status

    ranking = pd.read_csv(ranking_file, dtype={"stock_code": str}).fillna("")
    if "date" in ranking.columns and not ranking.empty:
        file_dates = {str(v) for v in ranking["date"].tolist() if str(v)}
        if ranking_date and file_dates and file_dates != {str(ranking_date)}:
            status = _status_payload(
                status="not_ready",
                ranking_date=ranking_date,
                price_date=price_date,
                count=0,
                priced_count=0,
                note="ランキングstatusとlatest CSVの日付が一致しません。前日データでは代用しません。",
            )
            atomic_json(target / "swipe_review_status.json", status)
            atomic_json(
                target / "swipe_review_universe.json",
                {"status": "not_ready", "ranking_date": ranking_date, "items": []},
            )
            return status

    ranking["rank"] = pd.to_numeric(ranking["rank"], errors="coerce")
    ranking = ranking[(ranking["rank"] >= 1) & (ranking["rank"] <= 100)].copy()
    ranking = ranking.sort_values("rank").drop_duplicates("stock_code", keep="first").head(100)

    items: list[dict[str, Any]] = []
    priced_count = 0
    for row in ranking.itertuples(index=False):
        code = str(row.stock_code).strip()
        recent, issue = _recent_prices(target / "stocks" / f"{code}.csv")
        if len(recent) >= 6:
            priced_count += 1
        item = {
            "bbs_rank": int(row.rank),
            "stock_code": code,
            "stock_name": str(getattr(row, "stock_name", "")),
            "market": str(getattr(row, "market", "")),
            "ranking_price": (
                float(getattr(row, "price"))
                if str(getattr(row, "price", "")).strip()
                and pd.notna(pd.to_numeric(getattr(row, "price"), errors="coerce"))
                else None
            ),
            "recent_prices": recent,
            "price_data_issue": issue,
            "ohlcv_path": f"stocks/{code}.csv",
        }
        items.append(item)

    status = _status_payload(
        status="success",
        ranking_date=str(ranking_date) if ranking_date is not None else None,
        price_date=price_date,
        count=len(items),
        priced_count=priced_count,
        note=(
            "母集団は当日掲示板ランキング1〜100位のみ。"
            "5営業日騰落率・RSI等は画面/分析時に生データから計算します。"
        ),
    )
    payload = {
        "status": "success",
        "ranking_date": status["ranking_date"],
        "price_date": price_date,
        "population_limit": 100,
        "sort_instruction": "recent_pricesの最終adj_close / 6本前adj_close - 1 を画面で計算し降順",
        "items": items,
    }
    atomic_json(target / "swipe_review_universe.json", payload)
    atomic_json(target / "swipe_review_status.json", status)
    return status
