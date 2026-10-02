"""Build the swipe-review input from the published 120-session snapshot.

The swipe population is the 100 TSE stocks with the highest estimated trading
value (latest close times volume) on the latest completed equity session.
Adds causal pattern monitoring candidates; derived returns remain UI-time calculations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from swing_data.collector import atomic_json
from swing_data.pattern_candidates import detect, RULES, VERSION

DECISION_PUBLIC = (
    "https://raw.githubusercontent.com/noumi0713/stock-analysis-tool/"
    "swipe-decisions/swipe_review"
)
POPULATION_TYPE = "trading_value_top_100_plus_pattern_candidates"


def _status_payload(
    *,
    status: str,
    price_date: str | None,
    count: int,
    priced_count: int,
    note: str,
) -> dict[str, Any]:
    date_part = price_date or "YYYY-MM-DD"
    return {
        "status": status,
        "ranking_date": price_date,
        "price_date": price_date,
        "population_type": POPULATION_TYPE,
        "population": "売買代金上位100銘柄＋三角持ち合い上抜け・底打ち候補",
        "population_limit": 100,
        "ranking_metric": "最新終値×出来高による推計売買代金（円）",
        "display_sort": "画面で直近6営業日の調整後終値から5営業日騰落率を計算し降順",
        "technical_indicators_persisted": False,
        "pattern_rule_version": VERSION,
        "pattern_signals_validated": False,
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


def _write_not_ready(target: Path, *, price_date: str | None, note: str) -> dict[str, Any]:
    status = _status_payload(
        status="not_ready",
        price_date=price_date,
        count=0,
        priced_count=0,
        note=note,
    )
    atomic_json(target / "swipe_review_status.json", status)
    atomic_json(
        target / "swipe_review_universe.json",
        {
            "status": "not_ready",
            "ranking_date": price_date,
            "price_date": price_date,
            "population_type": POPULATION_TYPE,
            "items": [],
        },
    )
    return status


def build_swipe_review(target: Path, price_date: str | None = None) -> dict[str, Any]:
    """Create the latest-volume swipe universe and status files in target."""
    target = Path(target)
    price_date = str(price_date).strip() if price_date else None
    if not price_date:
        return _write_not_ready(
            target,
            price_date=None,
            note="最新取引日を確認できないため、売買代金ランキングを作成しません。",
        )

    universe_file = target / "universe.csv"
    if not universe_file.exists():
        return _write_not_ready(
            target,
            price_date=price_date,
            note="universe.csv がないため、売買代金ランキングを作成しません。",
        )
    try:
        universe = pd.read_csv(universe_file, dtype=str).fillna("")
    except Exception:
        return _write_not_ready(
            target,
            price_date=price_date,
            note="universe.csv を読み込めないため、売買代金ランキングを作成しません。",
        )
    if not {"stock_code", "company_name"}.issubset(universe.columns):
        return _write_not_ready(
            target,
            price_date=price_date,
            note="universe.csv の必須列がないため、売買代金ランキングを作成しません。",
        )

    candidates: list[dict[str, Any]] = []
    for row in universe.drop_duplicates("stock_code", keep="first").itertuples(index=False):
        code = str(row.stock_code).strip().upper()
        if not code:
            continue
        recent, issue = _recent_prices(target / "stocks" / f"{code}.csv")
        if not recent or recent[-1]["date"] != price_date:
            continue
        latest = recent[-1]
        candidates.append(
            {
                "stock_code": code,
                "stock_name": str(getattr(row, "company_name", "")),
                "market": "",
                "sector": str(getattr(row, "sector17", "")),
                "ranking_price": latest["close"],
                "ranking_trading_value": latest["close"] * latest["volume"],
                "recent_prices": recent,
                "price_data_issue": issue,
                "ohlcv_path": f"stocks/{code}.csv",
            }
        )

    pattern_items = []
    for item in candidates:
        frame = pd.read_csv(target / item["ohlcv_path"])
        signals = detect(frame, price_date)
        item["pattern_signals"] = signals
        item["candidate_sources"] = ["pattern"] if signals else []
        if signals:
            pattern_items.append(item)
    candidates.sort(key=lambda item: (-item["ranking_trading_value"], item["stock_code"]))
    for rank, item in enumerate(candidates, start=1):
        item["trading_value_rank"] = rank
        if rank <= 100:
            item["candidate_sources"].insert(0, "trading_value_top_100")
    top = candidates[:100]
    top_codes = {x["stock_code"] for x in top}
    items = top + [x for x in pattern_items if x["stock_code"] not in top_codes]
    atomic_json(target / "pattern_candidates.json", {
        "status": "success", "price_date": price_date, "rule_version": VERSION,
        "rules": RULES, "scanned_count": len(candidates), "count": len(pattern_items),
        "reference_code": "285A", "reference_matches": next((x["pattern_signals"] for x in candidates if x["stock_code"] == "285A"), []),
        "items": pattern_items,
    })

    if not items:
        return _write_not_ready(
            target,
            price_date=price_date,
            note="最新取引日と一致する売買代金データがないため、母集団を作成しません。",
        )

    priced_count = sum(len(item["recent_prices"]) >= 6 for item in items)
    status = _status_payload(
        status="success",
        price_date=price_date,
        count=len(items),
        priced_count=priced_count,
        note=(
            "売買代金上位100銘柄に、直近3営業日の三角持ち合い上抜け・底打ち候補を追加。候補は買い推奨ではありません。"
            "5営業日騰落率・RSI等は画面/分析時に生データから計算します。"
        ),
    )
    status["pattern_count"] = len(pattern_items)
    status["pattern_added_count"] = len(items) - len(top)
    payload = {
        "status": "success",
        "ranking_date": price_date,
        "price_date": price_date,
        "population_type": POPULATION_TYPE,
        "population_limit": 100,
        "ranking_metric": "latest_close_times_volume_descending",
        "sort_instruction": "recent_pricesの最終adj_close / 6本前adj_close - 1 を画面で計算し降順",
        "items": items,
    }
    atomic_json(target / "swipe_review_universe.json", payload)
    atomic_json(target / "swipe_review_status.json", status)
    return status


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    manifest = _read_json(args.target / "manifest.json")
    price_date = manifest.get("expected_equity_date")
    result = build_swipe_review(args.target, price_date=str(price_date) if price_date else None)
    print(json.dumps(result, ensure_ascii=False))
