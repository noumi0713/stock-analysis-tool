from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_latest_signals(source: dict[str, Any]) -> dict[str, Any]:
    """Build the small, current-market-date payload from the dashboard export."""

    latest_date = str(source.get("latest_date") or "")
    if not latest_date:
        raise ValueError("Dashboard payload has no latest_date")

    ten_day = source.get("ten_day_signal_study") or {}
    study = ten_day.get("demo_trade_signal_study") or {}
    if study.get("status") != "completed":
        raise ValueError("424 signal study is unavailable")

    rows_by_ticker: dict[str, dict[str, Any]] = {}
    for row in source.get("stocks") or []:
        ticker = str(row.get("ticker") or "")
        if ticker:
            rows_by_ticker[ticker] = row
    for row in source.get("candidates") or []:
        ticker = str(row.get("ticker") or "")
        if ticker:
            rows_by_ticker[ticker] = row

    signals: list[dict[str, Any]] = []
    for live in study.get("live_signals") or []:
        ticker = str(live.get("ticker") or "")
        row = dict(rows_by_ticker.get(ticker) or {})
        rank = int(live.get("rank") or row.get("ml_ten_day_rank") or len(signals) + 1)
        code = str(row.get("code") or ticker.removesuffix(".T"))
        pattern = str(live.get("shape") or row.get("rise_pattern_shape") or "")
        close = row.get("close", live.get("signal_close_yen"))
        trading_value = row.get("turnover_value", live.get("daily_turnover_yen"))
        target_probability = row.get(
            "ml_ten_day_probability", live.get("target_probability")
        )
        down_5pct_probability = row.get(
            "ml_ten_day_down_5pct_probability", live.get("down_5pct_probability")
        )
        down_8pct_probability = row.get(
            "ml_ten_day_down_8pct_probability", live.get("down_8pct_probability")
        )
        expected_net_return = row.get(
            "ml_ten_day_expected_net_return", live.get("expected_net_return")
        )
        row.update(
            {
                "date": latest_date,
                "code": code,
                "ticker": ticker,
                "name": row.get("company_name"),
                "rank": rank,
                "signal": pattern,
                "pattern": pattern,
                "close": close,
                "trading_value": trading_value,
                "RSI": row.get("rsi_14"),
                "ATR": row.get("atr_14_pct"),
                "target_probability": target_probability,
                "down_5pct_probability": down_5pct_probability,
                "down_8pct_probability": down_8pct_probability,
                "expected_net_return": expected_net_return,
                # Keep the existing dashboard field names so the site can consume
                # this payload without changing ranking or display calculations.
                "turnover_value": trading_value,
                "ml_ten_day_signal": True,
                "ml_ten_day_rank": rank,
                "ml_ten_day_probability": target_probability,
                "ml_ten_day_down_5pct_probability": down_5pct_probability,
                "ml_ten_day_down_8pct_probability": down_8pct_probability,
                "ml_ten_day_expected_net_return": expected_net_return,
                "rise_pattern_shape": pattern,
            }
        )
        signals.append(row)

    signals.sort(key=lambda row: (int(row["rank"]), str(row["ticker"])))
    ten_day_summary = {
        key: ten_day.get(key)
        for key in (
            "method",
            "adopted_shape",
            "adopted_shape_label",
            "minimum_turnover_yen",
            "validation",
            "validation_folds",
        )
    }
    ten_day_summary["live_signal_count"] = len(signals)

    industry_trends = source.get("industry_trends") or {}
    return {
        "schema_version": 1,
        "generated_at": source.get("generated_at"),
        "date": latest_date,
        "latest_date": latest_date,
        "update": source.get("update"),
        "metrics": source.get("metrics") or {},
        "market_regime": source.get("market_regime"),
        "industry_trends": {"sector_17": industry_trends.get("sector_17") or []},
        "technical_method": source.get("technical_method"),
        "signal_model": source.get("signal_model"),
        "ten_day_signal_study": ten_day_summary,
        "signal_count": len(signals),
        "signals": signals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.dashboard.read_text(encoding="utf-8"))
    payload = build_latest_signals(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
