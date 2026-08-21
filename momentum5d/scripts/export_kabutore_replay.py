from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPLAY_YEAR = 2026
SIGNAL_SOURCE = "rsi14_three_day_frequency_10d_v1"


def _number(value: Any, digits: int = 6) -> float | None:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return round(float(parsed), digits)


def _probability(record: dict[str, Any], key: str, alias: str) -> float | None:
    return _number(record.get(key, record.get(alias)))


def _trusted_previous_signals(
    previous: dict[str, Any] | None,
    *,
    start_date: str,
    end_date: str,
) -> dict[str, list[dict]]:
    if not previous or (previous.get("meta") or {}).get("signalSource") != SIGNAL_SOURCE:
        return {}
    return {
        str(signal_date): list(records or [])
        for signal_date, records in (previous.get("signals") or {}).items()
        if start_date <= str(signal_date) <= end_date
    }


def build_replay_payload(
    source: dict[str, Any],
    prices: pd.DataFrame,
    previous: dict[str, Any] | None = None,
    start_date: str | None = None,
) -> dict[str, Any]:
    latest_date = str(source.get("latest_date") or "")
    range_start = start_date or f"{REPLAY_YEAR}-01-01"
    if not latest_date or range_start > latest_date:
        raise ValueError(f"Invalid replay range: {range_start} to {latest_date}")

    study = ((source.get("ten_day_signal_study") or {}).get("demo_trade_signal_study") or {})
    if study.get("status") != "completed":
        raise ValueError("RSI14 three-day signal study is unavailable")

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame = frame.loc[
        frame["date"].notna()
        & frame["date"].astype(str).ge(range_start)
        & frame["date"].astype(str).le(latest_date)
    ].copy()
    if frame.empty:
        raise ValueError(f"No market prices are available from {range_start}")
    frame["ticker"] = frame["ticker"].astype(str)
    dates = sorted(frame["date"].astype(str).unique().tolist())
    date_index = {value: index for index, value in enumerate(dates)}

    raw_signals = _trusted_previous_signals(
        previous,
        start_date=range_start,
        end_date=latest_date,
    )
    for record in study.get("historical_signals") or []:
        signal_date = str(record.get("signal_date") or "")
        if range_start <= signal_date <= latest_date:
            raw_signals.setdefault(signal_date, []).append(dict(record))
    # The latest market date is authoritative. Writing an empty list is essential:
    # it prevents a prior run's candidates from leaking into a zero-signal session.
    raw_signals[latest_date] = [dict(record) for record in study.get("live_signals") or []]

    maximum_signals = int(study.get("maximum_signals_per_day") or 3)
    for signal_date, records in raw_signals.items():
        deduplicated = {str(record.get("ticker") or ""): record for record in records}
        raw_signals[signal_date] = sorted(
            (record for ticker, record in deduplicated.items() if ticker),
            key=lambda record: (
                int(record.get("rank") or 9999),
                str(record.get("ticker") or ""),
            ),
        )[:maximum_signals]

    stock_rows = {
        str(row.get("ticker") or ""): row for row in source.get("stocks") or []
    }
    price_lookup = {
        (str(row.date), str(row.ticker)): row
        for row in frame.itertuples(index=False)
    }

    signals: dict[str, list[dict[str, Any]]] = {value: [] for value in dates}
    signal_tickers: set[str] = set()
    for signal_date in dates:
        normalized: list[dict[str, Any]] = []
        for fallback_rank, record in enumerate(raw_signals.get(signal_date, []), start=1):
            ticker = str(record.get("ticker") or "")
            if not ticker:
                continue
            stock = stock_rows.get(ticker) or {}
            price = price_lookup.get((signal_date, ticker))
            target_probability = _probability(
                record, "target_probability", "targetProbability"
            )
            down_5pct_probability = _probability(
                record, "down_5pct_probability", "down5Probability"
            )
            down_8pct_probability = _probability(
                record, "down_8pct_probability", "down8Probability"
            )
            expected_net_return = _probability(
                record, "expected_net_return", "expectedReturn"
            )
            close = _number(record.get("signal_close_yen", record.get("close")), 2)
            if close is None and price is not None:
                close = _number(getattr(price, "adjusted_close", price.close), 2)
            turnover = _number(
                record.get("daily_turnover_yen", record.get("turnover")), 0
            )
            if turnover is None and price is not None and close is not None:
                turnover = round(close * float(price.volume))
            rank = int(record.get("rank") or fallback_rank)
            code = str(stock.get("code") or ticker.removesuffix(".T"))
            normalized.append(
                {
                    "ticker": ticker,
                    "code": code[:4],
                    "name": stock.get("company_name") or stock.get("name") or ticker,
                    "sector": stock.get("sector_17_name") or stock.get("sector") or "その他",
                    "rank": rank,
                    "close": close or 0.0,
                    "return1d": (
                        _number(stock.get("return_1d"))
                        if signal_date == latest_date
                        else None
                    ),
                    "turnover": turnover or 0.0,
                    "volume": int(getattr(price, "volume", 0) or 0),
                    "reason": record.get("reason")
                    or "投げ売り反転・RSI14平均3営業日検知条件を通過",
                    "targetProbability": target_probability,
                    "down5Probability": down_5pct_probability,
                    "down8Probability": down_8pct_probability,
                    "expectedReturn": expected_net_return,
                    "limitPrice": _number(record.get("limit_price_yen"), 2),
                    "entryRule": record.get("entry_rule") or "翌営業日始値",
                    "score": round((target_probability or 0.0) * 100),
                }
            )
            signal_tickers.add(ticker)
        signals[signal_date] = normalized

    stocks: dict[str, dict[str, str]] = {}
    bars: dict[str, list[list[float | int]]] = {}
    for ticker, rows in frame.loc[frame["ticker"].isin(signal_tickers)].groupby(
        "ticker", sort=False
    ):
        stock = stock_rows.get(str(ticker)) or {}
        code = str(stock.get("code") or str(ticker).removesuffix(".T"))
        stocks[str(ticker)] = {
            "code": code[:4],
            "name": str(stock.get("company_name") or stock.get("name") or ticker),
            "sector": str(stock.get("sector_17_name") or stock.get("sector") or "その他"),
        }
        encoded: list[list[float | int]] = []
        for row in rows.sort_values("date").itertuples(index=False):
            close = float(row.close)
            adjusted_close = float(getattr(row, "adjusted_close", close))
            ratio = adjusted_close / close if close else 1.0
            encoded.append(
                [
                    date_index[str(row.date)],
                    round(float(row.open) * ratio, 2),
                    round(float(row.high) * ratio, 2),
                    round(float(row.low) * ratio, 2),
                    round(adjusted_close, 2),
                    int(row.volume),
                ]
            )
        bars[str(ticker)] = encoded

    conditions = (source.get("signal_model") or {}).get("conditions") or {}
    return {
        "meta": {
            "generatedAt": source.get("generated_at"),
            "startDate": dates[0],
            "endDate": latest_date,
            "lastStartDate": latest_date,
            "initialCash": 2_000_000,
            "maxHoldings": None,
            "lotSize": 100,
            "rankingSize": maximum_signals,
            "signalVersion": str(
                (source.get("signal_model") or {}).get("label")
                or "RSI14平均3営業日検知 Momentum10D"
            ),
            "signalSource": SIGNAL_SOURCE,
            "minimumTurnover": int(conditions.get("minimum_turnover_yen") or 300_000_000),
            "minimumProbability": conditions.get("minimum_probability"),
            "maximumDown5Probability": conditions.get("maximum_down_5pct_probability"),
            "maximumDown8Probability": conditions.get("maximum_down_8pct_probability"),
            "maxHoldingDays": 10,
            "targetReturn": 0.05,
            "stockCount": len(stocks),
            "latestSession": (source.get("update") or {}).get("session"),
            "latestSessionLabel": (source.get("update") or {}).get("session_label"),
        },
        "dates": dates,
        "stocks": stocks,
        "bars": bars,
        "signals": signals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lookback-years", type=int)
    args = parser.parse_args()

    source = json.loads(args.dashboard.read_text(encoding="utf-8"))
    previous = (
        json.loads(args.previous.read_text(encoding="utf-8"))
        if args.previous and args.previous.exists()
        else None
    )
    start_date = None
    if args.lookback_years:
        latest = pd.Timestamp(source["latest_date"])
        start_date = (latest - pd.DateOffset(years=args.lookback_years)).date().isoformat()
    payload = build_replay_payload(
        source,
        pd.read_parquet(args.prices),
        previous,
        start_date=start_date,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
