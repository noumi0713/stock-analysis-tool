from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

import pandas as pd

DAILY_SOURCE = "yfinance"
MARKET_TIMEZONE = "Asia/Tokyo"
DAILY_INTERVAL = "1d"
FINAL_SESSION = "close"


class MarketDataContractError(ValueError):
    """Raised when a market snapshot is not safe for signal publication."""


def daily_snapshot_fingerprint(prices: pd.DataFrame, market_date: object) -> str:
    required = {
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
        "stock_splits",
    }
    missing = required.difference(prices.columns)
    if missing:
        raise MarketDataContractError(
            f"Snapshot fingerprint is missing columns: {sorted(missing)}"
        )
    target = pd.to_datetime(market_date, errors="raise").date()
    work = prices.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.date
    work = work.loc[work["date"].eq(target)].copy()
    if work.empty:
        raise MarketDataContractError(f"No rows exist for market_date={target}")
    columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "stock_splits",
        "source",
    ]
    work = work[columns].sort_values(["ticker"]).reset_index(drop=True)
    canonical = work.to_json(
        orient="records", date_format="iso", double_precision=15, force_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_market_certification(
    certification: dict[str, Any],
    *,
    prices: pd.DataFrame | None = None,
    expected_date: object | None = None,
) -> dict[str, Any]:
    required_values = {
        "status": "certified",
        "source": DAILY_SOURCE,
        "session": FINAL_SESSION,
        "interval": DAILY_INTERVAL,
        "market_timezone": MARKET_TIMEZONE,
    }
    for key, expected in required_values.items():
        if certification.get(key) != expected:
            raise MarketDataContractError(
                f"Market certification requires {key}={expected}"
            )
    market_date = str(certification.get("market_date") or "")
    if not market_date:
        raise MarketDataContractError("Market certification has no market_date")
    expected_data_through = f"{market_date}T15:30:00+09:00"
    if certification.get("data_through") != expected_data_through:
        raise MarketDataContractError(
            "Certification data_through is not the TSE daily close"
        )
    if expected_date is not None:
        expected = pd.to_datetime(expected_date, errors="raise").date().isoformat()
        if market_date != expected:
            raise MarketDataContractError(
                f"Certification date mismatch: expected={expected} actual={market_date}"
            )
    acquired_at = str(certification.get("acquired_at") or "")
    try:
        acquired = datetime.fromisoformat(acquired_at)
    except ValueError as exc:
        raise MarketDataContractError("Certification acquired_at is invalid") from exc
    if acquired.tzinfo is None:
        raise MarketDataContractError("Certification acquired_at must be timezone-aware")
    successful = int(certification.get("successful_tickers", 0))
    expected_tickers = int(certification.get("expected_tickers", 0))
    if successful < 1 or expected_tickers < 1 or successful > expected_tickers:
        raise MarketDataContractError("Certification ticker counts are invalid")
    coverage = float(certification.get("coverage", 0.0))
    if abs(coverage - successful / expected_tickers) > 1e-9:
        raise MarketDataContractError("Certification coverage does not match ticker counts")
    if coverage < float(
        certification.get("minimum_coverage", 1.0)
    ):
        raise MarketDataContractError("Certified snapshot coverage is insufficient")
    if certification.get("rejected_sources") != []:
        raise MarketDataContractError("Certified snapshot contains rejected sources")
    fingerprint = str(certification.get("snapshot_fingerprint") or "")
    if not fingerprint:
        raise MarketDataContractError("Certification has no snapshot_fingerprint")
    if prices is not None:
        actual = daily_snapshot_fingerprint(prices, market_date)
        if actual != fingerprint:
            raise MarketDataContractError(
                "Certified snapshot fingerprint does not match signal input"
            )
    return certification


def load_market_certification(path: object) -> dict[str, Any]:
    from pathlib import Path

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MarketDataContractError("Market certification must be a JSON object")
    return validate_market_certification(value)
