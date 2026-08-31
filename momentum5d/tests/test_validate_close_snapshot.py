from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from scripts.validate_close_snapshot import validate_close_snapshot


def _prices(*, source: str = "yfinance", include_second: bool = True) -> pd.DataFrame:
    rows = [
        {
            "date": date(2026, 8, 31),
            "ticker": "1001.T",
            "open": 100.0,
            "high": 110.0,
            "low": 99.0,
            "close": 108.0,
            "volume": 1_000,
            "source": source,
        }
    ]
    if include_second:
        rows.append({**rows[0], "ticker": "1002.T"})
    return pd.DataFrame(rows)


def test_accepts_complete_daily_snapshot() -> None:
    result = validate_close_snapshot(
        _prices(),
        expected_date=date(2026, 8, 31),
        expected_tickers={"1001.T", "1002.T"},
        minimum_coverage=1.0,
    )

    assert result["source"] == "yfinance"
    assert result["coverage"] == 1.0


def test_rejects_intraday_close_even_if_date_is_current() -> None:
    with pytest.raises(ValueError, match="5分足"):
        validate_close_snapshot(
            _prices(source="yfinance_intraday_5m"),
            expected_date=date(2026, 8, 31),
            expected_tickers={"1001.T", "1002.T"},
        )


def test_rejects_stale_daily_snapshot() -> None:
    prices = _prices()
    prices["date"] = date(2026, 8, 28)

    with pytest.raises(ValueError, match="未確定"):
        validate_close_snapshot(
            prices,
            expected_date=date(2026, 8, 31),
            expected_tickers={"1001.T", "1002.T"},
        )


def test_rejects_insufficient_daily_coverage() -> None:
    with pytest.raises(ValueError, match="カバレッジ"):
        validate_close_snapshot(
            _prices(include_second=False),
            expected_date=date(2026, 8, 31),
            expected_tickers={"1001.T", "1002.T"},
            minimum_coverage=0.75,
        )
