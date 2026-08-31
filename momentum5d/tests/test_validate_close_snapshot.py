from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from scripts.validate_close_snapshot import (
    remove_non_daily_close_rows,
    validate_close_snapshot,
)


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


def test_removes_unconfirmed_intraday_rows_before_coverage_check() -> None:
    prices = pd.concat(
        [
            _prices(),
            _prices(source="yfinance_intraday_5m", include_second=False).assign(
                ticker="1003.T"
            ),
        ],
        ignore_index=True,
    )

    cleaned, removed = remove_non_daily_close_rows(
        prices,
        expected_date=date(2026, 8, 31),
    )
    result = validate_close_snapshot(
        cleaned,
        expected_date=date(2026, 8, 31),
        expected_tickers={"1001.T", "1002.T", "1003.T"},
        minimum_coverage=0.65,
    )

    assert removed == ["1003.T"]
    assert set(cleaned["ticker"]) == {"1001.T", "1002.T"}
    assert result["coverage"] == pytest.approx(2 / 3)


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
