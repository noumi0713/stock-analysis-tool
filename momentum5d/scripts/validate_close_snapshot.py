from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

DAILY_SOURCE = "yfinance"


def _normalize_ticker(value: str) -> str:
    ticker = value.strip().upper()
    if ticker and "." not in ticker:
        ticker = f"{ticker}.T"
    return ticker


def load_expected_tickers(path: Path) -> set[str]:
    values = {
        _normalize_ticker(line.split(",", 1)[0])
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    values.discard("TICKER.T")
    values.discard("SYMBOL.T")
    if not values:
        raise ValueError("期待銘柄ファイルが空です")
    return values


def validate_close_snapshot(
    prices: pd.DataFrame,
    *,
    expected_date: date,
    expected_tickers: set[str],
    minimum_coverage: float = 0.95,
) -> dict[str, object]:
    required_columns = {
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
    }
    missing_columns = sorted(required_columns - set(prices.columns))
    if missing_columns:
        raise ValueError(f"終値検証に必要な列がありません: {','.join(missing_columns)}")

    work = prices.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.date
    work["ticker"] = work["ticker"].astype(str).map(_normalize_ticker)
    latest_date = work["date"].max()
    if latest_date != expected_date:
        raise ValueError(
            f"当日の日足が未確定です: expected={expected_date.isoformat()} latest={latest_date}"
        )

    latest = work.loc[work["date"] == expected_date].copy()
    duplicate = latest.duplicated("ticker", keep=False)
    if duplicate.any():
        tickers = ",".join(sorted(latest.loc[duplicate, "ticker"].unique())[:10])
        raise ValueError(f"当日の日足が重複しています: {tickers}")

    non_daily = latest.loc[latest["source"].astype(str) != DAILY_SOURCE, "ticker"]
    if not non_daily.empty:
        tickers = ",".join(sorted(non_daily.unique())[:10])
        raise ValueError(
            "大引け値に5分足または不明な取得元が混在しています: "
            f"{tickers}"
        )

    numeric = latest[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    incomplete = numeric.isna().any(axis=1)
    if incomplete.any():
        tickers = ",".join(sorted(latest.loc[incomplete, "ticker"].unique())[:10])
        raise ValueError(f"当日の日足に欠損があります: {tickers}")

    actual_tickers = set(latest["ticker"]) & expected_tickers
    coverage = len(actual_tickers) / len(expected_tickers)
    if coverage < minimum_coverage:
        raise ValueError(
            "当日の日足カバレッジが不足しています: "
            f"{len(actual_tickers)}/{len(expected_tickers)} ({coverage:.1%}) "
            f"required>={minimum_coverage:.1%}"
        )

    return {
        "market_date": expected_date.isoformat(),
        "source": DAILY_SOURCE,
        "successful_tickers": len(actual_tickers),
        "expected_tickers": len(expected_tickers),
        "coverage": coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="大引け用Yahoo日足の完全性を検証")
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--expected-date", type=date.fromisoformat, required=True)
    parser.add_argument("--tickers-file", type=Path, required=True)
    parser.add_argument("--minimum-coverage", type=float, default=0.95)
    args = parser.parse_args()

    result = validate_close_snapshot(
        pd.read_parquet(args.prices),
        expected_date=args.expected_date,
        expected_tickers=load_expected_tickers(args.tickers_file),
        minimum_coverage=args.minimum_coverage,
    )
    print(
        "Certified daily close: "
        f"date={result['market_date']} source={result['source']} "
        f"coverage={result['coverage']:.1%} "
        f"tickers={result['successful_tickers']}/{result['expected_tickers']}"
    )


if __name__ == "__main__":
    main()
