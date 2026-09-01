from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from app.market_data_contract import (
    DAILY_INTERVAL,
    DAILY_SOURCE,
    FINAL_SESSION,
    MARKET_TIMEZONE,
    daily_snapshot_fingerprint,
)
from app.storage.parquet import ParquetStore


def validate_ingestion_status(
    status: dict[str, object], *, expected_date: date
) -> str:
    expected_values = {
        "status": "complete",
        "source": DAILY_SOURCE,
        "session": FINAL_SESSION,
        "interval": DAILY_INTERVAL,
        "market_timezone": MARKET_TIMEZONE,
        "as_of": expected_date.isoformat(),
        "market_date": expected_date.isoformat(),
    }
    for key, expected in expected_values.items():
        if status.get(key) != expected:
            raise ValueError(
                f"取得ステータスが不一致です: {key}={status.get(key)!r} "
                f"expected={expected!r}"
            )
    if status.get("intraday") is not None:
        raise ValueError("確定日足の取得ステータスに場中データが混在しています")
    if status.get("failed_batches"):
        raise ValueError("確定日足の取得に失敗バッチがあります")
    acquired_at = str(status.get("updated_at") or "")
    try:
        parsed = datetime.fromisoformat(acquired_at)
    except ValueError as exc:
        raise ValueError("取得時刻 updated_at が不正です") from exc
    if parsed.tzinfo is None:
        raise ValueError("取得時刻 updated_at にタイムゾーンがありません")
    return acquired_at


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


def remove_non_daily_close_rows(
    prices: pd.DataFrame,
    *,
    expected_date: date,
) -> tuple[pd.DataFrame, list[str]]:
    """Remove stale intraday overlays that could not be replaced by a daily bar."""

    if not {"date", "ticker", "source"}.issubset(prices.columns):
        return prices.copy(), []
    dates = pd.to_datetime(prices["date"], errors="coerce").dt.date
    stale = dates.eq(expected_date) & prices["source"].astype(str).ne(DAILY_SOURCE)
    removed = sorted(prices.loc[stale, "ticker"].astype(str).unique())
    return prices.loc[~stale].copy(), removed


def validate_close_snapshot(
    prices: pd.DataFrame,
    *,
    expected_date: date,
    expected_tickers: set[str],
    minimum_coverage: float = 0.95,
    acquired_at: str | None = None,
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
        "stock_splits",
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

    stamp = acquired_at or datetime.now(UTC).isoformat()
    return {
        "status": "certified",
        "market_date": expected_date.isoformat(),
        "source": DAILY_SOURCE,
        "session": FINAL_SESSION,
        "interval": DAILY_INTERVAL,
        "market_timezone": MARKET_TIMEZONE,
        "data_through": f"{expected_date.isoformat()}T15:30:00+09:00",
        "acquired_at": stamp,
        "successful_tickers": len(actual_tickers),
        "expected_tickers": len(expected_tickers),
        "coverage": coverage,
        "minimum_coverage": minimum_coverage,
        "rejected_sources": [],
        "snapshot_fingerprint": daily_snapshot_fingerprint(prices, expected_date),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="大引け用Yahoo日足の完全性を検証")
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--expected-date", type=date.fromisoformat, required=True)
    parser.add_argument("--tickers-file", type=Path, required=True)
    parser.add_argument("--ingestion-status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-coverage", type=float, default=0.95)
    parser.add_argument(
        "--remove-non-daily",
        action="store_true",
        help="当日に確定日足へ置換できなかった5分足行を候補母集団から除外する",
    )
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    excluded_tickers: list[str] = []
    if args.remove_non_daily:
        prices, excluded_tickers = remove_non_daily_close_rows(
            prices,
            expected_date=args.expected_date,
        )
        if excluded_tickers:
            ParquetStore._atomic_parquet(prices, args.prices)

    ingestion_status = json.loads(args.ingestion_status.read_text(encoding="utf-8"))
    acquired_at = validate_ingestion_status(
        ingestion_status,
        expected_date=args.expected_date,
    )
    result = validate_close_snapshot(
        prices,
        expected_date=args.expected_date,
        expected_tickers=load_expected_tickers(args.tickers_file),
        minimum_coverage=args.minimum_coverage,
        acquired_at=acquired_at,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(args.output)
    print(
        "Certified daily close: "
        f"date={result['market_date']} source={result['source']} "
        f"coverage={result['coverage']:.1%} "
        f"tickers={result['successful_tickers']}/{result['expected_tickers']} "
        f"excluded_non_daily={','.join(excluded_tickers) or '-'}"
    )


if __name__ == "__main__":
    main()
