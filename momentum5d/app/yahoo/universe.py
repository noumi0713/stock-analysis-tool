from __future__ import annotations

import argparse
from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

from app.point_in_time_universe import (
    DELISTING_URLS,
    NEW_LISTING_URLS,
    build_point_in_time_universe,
    download_listing_events,
)

JPX_LISTED_ISSUES_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)
TARGET_MARKETS = ("プライム", "スタンダード", "グロース")
SUPPLEMENTAL_MARKET_TICKERS = ("1306.T",)


def build_tse_universe(frame: pd.DataFrame) -> pd.DataFrame:
    """JPX上場銘柄一覧から東証の内国普通株式だけを抽出する。"""
    required = {
        "コード",
        "銘柄名",
        "市場・商品区分",
        "17業種コード",
        "33業種コード",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"JPX銘柄一覧の必須列がありません: {sorted(missing)}")

    work = frame.copy()
    market = work["市場・商品区分"].astype("string").fillna("")
    target = market.str.contains("内国株式", regex=False) & market.str.startswith(
        TARGET_MARKETS
    )
    work = work.loc[target, list(required)].copy()

    work["exchange_code"] = work["コード"].astype("string").str.strip().str.upper()
    work = work.loc[work["exchange_code"].str.fullmatch(r"[0-9A-Z]{4}", na=False)].copy()
    work["ticker"] = work["exchange_code"] + ".T"
    work["code"] = work["exchange_code"] + "0"
    work["company_name"] = work["銘柄名"].astype("string").str.strip()
    work["sector_17_code"] = (
        work["17業種コード"].astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    )
    work["sector_17_code"] = work["sector_17_code"].str.lstrip("0").replace("", pd.NA)
    work["sector_33_code"] = (
        work["33業種コード"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(4)
    )
    return (
        work[
            [
                "ticker",
                "code",
                "company_name",
                "sector_17_code",
                "sector_33_code",
            ]
        ]
        .drop_duplicates("ticker")
        .sort_values("ticker")
        .reset_index(drop=True)
    )


def download_tse_universe(url: str = JPX_LISTED_ISSUES_URL) -> pd.DataFrame:
    response = requests.get(
        url,
        timeout=45,
        headers={"User-Agent": "Momentum5D/1.0 personal-research"},
    )
    response.raise_for_status()
    listed = pd.read_excel(BytesIO(response.content), dtype="string", engine="xlrd")
    return build_tse_universe(listed)


def write_universe(universe: pd.DataFrame, config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    tickers = sorted(set(universe["ticker"]).union(SUPPLEMENTAL_MARKET_TICKERS))
    (config_dir / "prime_tickers.txt").write_text(
        "\n".join(tickers) + "\n",
        encoding="utf-8",
    )
    universe[["code", "company_name"]].to_csv(
        config_dir / "prime_names.csv",
        index=False,
        encoding="utf-8",
    )
    universe[["code", "sector_17_code", "sector_33_code"]].to_csv(
        config_dir / "prime_sectors.csv",
        index=False,
        encoding="utf-8",
    )


def write_point_in_time_universe(
    universe: pd.DataFrame,
    config_dir: Path,
    *,
    start: date,
    as_of: date,
) -> pd.DataFrame:
    new_listings = download_listing_events(NEW_LISTING_URLS, kind="new")
    delistings = download_listing_events(DELISTING_URLS, kind="delisted")
    history = build_point_in_time_universe(
        universe,
        new_listings,
        delistings,
        start=start,
        as_of=as_of,
    )
    history.to_csv(config_dir / "tse_universe_history.csv", index=False, encoding="utf-8")
    tickers = sorted(history.loc[~history["ticker_reused"], "ticker"].unique())
    (config_dir / "historical_tickers.txt").write_text(
        "\n".join(tickers) + "\n", encoding="utf-8"
    )
    return history


def main() -> int:
    parser = argparse.ArgumentParser(
        description="JPX公式一覧から東証全銘柄ユニバースを生成する"
    )
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--url", default=JPX_LISTED_ISSUES_URL)
    parser.add_argument("--history-start", type=date.fromisoformat)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()
    universe = download_tse_universe(args.url)
    write_universe(universe, args.config_dir)
    history_count = 0
    if args.history_start is not None:
        history = write_point_in_time_universe(
            universe,
            args.config_dir,
            start=args.history_start,
            as_of=args.as_of,
        )
        history_count = len(history)
    print(f"TSE universe: {len(universe):,} stocks history_intervals={history_count:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
