from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import pandas as pd

TARGET_MARKET_LABELS = ("プライム", "スタンダード", "グロース")
NEW_LISTING_URLS = (
    "https://www.jpx.co.jp/listing/stocks/new/",
    "https://www.jpx.co.jp/listing/stocks/new/00-archives-01.html",
    "https://www.jpx.co.jp/listing/stocks/new/00-archives-02.html",
    "https://www.jpx.co.jp/listing/stocks/new/00-archives-03.html",
)
DELISTING_URLS = (
    "https://www.jpx.co.jp/listing/stocks/delisted/",
    "https://www.jpx.co.jp/listing/stocks/delisted/archives-01.html",
    "https://www.jpx.co.jp/listing/stocks/delisted/archives-02.html",
    "https://www.jpx.co.jp/listing/stocks/delisted/archives-03.html",
)


def _date_prefix(value: object) -> pd.Timestamp:
    text = str(value).strip()[:10]
    return pd.to_datetime(text, format="%Y/%m/%d", errors="coerce")


def parse_new_listing_table(table: pd.DataFrame) -> pd.DataFrame:
    """Normalize JPX new-listing tables whose code and market occupy paired rows."""

    if table.shape[1] < 3:
        raise ValueError("JPX new-listing table has fewer than three columns")
    work = pd.DataFrame(
        {
            "listing_date": table.iloc[:, 0].map(_date_prefix),
            "company_name": table.iloc[:, 1].astype("string").str.strip(),
            "code_or_market": table.iloc[:, 2].astype("string").str.strip(),
        }
    ).dropna(subset=["listing_date", "company_name"])

    records: list[dict[str, object]] = []
    for (listing_date, company_name), rows in work.groupby(
        ["listing_date", "company_name"], sort=False, dropna=False
    ):
        values = rows["code_or_market"].dropna().astype(str).tolist()
        codes = [value for value in values if pd.Series([value]).str.fullmatch(r"[0-9A-Z]{4}")[0]]
        markets = [
            value
            for value in values
            if any(label in value for label in TARGET_MARKET_LABELS)
        ]
        if not codes or not markets:
            continue
        records.append(
            {
                "ticker": f"{codes[0]}.T",
                "company_name": str(company_name),
                "market": markets[0],
                "valid_from": pd.Timestamp(listing_date).normalize(),
            }
        )
    return pd.DataFrame.from_records(
        records, columns=["ticker", "company_name", "market", "valid_from"]
    )


def parse_delisting_table(table: pd.DataFrame) -> pd.DataFrame:
    required = {"上場廃止日", "銘柄名", "コード", "市場区分"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"JPX delisting table is missing columns: {sorted(missing)}")
    work = table.copy()
    work["valid_to"] = work["上場廃止日"].map(_date_prefix)
    work["exchange_code"] = work["コード"].astype("string").str.strip().str.upper()
    work = work.loc[
        work["exchange_code"].str.fullmatch(r"[0-9A-Z]{4}", na=False)
        & work["市場区分"].astype(str).str.contains("|".join(TARGET_MARKET_LABELS))
        & work["valid_to"].notna()
    ].copy()
    return pd.DataFrame(
        {
            "ticker": work["exchange_code"] + ".T",
            "company_name": work["銘柄名"].astype("string").str.strip(),
            "market": work["市場区分"].astype("string").str.strip(),
            "valid_to": work["valid_to"].dt.normalize(),
        }
    ).drop_duplicates(["ticker", "valid_to"])


def download_listing_events(
    urls: Iterable[str], *, kind: str
) -> pd.DataFrame:
    parser = parse_new_listing_table if kind == "new" else parse_delisting_table
    frames = [parser(pd.read_html(url)[0]) for url in urls]
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def build_point_in_time_universe(
    current_universe: pd.DataFrame,
    new_listings: pd.DataFrame,
    delistings: pd.DataFrame,
    *,
    start: date,
    as_of: date,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    as_of_ts = pd.Timestamp(as_of)
    listing_dates = (
        new_listings.loc[new_listings["valid_from"].le(as_of_ts)]
        .sort_values("valid_from")
        .drop_duplicates("ticker", keep="first")
        .set_index("ticker")["valid_from"]
        .to_dict()
    )

    records: list[dict[str, object]] = []
    for row in current_universe.itertuples(index=False):
        ticker = str(row.ticker)
        records.append(
            {
                "ticker": ticker,
                "company_name": str(row.company_name),
                "market": "current_tse",
                "valid_from": max(pd.Timestamp(listing_dates.get(ticker, start_ts)), start_ts),
                "valid_to": pd.NaT,
                "is_current": True,
                "source": "jpx_current_list",
            }
        )

    eligible_delistings = delistings.loc[
        delistings["valid_to"].between(start_ts, as_of_ts)
    ]
    for row in eligible_delistings.itertuples(index=False):
        ticker = str(row.ticker)
        records.append(
            {
                "ticker": ticker,
                "company_name": str(row.company_name),
                "market": str(row.market),
                "valid_from": max(pd.Timestamp(listing_dates.get(ticker, start_ts)), start_ts),
                "valid_to": pd.Timestamp(row.valid_to),
                "is_current": False,
                "source": "jpx_delisting_archive",
            }
        )

    history = pd.DataFrame.from_records(records)
    history = history.loc[
        history["valid_to"].isna() | history["valid_from"].le(history["valid_to"])
    ].copy()
    interval_counts = history.groupby("ticker")["valid_from"].transform("size")
    history["ticker_reused"] = interval_counts.gt(1)
    return history.sort_values(["ticker", "valid_from"]).reset_index(drop=True)


def filter_prices_by_point_in_time_universe(
    prices: pd.DataFrame, history: pd.DataFrame
) -> pd.DataFrame:
    required = {"ticker", "valid_from", "valid_to", "ticker_reused"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Universe history is missing columns: {sorted(missing)}")
    safe_history = history.loc[~history["ticker_reused"].astype(bool)].copy()
    safe_history["valid_from"] = pd.to_datetime(safe_history["valid_from"])
    safe_history["valid_to"] = pd.to_datetime(safe_history["valid_to"])

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    merged = frame.merge(
        safe_history[["ticker", "valid_from", "valid_to"]], on="ticker", how="inner"
    )
    eligible = merged["date"].ge(merged["valid_from"]) & (
        merged["valid_to"].isna() | merged["date"].le(merged["valid_to"])
    )
    return merged.loc[eligible, frame.columns].drop_duplicates(["ticker", "date"])
