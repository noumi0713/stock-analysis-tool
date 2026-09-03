from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yfinance as yf


def fetch_one(row: dict[str, str], retries: int = 2) -> dict[str, str]:
    ticker = (row.get("yahoo_ticker") or "").strip()
    base = dict(row)
    if not ticker:
        return {**base, "profile_status": "no_ticker", "profile_error": "missing ticker"}

    last_error = ""
    for attempt in range(retries + 1):
        try:
            info = yf.Ticker(ticker).get_info()
            quote_type = str(info.get("quoteType") or "")
            return {
                **base,
                "profile_status": "ok",
                "quote_type": quote_type,
                "long_name": str(info.get("longName") or info.get("shortName") or row.get("company_name") or ""),
                "sector": str(info.get("sector") or ""),
                "industry": str(info.get("industry") or ""),
                "website": str(info.get("website") or ""),
                "business_summary": str(info.get("longBusinessSummary") or ""),
                "market_cap": str(info.get("marketCap") or ""),
                "currency": str(info.get("currency") or ""),
                "profile_error": "",
            }
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1.0 + attempt * 1.5)
    return {
        **base,
        "profile_status": "error",
        "quote_type": "",
        "long_name": row.get("company_name", ""),
        "sector": "",
        "industry": "",
        "website": "",
        "business_summary": "",
        "market_cap": "",
        "currency": "",
        "profile_error": last_error,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dir", default="research/results/theme_relevance_batches")
    args = ap.parse_args()

    root = Path(args.dir)
    src = root / f"batch_{args.batch:03d}_stocks.csv"
    dst = root / f"batch_{args.batch:03d}_profiles.csv"
    with src.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    out: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(fetch_one, row): row for row in rows}
        for idx, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            out.append(result)
            print(idx, result.get("stock_code"), result.get("profile_status"), result.get("long_name"))

    order = {row["stock_code"]: i for i, row in enumerate(rows)}
    out.sort(key=lambda r: order.get(r.get("stock_code", ""), 10**9))
    fields = [
        "batch", "stock_code", "yahoo_ticker", "company_name", "theme_count",
        "profile_status", "quote_type", "long_name", "sector", "industry", "website",
        "business_summary", "market_cap", "currency", "profile_error",
    ]
    with dst.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)

    counts: dict[str, int] = {}
    for row in out:
        counts[row.get("profile_status", "unknown")] = counts.get(row.get("profile_status", "unknown"), 0) + 1
    print(json.dumps(counts, ensure_ascii=False))
    print(dst)


if __name__ == "__main__":
    main()
