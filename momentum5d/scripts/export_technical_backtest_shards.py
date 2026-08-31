from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.adjustments import price_adjustment_factor, split_adjusted_volume
from app.point_in_time_universe import filter_prices_by_point_in_time_universe


def _stock_metadata(source: dict[str, Any], tickers: list[str]) -> dict[str, dict[str, str]]:
    source_rows = {
        str(row.get("ticker") or ""): row for row in source.get("stocks") or []
    }
    result: dict[str, dict[str, str]] = {}
    for ticker in tickers:
        row = source_rows.get(ticker) or {}
        code = str(row.get("code") or ticker.removesuffix(".T"))[:4]
        result[ticker] = {
            "code": code,
            "name": str(row.get("company_name") or row.get("name") or ticker),
            "sector": str(
                row.get("sector_17_name") or row.get("sector") or "その他"
            ),
        }
    return result


def export_shards(
    source: dict[str, Any],
    prices: pd.DataFrame,
    output_dir: Path,
    *,
    lookback_years: int = 3,
    universe_history: pd.DataFrame | None = None,
) -> dict[str, Any]:
    latest = pd.Timestamp(str(source["latest_date"])).normalize()
    start = (latest - pd.DateOffset(years=lookback_years)).normalize()

    frame = prices.copy()
    universe_mode = "current_snapshot_unverified"
    if universe_history is not None:
        frame = filter_prices_by_point_in_time_universe(frame, universe_history)
        universe_mode = "jpx_point_in_time"
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame = frame.loc[
        frame["date"].notna()
        & frame["date"].ge(start)
        & frame["date"].le(latest)
    ].copy()
    if frame.empty:
        raise ValueError(f"No prices are available from {start.date()} to {latest.date()}")
    frame["ticker"] = frame["ticker"].astype(str)
    frame["adjustment_factor"] = price_adjustment_factor(
        frame["close"], frame.get("adjusted_close", frame["close"])
    )
    frame["adjusted_volume"] = split_adjusted_volume(
        frame["volume"], frame["adjustment_factor"]
    )
    frame["month"] = frame["date"].dt.strftime("%Y-%m")

    dates = sorted(frame["date"].dt.strftime("%Y-%m-%d").unique().tolist())
    date_index = {value: index for index, value in enumerate(dates)}
    tickers = sorted(frame["ticker"].unique().tolist())
    stocks = _stock_metadata(source, tickers)

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("*.json"):
        old_file.unlink()

    shards: list[dict[str, Any]] = []
    for month, month_rows in frame.groupby("month", sort=True):
        bars: dict[str, list[list[float | int]]] = {}
        row_count = 0
        for ticker, rows in month_rows.groupby("ticker", sort=False):
            encoded: list[list[float | int]] = []
            for row in rows.sort_values("date").itertuples(index=False):
                close = float(row.close)
                adjusted_close = float(getattr(row, "adjusted_close", close))
                ratio = float(row.adjustment_factor)
                encoded.append(
                    [
                        date_index[row.date.strftime("%Y-%m-%d")],
                        round(float(row.open) * ratio, 2),
                        round(float(row.high) * ratio, 2),
                        round(float(row.low) * ratio, 2),
                        round(adjusted_close, 2),
                        int(round(float(row.adjusted_volume))),
                    ]
                )
            bars[str(ticker)] = encoded
            row_count += len(encoded)

        filename = f"{month}.json"
        (output_dir / filename).write_text(
            json.dumps(
                {"month": month, "bars": bars},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        shards.append({"month": month, "path": filename, "rowCount": row_count})

    manifest = {
        "meta": {
            "generatedAt": source.get("generated_at"),
            "startDate": dates[0],
            "endDate": dates[-1],
            "lookbackYears": lookback_years,
            "stockCount": len(stocks),
            "dateCount": len(dates),
            "shardCount": len(shards),
            "dataScope": "all-tse",
            "universeMode": universe_mode,
        },
        "dates": dates,
        "stocks": stocks,
        "shards": shards,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lookback-years", type=int, default=3)
    parser.add_argument("--universe-history", type=Path)
    args = parser.parse_args()

    source = json.loads(args.dashboard.read_text(encoding="utf-8"))
    export_shards(
        source,
        pd.read_parquet(args.prices),
        args.output_dir,
        lookback_years=args.lookback_years,
        universe_history=(
            pd.read_csv(args.universe_history)
            if args.universe_history is not None
            else None
        ),
    )


if __name__ == "__main__":
    main()
