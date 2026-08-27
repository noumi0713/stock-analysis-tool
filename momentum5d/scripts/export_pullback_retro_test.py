from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from export_close_signals import (
    PULLBACK_MAX_SIGNALS,
    _load_names,
    _pullback_condition_results,
    calculate_indicators,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--names", type=Path)
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    prices["_parsed_date"] = pd.to_datetime(prices["date"], errors="coerce").dt.date
    indicators = calculate_indicators(prices.drop(columns=["_parsed_date"]))
    names = _load_names(args.names)
    results: dict[str, object] = {}

    for requested in args.dates:
        signal_date = pd.Timestamp(requested).date()
        day = indicators.loc[indicators["date"].eq(signal_date)].copy()
        conditions = _pullback_condition_results(day)
        selected = day.loc[conditions.all(axis=1)].copy()
        selected["_pullback_score"] = (
            selected["close_position"] * 2
            + selected["volume_ratio_1_20"]
            - selected["distance_from_ma25"] * 25
        )
        ranked = selected.sort_values(
            ["_pullback_score", "ticker"], ascending=[False, True]
        ).head(PULLBACK_MAX_SIGNALS)

        history = prices.loc[prices["_parsed_date"].le(signal_date)]
        observation_counts = history.groupby("ticker", sort=False).size()
        diagnostics = {
            "maximum_observations_per_ticker": int(observation_counts.max()),
            "median_observations_per_ticker": float(observation_counts.median()),
            "ma75_available_count": int(day["ma75"].notna().sum()),
            "ma75_slope_10d_available_count": int(
                day["ma75_slope_10d"].notna().sum()
            ),
            "condition_pass_counts": {
                key: int(conditions[key].sum()) for key in conditions.columns
            },
        }

        records: list[dict[str, object]] = []
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            ticker = str(row["ticker"])
            code = str(row.get("code") or ticker.removesuffix(".T"))[:4]
            future = indicators.loc[
                indicators["ticker"].eq(ticker)
                & indicators["date"].gt(signal_date)
            ].sort_values("date").head(1)
            entry_date = None
            entry_gap = None
            entry_eligible = None
            if not future.empty:
                next_row = future.iloc[0]
                entry_date = next_row["date"].isoformat()
                entry_gap = float(next_row["_open"] / row["_close"] - 1.0)
                entry_eligible = -0.04 <= entry_gap <= 0.03

            records.append(
                {
                    "rank": rank,
                    "date": requested,
                    "code": code,
                    "ticker": ticker,
                    "name": names.get(code) or ticker,
                    "close": round(float(row["_close"]), 2),
                    "score": round(float(row["_pullback_score"]), 6),
                    "entry_date": entry_date,
                    "entry_gap": None if entry_gap is None else round(entry_gap, 6),
                    "entry_eligible": entry_eligible,
                }
            )

        results[requested] = {
            "ticker_count": int(day["ticker"].nunique()),
            "raw_qualifying_count": int(len(selected)),
            "signal_count": len(records),
            "diagnostics": diagnostics,
            "signals": records,
        }

    payload = {
        "source": "production_yahoo_cache",
        "dates": args.dates,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
