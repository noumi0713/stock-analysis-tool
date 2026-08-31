from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.live_backtest import run_strategy_backtest
from app.point_in_time_universe import filter_prices_by_point_in_time_universe

SIGNAL_TYPES = ("capitulation_reversal", "first_pullback")
INPUT_MODES = (
    "raw_ohlcv_with_adjusted_close",
    "legacy_preadjusted_prices_raw_volume",
)


def audit_input_data(
    prices: pd.DataFrame, history: pd.DataFrame, *, input_mode: str
) -> dict[str, object]:
    required = {"ticker", "date", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(prices.columns))
    if missing:
        raise ValueError(f"Price data is missing columns: {missing}")
    if history.empty:
        raise ValueError("Point-in-time universe history is empty")
    if input_mode not in INPUT_MODES:
        raise ValueError(f"Unsupported input mode: {input_mode}")

    warnings: list[str] = []
    certified = input_mode == "raw_ohlcv_with_adjusted_close"
    if certified and "adjusted_close" not in prices.columns:
        raise ValueError("Certified input requires adjusted_close alongside raw OHLCV")
    if not certified:
        warnings.append(
            "Legacy shards contain pre-adjusted prices and raw volume; split-volume "
            "correction cannot be certified. Results are provisional only."
        )
    return {
        "status": "certified" if certified else "provisional",
        "input_mode": input_mode,
        "split_adjusted_volume": certified,
        "point_in_time_universe": True,
        "price_rows_before_universe_filter": len(prices),
        "price_start": str(pd.to_datetime(prices["date"]).min().date()),
        "price_end": str(pd.to_datetime(prices["date"]).max().date()),
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run both frozen live signals through one execution engine"
    )
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--universe-history", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-mode", choices=INPUT_MODES, required=True)
    args = parser.parse_args()

    prices = pd.read_parquet(args.prices)
    history = pd.read_csv(args.universe_history)
    audit = audit_input_data(prices, history, input_mode=args.input_mode)
    prices = filter_prices_by_point_in_time_universe(prices, history)
    audit["price_rows_after_universe_filter"] = len(prices)
    audit["eligible_tickers"] = int(prices["ticker"].nunique())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for signal_type in SIGNAL_TYPES:
        result = run_strategy_backtest(prices, signal_type=signal_type)
        summaries[signal_type] = result["summary"]
        result["candidate_paths"].to_csv(
            args.output_dir / f"{signal_type}_candidate_paths.csv", index=False
        )
        result["trades"].to_csv(
            args.output_dir / f"{signal_type}_trades.csv", index=False
        )
        result["equity_curve"].to_csv(
            args.output_dir / f"{signal_type}_equity_curve.csv", index=False
        )
    payload = {
        "status": f"completed_{audit['status']}",
        "universe_mode": "jpx_point_in_time",
        "signals_evaluated_separately": True,
        "input_data_audit": audit,
        "summaries": summaries,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
