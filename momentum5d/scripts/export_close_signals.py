from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RSI_PERIOD = 14
RSI_MIN = 25.0
RSI_MAX = 35.0
RETURN_1D_MIN = -0.03
RETURN_1D_MAX = 0.0
RETURN_5D_MIN = -0.15
RETURN_5D_MAX = 0.02
VOLUME_RATIO_MIN = 1.5
TURNOVER_MIN = 300_000_000.0
ATR_MIN = 0.02
ATR_MAX = 0.12
MAX_SIGNALS = 3

# These are descriptive reference values from the frozen three-year comparison.
# They are not used to decide whether a stock passes the technical screen.
REFERENCE_TARGET_PROBABILITY = 203 / 378
REFERENCE_DOWN_5_PROBABILITY = 0.31876606683804626
REFERENCE_DOWN_8_PROBABILITY = 0.16195372750642673
REFERENCE_EXPECTED_NET_RETURN = 0.01732925797149239


def _rolling_sum(values: pd.Series, period: int) -> pd.Series:
    return values.rolling(period, min_periods=period).sum()


def _site_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Match the simple rolling RSI used by Technical Backtest Lab."""

    delta = close.diff()
    up = delta.where(delta > 0, 0.0)
    down = (-delta).where(delta < 0, 0.0)
    up_sum = _rolling_sum(up, period)
    down_sum = _rolling_sum(down, period)
    ratio = up_sum.div(down_sum.replace(0.0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + ratio))
    rsi = rsi.where(down_sum.ne(0.0), 100.0)
    rsi.iloc[:period] = np.nan
    return rsi


def calculate_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    required = {
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    }
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Price data is missing columns: {sorted(missing)}")

    work = prices.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.date
    work = work.dropna(subset=["date", "ticker"]).sort_values(["ticker", "date"])
    raw_close = pd.to_numeric(work["close"], errors="coerce")
    adjusted_close = pd.to_numeric(work["adjusted_close"], errors="coerce")
    adjustment = adjusted_close.div(raw_close.replace(0.0, np.nan)).fillna(1.0)
    work["_close"] = adjusted_close
    work["_open"] = pd.to_numeric(work["open"], errors="coerce") * adjustment
    work["_high"] = pd.to_numeric(work["high"], errors="coerce") * adjustment
    work["_low"] = pd.to_numeric(work["low"], errors="coerce") * adjustment
    work["_volume"] = pd.to_numeric(work["volume"], errors="coerce")

    groups: list[pd.DataFrame] = []
    for _, stock in work.groupby("ticker", sort=False):
        stock = stock.copy()
        close = stock["_close"]
        volume = stock["_volume"]
        previous_close = close.shift(1)
        true_range = pd.concat(
            [
                stock["_high"] - stock["_low"],
                (stock["_high"] - previous_close).abs(),
                (stock["_low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        stock["RSI"] = _site_rsi(close)
        stock["return_1d"] = close.pct_change(fill_method=None)
        stock["return_5d"] = close.pct_change(5, fill_method=None)
        stock["volume_ratio_1_20"] = volume.div(volume.rolling(20, min_periods=20).mean())
        stock["trading_value"] = close * volume
        stock["ATR"] = true_range.rolling(14, min_periods=14).sum().div(14).div(close)
        stock["ma25"] = close.rolling(25, min_periods=25).mean()
        groups.append(stock)
    return pd.concat(groups, ignore_index=True) if groups else work.iloc[0:0].copy()


def select_latest_signals(indicators: pd.DataFrame) -> pd.DataFrame:
    if indicators.empty:
        return indicators.copy()
    latest_date = indicators["date"].max()
    latest = indicators.loc[indicators["date"].eq(latest_date)].copy()
    mask = (
        latest["RSI"].between(RSI_MIN, RSI_MAX)
        & latest["return_1d"].between(RETURN_1D_MIN, RETURN_1D_MAX)
        & latest["return_5d"].between(RETURN_5D_MIN, RETURN_5D_MAX)
        & latest["volume_ratio_1_20"].ge(VOLUME_RATIO_MIN)
        & latest["trading_value"].ge(TURNOVER_MIN)
        & latest["ATR"].between(ATR_MIN, ATR_MAX)
        & latest["_close"].lt(latest["ma25"])
        & latest["_close"].gt(latest["_open"])
    )
    return (
        latest.loc[mask]
        .sort_values(["volume_ratio_1_20", "ticker"], ascending=[False, True])
        .head(MAX_SIGNALS)
        .reset_index(drop=True)
    )


def _load_names(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    names = pd.read_csv(path, dtype="string")
    return {
        str(row.code).removesuffix("0"): str(row.company_name)
        for row in names.itertuples(index=False)
        if pd.notna(row.code) and pd.notna(row.company_name)
    }


def build_payload(
    prices: pd.DataFrame,
    *,
    names: dict[str, str] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    indicators = calculate_indicators(prices)
    if indicators.empty:
        raise ValueError("No usable daily prices were found")
    latest_date = indicators["date"].max().isoformat()
    latest_rows = indicators.loc[indicators["date"].astype(str).eq(latest_date)]
    signals = select_latest_signals(indicators)
    company_names = names or {}
    records: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(signals.iterrows(), start=1):
        ticker = str(row["ticker"])
        code = str(row.get("code") or ticker.removesuffix(".T"))[:4]
        records.append(
            {
                "date": latest_date,
                "code": code,
                "ticker": ticker,
                "name": company_names.get(code) or ticker,
                "rank": position,
                "signal": "capitulation_reversal",
                "pattern": "投げ売り反転",
                "close": round(float(row["_close"]), 2),
                "trading_value": round(float(row["trading_value"])),
                "turnover_value": round(float(row["trading_value"])),
                "RSI": round(float(row["RSI"]), 4),
                "ATR": round(float(row["ATR"]), 6),
                "return_1d": round(float(row["return_1d"]), 6),
                "return_5d": round(float(row["return_5d"]), 6),
                "volume_ratio_1_20": round(float(row["volume_ratio_1_20"]), 6),
                "target_probability": REFERENCE_TARGET_PROBABILITY,
                "down_5pct_probability": REFERENCE_DOWN_5_PROBABILITY,
                "down_8pct_probability": REFERENCE_DOWN_8_PROBABILITY,
                "expected_net_return": REFERENCE_EXPECTED_NET_RETURN,
                "entry_rule": "翌営業日始値",
                "take_profit_pct": 0.05,
                "stop_loss_pct": -0.12,
                "holding_days": 10,
            }
        )

    stamp = generated_at or datetime.now(UTC).isoformat()
    ticker_count = int(latest_rows["ticker"].nunique())
    return {
        "schema_version": 2,
        "generated_at": stamp,
        "date": latest_date,
        "latest_date": latest_date,
        "next_session": "翌営業日",
        "update": {
            "status": "complete",
            "session": "close",
            "session_label": "大引け",
            "market_date": latest_date,
            "data_through": f"{latest_date}T15:30:00+09:00",
            "interval": "1d",
            "successful_tickers": ticker_count,
            "coverage": 1.0,
            "generated_at": stamp,
        },
        "signal_model": {
            "key": "rsi14_three_day_frequency_10d_v2",
            "label": "RSI14投げ売り反転10D",
            "conditions": {
                "rsi_period": RSI_PERIOD,
                "rsi_min": RSI_MIN,
                "rsi_max": RSI_MAX,
                "return_1d_min": RETURN_1D_MIN,
                "return_1d_max": RETURN_1D_MAX,
                "return_5d_min": RETURN_5D_MIN,
                "return_5d_max": RETURN_5D_MAX,
                "volume_ratio_min": VOLUME_RATIO_MIN,
                "minimum_turnover_yen": int(TURNOVER_MIN),
                "atr_14_pct_min": ATR_MIN,
                "atr_14_pct_max": ATR_MAX,
                "ma25": "below",
                "bullish": True,
                "maximum_candidates_per_day": MAX_SIGNALS,
                "entry_rule": "翌営業日始値",
                "take_profit_pct": 0.05,
                "stop_loss_pct": -0.12,
                "holding_days": 10,
            },
        },
        "signal_count": len(records),
        "signals": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="大引け後の翌営業日用シグナルを生成")
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--names", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(
        pd.read_parquet(args.prices),
        names=_load_names(args.names),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    print(
        f"Close signals: date={payload['date']} count={payload['signal_count']} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
