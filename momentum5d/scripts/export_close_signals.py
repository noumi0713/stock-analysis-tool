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
RETURN_5D_MIN = -0.12
RETURN_5D_MAX = -0.05
VOLUME_RATIO_MIN = 1.5
TURNOVER_MIN = 300_000_000.0
ATR_MIN = 0.005
ATR_MAX = 0.08
MAX_SIGNALS = 3
MAX_NEAR_MISSES = 5
TAKE_PROFIT_PCT = 0.235
STOP_LOSS_PCT = -0.22
HOLDING_DAYS = 10

CONDITION_KEYS = (
    "rsi",
    "return_1d",
    "return_5d",
    "volume_ratio_1_20",
    "trading_value",
    "atr_14_pct",
    "ma25",
    "bullish",
)

# These are descriptive reference values from the frozen three-year stable-score comparison.
# They are not used to decide whether a stock passes the technical screen.
REFERENCE_TARGET_PROBABILITY = 0.5612903225806452
REFERENCE_DOWN_5_PROBABILITY = 0.23870967741935484
REFERENCE_DOWN_8_PROBABILITY = 0.10967741935483871
REFERENCE_EXPECTED_NET_RETURN = 0.04109912872314453


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


def _condition_results(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rsi": frame["RSI"].between(RSI_MIN, RSI_MAX),
            "return_1d": frame["return_1d"].between(
                RETURN_1D_MIN, RETURN_1D_MAX
            ),
            "return_5d": frame["return_5d"].between(
                RETURN_5D_MIN, RETURN_5D_MAX
            ),
            "volume_ratio_1_20": frame["volume_ratio_1_20"].ge(VOLUME_RATIO_MIN),
            "trading_value": frame["trading_value"].ge(TURNOVER_MIN),
            "atr_14_pct": frame["ATR"].between(ATR_MIN, ATR_MAX),
            "ma25": frame["_close"].lt(frame["ma25"]),
            "bullish": frame["_close"].gt(frame["_open"]),
        },
        index=frame.index,
    )


def _latest_rows(indicators: pd.DataFrame) -> pd.DataFrame:
    if indicators.empty:
        return indicators.copy()
    latest_date = indicators["date"].max()
    return indicators.loc[indicators["date"].eq(latest_date)].copy()


def select_latest_signals(indicators: pd.DataFrame) -> pd.DataFrame:
    latest = _latest_rows(indicators)
    if latest.empty:
        return latest
    mask = _condition_results(latest).all(axis=1)
    return (
        latest.loc[mask]
        .sort_values(["volume_ratio_1_20", "ticker"], ascending=[False, True])
        .head(MAX_SIGNALS)
        .reset_index(drop=True)
    )


def _miss_distance(row: pd.Series, condition: str) -> float:
    if condition == "rsi":
        return min(abs(float(row["RSI"]) - RSI_MIN), abs(float(row["RSI"]) - RSI_MAX)) / (
            RSI_MAX - RSI_MIN
        )
    if condition == "return_1d":
        value = float(row["return_1d"])
        gap = RETURN_1D_MIN - value if value < RETURN_1D_MIN else value - RETURN_1D_MAX
        return gap / (RETURN_1D_MAX - RETURN_1D_MIN)
    if condition == "return_5d":
        value = float(row["return_5d"])
        gap = RETURN_5D_MIN - value if value < RETURN_5D_MIN else value - RETURN_5D_MAX
        return gap / (RETURN_5D_MAX - RETURN_5D_MIN)
    if condition == "volume_ratio_1_20":
        return (VOLUME_RATIO_MIN - float(row["volume_ratio_1_20"])) / VOLUME_RATIO_MIN
    if condition == "trading_value":
        return (TURNOVER_MIN - float(row["trading_value"])) / TURNOVER_MIN
    if condition == "atr_14_pct":
        value = float(row["ATR"])
        gap = ATR_MIN - value if value < ATR_MIN else value - ATR_MAX
        return gap / (ATR_MAX - ATR_MIN)
    if condition == "ma25":
        return max(float(row["_close"]) / float(row["ma25"]) - 1.0, 0.0) / 0.05
    atr_yen = max(float(row["_close"]) * float(row["ATR"]), 1.0)
    return max(float(row["_open"]) - float(row["_close"]), 0.0) / atr_yen


def select_latest_near_misses(indicators: pd.DataFrame) -> pd.DataFrame:
    latest = _latest_rows(indicators)
    if latest.empty:
        return latest
    results = _condition_results(latest)
    near = latest.loc[results.sum(axis=1).eq(len(CONDITION_KEYS) - 1)].copy()
    if near.empty:
        return near
    near["_failed_condition"] = [
        next(key for key in CONDITION_KEYS if not bool(results.at[index, key]))
        for index in near.index
    ]
    near["_miss_distance"] = [
        _miss_distance(row, str(row["_failed_condition"]))
        for _, row in near.iterrows()
    ]
    return (
        near.sort_values(
            ["_miss_distance", "volume_ratio_1_20", "ticker"],
            ascending=[True, False, True],
        )
        .head(MAX_NEAR_MISSES)
        .reset_index(drop=True)
    )


def _failed_condition_payload(row: pd.Series) -> dict[str, Any]:
    condition = str(row["_failed_condition"])
    if condition == "rsi":
        actual = float(row["RSI"])
        return {
            "key": condition,
            "label": "RSI14",
            "actual_value": actual,
            "actual_label": f"{actual:.1f}",
            "required_label": "25〜35",
        }
    if condition == "return_1d":
        actual = float(row["return_1d"])
        return {
            "key": condition,
            "label": "前日比",
            "actual_value": actual,
            "actual_label": f"{actual * 100:+.2f}%",
            "required_label": "−3〜0%",
        }
    if condition == "return_5d":
        actual = float(row["return_5d"])
        return {
            "key": condition,
            "label": "5日騰落",
            "actual_value": actual,
            "actual_label": f"{actual * 100:+.2f}%",
            "required_label": "−12〜−5%",
        }
    if condition == "volume_ratio_1_20":
        actual = float(row["volume_ratio_1_20"])
        return {
            "key": condition,
            "label": "出来高比",
            "actual_value": actual,
            "actual_label": f"{actual:.2f}倍",
            "required_label": "1.5倍以上",
        }
    if condition == "trading_value":
        actual = float(row["trading_value"])
        return {
            "key": condition,
            "label": "売買代金",
            "actual_value": actual,
            "actual_label": f"{actual / 100_000_000:.2f}億円",
            "required_label": "3億円以上",
        }
    if condition == "atr_14_pct":
        actual = float(row["ATR"])
        return {
            "key": condition,
            "label": "ATR14",
            "actual_value": actual,
            "actual_label": f"{actual * 100:.1f}%",
            "required_label": "0.5〜8%",
        }
    if condition == "ma25":
        close = float(row["_close"])
        ma25 = float(row["ma25"])
        return {
            "key": condition,
            "label": "25日線",
            "actual_value": close,
            "actual_label": f"終値 {close:,.0f} / 25日線 {ma25:,.0f}",
            "required_label": "終値が25日線より下",
        }
    bullish = float(row["_close"]) > float(row["_open"])
    return {
        "key": condition,
        "label": "ローソク",
        "actual_value": bullish,
        "actual_label": "陽線" if bullish else "陰線または同値",
        "required_label": "陽線",
    }


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
    near_misses = select_latest_near_misses(indicators)
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
                "take_profit_pct": TAKE_PROFIT_PCT,
                "stop_loss_pct": STOP_LOSS_PCT,
                "holding_days": HOLDING_DAYS,
            }
        )

    near_miss_records: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(near_misses.iterrows(), start=1):
        ticker = str(row["ticker"])
        code = str(row.get("code") or ticker.removesuffix(".T"))[:4]
        near_miss_records.append(
            {
                "date": latest_date,
                "code": code,
                "ticker": ticker,
                "name": company_names.get(code) or ticker,
                "rank": position,
                "signal": "near_miss",
                "pattern": "投げ売り反転",
                "close": round(float(row["_close"]), 2),
                "trading_value": round(float(row["trading_value"])),
                "RSI": round(float(row["RSI"]), 4),
                "ATR": round(float(row["ATR"]), 6),
                "return_1d": round(float(row["return_1d"]), 6),
                "return_5d": round(float(row["return_5d"]), 6),
                "volume_ratio_1_20": round(float(row["volume_ratio_1_20"]), 6),
                "passed_conditions": len(CONDITION_KEYS) - 1,
                "total_conditions": len(CONDITION_KEYS),
                "failed_condition": _failed_condition_payload(row),
                "miss_distance": round(float(row["_miss_distance"]), 6),
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
            "key": "rsi14_stable_score_10d_v1",
            "label": "RSI14安定スコア1位10D",
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
                "take_profit_pct": TAKE_PROFIT_PCT,
                "stop_loss_pct": STOP_LOSS_PCT,
                "holding_days": HOLDING_DAYS,
            },
        },
        "signal_count": len(records),
        "signals": records,
        "near_miss_count": len(near_miss_records),
        "near_misses": near_miss_records,
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
        f"near_misses={payload['near_miss_count']} output={args.output}"
    )


if __name__ == "__main__":
    main()
