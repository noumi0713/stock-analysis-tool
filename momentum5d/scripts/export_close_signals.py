from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.adjustments import normalize_split_adjusted_ohlcv
from app.audit_metadata import build_audit_context
from app.live_strategy import load_frozen_strategy
from app.market_data_contract import (
    load_market_certification,
    validate_market_certification,
)
from app.signal_audit import build_audit_bundle, build_strategy_audit, write_gzip_json

LIVE_STRATEGY = load_frozen_strategy()
REVERSAL_SPEC = LIVE_STRATEGY["signals"]["capitulation_reversal"]
REVERSAL_CONDITIONS = REVERSAL_SPEC["conditions"]
REVERSAL_EXECUTION = REVERSAL_SPEC["execution"]
PULLBACK_SPEC = LIVE_STRATEGY["signals"]["first_pullback"]
PULLBACK_CONDITIONS = PULLBACK_SPEC["conditions"]
PULLBACK_EXECUTION = PULLBACK_SPEC["execution"]

RSI_PERIOD = int(REVERSAL_CONDITIONS["rsi_period"])
RSI_MIN = float(REVERSAL_CONDITIONS["rsi_min"])
RSI_MAX = float(REVERSAL_CONDITIONS["rsi_max"])
RETURN_1D_MIN = float(REVERSAL_CONDITIONS["return_1d_min"])
RETURN_1D_MAX = float(REVERSAL_CONDITIONS["return_1d_max"])
RETURN_5D_MIN = float(REVERSAL_CONDITIONS["return_5d_min"])
RETURN_5D_MAX = float(REVERSAL_CONDITIONS["return_5d_max"])
VOLUME_RATIO_MIN = float(REVERSAL_CONDITIONS["volume_ratio_min"])
TURNOVER_MIN = float(REVERSAL_CONDITIONS["minimum_turnover_yen"])
ATR_MIN = float(REVERSAL_CONDITIONS["atr_14_pct_min"])
ATR_MAX = float(REVERSAL_CONDITIONS["atr_14_pct_max"])
MAX_SIGNALS = int(REVERSAL_SPEC["maximum_candidates_per_day"])
MAX_NEAR_MISSES = 5
TAKE_PROFIT_PCT = float(REVERSAL_EXECUTION["take_profit_pct"])
STOP_LOSS_PCT = float(REVERSAL_EXECUTION["stop_loss_pct"])
HOLDING_DAYS = int(REVERSAL_EXECUTION["holding_days"])

PULLBACK_MAX_SIGNALS = int(PULLBACK_SPEC["maximum_candidates_per_day"])
PULLBACK_DRAWDOWN_MIN = float(PULLBACK_CONDITIONS["drawdown_from_20d_high_min"])
PULLBACK_DRAWDOWN_MAX = float(PULLBACK_CONDITIONS["drawdown_from_20d_high_max"])
PULLBACK_MA25_DISTANCE_MIN = float(PULLBACK_CONDITIONS["distance_from_ma25_min"])
PULLBACK_MA25_DISTANCE_MAX = float(PULLBACK_CONDITIONS["distance_from_ma25_max"])
PULLBACK_RETURN_1D_MIN = float(PULLBACK_CONDITIONS["return_1d_min"])
PULLBACK_RETURN_1D_MAX = float(PULLBACK_CONDITIONS["return_1d_max"])
PULLBACK_CLOSE_POSITION_MIN = float(PULLBACK_CONDITIONS["close_position_min"])
PULLBACK_VOLUME_RATIO_MIN = float(PULLBACK_CONDITIONS["volume_ratio_min"])
PULLBACK_TURNOVER_MIN = float(PULLBACK_CONDITIONS["minimum_turnover_yen"])
PULLBACK_ATR_MIN = float(PULLBACK_CONDITIONS["atr_14_pct_min"])
PULLBACK_ATR_MAX = float(PULLBACK_CONDITIONS["atr_14_pct_max"])
PULLBACK_MA25_SLOPE_MIN = float(PULLBACK_CONDITIONS["ma25_slope_5d_min"])
PULLBACK_MA75_SLOPE_MIN = float(PULLBACK_CONDITIONS["ma75_slope_10d_min"])
PULLBACK_TAKE_PROFIT_PCT = float(PULLBACK_EXECUTION["take_profit_pct"])
PULLBACK_STOP_LOSS_PCT = float(PULLBACK_EXECUTION["stop_loss_pct"])
PULLBACK_HOLDING_DAYS = int(PULLBACK_EXECUTION["holding_days"])
PULLBACK_REQUIRED_SESSIONS = int(LIVE_STRATEGY["data"]["history_sessions_required"])
PULLBACK_MIN_INDICATOR_COVERAGE = float(
    LIVE_STRATEGY["data"]["minimum_latest_ticker_coverage"]
)

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

REVERSAL_NAME = "投げ売り反転"
PULLBACK_NAME = "上昇トレンド初押し"

AUDIT_METRICS = [
    "date",
    "_open",
    "_high",
    "_low",
    "_close",
    "_volume",
    "stock_splits",
    "RSI",
    "ATR",
    "return_1d",
    "return_5d",
    "volume_ratio_1_20",
    "trading_value",
    "ma25",
    "ma75",
    "ma25_slope_5d",
    "ma75_slope_10d",
    "prior_high_20d",
    "drawdown_from_20d_high",
    "distance_from_ma25",
    "close_position",
]


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
    work = normalize_split_adjusted_ohlcv(work)

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
        stock["trading_value"] = stock["_turnover"]
        stock["ATR"] = true_range.rolling(14, min_periods=14).sum().div(14).div(close)
        stock["ma25"] = close.rolling(25, min_periods=25).mean()
        stock["ma75"] = close.rolling(75, min_periods=75).mean()
        stock["ma25_slope_5d"] = stock["ma25"].div(stock["ma25"].shift(5)).sub(1.0)
        stock["ma75_slope_10d"] = stock["ma75"].div(stock["ma75"].shift(10)).sub(1.0)
        stock["prior_high_20d"] = (
            stock["_high"].rolling(20, min_periods=20).max().shift(1)
        )
        stock["drawdown_from_20d_high"] = close.div(stock["prior_high_20d"]).sub(1.0)
        stock["distance_from_ma25"] = close.div(stock["ma25"]).sub(1.0)
        day_range = stock["_high"].sub(stock["_low"])
        stock["close_position"] = close.sub(stock["_low"]).div(
            day_range.replace(0.0, np.nan)
        ).fillna(0.5)
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


def select_historical_signals(indicators: pd.DataFrame) -> pd.DataFrame:
    if indicators.empty:
        return indicators.copy()
    selected = indicators.loc[_condition_results(indicators).all(axis=1)].copy()
    return (
        selected.sort_values(
            ["date", "volume_ratio_1_20", "ticker"],
            ascending=[True, False, True],
        )
        .groupby("date", sort=False)
        .head(MAX_SIGNALS)
        .reset_index(drop=True)
    )


def _pullback_condition_results(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trend_order": (frame["_close"] > frame["ma25"])
            & (frame["ma25"] > frame["ma75"]),
            "ma25_slope": frame["ma25_slope_5d"].gt(PULLBACK_MA25_SLOPE_MIN),
            "ma75_slope": frame["ma75_slope_10d"].gt(PULLBACK_MA75_SLOPE_MIN),
            "drawdown": frame["drawdown_from_20d_high"].between(
                PULLBACK_DRAWDOWN_MIN, PULLBACK_DRAWDOWN_MAX
            ),
            "ma25_distance": frame["distance_from_ma25"].between(
                PULLBACK_MA25_DISTANCE_MIN, PULLBACK_MA25_DISTANCE_MAX
            ),
            "return_1d": frame["return_1d"].between(
                PULLBACK_RETURN_1D_MIN, PULLBACK_RETURN_1D_MAX
            ),
            "bullish": frame["_close"].gt(frame["_open"]),
            "close_position": frame["close_position"].ge(
                PULLBACK_CLOSE_POSITION_MIN
            ),
            "volume_ratio": frame["volume_ratio_1_20"].ge(
                PULLBACK_VOLUME_RATIO_MIN
            ),
            "trading_value": frame["trading_value"].ge(PULLBACK_TURNOVER_MIN),
            "atr": frame["ATR"].between(PULLBACK_ATR_MIN, PULLBACK_ATR_MAX),
        },
        index=frame.index,
    )


def _reversal_condition_actuals(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "rsi": frame["RSI"],
        "return_1d": frame["return_1d"],
        "return_5d": frame["return_5d"],
        "volume_ratio_1_20": frame["volume_ratio_1_20"],
        "trading_value": frame["trading_value"],
        "atr_14_pct": frame["ATR"],
        "ma25": frame["_close"].div(frame["ma25"]).sub(1.0),
        "bullish": frame["_close"].sub(frame["_open"]),
    }


def _pullback_condition_actuals(frame: pd.DataFrame) -> dict[str, pd.Series]:
    trend_gap = pd.concat(
        [
            frame["_close"].div(frame["ma25"]).sub(1.0),
            frame["ma25"].div(frame["ma75"]).sub(1.0),
        ],
        axis=1,
    ).min(axis=1, skipna=False)
    return {
        "trend_order": trend_gap,
        "ma25_slope": frame["ma25_slope_5d"],
        "ma75_slope": frame["ma75_slope_10d"],
        "drawdown": frame["drawdown_from_20d_high"],
        "ma25_distance": frame["distance_from_ma25"],
        "return_1d": frame["return_1d"],
        "bullish": frame["_close"].sub(frame["_open"]),
        "close_position": frame["close_position"],
        "volume_ratio": frame["volume_ratio_1_20"],
        "trading_value": frame["trading_value"],
        "atr": frame["ATR"],
    }


REVERSAL_REQUIREMENTS = {
    "rsi": f"{RSI_MIN} <= RSI14 <= {RSI_MAX}",
    "return_1d": f"{RETURN_1D_MIN} <= return_1d <= {RETURN_1D_MAX}",
    "return_5d": f"{RETURN_5D_MIN} <= return_5d <= {RETURN_5D_MAX}",
    "volume_ratio_1_20": f"volume_ratio_1_20 >= {VOLUME_RATIO_MIN}",
    "trading_value": f"trading_value >= {TURNOVER_MIN}",
    "atr_14_pct": f"{ATR_MIN} <= ATR14/close <= {ATR_MAX}",
    "ma25": "close < ma25",
    "bullish": "close > open",
}

PULLBACK_REQUIREMENTS = {
    "trend_order": "close > ma25 > ma75",
    "ma25_slope": f"ma25_slope_5d > {PULLBACK_MA25_SLOPE_MIN}",
    "ma75_slope": f"ma75_slope_10d > {PULLBACK_MA75_SLOPE_MIN}",
    "drawdown": (
        f"{PULLBACK_DRAWDOWN_MIN} <= drawdown_from_20d_high "
        f"<= {PULLBACK_DRAWDOWN_MAX}"
    ),
    "ma25_distance": (
        f"{PULLBACK_MA25_DISTANCE_MIN} <= distance_from_ma25 "
        f"<= {PULLBACK_MA25_DISTANCE_MAX}"
    ),
    "return_1d": f"{PULLBACK_RETURN_1D_MIN} <= return_1d <= {PULLBACK_RETURN_1D_MAX}",
    "bullish": "close > open",
    "close_position": f"close_position >= {PULLBACK_CLOSE_POSITION_MIN}",
    "volume_ratio": f"volume_ratio_1_20 >= {PULLBACK_VOLUME_RATIO_MIN}",
    "trading_value": f"trading_value >= {PULLBACK_TURNOVER_MIN}",
    "atr": f"{PULLBACK_ATR_MIN} <= ATR14/close <= {PULLBACK_ATR_MAX}",
}


def _reversal_ranking(indicators: pd.DataFrame) -> pd.DataFrame:
    latest = _latest_rows(indicators)
    actuals = _reversal_condition_actuals(latest)
    valid = pd.DataFrame(actuals).notna().all(axis=1)
    eligible = _condition_results(latest).all(axis=1) & valid
    ranked = latest.loc[eligible, ["ticker", "volume_ratio_1_20"]].sort_values(
        ["volume_ratio_1_20", "ticker"], ascending=[False, True]
    )
    ranked = ranked.rename(columns={"volume_ratio_1_20": "ranking_value"})
    ranked.insert(1, "rank", range(1, len(ranked) + 1))
    ranked["method"] = "volume_ratio_descending"
    ranked["interpretation"] = "ordinal_only_not_win_rate_or_expected_return"
    return ranked.reset_index(drop=True)


def _pullback_ranking(indicators: pd.DataFrame) -> pd.DataFrame:
    latest = _latest_rows(indicators)
    actuals = _pullback_condition_actuals(latest)
    valid = pd.DataFrame(actuals).notna().all(axis=1)
    eligible = _pullback_condition_results(latest).all(axis=1) & valid
    ranking = PULLBACK_SPEC["ranking"]
    ranked = latest.loc[
        eligible,
        ["ticker", "close_position", "volume_ratio_1_20", "distance_from_ma25"],
    ].copy()
    ranked["close_position_contribution"] = ranked["close_position"] * float(
        ranking["close_position_weight"]
    )
    ranked["volume_ratio_contribution"] = ranked["volume_ratio_1_20"] * float(
        ranking["volume_ratio_weight"]
    )
    ranked["ma25_distance_contribution"] = ranked["distance_from_ma25"] * float(
        ranking["ma25_distance_weight"]
    )
    ranked["ranking_value"] = ranked[
        [
            "close_position_contribution",
            "volume_ratio_contribution",
            "ma25_distance_contribution",
        ]
    ].sum(axis=1)
    ranked = ranked.sort_values(["ranking_value", "ticker"], ascending=[False, True])
    ranked.insert(1, "rank", range(1, len(ranked) + 1))
    ranked["method"] = "frozen_linear_score_v1"
    ranked["interpretation"] = "ordinal_only_not_win_rate_or_expected_return"
    return ranked.reset_index(drop=True)


def select_latest_pullback_signals(indicators: pd.DataFrame) -> pd.DataFrame:
    latest = _latest_rows(indicators)
    if latest.empty:
        return latest
    mask = _pullback_condition_results(latest).all(axis=1)
    selected = latest.loc[mask].copy()
    ranking = PULLBACK_SPEC["ranking"]
    selected["_pullback_score"] = (
        selected["close_position"] * float(ranking["close_position_weight"])
        + selected["volume_ratio_1_20"] * float(ranking["volume_ratio_weight"])
        + selected["distance_from_ma25"] * float(ranking["ma25_distance_weight"])
    )
    return (
        selected.sort_values(
            ["_pullback_score", "ticker"], ascending=[False, True]
        )
        .head(PULLBACK_MAX_SIGNALS)
        .reset_index(drop=True)
    )


def select_historical_pullback_signals(indicators: pd.DataFrame) -> pd.DataFrame:
    if indicators.empty:
        return indicators.copy()
    selected = indicators.loc[
        _pullback_condition_results(indicators).all(axis=1)
    ].copy()
    ranking = PULLBACK_SPEC["ranking"]
    selected["_pullback_score"] = (
        selected["close_position"] * float(ranking["close_position_weight"])
        + selected["volume_ratio_1_20"] * float(ranking["volume_ratio_weight"])
        + selected["distance_from_ma25"] * float(ranking["ma25_distance_weight"])
    )
    return (
        selected.sort_values(
            ["date", "_pullback_score", "ticker"],
            ascending=[True, False, True],
        )
        .groupby("date", sort=False)
        .head(PULLBACK_MAX_SIGNALS)
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


def _load_theme_memberships(path: Path | None) -> dict[str, list[dict[str, str]]]:
    """Load descriptive theme metadata without affecting signal selection."""

    if path is None or not path.exists():
        return {}
    frame = pd.read_csv(path, dtype="string")
    required = {
        "theme_name",
        "cluster",
        "topix17_group",
        "stock_code",
        "source_url",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Theme data is missing columns: {sorted(missing)}")

    memberships: dict[str, list[dict[str, str]]] = {}
    for row in frame.itertuples(index=False):
        if pd.isna(row.stock_code) or pd.isna(row.theme_name):
            continue
        code = str(row.stock_code).strip()
        membership = {
            "theme": str(row.theme_name).strip(),
            "cluster": "" if pd.isna(row.cluster) else str(row.cluster).strip(),
            "topix17_group": (
                "" if pd.isna(row.topix17_group) else str(row.topix17_group).strip()
            ),
            "source_url": "" if pd.isna(row.source_url) else str(row.source_url).strip(),
        }
        bucket = memberships.setdefault(code, [])
        if membership not in bucket:
            bucket.append(membership)

    for bucket in memberships.values():
        bucket.sort(key=lambda item: (item["theme"], item["cluster"]))
    return memberships


def _theme_fields(
    code: str, memberships: dict[str, list[dict[str, str]]]
) -> dict[str, Any]:
    items = memberships.get(code, [])
    return {
        "theme_covered": bool(items),
        "themes": sorted({item["theme"] for item in items if item["theme"]}),
        "theme_clusters": sorted(
            {item["cluster"] for item in items if item["cluster"]}
        ),
        "topix17_groups": sorted(
            {item["topix17_group"] for item in items if item["topix17_group"]}
        ),
        "theme_memberships": items,
    }


def build_payload(
    prices: pd.DataFrame,
    *,
    certification: dict[str, Any],
    names: dict[str, str] | None = None,
    theme_memberships: dict[str, list[dict[str, str]]] | None = None,
    generated_at: str | None = None,
    git_commit: str | None = None,
    include_full_audit: bool = False,
) -> dict[str, Any]:
    certification = validate_market_certification(certification, prices=prices)
    indicators = calculate_indicators(prices)
    if indicators.empty:
        raise ValueError("No usable daily prices were found")
    latest_date = indicators["date"].max().isoformat()
    validate_market_certification(
        certification,
        expected_date=latest_date,
    )
    latest_rows = indicators.loc[indicators["date"].astype(str).eq(latest_date)]
    stamp = generated_at or datetime.now(UTC).isoformat()
    audit_context = build_audit_context(
        market_date=latest_date,
        acquired_at=str(certification["acquired_at"]),
        strategy_version=LIVE_STRATEGY["strategy_version"],
        strategy_names=["capitulation_reversal", "first_pullback"],
        snapshot_fingerprint=str(certification["snapshot_fingerprint"]),
        computed_at=stamp,
        git_commit=git_commit,
    )
    pullback_indicator_coverage = float(
        latest_rows["ma75_slope_10d"].notna().mean()
    )
    if pullback_indicator_coverage < PULLBACK_MIN_INDICATOR_COVERAGE:
        raise ValueError(
            "Insufficient history for first-pullback detection: "
            f"ma75_slope_10d coverage={pullback_indicator_coverage:.1%}, "
            f"required={PULLBACK_MIN_INDICATOR_COVERAGE:.1%}; "
            f"at least {PULLBACK_REQUIRED_SESSIONS} trading sessions are required"
        )
    signals = select_latest_signals(indicators)
    pullback_signals = select_latest_pullback_signals(indicators)
    near_misses = select_latest_near_misses(indicators)
    reversal_ranking = _reversal_ranking(indicators)
    pullback_ranking = _pullback_ranking(indicators)
    ranking_by_signal = {
        "capitulation_reversal": reversal_ranking.set_index("ticker").to_dict(
            orient="index"
        ),
        "first_pullback": pullback_ranking.set_index("ticker").to_dict(orient="index"),
    }
    company_names = names or {}
    themes_by_code = theme_memberships or {}
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
                **_theme_fields(code, themes_by_code),
                "rank": position,
                "audit_id": audit_context["audit_id"],
                "signal": "capitulation_reversal",
                "strategy_name": REVERSAL_NAME,
                "pattern": REVERSAL_NAME,
                "close": round(float(row["_close"]), 2),
                "trading_value": round(float(row["trading_value"])),
                "turnover_value": round(float(row["trading_value"])),
                "RSI": round(float(row["RSI"]), 4),
                "ATR": round(float(row["ATR"]), 6),
                "return_1d": round(float(row["return_1d"]), 6),
                "return_5d": round(float(row["return_5d"]), 6),
                "volume_ratio_1_20": round(float(row["volume_ratio_1_20"]), 6),
                "ranking": ranking_by_signal["capitulation_reversal"][ticker],
                "ranking_metrics_status": (
                    "ordinal_selection_only_not_probability_or_expected_return"
                ),
                "entry_rule": "翌営業日始値",
                "take_profit_pct": TAKE_PROFIT_PCT,
                "stop_loss_pct": STOP_LOSS_PCT,
                "holding_days": HOLDING_DAYS,
            }
        )

    pullback_records: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(pullback_signals.iterrows(), start=1):
        ticker = str(row["ticker"])
        code = str(row.get("code") or ticker.removesuffix(".T"))[:4]
        pullback_records.append(
            {
                "date": latest_date,
                "code": code,
                "ticker": ticker,
                "name": company_names.get(code) or ticker,
                **_theme_fields(code, themes_by_code),
                "rank": position,
                "audit_id": audit_context["audit_id"],
                "signal": "first_pullback",
                "strategy_name": PULLBACK_NAME,
                "pattern": PULLBACK_NAME,
                "close": round(float(row["_close"]), 2),
                "trading_value": round(float(row["trading_value"])),
                "turnover_value": round(float(row["trading_value"])),
                "RSI": round(float(row["RSI"]), 4),
                "ATR": round(float(row["ATR"]), 6),
                "return_1d": round(float(row["return_1d"]), 6),
                "return_5d": round(float(row["return_5d"]), 6),
                "volume_ratio_1_20": round(float(row["volume_ratio_1_20"]), 6),
                "ma25": round(float(row["ma25"]), 4),
                "ma75": round(float(row["ma75"]), 4),
                "ma25_slope_5d": round(float(row["ma25_slope_5d"]), 6),
                "ma75_slope_10d": round(float(row["ma75_slope_10d"]), 6),
                "drawdown_from_20d_high": round(
                    float(row["drawdown_from_20d_high"]), 6
                ),
                "distance_from_ma25": round(
                    float(row["distance_from_ma25"]), 6
                ),
                "close_position": round(float(row["close_position"]), 6),
                "ranking": ranking_by_signal["first_pullback"][ticker],
                "ranking_metrics_status": (
                    "ordinal_selection_only_not_probability_or_expected_return"
                ),
                "entry_rule": "翌営業日始値（前日終値比−4〜+3%のみ）",
                "take_profit_pct": PULLBACK_TAKE_PROFIT_PCT,
                "stop_loss_pct": PULLBACK_STOP_LOSS_PCT,
                "holding_days": PULLBACK_HOLDING_DAYS,
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
                **_theme_fields(code, themes_by_code),
                "rank": position,
                "audit_id": audit_context["audit_id"],
                "signal": "near_miss",
                "strategy_name": REVERSAL_NAME,
                "pattern": REVERSAL_NAME,
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

    theme_names = {
        item["theme"]
        for memberships in themes_by_code.values()
        for item in memberships
        if item["theme"]
    }
    reversal_audit = build_strategy_audit(
        latest_rows,
        signal_type="capitulation_reversal",
        strategy_name=REVERSAL_NAME,
        condition_results=_condition_results(latest_rows),
        condition_actuals=_reversal_condition_actuals(latest_rows),
        condition_requirements=REVERSAL_REQUIREMENTS,
        selected_tickers=signals["ticker"].astype(str).tolist(),
        ranking=reversal_ranking,
        ranking_definition={
            "method": "volume_ratio_descending",
            "formula": "volume_ratio_1_20 descending; ticker ascending tie-break",
            "components": ["volume_ratio_1_20"],
            "meaning": "ordinal_selection_only",
            "not_a_probability": True,
            "not_expected_return": True,
        },
        metric_keys=AUDIT_METRICS,
        audit_context=audit_context,
        missing_tickers=certification.get("missing_tickers") or [],
        company_names=company_names,
    )
    pullback_audit = build_strategy_audit(
        latest_rows,
        signal_type="first_pullback",
        strategy_name=PULLBACK_NAME,
        condition_results=_pullback_condition_results(latest_rows),
        condition_actuals=_pullback_condition_actuals(latest_rows),
        condition_requirements=PULLBACK_REQUIREMENTS,
        selected_tickers=pullback_signals["ticker"].astype(str).tolist(),
        ranking=pullback_ranking,
        ranking_definition={
            "method": "frozen_linear_score_v1",
            "formula": (
                "2.0*close_position + 1.0*volume_ratio_1_20 "
                "- 25.0*distance_from_ma25"
            ),
            "components": [
                {"metric": "close_position", "weight": 2.0},
                {"metric": "volume_ratio_1_20", "weight": 1.0},
                {"metric": "distance_from_ma25", "weight": -25.0},
            ],
            "meaning": "ordinal_selection_only",
            "not_a_probability": True,
            "not_expected_return": True,
        },
        metric_keys=AUDIT_METRICS,
        audit_context=audit_context,
        missing_tickers=certification.get("missing_tickers") or [],
        company_names=company_names,
    )
    strategy_audits = {
        "capitulation_reversal": reversal_audit,
        "first_pullback": pullback_audit,
    }
    full_audit = build_audit_bundle(
        audit_context=audit_context,
        certification=certification,
        strategies=strategy_audits,
    )
    audit_summary = {
        key: {field: value for field, value in audit.items() if field != "candidates"}
        for key, audit in strategy_audits.items()
    }
    payload = {
        "schema_version": 4,
        "strategy_version": LIVE_STRATEGY["strategy_version"],
        "strategy_status": LIVE_STRATEGY["status"],
        "portfolio_rules": LIVE_STRATEGY["portfolio"],
        "data_quality": {
            "status": "certified",
            "source": certification["source"],
            "adjusted_ohlc": True,
            "split_adjusted_volume": True,
            "adjustment_basis": "split_only_latest_share_basis",
            "snapshot_fingerprint": certification["snapshot_fingerprint"],
            "acquired_at": certification["acquired_at"],
            "signal_universe": "current_tse_as_of_signal_date",
            "historical_point_in_time_universe": "not_applicable_to_live_detection",
        },
        "generated_at": stamp,
        "audit_context": audit_context,
        "signal_audit_summary": audit_summary,
        "date": latest_date,
        "latest_date": latest_date,
        "next_session": "翌営業日",
        "update": {
            "status": "complete",
            "session": "close",
            "session_label": "大引け",
            "market_date": latest_date,
            "data_through": certification["data_through"],
            "interval": "1d",
            "source": certification["source"],
            "market_timezone": certification["market_timezone"],
            "successful_tickers": certification["successful_tickers"],
            "expected_tickers": certification["expected_tickers"],
            "coverage": certification["coverage"],
            "pullback_indicator_coverage": round(
                pullback_indicator_coverage, 6
            ),
            "generated_at": stamp,
        },
        "theme_catalog": {
            "enabled": bool(themes_by_code),
            "used_for_primary_selection": False,
            "theme_count": len(theme_names),
            "covered_stock_count": len(themes_by_code),
            "membership_count": sum(len(items) for items in themes_by_code.values()),
            "description": "株探テーマを参考にした関連銘柄分析用メタデータ",
        },
        "signal_model": {
            "key": "rsi14_stable_score_10d_v1",
            "label": "RSI14安定スコア1位10D",
            "ranking": reversal_audit["ranking_definition"],
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
        "pullback_signal_model": {
            "key": "uptrend_first_pullback_v1",
            "label": "上昇トレンド初押し",
            "ranking": pullback_audit["ranking_definition"],
            "conditions": {
                "trend_order": "close > ma25 > ma75",
                "ma25_slope_5d_min": PULLBACK_MA25_SLOPE_MIN,
                "ma75_slope_10d_min": PULLBACK_MA75_SLOPE_MIN,
                "drawdown_from_20d_high_min": PULLBACK_DRAWDOWN_MIN,
                "drawdown_from_20d_high_max": PULLBACK_DRAWDOWN_MAX,
                "distance_from_ma25_min": PULLBACK_MA25_DISTANCE_MIN,
                "distance_from_ma25_max": PULLBACK_MA25_DISTANCE_MAX,
                "return_1d_min": PULLBACK_RETURN_1D_MIN,
                "return_1d_max": PULLBACK_RETURN_1D_MAX,
                "bullish": True,
                "close_position_min": PULLBACK_CLOSE_POSITION_MIN,
                "volume_ratio_min": PULLBACK_VOLUME_RATIO_MIN,
                "minimum_turnover_yen": int(PULLBACK_TURNOVER_MIN),
                "atr_14_pct_min": PULLBACK_ATR_MIN,
                "atr_14_pct_max": PULLBACK_ATR_MAX,
                "maximum_candidates_per_day": PULLBACK_MAX_SIGNALS,
                "entry_gap_min": PULLBACK_EXECUTION["entry_gap_min"],
                "entry_gap_max": PULLBACK_EXECUTION["entry_gap_max"],
                "entry_rule": "翌営業日始値",
                "take_profit_pct": PULLBACK_TAKE_PROFIT_PCT,
                "stop_loss_pct": PULLBACK_STOP_LOSS_PCT,
                "holding_days": PULLBACK_HOLDING_DAYS,
            },
        },
        "total_signal_count": len(records) + len(pullback_records),
        "signal_count": len(records),
        "signals": records,
        "pullback_signal_count": len(pullback_records),
        "pullback_signals": pullback_records,
        "near_miss_count": len(near_miss_records),
        "near_misses": near_miss_records,
    }
    if include_full_audit:
        payload["_full_audit_bundle"] = full_audit
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="大引け後の翌営業日用シグナルを生成")
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--names", type=Path)
    parser.add_argument("--themes", type=Path)
    parser.add_argument("--certification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--audit-reference")
    parser.add_argument("--git-commit")
    args = parser.parse_args()
    payload = build_payload(
        pd.read_parquet(args.prices),
        certification=load_market_certification(args.certification),
        names=_load_names(args.names),
        theme_memberships=_load_theme_memberships(args.themes),
        git_commit=args.git_commit,
        include_full_audit=True,
    )
    full_audit = payload.pop("_full_audit_bundle")
    artifact = write_gzip_json(full_audit, args.audit_output)
    artifact["path"] = args.audit_reference or artifact["path"]
    artifact["candidate_rows"] = sum(
        len(strategy["candidates"])
        for strategy in full_audit["strategies"].values()
    )
    payload["signal_audit_artifact"] = artifact
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    print(
        f"Close signals: date={payload['date']} reversal={payload['signal_count']} "
        f"pullback={payload['pullback_signal_count']} "
        f"near_misses={payload['near_miss_count']} output={args.output}"
    )


if __name__ == "__main__":
    main()
