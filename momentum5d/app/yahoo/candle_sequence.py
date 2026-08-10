from __future__ import annotations

from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_HORIZONS = (1, 3, 5, 10)
TURNOVER_THRESHOLDS = (0, 50_000_000, 100_000_000, 200_000_000)
TARGET_TIMING_WINDOW_DAYS = 60


def _wilson_interval(successes: int, samples: int) -> list[float | None]:
    if samples <= 0:
        return [None, None]
    z = 1.96
    proportion = successes / samples
    denominator = 1 + (z**2 / samples)
    center = (proportion + z**2 / (2 * samples)) / denominator
    margin = (
        z * sqrt(proportion * (1 - proportion) / samples + z**2 / (4 * samples**2)) / denominator
    )
    return [center - margin, center + margin]


def _outcome_summary(frame: pd.DataFrame, horizon: int) -> dict[str, Any]:
    future_return = pd.to_numeric(
        frame[f"future_close_return_{horizon}d"], errors="coerce"
    ).dropna()
    if future_return.empty:
        return {
            "samples": 0,
            "up_rate": None,
            "down_rate": None,
            "flat_rate": None,
            "mean_return": None,
            "median_return": None,
            "up_rate_ci95": [None, None],
            "reached_plus_5pct_rate": None,
            "breached_minus_3pct_rate": None,
            "breached_minus_5pct_rate": None,
        }

    selected = frame.loc[future_return.index]
    samples = len(future_return)
    up_count = int(future_return.gt(0).sum())
    down_count = int(future_return.lt(0).sum())
    flat_count = samples - up_count - down_count
    return {
        "samples": samples,
        "up_rate": float(up_count / samples),
        "down_rate": float(down_count / samples),
        "flat_rate": float(flat_count / samples),
        "mean_return": float(future_return.mean()),
        "median_return": float(future_return.median()),
        "up_rate_ci95": _wilson_interval(up_count, samples),
        "reached_plus_5pct_rate": float(selected[f"future_max_return_{horizon}d"].ge(0.05).mean()),
        "breached_minus_3pct_rate": float(
            selected[f"future_min_return_{horizon}d"].le(-0.03).mean()
        ),
        "breached_minus_5pct_rate": float(
            selected[f"future_min_return_{horizon}d"].le(-0.05).mean()
        ),
    }


def _target_timing_summary(frame: pd.DataFrame, hit_day_column: str) -> dict[str, Any]:
    complete = frame.loc[frame["target_timing_complete"]].copy()
    hit_days = pd.to_numeric(complete[hit_day_column], errors="coerce").dropna()
    samples = len(complete)
    hits = len(hit_days)
    return {
        "samples_with_full_window": samples,
        "hits_within_window": hits,
        "hit_rate_within_window": float(hits / samples) if samples else None,
        "mean_business_days_to_plus_5pct": float(hit_days.mean()) if hits else None,
        "median_business_days_to_plus_5pct": float(hit_days.median()) if hits else None,
        "p25_business_days": float(hit_days.quantile(0.25)) if hits else None,
        "p75_business_days": float(hit_days.quantile(0.75)) if hits else None,
    }


def _prepare_sequence_frame(
    prices: pd.DataFrame,
    horizons: tuple[int, ...],
    *,
    target_timing_window_days: int,
) -> pd.DataFrame:
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(prices.columns))
    if missing:
        raise ValueError(f"ローソク足検証に必要な列がありません: {missing}")

    frame = prices.sort_values(["ticker", "date"]).copy()
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[
        frame[["open", "high", "low", "close"]].gt(0).all(axis=1) & frame["volume"].gt(0)
    ].copy()
    grouped = frame.groupby("ticker", sort=False)
    bearish = frame["close"].lt(frame["open"])
    frame["three_up_one_down"] = bearish
    for lag in (1, 2, 3):
        frame["three_up_one_down"] &= grouped["close"].shift(lag).gt(grouped["open"].shift(lag))

    frame["turnover_value"] = frame["close"] * frame["volume"]
    frame["day4_body_return"] = frame["close"] / frame["open"] - 1
    frame["day4_volume_up"] = frame["volume"].ge(grouped["volume"].shift(1))
    frame["prior_3d_gain"] = (
        frame["close"].groupby(frame["ticker"]).shift(1)
        / frame["open"].groupby(frame["ticker"]).shift(3)
        - 1
    )

    max_horizon = max(horizons)
    future_closes = [grouped["close"].shift(-step) for step in range(1, max_horizon + 1)]
    future_highs = [grouped["high"].shift(-step) for step in range(1, max_horizon + 1)]
    future_lows = [grouped["low"].shift(-step) for step in range(1, max_horizon + 1)]
    for horizon in horizons:
        future_close = future_closes[horizon - 1]
        future_max = pd.concat(future_highs[:horizon], axis=1).max(axis=1)
        future_min = pd.concat(future_lows[:horizon], axis=1).min(axis=1)
        complete = future_close.notna()
        frame[f"future_close_return_{horizon}d"] = np.where(
            complete, future_close / frame["close"] - 1, np.nan
        )
        frame[f"future_max_return_{horizon}d"] = np.where(
            complete, future_max / frame["close"] - 1, np.nan
        )
        frame[f"future_min_return_{horizon}d"] = np.where(
            complete, future_min / frame["close"] - 1, np.nan
        )

    target_price = frame["close"] * 1.05
    frame["target_timing_complete"] = grouped["close"].shift(-target_timing_window_days).notna()
    frame["first_plus_5pct_high_day"] = np.nan
    frame["first_plus_5pct_close_day"] = np.nan
    for day in range(1, target_timing_window_days + 1):
        future_high = grouped["high"].shift(-day)
        future_close = grouped["close"].shift(-day)
        high_hit = frame["first_plus_5pct_high_day"].isna() & future_high.ge(target_price)
        close_hit = frame["first_plus_5pct_close_day"].isna() & future_close.ge(target_price)
        frame.loc[high_hit, "first_plus_5pct_high_day"] = day
        frame.loc[close_hit, "first_plus_5pct_close_day"] = day
    return frame


def analyze_three_up_one_down(
    prices: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    target_timing_window_days: int = TARGET_TIMING_WINDOW_DAYS,
) -> dict[str, Any]:
    """3日連続陽線の後の陰線を、4日目終値基準で将来リターン集計する。"""
    if prices.empty:
        return {"pattern": "three_up_one_down", "sample_count": 0}
    if not horizons or min(horizons) <= 0:
        raise ValueError("検証日数は1以上で指定してください")
    if target_timing_window_days < max(horizons):
        raise ValueError("+5%到達日数の確認期間は最大の検証日数以上にしてください")

    frame = _prepare_sequence_frame(
        prices,
        horizons,
        target_timing_window_days=target_timing_window_days,
    )
    pattern = frame.loc[frame["three_up_one_down"]].copy()
    bearish_baseline = frame.loc[frame["close"].lt(frame["open"])].copy()

    by_horizon: dict[str, Any] = {}
    baseline_by_horizon: dict[str, Any] = {}
    for horizon in horizons:
        key = f"{horizon}d"
        pattern_summary = _outcome_summary(pattern, horizon)
        baseline_summary = _outcome_summary(bearish_baseline, horizon)
        if pattern_summary["up_rate"] is not None and baseline_summary["up_rate"] is not None:
            pattern_summary["up_rate_difference_vs_bearish_baseline"] = float(
                pattern_summary["up_rate"] - baseline_summary["up_rate"]
            )
        else:
            pattern_summary["up_rate_difference_vs_bearish_baseline"] = None
        by_horizon[key] = pattern_summary
        baseline_by_horizon[key] = baseline_summary

    liquidity: dict[str, Any] = {}
    for threshold in TURNOVER_THRESHOLDS:
        label = "all" if threshold == 0 else f"turnover_{threshold // 1_000_000}m_plus"
        selected = pattern.loc[pattern["turnover_value"].ge(threshold)]
        liquidity[label] = {
            "minimum_turnover_yen": threshold,
            "5d": _outcome_summary(selected, 5),
            "10d": _outcome_summary(selected, 10),
        }

    liquid_pattern = pattern.loc[pattern["turnover_value"].ge(200_000_000)]
    subgroups = {
        "day4_volume_up": _outcome_summary(liquid_pattern.loc[liquid_pattern["day4_volume_up"]], 5),
        "day4_volume_not_up": _outcome_summary(
            liquid_pattern.loc[~liquid_pattern["day4_volume_up"]], 5
        ),
        "day4_body_minus_1pct_or_worse": _outcome_summary(
            liquid_pattern.loc[liquid_pattern["day4_body_return"].le(-0.01)], 5
        ),
        "day4_body_between_0_and_minus_1pct": _outcome_summary(
            liquid_pattern.loc[liquid_pattern["day4_body_return"].gt(-0.01)], 5
        ),
        "prior_3d_gain_5pct_or_more": _outcome_summary(
            liquid_pattern.loc[liquid_pattern["prior_3d_gain"].ge(0.05)], 5
        ),
        "prior_3d_gain_under_5pct": _outcome_summary(
            liquid_pattern.loc[liquid_pattern["prior_3d_gain"].lt(0.05)], 5
        ),
    }

    timing_by_liquidity: dict[str, Any] = {}
    for threshold in TURNOVER_THRESHOLDS:
        label = "all" if threshold == 0 else f"turnover_{threshold // 1_000_000}m_plus"
        selected = pattern.loc[pattern["turnover_value"].ge(threshold)]
        timing_by_liquidity[label] = {
            "minimum_turnover_yen": threshold,
            "intraday_high_basis": _target_timing_summary(selected, "first_plus_5pct_high_day"),
            "closing_price_basis": _target_timing_summary(selected, "first_plus_5pct_close_day"),
        }

    timing_subgroups = {
        "prior_3d_gain_5pct_or_more": {
            "intraday_high_basis": _target_timing_summary(
                liquid_pattern.loc[liquid_pattern["prior_3d_gain"].ge(0.05)],
                "first_plus_5pct_high_day",
            ),
            "closing_price_basis": _target_timing_summary(
                liquid_pattern.loc[liquid_pattern["prior_3d_gain"].ge(0.05)],
                "first_plus_5pct_close_day",
            ),
        },
        "prior_3d_gain_under_5pct": {
            "intraday_high_basis": _target_timing_summary(
                liquid_pattern.loc[liquid_pattern["prior_3d_gain"].lt(0.05)],
                "first_plus_5pct_high_day",
            ),
            "closing_price_basis": _target_timing_summary(
                liquid_pattern.loc[liquid_pattern["prior_3d_gain"].lt(0.05)],
                "first_plus_5pct_close_day",
            ),
        },
    }

    return {
        "pattern": "three_up_one_down",
        "label": "3日連続陽線→4日目陰線",
        "entry_reference": "4日目終値",
        "continuation_definition": "各判定日の終値が4日目終値を下回る場合を続落、上回る場合を上昇",
        "latest_date": str(pd.to_datetime(frame["date"]).max().date()),
        "ticker_count": int(frame["ticker"].nunique()),
        "sample_count": int(pattern["three_up_one_down"].sum()),
        "by_horizon": by_horizon,
        "bearish_day_baseline": baseline_by_horizon,
        "liquidity": liquidity,
        "turnover_200m_subgroups_5d": subgroups,
        "plus_5pct_timing": {
            "maximum_observation_business_days": target_timing_window_days,
            "definition": (
                "4日目終値の1.05倍へ初めて到達した営業日。"
                "平均・中央値は60営業日以内に到達した事例のみ"
            ),
            "by_liquidity": timing_by_liquidity,
            "turnover_200m_subgroups": timing_subgroups,
        },
        "notes": [
            "陽線は終値>始値、陰線は終値<始値の厳密条件。寄引同値は除外",
            "+5%、-3%、-5%は4日目終値から判定期間中の高値・安値で判定",
            "同一銘柄の重複しない営業日系列で計算し、株式分割調整後価格を使用",
            "到達日数は右打ち切りを避けるため60営業日先まで確認できる事例だけを使用",
        ],
    }
