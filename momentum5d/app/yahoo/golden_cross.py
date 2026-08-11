from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class GoldenCrossBacktestConfig:
    moving_average_pairs: tuple[tuple[int, int], ...] = ((5, 25), (25, 75))
    volume_ratio_thresholds: tuple[float, ...] = (1.0, 1.2, 1.5, 2.0)
    horizon_days: int = 5
    take_profit: float = 0.05
    stop_loss: float = 0.03
    entry_gap_minimum: float = -0.03
    entry_gap_maximum: float = 0.015
    transaction_cost_bps: float = 20.0
    minimum_turnover_yen: float = 200_000_000.0
    maximum_candidates_per_day: int = 3
    minimum_calibration_trades: int = 30
    calibration_fraction: float = 0.70


def backtest_golden_cross_volume(
    features: pd.DataFrame,
    config: GoldenCrossBacktestConfig | None = None,
) -> dict[str, Any]:
    """Compare golden-cross signals using only information known at each close."""
    settings = config or GoldenCrossBacktestConfig()
    if features.empty:
        return _empty_report(settings)

    frame = _attach_signal_features(features, settings)
    evaluated = _attach_outcomes(frame, settings)
    available_dates = sorted(
        evaluated.loc[evaluated["trade_available"], "date"].drop_duplicates()
    )
    if not available_dates:
        return _empty_report(settings)
    split_index = min(
        max(int(len(available_dates) * settings.calibration_fraction), 1),
        len(available_dates) - 1,
    )
    validation_start = available_dates[split_index]

    strategies: dict[str, dict[str, Any]] = {}
    selected_trades: dict[str, pd.DataFrame] = {}
    for short_window, long_window in settings.moving_average_pairs:
        volume_rules: tuple[tuple[str, float | None], ...] = (
            ("previous_day", None),
            *(
                (f"prior20_{threshold:g}x", threshold)
                for threshold in settings.volume_ratio_thresholds
            ),
        )
        for volume_rule, threshold in volume_rules:
            key = f"gc_{short_window}_{long_window}_{volume_rule}"
            selected = _select_trades(
                evaluated,
                short_window=short_window,
                long_window=long_window,
                volume_threshold=threshold,
                config=settings,
            )
            selected_trades[key] = selected
            strategies[key] = {
                "label": _strategy_label(short_window, long_window, threshold),
                "short_window": short_window,
                "long_window": long_window,
                "volume_rule": volume_rule,
                "volume_ratio_threshold": threshold,
                "all": _summarize(selected),
                "calibration": _summarize(
                    selected.loc[selected["date"] < validation_start]
                ),
                "validation": _summarize(
                    selected.loc[selected["date"] >= validation_start]
                ),
            }

    eligible = [
        (key, values)
        for key, values in strategies.items()
        if values["calibration"]["trades"] >= settings.minimum_calibration_trades
    ]
    best_key = (
        max(
            eligible,
            key=lambda item: (
                item[1]["calibration"]["mean_net_return"] or -1.0,
                item[1]["calibration"]["target_hit_rate"] or -1.0,
            ),
        )[0]
        if eligible
        else None
    )
    best_trades = selected_trades.get(best_key, pd.DataFrame())
    return {
        "method": "golden_cross_volume_walk_forward_v1",
        "signal_timing": "営業日tの引け後に確定。t+1始値でエントリー",
        "leakage_control": "移動平均と出来高平均はtまで、結果はt+1以降だけを使用",
        "config": {
            "moving_average_pairs": [list(pair) for pair in settings.moving_average_pairs],
            "volume_ratio_thresholds": list(settings.volume_ratio_thresholds),
            "volume_average": "シグナル当日を含めない直前20営業日平均",
            "minimum_turnover_yen": settings.minimum_turnover_yen,
            "maximum_candidates_per_day": settings.maximum_candidates_per_day,
            "entry": "翌営業日始値",
            "take_profit": settings.take_profit,
            "stop_loss": -settings.stop_loss,
            "entry_gap_filter": [
                settings.entry_gap_minimum,
                settings.entry_gap_maximum,
            ],
            "same_day_target_and_stop": "stop_first",
            "holding_days": settings.horizon_days,
            "transaction_cost_bps": settings.transaction_cost_bps,
        },
        "period": {
            "start": str(available_dates[0]),
            "end": str(available_dates[-1]),
            "validation_start": str(validation_start),
            "calibration_fraction": settings.calibration_fraction,
        },
        "strategies": strategies,
        "selected_strategy": best_key,
        "selected_validation": (
            strategies[best_key]["validation"] if best_key is not None else None
        ),
        "selected_trade_examples": _trade_examples(best_trades),
        "limitations": [
            "決算発表の過去時点カレンダーがないため、決算跨ぎ除外は本検証に未反映",
            "日足で利確と損切りへ同日に到達した場合は損切りを先に処理",
            "選定条件は前半70%だけで決め、後半30%を未使用検証として表示",
        ],
    }


def _attach_signal_features(
    features: pd.DataFrame,
    config: GoldenCrossBacktestConfig,
) -> pd.DataFrame:
    frame = features.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    group = frame.groupby("ticker", sort=False)
    close = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    for window in sorted({window for pair in config.moving_average_pairs for window in pair}):
        frame[f"gc_ma_{window}"] = close.groupby(frame["ticker"], sort=False).transform(
            lambda values, period=window: values.rolling(
                period, min_periods=period
            ).mean()
        )
    prior_volume_average = group["volume"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=20).mean()
    )
    frame["gc_volume_ratio_prior20"] = volume / prior_volume_average
    frame["gc_volume_increased_previous_day"] = volume.gt(group["volume"].shift(1))
    for short_window, long_window in config.moving_average_pairs:
        short = frame[f"gc_ma_{short_window}"]
        long = frame[f"gc_ma_{long_window}"]
        previous_short = short.groupby(frame["ticker"], sort=False).shift(1)
        previous_long = long.groupby(frame["ticker"], sort=False).shift(1)
        frame[f"gc_{short_window}_{long_window}"] = (
            short.gt(long) & previous_short.le(previous_long)
        )
        frame[f"gc_strength_{short_window}_{long_window}"] = short / long - 1.0
    return frame


def _attach_outcomes(
    frame: pd.DataFrame,
    config: GoldenCrossBacktestConfig,
) -> pd.DataFrame:
    result = frame.copy()
    group = result.groupby("ticker", sort=False)
    result["entry_date"] = group["date"].shift(-1)
    result["entry_price"] = group["open"].shift(-1)
    result["entry_gap"] = result["entry_price"] / result["close"] - 1.0
    complete = result["entry_price"].gt(0)
    for day in range(1, config.horizon_days + 1):
        for column in ("date", "open", "high", "low", "close"):
            result[f"future_{column}_{day}"] = group[column].shift(-day)
        complete &= result[f"future_high_{day}"].notna()
        complete &= result[f"future_low_{day}"].notna()
        complete &= result[f"future_close_{day}"].notna()
    result["trade_available"] = complete

    entry = pd.to_numeric(result["entry_price"], errors="coerce")
    target = entry * (1.0 + config.take_profit)
    stop = entry * (1.0 - config.stop_loss)
    exit_price = pd.Series(np.nan, index=result.index, dtype="float64")
    exit_day = pd.Series(pd.NA, index=result.index, dtype="Int8")
    exit_reason = pd.Series("", index=result.index, dtype="object")
    unresolved = result["trade_available"].copy()
    for day in range(1, config.horizon_days + 1):
        open_ = pd.to_numeric(result[f"future_open_{day}"], errors="coerce")
        high = pd.to_numeric(result[f"future_high_{day}"], errors="coerce")
        low = pd.to_numeric(result[f"future_low_{day}"], errors="coerce")
        gap_stop = unresolved & open_.le(stop)
        intraday_stop = unresolved & ~gap_stop & low.le(stop)
        stopped = gap_stop | intraday_stop
        exit_price.loc[gap_stop] = open_.loc[gap_stop]
        exit_price.loc[intraday_stop] = stop.loc[intraday_stop]
        exit_day.loc[stopped] = day
        exit_reason.loc[stopped] = "stop_loss"
        unresolved &= ~stopped

        target_hit = unresolved & high.ge(target)
        exit_price.loc[target_hit] = target.loc[target_hit]
        exit_day.loc[target_hit] = day
        exit_reason.loc[target_hit] = "take_profit"
        unresolved &= ~target_hit

    time_exit = unresolved & result["trade_available"]
    exit_price.loc[time_exit] = pd.to_numeric(
        result.loc[time_exit, f"future_close_{config.horizon_days}"],
        errors="coerce",
    )
    exit_day.loc[time_exit] = config.horizon_days
    exit_reason.loc[time_exit] = "time_exit"
    result["exit_price"] = exit_price
    result["exit_day"] = exit_day
    result["exit_reason"] = exit_reason
    result["target_hit"] = exit_reason.eq("take_profit")
    result["stop_hit"] = exit_reason.eq("stop_loss")
    result["gross_return"] = exit_price / entry - 1.0
    result["net_return"] = (
        result["gross_return"] - config.transaction_cost_bps / 10_000.0
    )
    future_highs = result[
        [f"future_high_{day}" for day in range(1, config.horizon_days + 1)]
    ]
    future_lows = result[
        [f"future_low_{day}" for day in range(1, config.horizon_days + 1)]
    ]
    result["max_favorable_excursion"] = future_highs.max(axis=1) / entry - 1.0
    result["max_adverse_excursion"] = future_lows.min(axis=1) / entry - 1.0
    return result


def _select_trades(
    frame: pd.DataFrame,
    *,
    short_window: int,
    long_window: int,
    volume_threshold: float | None,
    config: GoldenCrossBacktestConfig,
) -> pd.DataFrame:
    mask = (
        frame[f"gc_{short_window}_{long_window}"].fillna(False)
        & frame["trade_available"]
        & pd.to_numeric(frame["turnover_value"], errors="coerce").ge(
            config.minimum_turnover_yen
        )
        & frame["entry_gap"].between(
            config.entry_gap_minimum,
            config.entry_gap_maximum,
        )
    )
    if volume_threshold is None:
        mask &= frame["gc_volume_increased_previous_day"].fillna(False)
    else:
        mask &= frame["gc_volume_ratio_prior20"].ge(volume_threshold)
    selected = frame.loc[mask].copy()
    selected = selected.sort_values(
        [
            "date",
            "gc_volume_ratio_prior20",
            f"gc_strength_{short_window}_{long_window}",
            "ticker",
        ],
        ascending=[True, False, False, True],
    )
    return (
        selected.groupby("date", sort=False, as_index=False)
        .head(config.maximum_candidates_per_day)
        .reset_index(drop=True)
    )


def _summarize(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "trades": 0,
            "target_hits": 0,
            "target_hit_rate": None,
            "win_rate": None,
            "mean_net_return": None,
            "median_net_return": None,
            "stop_hit_rate": None,
            "mean_mfe": None,
            "mean_mae": None,
            "profit_factor": None,
        }
    net = pd.to_numeric(trades["net_return"], errors="coerce")
    gains = net.loc[net > 0].sum()
    losses = -net.loc[net < 0].sum()
    return {
        "trades": len(trades),
        "target_hits": int(trades["target_hit"].sum()),
        "target_hit_rate": float(trades["target_hit"].mean()),
        "win_rate": float(net.gt(0).mean()),
        "mean_net_return": float(net.mean()),
        "median_net_return": float(net.median()),
        "stop_hit_rate": float(trades["stop_hit"].mean()),
        "mean_mfe": float(trades["max_favorable_excursion"].mean()),
        "mean_mae": float(trades["max_adverse_excursion"].mean()),
        "profit_factor": float(gains / losses) if losses > 0 else None,
    }


def _strategy_label(
    short_window: int,
    long_window: int,
    volume_threshold: float | None,
) -> str:
    volume = (
        "前日比増加"
        if volume_threshold is None
        else f"直前20日平均比{volume_threshold:g}倍以上"
    )
    return f"{short_window}日線×{long_window}日線GC＋出来高{volume}"


def _trade_examples(trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    columns = [
        "date",
        "ticker",
        "code",
        "entry_date",
        "entry_price",
        "exit_reason",
        "net_return",
        "gc_volume_ratio_prior20",
    ]
    available = [column for column in columns if column in trades]
    examples = trades.nlargest(min(10, len(trades)), "net_return")[available]
    return [
        {
            key: value.item() if hasattr(value, "item") else value
            for key, value in row.items()
        }
        for row in examples.to_dict("records")
    ]


def _empty_report(config: GoldenCrossBacktestConfig) -> dict[str, Any]:
    return {
        "method": "golden_cross_volume_walk_forward_v1",
        "config": {
            "moving_average_pairs": [list(pair) for pair in config.moving_average_pairs],
            "volume_ratio_thresholds": list(config.volume_ratio_thresholds),
        },
        "strategies": {},
        "selected_strategy": None,
        "selected_validation": None,
        "selected_trade_examples": [],
        "error": "評価可能なデータがありません",
    }
