from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class PerfectOrderBacktestConfig:
    moving_average_windows: tuple[int, int, int] = (5, 25, 75)
    slope_lookback_days: int = 5
    minimum_holding_days: int = 2
    maximum_holding_days: int = 7
    stop_loss: float = 0.03
    transaction_cost_bps: float = 20.0
    minimum_turnover_yen: float = 200_000_000.0
    maximum_candidates_per_day: int = 3
    entry_gap_minimum: float = -0.03
    entry_gap_maximum: float = 0.015
    maximum_entry_rsi: float = 70.0
    maximum_entry_ma25_deviation: float = 0.08
    calibration_fraction: float = 0.70
    minimum_calibration_trades: int = 30


PULLBACK_RULES: tuple[tuple[str, str], ...] = (
    ("ma5_touch", "5日線タッチ"),
    ("ma25_touch", "25日線タッチ"),
    ("three_day_pullback", "3日下落後も25日線上"),
    ("ma25_bullish_reversal", "25日線タッチ後の反転陽線"),
)

OVERHEAT_RULES: tuple[tuple[str, str], ...] = (
    ("rsi_65", "RSI65以上"),
    ("rsi_70", "RSI70以上"),
    ("rsi_75", "RSI75以上"),
    ("ma25_dev_5", "25日線乖離+5%以上"),
    ("ma25_dev_8", "25日線乖離+8%以上"),
    ("ma25_dev_12", "25日線乖離+12%以上"),
    ("rsi70_or_ma25_dev8", "RSI70以上または25日線乖離+8%以上"),
)


def backtest_perfect_order_pullbacks(
    features: pd.DataFrame,
    config: PerfectOrderBacktestConfig | None = None,
) -> dict[str, Any]:
    """Buy rising perfect-order pullbacks and exit after confirmed overheat."""
    settings = config or PerfectOrderBacktestConfig()
    if features.empty:
        return _empty_report(settings)
    signals = _attach_signal_features(features, settings)
    evaluated = _attach_future_values(signals, settings)
    dates = sorted(evaluated.loc[evaluated["trade_available"], "date"].unique())
    if len(dates) < 2:
        return _empty_report(settings)
    split_index = min(
        max(int(len(dates) * settings.calibration_fraction), 1),
        len(dates) - 1,
    )
    validation_start = dates[split_index]

    strategies: dict[str, dict[str, Any]] = {}
    trades_by_strategy: dict[str, pd.DataFrame] = {}
    for pullback_key, pullback_label in PULLBACK_RULES:
        candidates = _select_candidates(
            evaluated,
            pullback_rule=pullback_key,
            config=settings,
        )
        for exit_key, exit_label in OVERHEAT_RULES:
            key = f"{pullback_key}__{exit_key}"
            trades = _simulate_exits(
                candidates,
                exit_rule=exit_key,
                config=settings,
            )
            trades_by_strategy[key] = trades
            strategies[key] = {
                "label": f"{pullback_label}→{exit_label}",
                "pullback_rule": pullback_key,
                "overheat_exit_rule": exit_key,
                "all": _summarize(trades),
                "calibration": _summarize(trades.loc[trades["date"] < validation_start]),
                "validation": _summarize(trades.loc[trades["date"] >= validation_start]),
            }

    eligible = [
        (key, result)
        for key, result in strategies.items()
        if result["calibration"]["trades"] >= settings.minimum_calibration_trades
    ]
    selected_key = (
        max(
            eligible,
            key=lambda item: (
                item[1]["calibration"]["mean_net_return"] or -1.0,
                item[1]["calibration"]["win_rate"] or -1.0,
            ),
        )[0]
        if eligible
        else None
    )
    return {
        "method": "perfect_order_pullback_overheat_walk_forward_v1",
        "signal_timing": "営業日tの引け後に押し目確定、t+1始値で買い",
        "exit_timing": "過熱は引け後に確定し翌営業日始値で売却",
        "leakage_control": "移動平均・傾き・押し目は当日まで、売却判定は各将来日の引けまで",
        "config": {
            "perfect_order": "MA5 > MA25 > MA75、各線が5日前より上向き",
            "pullback_rules": {key: label for key, label in PULLBACK_RULES},
            "overheat_rules": {key: label for key, label in OVERHEAT_RULES},
            "entry": "翌営業日始値",
            "entry_gap_filter": [
                settings.entry_gap_minimum,
                settings.entry_gap_maximum,
            ],
            "minimum_holding_days": settings.minimum_holding_days,
            "maximum_holding_days": settings.maximum_holding_days,
            "stop_loss": -settings.stop_loss,
            "transaction_cost_bps": settings.transaction_cost_bps,
            "minimum_turnover_yen": settings.minimum_turnover_yen,
            "maximum_candidates_per_day": settings.maximum_candidates_per_day,
            "same_day_stop_and_rebound": "stop_first",
        },
        "period": {
            "start": str(dates[0]),
            "end": str(dates[-1]),
            "validation_start": str(validation_start),
            "calibration_fraction": settings.calibration_fraction,
        },
        "strategies": strategies,
        "selected_strategy": selected_key,
        "selected_validation": (strategies[selected_key]["validation"] if selected_key else None),
        "selected_trade_examples": _trade_examples(
            trades_by_strategy.get(selected_key, pd.DataFrame())
        ),
        "limitations": [
            "過去時点の決算予定履歴がないため決算跨ぎ除外は未反映",
            "日足内で損切りと反発が同時に起きた場合は損切りを先に処理",
            "前半70%だけで条件を選び、後半30%を未使用検証として表示",
        ],
    }


def _attach_signal_features(
    features: pd.DataFrame,
    config: PerfectOrderBacktestConfig,
) -> pd.DataFrame:
    frame = features.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    ticker = frame["ticker"]
    close = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    low = _adjusted_price(frame, "low")
    open_ = _adjusted_price(frame, "open")
    group_close = close.groupby(ticker, sort=False)
    short_window, middle_window, long_window = config.moving_average_windows
    for window in config.moving_average_windows:
        moving_average = group_close.transform(
            lambda values, period=window: values.rolling(period, min_periods=period).mean()
        )
        frame[f"po_ma_{window}"] = moving_average
        frame[f"po_ma_{window}_slope"] = (
            moving_average
            / moving_average.groupby(ticker, sort=False).shift(config.slope_lookback_days)
            - 1.0
        )
    short = frame[f"po_ma_{short_window}"]
    middle = frame[f"po_ma_{middle_window}"]
    long = frame[f"po_ma_{long_window}"]
    frame["po_perfect_order"] = (
        short.gt(middle)
        & middle.gt(long)
        & frame[f"po_ma_{short_window}_slope"].gt(0)
        & frame[f"po_ma_{middle_window}_slope"].gt(0)
        & frame[f"po_ma_{long_window}_slope"].gt(0)
    )
    frame["po_return_3d"] = close / group_close.shift(3) - 1.0
    frame["po_previous_return_1d"] = (
        pd.to_numeric(frame.get("return_1d"), errors="coerce").groupby(ticker, sort=False).shift(1)
    )
    frame["po_ma25_deviation"] = close / middle - 1.0
    frame["po_order_spread"] = short / long - 1.0
    frame["po_pullback_score"] = (
        0.55 * (1.0 - frame["po_ma25_deviation"].abs() / 0.08).clip(0, 1)
        + 0.25 * (frame["po_order_spread"] / 0.15).clip(0, 1)
        + 0.20 * (pd.to_numeric(frame.get("volume_ratio_1_20"), errors="coerce") / 2.0).clip(0, 1)
    )
    declining = pd.to_numeric(frame.get("return_1d"), errors="coerce").le(0)
    frame["po_pullback_ma5_touch"] = declining & low.le(short * 1.005) & close.ge(short)
    frame["po_pullback_ma25_touch"] = (
        declining & low.le(middle * 1.01) & close.ge(middle) & close.le(short * 1.02)
    )
    frame["po_pullback_three_day_pullback"] = frame["po_return_3d"].between(
        -0.08, -0.01
    ) & close.ge(middle)
    frame["po_pullback_ma25_bullish_reversal"] = (
        low.le(middle * 1.01)
        & close.ge(middle)
        & close.gt(open_)
        & frame["po_previous_return_1d"].lt(0)
    )

    rolling_mean20 = group_close.transform(lambda values: values.rolling(20, min_periods=20).mean())
    rolling_std20 = group_close.transform(lambda values: values.rolling(20, min_periods=20).std())
    frame["po_bollinger_upper"] = rolling_mean20 + 2.0 * rolling_std20
    return frame


def _attach_future_values(
    frame: pd.DataFrame,
    config: PerfectOrderBacktestConfig,
) -> pd.DataFrame:
    result = frame.copy()
    group = result.groupby("ticker", sort=False)
    result["entry_date"] = group["date"].shift(-1)
    result["entry_price"] = group["open"].shift(-1)
    result["entry_gap"] = result["entry_price"] / result["close"] - 1.0
    complete = result["entry_price"].gt(0)
    future_columns = (
        "date",
        "open",
        "high",
        "low",
        "close",
        "rsi_14",
        "po_ma25_deviation",
        "po_bollinger_upper",
    )
    for day in range(1, config.maximum_holding_days + 1):
        for column in future_columns:
            result[f"future_{column}_{day}"] = group[column].shift(-day)
        complete &= result[f"future_close_{day}"].notna()
        complete &= result[f"future_open_{day}"].notna()
    result["trade_available"] = complete
    return result


def _select_candidates(
    frame: pd.DataFrame,
    *,
    pullback_rule: str,
    config: PerfectOrderBacktestConfig,
) -> pd.DataFrame:
    mask = (
        frame["po_perfect_order"].fillna(False)
        & frame[f"po_pullback_{pullback_rule}"].fillna(False)
        & frame["trade_available"]
        & pd.to_numeric(frame["turnover_value"], errors="coerce").ge(config.minimum_turnover_yen)
        & frame["entry_gap"].between(
            config.entry_gap_minimum,
            config.entry_gap_maximum,
        )
        & pd.to_numeric(frame["rsi_14"], errors="coerce").lt(config.maximum_entry_rsi)
        & frame["po_ma25_deviation"].lt(config.maximum_entry_ma25_deviation)
    )
    return (
        frame.loc[mask]
        .sort_values(
            ["date", "po_pullback_score", "turnover_value", "ticker"],
            ascending=[True, False, False, True],
        )
        .groupby("date", sort=False, as_index=False)
        .head(config.maximum_candidates_per_day)
        .reset_index(drop=True)
    )


def _simulate_exits(
    candidates: pd.DataFrame,
    *,
    exit_rule: str,
    config: PerfectOrderBacktestConfig,
) -> pd.DataFrame:
    result = candidates.copy()
    if result.empty:
        for column in (
            "exit_price",
            "exit_day",
            "exit_reason",
            "net_return",
            "trade_win",
            "five_pct_reached",
            "stop_hit",
        ):
            result[column] = pd.Series(dtype="float64")
        return result
    entry = pd.to_numeric(result["entry_price"], errors="coerce")
    stop = entry * (1.0 - config.stop_loss)
    target = entry * 1.05
    exit_price = pd.Series(np.nan, index=result.index, dtype="float64")
    exit_day = pd.Series(pd.NA, index=result.index, dtype="Int8")
    exit_reason = pd.Series("", index=result.index, dtype="object")
    five_pct_reached = pd.Series(False, index=result.index, dtype="bool")
    unresolved = pd.Series(True, index=result.index, dtype="bool")
    scheduled_overheat = pd.Series(False, index=result.index, dtype="bool")
    for day in range(1, config.maximum_holding_days + 1):
        open_ = pd.to_numeric(result[f"future_open_{day}"], errors="coerce")
        high = pd.to_numeric(result[f"future_high_{day}"], errors="coerce")
        low = pd.to_numeric(result[f"future_low_{day}"], errors="coerce")

        overheat_exit = unresolved & scheduled_overheat
        exit_price.loc[overheat_exit] = open_.loc[overheat_exit]
        exit_day.loc[overheat_exit] = day
        exit_reason.loc[overheat_exit] = "overheat"
        five_pct_reached.loc[overheat_exit] |= open_.loc[overheat_exit].ge(
            target.loc[overheat_exit]
        )
        unresolved &= ~overheat_exit
        scheduled_overheat.loc[:] = False

        gap_stop = unresolved & open_.le(stop)
        intraday_stop = unresolved & ~gap_stop & low.le(stop)
        stopped = gap_stop | intraday_stop
        exit_price.loc[gap_stop] = open_.loc[gap_stop]
        exit_price.loc[intraday_stop] = stop.loc[intraday_stop]
        exit_day.loc[stopped] = day
        exit_reason.loc[stopped] = "stop_loss"
        unresolved &= ~stopped

        five_pct_reached.loc[unresolved] |= high.loc[unresolved].ge(target.loc[unresolved])
        if config.minimum_holding_days <= day < config.maximum_holding_days:
            scheduled_overheat = unresolved & _overheat_signal(
                result,
                day=day,
                rule=exit_rule,
            )

    time_exit = unresolved
    exit_price.loc[time_exit] = pd.to_numeric(
        result.loc[time_exit, f"future_close_{config.maximum_holding_days}"],
        errors="coerce",
    )
    exit_day.loc[time_exit] = config.maximum_holding_days
    exit_reason.loc[time_exit] = "time_exit"
    result["exit_price"] = exit_price
    result["exit_day"] = exit_day
    result["exit_reason"] = exit_reason
    result["gross_return"] = exit_price / entry - 1.0
    result["net_return"] = result["gross_return"] - config.transaction_cost_bps / 10_000.0
    result["trade_win"] = result["net_return"].gt(0)
    result["five_pct_reached"] = five_pct_reached
    result["stop_hit"] = exit_reason.eq("stop_loss")
    return result


def _overheat_signal(
    frame: pd.DataFrame,
    *,
    day: int,
    rule: str,
) -> pd.Series:
    rsi = pd.to_numeric(frame[f"future_rsi_14_{day}"], errors="coerce")
    deviation = pd.to_numeric(frame[f"future_po_ma25_deviation_{day}"], errors="coerce")
    close = pd.to_numeric(frame[f"future_close_{day}"], errors="coerce")
    upper = pd.to_numeric(frame[f"future_po_bollinger_upper_{day}"], errors="coerce")
    if rule.startswith("rsi_"):
        return rsi.ge(float(rule.removeprefix("rsi_")))
    if rule.startswith("ma25_dev_"):
        threshold = float(rule.removeprefix("ma25_dev_")) / 100.0
        return deviation.ge(threshold)
    if rule == "rsi70_or_ma25_dev8":
        return rsi.ge(70.0) | deviation.ge(0.08)
    return close.ge(upper)


def _summarize(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate": None,
            "five_pct_reach_rate": None,
            "mean_net_return": None,
            "median_net_return": None,
            "stop_hit_rate": None,
            "overheat_exit_rate": None,
            "mean_holding_days": None,
            "profit_factor": None,
        }
    net = pd.to_numeric(trades["net_return"], errors="coerce")
    gains = net.loc[net > 0].sum()
    losses = -net.loc[net < 0].sum()
    return {
        "trades": len(trades),
        "win_rate": float(trades["trade_win"].mean()),
        "five_pct_reach_rate": float(trades["five_pct_reached"].mean()),
        "mean_net_return": float(net.mean()),
        "median_net_return": float(net.median()),
        "stop_hit_rate": float(trades["stop_hit"].mean()),
        "overheat_exit_rate": float(trades["exit_reason"].eq("overheat").mean()),
        "mean_holding_days": float(pd.to_numeric(trades["exit_day"]).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else None,
    }


def _trade_examples(trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    columns = [
        "date",
        "ticker",
        "code",
        "entry_date",
        "entry_price",
        "exit_price",
        "exit_day",
        "exit_reason",
        "net_return",
    ]
    examples = trades.nlargest(min(10, len(trades)), "net_return")[columns]
    return [
        {key: value.item() if hasattr(value, "item") else value for key, value in row.items()}
        for row in examples.to_dict("records")
    ]


def _adjusted_price(frame: pd.DataFrame, column: str) -> pd.Series:
    ratio = pd.to_numeric(frame["adjusted_close"], errors="coerce") / pd.to_numeric(
        frame["close"], errors="coerce"
    )
    return pd.to_numeric(frame[column], errors="coerce") * ratio


def _empty_report(config: PerfectOrderBacktestConfig) -> dict[str, Any]:
    return {
        "method": "perfect_order_pullback_overheat_walk_forward_v1",
        "config": {
            "minimum_holding_days": config.minimum_holding_days,
            "maximum_holding_days": config.maximum_holding_days,
        },
        "strategies": {},
        "selected_strategy": None,
        "selected_validation": None,
        "selected_trade_examples": [],
        "error": "評価可能なデータがありません",
    }
