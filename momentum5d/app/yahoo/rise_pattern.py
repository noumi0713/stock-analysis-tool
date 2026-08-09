from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.yahoo.bottom_patterns import FEATURE_SPECS, SHAPE_LABELS

STRONG_SHAPES = frozenset(
    {"sharp_selloff", "capitulation_reversal", "rounded_base"}
)
STRONG_SHAPE_MIN_TURNOVER = 200_000_000.0


@dataclass(frozen=True)
class RisePatternConfig:
    min_signal_rate: float = 0.80
    min_samples: int = 30
    prior_strength: float = 20.0
    horizon_days: int = 5
    test_days: int = 60
    top_n: int = 20
    transaction_cost_bps: float = 20.0
    fixed_stop_pcts: tuple[float, ...] = (0.03, 0.04, 0.05)
    atr_stop_multipliers: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)


def add_latest_rise_pattern_signals(
    features: pd.DataFrame,
    bottom_events: pd.DataFrame,
    *,
    config: RisePatternConfig | None = None,
) -> pd.DataFrame:
    """Attach actionable pattern signals using information known by the latest close."""
    config = config or RisePatternConfig()
    frame = _add_live_bottom_features(features)
    frame["rise_pattern_probability"] = 0.0
    frame["rise_pattern_samples"] = 0
    frame["rise_pattern_signal"] = False
    frame["rise_pattern_shape"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame["rise_pattern_reason"] = ""
    if frame.empty or bottom_events.empty:
        return frame

    profiles = _calibrate_profiles(bottom_events, config)
    latest_date = frame["date"].max()
    latest_indexes = frame.index[frame["date"] == latest_date]
    for index in latest_indexes:
        row = frame.loc[index]
        if not bool(row["rise_pattern_live_bottom"]):
            continue
        profile = profiles.get(str(row["_rise_shape"]))
        if profile is None:
            continue
        subtype = _match_subtype(row, profile)
        if subtype is None:
            continue
        probability = float(subtype["smoothed_success_rate"])
        samples = int(subtype["samples"])
        signal = probability >= config.min_signal_rate and samples >= config.min_samples
        frame.at[index, "rise_pattern_probability"] = probability
        frame.at[index, "rise_pattern_samples"] = samples
        frame.at[index, "rise_pattern_signal"] = signal
        frame.at[index, "rise_pattern_shape"] = str(row["_rise_shape"])
        frame.at[index, "rise_pattern_reason"] = _signal_reason(
            str(row["_rise_shape"]), subtype, probability
        )
    return frame


def backtest_rise_pattern_signals(
    features: pd.DataFrame,
    bottom_events: pd.DataFrame,
    *,
    config: RisePatternConfig | None = None,
) -> dict[str, Any]:
    """Walk forward the pattern detector and compare it with the prior inflow signal."""
    config = config or RisePatternConfig()
    if features.empty or bottom_events.empty:
        return {"method": "walk_forward_rise_pattern_v1", "strategies": {}}

    frame = _attach_trade_outcomes(_add_live_bottom_features(features), config)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    all_dates = sorted(frame["date"].drop_duplicates())
    date_position = {value: position for position, value in enumerate(all_dates)}
    complete_dates = [
        value
        for value in all_dates
        if frame.loc[frame["date"] == value, "trade_outcome_available"].any()
    ]
    test_dates = complete_dates[-config.test_days :]
    if not test_dates:
        return {"method": "walk_forward_rise_pattern_v1", "strategies": {}}

    events = bottom_events.copy()
    events["date"] = pd.to_datetime(events["date"]).dt.date
    events["_date_position"] = events["date"].map(date_position)
    events = events.loc[events["_date_position"].notna()].copy()

    strategy_rows: dict[str, list[pd.DataFrame]] = {
        "observed_inflow": [],
        "rise_pattern": [],
        "rise_pattern_reversal": [],
        "rise_pattern_with_inflow": [],
        "strong_shape_200m": [],
        "strong_shape_200m_signal": [],
        "combined": [],
    }
    for test_date in test_dates:
        position = date_position[test_date]
        training = events.loc[
            events["_date_position"] <= position - config.horizon_days
        ]
        profiles = _calibrate_profiles(training, config)
        day = frame.loc[frame["date"] == test_date].copy()
        day = _score_day_patterns(day, profiles, config)
        eligible = day.loc[
            day["trade_outcome_available"].fillna(False)
            & (day["turnover_value"].fillna(0) >= 10_000_000)
            & day["rsi_14"].fillna(100).le(82.0)
        ].copy()
        if eligible.empty:
            continue

        old_mask = (
            eligible["observed_inflow_confirmed"].fillna(False).astype(bool)
            & eligible["return_1d"].between(0.002, 0.10)
            & eligible["return_5d"].between(-0.05, 0.18)
        )
        pattern_mask = eligible["rise_pattern_signal"].fillna(False).astype(bool)
        reversal_mask = pattern_mask & eligible["rise_pattern_reversal_confirmed"].fillna(
            False
        ).astype(bool)
        pattern_with_inflow_mask = pattern_mask & eligible[
            "observed_inflow_confirmed"
        ].fillna(False).astype(bool)
        strong_shape_mask = (
            eligible["rise_pattern_live_bottom"].fillna(False).astype(bool)
            & eligible["_rise_shape"].isin(STRONG_SHAPES)
            & eligible["turnover_value"].fillna(0).ge(STRONG_SHAPE_MIN_TURNOVER)
        )
        strong_shape_signal_mask = strong_shape_mask & pattern_mask
        old = _select_top(
            eligible.loc[old_mask], "observed_inflow_score", config.top_n
        )
        pattern = _select_top(
            eligible.loc[pattern_mask], "rise_pattern_probability", config.top_n
        )
        reversal = _select_top(
            eligible.loc[reversal_mask], "rise_pattern_probability", config.top_n
        )
        pattern_with_inflow = _select_top(
            eligible.loc[pattern_with_inflow_mask],
            "rise_pattern_probability",
            config.top_n,
        )
        strong_shape = eligible.loc[strong_shape_mask].copy()
        strong_shape["_strong_shape_score"] = strong_shape[
            "rise_pattern_probability"
        ].where(
            strong_shape["rise_pattern_probability"].gt(0),
            strong_shape["observed_inflow_score"].fillna(0),
        )
        strong_shape = _select_top(
            strong_shape, "_strong_shape_score", config.top_n
        )
        strong_shape_signal = _select_top(
            eligible.loc[strong_shape_signal_mask],
            "rise_pattern_probability",
            config.top_n,
        )
        combined = eligible.loc[old_mask | pattern_mask].copy()
        combined["_combined_score"] = combined[
            ["observed_inflow_score", "rise_pattern_probability"]
        ].max(axis=1)
        combined = _select_top(combined, "_combined_score", config.top_n)
        strategy_rows["observed_inflow"].append(old)
        strategy_rows["rise_pattern"].append(pattern)
        strategy_rows["rise_pattern_reversal"].append(reversal)
        strategy_rows["rise_pattern_with_inflow"].append(pattern_with_inflow)
        strategy_rows["strong_shape_200m"].append(strong_shape)
        strategy_rows["strong_shape_200m_signal"].append(strong_shape_signal)
        strategy_rows["combined"].append(combined)

    summaries: dict[str, Any] = {}
    strategy_trades: dict[str, pd.DataFrame] = {}
    for name, rows in strategy_rows.items():
        trades = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        strategy_trades[name] = trades
        summaries[name] = _strategy_summary(trades, test_dates, config)

    pattern_trades = strategy_trades["rise_pattern"]
    risk_management = {
        "assumptions": {
            "take_profit_pct": 0.05,
            "holding_days": config.horizon_days,
            "transaction_cost_bps": config.transaction_cost_bps,
            "same_day_stop_and_target": "stop_first",
            "gap_through_stop": "exit_at_open",
            "atr_period": 14,
        },
        "no_stop": _risk_summary(
            pattern_trades,
            test_dates,
            config,
            return_column="rise_trade_net_return",
            target_column="rise_trade_target_hit",
        ),
    }
    for stop_pct in config.fixed_stop_pcts:
        key = _fixed_stop_key(stop_pct)
        prefix = f"rise_trade_{key}"
        risk_management[key] = _risk_summary(
            pattern_trades,
            test_dates,
            config,
            return_column=f"{prefix}_net_return",
            target_column=f"{prefix}_target_hit",
            stop_column=f"{prefix}_stop_hit",
            ambiguous_column=f"{prefix}_ambiguous_both_hit",
            stop_distance_column=f"{prefix}_stop_pct",
        )
    for multiplier in config.atr_stop_multipliers:
        key = _atr_stop_key(multiplier)
        prefix = f"rise_trade_{key}"
        risk_management[key] = _risk_summary(
            pattern_trades,
            test_dates,
            config,
            return_column=f"{prefix}_net_return",
            target_column=f"{prefix}_target_hit",
            stop_column=f"{prefix}_stop_hit",
            ambiguous_column=f"{prefix}_ambiguous_both_hit",
            stop_distance_column=f"{prefix}_stop_pct",
        )

    strong_shape_trades = strategy_trades["strong_shape_200m"]
    strong_shape_risk_management = {
        "assumptions": risk_management["assumptions"],
        "no_stop": _risk_summary(
            strong_shape_trades,
            test_dates,
            config,
            return_column="rise_trade_net_return",
            target_column="rise_trade_target_hit",
        ),
    }
    for stop_pct in config.fixed_stop_pcts:
        key = _fixed_stop_key(stop_pct)
        prefix = f"rise_trade_{key}"
        strong_shape_risk_management[key] = _risk_summary(
            strong_shape_trades,
            test_dates,
            config,
            return_column=f"{prefix}_net_return",
            target_column=f"{prefix}_target_hit",
            stop_column=f"{prefix}_stop_hit",
            ambiguous_column=f"{prefix}_ambiguous_both_hit",
            stop_distance_column=f"{prefix}_stop_pct",
        )
    for multiplier in config.atr_stop_multipliers:
        key = _atr_stop_key(multiplier)
        prefix = f"rise_trade_{key}"
        strong_shape_risk_management[key] = _risk_summary(
            strong_shape_trades,
            test_dates,
            config,
            return_column=f"{prefix}_net_return",
            target_column=f"{prefix}_target_hit",
            stop_column=f"{prefix}_stop_hit",
            ambiguous_column=f"{prefix}_ambiguous_both_hit",
            stop_distance_column=f"{prefix}_stop_pct",
        )

    return {
        "method": "walk_forward_rise_pattern_v1",
        "signal_timing": "当日引けで検知し翌営業日始値でエントリー",
        "leakage_control": (
            "各評価日の5営業日前までに結果確定した底値イベントだけで特徴差と閾値を再学習"
        ),
        "config": asdict(config),
        "test_start": test_dates[0].isoformat(),
        "test_end": test_dates[-1].isoformat(),
        "test_days": len(test_dates),
        "strategies": summaries,
        "strong_shape_200m_definition": {
            "shapes": sorted(STRONG_SHAPES),
            "min_turnover": int(STRONG_SHAPE_MIN_TURNOVER),
            "entry": "next_trading_day_open",
            "top_n_per_day": config.top_n,
            "raw_ranking": "walk-forward subtype probability, fallback observed inflow score",
            "signal_overlay": "existing walk-forward probability >= 80% and samples >= 30",
        },
        "rise_pattern_risk_management": risk_management,
        "strong_shape_200m_risk_management": strong_shape_risk_management,
    }


def _add_live_bottom_features(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    ratio = frame["adjusted_close"] / frame["close"]
    frame["_rise_adjusted_low"] = frame["low"] * ratio
    frame["_rise_adjusted_high"] = frame["high"] * ratio
    low_group = frame.groupby("ticker", sort=False)["_rise_adjusted_low"]
    high_group = frame.groupby("ticker", sort=False)["_rise_adjusted_high"]
    prior_low_1 = low_group.shift(1)
    prior_low_2 = low_group.shift(2)
    prior_high_20 = high_group.transform(
        lambda values: values.shift(1).rolling(20, min_periods=20).max()
    )
    prior_bottom = low_group.transform(
        lambda values: values.shift(4).rolling(17, min_periods=17).min()
    )
    current_low = frame["_rise_adjusted_low"]
    frame["rise_pattern_live_bottom"] = (
        current_low.lt(pd.concat([prior_low_1, prior_low_2], axis=1).min(axis=1))
        & current_low.div(prior_high_20).sub(1).le(-0.04)
    )

    double_bottom = (
        prior_bottom.gt(0)
        & current_low.div(prior_bottom).sub(1).abs().le(0.03)
    )
    candle_span = frame["high"] - frame["low"]
    close_location = (frame["close"] - frame["low"]).div(candle_span.where(candle_span > 0))
    frame["rise_pattern_reversal_confirmed"] = close_location.fillna(0.5).ge(0.55)
    capitulation = (
        frame["return_20d"].le(-0.10)
        & frame["volume_ratio_5_20"].ge(1.20)
        & close_location.fillna(0.5).ge(0.55)
    )
    compression = frame["range_width_10d"].le(0.07) & frame["volatility_10d"].le(0.02)
    sharp = frame["return_5d"].le(-0.06)
    rounded = frame["return_20d"].le(-0.05) & frame["return_5d"].ge(-0.03)
    frame["_rise_shape"] = np.select(
        [double_bottom, capitulation, compression, sharp, rounded],
        [
            "double_bottom",
            "capitulation_reversal",
            "compression_base",
            "sharp_selloff",
            "rounded_base",
        ],
        default="other_swing_low",
    )
    return frame


def _calibrate_profiles(
    events: pd.DataFrame,
    config: RisePatternConfig,
) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    if events.empty:
        return profiles
    for shape, values in events.groupby("shape", sort=False):
        if len(values) < config.min_samples or values["target_5pct"].nunique() < 2:
            continue
        comparisons = _feature_comparison(values)
        if len(comparisons) < 2:
            continue
        selected = [comparisons[0]["feature"], comparisons[1]["feature"]]
        work = values.dropna(subset=selected).copy()
        if len(work) < config.min_samples:
            continue
        thresholds = {feature: float(work[feature].median()) for feature in selected}
        parent_rate = float(work["target_5pct"].mean())
        work["_side_1"] = np.where(work[selected[0]] >= thresholds[selected[0]], "high", "low")
        work["_side_2"] = np.where(work[selected[1]] >= thresholds[selected[1]], "high", "low")
        subtypes: dict[tuple[str, str], dict[str, Any]] = {}
        for sides, group in work.groupby(["_side_1", "_side_2"], sort=False):
            samples = int(len(group))
            successes = int(group["target_5pct"].sum())
            rate = successes / samples
            smoothed = (
                successes + parent_rate * config.prior_strength
            ) / (samples + config.prior_strength)
            subtypes[(str(sides[0]), str(sides[1]))] = {
                "samples": samples,
                "successes": successes,
                "success_rate": rate,
                "smoothed_success_rate": float(smoothed),
                "sides": [str(sides[0]), str(sides[1])],
            }
        profiles[str(shape)] = {
            "features": selected,
            "thresholds": thresholds,
            "subtypes": subtypes,
        }
    return profiles


def _feature_comparison(values: pd.DataFrame) -> list[dict[str, Any]]:
    hits = values.loc[values["target_5pct"]]
    misses = values.loc[~values["target_5pct"]]
    comparisons: list[dict[str, Any]] = []
    for feature in FEATURE_SPECS:
        all_values = pd.to_numeric(values[feature], errors="coerce").dropna()
        hit_values = pd.to_numeric(hits[feature], errors="coerce").dropna()
        miss_values = pd.to_numeric(misses[feature], errors="coerce").dropna()
        if all_values.empty or hit_values.empty or miss_values.empty:
            continue
        q25, q75 = all_values.quantile([0.25, 0.75])
        iqr = float(q75 - q25)
        gap = float(hit_values.median() - miss_values.median())
        comparisons.append(
            {"feature": feature, "effect": gap / iqr if iqr > 0 else 0.0}
        )
    return sorted(comparisons, key=lambda item: -abs(item["effect"]))


def _score_day_patterns(
    day: pd.DataFrame,
    profiles: dict[str, dict[str, Any]],
    config: RisePatternConfig,
) -> pd.DataFrame:
    result = day.copy()
    result["rise_pattern_probability"] = 0.0
    result["rise_pattern_samples"] = 0
    result["rise_pattern_signal"] = False
    for index in result.index[result["rise_pattern_live_bottom"].fillna(False)]:
        row = result.loc[index]
        profile = profiles.get(str(row["_rise_shape"]))
        if profile is None:
            continue
        subtype = _match_subtype(row, profile)
        if subtype is None:
            continue
        probability = float(subtype["smoothed_success_rate"])
        samples = int(subtype["samples"])
        result.at[index, "rise_pattern_probability"] = probability
        result.at[index, "rise_pattern_samples"] = samples
        result.at[index, "rise_pattern_signal"] = (
            probability >= config.min_signal_rate and samples >= config.min_samples
        )
    return result


def _match_subtype(
    row: pd.Series,
    profile: dict[str, Any],
) -> dict[str, Any] | None:
    features = profile["features"]
    if any(pd.isna(row.get(feature)) for feature in features):
        return None
    sides = tuple(
        "high" if float(row[feature]) >= profile["thresholds"][feature] else "low"
        for feature in features
    )
    return profile["subtypes"].get(sides)


def _signal_reason(shape: str, subtype: dict[str, Any], probability: float) -> str:
    return (
        f"{SHAPE_LABELS.get(shape, shape)}・過去類似{subtype['samples']}件・"
        f"補正+5%率{probability:.0%}"
    )


def _attach_trade_outcomes(
    features: pd.DataFrame,
    config: RisePatternConfig,
) -> pd.DataFrame:
    frame = features.copy()
    ratio = frame["adjusted_close"] / frame["close"]
    adjusted_open = frame["open"] * ratio
    adjusted_high = frame["high"] * ratio
    adjusted_low = frame["low"] * ratio
    group_key = frame["ticker"]
    entry = adjusted_open.groupby(group_key, sort=False).shift(-1)
    future_opens = [
        adjusted_open.groupby(group_key, sort=False).shift(-offset)
        for offset in range(1, config.horizon_days + 1)
    ]
    future_highs = [
        adjusted_high.groupby(group_key, sort=False).shift(-offset)
        for offset in range(1, config.horizon_days + 1)
    ]
    future_lows = [
        adjusted_low.groupby(group_key, sort=False).shift(-offset)
        for offset in range(1, config.horizon_days + 1)
    ]
    future_open = pd.concat(future_opens, axis=1)
    future_high = pd.concat(future_highs, axis=1)
    future_low = pd.concat(future_lows, axis=1)
    exit_close = frame["adjusted_close"].groupby(group_key, sort=False).shift(
        -config.horizon_days
    )
    complete = (
        entry.notna()
        & exit_close.notna()
        & future_open.notna().all(axis=1)
        & future_high.notna().all(axis=1)
        & future_low.notna().all(axis=1)
    )
    target = entry * 1.05
    hit = future_high.ge(target, axis=0).any(axis=1) & complete
    frame["trade_outcome_available"] = complete
    frame["rise_trade_target_hit"] = hit
    frame["rise_trade_future_max_return"] = future_high.max(axis=1) / entry - 1
    frame["rise_trade_gross_return"] = np.where(
        hit,
        0.05,
        exit_close / entry - 1,
    )
    frame["rise_trade_net_return"] = (
        frame["rise_trade_gross_return"] - config.transaction_cost_bps / 10_000.0
    )

    for stop_pct in config.fixed_stop_pcts:
        key = _fixed_stop_key(stop_pct)
        prefix = f"rise_trade_{key}"
        fixed_stop_return = pd.Series(-stop_pct, index=frame.index)
        fixed = _stopped_trade_outcome(
            entry,
            future_open,
            future_high,
            future_low,
            exit_close,
            complete,
            fixed_stop_return,
        )
        frame[f"{prefix}_stop_pct"] = stop_pct
        _assign_stop_outcome(frame, prefix, fixed, config)

    for multiplier in config.atr_stop_multipliers:
        key = _atr_stop_key(multiplier)
        prefix = f"rise_trade_{key}"
        atr_stop_return = -(multiplier * frame["atr_14"] / entry)
        atr = _stopped_trade_outcome(
            entry,
            future_open,
            future_high,
            future_low,
            exit_close,
            complete & atr_stop_return.notna(),
            atr_stop_return,
        )
        frame[f"{prefix}_stop_pct"] = -atr_stop_return
        _assign_stop_outcome(frame, prefix, atr, config)
    return frame


def _fixed_stop_key(stop_pct: float) -> str:
    return f"fixed_{stop_pct * 100:g}pct".replace(".", "_")


def _atr_stop_key(multiplier: float) -> str:
    return f"atr_{multiplier:g}x".replace(".", "_")


def _stopped_trade_outcome(
    entry: pd.Series,
    future_open: pd.DataFrame,
    future_high: pd.DataFrame,
    future_low: pd.DataFrame,
    exit_close: pd.Series,
    complete: pd.Series,
    stop_return: pd.Series,
) -> dict[str, pd.Series]:
    """Resolve target/stop order conservatively from daily OHLC bars."""
    valid = complete & stop_return.notna() & stop_return.lt(0)
    gross_return = pd.Series(np.nan, index=entry.index, dtype=float)
    target_hit = pd.Series(False, index=entry.index, dtype=bool)
    stop_hit = pd.Series(False, index=entry.index, dtype=bool)
    ambiguous_both_hit = pd.Series(False, index=entry.index, dtype=bool)
    unresolved = valid.copy()
    target_price = entry * 1.05
    stop_price = entry * (1.0 + stop_return)

    for day in range(future_high.shape[1]):
        day_open = future_open.iloc[:, day]
        day_high = future_high.iloc[:, day]
        day_low = future_low.iloc[:, day]
        touches_stop = day_low.le(stop_price)
        touches_target = day_high.ge(target_price)
        both = unresolved & touches_stop & touches_target
        ambiguous_both_hit.loc[both] = True

        gap_stop = unresolved & day_open.le(stop_price)
        intraday_stop = unresolved & ~gap_stop & touches_stop
        target = unresolved & ~gap_stop & ~intraday_stop & touches_target

        gross_return.loc[gap_stop] = day_open.loc[gap_stop] / entry.loc[gap_stop] - 1
        gross_return.loc[intraday_stop] = stop_return.loc[intraday_stop]
        gross_return.loc[target] = 0.05
        stop_hit.loc[gap_stop | intraday_stop] = True
        target_hit.loc[target] = True
        unresolved &= ~(gap_stop | intraday_stop | target)

    gross_return.loc[unresolved] = (
        exit_close.loc[unresolved] / entry.loc[unresolved] - 1
    )
    return {
        "gross_return": gross_return,
        "target_hit": target_hit,
        "stop_hit": stop_hit,
        "ambiguous_both_hit": ambiguous_both_hit,
    }


def _assign_stop_outcome(
    frame: pd.DataFrame,
    prefix: str,
    outcome: dict[str, pd.Series],
    config: RisePatternConfig,
) -> None:
    frame[f"{prefix}_gross_return"] = outcome["gross_return"]
    frame[f"{prefix}_net_return"] = (
        outcome["gross_return"] - config.transaction_cost_bps / 10_000.0
    )
    frame[f"{prefix}_target_hit"] = outcome["target_hit"]
    frame[f"{prefix}_stop_hit"] = outcome["stop_hit"]
    frame[f"{prefix}_ambiguous_both_hit"] = outcome["ambiguous_both_hit"]


def _select_top(frame: pd.DataFrame, score_column: str, top_n: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.sort_values(score_column, ascending=False).head(top_n).copy()


def _strategy_summary(
    trades: pd.DataFrame,
    test_dates: list[Any],
    config: RisePatternConfig,
) -> dict[str, Any]:
    if trades.empty:
        return {
            "selected_signals": 0,
            "target_hit_rate": None,
            "mean_trade_net_return": None,
            "median_trade_net_return": None,
            "trade_win_rate": None,
            "ending_equity": 1.0,
            "max_drawdown": 0.0,
        }
    daily = trades.groupby("date", as_index=False)["rise_trade_net_return"].mean()
    daily = daily.set_index("date").reindex(test_dates, fill_value=0.0).reset_index()
    daily["portfolio_return"] = daily["rise_trade_net_return"] / config.horizon_days
    daily["equity"] = (1 + daily["portfolio_return"]).cumprod()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1
    return {
        "selected_signals": int(len(trades)),
        "active_days": int(trades["date"].nunique()),
        "average_signals_per_test_day": float(len(trades) / len(test_dates)),
        "target_hit_rate": float(trades["rise_trade_target_hit"].mean()),
        "mean_future_max_return": float(trades["rise_trade_future_max_return"].mean()),
        "mean_trade_net_return": float(trades["rise_trade_net_return"].mean()),
        "median_trade_net_return": float(trades["rise_trade_net_return"].median()),
        "trade_win_rate": float((trades["rise_trade_net_return"] > 0).mean()),
        "ending_equity": float(daily["equity"].iloc[-1]),
        "max_drawdown": float(daily["drawdown"].min()),
    }


def _risk_summary(
    trades: pd.DataFrame,
    test_dates: list[Any],
    config: RisePatternConfig,
    *,
    return_column: str,
    target_column: str,
    stop_column: str | None = None,
    ambiguous_column: str | None = None,
    stop_distance_column: str | None = None,
) -> dict[str, Any]:
    if trades.empty or return_column not in trades:
        return {
            "selected_signals": 0,
            "target_hit_rate": None,
            "stop_hit_rate": None,
            "mean_trade_net_return": None,
            "median_trade_net_return": None,
            "trade_win_rate": None,
            "ending_equity": 1.0,
            "max_drawdown": 0.0,
        }

    valid = trades.loc[trades[return_column].notna()].copy()
    if valid.empty:
        return {
            "selected_signals": 0,
            "target_hit_rate": None,
            "stop_hit_rate": None,
            "mean_trade_net_return": None,
            "median_trade_net_return": None,
            "trade_win_rate": None,
            "ending_equity": 1.0,
            "max_drawdown": 0.0,
        }

    daily = valid.groupby("date", as_index=False)[return_column].mean()
    daily = daily.set_index("date").reindex(test_dates, fill_value=0.0).reset_index()
    daily["portfolio_return"] = daily[return_column] / config.horizon_days
    daily["equity"] = (1 + daily["portfolio_return"]).cumprod()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1
    result: dict[str, Any] = {
        "selected_signals": int(len(valid)),
        "active_days": int(valid["date"].nunique()),
        "target_hit_rate": float(valid[target_column].mean()),
        "stop_hit_rate": (
            float(valid[stop_column].mean()) if stop_column is not None else 0.0
        ),
        "mean_trade_net_return": float(valid[return_column].mean()),
        "median_trade_net_return": float(valid[return_column].median()),
        "trade_win_rate": float((valid[return_column] > 0).mean()),
        "ending_equity": float(daily["equity"].iloc[-1]),
        "max_drawdown": float(daily["drawdown"].min()),
    }
    if ambiguous_column is not None:
        result["ambiguous_both_hit_rate"] = float(valid[ambiguous_column].mean())
    if stop_distance_column is not None:
        result["mean_stop_distance_pct"] = float(valid[stop_distance_column].mean())
        result["median_stop_distance_pct"] = float(valid[stop_distance_column].median())
    return result

