from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DemandSupplyConfig:
    """Frozen rules for the out-of-sample demand/supply timing study."""

    test_days: int = 120
    horizon_days: int = 5
    target_pct: float = 0.04
    stop_pct: float = 0.03
    transaction_cost_bps: float = 20.0
    maximum_gap_up: float = 0.0
    minimum_gap_down: float = -0.10
    maximum_candidates_per_day: int = 1
    demand_thresholds: tuple[float, ...] = (0.65, 0.75, 0.85, 0.90)
    supply_thresholds: tuple[float, ...] = (0.55, 0.65, 0.75)
    minimum_turnovers: tuple[float, ...] = (
        50_000_000.0,
        100_000_000.0,
        200_000_000.0,
    )
    minimum_development_trades: int = 20
    minimum_one_day_volume_ratio: float = 1.20
    minimum_one_day_turnover_ratio: float = 1.20
    minimum_up_volume_share: float = 0.55
    minimum_demand_supply_balance: float = 0.05


def add_supply_pressure_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add a symmetric, close-known proxy for sell-side supply pressure."""
    frame = features.copy()
    volume_rank = _numeric(frame, "observed_volume_ratio_rank")
    turnover_rank = _numeric(frame, "observed_turnover_ratio_rank")
    volume_intensity = _numeric(frame, "observed_volume_intensity_score")
    turnover_intensity = _numeric(frame, "observed_turnover_intensity_score")
    negative_return = (-_numeric(frame, "return_1d") / 0.06).clip(0.0, 1.0)
    negative_candle = (-_numeric(frame, "intraday_return") / 0.05).clip(0.0, 1.0)
    price_confirmation = 0.65 * negative_return + 0.35 * negative_candle
    down_volume = ((0.55 - _numeric(frame, "up_volume_share_10d")) / 0.35).clip(
        0.0, 1.0
    )
    frame["observed_supply_price_confirmation_score"] = price_confirmation
    frame["observed_supply_pressure_score"] = (
        0.25 * volume_rank
        + 0.25 * turnover_rank
        + 0.15 * volume_intensity
        + 0.15 * turnover_intensity
        + 0.10 * price_confirmation
        + 0.10 * down_volume
    ).clip(0.0, 1.0)
    frame["observed_demand_supply_balance"] = (
        _numeric(frame, "observed_inflow_score")
        - frame["observed_supply_pressure_score"]
    )
    frame["observed_supply_pressure_confirmed"] = _supply_trigger(frame, 0.55)
    return frame


def analyze_demand_supply_timing(
    features: pd.DataFrame,
    *,
    config: DemandSupplyConfig | None = None,
) -> dict[str, Any]:
    """Tune on the first half, then compare fixed and supply exits on the second half."""
    config = config or DemandSupplyConfig()
    if features.empty:
        return {"method": "demand_supply_walk_forward_v1", "status": "no_data"}

    frame = add_supply_pressure_features(features)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    fixed_outcomes = _simulate_outcomes(
        frame,
        config,
        supply_threshold=None,
        use_fixed_stop=True,
    )
    supply_outcomes = {
        threshold: _simulate_outcomes(
            frame,
            config,
            supply_threshold=threshold,
            use_fixed_stop=False,
        )
        for threshold in config.supply_thresholds
    }
    hybrid_outcomes = {
        threshold: _simulate_outcomes(
            frame,
            config,
            supply_threshold=threshold,
            use_fixed_stop=True,
        )
        for threshold in config.supply_thresholds
    }
    complete = fixed_outcomes["trade_available"]
    available_dates = sorted(frame.loc[complete, "date"].drop_duplicates())
    test_dates = available_dates[-config.test_days :]
    midpoint = len(test_dates) // 2
    development_dates = test_dates[:midpoint]
    validation_dates = test_dates[midpoint:]
    if not development_dates or not validation_dates:
        return {"method": "demand_supply_walk_forward_v1", "status": "insufficient_dates"}

    grid: list[dict[str, Any]] = []
    for demand_threshold in config.demand_thresholds:
        for supply_threshold in config.supply_thresholds:
            for minimum_turnover in config.minimum_turnovers:
                selected = _select_buys(
                    frame,
                    fixed_outcomes,
                    dates=development_dates,
                    demand_threshold=demand_threshold,
                    minimum_turnover=minimum_turnover,
                    config=config,
                )
                summary = _summary(
                    selected,
                    supply_outcomes[supply_threshold],
                    development_dates,
                    config,
                )
                grid.append(
                    {
                        "demand_threshold": demand_threshold,
                        "supply_threshold": supply_threshold,
                        "minimum_turnover_yen": int(minimum_turnover),
                        **summary,
                    }
                )

    eligible_grid = [
        result
        for result in grid
        if result["selected_signals"] >= config.minimum_development_trades
    ]
    candidates = eligible_grid or grid
    best = max(
        candidates,
        key=lambda result: (
            result["ending_equity"],
            result["mean_net_return_per_holding_day"] or -1.0,
            result["selected_signals"],
        ),
    )
    selected_config = {
        "demand_threshold": best["demand_threshold"],
        "supply_threshold": best["supply_threshold"],
        "minimum_turnover_yen": best["minimum_turnover_yen"],
    }

    development_buys = _select_buys(
        frame,
        fixed_outcomes,
        dates=development_dates,
        demand_threshold=float(best["demand_threshold"]),
        minimum_turnover=float(best["minimum_turnover_yen"]),
        config=config,
    )
    validation_buys = _select_buys(
        frame,
        fixed_outcomes,
        dates=validation_dates,
        demand_threshold=float(best["demand_threshold"]),
        minimum_turnover=float(best["minimum_turnover_yen"]),
        config=config,
    )
    supply_threshold = float(best["supply_threshold"])
    development = {
        "fixed_exit": _summary(
            development_buys,
            fixed_outcomes,
            development_dates,
            config,
        ),
        "supply_exit": _summary(
            development_buys,
            supply_outcomes[supply_threshold],
            development_dates,
            config,
        ),
        "hybrid_exit": _summary(
            development_buys,
            hybrid_outcomes[supply_threshold],
            development_dates,
            config,
        ),
    }
    validation = {
        "fixed_exit": _summary(
            validation_buys,
            fixed_outcomes,
            validation_dates,
            config,
        ),
        "supply_exit": _summary(
            validation_buys,
            supply_outcomes[supply_threshold],
            validation_dates,
            config,
        ),
        "hybrid_exit": _summary(
            validation_buys,
            hybrid_outcomes[supply_threshold],
            validation_dates,
            config,
        ),
    }

    return {
        "method": "demand_supply_walk_forward_v1",
        "status": "complete",
        "definition": {
            "demand": (
                "当日引け時点の出来高・売買代金の20日比と市場内順位、"
                "陽線・上昇確認、10日上昇日出来高比率"
            ),
            "supply": (
                "同じ出来高・売買代金強度に陰線・下落確認と"
                "10日下落日出来高比率を重ねた売り圧力代理値"
            ),
            "credit_balance": "日次で同条件取得できないため未使用",
        },
        "assumptions": {
            **asdict(config),
            "entry": "signal_close_then_next_open_only_if_open_not_above_signal_close",
            "supply_exit": (
                "no_fixed_stop; supply_detected_at_close_then_exit_next_open"
            ),
            "hybrid_exit": "fixed_3pct_stop_plus_supply_exit",
            "same_day_stop_and_target": "stop_first",
            "gap_through_stop_or_target": "exit_at_open",
            "selection": "development_first_half_then_frozen_validation_second_half",
        },
        "period": {
            "development_start": str(development_dates[0]),
            "development_end": str(development_dates[-1]),
            "validation_start": str(validation_dates[0]),
            "validation_end": str(validation_dates[-1]),
        },
        "selected_config": selected_config,
        "development": development,
        "validation": validation,
        "development_grid": grid,
    }


def _select_buys(
    frame: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    dates: list[Any],
    demand_threshold: float,
    minimum_turnover: float,
    config: DemandSupplyConfig,
) -> pd.Index:
    demand = _numeric(frame, "observed_inflow_score")
    balance = _numeric(frame, "observed_demand_supply_balance")
    mask = (
        frame["date"].isin(dates)
        & baseline["trade_available"]
        & baseline["entry_gap_return"].between(
            config.minimum_gap_down,
            config.maximum_gap_up,
        )
        & _numeric(frame, "turnover_value").ge(minimum_turnover)
        & demand.ge(demand_threshold)
        & balance.ge(config.minimum_demand_supply_balance)
        & _numeric(frame, "observed_volume_ratio_rank").ge(0.70)
        & _numeric(frame, "observed_turnover_ratio_rank").ge(0.70)
        & _numeric(frame, "volume_ratio_1_20").ge(
            config.minimum_one_day_volume_ratio
        )
        & _numeric(frame, "turnover_ratio_1_20").ge(
            config.minimum_one_day_turnover_ratio
        )
        & _numeric(frame, "return_1d").gt(0.002)
        & _numeric(frame, "intraday_return").ge(0.0)
        & _numeric(frame, "up_volume_share_10d").ge(
            config.minimum_up_volume_share
        )
        & _numeric(frame, "rsi_14", default=100.0).le(82.0)
    )
    selected = frame.loc[mask, ["date"]].copy()
    selected["_rank_score"] = demand.loc[mask] + 0.25 * balance.loc[mask]
    return (
        selected.sort_values(["date", "_rank_score"], ascending=[True, False])
        .groupby("date", sort=False)
        .head(config.maximum_candidates_per_day)
        .index
    )


def _simulate_outcomes(
    frame: pd.DataFrame,
    config: DemandSupplyConfig,
    *,
    supply_threshold: float | None,
    use_fixed_stop: bool = True,
) -> pd.DataFrame:
    ticker = frame["ticker"]
    ratio = _numeric(frame, "adjusted_close") / _numeric(frame, "close")
    adjusted_open = _numeric(frame, "open") * ratio
    adjusted_high = _numeric(frame, "high") * ratio
    adjusted_low = _numeric(frame, "low") * ratio
    adjusted_close = _numeric(frame, "adjusted_close")
    future_open = pd.concat(
        [
            adjusted_open.groupby(ticker, sort=False).shift(-offset)
            for offset in range(1, config.horizon_days + 1)
        ],
        axis=1,
    )
    future_high = pd.concat(
        [
            adjusted_high.groupby(ticker, sort=False).shift(-offset)
            for offset in range(1, config.horizon_days + 1)
        ],
        axis=1,
    )
    future_low = pd.concat(
        [
            adjusted_low.groupby(ticker, sort=False).shift(-offset)
            for offset in range(1, config.horizon_days + 1)
        ],
        axis=1,
    )
    future_close = pd.concat(
        [
            adjusted_close.groupby(ticker, sort=False).shift(-offset)
            for offset in range(1, config.horizon_days + 1)
        ],
        axis=1,
    )
    entry = future_open.iloc[:, 0]
    complete = (
        future_open.notna().all(axis=1)
        & future_high.notna().all(axis=1)
        & future_low.notna().all(axis=1)
        & future_close.notna().all(axis=1)
    )
    supply = (
        _supply_trigger(frame, supply_threshold)
        if supply_threshold is not None
        else pd.Series(False, index=frame.index)
    )
    future_supply = pd.concat(
        [
            supply.groupby(ticker, sort=False)
            .shift(-offset)
            .astype("boolean")
            .fillna(False)
            .astype(bool)
            for offset in range(1, config.horizon_days)
        ],
        axis=1,
    )

    target_price = entry * (1.0 + config.target_pct)
    stop_price = entry * (1.0 - config.stop_pct)
    gross_return = pd.Series(np.nan, index=frame.index, dtype="float64")
    target_hit = pd.Series(False, index=frame.index)
    stop_hit = pd.Series(False, index=frame.index)
    supply_exit = pd.Series(False, index=frame.index)
    holding_days = pd.Series(np.nan, index=frame.index, dtype="float64")
    unresolved = complete.copy()
    never = pd.Series(False, index=frame.index)

    for day in range(config.horizon_days):
        day_open = future_open.iloc[:, day]
        day_high = future_high.iloc[:, day]
        day_low = future_low.iloc[:, day]
        if day > 0 and supply_threshold is not None:
            exit_on_supply = unresolved & future_supply.iloc[:, day - 1].astype(bool)
            gross_return.loc[exit_on_supply] = (
                day_open.loc[exit_on_supply] / entry.loc[exit_on_supply] - 1.0
            )
            supply_exit.loc[exit_on_supply] = True
            holding_days.loc[exit_on_supply] = float(day)
            unresolved &= ~exit_on_supply

        gap_stop = unresolved & day_open.le(stop_price) if use_fixed_stop else never
        gap_target = unresolved & ~gap_stop & day_open.ge(target_price)
        touches_stop = day_low.le(stop_price) if use_fixed_stop else never
        touches_target = day_high.ge(target_price)
        intraday_stop = unresolved & ~gap_stop & ~gap_target & touches_stop
        intraday_target = (
            unresolved
            & ~gap_stop
            & ~gap_target
            & ~intraday_stop
            & touches_target
        )
        gross_return.loc[gap_stop | gap_target] = (
            day_open.loc[gap_stop | gap_target] / entry.loc[gap_stop | gap_target] - 1.0
        )
        gross_return.loc[intraday_stop] = -config.stop_pct
        gross_return.loc[intraday_target] = config.target_pct
        stop_hit.loc[gap_stop | intraday_stop] = True
        target_hit.loc[gap_target | intraday_target] = True
        resolved = gap_stop | gap_target | intraday_stop | intraday_target
        holding_days.loc[resolved] = float(day + 1)
        unresolved &= ~resolved

    gross_return.loc[unresolved] = (
        future_close.iloc[:, -1].loc[unresolved] / entry.loc[unresolved] - 1.0
    )
    holding_days.loc[unresolved] = float(config.horizon_days)
    return pd.DataFrame(
        {
            "signal_date": frame["date"],
            "trade_available": complete,
            "entry_gap_return": entry / adjusted_close - 1.0,
            "gross_return": gross_return,
            "net_return": gross_return - config.transaction_cost_bps / 10_000.0,
            "target_hit": target_hit,
            "stop_hit": stop_hit,
            "supply_exit": supply_exit,
            "holding_days": holding_days,
        },
        index=frame.index,
    )


def _summary(
    selected: pd.Index,
    outcomes: pd.DataFrame,
    dates: list[Any],
    config: DemandSupplyConfig,
) -> dict[str, Any]:
    trades = outcomes.loc[selected].copy()
    if trades.empty:
        return {
            "selected_signals": 0,
            "target_hit_rate": None,
            "stop_hit_rate": None,
            "supply_exit_rate": None,
            "trade_win_rate": None,
            "mean_trade_net_return": None,
            "median_trade_net_return": None,
            "mean_holding_days": None,
            "mean_net_return_per_holding_day": None,
            "ending_equity": 1.0,
            "max_drawdown": 0.0,
        }
    daily = trades.groupby("signal_date")["net_return"].mean()
    daily = daily.reindex(dates, fill_value=0.0)
    # One fifth of capital per daily cohort prevents overlapping five-day trades from
    # implicitly using leverage and matches the existing Momentum 5D summaries.
    equity = (1.0 + daily / config.horizon_days).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    holding_days = trades["holding_days"].replace(0, np.nan)
    return {
        "selected_signals": int(len(trades)),
        "active_days": int(trades["signal_date"].nunique()),
        "target_hit_rate": float(trades["target_hit"].mean()),
        "stop_hit_rate": float(trades["stop_hit"].mean()),
        "supply_exit_rate": float(trades["supply_exit"].mean()),
        "trade_win_rate": float((trades["net_return"] > 0).mean()),
        "mean_trade_net_return": float(trades["net_return"].mean()),
        "median_trade_net_return": float(trades["net_return"].median()),
        "mean_holding_days": float(holding_days.mean()),
        "mean_net_return_per_holding_day": float(
            (trades["net_return"] / holding_days).mean()
        ),
        "ending_equity": float(equity.iloc[-1]),
        "max_drawdown": float(drawdown.min()),
    }


def _supply_trigger(frame: pd.DataFrame, threshold: float) -> pd.Series:
    return (
        _numeric(frame, "observed_supply_pressure_score").ge(threshold)
        & _numeric(frame, "observed_volume_ratio_rank").ge(0.70)
        & _numeric(frame, "observed_turnover_ratio_rank").ge(0.70)
        & _numeric(frame, "return_1d").lt(-0.002)
        & _numeric(frame, "intraday_return").le(0.0)
        & _numeric(frame, "up_volume_share_10d", default=1.0).le(0.52)
    )


def _numeric(frame: pd.DataFrame, column: str, *, default: float = 0.0) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)
