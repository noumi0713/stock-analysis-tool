from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.adjustments import normalize_split_adjusted_ohlcv
from app.live_strategy import StrategySpecError, load_frozen_strategy
from scripts.export_close_signals import (
    calculate_indicators,
    select_historical_pullback_signals,
    select_historical_signals,
)

EXECUTION_ENGINE_ID = "shared_next_open_ohlc_v1"


def _resolve_strategy(strategy: dict[str, Any] | None) -> dict[str, Any]:
    frozen = load_frozen_strategy()
    if strategy is not None and strategy != frozen:
        raise StrategySpecError(
            "Backtest strategy differs from the frozen production specification"
        )
    return frozen


@dataclass(frozen=True, slots=True)
class TradePath:
    signal_type: str
    signal_date: object
    entry_date: object
    exit_date: object
    ticker: str
    candidate_rank: int
    signal_close: float
    entry_price: float
    exit_price: float
    gross_return: float
    net_return: float
    exit_reason: str
    holding_sessions: int


def _profit_factor(returns: pd.Series) -> float | None:
    gains = float(returns.loc[returns.gt(0)].sum())
    losses = float(-returns.loc[returns.lt(0)].sum())
    return gains / losses if losses > 0 else None


def build_trade_paths(
    prices: pd.DataFrame, *, signal_type: str, strategy: dict[str, Any] | None = None
) -> pd.DataFrame:
    spec = _resolve_strategy(strategy)
    indicators = calculate_indicators(prices)
    if signal_type == "capitulation_reversal":
        selected = select_historical_signals(indicators)
    elif signal_type == "first_pullback":
        selected = select_historical_pullback_signals(indicators)
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")
    if selected.empty:
        return pd.DataFrame(columns=[field for field in TradePath.__annotations__])

    execution = spec["signals"][signal_type]["execution"]
    slippage = float(spec["portfolio"]["slippage_bps_per_side"]) / 10_000.0
    transaction_cost = (
        float(spec["portfolio"]["round_trip_transaction_cost_bps"]) / 10_000.0
    )
    market_dates = sorted(indicators["date"].drop_duplicates())
    next_date = {day: market_dates[index + 1] for index, day in enumerate(market_dates[:-1])}
    bars = {
        ticker: stock.sort_values("date").reset_index(drop=True)
        for ticker, stock in indicators.groupby("ticker", sort=False)
    }

    selected = selected.copy()
    selected["candidate_rank"] = selected.groupby("date").cumcount().add(1)
    paths: list[TradePath] = []
    for _, row in selected.iterrows():
        entry_date = next_date.get(row["date"])
        if entry_date is None:
            continue
        stock = bars[str(row["ticker"])]
        entry_rows = stock.loc[stock["date"].eq(entry_date)]
        if entry_rows.empty:
            continue
        entry_row = entry_rows.iloc[0]
        raw_entry = float(entry_row["_open"])
        signal_close = float(row["_close"])
        gap = raw_entry / signal_close - 1.0
        if "entry_gap_min" in execution and not (
            float(execution["entry_gap_min"])
            <= gap
            <= float(execution["entry_gap_max"])
        ):
            continue

        entry_price = raw_entry * (1.0 + slippage)
        target_price = entry_price * (1.0 + float(execution["take_profit_pct"]))
        stop_price = entry_price * (1.0 + float(execution["stop_loss_pct"]))
        future = stock.loc[stock["date"].ge(entry_date)].head(
            int(execution["holding_days"])
        )
        if future.empty:
            continue
        exit_price = float(future.iloc[-1]["_close"]) * (1.0 - slippage)
        exit_date = future.iloc[-1]["date"]
        exit_reason = "time"
        holding_sessions = len(future)
        for session, (_, bar) in enumerate(future.iterrows(), start=1):
            if float(bar["_low"]) <= stop_price:
                exit_price = stop_price * (1.0 - slippage)
                exit_date = bar["date"]
                exit_reason = "stop_loss"
                holding_sessions = session
                break
            if float(bar["_high"]) >= target_price:
                exit_price = target_price * (1.0 - slippage)
                exit_date = bar["date"]
                exit_reason = "take_profit"
                holding_sessions = session
                break
        gross_return = exit_price / entry_price - 1.0
        paths.append(
            TradePath(
                signal_type=signal_type,
                signal_date=row["date"],
                entry_date=entry_date,
                exit_date=exit_date,
                ticker=str(row["ticker"]),
                candidate_rank=int(row["candidate_rank"]),
                signal_close=signal_close,
                entry_price=entry_price,
                exit_price=exit_price,
                gross_return=gross_return,
                net_return=gross_return - transaction_cost,
                exit_reason=exit_reason,
                holding_sessions=holding_sessions,
            )
        )
    return pd.DataFrame([asdict(path) for path in paths]) if paths else pd.DataFrame()


def simulate_portfolio(
    trade_paths: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    strategy: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    spec = _resolve_strategy(strategy)
    portfolio = spec["portfolio"]
    initial_capital = float(portfolio["initial_capital_yen"])
    lot_size = int(portfolio["lot_size"])
    max_positions = int(portfolio["maximum_open_positions"])
    max_new = int(portfolio["maximum_new_positions_per_day"])
    one_way_cost = (
        float(portfolio["round_trip_transaction_cost_bps"]) / 20_000.0
    )
    if trade_paths.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "initial_equity_yen": initial_capital,
                "ending_equity_yen": initial_capital,
                "total_return": 0.0,
                "maximum_drawdown": 0.0,
                "completed_trades": 0,
            },
        )

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame = normalize_split_adjusted_ohlcv(frame)
    frame["mark_close"] = frame["_close"]
    marks = frame.set_index(["date", "ticker"])["mark_close"].to_dict()
    dates = sorted(frame["date"].dropna().unique())
    entries = {
        day: group.sort_values(["candidate_rank", "ticker"])
        for day, group in trade_paths.groupby("entry_date")
    }

    cash = initial_capital
    positions: dict[str, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    for day in dates:
        exiting = [ticker for ticker, item in positions.items() if item["exit_date"] == day]
        for ticker in exiting:
            item = positions.pop(ticker)
            proceeds = item["shares"] * item["exit_price"] * (1.0 - one_way_cost)
            cash += proceeds
            completed.append(
                {
                    **item,
                    "net_profit_yen": proceeds - item["cash_out"],
                    "portfolio_exit_date": day,
                }
            )

        new_count = 0
        for row in entries.get(day, pd.DataFrame()).itertuples(index=False):
            if new_count >= max_new or len(positions) >= max_positions:
                break
            if row.ticker in positions:
                continue
            remaining_slots = max_positions - len(positions)
            budget = cash / remaining_slots
            unit_cost = float(row.entry_price) * lot_size * (1.0 + one_way_cost)
            lots = int(np.floor(budget / unit_cost))
            if lots < 1:
                continue
            shares = lots * lot_size
            cash_out = shares * float(row.entry_price) * (1.0 + one_way_cost)
            cash -= cash_out
            positions[str(row.ticker)] = {
                **row._asdict(),
                "shares": shares,
                "cash_out": cash_out,
            }
            new_count += 1

        immediate = [ticker for ticker, item in positions.items() if item["exit_date"] == day]
        for ticker in immediate:
            item = positions.pop(ticker)
            proceeds = item["shares"] * item["exit_price"] * (1.0 - one_way_cost)
            cash += proceeds
            completed.append(
                {
                    **item,
                    "net_profit_yen": proceeds - item["cash_out"],
                    "portfolio_exit_date": day,
                }
            )

        market_value = sum(
            item["shares"]
            * float(marks.get((day, ticker), item["entry_price"]))
            for ticker, item in positions.items()
        )
        curve.append(
            {
                "date": day,
                "cash_yen": cash,
                "market_value_yen": market_value,
                "equity_yen": cash + market_value,
                "open_positions": len(positions),
            }
        )

    completed_frame = pd.DataFrame(completed)
    curve_frame = pd.DataFrame(curve)
    curve_frame["drawdown"] = (
        curve_frame["equity_yen"] / curve_frame["equity_yen"].cummax() - 1.0
    )
    realized_returns = (
        completed_frame["net_profit_yen"] / completed_frame["cash_out"]
        if not completed_frame.empty
        else pd.Series(dtype="float64")
    )
    ending_equity = float(curve_frame.iloc[-1]["equity_yen"])
    annual = {}
    previous_year_end = initial_capital
    years = pd.Series(curve_frame["date"]).map(lambda day: day.year)
    for year, rows in curve_frame.groupby(years, sort=True):
        end_equity = float(rows.iloc[-1]["equity_yen"])
        annual[str(year)] = end_equity / previous_year_end - 1.0
        previous_year_end = end_equity
    summary = {
        "strategy_version": spec["strategy_version"],
        "execution_engine_id": EXECUTION_ENGINE_ID,
        "initial_equity_yen": initial_capital,
        "ending_equity_yen": ending_equity,
        "total_return": ending_equity / initial_capital - 1.0,
        "maximum_drawdown": float(curve_frame["drawdown"].min()),
        "completed_trades": len(completed_frame),
        "trade_win_rate": float(realized_returns.gt(0).mean())
        if not realized_returns.empty
        else None,
        "mean_trade_net_return": float(realized_returns.mean())
        if not realized_returns.empty
        else None,
        "profit_factor": _profit_factor(realized_returns),
        "annual_returns": annual,
    }
    return completed_frame, curve_frame, summary


def run_strategy_backtest(
    prices: pd.DataFrame, *, signal_type: str, strategy: dict[str, Any] | None = None
) -> dict[str, Any]:
    spec = _resolve_strategy(strategy)
    paths = build_trade_paths(prices, signal_type=signal_type, strategy=spec)
    trades, curve, summary = simulate_portfolio(paths, prices, strategy=spec)
    summary.update(
        {
            "signal_type": signal_type,
            "candidate_trade_paths": len(paths),
            "take_profit_count": int(trades["exit_reason"].eq("take_profit").sum())
            if not trades.empty
            else 0,
            "stop_loss_count": int(trades["exit_reason"].eq("stop_loss").sum())
            if not trades.empty
            else 0,
            "timed_exit_count": int(trades["exit_reason"].eq("time").sum())
            if not trades.empty
            else 0,
        }
    )
    return {
        "summary": summary,
        "candidate_paths": paths,
        "trades": trades,
        "equity_curve": curve,
    }
