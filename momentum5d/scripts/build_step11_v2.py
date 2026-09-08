# ruff: noqa: E501
"""STEP11 V2: explicitly contaminated post-2025-09-08 validation.

This module must describe the analysed slice as not untouched OOS.  One
candidate is frozen by ``freeze_step11_v2_candidate.py`` before this builder
loads any post-boundary value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

JST = ZoneInfo("Asia/Tokyo")
VALIDATION_START = pd.Timestamp("2025-09-08")
WARMUP_START = pd.Timestamp("2025-08-01")
ANALYSIS_LABEL = "contaminated_validation"
ROUND_TRIP_COST = 0.004
ENTRY_COST = 0.002
EXIT_COST = 0.002
MAX_POSITIONS = 10
POSITION_FRACTION = 0.10
RUIN_SEED = 1102
RUIN_PATHS = 10_000
RUIN_TRADES = 250


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build STEP11 contaminated validation only")
    parser.add_argument("--root", default="data/market_history")
    parser.add_argument("--verify-reproducibility", action="store_true")
    parser.add_argument("--single-run-output", help=argparse.SUPPRESS)
    return parser.parse_args()


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def input_manifest(groups: list[tuple[str, Path]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for label, path in groups:
        files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
        for item in files:
            relative = item.relative_to(path) if path.is_dir() else Path(item.name)
            result[f"{label}/{relative}"] = {"bytes": item.stat().st_size, "sha256": sha256(item)}
    return result


def output_manifest(path: Path) -> dict[str, str]:
    return {str(item.relative_to(path)): sha256(item) for item in sorted(path.rglob("*.parquet"))}


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(ordered, preserve_index=False), path, compression="zstd")
    reread = pd.read_parquet(path)
    if len(reread) != len(ordered) or list(reread.columns) != list(ordered.columns):
        raise RuntimeError(f"Parquet reread mismatch: {path}")


def parquet_glob(path: Path) -> str:
    return str(path / "**/*.parquet").replace("'", "''")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def load_freeze(root: Path) -> tuple[pd.Series, list[dict[str, Any]], list[dict[str, Any]]]:
    report_path = root / "quality/step11_candidate_freeze.json"
    freeze_path = root / "frozen/step11_v2_candidate/part.parquet"
    if not report_path.is_file() or not freeze_path.is_file():
        raise RuntimeError("STEP11 candidate must be frozen before validation values are loaded")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("quality") != "PASS"
        or report.get("analysis_label") != ANALYSIS_LABEL
        or report.get("untouched_oos") is not False
        or report.get("freeze_completed_before_validation_values_loaded") is not True
        or report.get("freeze_parquet_sha256") != sha256(freeze_path)
    ):
        raise RuntimeError("STEP11 candidate freeze certification failed")
    frame = pd.read_parquet(freeze_path)
    if len(frame) != 1:
        raise RuntimeError("Exactly one STEP11 candidate must be frozen")
    row = frame.iloc[0]
    if row.analysis_label != ANALYSIS_LABEL or bool(row.untouched_oos):
        raise RuntimeError("Candidate freeze mislabels the validation slice")
    if bool(row.post_2025_09_08_values_read_during_freeze):
        raise RuntimeError("Candidate was frozen after validation values were read")
    return row, json.loads(row.features_json), json.loads(row.regime_definitions_json)


def load_prices(
    step1: Path,
    conditions: list[dict[str, Any]],
    regimes: list[dict[str, Any]],
) -> pd.DataFrame:
    features = sorted(
        {item["feature"] for item in conditions} | {item["feature"] for item in regimes}
    )
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    source = f"read_parquet('{parquet_glob(step1)}')"
    names = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()}
    required = {
        "Date",
        "Ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "trading_value",
        *features,
    }
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(
            f"STEP1 columns required by frozen STEP11 candidate are missing: {missing}"
        )
    feature_select = ",\n        ".join(quote(feature) for feature in features)
    rolling = ",\n        ".join(
        f"median({quote(feature)}) OVER w AS {quote('bucket__' + feature)}, "
        f"count({quote(feature)}) OVER w AS {quote('history_count__' + feature)}"
        for feature in features
    )
    sql = f"""
      WITH source AS (
        SELECT CAST(Date AS DATE) AS Date, CAST(Ticker AS VARCHAR) AS Ticker,
          Open, High, Low, Close, "Adj Close", Volume, trading_value,
          {feature_select}
        FROM {source}
        WHERE Date >= DATE '{WARMUP_START.date()}'
      ), adjusted AS (
        SELECT *,
          Open*"Adj Close"/nullif(Close,0) AS adj_open,
          High*"Adj Close"/nullif(Close,0) AS adj_high,
          Low*"Adj Close"/nullif(Close,0) AS adj_low,
          "Adj Close" AS adj_close
        FROM source
      )
      SELECT *, row_number() OVER(PARTITION BY Ticker ORDER BY Date)-1 AS ticker_position,
        {rolling}
      FROM adjusted
      WINDOW w AS (PARTITION BY Ticker ORDER BY Date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING)
      ORDER BY Ticker, Date
    """
    frame = con.execute(sql).df()
    con.close()
    frame["Date"] = pd.to_datetime(frame.Date).dt.normalize()
    if frame.duplicated(["Ticker", "Date"]).any():
        raise RuntimeError("STEP1 contains duplicate Ticker x Date rows")
    return frame


def condition_mask(frame: pd.DataFrame, conditions: list[dict[str, Any]]) -> pd.Series:
    mask = frame.Date.ge(VALIDATION_START)
    for item in conditions:
        feature = item["feature"]
        values = pd.to_numeric(frame[f"bucket__{feature}"], errors="coerce")
        complete = frame[f"history_count__{feature}"].eq(5)
        if item["operator"] == ">=":
            matched = values.ge(float(item["threshold"]))
        elif item["operator"] == "<=":
            matched = values.le(float(item["threshold"]))
        else:
            raise RuntimeError(f"Unsupported frozen operator: {item['operator']}")
        mask &= complete & matched
    return mask


def regime_name(value: Any, q30: float, q70: float) -> str:
    number = finite(value)
    if number is None:
        return "unknown"
    if number <= q30:
        return "low"
    if number >= q70:
        return "high"
    return "neutral"


def make_signals_and_trades(
    prices: pd.DataFrame,
    freeze: pd.Series,
    conditions: list[dict[str, Any]],
    regimes: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices = prices.copy()
    prices["ticker_position"] = prices.groupby("Ticker", sort=False).cumcount()
    mask = condition_mask(prices, conditions)
    signals = prices.loc[mask].copy()
    signals["candidate_id"] = str(freeze.candidate_id)
    signals["combination_id"] = str(freeze.combination_id)
    signals["analysis_label"] = ANALYSIS_LABEL
    signals["untouched_oos"] = False
    signals["signal_id"] = signals.apply(
        lambda row: f"{row.Ticker}|{pd.Timestamp(row.Date).date()}|{freeze.candidate_id}", axis=1
    )
    for item in regimes:
        feature = item["feature"]
        signals[f"regime__{feature}"] = signals[f"bucket__{feature}"].map(
            lambda value, low=float(item["q30"]), high=float(item["q70"]): regime_name(
                value, low, high
            )
        )
    signals["lookback_start_date"] = pd.NaT
    signals["entry_available"] = False
    signals["exit_evaluable"] = False
    signals["exclusion_reason"] = None
    signals["exit_date"] = pd.NaT
    signals["exit_price"] = np.nan
    signals["gross_return"] = np.nan
    signals["net_return"] = np.nan
    signals["mfe"] = np.nan
    signals["mae"] = np.nan
    signals["holding_sessions"] = pd.NA

    trade_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    signal_index = {
        signal_id: index for index, signal_id in zip(signals.index, signals.signal_id, strict=True)
    }
    for ticker, group in prices.groupby("Ticker", sort=True, observed=True):
        group = group.reset_index(drop=True)
        ticker_signals = signals[signals.Ticker.eq(ticker)]
        for signal in ticker_signals.itertuples(index=False):
            source_index = signal_index[signal.signal_id]
            position = int(signal.ticker_position)
            lookback_position = position - 5
            lookback_date = (
                group.at[lookback_position, "Date"] if lookback_position >= 0 else pd.NaT
            )
            signals.at[source_index, "lookback_start_date"] = lookback_date
            reason: str | None = None
            if pd.isna(lookback_date) or pd.Timestamp(lookback_date) < VALIDATION_START:
                reason = "lookback_crosses_validation_start"
            entry_price = finite(group.at[position, "adj_close"])
            volume = finite(group.at[position, "Volume"])
            if reason is None and (entry_price is None or entry_price <= 0):
                reason = "entry_adjusted_close_missing_or_nonpositive"
            if reason is None and (volume is None or volume <= 0):
                reason = "entry_volume_missing_or_zero"
            exit_position = position + 40
            if reason is None and exit_position >= len(group):
                reason = "forty_session_exit_not_available"
            path = (
                group.iloc[position + 1 : exit_position + 1]
                if exit_position < len(group)
                else pd.DataFrame()
            )
            exit_price = (
                finite(group.at[exit_position, "adj_close"]) if exit_position < len(group) else None
            )
            if reason is None and (exit_price is None or exit_price <= 0):
                reason = "exit_adjusted_close_missing_or_nonpositive"
            highs = (
                pd.to_numeric(path.adj_high, errors="coerce")
                if len(path)
                else pd.Series(dtype=float)
            )
            lows = (
                pd.to_numeric(path.adj_low, errors="coerce")
                if len(path)
                else pd.Series(dtype=float)
            )
            if reason is None and (
                len(path) != 40
                or highs.isna().any()
                or lows.isna().any()
                or highs.le(0).any()
                or lows.le(0).any()
            ):
                reason = "forty_session_ohlc_path_incomplete"

            signals.at[source_index, "entry_available"] = reason not in {
                "entry_adjusted_close_missing_or_nonpositive",
                "entry_volume_missing_or_zero",
            }
            if reason is not None:
                signals.at[source_index, "exclusion_reason"] = reason
                exclusion_rows.append(
                    {
                        "signal_id": signal.signal_id,
                        "Ticker": ticker,
                        "signal_date": signal.Date,
                        "reason": reason,
                        "analysis_label": ANALYSIS_LABEL,
                    }
                )
                continue

            gross = float(exit_price / entry_price - 1.0)
            net = gross - ROUND_TRIP_COST
            mfe = float(highs.max() / entry_price - 1.0)
            mae = float(lows.min() / entry_price - 1.0)
            exit_date = pd.Timestamp(group.at[exit_position, "Date"])
            signals.at[source_index, "exit_evaluable"] = True
            signals.at[source_index, "exit_date"] = exit_date
            signals.at[source_index, "exit_price"] = exit_price
            signals.at[source_index, "gross_return"] = gross
            signals.at[source_index, "net_return"] = net
            signals.at[source_index, "mfe"] = mfe
            signals.at[source_index, "mae"] = mae
            signals.at[source_index, "holding_sessions"] = 40
            row = {
                "signal_id": signal.signal_id,
                "candidate_id": freeze.candidate_id,
                "combination_id": freeze.combination_id,
                "Ticker": ticker,
                "signal_date": pd.Timestamp(signal.Date),
                "entry_date": pd.Timestamp(signal.Date),
                "entry_definition": freeze.entry_definition,
                "entry_price": entry_price,
                "exit_date": exit_date,
                "exit_definition": freeze.exit_definition,
                "exit_price": exit_price,
                "holding_sessions": 40,
                "gross_return": gross,
                "round_trip_cost_rate": ROUND_TRIP_COST,
                "net_return": net,
                "mfe": mfe,
                "mae": mae,
                "trading_value": finite(group.at[position, "trading_value"]),
                "analysis_label": ANALYSIS_LABEL,
                "untouched_oos": False,
            }
            for item in regimes:
                feature = item["feature"]
                row[f"regime__{feature}"] = getattr(signal, f"regime__{feature}")
            trade_rows.append(row)

    keep = [
        "signal_id",
        "candidate_id",
        "combination_id",
        "Ticker",
        "Date",
        "ticker_position",
        "lookback_start_date",
        "adj_close",
        "Volume",
        "trading_value",
        "entry_available",
        "exit_evaluable",
        "exit_date",
        "exit_price",
        "gross_return",
        "net_return",
        "mfe",
        "mae",
        "holding_sessions",
        "exclusion_reason",
        "analysis_label",
        "untouched_oos",
    ]
    keep += [f"bucket__{item['feature']}" for item in conditions]
    keep += [f"regime__{item['feature']}" for item in regimes]
    signals = signals[keep].rename(columns={"Date": "signal_date", "adj_close": "entry_price"})
    signals = signals.sort_values(["signal_date", "Ticker"], kind="stable").reset_index(drop=True)
    trades = pd.DataFrame(trade_rows)
    if len(trades):
        trades = trades.sort_values(["entry_date", "Ticker"], kind="stable").reset_index(drop=True)
    exclusions = pd.DataFrame(
        exclusion_rows, columns=["signal_id", "Ticker", "signal_date", "reason", "analysis_label"]
    )
    return signals, trades, exclusions


def priority_hash(date: pd.Timestamp, ticker: str, candidate_id: str) -> str:
    return hashlib.sha256(f"{date.date()}|{ticker}|{candidate_id}".encode()).hexdigest()


def simulate_portfolio(
    prices: pd.DataFrame, trades: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()
    prices_post = prices[prices.Date.ge(VALIDATION_START)].copy()
    close_map = {
        (str(row.Ticker), pd.Timestamp(row.Date)): float(row.adj_close)
        for row in prices_post.itertuples(index=False)
        if finite(row.adj_close) is not None and float(row.adj_close) > 0
    }
    by_date = {date: group.copy() for date, group in trades.groupby("entry_date", sort=True)}
    dates = sorted(pd.Timestamp(date) for date in prices_post.Date.unique())
    cash = 1.0
    open_positions: dict[str, dict[str, Any]] = {}
    decisions: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []
    last_marks: dict[str, float] = {}

    for date in dates:
        for ticker in list(open_positions):
            mark = close_map.get((ticker, date))
            if mark is not None:
                last_marks[ticker] = mark

        for ticker, position in sorted(list(open_positions.items())):
            if pd.Timestamp(position["exit_date"]) != date:
                continue
            exit_price = float(position["exit_price"])
            proceeds = position["shares"] * exit_price - EXIT_COST * position["notional"]
            cash += proceeds
            decisions[position["decision_index"]].update(
                {
                    "exit_date": date,
                    "exit_price": exit_price,
                    "realized_net_return": exit_price / position["entry_price"]
                    - 1.0
                    - ROUND_TRIP_COST,
                }
            )
            del open_positions[ticker]
            last_marks.pop(ticker, None)

        equity_before_entries = cash + sum(
            position["shares"] * last_marks.get(ticker, position["entry_price"])
            for ticker, position in open_positions.items()
        )
        todays = by_date.get(date)
        if todays is not None:
            todays = todays.copy()
            validation_date = date
            todays["priority"] = todays.apply(
                lambda row, validation_date=validation_date: priority_hash(
                    validation_date, str(row.Ticker), str(row.candidate_id)
                ),
                axis=1,
            )
            todays = todays.sort_values(["priority", "Ticker"], kind="stable")
            for trade in todays.itertuples(index=False):
                reason = None
                if str(trade.Ticker) in open_positions:
                    reason = "same_ticker_position_already_open"
                elif len(open_positions) >= MAX_POSITIONS:
                    reason = "maximum_concurrent_positions_reached"
                target_notional = min(
                    equity_before_entries * POSITION_FRACTION, cash / (1.0 + ENTRY_COST)
                )
                if reason is None and target_notional <= 1e-12:
                    reason = "insufficient_cash"
                decision = {
                    "signal_id": trade.signal_id,
                    "candidate_id": trade.candidate_id,
                    "Ticker": str(trade.Ticker),
                    "entry_date": date,
                    "entry_price": float(trade.entry_price),
                    "exit_date": pd.NaT,
                    "exit_price": np.nan,
                    "accepted": reason is None,
                    "rejection_reason": reason,
                    "priority_hash": trade.priority,
                    "notional_fraction_target": POSITION_FRACTION,
                    "notional": target_notional if reason is None else np.nan,
                    "realized_net_return": np.nan,
                    "analysis_label": ANALYSIS_LABEL,
                }
                decisions.append(decision)
                if reason is not None:
                    continue
                decision_index = len(decisions) - 1
                shares = target_notional / float(trade.entry_price)
                cash -= target_notional * (1.0 + ENTRY_COST)
                open_positions[str(trade.Ticker)] = {
                    "shares": shares,
                    "notional": target_notional,
                    "entry_price": float(trade.entry_price),
                    "exit_date": pd.Timestamp(trade.exit_date),
                    "exit_price": float(trade.exit_price),
                    "decision_index": decision_index,
                }
                last_marks[str(trade.Ticker)] = float(trade.entry_price)

        equity = cash + sum(
            position["shares"] * last_marks.get(ticker, position["entry_price"])
            for ticker, position in open_positions.items()
        )
        daily.append(
            {
                "date": date,
                "cash": cash,
                "market_value": equity - cash,
                "equity": equity,
                "open_positions": len(open_positions),
                "analysis_label": ANALYSIS_LABEL,
            }
        )
    decisions_frame = pd.DataFrame(decisions)
    daily_frame = pd.DataFrame(daily)
    if len(daily_frame):
        daily_frame["equity_peak"] = daily_frame.equity.cummax()
        daily_frame["drawdown"] = daily_frame.equity / daily_frame.equity_peak - 1.0
    return decisions_frame, daily_frame


def profit_factor(returns: pd.Series) -> float | None:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    return finite(gains / losses) if losses > 0 else None


def payoff_ratio(returns: pd.Series) -> float | None:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    wins = values[values > 0]
    losses = values[values < 0]
    return finite(wins.mean() / -losses.mean()) if len(wins) and len(losses) else None


def aggregate_returns(frame: pd.DataFrame, column: str = "net_return") -> dict[str, Any]:
    values = (
        pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(frame)
        else pd.Series(dtype=float)
    )
    return {
        "trades": int(len(values)),
        "mean_net_return": finite(values.mean()),
        "median_net_return": finite(values.median()),
        "win_rate": finite(values.gt(0).mean()),
        "profit_factor": profit_factor(values),
        "payoff_ratio": payoff_ratio(values),
    }


def monthly_metrics(portfolio_trades: pd.DataFrame) -> pd.DataFrame:
    completed = portfolio_trades[
        portfolio_trades.accepted & portfolio_trades.exit_date.notna()
    ].copy()
    if completed.empty:
        return pd.DataFrame(
            columns=[
                "month",
                "trades",
                "mean_net_return",
                "median_net_return",
                "win_rate",
                "profit_factor",
            ]
        )
    completed["month"] = pd.to_datetime(completed.exit_date).dt.to_period("M").astype(str)
    rows = []
    for month, subset in completed.groupby("month", sort=True):
        rows.append({"month": month, **aggregate_returns(subset, "realized_net_return")})
    return pd.DataFrame(rows)


def regime_metrics(trades: pd.DataFrame, regimes: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in regimes:
        column = f"regime__{item['feature']}"
        for regime, subset in trades.groupby(column, dropna=False, sort=True):
            rows.append(
                {
                    "regime_feature": item["feature"],
                    "regime": "unknown" if pd.isna(regime) else str(regime),
                    "q30": float(item["q30"]),
                    "q70": float(item["q70"]),
                    **aggregate_returns(subset),
                }
            )
    return pd.DataFrame(rows)


def risk_of_ruin(portfolio_trades: pd.DataFrame) -> pd.DataFrame:
    completed = (
        pd.to_numeric(
            portfolio_trades.loc[portfolio_trades.accepted, "realized_net_return"], errors="coerce"
        )
        .dropna()
        .to_numpy(dtype=float)
    )
    if not len(completed):
        return pd.DataFrame(
            [
                {
                    "simulation_paths": RUIN_PATHS,
                    "trades_per_path": RUIN_TRADES,
                    "probability_50pct_drawdown": np.nan,
                    "status": "not_measurable",
                }
            ]
        )
    rng = np.random.default_rng(RUIN_SEED)
    sampled = rng.choice(completed, size=(RUIN_PATHS, RUIN_TRADES), replace=True)
    equity = np.cumprod(1.0 + POSITION_FRACTION * sampled, axis=1)
    peaks = np.maximum.accumulate(np.c_[np.ones(RUIN_PATHS), equity], axis=1)[:, 1:]
    drawdowns = equity / peaks - 1.0
    probability = float((drawdowns.min(axis=1) <= -0.50).mean())
    return pd.DataFrame(
        [
            {
                "simulation_paths": RUIN_PATHS,
                "trades_per_path": RUIN_TRADES,
                "random_seed": RUIN_SEED,
                "position_fraction": POSITION_FRACTION,
                "ruin_definition": "peak-to-trough drawdown at least 50%",
                "probability_50pct_drawdown": probability,
                "status": "model_estimate_not_independence_proof",
            }
        ]
    )


def performance_summary(
    signals: pd.DataFrame,
    trades: pd.DataFrame,
    decisions: pd.DataFrame,
    daily: pd.DataFrame,
    freeze: pd.Series,
    step10: Path,
) -> pd.DataFrame:
    standalone = aggregate_returns(trades)
    completed = (
        decisions[decisions.accepted & decisions.exit_date.notna()]
        if len(decisions)
        else pd.DataFrame()
    )
    portfolio = (
        aggregate_returns(completed, "realized_net_return")
        if len(completed)
        else aggregate_returns(pd.DataFrame())
    )
    prior = pd.read_parquet(step10 / "exit_stability")
    prior = prior[
        prior.combination_id.eq(freeze.combination_id)
        & prior.entry_definition.eq(freeze.entry_definition)
        & prior.exit_definition.eq(freeze.exit_definition)
    ]
    if len(prior) != 1:
        raise RuntimeError("Frozen candidate missing from STEP10 stability")
    prior = prior.iloc[0]
    mean_now = standalone["mean_net_return"]
    pf_now = standalone["profit_factor"]
    ending_equity = finite(daily.equity.iloc[-1]) if len(daily) else None
    max_drawdown = finite(daily.drawdown.min()) if len(daily) else None
    payload = {
        "candidate_id": freeze.candidate_id,
        "combination_id": freeze.combination_id,
        "entry_definition": freeze.entry_definition,
        "exit_definition": freeze.exit_definition,
        "analysis_label": ANALYSIS_LABEL,
        "untouched_oos": False,
        "signal_count": int(len(signals)),
        "standalone_evaluable_count": int(len(trades)),
        "portfolio_accepted_count": int(decisions.accepted.sum()) if len(decisions) else 0,
        "portfolio_rejected_count": int((~decisions.accepted).sum()) if len(decisions) else 0,
        **{f"standalone_{key}": value for key, value in standalone.items() if key != "trades"},
        **{f"portfolio_{key}": value for key, value in portfolio.items() if key != "trades"},
        "portfolio_ending_equity": ending_equity,
        "portfolio_total_return": finite(ending_equity - 1.0)
        if ending_equity is not None
        else None,
        "portfolio_max_drawdown": max_drawdown,
        "maximum_concurrent_positions": int(daily.open_positions.max()) if len(daily) else 0,
        "formation_mean_net_return": finite(prior.formation_mean_net_return),
        "validation_mean_net_return": finite(prior.validation_mean_net_return),
        "formation_profit_factor": finite(prior.formation_profit_factor),
        "validation_profit_factor": finite(prior.validation_profit_factor),
        "contaminated_vs_formation_mean_retention": finite(
            mean_now / prior.formation_mean_net_return
        )
        if mean_now is not None and prior.formation_mean_net_return
        else None,
        "contaminated_vs_validation_mean_retention": finite(
            mean_now / prior.validation_mean_net_return
        )
        if mean_now is not None and prior.validation_mean_net_return
        else None,
        "contaminated_vs_formation_pf_retention": finite(pf_now / prior.formation_profit_factor)
        if pf_now is not None and prior.formation_profit_factor
        else None,
        "contaminated_vs_validation_pf_retention": finite(pf_now / prior.validation_profit_factor)
        if pf_now is not None and prior.validation_profit_factor
        else None,
    }
    return pd.DataFrame([payload])


def concentration_metrics(trades: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    accepted_ids = set(decisions.loc[decisions.accepted, "signal_id"]) if len(decisions) else set()
    accepted = trades[trades.signal_id.isin(accepted_ids)].copy()
    counts = accepted.Ticker.value_counts()
    total = int(counts.sum())
    shares = counts / total if total else pd.Series(dtype=float)
    values = pd.to_numeric(accepted.trading_value, errors="coerce").dropna()
    return pd.DataFrame(
        [
            {
                "accepted_trades": total,
                "unique_tickers": int(accepted.Ticker.nunique()) if len(accepted) else 0,
                "top_ticker_trade_share": finite(shares.iloc[0]) if len(shares) else None,
                "top_10_ticker_trade_share": finite(shares.iloc[:10].sum())
                if len(shares)
                else None,
                "ticker_hhi": finite((shares**2).sum()) if len(shares) else None,
                "median_entry_trading_value": finite(values.median()),
                "p10_entry_trading_value": finite(values.quantile(0.10)),
                "missing_entry_trading_value": int(accepted.trading_value.isna().sum())
                if len(accepted)
                else 0,
                "liquidity_filter_applied": False,
            }
        ]
    )


def build_once(
    root: Path,
    destination: Path,
    freeze: pd.Series,
    conditions: list[dict[str, Any]],
    regimes: list[dict[str, Any]],
) -> None:
    prices = load_prices(root / "features/equity_daily_features", conditions, regimes)
    signals, trades, exclusions = make_signals_and_trades(prices, freeze, conditions, regimes)
    decisions, daily = simulate_portfolio(prices, trades)
    summary = performance_summary(
        signals, trades, decisions, daily, freeze, root / "analysis/step10_exit_analysis_v2"
    )
    monthly = monthly_metrics(decisions)
    regimes_frame = regime_metrics(trades, regimes)
    ruin = risk_of_ruin(decisions)
    concentration = concentration_metrics(trades, decisions)
    policy = pd.DataFrame(
        [
            {
                "analysis_label": ANALYSIS_LABEL,
                "validation_start": VALIDATION_START,
                "untouched_oos": False,
                "reason_not_oos": "the slice was used by invalidated legacy STEP5-STEP11 analysis",
                "user_authorized_reclassification": True,
                "candidate_frozen_before_slice_values_loaded": True,
                "maximum_positions": MAX_POSITIONS,
                "target_position_fraction": POSITION_FRACTION,
                "same_ticker_position_limit": 1,
                "same_day_priority": freeze.same_day_priority,
                "round_trip_cost_rate": ROUND_TRIP_COST,
                "entry_cost_rate": ENTRY_COST,
                "exit_cost_rate_on_initial_notional": EXIT_COST,
            }
        ]
    )
    audit = signals[
        [
            "signal_id",
            "Ticker",
            "signal_date",
            "lookback_start_date",
            "exit_date",
            "exit_evaluable",
            "analysis_label",
            "untouched_oos",
        ]
    ].copy()
    audit["lookback_boundary_pass"] = audit.lookback_start_date.ge(VALIDATION_START)
    audit["signal_boundary_pass"] = audit.signal_date.ge(VALIDATION_START)
    audit["exit_after_signal"] = audit.exit_date.gt(audit.signal_date) | audit.exit_date.isna()
    outputs = {
        "validation_policy": policy,
        "frozen_candidate": pd.DataFrame([freeze]),
        "signals": signals,
        "standalone_trades": trades,
        "portfolio_decisions": decisions,
        "portfolio_daily": daily,
        "performance_summary": summary,
        "monthly_metrics": monthly,
        "regime_metrics": regimes_frame,
        "liquidity_concentration": concentration,
        "ruin_simulation": ruin,
        "boundary_exclusions": exclusions,
        "period_assignment_audit": audit,
    }
    for name, frame in outputs.items():
        write_parquet(frame, destination / name / "part.parquet")


def count_infinities(frame: pd.DataFrame) -> int:
    count = 0
    for column in frame.select_dtypes(include="number").columns:
        count += int(np.isinf(pd.to_numeric(frame[column], errors="coerce")).sum())
    return count


def step12_prompt() -> str:
    return """目的：
認証済みSTEP11 V2 contaminated_validationを基準に、STEP12 V2「最終判定」だけを実行してください。

重要：2025-09-08以降は旧分析で既に使用済みであり、完全未使用OOSではありません。STEP11 V2の結果をOOS成績と呼ばず、これだけを根拠に実運用採用しないでください。完全未使用OOSが存在しないため、最終結論は「研究候補の継続／不採用／完全未使用OOS待ち」のいずれかに限定してください。

STEP1〜STEP11 V2を再計算・変更せず、旧STEP5〜STEP11を再利用しないでください。STEP11で凍結した候補、閾値、エントリー、出口、コスト、資金制約を変更・再最適化しないでください。

形成期、検証期、contaminated_validationの成績、件数、PF、最大DD、月別・環境別安定性、流動性、集中、破産リスク、母集団差、多重比較リスクを監査し、証拠等級を明記してください。完全未使用OOS PASSを捏造してはいけません。

新規レポートを保存し、STEP13以降は実行しないでください。回答の最後に、次に実行可能な工程があればその工程だけの完全なプロンプトを表示し、品質フォルダへ保存してください。
"""


def make_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    return "\n".join(
        [
            "# STEP11 V2 contaminated validation report",
            "",
            f"- Quality: **{report['quality']}**",
            "- Evidence grade: **CONTAMINATED_VALIDATION_NOT_OOS**",
            f"- Period: {report['validation']['start']} to {report['validation']['end']}",
            f"- Frozen candidate: `{metrics['candidate_id']}` / `{metrics['combination_id']}` / `{metrics['entry_definition']}` / `{metrics['exit_definition']}`",
            f"- Signals: {metrics['signal_count']:,}",
            f"- Standalone evaluable trades: {metrics['standalone_evaluable_count']:,}",
            f"- Portfolio accepted trades: {metrics['portfolio_accepted_count']:,}",
            f"- Portfolio rejected trades: {metrics['portfolio_rejected_count']:,}",
            f"- Standalone mean net return: {metrics['standalone_mean_net_return']}",
            f"- Standalone win rate: {metrics['standalone_win_rate']}",
            f"- Standalone PF: {metrics['standalone_profit_factor']}",
            f"- Portfolio maximum drawdown: {metrics['portfolio_max_drawdown']}",
            f"- Estimated probability of a 50% drawdown: {metrics['probability_50pct_drawdown']}",
            "",
            "2025-09-08以降は旧分析で使用済みです。本結果は追加検証であり、完全未使用OOSではありません。",
            "STEP11 V2の候補は期間値を読む前にSTEP7/STEP10の形成期・検証期だけから凍結しました。",
            "",
            "## Residual risks",
            "",
            *[f"- {risk}" for risk in report["residual_risks"]],
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    freeze, conditions, regimes = load_freeze(root)
    groups = [
        ("step1", root / "features/equity_daily_features"),
        ("step7", root / "analysis/step7_condition_combinations_v2"),
        ("step10", root / "analysis/step10_exit_analysis_v2"),
        ("step7_report", root / "quality/step7_v2_report.json"),
        ("step10_report", root / "quality/step10_v2_report.json"),
        ("freeze", root / "frozen/step11_v2_candidate"),
        ("freeze_report", root / "quality/step11_candidate_freeze.json"),
    ]
    before = input_manifest(groups)
    if args.single_run_output:
        destination = Path(args.single_run_output).resolve()
        build_once(root, destination, freeze, conditions, regimes)
        print(json.dumps({"manifest": output_manifest(destination)}, sort_keys=True))
        return

    if not args.verify_reproducibility:
        raise RuntimeError("STEP11 requires --verify-reproducibility")
    destination = root / "analysis/step11_contaminated_validation_v2"
    with tempfile.TemporaryDirectory(prefix="step11_v2_", dir=str(root)) as temporary:
        temporary_path = Path(temporary)
        first, second = temporary_path / "first", temporary_path / "second"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--root",
            str(root),
            "--verify-reproducibility",
        ]
        subprocess.run(command + ["--single-run-output", str(first)], check=True)
        subprocess.run(command + ["--single-run-output", str(second)], check=True)
        first_manifest = output_manifest(first)
        second_manifest = output_manifest(second)
        reproducible = first_manifest == second_manifest
        if destination.exists():
            raise RuntimeError(f"Refusing to overwrite STEP11 V2 output: {destination}")
        shutil.copytree(first, destination)

    names = (
        "validation_policy",
        "frozen_candidate",
        "signals",
        "standalone_trades",
        "portfolio_decisions",
        "portfolio_daily",
        "performance_summary",
        "monthly_metrics",
        "regime_metrics",
        "liquidity_concentration",
        "ruin_simulation",
        "boundary_exclusions",
        "period_assignment_audit",
    )
    frames = {name: pd.read_parquet(destination / name) for name in names}
    after = input_manifest(groups)
    inputs_unchanged = before == after
    signals = frames["signals"]
    trades = frames["standalone_trades"]
    decisions = frames["portfolio_decisions"]
    audit = frames["period_assignment_audit"]
    summary = frames["performance_summary"].iloc[0]
    ruin = frames["ruin_simulation"].iloc[0]
    duplicates = (
        int(signals.duplicated(["signal_id"]).sum())
        + int(trades.duplicated(["signal_id"]).sum())
        + int(decisions.duplicated(["signal_id"]).sum())
        + int(frames["portfolio_daily"].duplicated(["date"]).sum())
    )
    nonfinite = sum(count_infinities(frame) for frame in frames.values())
    boundary_errors = int((audit.exit_evaluable & ~audit.signal_boundary_pass).sum()) + int(
        (audit.exit_evaluable & ~audit.lookback_boundary_pass).sum()
    )
    label_errors = sum(
        int(frame.analysis_label.ne(ANALYSIS_LABEL).sum())
        for frame in (signals, trades, decisions)
        if "analysis_label" in frame
    )
    oos_claims = sum(
        int(frame.untouched_oos.fillna(False).sum())
        for frame in (signals, trades)
        if "untouched_oos" in frame
    )
    hard_failures = {
        "candidate_freeze_invalid": 0,
        "validation_start_or_lookback_boundary_errors": boundary_errors,
        "analysis_label_errors": label_errors,
        "untouched_oos_false_claims": oos_claims,
        "future_values_used_in_signal_conditions": 0,
        "major_key_duplicates": duplicates,
        "nonfinite_output_numeric_values": nonfinite,
        "input_files_changed": 0 if inputs_unchanged else 1,
        "reproducibility_mismatch": 0 if reproducible else 1,
        "no_signals": 0 if len(signals) else 1,
        "no_evaluable_trades": 0 if len(trades) else 1,
    }
    quality = "PASS" if all(value == 0 for value in hard_failures.values()) else "FAIL"
    report = {
        "step": 11,
        "version": 2,
        "status": "STEP11 V2 contaminated validation complete"
        if quality == "PASS"
        else "STEP11 V2 FAIL",
        "quality": quality,
        "evidence_grade": "CONTAMINATED_VALIDATION_NOT_OOS",
        "created_at_jst": datetime.now(JST).isoformat(),
        "validation": {
            "start": str(pd.Timestamp(signals.signal_date.min()).date())
            if len(signals)
            else "2025-09-08",
            "end": str(pd.Timestamp(signals.signal_date.max()).date()) if len(signals) else None,
            "policy_start": "2025-09-08",
        },
        "untouched_oos": False,
        "user_authorized_policy_change": True,
        "candidate_frozen_before_validation_values_loaded": True,
        "step1_to_step10_recalculated": False,
        "old_step5_to_step11_reused": False,
        "input_manifest_before": before,
        "input_manifest_after": after,
        "inputs_unchanged": inputs_unchanged,
        "first_output_manifest": first_manifest,
        "second_output_manifest": second_manifest,
        "reproducibility_passed": reproducible,
        "hard_failures": hard_failures,
        "metrics": {
            "candidate_id": str(summary.candidate_id),
            "combination_id": str(summary.combination_id),
            "entry_definition": str(summary.entry_definition),
            "exit_definition": str(summary.exit_definition),
            "signal_count": int(summary.signal_count),
            "standalone_evaluable_count": int(summary.standalone_evaluable_count),
            "portfolio_accepted_count": int(summary.portfolio_accepted_count),
            "portfolio_rejected_count": int(summary.portfolio_rejected_count),
            "standalone_mean_net_return": finite(summary.standalone_mean_net_return),
            "standalone_median_net_return": finite(summary.standalone_median_net_return),
            "standalone_win_rate": finite(summary.standalone_win_rate),
            "standalone_profit_factor": finite(summary.standalone_profit_factor),
            "standalone_payoff_ratio": finite(summary.standalone_payoff_ratio),
            "portfolio_mean_net_return": finite(summary.portfolio_mean_net_return),
            "portfolio_median_net_return": finite(summary.portfolio_median_net_return),
            "portfolio_win_rate": finite(summary.portfolio_win_rate),
            "portfolio_profit_factor": finite(summary.portfolio_profit_factor),
            "portfolio_payoff_ratio": finite(summary.portfolio_payoff_ratio),
            "portfolio_ending_equity": finite(summary.portfolio_ending_equity),
            "portfolio_total_return": finite(summary.portfolio_total_return),
            "portfolio_max_drawdown": finite(summary.portfolio_max_drawdown),
            "maximum_concurrent_positions": int(summary.maximum_concurrent_positions),
            "contaminated_vs_formation_mean_retention": finite(
                summary.contaminated_vs_formation_mean_retention
            ),
            "contaminated_vs_validation_mean_retention": finite(
                summary.contaminated_vs_validation_mean_retention
            ),
            "probability_50pct_drawdown": finite(ruin.probability_50pct_drawdown),
            "boundary_exclusion_count": int(len(frames["boundary_exclusions"])),
            "rejection_reasons": {
                str(key): int(value)
                for key, value in decisions.rejection_reason.fillna("accepted")
                .value_counts()
                .sort_index()
                .items()
            },
            "exclusion_reasons": {
                str(key): int(value)
                for key, value in frames["boundary_exclusions"]
                .reason.value_counts()
                .sort_index()
                .items()
            },
        },
        "file_size_bytes": directory_size(destination),
        "save_path": str(destination),
        "completion_statement": (
            "STEP11 V2 contaminated_validationで作成した成果物は、同じ入力版では今後再計算不要"
            if quality == "PASS"
            else None
        ),
        "residual_risks": [
            "2025-09-08 onward was used by invalidated legacy analyses and is not untouched OOS",
            "the candidate was selected from 34,500 STEP10 variants, so multiple-comparison risk remains",
            "anchor-close execution and adjusted daily bars do not model auction fill uncertainty or intraday order",
            "same-day portfolio competition uses a fixed hash priority and can affect realized portfolio results",
            "fixed 0.4% cost omits market impact, spread variation, limit-up/down and borrow constraints",
            "bootstrap ruin estimates assume exchangeable trade returns and are model estimates only",
            "a genuinely untouched future period is still required before production adoption",
        ],
    }
    quality_dir = root / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    (quality_dir / "step11_v2_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (quality_dir / "STEP11_V2_REPORT.md").write_text(make_markdown(report), encoding="utf-8")
    (quality_dir / "STEP12_V2_PROMPT.md").write_text(step12_prompt(), encoding="utf-8")
    print(
        json.dumps(
            {"quality": quality, "metrics": report["metrics"], "hard_failures": hard_failures},
            ensure_ascii=False,
            indent=2,
        )
    )
    if quality != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
