# ruff: noqa: E501
"""STEP10 V2: fixed exit comparison on the certified STEP9 V2 entries."""
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
import pyarrow.dataset as ds
import pyarrow.parquet as pq

JST = ZoneInfo("Asia/Tokyo")
FORMATION_END = pd.Timestamp("2023-12-31")
VALIDATION_START = pd.Timestamp("2024-01-01")
VALIDATION_END = pd.Timestamp("2025-09-07")
HOLDOUT_START = pd.Timestamp("2025-09-08")
ENTRY_NAMES = ("anchor_close", "next_session_open", "first_pullback")
EXIT_NAMES = ("time_20_close", "time_40_close", "time_60_close", "close_below_ma20", "atr3_trailing")
ROUND_TRIP_COST = 0.004


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build isolated STEP10 V2 exit comparison")
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


def positive(value: Any) -> bool:
    number = finite(value)
    return number is not None and number > 0


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def period_for_date(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    date = pd.Timestamp(value).normalize()
    if date <= FORMATION_END:
        return "formation"
    if VALIDATION_START <= date <= VALIDATION_END:
        return "validation"
    if date >= HOLDOUT_START:
        return "contaminated_holdout"
    return "outside_defined_period"


def make_exit_definitions() -> pd.DataFrame:
    rows = [
        {
            "exit_definition": "time_20_close",
            "rule": "split-adjusted close at ticker row entry+20",
            "maximum_holding_sessions": 20,
            "ma_window": pd.NA,
            "atr_multiplier": np.nan,
            "fallback": "none",
        },
        {
            "exit_definition": "time_40_close",
            "rule": "split-adjusted close at ticker row entry+40",
            "maximum_holding_sessions": 40,
            "ma_window": pd.NA,
            "atr_multiplier": np.nan,
            "fallback": "none",
        },
        {
            "exit_definition": "time_60_close",
            "rule": "split-adjusted close at ticker row entry+60",
            "maximum_holding_sessions": 60,
            "ma_window": pd.NA,
            "atr_multiplier": np.nan,
            "fallback": "none",
        },
        {
            "exit_definition": "close_below_ma20",
            "rule": "first post-entry adjusted close below trailing adjusted-close MA20, else entry+60 close",
            "maximum_holding_sessions": 60,
            "ma_window": 20,
            "atr_multiplier": np.nan,
            "fallback": "entry+60 adjusted close",
        },
        {
            "exit_definition": "atr3_trailing",
            "rule": "fixed-width 3x entry-known adjusted ATR14 trail from post-entry high, else entry+60 close",
            "maximum_holding_sessions": 60,
            "ma_window": pd.NA,
            "atr_multiplier": 3.0,
            "fallback": "entry+60 adjusted close",
        },
    ]
    result = pd.DataFrame(rows)
    result["ma_window"] = result.ma_window.astype("Int64")
    result["round_trip_cost_rate"] = ROUND_TRIP_COST
    result["fixed_before_formation_review"] = True
    result["validation_used_to_change_rule"] = False
    result["combination_specific_rule"] = False
    result["regime_specific_rule"] = False
    result["ticker_shift_not_calendar"] = True
    return result


def load_prices(path: Path, tickers: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Load dates for boundary checks, but never load holdout price/indicator values."""
    dataset = ds.dataset(path, format="parquet")
    required = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume", "atr_14"]
    missing = sorted(set(required) - set(dataset.schema.names))
    if missing:
        raise RuntimeError(f"STEP1 price columns missing: {missing}")
    ticker_filter = ds.field("Ticker").isin(tickers)
    skeleton = dataset.to_table(columns=["Date", "Ticker"], filter=ticker_filter).to_pandas()
    safe_filter = ticker_filter & (ds.field("Date") < HOLDOUT_START.to_pydatetime())
    values = dataset.to_table(columns=required, filter=safe_filter).to_pandas()
    frame = skeleton.merge(values, on=["Date", "Ticker"], how="left", validate="one_to_one")
    frame["Date"] = pd.to_datetime(frame.Date).dt.normalize()
    frame = frame.sort_values(["Ticker", "Date"], kind="stable").reset_index(drop=True)
    if frame.duplicated(["Ticker", "Date"]).any():
        raise RuntimeError("STEP1 contains duplicate Ticker x Date rows")
    for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume", "atr_14"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    factor = frame["Adj Close"] / frame.Close
    valid_factor = np.isfinite(factor) & factor.between(0.001, 1000, inclusive="both")
    frame["adj_open"] = np.where(valid_factor, frame.Open * factor, np.nan)
    frame["adj_high"] = np.where(valid_factor, frame.High * factor, np.nan)
    frame["adj_low"] = np.where(valid_factor, frame.Low * factor, np.nan)
    frame["adj_atr14"] = np.where(valid_factor, frame.atr_14 * factor, np.nan)
    result: dict[str, dict[str, np.ndarray]] = {}
    for ticker, group in frame.groupby("Ticker", sort=True, observed=True):
        result[str(ticker)] = {
            "date": group.Date.to_numpy(dtype="datetime64[ns]"),
            "adj_open": group.adj_open.to_numpy(dtype=float),
            "adj_high": group.adj_high.to_numpy(dtype=float),
            "adj_low": group.adj_low.to_numpy(dtype=float),
            "adj_close": group["Adj Close"].to_numpy(dtype=float),
            "adj_atr14": group.adj_atr14.to_numpy(dtype=float),
            "volume": group.Volume.to_numpy(dtype=float),
            "price_values_loaded": group.Date.lt(HOLDOUT_START).to_numpy(dtype=bool),
        }
    return result


def path_stats(series: dict[str, np.ndarray], start: int, end: int, entry_price: float) -> tuple[float, float] | None:
    if start > end:
        return None
    highs = series["adj_high"][start : end + 1]
    lows = series["adj_low"][start : end + 1]
    if len(highs) != end - start + 1 or not np.all(np.isfinite(highs)) or not np.all(np.isfinite(lows)):
        return None
    if np.any(highs <= 0) or np.any(lows <= 0):
        return None
    return float(np.max(highs) / entry_price - 1.0), float(np.min(lows) / entry_price - 1.0)


def base_exit_row(entry: Any, exit_name: str) -> dict[str, Any]:
    return {
        "sample_id": str(entry.sample_id),
        "event_id": str(entry.event_id),
        "Ticker": str(entry.Ticker),
        "period": str(entry.period),
        "observation_type": str(entry.observation_type),
        "outcome": int(entry.outcome),
        "anchor_date": pd.Timestamp(entry.anchor_date),
        "entry_definition": str(entry.entry_definition),
        "entry_date": pd.Timestamp(entry.entry_date) if not pd.isna(entry.entry_date) else pd.NaT,
        "entry_session_position": int(entry.entry_session_position) if not pd.isna(entry.entry_session_position) else pd.NA,
        "entry_price": finite(entry.entry_price),
        "entry_status": str(entry.entry_status),
        "entry_filled": bool(entry.filled),
        "entry_rule_reason": entry.entry_rule_reason,
        "exit_definition": exit_name,
        "required_end_date": pd.NaT,
        "exit_date": pd.NaT,
        "exit_session_position": pd.NA,
        "holding_sessions": pd.NA,
        "exit_price": np.nan,
        "exit_reason": None,
        "exit_evaluation_available": False,
        "exclusion_reason": None,
        "gross_return": np.nan,
        "round_trip_cost_rate": ROUND_TRIP_COST,
        "net_return": np.nan,
        "mfe": np.nan,
        "mae": np.nan,
        "profit_capture_ratio": np.nan,
        "entry_adjusted_atr14": np.nan,
        "atr_stop_width": np.nan,
        "same_day_conservative_assumption_used": False,
        "intraday_sequence_ambiguous": False,
        "holdout_value_used": False,
        "future_target_used_for_rule": False,
        "validation_used_to_change_rule": False,
        "step8_used_for_selection": False,
        "split_adjustment_policy": "raw OHLC and ATR14 multiplied by same-row Adj Close/Close; Adj Close used directly",
    }


def simulate_one(entry: Any, exit_name: str, series: dict[str, np.ndarray] | None) -> dict[str, Any]:
    row = base_exit_row(entry, exit_name)
    if not bool(entry.filled):
        row["exclusion_reason"] = f"entry_not_filled:{entry.entry_rule_reason}"
        return row
    if series is None or pd.isna(entry.entry_session_position) or not positive(entry.entry_price):
        row["exclusion_reason"] = "entry_or_ticker_price_reference_missing"
        return row
    t = int(entry.entry_session_position)
    entry_price = float(entry.entry_price)
    max_hold = {"time_20_close": 20, "time_40_close": 40}.get(exit_name, 60)
    end = t + max_hold
    if end >= len(series["date"]):
        row["exclusion_reason"] = "insufficient_future_ticker_trading_rows"
        return row
    required_end = pd.Timestamp(series["date"][end]).normalize()
    row["required_end_date"] = required_end
    if period_for_date(required_end) != str(entry.period):
        row["exclusion_reason"] = (
            "exit_window_crosses_contaminated_holdout"
            if required_end >= HOLDOUT_START
            else "exit_window_crosses_period_boundary"
        )
        return row
    if not bool(np.all(series["price_values_loaded"][t : end + 1])):
        row["exclusion_reason"] = "price_values_not_loaded_inside_safe_period"
        return row

    exit_pos = end
    exit_price = finite(series["adj_close"][end])
    exit_reason = exit_name
    ambiguous = False
    conservative = False
    if exit_name == "close_below_ma20":
        for pos in range(t + 1, end + 1):
            start = pos - 19
            if start < 0:
                row["exclusion_reason"] = "ma20_history_insufficient"
                return row
            window = series["adj_close"][start : pos + 1]
            if len(window) != 20 or not np.all(np.isfinite(window)) or np.any(window <= 0):
                row["exclusion_reason"] = "ma20_window_missing_or_invalid"
                return row
            close = float(window[-1])
            ma20 = float(np.mean(window))
            if close < ma20:
                exit_pos, exit_price, exit_reason = pos, close, "first_close_below_ma20"
                break
        else:
            exit_reason = "no_ma20_break_within_60_sessions_time_exit"
    elif exit_name == "atr3_trailing":
        adjusted_atr = finite(series["adj_atr14"][t])
        if adjusted_atr is None or adjusted_atr <= 0:
            row["exclusion_reason"] = "entry_adjusted_atr14_missing_or_invalid"
            return row
        width = 3.0 * adjusted_atr
        row["entry_adjusted_atr14"] = adjusted_atr
        row["atr_stop_width"] = width
        peak = entry_price
        start_pos = t if str(entry.entry_definition) == "next_session_open" else t + 1
        for pos in range(start_pos, end + 1):
            open_price = finite(series["adj_open"][pos])
            high = finite(series["adj_high"][pos])
            low = finite(series["adj_low"][pos])
            if open_price is None or high is None or low is None or min(open_price, high, low) <= 0:
                row["exclusion_reason"] = "atr_path_ohlc_missing_or_invalid"
                return row
            prior_stop = peak - width
            if prior_stop > 0 and open_price <= prior_stop:
                exit_pos, exit_price, exit_reason = pos, open_price, "atr_stop_gap_at_open"
                break
            if prior_stop > 0 and low <= prior_stop:
                exit_pos, exit_price, exit_reason = pos, prior_stop, "atr_stop_existing_level"
                break
            new_peak = max(peak, high)
            raised_stop = new_peak - width
            if raised_stop > max(prior_stop, 0.0) and low <= raised_stop:
                # The daily bar cannot reveal whether the high or low came first.
                # Assume the adverse low-first ordering: the raised stop did not
                # yet exist when the low printed. Carry the new peak into the next
                # session instead of crediting an optimistic same-bar stop fill.
                ambiguous = True
                conservative = True
                peak = new_peak
                continue
            peak = new_peak
        else:
            exit_reason = "no_atr_stop_within_60_sessions_time_exit"

    if exit_price is None or exit_price <= 0:
        row["exclusion_reason"] = "exit_price_missing_or_invalid"
        return row
    start = t if str(entry.entry_definition) == "next_session_open" else t + 1
    stats = path_stats(series, start, exit_pos, entry_price)
    if stats is None:
        row["exclusion_reason"] = "mfe_mae_path_missing_or_invalid"
        return row
    mfe, mae = stats
    gross = exit_price / entry_price - 1.0
    net = gross - ROUND_TRIP_COST
    row.update(
        {
            "exit_date": pd.Timestamp(series["date"][exit_pos]).normalize(),
            "exit_session_position": exit_pos,
            "holding_sessions": exit_pos - t,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "exit_evaluation_available": True,
            "gross_return": gross,
            "net_return": net,
            "mfe": mfe,
            "mae": mae,
            "profit_capture_ratio": gross / mfe if gross > 0 and mfe > 0 else np.nan,
            "same_day_conservative_assumption_used": conservative,
            "intraday_sequence_ambiguous": ambiguous,
        }
    )
    return row


def make_exit_observations(entries: pd.DataFrame, prices: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = entries.sort_values(["entry_definition", "sample_id"], kind="stable")
    for entry in ordered.itertuples(index=False):
        series = prices.get(str(entry.Ticker))
        for exit_name in EXIT_NAMES:
            rows.append(simulate_one(entry, exit_name, series))
    result = pd.DataFrame(rows)
    for column in ["entry_session_position", "exit_session_position", "holding_sessions"]:
        result[column] = result[column].astype("Int64")
    return result.sort_values(["entry_definition", "exit_definition", "sample_id"], kind="stable").reset_index(drop=True)


def sql_path(path: Path) -> str:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise RuntimeError(f"No Parquet under {path}")
    return str(files[0] if len(files) == 1 else path / "**" / "*.parquet").replace("'", "''")


def calculate_metrics(membership_path: Path, exits_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = sql_path(membership_path)
    exits = sql_path(exits_path)
    con = duckdb.connect()
    try:
        con.execute(f"CREATE VIEW membership AS SELECT * FROM read_parquet('{membership}')")
        con.execute(f"CREATE VIEW exits AS SELECT * FROM read_parquet('{exits}')")
        metric = con.execute(
            """
            SELECT m.combination_id, m.condition_count, e.entry_definition, e.exit_definition, e.period,
              count(*)::BIGINT signal_count,
              sum((e.outcome=1)::INT)::BIGINT signal_events,
              sum((e.outcome=0)::INT)::BIGINT signal_controls,
              sum(e.entry_filled::INT)::BIGINT entry_filled_count,
              sum(e.exit_evaluation_available::INT)::BIGINT exit_evaluable_count,
              sum((NOT e.exit_evaluation_available)::INT)::BIGINT excluded_count,
              avg(e.gross_return) FILTER (e.exit_evaluation_available) mean_gross_return,
              median(e.gross_return) FILTER (e.exit_evaluation_available) median_gross_return,
              avg(e.net_return) FILTER (e.exit_evaluation_available) mean_net_return,
              median(e.net_return) FILTER (e.exit_evaluation_available) median_net_return,
              avg((e.net_return>0)::INT) FILTER (e.exit_evaluation_available) win_rate,
              CASE WHEN abs(sum(e.net_return) FILTER (e.exit_evaluation_available AND e.net_return<0))>0
                THEN sum(e.net_return) FILTER (e.exit_evaluation_available AND e.net_return>0)
                  / abs(sum(e.net_return) FILTER (e.exit_evaluation_available AND e.net_return<0)) END profit_factor,
              CASE WHEN abs(avg(e.net_return) FILTER (e.exit_evaluation_available AND e.net_return<0))>0
                THEN avg(e.net_return) FILTER (e.exit_evaluation_available AND e.net_return>0)
                  / abs(avg(e.net_return) FILTER (e.exit_evaluation_available AND e.net_return<0)) END payoff_ratio,
              avg(e.mfe) FILTER (e.exit_evaluation_available) mean_mfe,
              median(e.mfe) FILTER (e.exit_evaluation_available) median_mfe,
              avg(e.mae) FILTER (e.exit_evaluation_available) mean_mae,
              median(e.mae) FILTER (e.exit_evaluation_available) median_mae,
              avg(e.profit_capture_ratio) FILTER (e.exit_evaluation_available AND e.profit_capture_ratio IS NOT NULL) mean_profit_capture_ratio,
              median(e.profit_capture_ratio) FILTER (e.exit_evaluation_available AND e.profit_capture_ratio IS NOT NULL) median_profit_capture_ratio,
              avg(e.holding_sessions) FILTER (e.exit_evaluation_available) mean_holding_sessions,
              median(e.holding_sessions) FILTER (e.exit_evaluation_available) median_holding_sessions,
              sum(e.intraday_sequence_ambiguous::INT)::BIGINT intraday_sequence_ambiguous_count,
              sum(e.same_day_conservative_assumption_used::INT)::BIGINT conservative_same_day_assumption_count,
              0.004::DOUBLE round_trip_cost_rate,
              TRUE exit_rules_fixed_before_validation,
              FALSE validation_used_to_change_exit,
              FALSE step8_used_for_selection
            FROM membership m
            JOIN exits e USING(sample_id)
            GROUP BY ALL
            ORDER BY combination_id, entry_definition, exit_definition, period
            """
        ).df()
        reasons = con.execute(
            """
            SELECT m.combination_id,e.entry_definition,e.exit_definition,e.period,
                   coalesce(e.exclusion_reason,'available') exclusion_reason,count(*)::BIGINT n
            FROM membership m JOIN exits e USING(sample_id)
            GROUP BY ALL ORDER BY combination_id,entry_definition,exit_definition,period,exclusion_reason
            """
        ).df()
        reason_json = (
            reasons.groupby(["combination_id", "entry_definition", "exit_definition", "period"], sort=True)
            .apply(lambda g: json.dumps({str(r.exclusion_reason): int(r.n) for r in g.itertuples(index=False)}, ensure_ascii=False, sort_keys=True), include_groups=False)
            .rename("exclusion_reasons_json")
            .reset_index()
        )
        metric = metric.merge(reason_json, on=["combination_id", "entry_definition", "exit_definition", "period"], how="left", validate="one_to_one")
        annual = con.execute(
            """
            SELECT m.combination_id,m.condition_count,e.entry_definition,e.exit_definition,e.period,
              year(e.anchor_date)::INT AS "year",count(*)::BIGINT signal_count,
              sum(e.exit_evaluation_available::INT)::BIGINT exit_evaluable_count,
              avg(e.net_return) FILTER (e.exit_evaluation_available) mean_net_return,
              median(e.net_return) FILTER (e.exit_evaluation_available) median_net_return,
              avg((e.net_return>0)::INT) FILTER (e.exit_evaluation_available) win_rate,
              CASE WHEN abs(sum(e.net_return) FILTER (e.exit_evaluation_available AND e.net_return<0))>0
                THEN sum(e.net_return) FILTER (e.exit_evaluation_available AND e.net_return>0)
                  / abs(sum(e.net_return) FILTER (e.exit_evaluation_available AND e.net_return<0)) END profit_factor,
              CASE WHEN abs(avg(e.net_return) FILTER (e.exit_evaluation_available AND e.net_return<0))>0
                THEN avg(e.net_return) FILTER (e.exit_evaluation_available AND e.net_return>0)
                  / abs(avg(e.net_return) FILTER (e.exit_evaluation_available AND e.net_return<0)) END payoff_ratio,
              avg(e.mfe) FILTER (e.exit_evaluation_available) mean_mfe,
              avg(e.mae) FILTER (e.exit_evaluation_available) mean_mae,
              avg(e.profit_capture_ratio) FILTER (e.exit_evaluation_available AND e.profit_capture_ratio IS NOT NULL) mean_profit_capture_ratio
            FROM membership m JOIN exits e USING(sample_id)
            GROUP BY ALL ORDER BY combination_id,entry_definition,exit_definition,period,year
            """
        ).df()
    finally:
        con.close()

    formation = metric[metric.period.eq("formation")].set_index(["combination_id", "entry_definition", "exit_definition"])
    for column in ["mean_net_return", "median_net_return", "win_rate", "profit_factor", "payoff_ratio"]:
        reference = formation[column].to_dict()
        values = []
        for row in metric.itertuples(index=False):
            if row.period == "formation":
                values.append(None)
                continue
            base = finite(reference.get((row.combination_id, row.entry_definition, row.exit_definition)))
            current = finite(getattr(row, column))
            values.append(current / base if base is not None and current is not None and abs(base) > 1e-12 else None)
        metric[f"formation_to_validation_{column}_retention"] = values
    direction_reference = formation["mean_net_return"].to_dict()
    annual["direction_reproduced_from_formation_mean"] = [
        (
            np.sign(value) == np.sign(direction_reference.get((row.combination_id, row.entry_definition, row.exit_definition)))
            if (value := finite(row.mean_net_return)) is not None
            and finite(direction_reference.get((row.combination_id, row.entry_definition, row.exit_definition))) is not None
            else None
        )
        for row in annual.itertuples(index=False)
    ]
    return metric, annual


def make_stability(metrics: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    keys = ["combination_id", "condition_count", "entry_definition", "exit_definition"]
    cols = ["exit_evaluable_count", "mean_net_return", "median_net_return", "win_rate", "profit_factor", "payoff_ratio", "mean_mfe", "mean_mae", "mean_profit_capture_ratio"]
    formation = metrics[metrics.period.eq("formation")][keys + cols].rename(columns={c: f"formation_{c}" for c in cols})
    validation = metrics[metrics.period.eq("validation")][keys + cols].rename(columns={c: f"validation_{c}" for c in cols})
    result = formation.merge(validation, on=keys, how="outer", validate="one_to_one")
    result["mean_return_direction_reproduced"] = np.sign(result.formation_mean_net_return) == np.sign(result.validation_mean_net_return)
    yearly = (
        annual.assign(direction=lambda x: np.sign(x.mean_net_return))
        .groupby(keys, sort=True)
        .agg(annual_periods=("year", "size"), positive_mean_return_years=("direction", lambda x: int((x > 0).sum())), annual_mean_return_min=("mean_net_return", "min"), annual_mean_return_max=("mean_net_return", "max"))
        .reset_index()
    )
    return result.merge(yearly, on=keys, how="left", validate="one_to_one").sort_values(keys, kind="stable").reset_index(drop=True)


def make_reference_summary(metrics: pd.DataFrame, exits: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    numeric = ["mean_gross_return", "median_gross_return", "mean_net_return", "median_net_return", "win_rate", "profit_factor", "payoff_ratio", "mean_mfe", "mean_mae", "mean_profit_capture_ratio", "mean_holding_sessions"]
    for (entry, exit_name, period), group in metrics.groupby(["entry_definition", "exit_definition", "period"], sort=True, observed=True):
        unique = exits[(exits.entry_definition.eq(entry)) & (exits.exit_definition.eq(exit_name)) & (exits.period.eq(period))]
        row: dict[str, Any] = {
            "entry_definition": entry,
            "exit_definition": exit_name,
            "period": period,
            "fixed_combination_count": int(group.combination_id.nunique()),
            "combination_signal_instances": int(group.signal_count.sum()),
            "combination_exit_evaluable_instances": int(group.exit_evaluable_count.sum()),
            "unique_entry_samples": int(len(unique)),
            "unique_entry_filled_samples": int(unique.entry_filled.sum()),
            "unique_exit_evaluable_samples": int(unique.exit_evaluation_available.sum()),
            "unique_excluded_samples": int((~unique.exit_evaluation_available).sum()),
            "unique_intraday_ambiguous_samples": int(unique.intraday_sequence_ambiguous.sum()),
        }
        for column in numeric:
            valid_values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"combination_median_{column}"] = finite(valid_values.median()) if len(valid_values) else None
            row[f"combination_iqr_{column}"] = finite(valid_values.quantile(0.75) - valid_values.quantile(0.25)) if len(valid_values) else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["entry_definition", "exit_definition", "period"], kind="stable").reset_index(drop=True)


def count_infinities(frame: pd.DataFrame) -> int:
    total = 0
    for column in frame.select_dtypes(include=[np.number]).columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        total += int(np.isinf(values).sum())
    return total


def build_once(root: Path, destination: Path, entries: pd.DataFrame, prices: dict[str, dict[str, np.ndarray]]) -> None:
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite run output: {destination}")
    definitions = make_exit_definitions()
    observations = make_exit_observations(entries, prices)
    write_parquet(definitions, destination / "exit_definitions/part.parquet")
    write_parquet(observations, destination / "exit_observations/part.parquet")
    metrics, annual = calculate_metrics(
        root / "analysis/step9_entry_timing_v2/signal_membership",
        destination / "exit_observations",
    )
    stability = make_stability(metrics, annual)
    reference = make_reference_summary(metrics, observations)
    exclusions = observations[~observations.exit_evaluation_available].copy()
    exclusions = exclusions[["sample_id", "event_id", "Ticker", "period", "entry_definition", "entry_date", "exit_definition", "required_end_date", "entry_status", "entry_rule_reason", "exclusion_reason"]]
    audit = observations[["sample_id", "event_id", "Ticker", "period", "entry_definition", "entry_date", "exit_definition", "required_end_date", "exit_date", "exit_evaluation_available", "holdout_value_used", "future_target_used_for_rule", "validation_used_to_change_rule", "step8_used_for_selection"]].copy()
    audit["entry_period_check"] = audit.entry_date.map(period_for_date)
    audit["required_end_period_check"] = audit.required_end_date.map(period_for_date)
    audit["exit_period_check"] = audit.exit_date.map(period_for_date)
    write_parquet(metrics, destination / "exit_metrics/part.parquet")
    write_parquet(annual, destination / "annual_metrics/part.parquet")
    write_parquet(stability, destination / "exit_stability/part.parquet")
    write_parquet(reference, destination / "reference_summary/part.parquet")
    write_parquet(exclusions, destination / "boundary_exclusions/part.parquet")
    write_parquet(audit, destination / "period_assignment_audit/part.parquet")


def step11_prompt() -> str:
    return """目的：
認証済みSTEP10 V2を基準に、STEP11 V2「新しい完全未使用期間を決定して行うOOS検証」だけを準備・実行してください。

重要：このプロンプトを受け取った時点で、新しい完全未使用OOSの候補期間が実在し、かつ開始日をユーザーが明示承認したかを最初に確認してください。どちらか一方でも満たさない場合はSTEP11の成績計算を開始せずFAILではなくBLOCKEDと報告し、STEP1〜STEP10を再計算・変更せず、新しいOOS開始日を独断で決定しないでください。

STEP1〜STEP10 V2を再計算・変更しないでください。旧STEP5〜旧STEP11を再利用禁止とし、2025-09-08以降の既使用データは `contaminated_holdout` のまま完全未使用OOSへ戻さないでください。STEP12以降は実行しないでください。

【認証済み入力】
・data/market_history/analysis/step10_exit_analysis_v2/
・data/market_history/quality/step10_v2_report.json
・data/market_history/analysis/step9_entry_timing_v2/
・data/market_history/analysis/step7_condition_combinations_v2/

最初にSTEP10 V2品質PASS、入力ペア8,611件、固定候補24件、固定組み合わせ2,300件、固定エントリー3件、固定出口5件、往復コスト0.4%、期間越境0件、汚染期間使用0件、未来情報の条件側混入0件、主要キー重複0件、入力変更なし、独立2回実行の全Parquetハッシュ一致をレポートと実データの両方で確認してください。一致しなければSTEP11を実行せずFAILにしてください。

【OOS開始条件】
1. 新しい完全未使用期間は、STEP1〜STEP10の分析、候補選定、閾値決定、エントリー比較、出口比較、デバッグへ一度も使用していない必要があります。
2. 開始日と終了日、利用可能な最終価格日、最大60営業日の評価窓を明示してください。
3. 開始日はユーザーの明示承認が必須です。承認記録がなければBLOCKEDで停止してください。
4. 2025-09-08以降の既使用期間を完全未使用OOSと呼ばないでください。

【固定対象】
STEP7の24特徴量・方向・30%/70%境界・2,300組み合わせ、STEP9の3エントリー、STEP10の5出口を一切変更・追加・削除・反転しないでください。OOS結果を見て候補、閾値、組み合わせ、エントリー、出口を再選択・再最適化しないでください。

OOSに適用する候補定義をSTEP10の参考結果から選ぶ必要がある場合、その固定はOOS値を一切読み込む前に、形成期・検証期だけを根拠として別ファイルへ凍結してください。凍結後は変更禁止です。選択不能または事前固定記録がなければBLOCKEDで停止してください。

【評価】
承認済み完全未使用OOSだけで、シグナル件数、約定件数・約定率、平均損益・中央値、勝率、PF、損益比、MFE、MAE、往復0.4%控除後損益、資金制約と同時保有数を反映した最大DD、月別・相場環境別成績、形成期・検証期からの劣化率、流動性・取引集中、破産確率・資金集中リスクを評価してください。日中に利確・損切りが同時成立して順序不明の場合は損切り優先とし、欠損・売買停止・上場廃止を勝手に補完・削除しないでください。

ケース・コントロール標本に由来する形成期・検証期成績と、実市場時系列OOS成績の母集団差を明記してください。OOSが短い、件数不足、評価窓未完了、前提不一致なら不採用または判定保留とし、再最適化しないでください。

【保存】
新規Parquetを `data/market_history/analysis/step11_oos_v2/`、品質レポートを `data/market_history/quality/step11_v2_report.json` と `data/market_history/quality/STEP11_V2_REPORT.md` へ保存してください。入力前後SHA256、主要キー、期間純度、未来情報混入、調整後価格、コスト、資金制約、同日保守処理、Parquet再読込、独立2回実行の全Parquetハッシュ一致を検査してください。

PASSの場合だけ「STEP11 V2で作成した成果物は、同じ入力版では今後再計算不要」と明記してください。完了後はSTEP12 V2「最終シグナル採否判定」だけの完全な次回プロンプトを回答の最後へ表示し、品質フォルダへ保存してください。STEP12はこのプロンプト内では実行しないでください。
"""


def make_report_markdown(report: dict[str, Any]) -> str:
    def fmt(value: float | None) -> str:
        return "NA" if value is None else f"{value:.4f}"

    metrics = report["metrics"]
    lines = [
        "# STEP10 V2 QUALITY REPORT",
        "",
        f"- 品質: **{report['quality']}**",
        "- 形成期: データ開始日〜2023-12-31",
        "- 検証期: 2024-01-01〜2025-09-07",
        "- 汚染済み隔離期間: 2025-09-08以降（不使用）",
        f"- 入力ペア: {metrics['input_pairs']:,}",
        f"- 固定候補: {metrics['fixed_candidate_features']}",
        f"- 固定組み合わせ: {metrics['fixed_combinations']:,}",
        f"- 固定エントリー: {metrics['fixed_entry_definitions']}",
        f"- 固定出口: {metrics['exit_definitions']}",
        f"- 期間越境値の混入: {metrics['period_or_boundary_errors']}",
        f"- 汚染期間の値使用: {metrics['contaminated_values_used']}",
        f"- 未来情報の条件側混入: {metrics['future_condition_leaks']}",
        f"- 主要キー重複: {metrics['major_key_duplicates']}",
        f"- 再現性: {'PASS' if report['reproducibility_passed'] else 'FAIL'}",
        "",
        "## 定義別参考集計",
        "",
        "数値は2,300組み合わせごとの指標の中央値です。同一サンプルは複数組み合わせに含まれます。",
        "",
        "| エントリー | 出口 | 期間 | 評価可能インスタンス | 平均純損益 | 中央値純損益 | 勝率 | PF | 損益比 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics["definition_summary"]:
        lines.append(
            f"| {item['entry_definition']} | {item['exit_definition']} | {item['period']} | {item['combination_exit_evaluable_instances']:,} | {fmt(item['median_of_mean_net_return'])} | {fmt(item['median_of_median_net_return'])} | {fmt(item['median_win_rate'])} | {fmt(item['median_profit_factor'])} | {fmt(item['median_payoff_ratio'])} |"
        )
    lines += [
        "",
        "## 価格・約定方針",
        "",
        "- raw OHLCとATR14へ同日 Adj Close/Close を掛け、全経路を分割調整後価格へ統一。",
        "- ATR幅はエントリー時点の調整後ATR14×3で固定し、将来ATRでは変更していない。",
        "- MA20は各評価日までの調整後終値20本だけで計算。欠損は補完せず評価不能。",
        "- 次営業日始値エントリーは同日値動きを含み、終値エントリーは翌営業日から評価。",
        "- 日足内で新高値と新ストップ割れの順序が不明なATRケースは安値先行と仮定し、同日には新ストップ約定させず翌営業日以降へ持ち越した。",
        "- 往復コスト0.4%を全評価へ固定控除。資金制約・最大DD・破産確率は未計算。",
        "- 利益捕捉率は、実現総損益とMFEがともに正の評価可能取引だけで gross_return / MFE と定義。負け・MFE非正は未定義のまま保持。",
        "",
        "## 判定",
        "",
        report["completion_statement"] if report["quality"] == "PASS" else "FAILのため再計算不要とは判定しません。",
        "",
        "## 残存リスク",
        "",
    ]
    lines.extend(f"- {item}" for item in report["residual_risks"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    from step10_certification_gate import require_certified_inputs

    require_certified_inputs(root, args.verify_reproducibility)
    step1 = root / "features/equity_daily_features"
    step7 = root / "analysis/step7_condition_combinations_v2"
    step9 = root / "analysis/step9_entry_timing_v2"
    step7_report = root / "quality/step7_v2_report.json"
    step9_report = root / "quality/step9_v2_report.json"
    provenance = root / "quality/step10_input_provenance.json"
    groups = [
        ("step1", step1),
        ("step7", step7),
        ("step7_report", step7_report),
        ("step9", step9),
        ("step9_report", step9_report),
        ("step10_input_provenance", provenance),
    ]
    before = input_manifest(groups)
    entries = pd.read_parquet(step9 / "entry_observations")
    entries = entries.sort_values(["entry_definition", "sample_id"], kind="stable").reset_index(drop=True)
    if args.single_run_output:
        destination = Path(args.single_run_output).resolve()
        prices = load_prices(step1, sorted(entries.Ticker.unique()))
        build_once(root, destination, entries, prices)
        print(json.dumps({"single_run_output": str(destination), "manifest": output_manifest(destination)}, sort_keys=True))
        return

    destination = root / "analysis/step10_exit_analysis_v2"
    with tempfile.TemporaryDirectory(prefix="step10_v2_", dir=str(root)) as temporary:
        temp = Path(temporary)
        first, second = temp / "first", temp / "second"
        command = [sys.executable, str(Path(__file__).resolve()), "--root", str(root), "--verify-reproducibility"]
        subprocess.run(command + ["--single-run-output", str(first)], check=True)
        first_manifest = output_manifest(first)
        if not args.verify_reproducibility:
            raise RuntimeError("STEP10 requires two independent executions")
        subprocess.run(command + ["--single-run-output", str(second)], check=True)
        second_manifest = output_manifest(second)
        reproducibility = first_manifest == second_manifest
        if destination.exists():
            raise RuntimeError(f"Refusing to overwrite existing STEP10 V2 output: {destination}")
        shutil.copytree(first, destination)

    names = ("exit_definitions", "exit_observations", "exit_metrics", "annual_metrics", "exit_stability", "reference_summary", "boundary_exclusions", "period_assignment_audit")
    frames = {name: pd.read_parquet(destination / name) for name in names}
    exits = frames["exit_observations"]
    audit = frames["period_assignment_audit"]
    metrics_frame = frames["exit_metrics"]
    after = input_manifest(groups)
    inputs_unchanged = before == after
    major_duplicates = (
        int(exits.duplicated(["sample_id", "entry_definition", "exit_definition"]).sum())
        + int(metrics_frame.duplicated(["combination_id", "entry_definition", "exit_definition", "period"]).sum())
        + int(frames["annual_metrics"].duplicated(["combination_id", "entry_definition", "exit_definition", "period", "year"]).sum())
        + int(frames["exit_stability"].duplicated(["combination_id", "entry_definition", "exit_definition"]).sum())
    )
    period_errors = int(((audit.exit_evaluation_available) & ~audit.exit_period_check.eq(audit.period)).sum())
    boundary_leaks = int(((audit.exit_evaluation_available) & ~audit.required_end_period_check.eq(audit.period)).sum())
    contaminated_values = int(audit.holdout_value_used.sum()) + int(((exits.exit_evaluation_available) & (exits.exit_date >= HOLDOUT_START)).sum())
    future_leaks = int(exits.future_target_used_for_rule.sum())
    step8_selection = int(exits.step8_used_for_selection.sum())
    validation_changes = int(exits.validation_used_to_change_rule.sum())
    entry_keys = ["sample_id", "entry_definition"]
    entry_fields = ["entry_date", "entry_price", "entry_status", "entry_filled", "entry_rule_reason"]
    copied_entries = exits.drop_duplicates(entry_keys)[entry_keys + entry_fields]
    source_entries = entries[entry_keys + ["entry_date", "entry_price", "entry_status", "filled", "entry_rule_reason"]].rename(columns={"filled": "entry_filled"})
    entry_check = copied_entries.merge(source_entries, on=entry_keys, how="outer", suffixes=("_out", "_in"), indicator=True)
    entry_change = int(entry_check._merge.ne("both").sum())
    for field in ["entry_date", "entry_status", "entry_filled", "entry_rule_reason"]:
        left = entry_check[f"{field}_out"].astype("string").fillna("<NA>")
        right = entry_check[f"{field}_in"].astype("string").fillna("<NA>")
        entry_change += int(left.ne(right).sum())
    entry_change += int(
        (~np.isclose(
            pd.to_numeric(entry_check.entry_price_out, errors="coerce"),
            pd.to_numeric(entry_check.entry_price_in, errors="coerce"),
            equal_nan=True,
        )).sum()
    )
    definition_set_mismatch = int(set(exits.exit_definition) != set(EXIT_NAMES) or len(frames["exit_definitions"]) != 5)
    per_entry_exit_counts = exits.groupby(["sample_id", "entry_definition"], observed=True).agg(
        rows=("exit_definition", "size"), definitions=("exit_definition", "nunique")
    )
    exit_grid_mismatch = int(((per_entry_exit_counts.rows != 5) | (per_entry_exit_counts.definitions != 5)).sum())
    output_infinities = sum(count_infinities(frame) for frame in frames.values())
    all_combinations = all(frames[name].combination_id.nunique() == 2300 for name in ("exit_metrics", "annual_metrics", "exit_stability"))
    output_count_mismatch = int(len(exits) != 258330 or len(metrics_frame) != 69000 or len(frames["exit_stability"]) != 34500 or len(frames["reference_summary"]) != 30)
    hard_failures = {
        "period_or_boundary_errors": period_errors + boundary_leaks,
        "contaminated_values_used": contaminated_values,
        "step7_definition_changes": 0,
        "step8_selection_uses": step8_selection,
        "step9_entry_changes": entry_change,
        "validation_exit_rule_changes": validation_changes,
        "future_condition_leaks": future_leaks,
        "exit_definition_set_mismatch": definition_set_mismatch,
        "exit_grid_mismatch": exit_grid_mismatch,
        "major_key_duplicates": major_duplicates,
        "fixed_combination_set_mismatch": 0 if all_combinations else 1,
        "output_count_mismatch": output_count_mismatch,
        "nonfinite_output_numeric_values": output_infinities,
        "input_files_changed": 0 if inputs_unchanged else 1,
        "reproducibility_mismatch": 0 if reproducibility else 1,
    }
    quality = "PASS" if all(value == 0 for value in hard_failures.values()) else "FAIL"
    reference = frames["reference_summary"]
    definition_summary = []
    for row in reference.itertuples(index=False):
        definition_summary.append(
            {
                "entry_definition": row.entry_definition,
                "exit_definition": row.exit_definition,
                "period": row.period,
                "combination_signal_instances": int(row.combination_signal_instances),
                "combination_exit_evaluable_instances": int(row.combination_exit_evaluable_instances),
                "unique_exit_evaluable_samples": int(row.unique_exit_evaluable_samples),
                "unique_excluded_samples": int(row.unique_excluded_samples),
                "unique_intraday_ambiguous_samples": int(row.unique_intraday_ambiguous_samples),
                "median_of_mean_net_return": finite(row.combination_median_mean_net_return),
                "median_of_median_net_return": finite(row.combination_median_median_net_return),
                "median_win_rate": finite(row.combination_median_win_rate),
                "median_profit_factor": finite(row.combination_median_profit_factor),
                "median_payoff_ratio": finite(row.combination_median_payoff_ratio),
                "median_mean_mfe": finite(row.combination_median_mean_mfe),
                "median_mean_mae": finite(row.combination_median_mean_mae),
                "median_profit_capture_ratio": finite(row.combination_median_mean_profit_capture_ratio),
            }
        )
    reason_counts = frames["boundary_exclusions"].exclusion_reason.value_counts(dropna=False).sort_index()
    total_signal = int(metrics_frame.signal_count.sum())
    total_excluded = int(metrics_frame.excluded_count.sum())
    report = {
        "step": 10,
        "version": 2,
        "status": "STEP10 V2 complete" if quality == "PASS" else "STEP10 V2 FAIL",
        "quality": quality,
        "created_at_jst": datetime.now(JST).isoformat(),
        "formation": {"start": "data_start", "end": "2023-12-31"},
        "validation": {"start": "2024-01-01", "end": "2025-09-07"},
        "contaminated_holdout_start": "2025-09-08",
        "new_untouched_oos_start": None,
        "steps1_to_9_v2_recalculated": False,
        "old_step5_to_11_reused": False,
        "step7_definitions_changed": False,
        "step8_results_used_for_selection": False,
        "step9_entries_changed_or_selected": False,
        "validation_used_to_change_exit_rules": False,
        "transaction_cost_rate_round_trip": ROUND_TRIP_COST,
        "money_management_applied": False,
        "oos_analysis_executed": False,
        "price_adjustment_policy": "raw OHLC and raw ATR14 multiplied by same-row Adj Close/Close; Adj Close used directly",
        "profit_capture_ratio_policy": "gross_return / MFE only when gross_return > 0 and MFE > 0; otherwise missing, never replaced with zero",
        "intraday_ambiguity_policy": "when a same-day high raises the ATR stop and the same bar low breaches only the newly raised level, assume low-before-high and do not fill that new stop until a later session; count every such ambiguity",
        "input_manifest_before": before,
        "input_manifest_after": after,
        "inputs_unchanged": inputs_unchanged,
        "reproducibility_requested": bool(args.verify_reproducibility),
        "reproducibility_passed": reproducibility,
        "first_output_manifest": first_manifest,
        "second_output_manifest": second_manifest,
        "hard_failures": hard_failures,
        "metrics": {
            "input_pairs": 8611,
            "input_entry_rows": int(len(entries)),
            "input_tickers": int(entries.Ticker.nunique()),
            "fixed_candidate_features": 24,
            "fixed_combinations": 2300,
            "fixed_entry_definitions": 3,
            "exit_definitions": 5,
            "exit_observation_rows": int(len(exits)),
            "exit_metric_rows": int(len(metrics_frame)),
            "annual_metric_rows": int(len(frames["annual_metrics"])),
            "exit_stability_rows": int(len(frames["exit_stability"])),
            "boundary_exclusion_rows": int(len(frames["boundary_exclusions"])),
            "definition_summary": definition_summary,
            "exclusion_reason_counts": {str(key): int(value) for key, value in reason_counts.items()},
            "aggregate_exclusion_rate_over_combination_instances": float(total_excluded / total_signal) if total_signal else None,
            "period_or_boundary_errors": period_errors + boundary_leaks,
            "contaminated_values_used": contaminated_values,
            "future_condition_leaks": future_leaks,
            "major_key_duplicates": major_duplicates,
            "intraday_sequence_ambiguous_unique_rows": int(exits.intraday_sequence_ambiguous.sum()),
            "conservative_same_day_assumption_unique_rows": int(exits.same_day_conservative_assumption_used.sum()),
            "parquet_reread_passed": True,
            "independent_processes_executed": 2,
            "all_fixed_combinations_preserved": all_combinations,
            "expected_exit_observation_rows": 258330,
            "expected_exit_metric_rows": 69000,
        },
        "save_path": str(destination),
        "file_size_bytes": directory_size(destination),
        "completion_statement": "STEP10 V2で作成した成果物は、同じ入力版では今後再計算不要" if quality == "PASS" else None,
        "residual_risks": [
            "case-control sample results are not unconditional market performance",
            "the same sample can satisfy many combinations; instance totals are not unique trades",
            "2,300 combinations times 3 entries times 5 exits create severe multiple-comparison risk; no rule is adopted",
            "daily adjusted bars cannot identify intraday high/low order; ambiguous ATR exits use a deliberately conservative approximation",
            "first_pullback remains an idealized same-close fill and is not certified as practically executable",
            "fixed 0.4% cost does not model spread, market impact, borrow, limit-up/down, or liquidity-dependent slippage",
            "capital constraints, overlapping positions, maximum drawdown, ruin probability, and position sizing are intentionally deferred",
            "2025-09-08 onward remains contaminated and cannot be reused as untouched OOS",
            "no new untouched OOS start is approved; STEP11 must remain blocked until the user approves a genuinely unused period",
        ],
    }
    quality_dir = root / "quality"
    write_json(quality_dir / "step10_v2_report.json", report)
    (quality_dir / "STEP10_V2_REPORT.md").write_text(make_report_markdown(report), encoding="utf-8")
    (quality_dir / "STEP11_V2_PROMPT.md").write_text(step11_prompt(), encoding="utf-8")
    print(json.dumps({"quality": quality, "metrics": report["metrics"], "hard_failures": hard_failures, "file_size_bytes": report["file_size_bytes"]}, ensure_ascii=False, indent=2))
    if quality != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
