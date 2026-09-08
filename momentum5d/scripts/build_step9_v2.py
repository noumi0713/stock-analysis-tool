# ruff: noqa: E501
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import shutil
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
MAIN_BUCKET = "d-5_to_d-1"
HORIZONS = (5, 10, 20, 40, 60)
ENTRY_NAMES = ("anchor_close", "next_session_open", "first_pullback")
FORBIDDEN = ("future", "forward", "target", "label", "mfe", "mae", "peak_date", "event_end", "outcome")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build isolated STEP9 V2 entry timing comparison")
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(groups: list[tuple[str, Path]]) -> dict[str, dict[str, Any]]:
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


def validate_step7(report: dict[str, Any]) -> None:
    if report.get("quality") != "PASS" or report.get("version") != 2 or not report.get("reproducibility_passed"):
        raise RuntimeError("STEP7 V2 certification is not PASS/reproducible")
    metrics = report.get("metrics", {})
    expected = {
        "input_pairs": 8611,
        "input_events": 8611,
        "input_controls": 8611,
        "input_tickers": 2637,
        "selected_candidate_features": 24,
        "total_combinations": 2300,
        "period_or_boundary_errors": 0,
        "contaminated_samples": 0,
        "definition_threshold_mismatches": 0,
        "major_key_duplicates": 0,
        "validation_used_for_candidate_selection": False,
        "validation_used_for_correlation": False,
        "validation_used_for_definition": False,
        "validation_thresholds_reestimated": False,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise RuntimeError(f"STEP7 V2 precondition mismatch: {key}")
    if metrics.get("forbidden_candidate_features") != [] or metrics.get("selected_strong_correlation_pairs") != 0:
        raise RuntimeError("STEP7 V2 forbidden/strongly-correlated candidate precondition mismatch")


def validate_step8(report: dict[str, Any]) -> None:
    if report.get("quality") != "PASS" or report.get("version") != 2 or not report.get("reproducibility_passed"):
        raise RuntimeError("STEP8 V2 certification is not PASS/reproducible")
    metrics = report.get("metrics", {})
    expected = {
        "input_pairs": 8611,
        "input_events": 8611,
        "input_controls": 8611,
        "input_tickers": 2637,
        "fixed_candidate_features": 24,
        "fixed_combinations": 2300,
        "environment_variables": 9,
        "period_or_boundary_errors": 0,
        "contaminated_samples": 0,
        "combination_definition_mismatches": 0,
        "combination_definition_field_mismatches": 0,
        "validation_regime_boundaries_reestimated": False,
        "major_key_duplicates": 0,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise RuntimeError(f"STEP8 V2 precondition mismatch: {key}")
    if metrics.get("forbidden_features_used") != []:
        raise RuntimeError("STEP8 V2 contains forbidden features")


def validate_step2(report: dict[str, Any]) -> None:
    expected = {
        "quality": "PASS",
        "step": 2,
        "step1_recalculated": False,
        "future_values_in_feature_columns": False,
        "target_columns_only": True,
        "calculation_basis": "per-ticker trading-row shift, not calendar-day shift",
        "insufficient_future_rows_imputed": False,
        "split_delisted_missing_rows_deleted": False,
        "label_only_schema": True,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(f"STEP2 precondition mismatch: {key}")


def period_for_date(value: pd.Timestamp) -> str:
    if pd.isna(value):
        return "missing"
    value = pd.Timestamp(value).normalize()
    if value <= FORMATION_END:
        return "formation"
    if VALIDATION_START <= value <= VALIDATION_END:
        return "validation"
    if value >= HOLDOUT_START:
        return "contaminated_holdout"
    return "outside_defined_period"


def period_end(period: str) -> pd.Timestamp:
    if period == "formation":
        return FORMATION_END
    if period == "validation":
        return VALIDATION_END
    raise ValueError(period)


def make_entry_definitions() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "entry_definition": "anchor_close",
            "entry_rule": "adjusted close on anchor_date",
            "entry_price_type": "split_adjusted_close",
            "search_start_sessions_after_anchor": 0,
            "search_end_sessions_after_anchor": 0,
            "unfilled_rule": "not_applicable_unless_certified_price_missing",
            "execution_assumption": "hypothetical adjusted-close fill; tradability requires positive reported volume",
            "fixed_before_validation": True,
            "validation_used_to_change_rule": False,
            "step8_used_to_select_rule": False,
        },
        {
            "entry_definition": "next_session_open",
            "entry_rule": "split-adjusted open on the next ticker trading row",
            "entry_price_type": "raw_open_times_same_row_adj_close_over_raw_close",
            "search_start_sessions_after_anchor": 1,
            "search_end_sessions_after_anchor": 1,
            "unfilled_rule": "unfilled if next ticker trading row or valid adjusted open is unavailable inside the same period",
            "execution_assumption": "hypothetical adjusted-open fill; tradability requires positive reported volume",
            "fixed_before_validation": True,
            "validation_used_to_change_rule": False,
            "step8_used_to_select_rule": False,
        },
        {
            "entry_definition": "first_pullback",
            "entry_rule": "first adjusted close <= prior adjusted close on ticker rows anchor+1 through anchor+5",
            "entry_price_type": "split_adjusted_close",
            "search_start_sessions_after_anchor": 1,
            "search_end_sessions_after_anchor": 5,
            "unfilled_rule": "unfilled only if all five rows are observable and no qualifying close occurs; missing/boundary/tradability uncertainty is indeterminate",
            "execution_assumption": "idealized same-close fill after observing the closing condition; not certified as practically executable",
            "fixed_before_validation": True,
            "validation_used_to_change_rule": False,
            "step8_used_to_select_rule": False,
        },
    ])


def definition_conditions(row: Any) -> list[tuple[str, str, float]]:
    result: list[tuple[str, str, float]] = []
    for index in range(1, int(row.condition_count) + 1):
        result.append((str(getattr(row, f"feature_{index}")), str(getattr(row, f"direction_{index}")), float(getattr(row, f"threshold_{index}"))))
    return result


def condition_masks(samples: pd.DataFrame, definitions: pd.DataFrame) -> dict[str, np.ndarray]:
    features = sorted(set(definitions.feature_1) | set(definitions.feature_2) | set(definitions.feature_3.dropna()))
    cache = {feature: pd.to_numeric(samples[feature], errors="coerce").to_numpy(dtype=float) for feature in features}
    result: dict[str, np.ndarray] = {}
    for row in definitions.itertuples(index=False):
        mask = np.ones(len(samples), dtype=bool)
        for feature, direction, threshold in definition_conditions(row):
            values = cache[feature]
            valid = np.isfinite(values)
            selected = values >= threshold if direction == "high" else values <= threshold
            mask &= valid & selected
        result[str(row.combination_id)] = mask
    return result


def load_prices(path: Path, tickers: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Load the full date/index skeleton but never load holdout OHLCV values."""
    dataset = ds.dataset(path, format="parquet")
    required = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = sorted(set(required) - set(dataset.schema.names))
    if missing:
        raise RuntimeError(f"STEP1 price columns missing: {missing}")
    ticker_filter = ds.field("Ticker").isin(tickers)
    index_frame = dataset.to_table(columns=["Date", "Ticker"], filter=ticker_filter).to_pandas()
    value_filter = ticker_filter & (ds.field("Date") < HOLDOUT_START.to_pydatetime())
    values = dataset.to_table(columns=required, filter=value_filter).to_pandas()
    frame = index_frame.merge(values, on=["Date", "Ticker"], how="left", validate="one_to_one")
    frame["Date"] = pd.to_datetime(frame.Date).dt.normalize()
    frame = frame.sort_values(["Ticker", "Date"], kind="stable").reset_index(drop=True)
    if frame.duplicated(["Ticker", "Date"]).any():
        raise RuntimeError("STEP1 contains duplicate Ticker x Date price rows")
    for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    factor = frame["Adj Close"] / frame.Close
    valid_factor = np.isfinite(factor) & factor.between(0.001, 1000, inclusive="both")
    frame["adj_open"] = np.where(valid_factor, frame.Open * factor, np.nan)
    frame["adj_high"] = np.where(valid_factor, frame.High * factor, np.nan)
    frame["adj_low"] = np.where(valid_factor, frame.Low * factor, np.nan)
    result: dict[str, dict[str, np.ndarray]] = {}
    for ticker, group in frame.groupby("Ticker", sort=True, observed=True):
        result[str(ticker)] = {
            "date": group.Date.to_numpy(dtype="datetime64[ns]"),
            "adj_open": group.adj_open.to_numpy(dtype=float),
            "adj_high": group.adj_high.to_numpy(dtype=float),
            "adj_low": group.adj_low.to_numpy(dtype=float),
            "adj_close": group["Adj Close"].to_numpy(dtype=float),
            "volume": group.Volume.to_numpy(dtype=float),
            "price_values_loaded": group.Date.lt(HOLDOUT_START).to_numpy(dtype=bool),
        }
    return result


def positive_finite(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0)


def make_entry_observations(samples: pd.DataFrame, prices: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_pos, sample in enumerate(samples.itertuples(index=False)):
        ticker = str(sample.Ticker)
        anchor = pd.Timestamp(sample.anchor_date).normalize()
        series = prices.get(ticker)
        anchor_pos = -1
        if series is not None:
            candidates = np.flatnonzero(series["date"] == np.datetime64(anchor))
            if len(candidates) == 1:
                anchor_pos = int(candidates[0])
        common = {
            "sample_pos": sample_pos,
            "sample_id": str(sample.sample_id),
            "event_id": str(sample.event_id),
            "Ticker": ticker,
            "period": str(sample.period),
            "observation_type": str(sample.observation_type),
            "outcome": int(sample.outcome),
            "anchor_date": anchor,
            "anchor_year": int(sample.anchor_year),
            "anchor_period_check": period_for_date(anchor),
        }
        for entry_name in ENTRY_NAMES:
            row = dict(common)
            row.update({
                "entry_definition": entry_name,
                "entry_date": pd.NaT,
                "entry_session_position": pd.NA,
                "entry_price": np.nan,
                "entry_adj_close": np.nan,
                "entry_adj_high": np.nan,
                "entry_adj_low": np.nan,
                "price_reference_available": False,
                "volume": np.nan,
                "tradability_confirmed": False,
                "entry_status": "indeterminate",
                "hypothetical_filled": False,
                "filled": False,
                "entry_rule_reason": None,
                "execution_realism": "hypothetical_only",
                "entry_period_check": "missing",
                "entry_rule_uses_target_labels": False,
                "step8_used_for_selection": False,
                "validation_used_to_change_rule": False,
            })
            if series is None or anchor_pos < 0:
                row["entry_rule_reason"] = "anchor_price_row_missing"
                rows.append(row)
                continue
            dates = series["date"]
            adj_close = series["adj_close"]
            if entry_name == "anchor_close":
                entry_pos = anchor_pos
            elif entry_name == "next_session_open":
                entry_pos = anchor_pos + 1
                if entry_pos >= len(dates):
                    row["entry_rule_reason"] = "next_ticker_trading_row_missing"
                    rows.append(row)
                    continue
                next_date = pd.Timestamp(dates[entry_pos])
                if period_for_date(next_date) != str(sample.period):
                    row["entry_rule_reason"] = "next_session_crosses_period_boundary"
                    rows.append(row)
                    continue
            else:
                entry_pos = -1
                search_missing = False
                search_boundary = False
                for offset in range(1, 6):
                    pos = anchor_pos + offset
                    if pos >= len(dates):
                        search_missing = True
                        break
                    candidate_date = pd.Timestamp(dates[pos])
                    if period_for_date(candidate_date) != str(sample.period):
                        search_boundary = True
                        break
                    current = adj_close[pos]
                    prior = adj_close[pos - 1]
                    if not positive_finite(current) or not positive_finite(prior):
                        search_missing = True
                        break
                    if current <= prior:
                        entry_pos = pos
                        break
                if entry_pos < 0:
                    if search_missing:
                        row["entry_rule_reason"] = "entry_search_source_missing"
                        row["entry_status"] = "indeterminate"
                    elif search_boundary:
                        row["entry_rule_reason"] = "entry_search_window_crosses_period_boundary"
                        row["entry_status"] = "indeterminate"
                    else:
                        row["entry_rule_reason"] = "no_pullback_within_5_sessions"
                        row["entry_status"] = "unfilled"
                    rows.append(row)
                    continue
            entry_date = pd.Timestamp(dates[entry_pos]).normalize()
            if period_for_date(entry_date) != str(sample.period):
                row["entry_rule_reason"] = "entry_date_period_mismatch"
                rows.append(row)
                continue
            entry_price = series["adj_open"][entry_pos] if entry_name == "next_session_open" else adj_close[entry_pos]
            if not positive_finite(entry_price):
                row["entry_rule_reason"] = "entry_price_missing_or_invalid"
                rows.append(row)
                continue
            volume = finite(series["volume"][entry_pos])
            row.update({
                "entry_date": entry_date,
                "entry_session_position": entry_pos,
                "entry_price": float(entry_price),
                "entry_adj_close": float(adj_close[entry_pos]) if positive_finite(adj_close[entry_pos]) else np.nan,
                "entry_adj_high": float(series["adj_high"][entry_pos]) if positive_finite(series["adj_high"][entry_pos]) else np.nan,
                "entry_adj_low": float(series["adj_low"][entry_pos]) if positive_finite(series["adj_low"][entry_pos]) else np.nan,
                "price_reference_available": True,
                "volume": volume,
                "entry_period_check": period_for_date(entry_date),
            })
            if volume is None or volume <= 0:
                row["entry_status"] = "indeterminate"
                row["entry_rule_reason"] = "entry_tradability_unconfirmed_zero_or_missing_volume"
                rows.append(row)
                continue
            row.update({
                "tradability_confirmed": True,
                "entry_status": "hypothetical_fill",
                "hypothetical_filled": True,
                "filled": True,
                "entry_rule_reason": "idealized_same_close_fill" if entry_name == "first_pullback" else "hypothetical_fill",
                "execution_realism": "idealized_same_close_after_condition_confirmation" if entry_name == "first_pullback" else "hypothetical_daily_bar_fill",
            })
            rows.append(row)
    result = pd.DataFrame(rows)
    result["entry_session_position"] = result.entry_session_position.astype("Int64")
    return result.sort_values(["entry_definition", "sample_pos"], kind="stable").reset_index(drop=True)


def target_glob(path: Path) -> str:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise RuntimeError(f"No STEP2 target Parquet found under {path}")
    if len(files) == 1:
        return str(files[0])
    return str(path / "**" / "*.parquet")


def attach_targets(entries: pd.DataFrame, prices: dict[str, dict[str, np.ndarray]], target_path: Path) -> pd.DataFrame:
    result = entries.copy()
    for horizon in HORIZONS:
        result[f"target_end_date_{horizon}d"] = pd.NaT
        result[f"forward_return_{horizon}d"] = np.nan
        result[f"mfe_{horizon}d"] = np.nan
        result[f"mae_{horizon}d"] = np.nan
        result[f"evaluation_available_{horizon}d"] = False
        result[f"evaluation_exclusion_reason_{horizon}d"] = None

    for idx, row in result.iterrows():
        if not bool(row.filled):
            for horizon in HORIZONS:
                result.at[idx, f"evaluation_exclusion_reason_{horizon}d"] = row.entry_rule_reason
            continue
        series = prices[str(row.Ticker)]
        entry_pos = int(row.entry_session_position)
        for horizon in HORIZONS:
            target_pos = entry_pos + horizon
            if target_pos >= len(series["date"]):
                result.at[idx, f"evaluation_exclusion_reason_{horizon}d"] = "insufficient_future_ticker_trading_rows"
                continue
            target_date = pd.Timestamp(series["date"][target_pos]).normalize()
            result.at[idx, f"target_end_date_{horizon}d"] = target_date
            if period_for_date(target_date) != str(row.period):
                reason = "evaluation_window_crosses_contaminated_holdout" if target_date >= HOLDOUT_START else "evaluation_window_crosses_period_boundary"
                result.at[idx, f"evaluation_exclusion_reason_{horizon}d"] = reason

    con = duckdb.connect()
    parquet = target_glob(target_path).replace("'", "''")
    try:
        for horizon in HORIZONS:
            eligible = result[
                result.filled
                & result[f"target_end_date_{horizon}d"].notna()
                & result[f"evaluation_exclusion_reason_{horizon}d"].isna()
            ][["Ticker", "entry_date"]].copy()
            eligible["obs_index"] = eligible.index.astype("int64")
            eligible = eligible.rename(columns={"entry_date": "Date"})
            con.register("eligible_keys", eligible)
            queried = con.execute(f"""
                SELECT k.obs_index,
                       t.forward_return_{horizon}d AS source_return,
                       t.mfe_{horizon}d AS source_mfe,
                       t.mae_{horizon}d AS source_mae
                FROM eligible_keys k
                LEFT JOIN read_parquet('{parquet}') t
                  ON k.Ticker = t.Ticker AND CAST(k.Date AS DATE) = t.Date
                ORDER BY k.obs_index
            """).df()
            for value in queried.itertuples(index=False):
                idx = int(value.obs_index)
                source = (finite(value.source_return), finite(value.source_mfe), finite(value.source_mae))
                if any(item is None for item in source):
                    result.at[idx, f"evaluation_exclusion_reason_{horizon}d"] = "step2_target_triplet_missing_or_invalid"
                    continue
                source_return, source_mfe, source_mae = source
                if result.at[idx, "entry_definition"] == "next_session_open":
                    entry_price = float(result.at[idx, "entry_price"])
                    entry_close = finite(result.at[idx, "entry_adj_close"])
                    same_high = finite(result.at[idx, "entry_adj_high"])
                    same_low = finite(result.at[idx, "entry_adj_low"])
                    if entry_close is None or same_high is None or same_low is None:
                        result.at[idx, f"evaluation_exclusion_reason_{horizon}d"] = "adjusted_entry_session_ohlc_missing_or_invalid"
                        continue
                    target_close = entry_close * (1.0 + source_return)
                    future_high = entry_close * (1.0 + source_mfe)
                    future_low = entry_close * (1.0 + source_mae)
                    output = (
                        target_close / entry_price - 1.0,
                        max(same_high, future_high) / entry_price - 1.0,
                        min(same_low, future_low) / entry_price - 1.0,
                    )
                else:
                    output = (source_return, source_mfe, source_mae)
                if not all(math.isfinite(float(item)) for item in output):
                    result.at[idx, f"evaluation_exclusion_reason_{horizon}d"] = "nonfinite_derived_evaluation"
                    continue
                result.at[idx, f"forward_return_{horizon}d"] = float(output[0])
                result.at[idx, f"mfe_{horizon}d"] = float(output[1])
                result.at[idx, f"mae_{horizon}d"] = float(output[2])
                result.at[idx, f"evaluation_available_{horizon}d"] = True
    finally:
        con.close()
    return result


def json_counts(series: pd.Series) -> str:
    counts = series.fillna("available").value_counts(dropna=False).sort_index()
    return json.dumps({str(key): int(value) for key, value in counts.items()}, ensure_ascii=False, sort_keys=True)


def summarize_returns(values: np.ndarray) -> dict[str, Any]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean_return": None, "median_return": None, "win_rate": None}
    return {
        "mean_return": float(np.mean(values)),
        "median_return": float(np.median(values)),
        "win_rate": float(np.mean(values > 0)),
    }


def build_metrics(
    samples: pd.DataFrame,
    entries: pd.DataFrame,
    definitions: pd.DataFrame,
    step7_metrics: pd.DataFrame,
    selected_masks: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered_entries = {
        name: entries[entries.entry_definition.eq(name)].sort_values("sample_pos", kind="stable").reset_index(drop=True)
        for name in ENTRY_NAMES
    }
    period_arrays = {period: samples.period.eq(period).to_numpy() for period in ("formation", "validation")}
    step7_lookup = step7_metrics.set_index(["combination_id", "period"])["condition_rows"].to_dict()
    metric_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    for definition in definitions.itertuples(index=False):
        combination_id = str(definition.combination_id)
        signal = selected_masks[combination_id]
        for period, period_mask in period_arrays.items():
            signal_period = signal & period_mask
            signal_indices = np.flatnonzero(signal_period)
            if len(signal_indices) != int(step7_lookup[(combination_id, period)]):
                raise RuntimeError(f"STEP7 signal count mismatch: {combination_id}/{period}")
            for entry_name in ENTRY_NAMES:
                frame = ordered_entries[entry_name]
                selected = frame.iloc[signal_indices]
                for horizon in HORIZONS:
                    return_col = f"forward_return_{horizon}d"
                    mfe_col = f"mfe_{horizon}d"
                    mae_col = f"mae_{horizon}d"
                    available_col = f"evaluation_available_{horizon}d"
                    reason_col = f"evaluation_exclusion_reason_{horizon}d"
                    valid = selected[available_col].to_numpy(dtype=bool)
                    valid_rows = selected.loc[valid]
                    determinate = selected.entry_status.ne("indeterminate")
                    fill_denominator = int(determinate.sum())
                    stats = summarize_returns(valid_rows[return_col].to_numpy(dtype=float))
                    metric_rows.append({
                        "combination_id": combination_id,
                        "condition_count": int(definition.condition_count),
                        "entry_definition": entry_name,
                        "period": period,
                        "horizon": horizon,
                        "signal_count": int(len(selected)),
                        "signal_events": int(selected.outcome.eq(1).sum()),
                        "signal_controls": int(selected.outcome.eq(0).sum()),
                        "price_reference_available": int(selected.price_reference_available.sum()),
                        "filled_count": int(selected.filled.sum()),
                        "hypothetical_fill_count": int(selected.hypothetical_filled.sum()),
                        "unfilled_count": int(selected.entry_status.eq("unfilled").sum()),
                        "indeterminate_count": int(selected.entry_status.eq("indeterminate").sum()),
                        "fill_rate_denominator_count": fill_denominator,
                        "fill_rate": float(selected.hypothetical_filled.sum() / fill_denominator) if fill_denominator else None,
                        "total_signal_based_fill_rate": float(selected.hypothetical_filled.mean()) if len(selected) else None,
                        "evaluation_available_count": int(valid.sum()),
                        "evaluation_missing_count": int((~valid).sum()),
                        "evaluation_missing_rate": float((~valid).mean()) if len(selected) else None,
                        **stats,
                        "mean_mfe": float(valid_rows[mfe_col].mean()) if len(valid_rows) else None,
                        "median_mfe": float(valid_rows[mfe_col].median()) if len(valid_rows) else None,
                        "mean_mae": float(valid_rows[mae_col].mean()) if len(valid_rows) else None,
                        "median_mae": float(valid_rows[mae_col].median()) if len(valid_rows) else None,
                        "exclusion_reasons_json": json_counts(selected[reason_col]),
                        "formation_to_validation_mean_return_retention": None,
                        "formation_to_validation_median_return_retention": None,
                        "formation_to_validation_win_rate_retention": None,
                        "retention_unavailable_reason": None if period == "formation" else "pending_formation_reference",
                        "definition_fixed_before_validation": True,
                        "step8_used_for_selection": False,
                    })
                    years = sorted(pd.Series(selected.loc[valid, "entry_date"]).dt.year.dropna().astype(int).unique())
                    for year in years:
                        year_rows = valid_rows[pd.to_datetime(valid_rows.entry_date).dt.year.eq(year)]
                        annual_stats = summarize_returns(year_rows[return_col].to_numpy(dtype=float))
                        annual_rows.append({
                            "combination_id": combination_id,
                            "condition_count": int(definition.condition_count),
                            "entry_definition": entry_name,
                            "horizon": horizon,
                            "period": period,
                            "year": int(year),
                            "signal_count": int(len(selected[pd.to_datetime(selected.anchor_date).dt.year.eq(year)])),
                            "hypothetical_fill_count": int(selected[pd.to_datetime(selected.entry_date).dt.year.eq(year)].hypothetical_filled.sum()),
                            "unfilled_count": int(selected[pd.to_datetime(selected.anchor_date).dt.year.eq(year)].entry_status.eq("unfilled").sum()),
                            "indeterminate_count": int(selected[pd.to_datetime(selected.anchor_date).dt.year.eq(year)].entry_status.eq("indeterminate").sum()),
                            "evaluation_available_count": int(len(year_rows)),
                            **annual_stats,
                            "formation_direction": None,
                            "annual_direction_reproduced": None,
                        })
    metrics = pd.DataFrame(metric_rows)
    key = ["combination_id", "entry_definition", "horizon"]
    formation = metrics[metrics.period.eq("formation")].set_index(key)
    for idx in metrics[metrics.period.eq("validation")].index:
        row = metrics.loc[idx]
        base = formation.loc[(row.combination_id, row.entry_definition, row.horizon)]
        reasons: list[str] = []
        for source, target in [
            ("mean_return", "formation_to_validation_mean_return_retention"),
            ("median_return", "formation_to_validation_median_return_retention"),
            ("win_rate", "formation_to_validation_win_rate_retention"),
        ]:
            denominator = finite(base[source])
            numerator = finite(row[source])
            if denominator is None or numerator is None:
                reasons.append(f"{source}_not_evaluable")
            elif denominator == 0:
                reasons.append(f"formation_{source}_zero")
            else:
                metrics.at[idx, target] = numerator / denominator
        metrics.at[idx, "retention_unavailable_reason"] = ";".join(reasons) if reasons else None
    annual = pd.DataFrame(annual_rows)
    formation_direction = metrics[metrics.period.eq("formation")].set_index(key).mean_return.map(
        lambda value: None if finite(value) is None else ("positive" if float(value) >= 0 else "negative")
    ).to_dict()
    for idx, row in annual.iterrows():
        direction = formation_direction[(row.combination_id, row.entry_definition, row.horizon)]
        annual.at[idx, "formation_direction"] = direction
        value = finite(row.mean_return)
        annual.at[idx, "annual_direction_reproduced"] = None if direction is None or value is None else bool((value >= 0) == (direction == "positive"))
    return (
        metrics.sort_values(["combination_id", "entry_definition", "period", "horizon"], kind="stable").reset_index(drop=True),
        annual.sort_values(["combination_id", "entry_definition", "horizon", "period", "year"], kind="stable").reset_index(drop=True),
    )


def make_stability(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (combination_id, period, horizon), group in metrics.groupby(["combination_id", "period", "horizon"], sort=True, observed=True):
        lookup = group.set_index("entry_definition")
        values = {name: finite(lookup.loc[name, "mean_return"]) for name in ENTRY_NAMES}
        valid = [value for value in values.values() if value is not None]
        rows.append({
            "combination_id": combination_id,
            "period": period,
            "horizon": int(horizon),
            "anchor_close_mean_return": values["anchor_close"],
            "next_session_open_mean_return": values["next_session_open"],
            "first_pullback_mean_return": values["first_pullback"],
            "anchor_minus_next_open": None if values["anchor_close"] is None or values["next_session_open"] is None else values["anchor_close"] - values["next_session_open"],
            "anchor_minus_first_pullback": None if values["anchor_close"] is None or values["first_pullback"] is None else values["anchor_close"] - values["first_pullback"],
            "next_open_minus_first_pullback": None if values["next_session_open"] is None or values["first_pullback"] is None else values["next_session_open"] - values["first_pullback"],
            "timing_mean_return_range": max(valid) - min(valid) if valid else None,
            "descriptive_only": True,
            "entry_timing_adopted": False,
        })
    return pd.DataFrame(rows)


def make_reference_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (entry_name, period, horizon), group in metrics.groupby(["entry_definition", "period", "horizon"], sort=True, observed=True):
        rows.append({
            "entry_definition": entry_name,
            "period": period,
            "horizon": int(horizon),
            "combination_count": int(group.combination_id.nunique()),
            "combination_signal_instances": int(group.signal_count.sum()),
            "median_signal_count_per_combination": float(group.signal_count.median()),
            "combination_filled_instances": int(group.filled_count.sum()),
            "combination_unfilled_instances": int(group.unfilled_count.sum()),
            "combination_indeterminate_instances": int(group.indeterminate_count.sum()),
            "fill_rate_denominator_instances": int(group.fill_rate_denominator_count.sum()),
            "weighted_fill_rate": float(group.hypothetical_fill_count.sum() / group.fill_rate_denominator_count.sum()) if group.fill_rate_denominator_count.sum() else None,
            "total_signal_based_fill_rate": float(group.hypothetical_fill_count.sum() / group.signal_count.sum()) if group.signal_count.sum() else None,
            "median_combination_mean_return": float(group.mean_return.median()) if group.mean_return.notna().any() else None,
            "median_combination_median_return": float(group.median_return.median()) if group.median_return.notna().any() else None,
            "median_combination_win_rate": float(group.win_rate.median()) if group.win_rate.notna().any() else None,
            "case_control_rate_is_unconditional_market_probability": False,
            "descriptive_only": True,
            "entry_timing_adopted": False,
        })
    return pd.DataFrame(rows)


def make_boundary_exclusions(entries: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base = ["sample_id", "event_id", "Ticker", "period", "observation_type", "anchor_date", "entry_definition", "entry_date"]
    for horizon in HORIZONS:
        reason = f"evaluation_exclusion_reason_{horizon}d"
        frame = entries[entries[reason].notna()][base + [f"target_end_date_{horizon}d", reason]].copy()
        frame["horizon"] = horizon
        frame = frame.rename(columns={f"target_end_date_{horizon}d": "target_end_date", reason: "exclusion_reason"})
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(["entry_definition", "sample_id", "horizon"], kind="stable").reset_index(drop=True)


def make_period_audit(entries: pd.DataFrame) -> pd.DataFrame:
    columns = ["sample_id", "event_id", "Ticker", "period", "observation_type", "anchor_date", "anchor_period_check", "entry_definition", "entry_date", "entry_period_check", "price_reference_available", "tradability_confirmed", "entry_status", "filled"]
    result = entries[columns].copy()
    for horizon in HORIZONS:
        result[f"target_end_date_{horizon}d"] = entries[f"target_end_date_{horizon}d"]
        result[f"evaluation_available_{horizon}d"] = entries[f"evaluation_available_{horizon}d"]
        result[f"period_check_pass_{horizon}d"] = (~entries[f"evaluation_available_{horizon}d"]) | entries[f"target_end_date_{horizon}d"].map(period_for_date).eq(entries.period)
    result["contaminated_value_used"] = False
    return result.sort_values(["entry_definition", "sample_id"], kind="stable").reset_index(drop=True)


def make_signal_membership(
    samples: pd.DataFrame,
    definitions: pd.DataFrame,
    selected_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    identity = samples[["sample_id", "event_id", "Ticker", "period", "observation_type", "outcome", "anchor_date"]]
    parts: list[pd.DataFrame] = []
    counts = definitions.set_index("combination_id").condition_count.to_dict()
    for combination_id in sorted(selected_masks):
        selected = identity.loc[selected_masks[combination_id]].copy()
        selected.insert(0, "combination_id", combination_id)
        selected.insert(1, "condition_count", int(counts[combination_id]))
        parts.append(selected)
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(["combination_id", "sample_id"]).any():
        raise RuntimeError("signal_membership duplicate combination_id x sample_id")
    return result.sort_values(["combination_id", "sample_id"], kind="stable").reset_index(drop=True)


def make_paired_timing_comparison(
    samples: pd.DataFrame,
    entries: pd.DataFrame,
    definitions: pd.DataFrame,
    selected_masks: dict[str, np.ndarray],
) -> pd.DataFrame:
    ordered = {
        name: entries[entries.entry_definition.eq(name)].sort_values("sample_pos", kind="stable").reset_index(drop=True)
        for name in ENTRY_NAMES
    }
    period_masks = {period: samples.period.eq(period).to_numpy() for period in ("formation", "validation")}
    rows: list[dict[str, Any]] = []
    for definition in definitions.itertuples(index=False):
        combination_id = str(definition.combination_id)
        signal = selected_masks[combination_id]
        for period, period_mask in period_masks.items():
            base = signal & period_mask
            for horizon in HORIZONS:
                available = base.copy()
                for entry_name in ENTRY_NAMES:
                    available &= ordered[entry_name][f"evaluation_available_{horizon}d"].to_numpy(dtype=bool)
                indexes = np.flatnonzero(available)
                values = {
                    entry_name: ordered[entry_name].iloc[indexes][f"forward_return_{horizon}d"].to_numpy(dtype=float)
                    for entry_name in ENTRY_NAMES
                }
                row: dict[str, Any] = {
                    "combination_id": combination_id,
                    "condition_count": int(definition.condition_count),
                    "period": period,
                    "horizon": horizon,
                    "signal_count": int(base.sum()),
                    "paired_sample_count": int(len(indexes)),
                    "paired_events": int(samples.iloc[indexes].outcome.eq(1).sum()),
                    "paired_controls": int(samples.iloc[indexes].outcome.eq(0).sum()),
                    "comparison_scope": "same combination-sample instances evaluable for all three entry definitions",
                    "descriptive_only": True,
                }
                for entry_name in ENTRY_NAMES:
                    stats = summarize_returns(values[entry_name])
                    row[f"{entry_name}_mean_return"] = stats["mean_return"]
                    row[f"{entry_name}_median_return"] = stats["median_return"]
                    row[f"{entry_name}_win_rate"] = stats["win_rate"]
                for left, right, label in (
                    ("anchor_close", "next_session_open", "anchor_minus_next_open"),
                    ("anchor_close", "first_pullback", "anchor_minus_first_pullback"),
                    ("next_session_open", "first_pullback", "next_open_minus_first_pullback"),
                ):
                    diff = values[left] - values[right]
                    row[f"{label}_mean"] = float(np.mean(diff)) if len(diff) else None
                    row[f"{label}_median"] = float(np.median(diff)) if len(diff) else None
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["combination_id", "period", "horizon"], kind="stable").reset_index(drop=True)


def count_infinities(frame: pd.DataFrame) -> int:
    total = 0
    for column in frame.select_dtypes(include=[np.number]).columns:
        total += int(np.isinf(pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)).sum())
    return total


def build_once(
    root: Path,
    output: Path,
    samples: pd.DataFrame,
    definitions: pd.DataFrame,
    step7_metrics: pd.DataFrame,
    prices: dict[str, dict[str, np.ndarray]],
    target_path: Path,
) -> dict[str, pd.DataFrame]:
    selected_masks = condition_masks(samples, definitions)
    entries = attach_targets(make_entry_observations(samples, prices), prices, target_path)
    metrics, annual = build_metrics(samples, entries, definitions, step7_metrics, selected_masks)
    frames = {
        "entry_definitions": make_entry_definitions(),
        "signal_membership": make_signal_membership(samples, definitions, selected_masks),
        "entry_observations": entries.drop(columns=["sample_pos"]),
        "entry_timing_metrics": metrics,
        "paired_timing_comparison": make_paired_timing_comparison(samples, entries, definitions, selected_masks),
        "annual_metrics": annual,
        "timing_stability": make_stability(metrics),
        "reference_summary": make_reference_summary(metrics),
        "boundary_exclusions": make_boundary_exclusions(entries),
        "period_assignment_audit": make_period_audit(entries),
    }
    for name, frame in frames.items():
        write_parquet(frame, output / name / "part.parquet")
    return frames


def step10_prompt(metrics: dict[str, Any]) -> str:
    return """目的：
認証済みSTEP9 V2を基準に、STEP10 V2「利益を伸ばす出口検証」だけを実行してください。

STEP1〜STEP9 V2を再計算・変更しないでください。旧STEP5〜旧STEP11の成果物は再利用禁止です。STEP7 V2の固定候補24特徴量、方向、30%・70%分位境界、2条件・3条件AND定義2,300件を変更・追加・削除・反転しないでください。STEP8 V2の相場環境別結果で組み合わせを選別しないでください。STEP9 V2の3エントリー定義を変更せず、検証期の結果を見てエントリーを選別しないでください。

今回は固定済み2,300組み合わせ×3エントリー定義について、事前固定した出口の差だけを比較します。最終シグナル採用、資金管理、ポジションサイズ、同時保有制御、OOS検証は行わず、STEP11以降を実行しないでください。

【認証済み入力】
・data/market_history/analysis/step9_entry_timing_v2/
・data/market_history/quality/step9_v2_report.json
・data/market_history/analysis/step7_condition_combinations_v2/
・data/market_history/quality/step7_v2_report.json

価格経路、20日移動平均、ATRの参照に限り、STEP1認証済み特徴量マスタを読み取り専用で使用できます。
・data/market_history/features/equity_daily_features/

最初にSTEP9 V2品質PASS、入力ペア8,611件、固定候補24件、固定組み合わせ2,300件、比較エントリー3件、期間越境0件、汚染期間使用0件、未来目的変数の条件側混入0件、主要キー重複0件、入力変更なし、再現性PASSを確認してください。一致しなければSTEP10 V2を実行せずFAILにしてください。

【期間】
・形成期：データ開始日〜2023-12-31
・検証期：2024-01-01〜2025-09-07
・汚染済み隔離期間：2025-09-08以降

2025-09-08以降は `contaminated_holdout` として、出口定義、価格参照、閾値計算、分析、集計、比較、順位、可視化へ使用しないでください。出口評価窓が期間境界または2025-09-08以降へ越えるサンプルは、値を読まず理由を保存して除外してください。新しい完全未使用OOS開始日を独断で決定しないでください。

【固定エントリー】
STEP9 V2の以下をそのまま使用してください。
・anchor_close
・next_session_open
・first_pullback

未約定サンプルを約定扱いにせず、エントリー日・調整後エントリー価格・欠損理由をSTEP9 V2から引き継いでください。検証期結果によるエントリー選別は禁止します。

【出口定義】
形成期を見る前から次の5規則を固定し、全組み合わせ・全エントリーへ機械的に適用してください。

1. time_20_close：エントリー20営業日後の調整後終値
2. time_40_close：エントリー40営業日後の調整後終値
3. time_60_close：エントリー60営業日後の調整後終値
4. close_below_ma20：エントリー翌日以降、調整後終値がその日までの調整後終値20日移動平均を初めて下回った日の調整後終値。60営業日以内に発生しなければ60日終値
5. atr3_trailing：エントリー後の最高調整後高値から、エントリー時点で既知のATR14の3倍を引いた固定幅トレール。日中安値がストップ以下なら、調整後始値がストップ以下の場合は調整後始値、それ以外はストップ価格で退出。60営業日以内に発生しなければ60日終値

ATR14はエントリー日以前に計算済みの値だけを使い、将来ATRで幅を変更しないでください。MA20は各評価日の当日までの調整後終値だけで計算してください。出口規則、20/40/60日、20MA、ATR倍率3.0、最大保有60日を検証期で変更してはいけません。細かな利確・損切り閾値探索、最良出口の選択、組み合わせ別・環境別の出口変更は禁止します。

【価格と約定】
銘柄ごとの営業日行位置を使用し、暦日シフトは禁止します。全価格を分割調整後へ統一し、調整後OHLCの作成方法を明記してください。売買停止、欠損、上場廃止、分割を勝手に削除・補完しないでください。

同一日に複数の出口条件が成立し得る規則では、保守的な価格を優先してください。ATRストップのギャップ約定規則を上記どおり固定してください。日足だけでは日中順序を判定できない場合は、利益を大きく見せない保守的仮定と件数を保存してください。

【評価】
形成期と検証期を完全分離し、固定2,300組み合わせ×3エントリー×5出口について最低限以下を保存してください。
・組み合わせID、エントリー定義、出口定義、期間
・シグナル数、エントリー約定数、出口評価可能数、欠損・除外数
・退出日、退出価格、保有営業日数、退出理由
・総損益率と往復0.4%控除後の純損益率
・平均損益、中央値、勝率、PF、損益比
・MFE、MAE、利益捕捉率
・形成期から検証期への成績維持率
・年別件数、年別成績、方向再現性

取引コストは往復0.4%を固定してください。資金制約、同時保有数、ポジションサイズ、最大DD、破産確率はSTEP11以降の対象なので今回は計算しないでください。ケース・コントロール標本上の成績は実市場の無条件成績ではないことを明記してください。今回は参考比較だけで、出口・エントリー・組み合わせを採用しないでください。

【保存】
旧STEP10を上書きせず、以下へ新規Parquetを保存してください。
data/market_history/analysis/step10_exit_analysis_v2/

最低限：
・exit_definitions/
・exit_observations/
・exit_metrics/
・annual_metrics/
・exit_stability/
・reference_summary/
・boundary_exclusions/
・period_assignment_audit/

品質レポート：
data/market_history/quality/step10_v2_report.json
data/market_history/quality/STEP10_V2_REPORT.md

【品質検査】
入力ファイル前後SHA256、形成期・検証期分離、2025-09-08以降の不使用、銘柄別営業日シフト、STEP7固定定義との完全一致、STEP8による選別なし、STEP9エントリー定義・約定結果との完全一致、出口5規則の固定、検証期による出口変更なし、未来情報のシグナル・エントリー条件側混入なし、調整後価格の一貫性、同日約定の保守的処理、欠損・除外理由、主要キー、件数整合、無限大、Parquet再読込、同じ入力による2回実行のハッシュ一致を検査してください。

期間越境、汚染期間使用、STEP7定義変更、STEP8結果による選別、STEP9エントリー変更、検証期による出口規則変更、未来情報の条件側混入、入力変更、主要キー重複、再現性不一致が1件でもあればFAILにしてください。

【完了報告】
STEP10 V2 完了 / FAIL、期間、入力ペア数、固定候補数、固定組み合わせ数、固定エントリー数、出口定義数、定義別評価件数・平均・中央値・勝率・PF・損益比、形成期・検証期結果、欠損・除外理由、期間越境、未来情報混入、重複、作成ファイル、保存先、容量、入力変更、再現性、品質、残存リスクを報告してください。

PASSの場合だけ「STEP10 V2で作成した成果物は、同じ入力版では今後再計算不要」と明記してください。

処理完了後、回答の最後にSTEP11 V2「新しい完全未使用期間を決定して行うOOS検証」だけの完全な次回用プロンプトを表示し、同じ内容を `data/market_history/quality/STEP11_V2_PROMPT.md` へ保存してください。ただし、新しい完全未使用OOS期間が存在しない、または開始日をユーザーが承認していない場合はSTEP11を実行せず、その事実を明記するプロンプトにしてください。STEP12以降を実行させないでください。

今後も各STEP終了時に次のSTEPだけの完全なプロンプトを回答の最後へ表示し、品質フォルダへ保存してください。
"""


def make_report_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# STEP9 V2 QUALITY REPORT",
        "",
        f"- 品質: **{report['quality']}**",
        "- 形成期: データ開始日〜2023-12-31",
        "- 検証期: 2024-01-01〜2025-09-07",
        "- 汚染済み隔離期間: 2025-09-08以降（不使用）",
        f"- 入力ペア: {metrics['input_pairs']:,}",
        f"- 固定候補特徴量: {metrics['fixed_candidate_features']}",
        f"- 固定組み合わせ: {metrics['fixed_combinations']:,}",
        f"- 比較エントリー定義: {metrics['entry_definitions']}",
        f"- 期間越境値の混入: {metrics['period_or_boundary_errors']}",
        f"- 汚染期間の値使用: {metrics['contaminated_values_used']}",
        f"- 未来目的変数の条件側混入: {metrics['future_target_condition_leaks']}",
        f"- 主要キー重複: {metrics['major_key_duplicates']}",
        f"- 入力変更: {not report['inputs_unchanged']}",
        f"- 再現性: {'PASS' if report['reproducibility_passed'] else 'FAIL'}",
        "",
        "## 定義別集計",
        "",
        "件数は2,300組み合わせ内のシグナル該当を合計した combination-sample instances で、同じサンプルが複数組み合わせへ含まれます。",
        "",
        "| エントリー | 期間 | シグナル | 仮想約定 | 未約定 | 判定不能 | 約定率分母 | 約定率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics["entry_definition_summary"]:
        rate = "NA" if item["weighted_fill_rate"] is None else f"{item['weighted_fill_rate']:.4f}"
        lines.append(f"| {item['entry_definition']} | {item['period']} | {item['combination_signal_instances']:,} | {item['combination_filled_instances']:,} | {item['combination_unfilled_instances']:,} | {item['combination_indeterminate_instances']:,} | {item['fill_rate_denominator_instances']:,} | {rate} |")
    lines += [
        "",
        "## 価格・評価方針",
        "",
        "- anchor_close と first_pullback はSTEP2の調整後終値基準ターゲットをそのまま使用。",
        "- next_session_open は同じSTEP2将来終値・将来高値・将来安値を、当日の分割調整後始値基準へ換算。同日高値・安値もMFE/MAEへ含める。",
        "- 全方式のN日先はエントリー銘柄行の t+N。暦日シフトは不使用。",
        "- 日足の価格存在と約定可能性を分離。出来高がゼロまたは欠損なら判定不能として仮想約定に含めない。",
        "- first_pullbackは終値条件の確認と同じ終値で約定する理想化比較であり、実約定可能性の認証ではない。",
        "- 期間境界を越える評価値は読み込まず、NULLと理由を保存。欠損の補完・無条件削除はなし。",
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
    from step9_certification_gate import require_certified_inputs
    require_certified_inputs(root, args.verify_reproducibility)
    step1 = root / "features" / "equity_daily_features"
    step2 = root / "targets" / "equity_daily_forward_targets"
    step2_report_path = root / "quality" / "step2_report.json"
    step6 = root / "analysis" / "step6_univariate_v2"
    step7 = root / "analysis" / "step7_condition_combinations_v2"
    step8 = root / "analysis" / "step8_regime_analysis_v2"
    step7_report_path = root / "quality" / "step7_v2_report.json"
    step8_report_path = root / "quality" / "step8_v2_report.json"
    provenance_path = root / "quality" / "step9_input_provenance.json"
    for path in [step1, step2, step2_report_path, step6 / "analysis_samples", step7, step8, step7_report_path, step8_report_path, provenance_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    step2_report = json.loads(step2_report_path.read_text(encoding="utf-8"))
    step7_report = json.loads(step7_report_path.read_text(encoding="utf-8"))
    step8_report = json.loads(step8_report_path.read_text(encoding="utf-8"))
    validate_step2(step2_report)
    validate_step7(step7_report)
    validate_step8(step8_report)

    groups = [
        ("step1_prices", step1),
        ("step2_targets", step2),
        ("step2_report", step2_report_path),
        ("step6_analysis_samples", step6 / "analysis_samples"),
        ("step7_v2", step7),
        ("step7_v2_report", step7_report_path),
        ("step8_v2", step8),
        ("step8_v2_report", step8_report_path),
        ("step9_input_provenance", provenance_path),
    ]
    input_before = manifest(groups)
    samples_all = pd.read_parquet(step6 / "analysis_samples")
    samples = samples_all[samples_all.bucket.eq(MAIN_BUCKET)].sort_values("sample_id", kind="stable").reset_index(drop=True)
    definitions = pd.read_parquet(step7 / "combination_definitions").sort_values("combination_id", kind="stable").reset_index(drop=True)
    step7_metrics = pd.read_parquet(step7 / "combination_metrics")
    step8_definitions = pd.read_parquet(step8 / "regime_definitions")
    candidate_features = pd.read_parquet(step7 / "candidate_features")

    duplicate_samples = int(samples.sample_id.duplicated().sum())
    pair_counts = samples.groupby("event_id", observed=True).agg(rows=("sample_id", "size"), events=("outcome", "sum"), controls=("outcome", lambda values: int((values == 0).sum())), periods=("period", "nunique"))
    invalid_pairs = int((~pair_counts.eq(pd.Series({"rows": 2, "events": 1, "controls": 1, "periods": 1}))).any(axis=1).sum())
    period_imbalance = int(sum(abs(int(group.outcome.eq(1).sum()) - int(group.outcome.eq(0).sum())) for _, group in samples.groupby("period", observed=True)))
    contaminated_samples = int((pd.to_datetime(samples.anchor_date) >= HOLDOUT_START).sum())
    period_errors = int((samples.apply(lambda row: period_for_date(row.anchor_date) != row.period, axis=1)).sum())
    forbidden_definitions = sorted({feature for row in definitions.itertuples(index=False) for feature, _, _ in definition_conditions(row) if any(token in feature.lower() for token in FORBIDDEN)})
    if len(samples) != 17222 or len(pair_counts) != 8611 or int(samples.outcome.sum()) != 8611 or int(samples.outcome.eq(0).sum()) != 8611 or samples.Ticker.nunique() != 2637:
        raise RuntimeError("STEP6 main-bucket sample precondition mismatch")
    if len(definitions) != 2300 or len(candidate_features[candidate_features.selected_candidate]) != 24 or len(step8_definitions) != 9:
        raise RuntimeError("STEP7/STEP8 fixed-definition count mismatch")
    if duplicate_samples or invalid_pairs or period_imbalance or contaminated_samples or period_errors or forbidden_definitions:
        raise RuntimeError("Input sample integrity precondition failed")

    if args.single_run_output:
        single_output = Path(args.single_run_output).resolve()
        if single_output.exists():
            raise RuntimeError(f"Refusing to overwrite independent run output: {single_output}")
        prices = load_prices(step1, sorted(samples.Ticker.unique()))
        build_once(root, single_output, samples, definitions, step7_metrics, prices, step2)
        print(json.dumps({"single_run_output": str(single_output), "manifest": output_manifest(single_output)}, sort_keys=True))
        return

    destination = root / "analysis" / "step9_entry_timing_v2"
    with tempfile.TemporaryDirectory(prefix="step9_v2_", dir=str(root)) as temporary:
        temp = Path(temporary)
        first = temp / "first"
        second = temp / "second"
        common = [sys.executable, str(Path(__file__).resolve()), "--root", str(root), "--verify-reproducibility"]
        subprocess.run(common + ["--single-run-output", str(first)], check=True)
        first_manifest = output_manifest(first)
        if args.verify_reproducibility:
            subprocess.run(common + ["--single-run-output", str(second)], check=True)
            second_manifest = output_manifest(second)
            reproducibility = first_manifest == second_manifest
        else:
            raise RuntimeError("STEP9 requires two independent executions; no single-run PASS")
        if destination.exists():
            raise RuntimeError(f"Refusing to overwrite existing STEP9 V2 output: {destination}")
        shutil.copytree(first, destination)

    frames = {
        name: pd.read_parquet(destination / name)
        for name in (
            "entry_definitions", "signal_membership", "entry_observations", "entry_timing_metrics",
            "paired_timing_comparison", "annual_metrics", "timing_stability", "reference_summary",
            "boundary_exclusions", "period_assignment_audit",
        )
    }
    entries = frames["entry_observations"]
    metrics_frame = frames["entry_timing_metrics"]
    audit = frames["period_assignment_audit"]
    input_after = manifest(groups)
    inputs_unchanged = input_before == input_after
    entry_key_duplicates = int(entries.duplicated(["sample_id", "entry_definition"]).sum())
    membership_key_duplicates = int(frames["signal_membership"].duplicated(["combination_id", "sample_id"]).sum())
    metric_key_duplicates = int(metrics_frame.duplicated(["combination_id", "entry_definition", "period", "horizon"]).sum())
    annual_key_duplicates = int(frames["annual_metrics"].duplicated(["combination_id", "entry_definition", "horizon", "period", "year"]).sum())
    paired_key_duplicates = int(frames["paired_timing_comparison"].duplicated(["combination_id", "period", "horizon"]).sum())
    major_duplicates = entry_key_duplicates + membership_key_duplicates + metric_key_duplicates + annual_key_duplicates + paired_key_duplicates
    boundary_errors = 0
    contaminated_values_used = int(
        ((entries.entry_date >= HOLDOUT_START) & entries.entry_price.notna()).sum()
        + audit.contaminated_value_used.sum()
    )
    for horizon in HORIZONS:
        retained = entries[f"evaluation_available_{horizon}d"]
        boundary_errors += int((retained & ~entries[f"target_end_date_{horizon}d"].map(period_for_date).eq(entries.period)).sum())
        contaminated_values_used += int((retained & (entries[f"target_end_date_{horizon}d"] >= HOLDOUT_START)).sum())
    future_target_condition_leaks = len(forbidden_definitions)
    definition_change_count = int(definitions.validation_used_for_definition.fillna(False).sum()) + int(definitions.logical_operator.ne("AND").sum())
    step8_selection_count = int(entries.step8_used_for_selection.sum())
    validation_rule_change_count = int(entries.validation_used_to_change_rule.sum())
    adjustment_inconsistency_count = int((entries.filled & (~np.isfinite(entries.entry_price) | (entries.entry_price <= 0))).sum())
    output_infinities = sum(count_infinities(frame) for frame in frames.values())
    all_combinations_preserved = all(
        int(frames[name].combination_id.nunique()) == 2300
        for name in ("signal_membership", "entry_timing_metrics", "paired_timing_comparison", "timing_stability")
    )
    hard_failures = {
        "input_sample_id_duplicates": duplicate_samples,
        "input_invalid_pair_rows": invalid_pairs,
        "input_event_control_imbalance": period_imbalance,
        "period_or_boundary_errors": boundary_errors + period_errors,
        "contaminated_samples_or_values_used": contaminated_samples + contaminated_values_used,
        "future_target_condition_leaks": future_target_condition_leaks,
        "step7_definition_changes": definition_change_count,
        "step8_selection_uses": step8_selection_count,
        "validation_entry_rule_changes": validation_rule_change_count,
        "adjusted_price_inconsistencies": adjustment_inconsistency_count,
        "major_key_duplicates": major_duplicates,
        "fixed_combination_set_mismatch": 0 if all_combinations_preserved else 1,
        "nonfinite_output_numeric_values": output_infinities,
        "input_files_changed": 0 if inputs_unchanged else 1,
        "reproducibility_mismatch": 0 if reproducibility else 1,
    }
    quality = "PASS" if all(value == 0 for value in hard_failures.values()) else "FAIL"
    summary = frames["reference_summary"]
    summary20 = summary[summary.horizon.eq(20)].copy()
    entry_summary = []
    for row in summary20.itertuples(index=False):
        entry_summary.append({
            "entry_definition": row.entry_definition,
            "period": row.period,
            "combination_signal_instances": int(row.combination_signal_instances),
            "combination_filled_instances": int(row.combination_filled_instances),
            "combination_unfilled_instances": int(row.combination_unfilled_instances),
            "combination_indeterminate_instances": int(row.combination_indeterminate_instances),
            "fill_rate_denominator_instances": int(row.fill_rate_denominator_instances),
            "weighted_fill_rate": finite(row.weighted_fill_rate),
            "total_signal_based_fill_rate": finite(row.total_signal_based_fill_rate),
            "horizon_for_count_summary": 20,
        })
    exclusions = frames["boundary_exclusions"].exclusion_reason.value_counts().sort_index()
    total_signal_cells = int(metrics_frame.signal_count.sum())
    total_missing_cells = int(metrics_frame.evaluation_missing_count.sum())
    report = {
        "step": 9,
        "version": 2,
        "status": "STEP9 V2 complete" if quality == "PASS" else "STEP9 V2 FAIL",
        "quality": quality,
        "created_at_jst": datetime.now(JST).isoformat(),
        "formation": {"start": "data_start", "end": "2023-12-31"},
        "validation": {"start": "2024-01-01", "end": "2025-09-07"},
        "contaminated_holdout_start": "2025-09-08",
        "new_untouched_oos_start": None,
        "steps1_to_8_v2_recalculated": False,
        "old_step5_to_11_reused": False,
        "step7_combination_definitions_changed": False,
        "step8_results_used_for_selection": False,
        "entry_definition_source": "fixed before validation",
        "validation_used_to_change_entry_rules": False,
        "entry_exit_money_management_adopted": False,
        "transaction_costs_applied": False,
        "price_adjustment_policy": {
            "anchor_close": "STEP2 adjusted-close target triplets, unchanged",
            "first_pullback": "STEP2 adjusted-close target triplets at the filled pullback date, unchanged",
            "next_session_open": "raw Open multiplied by same-row Adj Close/Close; STEP2 future close/high/low levels rebased to that adjusted open; same-day adjusted high/low included in MFE/MAE",
            "horizon": "ticker trading row t+N, never calendar days",
        },
        "step6_analysis_samples_use": "fixed STEP7 combination application only; no ranking, selection, direction, threshold, or definition was recomputed",
        "input_artifact_provenance": json.loads(provenance_path.read_text(encoding="utf-8")),
        "input_manifest_before": input_before,
        "input_manifest_after": input_after,
        "inputs_unchanged": inputs_unchanged,
        "reproducibility_requested": bool(args.verify_reproducibility),
        "reproducibility_passed": reproducibility,
        "first_output_manifest": first_manifest,
        "second_output_manifest": second_manifest,
        "hard_failures": hard_failures,
        "metrics": {
            "input_pairs": 8611,
            "input_rows": 17222,
            "input_events": 8611,
            "input_controls": 8611,
            "input_tickers": 2637,
            "fixed_candidate_features": 24,
            "fixed_combinations": 2300,
            "entry_definitions": 3,
            "step8_environment_variables": 9,
            "entry_observation_rows": int(len(entries)),
            "signal_membership_rows": int(len(frames["signal_membership"])),
            "entry_metric_rows": int(len(metrics_frame)),
            "paired_timing_comparison_rows": int(len(frames["paired_timing_comparison"])),
            "annual_metric_rows": int(len(frames["annual_metrics"])),
            "boundary_exclusion_rows": int(len(frames["boundary_exclusions"])),
            "entry_definition_summary": entry_summary,
            "exclusion_reason_counts": {str(key): int(value) for key, value in exclusions.items()},
            "aggregate_evaluation_missing_rate": float(total_missing_cells / total_signal_cells) if total_signal_cells else None,
            "period_or_boundary_errors": boundary_errors + period_errors,
            "contaminated_values_used": contaminated_values_used,
            "future_target_condition_leaks": future_target_condition_leaks,
            "major_key_duplicates": major_duplicates,
            "parquet_reread_passed": True,
            "independent_processes_executed": 2,
            "all_fixed_combinations_preserved": all_combinations_preserved,
        },
        "save_path": str(destination),
        "file_size_bytes": directory_size(destination),
        "completion_statement": "STEP9 V2で作成した成果物は、同じ入力版では今後再計算不要" if quality == "PASS" else None,
        "residual_risks": [
            "case-control sample returns are not unconditional market performance",
            "the same sample can satisfy many of 2,300 combinations, so combination-signal instance totals are not unique trades",
            "first_pullback is a contingent order rule and its no-fill rate can materially change comparisons",
            "daily adjusted OHLC cannot reconstruct intraday execution order or slippage",
            "next-session-open MFE/MAE include the entry session while close-entry MFE/MAE begin on the next session, consistent with executable time after entry",
            "STEP2 source-invalid target triplets remain missing and were not imputed",
            "no transaction cost, exit, capital constraint, position sizing, or overlapping-position control is applied in STEP9 V2",
            "2,300 combinations create substantial multiple-comparison risk; no combination or timing is adopted",
            "2025-09-08 onward remains contaminated and cannot be reused as untouched OOS",
        ],
    }
    quality_dir = root / "quality"
    write_json(quality_dir / "step9_v2_report.json", report)
    (quality_dir / "STEP9_V2_REPORT.md").write_text(make_report_markdown(report), encoding="utf-8")
    prompt = step10_prompt(report["metrics"])
    (quality_dir / "STEP10_V2_PROMPT.md").write_text(prompt, encoding="utf-8")
    print(json.dumps({"quality": quality, "metrics": report["metrics"], "hard_failures": hard_failures, "file_size_bytes": report["file_size_bytes"]}, ensure_ascii=False, indent=2))
    if quality != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
