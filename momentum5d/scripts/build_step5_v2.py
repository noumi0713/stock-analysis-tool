# ruff: noqa: E501
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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
import pyarrow.parquet as pq

JST = ZoneInfo("Asia/Tokyo")
FORMATION_END = pd.Timestamp("2023-12-31")
VALIDATION_START = pd.Timestamp("2024-01-01")
VALIDATION_END = pd.Timestamp("2025-09-07")
CONTAMINATED_HOLDOUT_START = pd.Timestamp("2025-09-08")
PRE_WINDOW = 20
TARGET_HORIZON = 40
EVENT_EXCLUSION_RADIUS = 60
BUCKETS = (
    (-20, -11, "d-20_to_d-11"),
    (-10, -6, "d-10_to_d-6"),
    (-5, -1, "d-5_to_d-1"),
    (0, 0, "d0"),
)
VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build isolated STEP5 V2 control comparison")
    parser.add_argument("--root", default="data/market_history")
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--verify-reproducibility", action="store_true")
    return parser.parse_args()


def parquet_glob(path: Path) -> str:
    return str(path.resolve() / "**" / "*.parquet").replace("'", "''")


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_manifest(groups: list[tuple[str, Path]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for label, path in groups:
        for item in sorted(path.rglob("*.parquet")):
            key = f"{label}/{item.relative_to(path)}"
            result[key] = {"bytes": item.stat().st_size, "sha256": sha256(item)}
    return result


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def output_manifest(path: Path) -> dict[str, str]:
    return {str(item.relative_to(path)): sha256(item) for item in sorted(path.rglob("*.parquet"))}


def report_ok(path: Path, step: int) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"STEP{step} report missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("quality") != "PASS":
        raise RuntimeError(f"STEP{step} is not certified PASS: {path}")
    return report


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.reset_index(drop=True)
    pq.write_table(pa.Table.from_pandas(ordered, preserve_index=False), path, compression="zstd")
    if pq.ParquetFile(path).metadata.num_rows != len(ordered):
        raise RuntimeError(f"Parquet row count mismatch: {path}")


def classify_period(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values)
    return pd.Series(
        np.select(
            [dates <= FORMATION_END, dates.between(VALIDATION_START, VALIDATION_END)],
            ["formation", "validation"],
            default="contaminated_holdout",
        ),
        index=values.index,
        dtype="string",
    )


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def numeric_features(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "event_id",
        "Ticker",
        "event_start_date",
        "observation_date",
        "relative_day",
        "control_start_date",
        "bucket",
        "event_period",
        "control_period",
    }
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def bucket_medians(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for start, end, label in BUCKETS:
        part = frame.loc[frame.relative_day.between(start, end), ["event_id", *features]]
        medians = (
            part.groupby("event_id", sort=True)[features].median(numeric_only=True).reset_index()
        )
        medians.insert(1, "bucket", label)
        rows.append(medians)
    return pd.concat(rows, ignore_index=True)


def deterministic_matches(
    events: pd.DataFrame, controls: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match within ticker/period using a stable order and no control reuse."""
    matched: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    pools = {
        (ticker, period): group.sort_values(["control_start_date", "control_row_position"])
        for (ticker, period), group in controls.groupby(["Ticker", "control_period"], sort=True)
    }
    order = events.sort_values(["Ticker", "event_period", "event_start_date", "event_id"])
    used: set[tuple[str, pd.Timestamp]] = set()
    for event in order.itertuples(index=False):
        pool = pools.get((event.Ticker, event.event_period))
        if pool is None or pool.empty:
            unmatched.append(
                {
                    "event_id": event.event_id,
                    "Ticker": event.Ticker,
                    "event_start_date": event.event_start_date,
                    "event_period": event.event_period,
                    "reason": "no_eligible_control_same_ticker_period",
                }
            )
            continue
        candidates = pool.copy()
        candidates["same_year_sort"] = (
            candidates.control_start_date.dt.year != pd.Timestamp(event.event_start_date).year
        )
        candidates["distance_sort"] = (
            candidates.control_row_position.astype(int) - int(event.event_row_position)
        ).abs()
        candidates = candidates.sort_values(
            ["same_year_sort", "distance_sort", "control_start_date", "control_row_position"]
        )
        selected = None
        for candidate in candidates.itertuples(index=False):
            key = (candidate.Ticker, pd.Timestamp(candidate.control_start_date))
            if key not in used:
                selected = candidate
                used.add(key)
                break
        if selected is None:
            unmatched.append(
                {
                    "event_id": event.event_id,
                    "Ticker": event.Ticker,
                    "event_start_date": event.event_start_date,
                    "event_period": event.event_period,
                    "reason": "eligible_controls_exhausted_by_no_reuse",
                }
            )
            continue
        matched.append(
            {
                "event_id": event.event_id,
                "Ticker": event.Ticker,
                "event_start_date": event.event_start_date,
                "event_period": event.event_period,
                "event_window_start_date": event.event_window_start_date,
                "event_target_end_date": event.event_target_end_date,
                "event_row_position": int(event.event_row_position),
                "control_start_date": selected.control_start_date,
                "control_period": selected.control_period,
                "control_window_start_date": selected.control_window_start_date,
                "control_target_end_date": selected.control_target_end_date,
                "control_row_position": int(selected.control_row_position),
                "trading_day_distance": abs(
                    int(selected.control_row_position) - int(event.event_row_position)
                ),
                "same_calendar_year": bool(
                    pd.Timestamp(selected.control_start_date).year
                    == pd.Timestamp(event.event_start_date).year
                ),
                "control_reused": False,
                "boundary_check_pass": True,
                "exclusion_reason": pd.NA,
            }
        )
    return pd.DataFrame(matched), pd.DataFrame(unmatched)


def comparison_summary(
    event_bucket: pd.DataFrame,
    control_bucket: pd.DataFrame,
    matches: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    metadata = matches[["event_id", "event_period", "event_start_date", "control_start_date"]]
    event = event_bucket.merge(metadata, on="event_id", validate="many_to_one")
    control = control_bucket.merge(metadata, on="event_id", validate="many_to_one")
    rows: list[dict[str, Any]] = []
    for period in ("formation", "validation"):
        event_period = event[event.event_period == period]
        control_period = control[control.event_period == period]
        years = sorted(pd.to_datetime(event_period.event_start_date).dt.year.unique())
        scopes: list[tuple[str, int | None]] = [
            ("period", None),
            *(("year", int(y)) for y in years),
        ]
        for scope_type, year in scopes:
            for bucket in [item[2] for item in BUCKETS]:
                left = event_period[event_period.bucket == bucket]
                right = control_period[control_period.bucket == bucket]
                if year is not None:
                    left = left[pd.to_datetime(left.event_start_date).dt.year == year]
                    right = right[pd.to_datetime(right.event_start_date).dt.year == year]
                joined = left[["event_id", *features]].merge(
                    right[["event_id", *features]],
                    on="event_id",
                    suffixes=("_event", "_control"),
                    validate="one_to_one",
                )
                for feature in features:
                    event_values = pd.to_numeric(joined[f"{feature}_event"], errors="coerce")
                    control_values = pd.to_numeric(joined[f"{feature}_control"], errors="coerce")
                    valid = (
                        event_values.notna()
                        & control_values.notna()
                        & np.isfinite(event_values)
                        & np.isfinite(control_values)
                    )
                    differences = (event_values[valid] - control_values[valid]).astype(float)
                    standard_deviation = differences.std(ddof=1)
                    nonzero = differences[differences != 0]
                    consistency = None
                    if len(nonzero):
                        consistency = finite(max((nonzero > 0).mean(), (nonzero < 0).mean()))
                    rows.append(
                        {
                            "period": period,
                            "scope_type": scope_type,
                            "scope_year": year,
                            "bucket": bucket,
                            "feature": feature,
                            "pairs": int(len(joined)),
                            "valid_pairs": int(valid.sum()),
                            "event_missing_rate": finite(event_values.isna().mean()),
                            "control_missing_rate": finite(control_values.isna().mean()),
                            "event_median": finite(event_values[valid].median())
                            if valid.any()
                            else None,
                            "control_median": finite(control_values[valid].median())
                            if valid.any()
                            else None,
                            "mean_paired_difference": finite(differences.mean())
                            if len(differences)
                            else None,
                            "median_paired_difference": finite(differences.median())
                            if len(differences)
                            else None,
                            "paired_standardized_effect": (
                                finite(differences.mean() / standard_deviation)
                                if len(differences) > 1
                                and standard_deviation
                                and math.isfinite(standard_deviation)
                                else None
                            ),
                            "paired_direction_consistency": consistency,
                        }
                    )
    return pd.DataFrame(rows)


def reason_for_boundary(row: Any, prefix: str) -> str | None:
    period = getattr(row, f"{prefix}_period")
    if period == "contaminated_holdout":
        return "contaminated_holdout"
    if pd.isna(getattr(row, f"{prefix}_window_start_date")):
        return "missing_full_pre_window"
    if pd.isna(getattr(row, f"{prefix}_target_end_date")):
        return "missing_full_40d_target_horizon"
    if getattr(row, f"{prefix}_window_period") != period:
        return "pre_window_crosses_period_boundary"
    if getattr(row, f"{prefix}_target_period") != period:
        return "target_horizon_crosses_period_boundary"
    return None


def build_once(root: Path, stage: Path, memory_limit: str) -> dict[str, Any]:
    for name in (
        "matched_pairs",
        "unmatched_events",
        "control_pre_event_windows",
        "control_bucket_medians",
        "feature_comparison",
        "boundary_exclusions",
        "period_assignment_audit",
        "old_vs_v2_comparison",
        "matching_summary",
    ):
        (stage / name).mkdir(parents=True, exist_ok=True)

    features_dir = root / "features/equity_daily_features"
    labels_dir = root / "targets/large_move_events/event_labels"
    events_dir = root / "targets/large_move_events/independent_events"
    step4_dir = root / "analysis/step4_pre_event_common_features"
    work = stage.parent / "step5_v2.duckdb"
    connection = duckdb.connect(str(work))
    connection.execute(f"SET memory_limit='{memory_limit}'")
    connection.execute("SET threads=2")
    connection.execute(f"CREATE VIEW features AS FROM read_parquet('{parquet_glob(features_dir)}')")
    connection.execute(f"CREATE VIEW labels AS FROM read_parquet('{parquet_glob(labels_dir)}')")
    connection.execute(f"CREATE VIEW events AS FROM read_parquet('{parquet_glob(events_dir)}')")
    connection.execute(
        f"""
        CREATE TABLE positions AS
        SELECT Date,Ticker,
          row_number() OVER(PARTITION BY Ticker ORDER BY Date)-1 AS row_position,
          lag(Date,{PRE_WINDOW}) OVER(PARTITION BY Ticker ORDER BY Date) AS window_start_date,
          lead(Date,{TARGET_HORIZON}) OVER(PARTITION BY Ticker ORDER BY Date) AS target_end_date
        FROM features
        """
    )
    positions = connection.execute("SELECT * FROM positions ORDER BY Ticker,Date").df()
    positions["Date"] = pd.to_datetime(positions.Date)
    positions["window_start_date"] = pd.to_datetime(positions.window_start_date)
    positions["target_end_date"] = pd.to_datetime(positions.target_end_date)
    positions["period"] = classify_period(positions.Date)
    positions["window_period"] = classify_period(positions.window_start_date)
    positions["target_period"] = classify_period(positions.target_end_date)

    independent = connection.execute(
        "SELECT event_id,Ticker,event_start_date FROM events ORDER BY Ticker,event_start_date,event_id"
    ).df()
    independent["event_start_date"] = pd.to_datetime(independent.event_start_date)
    anchors = independent.merge(
        positions,
        left_on=["Ticker", "event_start_date"],
        right_on=["Ticker", "Date"],
        how="left",
        validate="one_to_one",
    ).drop(columns="Date")
    anchors = anchors.rename(
        columns={
            "period": "event_period",
            "window_start_date": "event_window_start_date",
            "target_end_date": "event_target_end_date",
            "window_period": "event_window_period",
            "target_period": "event_target_period",
            "row_position": "event_row_position",
        }
    )
    anchors["boundary_exclusion_reason"] = [
        "event_anchor_missing_from_step1"
        if pd.isna(row.event_row_position)
        else reason_for_boundary(row, "event")
        for row in anchors.itertuples(index=False)
    ]
    eligible_events = anchors[anchors.boundary_exclusion_reason.isna()].copy()
    eligible_events["event_row_position"] = eligible_events.event_row_position.astype(int)

    event_positions = eligible_events[["Ticker", "event_row_position"]].drop_duplicates()
    connection.register("eligible_event_positions", event_positions)
    labels = connection.execute(
        """
        SELECT p.Date AS control_start_date,p.Ticker,p.row_position AS control_row_position,
          p.window_start_date AS control_window_start_date,p.target_end_date AS control_target_end_date,
          l.event_40d_ge_30pct,l.max_forward_close_return_40d
        FROM positions p JOIN labels l ON p.Date=l.Date AND p.Ticker=l.Ticker
        WHERE l.event_40d_ge_30pct=false AND l.max_forward_close_return_40d IS NOT NULL
        ORDER BY p.Ticker,p.Date
        """
    ).df()
    for column in ("control_start_date", "control_window_start_date", "control_target_end_date"):
        labels[column] = pd.to_datetime(labels[column])
    labels["control_period"] = classify_period(labels.control_start_date)
    labels["control_window_period"] = classify_period(labels.control_window_start_date)
    labels["control_target_period"] = classify_period(labels.control_target_end_date)
    labels["boundary_exclusion_reason"] = [
        reason_for_boundary(row, "control") for row in labels.itertuples(index=False)
    ]

    # Use only eligible in-period event anchors for proximity exclusion. Quarantined events
    # are never allowed to influence control selection.
    event_positions_by_ticker = {
        ticker: np.sort(group.event_row_position.to_numpy(dtype=np.int64))
        for ticker, group in eligible_events.groupby("Ticker", sort=True)
    }
    near_event = np.zeros(len(labels), dtype=bool)
    for ticker, index in labels.groupby("Ticker", sort=True).groups.items():
        event_pos = event_positions_by_ticker.get(ticker)
        if event_pos is None or not len(event_pos):
            continue
        candidate_pos = labels.loc[index, "control_row_position"].to_numpy(dtype=np.int64)
        insertion = np.searchsorted(event_pos, candidate_pos)
        left = np.where(insertion > 0, event_pos[np.maximum(insertion - 1, 0)], -(10**12))
        right = np.where(
            insertion < len(event_pos), event_pos[np.minimum(insertion, len(event_pos) - 1)], 10**12
        )
        near_event[np.asarray(index)] = (
            np.minimum(abs(candidate_pos - left), abs(candidate_pos - right))
            <= EVENT_EXCLUSION_RADIUS
        )
    labels["near_eligible_event_60d"] = near_event
    labels.loc[
        labels.boundary_exclusion_reason.isna() & labels.near_eligible_event_60d,
        "boundary_exclusion_reason",
    ] = "within_60_trading_rows_of_eligible_event"
    eligible_controls = labels[labels.boundary_exclusion_reason.isna()].copy()

    matched, unmatched_pool = deterministic_matches(eligible_events, eligible_controls)
    excluded_events = anchors[anchors.boundary_exclusion_reason.notna()][
        ["event_id", "Ticker", "event_start_date", "event_period", "boundary_exclusion_reason"]
    ].rename(columns={"boundary_exclusion_reason": "reason"})
    unmatched = pd.concat([excluded_events, unmatched_pool], ignore_index=True).sort_values(
        ["Ticker", "event_start_date", "event_id"]
    )

    audit_events = anchors[
        [
            "event_id",
            "Ticker",
            "event_start_date",
            "event_period",
            "event_window_start_date",
            "event_window_period",
            "event_target_end_date",
            "event_target_period",
            "boundary_exclusion_reason",
        ]
    ].copy()
    audit_events.insert(0, "record_type", "event")
    audit_controls = labels[
        [
            "Ticker",
            "control_start_date",
            "control_period",
            "control_window_start_date",
            "control_window_period",
            "control_target_end_date",
            "control_target_period",
            "boundary_exclusion_reason",
        ]
    ].copy()
    audit_controls.insert(0, "record_type", "control_candidate")
    audit_controls.insert(
        1, "event_id", pd.Series(pd.NA, index=audit_controls.index, dtype="string")
    )
    audit_controls = audit_controls.rename(
        columns={
            "control_start_date": "event_start_date",
            "control_period": "event_period",
            "control_window_start_date": "event_window_start_date",
            "control_window_period": "event_window_period",
            "control_target_end_date": "event_target_end_date",
            "control_target_period": "event_target_period",
        }
    )
    period_audit = pd.concat([audit_events, audit_controls], ignore_index=True)

    boundary_event = excluded_events.assign(record_type="event").rename(
        columns={"event_start_date": "anchor_date", "event_period": "period"}
    )
    boundary_control = labels[labels.boundary_exclusion_reason.notna()][
        ["Ticker", "control_start_date", "control_period", "boundary_exclusion_reason"]
    ].rename(
        columns={
            "control_start_date": "anchor_date",
            "control_period": "period",
            "boundary_exclusion_reason": "reason",
        }
    )
    boundary_control.insert(
        0, "event_id", pd.Series(pd.NA, index=boundary_control.index, dtype="string")
    )
    boundary_control["record_type"] = "control_candidate"
    boundary_exclusions = pd.concat([boundary_event, boundary_control], ignore_index=True)

    matched_ids = matched[["event_id"]]
    event_windows = pd.read_parquet(step4_dir / "pre_event_windows")
    for column in event_windows.select_dtypes(include="boolean").columns:
        event_windows[column] = event_windows[column].astype("Float64")
    event_windows["observation_date"] = pd.to_datetime(event_windows.observation_date)
    event_windows = event_windows.merge(
        matched_ids, on="event_id", how="inner", validate="many_to_one"
    )
    matched_window_counts = event_windows.groupby("event_id").agg(
        rows=("relative_day", "size"),
        minimum=("relative_day", "min"),
        maximum=("relative_day", "max"),
    )
    bad_event_windows = matched_window_counts[
        (matched_window_counts.rows != 21)
        | (matched_window_counts.minimum != -20)
        | (matched_window_counts.maximum != 0)
    ]
    if len(bad_event_windows):
        raise RuntimeError(f"Matched events with incomplete STEP4 window: {len(bad_event_windows)}")
    features = numeric_features(event_windows)

    connection.register(
        "matched", matched[["event_id", "Ticker", "control_row_position", "control_start_date"]]
    )
    schema = connection.execute("DESCRIBE features").fetchall()
    feature_columns = [name for name, *_ in schema if name not in {"Date", "Ticker"}]
    selected = ",".join(f"f.{quote(name)}" for name in feature_columns)
    control_windows = connection.execute(
        f"""
        SELECT m.event_id,m.Ticker,m.control_start_date,f.Date AS observation_date,
          CAST(p.row_position-m.control_row_position AS SMALLINT) AS relative_day,{selected}
        FROM matched m
        JOIN positions p ON m.Ticker=p.Ticker
          AND p.row_position BETWEEN m.control_row_position-{PRE_WINDOW} AND m.control_row_position
        JOIN features f ON p.Ticker=f.Ticker AND p.Date=f.Date
        ORDER BY m.event_id,relative_day
        """
    ).df()
    connection.close()
    work.unlink(missing_ok=True)
    for column in control_windows.select_dtypes(include="boolean").columns:
        control_windows[column] = control_windows[column].astype("Float64")
    control_windows["observation_date"] = pd.to_datetime(control_windows.observation_date)
    counts = control_windows.groupby("event_id").agg(
        rows=("relative_day", "size"),
        minimum=("relative_day", "min"),
        maximum=("relative_day", "max"),
    )
    if ((counts.rows != 21) | (counts.minimum != -20) | (counts.maximum != 0)).any():
        raise RuntimeError("Control window row-count invariant failed")

    event_bucket = bucket_medians(event_windows, features)
    control_bucket = bucket_medians(control_windows, features)
    comparison = comparison_summary(event_bucket, control_bucket, matched, features)

    old_dir = root / "analysis/step5_control_comparison"
    old_rows: list[dict[str, Any]] = []
    if old_dir.exists():
        old_matches = pd.read_parquet(old_dir / "matched_pairs")
        old_matches["event_start_date"] = pd.to_datetime(old_matches.event_start_date)
        old_matches["control_start_date"] = pd.to_datetime(old_matches.control_start_date)
        old_windows = pd.read_parquet(
            old_dir / "control_pre_event_windows",
            columns=["event_id", "observation_date", "relative_day"],
        ).merge(
            old_matches[["event_id", "event_start_date"]], on="event_id", validate="many_to_one"
        )
        old_windows["observation_date"] = pd.to_datetime(old_windows.observation_date)
        old_rows.extend(
            [
                {
                    "metric": "pre_oos_assigned_oos_controls",
                    "old_value": int(
                        (
                            (old_matches.event_start_date < CONTAMINATED_HOLDOUT_START)
                            & (old_matches.control_start_date >= CONTAMINATED_HOLDOUT_START)
                        ).sum()
                    ),
                    "v2_value": int(
                        (
                            (matched.event_start_date < CONTAMINATED_HOLDOUT_START)
                            & (matched.control_start_date >= CONTAMINATED_HOLDOUT_START)
                        ).sum()
                    ),
                },
                {
                    "metric": "formation_assigned_later_controls",
                    "old_value": int(
                        (
                            (old_matches.event_start_date <= FORMATION_END)
                            & (old_matches.control_start_date > FORMATION_END)
                        ).sum()
                    ),
                    "v2_value": int(
                        (
                            (matched.event_period == "formation")
                            & (matched.control_start_date > FORMATION_END)
                        ).sum()
                    ),
                },
                {
                    "metric": "oos_observations_in_d5_to_d1",
                    "old_value": int(
                        (
                            (old_windows.event_start_date < CONTAMINATED_HOLDOUT_START)
                            & (old_windows.observation_date >= CONTAMINATED_HOLDOUT_START)
                            & old_windows.relative_day.between(-5, -1)
                        ).sum()
                    ),
                    "v2_value": int(
                        (
                            (control_windows.observation_date >= CONTAMINATED_HOLDOUT_START)
                            & control_windows.relative_day.between(-5, -1)
                        ).sum()
                    ),
                },
            ]
        )
    old_vs_v2 = pd.DataFrame(old_rows, columns=["metric", "old_value", "v2_value"])

    summary_rows: list[dict[str, Any]] = []
    for period in ("formation", "validation"):
        eligible = eligible_events[eligible_events.event_period == period]
        period_matches = matched[matched.event_period == period]
        period_unmatched = unmatched[unmatched.event_period == period]
        distance = period_matches.trading_day_distance
        summary_rows.append(
            {
                "period": period,
                "eligible_events": int(len(eligible)),
                "matched_events": int(len(period_matches)),
                "unmatched_events": int(len(period_unmatched)),
                "match_rate": finite(len(period_matches) / len(eligible))
                if len(eligible)
                else None,
                "tickers": int(period_matches.Ticker.nunique()),
                "same_calendar_year_rate": finite(period_matches.same_calendar_year.mean())
                if len(period_matches)
                else None,
                "distance_median": finite(distance.median()) if len(distance) else None,
                "distance_q25": finite(distance.quantile(0.25)) if len(distance) else None,
                "distance_q75": finite(distance.quantile(0.75)) if len(distance) else None,
            }
        )
    matching_summary = pd.DataFrame(summary_rows)

    frames = {
        "matched_pairs": matched.sort_values(["Ticker", "event_start_date", "event_id"]),
        "unmatched_events": unmatched,
        "control_pre_event_windows": control_windows,
        "control_bucket_medians": control_bucket,
        "feature_comparison": comparison,
        "boundary_exclusions": boundary_exclusions,
        "period_assignment_audit": period_audit,
        "old_vs_v2_comparison": old_vs_v2,
        "matching_summary": matching_summary,
    }
    for name, frame in frames.items():
        write_parquet(frame, stage / name / "part.parquet")

    event_nulls = int(event_windows[features].isna().sum().sum())
    control_nulls = int(control_windows[features].isna().sum().sum())
    feature_cells = int((len(event_windows) + len(control_windows)) * len(features))
    metrics = {
        "input_events": int(len(anchors)),
        "eligible_events": int(len(eligible_events)),
        "matched_events": int(len(matched)),
        "unmatched_or_boundary_excluded_events": int(len(unmatched)),
        "matched_tickers": int(matched.Ticker.nunique()),
        "event_window_rows": int(len(event_windows)),
        "control_window_rows": int(len(control_windows)),
        "features_compared": int(len(features)),
        "feature_missing_cells": event_nulls + control_nulls,
        "feature_cells": feature_cells,
        "feature_missing_rate": finite((event_nulls + control_nulls) / feature_cells)
        if feature_cells
        else None,
        "boundary_exclusion_reasons": {
            str(key): int(value)
            for key, value in boundary_exclusions.reason.value_counts(dropna=False)
            .sort_index()
            .items()
        },
        "unmatched_reasons": {
            str(key): int(value)
            for key, value in unmatched.reason.value_counts(dropna=False).sort_index().items()
        },
        "event_id_duplicates": int(matched.event_id.duplicated().sum()),
        "control_ticker_date_duplicates": int(
            matched.duplicated(["Ticker", "control_start_date"]).sum()
        ),
        "control_reuse_rows": int(matched.control_reused.sum()),
        "cross_period_matches": int((matched.event_period != matched.control_period).sum()),
        "pre_window_boundary_crossings": int(
            (
                (matched.event_window_start_date < VALIDATION_START)
                & (matched.event_period == "validation")
                | (matched.control_window_start_date < VALIDATION_START)
                & (matched.control_period == "validation")
            ).sum()
        ),
        "target_boundary_crossings": int(
            (
                (matched.event_target_end_date > VALIDATION_END)
                & (matched.event_period == "validation")
                | (matched.control_target_end_date > VALIDATION_END)
                & (matched.control_period == "validation")
                | (matched.event_target_end_date > FORMATION_END)
                & (matched.event_period == "formation")
                | (matched.control_target_end_date > FORMATION_END)
                & (matched.control_period == "formation")
            ).sum()
        ),
        "contaminated_dates_in_matched": int(
            (
                (matched.event_start_date >= CONTAMINATED_HOLDOUT_START)
                | (matched.control_start_date >= CONTAMINATED_HOLDOUT_START)
            ).sum()
        ),
        "contaminated_observations_in_analysis": int(
            (control_windows.observation_date >= CONTAMINATED_HOLDOUT_START).sum()
            + (event_windows.observation_date >= CONTAMINATED_HOLDOUT_START).sum()
        ),
        "missing_matched_dates": int(
            matched[
                [
                    "event_start_date",
                    "event_window_start_date",
                    "event_target_end_date",
                    "control_start_date",
                    "control_window_start_date",
                    "control_target_end_date",
                ]
            ]
            .isna()
            .sum()
            .sum()
        ),
        "nonfinite_comparison_values": int(
            np.isinf(
                comparison.select_dtypes(include="number").to_numpy(dtype=float, na_value=np.nan)
            ).sum()
        ),
        "old_comparison": old_rows,
    }
    return metrics


def step6_prompt() -> str:
    return """# STEP6 V2 only: corrected univariate analysis

STEP5 V2の認証済み成果物 `data/market_history/analysis/step5_control_comparison_v2/`
だけを対照比較入力にし、STEP1〜STEP5 V2を再計算せず、STEP6 V2「単変量分析」だけを実行してください。

旧STEP5〜STEP11の成果物は再利用禁止です。形成期はデータ開始日〜2023-12-31、検証期は2024-01-01〜2025-09-07として完全分離してください。2025-09-08以降は既に使用済みの `contaminated_holdout` であり、分析、集計、特徴量選択、閾値計算、可視化に一切使用しないでください。新しい完全未使用OOSの開始日は独断で決定しないでください。

各特徴を単独で評価し、形成期と検証期を分離して、AUC、効果量、欠損率、年別方向再現性、形成期で固定した粗い分位の検証期イベント率を新規Parquetへ保存してください。方向を検証期ごとに反転してAUCを良く見せないでください。特徴量採用、組み合わせ、シグナル、売買ルール、細かな閾値最適化は行わないでください。

期間越境、OOS混入、キー重複、入力変更、実行非再現が1件でもあればFAILにしてください。完了時はSTEP6 V2の品質、件数、欠損、容量、残存リスクを報告し、PASSの場合だけ同じ入力版で再計算不要と明記してください。最後にSTEP7 V2だけの次回プロンプトを保存してください。
"""


def invalidation_text() -> str:
    return """# STEP5 V1由来成果物の無効化

旧STEP5は対照群の期間を `control_start_date` ではなく、対応イベントの `event_start_date` で分類していました。形成期・検証期へOOS対照群514件（形成期8件、検証期506件）が混入し、-5〜-1区間にはOOS観測2,432行（506対照群、293銘柄）が含まれました。形成期へ形成期終了後の対照群305件も混入しました。

このため、旧STEP5を入力にした旧STEP6、STEP7、STEP8、STEP9、STEP10、STEP11は再利用禁止です。ファイルは監査証跡として削除しません。修正版は `data/market_history/analysis/step5_control_comparison_v2/` です。
"""


def markdown_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    status = report["status"]
    return f"""# STEP5 V2 品質レポート

判定：**{status}**

- 形成期：データ開始日〜2023-12-31
- 検証期：2024-01-01〜2025-09-07
- 汚染済み隔離期間：2025-09-08以降
- 入力イベント：{metrics["input_events"]:,}
- 境界条件を満たすイベント：{metrics["eligible_events"]:,}
- マッチ成功：{metrics["matched_events"]:,}
- 未マッチ・境界除外：{metrics["unmatched_or_boundary_excluded_events"]:,}
- 対象銘柄：{metrics["matched_tickers"]:,}
- 修正後の期間越境：{metrics["cross_period_matches"]:,}
- 修正後の隔離期間観測混入：{metrics["contaminated_observations_in_analysis"]:,}
- event_id重複：{metrics["event_id_duplicates"]:,}
- 対照銘柄×日付重複：{metrics["control_ticker_date_duplicates"]:,}
- 特徴量欠損率：{metrics["feature_missing_rate"]:.6%}
- 入力変更：{not report["inputs_unchanged"]}
- 2回実行のParquet一致：{report["reproducibility_passed"]}
- 保存容量：{report["file_size_bytes"]:,} bytes

旧STEP5〜STEP11は再利用禁止です。2025-09-08以降は完全未使用OOSへ戻していません。新しいOOS開始日はSTEP10 V2まで条件を再固定した後に別途決定します。

{report["completion_statement"]}
"""


def main() -> int:
    options = parse_args()
    repository = Path(__file__).resolve().parents[2]
    root = (repository / "momentum5d" / options.root).resolve()
    reports = {
        step: report_ok(root / f"quality/step{step}_report.json", step) for step in (1, 3, 4)
    }
    groups = [
        ("step1", root / "features/equity_daily_features"),
        ("step3_labels", root / "targets/large_move_events/event_labels"),
        ("step3_events", root / "targets/large_move_events/independent_events"),
        ("step4", root / "analysis/step4_pre_event_common_features"),
    ]
    before = parquet_manifest(groups)
    target = root / "analysis/step5_control_comparison_v2"
    quality_dir = root / "quality"
    if target.exists() or (quality_dir / "step5_v2_report.json").exists():
        raise RuntimeError("STEP5 V2 output already exists; immutable outputs are not overwritten")

    temporary = Path(tempfile.mkdtemp(prefix="step5_v2_", dir=root.parent))
    first = temporary / "first"
    metrics = build_once(root, first, options.memory_limit)
    first_manifest = output_manifest(first)
    reproducibility_passed = False
    second_manifest: dict[str, str] | None = None
    if options.verify_reproducibility:
        gc.collect()
        second = temporary / "second"
        second_metrics = build_once(root, second, options.memory_limit)
        second_manifest = output_manifest(second)
        reproducibility_passed = first_manifest == second_manifest and metrics == second_metrics
    after = parquet_manifest(groups)
    inputs_unchanged = before == after

    old_lookup = {row["metric"]: row for row in metrics["old_comparison"]}
    known_old_counts_verified = (
        old_lookup.get("pre_oos_assigned_oos_controls", {}).get("old_value") == 514
        and old_lookup.get("oos_observations_in_d5_to_d1", {}).get("old_value") == 2432
        and old_lookup.get("formation_assigned_later_controls", {}).get("old_value") == 305
    )
    hard_failures = {
        "cross_period_matches": metrics["cross_period_matches"],
        "pre_window_boundary_crossings": metrics["pre_window_boundary_crossings"],
        "target_boundary_crossings": metrics["target_boundary_crossings"],
        "contaminated_dates_in_matched": metrics["contaminated_dates_in_matched"],
        "contaminated_observations_in_analysis": metrics["contaminated_observations_in_analysis"],
        "event_id_duplicates": metrics["event_id_duplicates"],
        "control_ticker_date_duplicates": metrics["control_ticker_date_duplicates"],
        "control_reuse_rows": metrics["control_reuse_rows"],
        "missing_matched_dates": metrics["missing_matched_dates"],
        "nonfinite_comparison_values": metrics["nonfinite_comparison_values"],
    }
    passed = (
        all(value == 0 for value in hard_failures.values())
        and inputs_unchanged
        and options.verify_reproducibility
        and reproducibility_passed
        and known_old_counts_verified
        and metrics["matched_events"] > 0
    )
    status = "STEP5 V2 complete" if passed else "STEP5 V2 FAIL"
    target.parent.mkdir(parents=True, exist_ok=True)
    first.rename(target)
    report = {
        "step": 5,
        "version": VERSION,
        "status": status,
        "quality": "PASS" if passed else "FAIL",
        "created_at_jst": datetime.now(JST).isoformat(),
        "formation": {"start": "data_start", "end": str(FORMATION_END.date())},
        "validation": {"start": str(VALIDATION_START.date()), "end": str(VALIDATION_END.date())},
        "contaminated_holdout_start": str(CONTAMINATED_HOLDOUT_START.date()),
        "new_untouched_oos_start": None,
        "steps1_to_4_recalculated": False,
        "strategy_modified": False,
        "optimization_performed": False,
        "input_reports": {str(step): reports[step].get("input_fingerprint") for step in reports},
        "input_parquet_manifest_before": before,
        "input_parquet_manifest_after": after,
        "inputs_unchanged": inputs_unchanged,
        "known_old_counts_verified": known_old_counts_verified,
        "reproducibility_requested": options.verify_reproducibility,
        "reproducibility_passed": reproducibility_passed,
        "first_output_manifest": first_manifest,
        "second_output_manifest": second_manifest,
        "hard_failures": hard_failures,
        "metrics": metrics,
        "save_path": str(target),
        "file_size_bytes": directory_size(target),
        "completion_statement": (
            "STEP5 V2で作成した成果物は、同じ入力版では今後再計算不要"
            if passed
            else "STEP5 V2はFAILのため再計算不要とは認証しない"
        ),
        "downstream_status": "old STEP6 through STEP11 invalidated",
        "residual_risks": [
            "2025-09-08以降は既使用のため完全未使用OOSには戻らない",
            "対照群マッチングは同一銘柄・同一期間の決定的貪欲法であり、未マッチが残り得る",
            "日足欠損により銘柄ごとの40行が市場共通の40営業日と一致しない場合がある",
            "STEP4自体は隔離期間を含むが、V2分析ではマッチ済み期間内event_idの21行だけを抽出した",
        ],
    }
    write_json(quality_dir / "step5_v2_report.json", report)
    (quality_dir / "STEP5_V2_REPORT.md").write_text(markdown_report(report), encoding="utf-8")
    (quality_dir / "STEP6_V2_PROMPT.md").write_text(step6_prompt(), encoding="utf-8")
    (quality_dir / "STEP6_TO_STEP11_INVALIDATED_BY_STEP5_V1.md").write_text(
        invalidation_text(), encoding="utf-8"
    )
    shutil.rmtree(temporary, ignore_errors=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
