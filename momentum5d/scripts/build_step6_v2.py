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

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

JST = ZoneInfo("Asia/Tokyo")
FORMATION_END = pd.Timestamp("2023-12-31")
VALIDATION_START = pd.Timestamp("2024-01-01")
VALIDATION_END = pd.Timestamp("2025-09-07")
HOLDOUT_START = pd.Timestamp("2025-09-08")
MAIN_BUCKET = "d-5_to_d-1"
BUCKETS = ("d-20_to_d-11", "d-10_to_d-6", MAIN_BUCKET, "d0")
FORBIDDEN = ("future", "forward", "target", "label", "mfe", "mae", "peak_date", "event_end", "outcome")
META = {
    "sample_id", "event_id", "Ticker", "period", "bucket", "observation_type", "outcome",
    "event_start_date", "control_start_date", "anchor_date", "pair_year", "anchor_year",
}
ABSOLUTE = {
    "Open", "High", "Low", "Close", "Adj Close", "Volume", "trading_value", "atr_14",
    "market_trading_value_total", "sector_trading_value",
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STEP6 V2 leakage-safe univariate analysis")
    parser.add_argument("--root", default="data/market_history")
    parser.add_argument("--verify-reproducibility", action="store_true")
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


def parquet_manifest(path: Path) -> dict[str, str]:
    return {str(item.relative_to(path)): sha256(item) for item in sorted(path.rglob("*.parquet"))}


def size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame.reset_index(drop=True), preserve_index=False)
    pq.write_table(table, path, compression="zstd")
    reread = pd.read_parquet(path)
    if len(reread) != len(frame) or list(reread.columns) != list(frame.columns):
        raise RuntimeError(f"Parquet reread mismatch: {path}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def base_exclusion(column: str, dtype: Any) -> str | None:
    lower = column.lower()
    if column in META:
        return "identifier_or_metadata"
    if any(token in lower for token in FORBIDDEN):
        return "future_or_target_information"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "date"
    if not (pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)):
        return "non_numeric_or_categorical"
    if column in ABSOLUTE:
        return "absolute_price_volume_or_trading_value_level"
    if lower.startswith("ma_") and not any(word in lower for word in ("deviation", "slope", "rank")):
        return "moving_average_absolute_level"
    if lower.endswith("_close"):
        return "external_or_index_absolute_level"
    if lower.endswith("_mean_5d") or lower.endswith("_mean_20d"):
        return "volume_or_trading_value_average_level"
    return None


def validate_step5(report: dict[str, Any]) -> None:
    required = {
        "quality": "PASS", "version": 2, "inputs_unchanged": True,
        "known_old_counts_verified": True, "reproducibility_passed": True,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise RuntimeError(f"STEP5 V2 precondition mismatch: {key}")
    expected_metrics = {
        "input_events": 13025, "eligible_events": 8896, "matched_events": 8611,
        "matched_tickers": 2637, "cross_period_matches": 0,
        "contaminated_observations_in_analysis": 0, "event_id_duplicates": 0,
        "control_ticker_date_duplicates": 0, "control_reuse_rows": 0,
    }
    for key, expected in expected_metrics.items():
        if report.get("metrics", {}).get(key) != expected:
            raise RuntimeError(f"STEP5 V2 metric mismatch: {key}")


def make_samples(root: Path) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    step5 = root / "analysis/step5_control_comparison_v2"
    matched = pd.read_parquet(step5 / "matched_pairs")
    events = pd.read_parquet(root / "analysis/step4_pre_event_common_features/event_bucket_medians")
    controls = pd.read_parquet(step5 / "control_bucket_medians")
    for column in (
        "event_start_date", "control_start_date", "event_window_start_date", "event_target_end_date",
        "control_window_start_date", "control_target_end_date",
    ):
        matched[column] = pd.to_datetime(matched[column], errors="raise")
    if (matched.event_period != matched.control_period).any():
        raise RuntimeError("Cross-period STEP5 V2 match")
    if ((matched.event_start_date >= HOLDOUT_START) | (matched.control_start_date >= HOLDOUT_START)).any():
        raise RuntimeError("Contaminated anchor in STEP5 V2")

    metadata = matched[["event_id", "Ticker", "event_period", "event_start_date", "control_start_date"]]
    events = events.merge(metadata, on="event_id", how="inner", validate="many_to_one")
    controls = controls.merge(metadata, on="event_id", how="inner", validate="many_to_one")
    common = sorted((set(events.columns) & set(controls.columns)) - {"event_id", "bucket"})
    payload = [c for c in common if c not in {"Ticker", "event_period", "event_start_date", "control_start_date"}]
    columns = ["event_id", "bucket", *payload, "Ticker", "event_period", "event_start_date", "control_start_date"]
    event_rows = events[columns].copy()
    control_rows = controls[columns].copy()
    event_rows["observation_type"], event_rows["outcome"] = "event", 1
    control_rows["observation_type"], control_rows["outcome"] = "control", 0
    event_rows["anchor_date"] = event_rows.event_start_date
    control_rows["anchor_date"] = control_rows.control_start_date
    samples = pd.concat([event_rows, control_rows], ignore_index=True).rename(columns={"event_period": "period"})
    samples["pair_year"] = samples.event_start_date.dt.year.astype("int16")
    samples["anchor_year"] = samples.anchor_date.dt.year.astype("int16")
    samples["sample_id"] = samples.event_id.astype(str) + "-" + samples.observation_type + "-" + samples.bucket
    front = [
        "sample_id", "event_id", "Ticker", "period", "bucket", "observation_type", "outcome",
        "event_start_date", "control_start_date", "anchor_date", "pair_year", "anchor_year",
    ]
    samples = samples[front + payload].sort_values(["event_id", "bucket", "observation_type"], kind="stable").reset_index(drop=True)

    formation = samples[(samples.period == "formation") & (samples.bucket == MAIN_BUCKET)]
    eligibility: list[dict[str, Any]] = []
    eligible: list[str] = []
    for column in samples.columns:
        reason = base_exclusion(column, samples[column].dtype)
        valid_count: int | None = None
        unique_count: int | None = None
        if reason is None:
            numeric = pd.to_numeric(formation[column], errors="coerce")
            valid = numeric.notna() & np.isfinite(numeric)
            valid_count = int(valid.sum())
            unique_count = int(numeric[valid].nunique())
            if valid_count == 0:
                reason = "all_missing_in_formation_main_bucket"
            elif valid_count < 20:
                reason = "insufficient_formation_valid_rows"
            elif unique_count < 2:
                reason = "constant_in_formation_main_bucket"
            elif formation.loc[valid, "outcome"].nunique() != 2:
                reason = "one_class_after_formation_missingness"
        selected = reason is None
        missing = int(samples[column].isna().sum())
        eligibility.append({
            "column": column, "dtype": str(samples[column].dtype), "eligible": selected,
            "exclusion_reason": reason, "missing_count": missing,
            "missing_rate": finite(missing / len(samples)),
            "formation_main_valid_count": valid_count, "formation_main_unique_count": unique_count,
            "missing_reason": "source_structural_or_unavailable; retained without imputation" if selected and missing else None,
        })
        if selected:
            eligible.append(column)
    return samples, eligible, pd.DataFrame(eligibility)


def auc(outcome: pd.Series, values: pd.Series) -> tuple[float | None, int]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = outcome.notna() & numeric.notna() & np.isfinite(numeric)
    if valid.sum() < 20 or outcome[valid].nunique() != 2 or numeric[valid].nunique() < 2:
        return None, int(valid.sum())
    return float(roc_auc_score(outcome[valid], numeric[valid])), int(valid.sum())


def oriented(raw: float | None, direction: str | None) -> float | None:
    if raw is None or direction is None:
        return None
    return raw if direction == "high" else 1.0 - raw


def effects(frame: pd.DataFrame, feature: str, direction: str | None) -> dict[str, Any]:
    left = frame[frame.outcome == 1][["event_id", feature]].rename(columns={feature: "event"})
    right = frame[frame.outcome == 0][["event_id", feature]].rename(columns={feature: "control"})
    pair = left.merge(right, on="event_id", validate="one_to_one")
    pair["event"] = pd.to_numeric(pair.event, errors="coerce")
    pair["control"] = pd.to_numeric(pair.control, errors="coerce")
    pair = pair[pair.event.notna() & pair.control.notna() & np.isfinite(pair.event) & np.isfinite(pair.control)]
    event = pd.to_numeric(frame.loc[frame.outcome == 1, feature], errors="coerce").dropna()
    control = pd.to_numeric(frame.loc[frame.outcome == 0, feature], errors="coerce").dropna()
    event, control = event[np.isfinite(event)], control[np.isfinite(control)]
    smd = None
    if len(event) > 1 and len(control) > 1:
        pooled = math.sqrt((event.var(ddof=1) + control.var(ddof=1)) / 2)
        smd = finite((event.mean() - control.mean()) / pooled) if pooled else None
    difference = pair.event - pair.control
    pair_sd = difference.std(ddof=1)
    pair_effect = finite(difference.mean() / pair_sd) if len(difference) > 1 and pair_sd else None
    nonzero = difference[difference != 0]
    consistency = None
    if len(nonzero) and direction:
        consistency = finite((nonzero > 0).mean() if direction == "high" else (nonzero < 0).mean())
    event_median = finite(event.median()) if len(event) else None
    control_median = finite(control.median()) if len(control) else None
    return {
        "event_median": event_median, "control_median": control_median,
        "median_difference": finite(event_median - control_median) if event_median is not None and control_median is not None else None,
        "standardized_mean_difference": smd, "paired_valid": int(len(pair)),
        "paired_difference_median": finite(difference.median()) if len(difference) else None,
        "paired_standardized_effect": pair_effect, "paired_direction_consistency": consistency,
    }


def metrics_for_bucket(frame: pd.DataFrame, features: list[str], bucket: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    annual: list[dict[str, Any]] = []
    formation, validation = frame[frame.period == "formation"], frame[frame.period == "validation"]
    for feature in features:
        formation_raw, formation_valid = auc(formation.outcome, formation[feature])
        direction = None if formation_raw is None else ("high" if formation_raw >= 0.5 else "low")
        validation_raw, validation_valid = auc(validation.outcome, validation[feature])
        row: dict[str, Any] = {
            "feature": feature, "bucket": bucket,
            "formation_rows": int(len(formation)), "formation_events": int((formation.outcome == 1).sum()),
            "formation_controls": int((formation.outcome == 0).sum()), "formation_valid_rows": formation_valid,
            "formation_missing": int(formation[feature].isna().sum()),
            "formation_missing_rate": finite(formation[feature].isna().mean()),
            "formation_raw_auc": formation_raw, "formation_direction": direction,
            "formation_fixed_direction_auc": oriented(formation_raw, direction),
            "validation_rows": int(len(validation)), "validation_events": int((validation.outcome == 1).sum()),
            "validation_controls": int((validation.outcome == 0).sum()), "validation_valid_rows": validation_valid,
            "validation_missing": int(validation[feature].isna().sum()),
            "validation_missing_rate": finite(validation[feature].isna().mean()),
            "validation_raw_auc": validation_raw,
            "validation_fixed_direction_auc": oriented(validation_raw, direction),
            "direction_reproduced_in_validation": oriented(validation_raw, direction) >= 0.5 if oriented(validation_raw, direction) is not None else None,
        }
        row.update({f"formation_{key}": value for key, value in effects(formation, feature, direction).items()})
        row.update({f"validation_{key}": value for key, value in effects(validation, feature, direction).items()})
        rows.append(row)
        for year in sorted(frame.pair_year.unique()):
            part = frame[frame.pair_year == year]
            raw, valid = auc(part.outcome, part[feature])
            annual.append({
                "feature": feature, "bucket": bucket, "pair_year": int(year),
                "period": str(part.period.iloc[0]), "rows": int(len(part)),
                "events": int((part.outcome == 1).sum()), "controls": int((part.outcome == 0).sum()),
                "valid_rows": valid, "missing": int(part[feature].isna().sum()),
                "missing_rate": finite(part[feature].isna().mean()), "raw_auc": raw,
                "formation_direction": direction, "fixed_direction_auc": oriented(raw, direction),
                "direction_reproduced": oriented(raw, direction) >= 0.5 if oriented(raw, direction) is not None else None,
                **effects(part, feature, direction),
            })
    return pd.DataFrame(rows), pd.DataFrame(annual)


def quantile_definitions(values: pd.Series) -> list[tuple[str, list[float]]]:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[numeric.notna() & np.isfinite(numeric)]
    if numeric.nunique() < 5:
        return []
    quintiles = np.unique(numeric.quantile([0.2, 0.4, 0.6, 0.8]).to_numpy(float)).tolist()
    coarse = np.unique(numeric.quantile([0.3, 0.7]).to_numpy(float)).tolist()
    result: list[tuple[str, list[float]]] = []
    if len(quintiles) == 4:
        result.append(("quintile_20pct", quintiles))
    if len(coarse) == 2:
        result.append(("coarse_30_70", coarse))
    return result


def assign_bins(values: pd.Series, thresholds: list[float]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = numeric.notna() & np.isfinite(numeric)
    result.loc[valid] = np.searchsorted(np.asarray(thresholds), numeric.loc[valid], side="right") + 1
    return result


def quantile_table(main: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    formation = main[main.period == "formation"]
    rows: list[dict[str, Any]] = []
    for feature in features:
        for scheme, thresholds in quantile_definitions(formation[feature]):
            encoded = json.dumps(thresholds, separators=(",", ":"))
            for period in ("formation", "validation"):
                part = main[main.period == period]
                bins = assign_bins(part[feature], thresholds)
                valid = bins.notna()
                base = finite(part.loc[valid, "outcome"].mean()) if valid.any() else None
                for number in range(1, len(thresholds) + 2):
                    selected = part[bins == number]
                    event_rate = finite(selected.outcome.mean()) if len(selected) else None
                    rows.append({
                        "feature": feature, "scheme": scheme, "period": period, "bin": number,
                        "formation_thresholds_json": encoded,
                        "lower_exclusive": None if number == 1 else thresholds[number - 2],
                        "upper_inclusive": None if number == len(thresholds) + 1 else thresholds[number - 1],
                        "rows": int(len(selected)), "events": int(selected.outcome.sum()),
                        "controls": int(len(selected) - selected.outcome.sum()), "event_rate": event_rate,
                        "base_event_rate": base, "lift": finite(event_rate / base) if event_rate is not None and base else None,
                        "missing": int((~valid).sum()),
                    })
    return pd.DataFrame(rows)


def ranking_table(metrics: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    yearly = annual.groupby("feature", sort=True).direction_reproduced.agg(
        years_evaluable="count", years_reproduced="sum", year_direction_reproduction_rate="mean"
    ).reset_index()
    ranking = metrics.merge(yearly, on="feature", how="left", validate="one_to_one")
    ranking["effect_sign_reproduced"] = np.sign(ranking.formation_standardized_mean_difference) == np.sign(ranking.validation_standardized_mean_difference)
    ranking["reference_only"], ranking["adopted_signal"], ranking["adopted_threshold"] = True, False, False
    ranking["ranking_note"] = "descriptive reference only; not a signal, combination, rule, or adopted threshold"
    ranking = ranking.sort_values(
        ["validation_fixed_direction_auc", "direction_reproduced_in_validation", "effect_sign_reproduced",
         "year_direction_reproduction_rate", "validation_missing_rate", "validation_valid_rows", "feature"],
        ascending=[False, False, False, False, True, False, True], na_position="last",
    ).reset_index(drop=True)
    ranking.insert(0, "reference_rank", np.arange(1, len(ranking) + 1))
    return ranking[[
        "reference_rank", "feature", "formation_direction", "formation_raw_auc", "formation_fixed_direction_auc",
        "validation_raw_auc", "validation_fixed_direction_auc", "direction_reproduced_in_validation",
        "formation_standardized_mean_difference", "validation_standardized_mean_difference", "effect_sign_reproduced",
        "years_evaluable", "years_reproduced", "year_direction_reproduction_rate", "formation_valid_rows",
        "validation_valid_rows", "formation_missing_rate", "validation_missing_rate", "reference_only",
        "adopted_signal", "adopted_threshold", "ranking_note",
    ]]


def compare_old(root: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    old_path = root / "analysis/step6_univariate/univariate_metrics"
    columns = ["feature", "old_formation_oriented_auc", "v2_formation_fixed_auc", "old_validation_oriented_auc", "v2_validation_fixed_auc"]
    if not old_path.exists():
        return pd.DataFrame(columns=columns)
    old = pd.read_parquet(old_path)
    formation = old[old.scope == "formation"][["feature", "auc_oriented", "direction"]].rename(columns={"auc_oriented": "old_formation_oriented_auc", "direction": "old_formation_direction"})
    validation = old[old.scope == "validation"][["feature", "auc_oriented", "direction"]].rename(columns={"auc_oriented": "old_validation_oriented_auc", "direction": "old_validation_direction"})
    prior = formation.merge(validation, on="feature", how="outer", validate="one_to_one")
    current = metrics[["feature", "formation_direction", "formation_fixed_direction_auc", "validation_fixed_direction_auc"]].rename(columns={
        "formation_direction": "v2_formation_direction", "formation_fixed_direction_auc": "v2_formation_fixed_auc",
        "validation_fixed_direction_auc": "v2_validation_fixed_auc",
    })
    result = prior.merge(current, on="feature", how="outer", validate="one_to_one")
    result["direction_changed"] = result.old_formation_direction != result.v2_formation_direction
    result["old_validation_was_independently_oriented"], result["used_for_v2_selection"] = True, False
    return result.sort_values("feature").reset_index(drop=True)


def build_once(root: Path, stage: Path) -> dict[str, Any]:
    output_names = (
        "analysis_samples", "feature_eligibility", "univariate_metrics", "quantile_bins",
        "annual_metrics", "reference_ranking", "period_assignment_audit", "old_vs_v2_comparison",
    )
    for name in output_names:
        (stage / name).mkdir(parents=True, exist_ok=True)

    samples, features, eligibility = make_samples(root)
    metric_frames: list[pd.DataFrame] = []
    annual_frames: list[pd.DataFrame] = []
    for bucket in BUCKETS:
        frame = samples[samples.bucket == bucket]
        bucket_metrics, bucket_annual = metrics_for_bucket(frame, features, bucket)
        metric_frames.append(bucket_metrics)
        annual_frames.append(bucket_annual)
    metrics = pd.concat(metric_frames, ignore_index=True)
    annual = pd.concat(annual_frames, ignore_index=True)
    main = samples[samples.bucket == MAIN_BUCKET]
    main_metrics = metrics[metrics.bucket == MAIN_BUCKET].drop(columns="bucket")
    main_annual = annual[annual.bucket == MAIN_BUCKET].drop(columns="bucket")
    quantiles = quantile_table(main, features)
    ranking = ranking_table(main_metrics, main_annual)
    old_comparison = compare_old(root, main_metrics)

    matches = pd.read_parquet(root / "analysis/step5_control_comparison_v2/matched_pairs")
    for column in (
        "event_start_date", "control_start_date", "event_window_start_date", "event_target_end_date",
        "control_window_start_date", "control_target_end_date",
    ):
        matches[column] = pd.to_datetime(matches[column], errors="raise")
    audit = matches[[
        "event_id", "Ticker", "event_period", "control_period", "event_start_date", "control_start_date",
        "event_window_start_date", "event_target_end_date", "control_window_start_date",
        "control_target_end_date", "boundary_check_pass", "control_reused",
    ]].copy()
    audit["period_equal"] = audit.event_period == audit.control_period
    audit["contaminated_anchor"] = (audit.event_start_date >= HOLDOUT_START) | (audit.control_start_date >= HOLDOUT_START)
    audit["formation_target_crossing"] = (
        ((audit.event_period == "formation") & (audit.event_target_end_date > FORMATION_END))
        | ((audit.control_period == "formation") & (audit.control_target_end_date > FORMATION_END))
    )
    audit["validation_window_crossing"] = (
        ((audit.event_period == "validation") & (audit.event_window_start_date < VALIDATION_START))
        | ((audit.control_period == "validation") & (audit.control_window_start_date < VALIDATION_START))
    )
    audit["validation_target_crossing"] = (
        ((audit.event_period == "validation") & (audit.event_target_end_date > VALIDATION_END))
        | ((audit.control_period == "validation") & (audit.control_target_end_date > VALIDATION_END))
    )

    frames = {
        "analysis_samples": samples, "feature_eligibility": eligibility, "univariate_metrics": metrics,
        "quantile_bins": quantiles, "annual_metrics": annual, "reference_ranking": ranking,
        "period_assignment_audit": audit, "old_vs_v2_comparison": old_comparison,
    }
    for name, frame in frames.items():
        write_parquet(frame, stage / name / "part.parquet")

    pairs = samples.groupby(["event_id", "bucket"]).agg(
        rows=("sample_id", "size"), outcome_sum=("outcome", "sum"), periods=("period", "nunique"),
        observation_types=("observation_type", "nunique"),
    )
    numeric_samples = samples[features].apply(pd.to_numeric, errors="coerce")
    numeric_outputs = pd.concat([
        metrics.select_dtypes(include="number"), quantiles.select_dtypes(include="number"),
        annual.select_dtypes(include="number"), ranking.select_dtypes(include="number"),
    ], axis=1)
    auc_errors = 0
    for frame, names in (
        (metrics, [c for c in metrics if c.endswith("auc")]),
        (annual, ["raw_auc", "fixed_direction_auc"]),
    ):
        for column in names:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            auc_errors += int((~values.between(0, 1)).sum())
    expected_validation = np.where(
        metrics.formation_direction.eq("low"), 1 - metrics.validation_raw_auc, metrics.validation_raw_auc
    )
    direction_mismatches = int((
        metrics.validation_fixed_direction_auc.notna()
        & ~np.isclose(metrics.validation_fixed_direction_auc, expected_validation, equal_nan=True)
    ).sum())
    quantile_mismatches = int(quantiles.groupby(["feature", "scheme"]).formation_thresholds_json.nunique().gt(1).sum())
    key_duplicates = int(
        metrics.duplicated(["feature", "bucket"]).sum()
        + annual.duplicated(["feature", "bucket", "pair_year"]).sum()
        + quantiles.duplicated(["feature", "scheme", "period", "bin"]).sum()
        + ranking.duplicated(["feature"]).sum()
        + eligibility.duplicated(["column"]).sum()
        + audit.duplicated(["event_id"]).sum()
    )
    balances = main.groupby("period").outcome.agg(["sum", "count"])
    period_imbalance = int((balances["sum"] - (balances["count"] - balances["sum"])).abs().sum())
    event_count, control_count = int((main.outcome == 1).sum()), int((main.outcome == 0).sum())
    return {
        "matched_pairs": int(main.event_id.nunique()), "analysis_sample_rows_all_buckets": int(len(samples)),
        "main_bucket_rows": int(len(main)), "event_rows": event_count, "control_rows": control_count,
        "tickers": int(main.Ticker.nunique()), "eligible_features": int(len(features)),
        "excluded_columns": int((~eligibility.eligible).sum()),
        "excluded_by_reason": {str(k): int(v) for k, v in eligibility.loc[~eligibility.eligible, "exclusion_reason"].value_counts().sort_index().items()},
        "formation_rows": int((main.period == "formation").sum()), "validation_rows": int((main.period == "validation").sum()),
        "formation_valid_rows_min": int(main_metrics.formation_valid_rows.min()), "formation_valid_rows_max": int(main_metrics.formation_valid_rows.max()),
        "validation_valid_rows_min": int(main_metrics.validation_valid_rows.min()), "validation_valid_rows_max": int(main_metrics.validation_valid_rows.max()),
        "feature_cells": int(numeric_samples.size), "feature_missing_cells": int(numeric_samples.isna().sum().sum()),
        "feature_missing_rate": finite(numeric_samples.isna().sum().sum() / numeric_samples.size),
        "sample_id_duplicates": int(samples.sample_id.duplicated().sum()),
        "invalid_pair_rows": int(((pairs.rows != 2) | (pairs.outcome_sum != 1) | (pairs.periods != 1) | (pairs.observation_types != 2)).sum()),
        "event_control_imbalance": abs(event_count - control_count), "period_event_control_imbalance": period_imbalance,
        "period_or_boundary_errors": int((~audit.period_equal).sum() + audit.contaminated_anchor.sum() + audit.formation_target_crossing.sum() + audit.validation_window_crossing.sum() + audit.validation_target_crossing.sum()),
        "contaminated_samples": int(((samples.event_start_date >= HOLDOUT_START) | (samples.control_start_date >= HOLDOUT_START)).sum()),
        "forbidden_predictor_columns": [c for c in features if any(token in c.lower() for token in FORBIDDEN)],
        "validation_direction_reestimated": False, "validation_quantiles_reestimated": False,
        "validation_fixed_direction_mismatches": direction_mismatches, "quantile_threshold_mismatches": quantile_mismatches,
        "major_key_duplicates": key_duplicates, "parquet_reread_passed": True, "auc_out_of_range": auc_errors,
        "nonfinite_output_numeric_values": int(np.isinf(numeric_outputs.to_numpy(dtype=float, na_value=np.nan)).sum()),
        "univariate_rows": int(len(metrics)), "univariate_main_bucket_rows": int(len(main_metrics)),
        "annual_rows": int(len(annual)), "quantile_rows": int(len(quantiles)), "ranking_rows": int(len(ranking)),
    }


def step7_prompt() -> str:
    return """目的：
認証済みSTEP6 V2を基準に、STEP7 V2「2〜3条件組み合わせ検証」だけを実行してください。

STEP1〜STEP6 V2を再計算・変更しないでください。旧STEP5〜旧STEP11の成果物は再利用禁止です。今回は粗い2条件・3条件の組み合わせを検証する工程です。エントリー、出口、資金管理、売買ルールは作成・変更しないでください。STEP8以降は実行しないでください。

【認証済み入力】
・data/market_history/analysis/step6_univariate_v2/
・data/market_history/quality/step6_v2_report.json

最初にSTEP6 V2がPASS、分析ペア8,611件、イベント8,611件、対照8,611件、対象2,637銘柄、期間越境0、汚染期間混入0、未来情報混入0、主要キー重複0、形成期方向固定、形成期分位固定、再現性PASSであることを確認してください。一致しなければSTEP7 V2を実行せずFAILにしてください。

【期間】
・形成期：データ開始日〜2023-12-31
・検証期：2024-01-01〜2025-09-07
・汚染済み隔離期間：2025-09-08以降

2025-09-08以降は `contaminated_holdout` として、分析、候補選定、相関計算、条件作成、閾値計算、集計、可視化へ使用しないでください。OOS、形成期、検証期として扱わず、新しい完全未使用OOSの開始日を独断で決定しないでください。

【候補と条件の固定】
主区間 `d-5_to_d-1` だけを使用してください。STEP6 V2の `reference_ranking` は参考情報であり採用済みシグナルではありません。形成期だけを用いて、検証期を見ずに候補特徴量、方向、組み合わせを決定してください。分位境界はSTEP6 V2で形成期から固定した `coarse_30_70` の30%・70%境界だけを使用してください。検証期で特徴量、方向、分位境界、組み合わせを変更してはいけません。

欠損率、有効件数、形成期と検証期の方向一致、効果量再現性、年別再現性を考慮し、強い相関を持つ実質同義特徴量の重複採用を避けてください。候補削減の根拠、除外理由、形成期内の特徴量間相関を保存してください。検証期AUCや検証期リフトを見て組み合わせ候補を追加・削除・反転してはいけません。

形成期で固定した候補について、2条件ANDと3条件ANDを粗く評価してください。細かな閾値探索、最良閾値選択、OR条件、重み最適化、機械学習、検証期への適合は禁止します。各組み合わせについて、形成期・検証期を完全分離して以下を保存してください。

・組み合わせIDと構成特徴量
・各特徴量の形成期固定方向と境界値
・全サンプル数、有効サンプル数、欠損数、欠損率
・条件該当行数、イベント数、対照数
・イベント率、全体イベント率、リフト
・形成期から検証期へのリフト維持率
・年別該当件数、イベント率、リフト、方向再現性
・特徴量間相関

ケース・コントロール比較のイベント率は実市場の無条件発生確率ではないことをレポートへ明記してください。今回は参考組み合わせの記述だけを行い、採用シグナル、売買ルール、エントリー、出口を確定しないでください。

【保存】
旧STEP7を上書きせず、以下へ新規Parquetを保存してください。

data/market_history/analysis/step7_condition_combinations_v2/

最低限：candidate_features/、feature_correlations/、combination_definitions/、combination_metrics/、annual_metrics/、reference_ranking/、period_assignment_audit/、old_vs_v2_comparison/

品質レポート：
data/market_history/quality/step7_v2_report.json
data/market_history/quality/STEP7_V2_REPORT.md

【品質検査】
入力ファイル前後SHA256、主要キー、イベント・対照1対1、期間分離、2025-09-08以降の不使用、未来列不使用、形成期だけの候補作成、形成期固定方向、形成期固定分位、検証期による候補変更の不存在、欠損、無限大、件数整合、出力Parquet再読込、同じ入力による2回実行のハッシュ一致を検査してください。

期間越境、汚染期間使用、未来情報混入、検証期による候補・方向・境界・組み合わせ変更、イベント対照不均衡、入力変更、主要キー重複、再現性不一致が1件でもあればFAILにしてください。欠損を無条件削除・補完せず、組み合わせ別の有効行で計算し、欠損数と理由を保存してください。

【完了報告】
STEP7 V2 完了 / FAIL、期間、入力ペア数、候補特徴量数と除外理由、2条件・3条件の評価数、形成期・検証期の有効件数、欠損率、期間越境、未来情報、重複、作成ファイル、保存先、容量、入力変更、再現性、品質、残存リスクを報告してください。

PASSの場合だけ「STEP7 V2で作成した成果物は、同じ入力版では今後再計算不要」と明記してください。

処理完了後、回答の最後にSTEP8 V2「相場環境別検証」だけを実行する完全な次回用プロンプトを表示し、同じ内容を `data/market_history/quality/STEP8_V2_PROMPT.md` へ保存してください。STEP9以降を実行させないでください。今後も各STEP終了時に次のSTEPだけの完全なプロンプトを回答の最後へ表示し、品質フォルダへ保存してください。
"""


def report_markdown(report: dict[str, Any]) -> str:
    m = report["metrics"]
    return f"""# STEP6 V2 品質レポート

判定：**{report['quality']}**

- 形成期：データ開始日〜2023-12-31
- 検証期：2024-01-01〜2025-09-07
- 汚染済み隔離期間：2025-09-08以降
- 分析ペア：{m['matched_pairs']:,}
- 主区間サンプル：{m['main_bucket_rows']:,}（イベント{m['event_rows']:,}、対照{m['control_rows']:,}）
- 対象銘柄：{m['tickers']:,}
- 評価特徴量：{m['eligible_features']:,}
- 除外列：{m['excluded_columns']:,}
- 形成期サンプル：{m['formation_rows']:,}（特徴量別有効{m['formation_valid_rows_min']:,}〜{m['formation_valid_rows_max']:,}）
- 検証期サンプル：{m['validation_rows']:,}（特徴量別有効{m['validation_valid_rows_min']:,}〜{m['validation_valid_rows_max']:,}）
- 評価特徴量の全区間欠損率：{m['feature_missing_rate']:.6%}
- 期間・境界エラー：{m['period_or_boundary_errors']:,}
- 汚染期間サンプル：{m['contaminated_samples']:,}
- 未来・目的変数の説明変数混入：{len(m['forbidden_predictor_columns']):,}
- sample_id重複：{m['sample_id_duplicates']:,}
- 不正な1対1ペア：{m['invalid_pair_rows']:,}
- イベント・対照不均衡：{m['event_control_imbalance']:,}
- 期間別イベント・対照不均衡：{m['period_event_control_imbalance']:,}
- 主要キー重複：{m['major_key_duplicates']:,}
- 入力変更：{not report['inputs_unchanged']}
- 2回実行のParquet一致：{report['reproducibility_passed']}
- 保存容量：{report['file_size_bytes']:,} bytes

主順位と分位分析は `d-5_to_d-1` だけで作成し、ほかの3区間は補助時系列確認用に保存しました。方向は形成期だけで決めて検証期と年別評価へ固定し、分位境界も形成期だけで計算しました。参考順位は採用シグナル、組み合わせ、売買ルール、採用閾値ではありません。ケース・コントロール上のイベント率は実市場の無条件発生確率ではありません。

旧STEP6〜旧STEP11は再利用禁止です。2025-09-08以降は完全未使用OOSへ戻していません。

{report['completion_statement']}
"""


def main() -> int:
    options = args()
    repository = Path(__file__).resolve().parents[2]
    root = (repository / "momentum5d" / options.root).resolve()
    step5_report_path = root / "quality/step5_v2_report.json"
    if not step5_report_path.exists():
        raise RuntimeError(f"STEP5 V2 report missing: {step5_report_path}")
    step5_report = json.loads(step5_report_path.read_text(encoding="utf-8"))
    validate_step5(step5_report)
    groups = [
        ("step5_v2", root / "analysis/step5_control_comparison_v2"),
        ("step5_v2_report", step5_report_path),
        ("step4_matched_event_values", root / "analysis/step4_pre_event_common_features/event_bucket_medians"),
    ]
    old_step6 = root / "analysis/step6_univariate"
    if old_step6.exists():
        groups.append(("old_step6_audit_only", old_step6))
    before = manifest(groups)
    target = root / "analysis/step6_univariate_v2"
    quality = root / "quality"
    if target.exists() or (quality / "step6_v2_report.json").exists():
        raise RuntimeError("STEP6 V2 immutable output already exists")

    temporary = Path(tempfile.mkdtemp(prefix="step6_v2_", dir=root.parent))
    first = temporary / "first"
    metrics = build_once(root, first)
    first_hashes = parquet_manifest(first)
    second_hashes: dict[str, str] | None = None
    reproducible = False
    if options.verify_reproducibility:
        gc.collect()
        second = temporary / "second"
        second_metrics = build_once(root, second)
        second_hashes = parquet_manifest(second)
        reproducible = first_hashes == second_hashes and metrics == second_metrics
    inputs_unchanged = before == manifest(groups)
    hard_failures = {
        key: metrics[key]
        for key in (
            "sample_id_duplicates", "invalid_pair_rows", "event_control_imbalance",
            "period_event_control_imbalance", "period_or_boundary_errors", "contaminated_samples",
            "validation_fixed_direction_mismatches", "quantile_threshold_mismatches",
            "major_key_duplicates", "auc_out_of_range", "nonfinite_output_numeric_values",
        )
    }
    hard_failures["forbidden_predictor_columns"] = len(metrics["forbidden_predictor_columns"])
    passed = (
        all(value == 0 for value in hard_failures.values()) and inputs_unchanged
        and options.verify_reproducibility and reproducible
        and not metrics["validation_direction_reestimated"]
        and not metrics["validation_quantiles_reestimated"]
        and metrics["parquet_reread_passed"] and metrics["matched_pairs"] == 8611
        and metrics["event_rows"] == metrics["control_rows"] == 8611
        and metrics["tickers"] == 2637 and metrics["eligible_features"] > 0
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    first.rename(target)
    report = {
        "step": 6, "version": 2, "status": "STEP6 V2 complete" if passed else "STEP6 V2 FAIL",
        "quality": "PASS" if passed else "FAIL", "created_at_jst": datetime.now(JST).isoformat(),
        "formation": {"start": "data_start", "end": "2023-12-31"},
        "validation": {"start": "2024-01-01", "end": "2025-09-07"},
        "contaminated_holdout_start": "2025-09-08", "new_untouched_oos_start": None,
        "steps1_to_5_v2_recalculated": False, "old_step6_used_for_v2_analysis": False,
        "strategy_or_signal_created": False, "combination_or_threshold_adopted": False,
        "direction_source": "formation only", "validation_direction_reestimated": False,
        "quantile_boundary_source": "formation only", "validation_quantiles_reestimated": False,
        "input_manifest_before": before, "input_manifest_after": manifest(groups),
        "inputs_unchanged": inputs_unchanged, "reproducibility_requested": options.verify_reproducibility,
        "reproducibility_passed": reproducible, "first_output_manifest": first_hashes,
        "second_output_manifest": second_hashes, "hard_failures": hard_failures, "metrics": metrics,
        "save_path": str(target), "file_size_bytes": size(target),
        "completion_statement": "STEP6 V2で作成した成果物は、同じ入力版では今後再計算不要" if passed else "STEP6 V2はFAILのため再計算不要とは認証しない",
        "residual_risks": [
            "case-control event rates are not unconditional market probabilities",
            "multiple testing remains substantial and reference ranking is not adoption",
            "repeated tickers and shared market regimes reduce effective sample independence",
            "sector features retain structural missingness and reduced sample sizes",
            "2025-09-08 onward remains contaminated and cannot be reused as untouched OOS",
        ],
    }
    write_json(quality / "step6_v2_report.json", report)
    (quality / "STEP6_V2_REPORT.md").write_text(report_markdown(report), encoding="utf-8")
    (quality / "STEP7_V2_PROMPT.md").write_text(step7_prompt(), encoding="utf-8")
    shutil.rmtree(temporary, ignore_errors=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
