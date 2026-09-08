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

JST = ZoneInfo("Asia/Tokyo")
FORMATION_END = pd.Timestamp("2023-12-31")
VALIDATION_START = pd.Timestamp("2024-01-01")
VALIDATION_END = pd.Timestamp("2025-09-07")
HOLDOUT_START = pd.Timestamp("2025-09-08")
MAIN_BUCKET = "d-5_to_d-1"
FORBIDDEN = ("future", "forward", "target", "label", "mfe", "mae", "peak_date", "event_end", "outcome")

REGIME_SPECS = (
    ("nikkei225", "index_nikkei225_change_20d", "weak", "neutral", "strong", "弱い", "中立", "強い"),
    ("sp500", "external_sp500_change_20d", "weak", "neutral", "strong", "弱い", "中立", "強い"),
    ("nasdaq_composite", "external_nasdaq_composite_change_20d", "weak", "neutral", "strong", "弱い", "中立", "強い"),
    ("sox", "external_sox_change_20d", "weak", "neutral", "strong", "弱い", "中立", "強い"),
    ("vix", "external_vix_vs_ma20", "low", "neutral", "high", "低い", "中立", "高い"),
    ("usdjpy", "external_usd_jpy_change_20d", "yen_appreciation", "neutral", "yen_depreciation", "円高方向", "中立", "円安方向"),
    ("us10y", "external_us10y_change_20d", "falling", "neutral", "rising", "低下", "中立", "上昇"),
    ("market_internals", "market_advancer_ratio", "weak", "neutral", "strong", "弱い", "中立", "強い"),
    ("market_trading_value", "market_trading_value_change_1d", "decreasing", "neutral", "increasing", "減少", "中立", "増加"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build isolated STEP8 V2 regime analysis")
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
    expected = {
        "quality": "PASS",
        "version": 2,
        "inputs_unchanged": True,
        "reproducibility_passed": True,
        "candidate_selection_source": "formation_only",
        "correlation_source": "formation_only",
        "threshold_source": "STEP6 V2 formation coarse_30_70",
        "validation_used_after_definition_freeze_only": True,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(f"STEP7 V2 precondition mismatch: {key}")
    expected_metrics = {
        "input_pairs": 8611,
        "input_events": 8611,
        "input_controls": 8611,
        "input_tickers": 2637,
        "selected_candidate_features": 24,
        "selected_strong_correlation_pairs": 0,
        "total_combinations": 2300,
        "period_or_boundary_errors": 0,
        "contaminated_samples": 0,
        "major_key_duplicates": 0,
        "validation_used_for_candidate_selection": False,
        "validation_used_for_correlation": False,
        "validation_used_for_definition": False,
        "validation_thresholds_reestimated": False,
    }
    metrics = report.get("metrics", {})
    for key, value in expected_metrics.items():
        if metrics.get(key) != value:
            raise RuntimeError(f"STEP7 V2 metric mismatch: {key}")
    if metrics.get("forbidden_candidate_features") != []:
        raise RuntimeError("STEP7 V2 contains forbidden candidate features")


def load_formation_thresholds(quantiles: pd.DataFrame) -> dict[str, tuple[float, float]]:
    formation = quantiles[(quantiles.scheme == "coarse_30_70") & (quantiles.period == "formation")]
    result: dict[str, tuple[float, float]] = {}
    for _, row in formation.sort_values(["feature", "bin"], kind="stable").drop_duplicates("feature").iterrows():
        values = json.loads(row.formation_thresholds_json)
        if len(values) != 2 or finite(values[0]) is None or finite(values[1]) is None:
            continue
        result[str(row.feature)] = (float(values[0]), float(values[1]))
    return result


def make_regime_definitions(thresholds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable, feature, low, middle, high, low_ja, middle_ja, high_ja in REGIME_SPECS:
        if feature not in thresholds:
            raise RuntimeError(f"Missing formation coarse_30_70 boundary: {feature}")
        q30, q70 = thresholds[feature]
        rows.append({
            "environment_variable": variable,
            "source_feature": feature,
            "q30": q30,
            "q70": q70,
            "low_regime": low,
            "middle_regime": middle,
            "high_regime": high,
            "low_regime_ja": low_ja,
            "middle_regime_ja": middle_ja,
            "high_regime_ja": high_ja,
            "unknown_regime": "unknown",
            "boundary_source": "STEP6 V2 formation coarse_30_70",
            "boundary_calculation_period": "formation_only",
            "validation_used_for_boundary": False,
            "combined_regime_definition": False,
            "optimized_for_combination_performance": False,
        })
    return pd.DataFrame(rows)


def classify(values: pd.Series, q30: float, q70: float, labels: tuple[str, str, str]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna() & np.isfinite(numeric)
    result = pd.Series("unknown", index=values.index, dtype="object")
    result.loc[valid & (numeric <= q30)] = labels[0]
    result.loc[valid & (numeric > q30) & (numeric < q70)] = labels[1]
    result.loc[valid & (numeric >= q70)] = labels[2]
    return result


def make_sample_regimes(samples: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    meta = ["sample_id", "event_id", "Ticker", "period", "observation_type", "outcome", "anchor_date", "pair_year"]
    label_ja: dict[tuple[str, str], str] = {}
    rows: list[pd.DataFrame] = []
    for definition in definitions.itertuples(index=False):
        labels = (definition.low_regime, definition.middle_regime, definition.high_regime)
        for english, japanese in zip(labels, (definition.low_regime_ja, definition.middle_regime_ja, definition.high_regime_ja), strict=True):
            label_ja[(definition.environment_variable, english)] = japanese
        label_ja[(definition.environment_variable, "unknown")] = "不明"
        frame = samples[meta].copy()
        frame["environment_variable"] = definition.environment_variable
        frame["source_feature"] = definition.source_feature
        frame["environment_value"] = pd.to_numeric(samples[definition.source_feature], errors="coerce")
        frame["regime"] = classify(frame.environment_value, definition.q30, definition.q70, labels)
        frame["regime_ja"] = frame.regime.map(lambda value: label_ja[(definition.environment_variable, value)])
        frame["q30"] = definition.q30
        frame["q70"] = definition.q70
        frame["boundary_source"] = definition.boundary_source
        frame["boundary_calculation_period"] = "formation_only"
        frame["validation_used_for_boundary"] = False
        frame["missing_reason"] = np.where(frame.regime.eq("unknown"), "environment_feature_missing_or_nonfinite; retained without imputation", None)
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(["environment_variable", "period", "sample_id"], kind="stable").reset_index(drop=True)


def definition_conditions(row: Any) -> list[tuple[str, str, float]]:
    result: list[tuple[str, str, float]] = []
    for index in range(1, int(row.condition_count) + 1):
        result.append((
            str(getattr(row, f"feature_{index}")),
            str(getattr(row, f"direction_{index}")),
            float(getattr(row, f"threshold_{index}")),
        ))
    return result


def condition_masks(samples: pd.DataFrame, definitions: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    valid_masks: dict[str, np.ndarray] = {}
    selected_masks: dict[str, np.ndarray] = {}
    numeric_cache: dict[str, np.ndarray] = {}
    for feature in sorted(set(definitions.feature_1) | set(definitions.feature_2) | set(definitions.feature_3.dropna())):
        numeric_cache[feature] = pd.to_numeric(samples[feature], errors="coerce").to_numpy(dtype=float)
    for definition in definitions.itertuples(index=False):
        valid = np.ones(len(samples), dtype=bool)
        selected = np.ones(len(samples), dtype=bool)
        for feature, direction, threshold in definition_conditions(definition):
            values = numeric_cache[feature]
            feature_valid = np.isfinite(values)
            valid &= feature_valid
            selected &= feature_valid & ((values >= threshold) if direction == "high" else (values <= threshold))
        valid_masks[definition.combination_id] = valid
        selected_masks[definition.combination_id] = selected
    return valid_masks, selected_masks


def summarize_mask(
    outcomes: np.ndarray,
    group: np.ndarray,
    valid: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    valid_group = group & valid
    selected_group = group & selected
    total_rows = int(group.sum())
    valid_rows = int(valid_group.sum())
    condition_rows = int(selected_group.sum())
    condition_events = int(outcomes[selected_group].sum())
    base_events = int(outcomes[valid_group].sum())
    event_rate = finite(condition_events / condition_rows) if condition_rows else None
    base_rate = finite(base_events / valid_rows) if valid_rows else None
    return {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "missing_rows": total_rows - valid_rows,
        "missing_rate": finite((total_rows - valid_rows) / total_rows) if total_rows else None,
        "condition_rows": condition_rows,
        "condition_events": condition_events,
        "condition_controls": condition_rows - condition_events,
        "event_rate": event_rate,
        "base_event_rate": base_rate,
        "lift": finite(event_rate / base_rate) if event_rate is not None and base_rate else None,
        "missing_reason": "one_or_more_component_features_missing; retained without imputation" if total_rows > valid_rows else None,
    }


def evaluate(
    samples: pd.DataFrame,
    sample_regimes: pd.DataFrame,
    definitions: pd.DataFrame,
    valid_masks: dict[str, np.ndarray],
    selected_masks: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = samples.outcome.to_numpy(dtype=np.int8)
    sample_order = pd.Series(np.arange(len(samples)), index=samples.sample_id)
    regimes = sample_regimes.copy()
    regimes["sample_position"] = regimes.sample_id.map(sample_order)
    if regimes.sample_position.isna().any():
        raise RuntimeError("Sample-regime mapping failed")
    regimes = regimes.sort_values("sample_position", kind="stable")
    masks: dict[tuple[str, str, str], np.ndarray] = {}
    annual_masks: dict[tuple[str, str, int], np.ndarray] = {}
    regime_info: dict[tuple[str, str], tuple[str, float, float]] = {}
    for env, part in regimes.groupby("environment_variable", sort=True):
        ordered = part.sort_values("sample_position", kind="stable")
        if not np.array_equal(ordered.sample_position.to_numpy(), np.arange(len(samples))):
            raise RuntimeError(f"Incomplete regime sample coverage: {env}")
        labels = ordered.regime.to_numpy(dtype=object)
        periods = ordered.period.to_numpy(dtype=object)
        years = ordered.pair_year.to_numpy(dtype=int)
        source = str(ordered.source_feature.iloc[0])
        q30 = float(ordered.q30.iloc[0])
        q70 = float(ordered.q70.iloc[0])
        for label in sorted(set(labels)):
            regime_info[(env, str(label))] = (source, q30, q70)
            for period in ("formation", "validation"):
                masks[(env, str(label), period)] = (labels == label) & (periods == period)
            for year in sorted(set(years)):
                annual_masks[(env, str(label), int(year))] = (labels == label) & (years == year)

    rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    for definition in definitions.itertuples(index=False):
        valid = valid_masks[definition.combination_id]
        selected = selected_masks[definition.combination_id]
        for (env, label), (source, q30, q70) in sorted(regime_info.items()):
            for period in ("formation", "validation"):
                rows.append({
                    "combination_id": definition.combination_id,
                    "condition_count": int(definition.condition_count),
                    "environment_variable": env,
                    "source_feature": source,
                    "regime": label,
                    "period": period,
                    "q30": q30,
                    "q70": q70,
                    "boundary_source": "STEP6 V2 formation coarse_30_70",
                    "boundary_calculation_period": "formation_only",
                    "validation_used_for_boundary": False,
                    **summarize_mask(outcomes, masks[(env, label, period)], valid, selected),
                    "combination_definition_frozen_from_step7": True,
                })
            for year in sorted(samples.pair_year.unique()):
                evaluated = summarize_mask(outcomes, annual_masks[(env, label, int(year))], valid, selected)
                annual_rows.append({
                    "combination_id": definition.combination_id,
                    "condition_count": int(definition.condition_count),
                    "environment_variable": env,
                    "source_feature": source,
                    "regime": label,
                    "pair_year": int(year),
                    "period": "formation" if int(year) <= FORMATION_END.year else "validation",
                    "q30": q30,
                    "q70": q70,
                    **evaluated,
                    "direction_reproduced": evaluated["lift"] >= 1.0 if evaluated["lift"] is not None else None,
                    "combination_definition_frozen_from_step7": True,
                    "validation_used_for_boundary": False,
                })
    metrics = pd.DataFrame(rows)
    annual = pd.DataFrame(annual_rows)

    known = metrics[metrics.regime != "unknown"]
    spread = known.groupby(["combination_id", "environment_variable", "period"], sort=True).lift.agg(
        minimum_regime_lift="min", maximum_regime_lift="max"
    ).reset_index()
    spread["max_lift_difference_across_regimes"] = spread.maximum_regime_lift - spread.minimum_regime_lift
    metrics = metrics.merge(spread, on=["combination_id", "environment_variable", "period"], how="left", validate="many_to_one")
    pivot = metrics.pivot_table(
        index=["combination_id", "environment_variable", "regime"], columns="period", values="lift", aggfunc="first"
    ).reset_index()
    formation_lift = pd.to_numeric(pivot.get("formation"), errors="coerce")
    validation_lift = pd.to_numeric(pivot.get("validation"), errors="coerce")
    pivot["formation_to_validation_lift_retention"] = np.where(
        formation_lift.notna() & validation_lift.notna() & (formation_lift != 0),
        validation_lift / formation_lift,
        np.nan,
    )
    pivot["lift_retention_missing_reason"] = np.where(
        formation_lift.isna(),
        "formation_lift_not_evaluable",
        np.where(
            formation_lift == 0,
            "formation_lift_zero; ratio undefined",
            np.where(validation_lift.isna(), "validation_lift_not_evaluable", None),
        ),
    )
    metrics = metrics.merge(
        pivot[["combination_id", "environment_variable", "regime", "formation_to_validation_lift_retention", "lift_retention_missing_reason"]],
        on=["combination_id", "environment_variable", "regime"], how="left", validate="many_to_one",
    )
    metrics["case_control_rate_not_market_probability"] = True
    return (
        metrics.sort_values(["combination_id", "environment_variable", "regime", "period"], kind="stable").reset_index(drop=True),
        annual.sort_values(["combination_id", "environment_variable", "regime", "pair_year"], kind="stable").reset_index(drop=True),
    )


def make_stability(metrics: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    known = metrics[metrics.regime != "unknown"]
    formation = known[known.period == "formation"].groupby(["combination_id", "condition_count", "environment_variable"], sort=True).agg(
        formation_regimes_evaluable=("lift", "count"),
        formation_min_lift=("lift", "min"),
        formation_median_lift=("lift", "median"),
        formation_max_lift=("lift", "max"),
        formation_max_lift_difference=("max_lift_difference_across_regimes", "first"),
        formation_condition_rows=("condition_rows", "sum"),
    ).reset_index()
    validation = known[known.period == "validation"].groupby(["combination_id", "condition_count", "environment_variable"], sort=True).agg(
        validation_regimes_evaluable=("lift", "count"),
        validation_min_lift=("lift", "min"),
        validation_median_lift=("lift", "median"),
        validation_max_lift=("lift", "max"),
        validation_max_lift_difference=("max_lift_difference_across_regimes", "first"),
        validation_condition_rows=("condition_rows", "sum"),
        validation_direction_reproduction_rate=("lift", lambda values: float((values >= 1.0).mean()) if len(values) else np.nan),
    ).reset_index()
    retention = known.groupby(["combination_id", "condition_count", "environment_variable"], sort=True).formation_to_validation_lift_retention.agg(
        regimes_with_retention="count",
        minimum_lift_retention="min",
        median_lift_retention="median",
        maximum_lift_retention="max",
    ).reset_index()
    yearly = annual[(annual.regime != "unknown") & annual.direction_reproduced.notna()].groupby(
        ["combination_id", "condition_count", "environment_variable"], sort=True
    ).direction_reproduced.agg(year_regime_cells_evaluable="count", year_regime_cells_reproduced="sum", year_regime_direction_reproduction_rate="mean").reset_index()
    result = formation.merge(validation, on=["combination_id", "condition_count", "environment_variable"], validate="one_to_one")
    result = result.merge(retention, on=["combination_id", "condition_count", "environment_variable"], validate="one_to_one")
    result = result.merge(yearly, on=["combination_id", "condition_count", "environment_variable"], how="left", validate="one_to_one")
    result["reference_only"] = True
    result["adopted_signal"] = False
    result["adopted_regime_filter"] = False
    result["case_control_rate_not_market_probability"] = True
    return result.sort_values(["combination_id", "environment_variable"], kind="stable").reset_index(drop=True)


def make_reference_summary(stability: pd.DataFrame) -> pd.DataFrame:
    result = stability.groupby(["combination_id", "condition_count"], sort=True).agg(
        environment_variables=("environment_variable", "nunique"),
        formation_worst_environment_lift=("formation_min_lift", "min"),
        formation_median_environment_lift=("formation_median_lift", "median"),
        formation_largest_environment_spread=("formation_max_lift_difference", "max"),
        validation_worst_environment_lift=("validation_min_lift", "min"),
        validation_median_environment_lift=("validation_median_lift", "median"),
        validation_largest_environment_spread=("validation_max_lift_difference", "max"),
        median_lift_retention=("median_lift_retention", "median"),
        worst_lift_retention=("minimum_lift_retention", "min"),
        validation_regime_direction_reproduction_rate=("validation_direction_reproduction_rate", "mean"),
        year_regime_direction_reproduction_rate=("year_regime_direction_reproduction_rate", "mean"),
    ).reset_index()
    result = result.sort_values(
        ["formation_worst_environment_lift", "formation_median_environment_lift", "combination_id"],
        ascending=[False, False, True], kind="stable", na_position="last",
    ).reset_index(drop=True)
    result.insert(0, "formation_only_reference_rank", np.arange(1, len(result) + 1))
    result["ranking_source"] = "formation regime results only; validation appended after freeze"
    result["reference_only"] = True
    result["adopted_signal"] = False
    result["adopted_regime_filter"] = False
    result["adopted_trading_rule"] = False
    return result


def make_audit(root: Path, samples: pd.DataFrame) -> pd.DataFrame:
    audit = pd.read_parquet(root / "analysis/step7_condition_combinations_v2/period_assignment_audit")
    audit = audit.copy()
    audit["step8_used_main_bucket_only"] = True
    audit["step8_regime_boundary_period"] = "formation_only"
    audit["step8_validation_boundary_reestimated"] = False
    audit["step8_combination_definition_changed"] = False
    audit["step8_contaminated_holdout_used"] = False
    audit["step8_sample_rows"] = audit.event_id.map(samples.groupby("event_id").size()).astype("Int64")
    return audit.sort_values("event_id", kind="stable").reset_index(drop=True)


def old_comparison(root: Path, metrics: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    old = root / "analysis/step8_regime_analysis"
    if not old.exists():
        return pd.DataFrame([{
            "metric": "old_step8_available", "old_value": 0, "v2_value": 1,
            "used_for_v2_definition_or_analysis": False, "note": "old STEP8 unavailable and not required",
        }])
    old_files = list(old.rglob("*.parquet"))
    old_rows = sum(len(pd.read_parquet(path)) for path in old_files)
    return pd.DataFrame([
        {"metric": "old_step8_parquet_rows", "old_value": old_rows, "v2_value": len(metrics), "used_for_v2_definition_or_analysis": False, "note": "old result invalidated; audit comparison only"},
        {"metric": "combination_count", "old_value": None, "v2_value": definitions.combination_id.nunique(), "used_for_v2_definition_or_analysis": False, "note": "old result not used to select or change combinations"},
    ])


def numeric_infinities(frames: list[pd.DataFrame]) -> int:
    total = 0
    for frame in frames:
        numeric = frame.select_dtypes(include=[np.number])
        if not numeric.empty:
            total += int(np.isinf(numeric.to_numpy(dtype=float)).sum())
    return total


def environment_summary(sample_regimes: pd.DataFrame, metrics: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for env in sorted(sample_regimes.environment_variable.unique()):
        for period in ("formation", "validation"):
            samples = sample_regimes[(sample_regimes.environment_variable == env) & (sample_regimes.period == period)]
            cells = metrics[(metrics.environment_variable == env) & (metrics.period == period) & (metrics.regime != "unknown")]
            lifts = cells.lift.dropna()
            rows.append({
                "environment_variable": env,
                "period": period,
                "sample_rows": int(len(samples)),
                "valid_environment_rows": int((samples.regime != "unknown").sum()),
                "unknown_rows": int((samples.regime == "unknown").sum()),
                "combination_regime_cells": int(len(cells)),
                "evaluable_lift_cells": int(len(lifts)),
                "median_lift": finite(lifts.median()) if len(lifts) else None,
                "lift_q25": finite(lifts.quantile(0.25)) if len(lifts) else None,
                "lift_q75": finite(lifts.quantile(0.75)) if len(lifts) else None,
                "cells_lift_at_least_one": int((lifts >= 1.0).sum()),
                "median_lift_retention": finite(cells.formation_to_validation_lift_retention.median()) if period == "validation" else None,
            })
    return rows


def build_once(root: Path, stage: Path) -> dict[str, Any]:
    names = (
        "regime_definitions", "sample_regimes", "combination_regime_metrics", "annual_regime_metrics",
        "regime_stability", "reference_summary", "period_assignment_audit", "old_vs_v2_comparison",
    )
    for name in names:
        (stage / name).mkdir(parents=True, exist_ok=True)

    step7 = root / "analysis/step7_condition_combinations_v2"
    definitions = pd.read_parquet(step7 / "combination_definitions").sort_values("combination_id", kind="stable").reset_index(drop=True)
    candidates = pd.read_parquet(step7 / "candidate_features")
    samples = pd.read_parquet(root / "analysis/step6_univariate_v2/analysis_samples")
    samples = samples[samples.bucket == MAIN_BUCKET].sort_values("sample_id", kind="stable").reset_index(drop=True)
    quantiles = pd.read_parquet(root / "analysis/step6_univariate_v2/quantile_bins")

    thresholds = load_formation_thresholds(quantiles)
    regime_definitions = make_regime_definitions(thresholds)
    sample_regimes = make_sample_regimes(samples, regime_definitions)
    valid_masks, selected_masks = condition_masks(samples, definitions)
    metrics, annual = evaluate(samples, sample_regimes, definitions, valid_masks, selected_masks)
    stability = make_stability(metrics, annual)
    reference = make_reference_summary(stability)
    audit = make_audit(root, samples)
    old = old_comparison(root, metrics, definitions)

    frames = {
        "regime_definitions": regime_definitions,
        "sample_regimes": sample_regimes,
        "combination_regime_metrics": metrics,
        "annual_regime_metrics": annual,
        "regime_stability": stability,
        "reference_summary": reference,
        "period_assignment_audit": audit,
        "old_vs_v2_comparison": old,
    }
    for name, frame in frames.items():
        write_parquet(frame, stage / name / "part.parquet")

    selected_candidates = candidates[candidates.selected_candidate]
    definition_features = set(definitions.feature_1) | set(definitions.feature_2) | set(definitions.feature_3.dropna())
    candidate_features = set(selected_candidates.feature)
    combination_definition_mismatches = int(definition_features != candidate_features)
    candidate_lookup = selected_candidates.set_index("feature")
    definition_field_mismatches = 0
    for definition in definitions.itertuples(index=False):
        if definition.logical_operator != "AND" or int(definition.condition_count) not in (2, 3):
            definition_field_mismatches += 1
        if definition.candidate_source != "formation_only" or bool(definition.validation_used_for_definition):
            definition_field_mismatches += 1
        for position, (feature, direction, threshold) in enumerate(definition_conditions(definition), start=1):
            candidate = candidate_lookup.loc[feature]
            if direction != candidate.formation_direction or not np.isclose(threshold, float(candidate.condition_threshold), rtol=0, atol=1e-15):
                definition_field_mismatches += 1
            expected_operator = ">=" if direction == "high" else "<="
            if getattr(definition, f"operator_{position}") != expected_operator:
                definition_field_mismatches += 1
    forbidden_features = sorted(feature for feature in definition_features | set(regime_definitions.source_feature) if any(token in feature.lower() for token in FORBIDDEN))
    periods = set(samples.period)
    period_dates = pd.to_datetime(samples.anchor_date)
    period_errors = int(((samples.period == "formation") & (period_dates > FORMATION_END)).sum())
    period_errors += int(((samples.period == "validation") & ((period_dates < VALIDATION_START) | (period_dates > VALIDATION_END))).sum())
    contaminated = int((period_dates >= HOLDOUT_START).sum())
    input_pair_counts = samples.groupby(["event_id", "observation_type"], sort=True).size().unstack(fill_value=0)
    invalid_pairs = int(((input_pair_counts.get("event", 0) != 1) | (input_pair_counts.get("control", 0) != 1)).sum())
    event_control_imbalance = abs(int((samples.outcome == 1).sum()) - int((samples.outcome == 0).sum()))
    period_imbalance = int(samples.groupby("period").outcome.agg(lambda values: abs(int((values == 1).sum()) - int((values == 0).sum()))).sum())
    regime_sample_imbalance = int(sample_regimes.groupby(["environment_variable", "period"]).outcome.agg(lambda values: abs(int((values == 1).sum()) - int((values == 0).sum()))).sum())
    major_duplicates = (
        int(regime_definitions.duplicated("environment_variable").sum())
        + int(sample_regimes.duplicated(["sample_id", "environment_variable"]).sum())
        + int(metrics.duplicated(["combination_id", "environment_variable", "regime", "period"]).sum())
        + int(annual.duplicated(["combination_id", "environment_variable", "regime", "pair_year"]).sum())
        + int(stability.duplicated(["combination_id", "environment_variable"]).sum())
        + int(reference.duplicated("combination_id").sum())
        + int(audit.duplicated("event_id").sum())
    )
    expected_definition_hash = sha256(step7 / "combination_definitions/part.parquet")
    unknown = sample_regimes[sample_regimes.regime == "unknown"]
    env_counts = sample_regimes.groupby(["environment_variable", "period"], sort=True).agg(
        total_rows=("sample_id", "size"), valid_rows=("regime", lambda values: int((values != "unknown").sum())), unknown_rows=("regime", lambda values: int((values == "unknown").sum()))
    ).reset_index()
    return {
        "input_pairs": int(samples.event_id.nunique()),
        "input_rows": int(len(samples)),
        "input_events": int((samples.outcome == 1).sum()),
        "input_controls": int((samples.outcome == 0).sum()),
        "input_tickers": int(samples.Ticker.nunique()),
        "fixed_candidate_features": int(selected_candidates.feature.nunique()),
        "fixed_combinations": int(definitions.combination_id.nunique()),
        "two_condition_combinations": int((definitions.condition_count == 2).sum()),
        "three_condition_combinations": int((definitions.condition_count == 3).sum()),
        "environment_variables": int(len(regime_definitions)),
        "environment_sample_rows": int(len(sample_regimes)),
        "environment_valid_rows": int((sample_regimes.regime != "unknown").sum()),
        "environment_unknown_rows": int(len(unknown)),
        "unknown_by_environment_period": env_counts.to_dict("records"),
        "environment_summary": environment_summary(sample_regimes, metrics),
        "combination_regime_metric_rows": int(len(metrics)),
        "annual_regime_metric_rows": int(len(annual)),
        "regime_stability_rows": int(len(stability)),
        "reference_summary_rows": int(len(reference)),
        "formation_sample_rows": int((samples.period == "formation").sum()),
        "validation_sample_rows": int((samples.period == "validation").sum()),
        "input_sample_id_duplicates": int(samples.duplicated("sample_id").sum()),
        "input_invalid_pair_rows": invalid_pairs,
        "input_event_control_imbalance": event_control_imbalance,
        "input_period_event_control_imbalance": period_imbalance,
        "regime_sample_event_control_imbalance": regime_sample_imbalance,
        "period_or_boundary_errors": period_errors,
        "contaminated_samples": contaminated,
        "forbidden_features_used": forbidden_features,
        "combination_definition_mismatches": combination_definition_mismatches,
        "combination_definition_field_mismatches": definition_field_mismatches,
        "step7_combination_definition_sha256": expected_definition_hash,
        "validation_used_for_regime_boundary": False,
        "validation_regime_boundaries_reestimated": False,
        "combined_regime_definitions": 0,
        "optimized_regime_thresholds": 0,
        "major_key_duplicates": major_duplicates,
        "nonfinite_output_numeric_values": numeric_infinities(list(frames.values())),
        "parquet_reread_passed": True,
        "input_periods": sorted(periods),
    }


def step9_prompt() -> str:
    return """目的：
認証済みSTEP8 V2を基準に、STEP9 V2「エントリータイミング比較」だけを実行してください。

STEP1〜STEP8 V2を再計算・変更しないでください。旧STEP5〜旧STEP11の成果物は再利用禁止です。STEP7 V2で固定した候補特徴量、方向、30%・70%分位境界、2条件・3条件AND定義を変更・追加・削除・反転しないでください。STEP8 V2の相場環境別結果を見て組み合わせや環境を選別・変更してはいけません。

今回は固定済み全2,300組み合わせに対し、機械的に事前定義したエントリータイミングの差だけを比較します。採用シグナル、出口、資金管理、ポジションサイズ、売買ルールを確定しないでください。STEP10以降は実行しないでください。

【認証済み入力】
・data/market_history/analysis/step7_condition_combinations_v2/
・data/market_history/quality/step7_v2_report.json
・data/market_history/analysis/step8_regime_analysis_v2/
・data/market_history/quality/step8_v2_report.json

価格と営業日順の参照に限り、STEP1認証済み特徴量マスタを読み取り専用で使用できます。
・data/market_history/features/equity_daily_features/

必要な目的変数・価格経路の参照に限り、STEP2認証済み目的変数を読み取り専用で使用できます。
・data/market_history/targets/equity_daily_forward_targets/

分析対象event_id、期間、固定候補、方向、境界、組み合わせはSTEP7 V2を正本としてください。STEP8 V2は品質確認と環境別の記述結果にだけ使用し、エントリー定義の選択・変更に使用しないでください。

最初に以下を確認してください。
・STEP7 V2品質：PASS
・STEP8 V2品質：PASS
・入力ペア：8,611件
・イベント：8,611件
・対照：8,611件
・対象銘柄：2,637銘柄
・固定候補特徴量：24
・固定組み合わせ：2,300
・STEP8 V2環境変数：9
・期間越境：0件
・汚染期間混入：0件
・未来情報混入：0件
・主要キー重複：0件
・STEP7 V2とSTEP8 V2の再現性検査：PASS

一致しない場合はSTEP9 V2を実行せずFAILにしてください。

【期間】
・形成期：データ開始日〜2023-12-31
・検証期：2024-01-01〜2025-09-07
・汚染済み隔離期間：2025-09-08以降

2025-09-08以降は `contaminated_holdout` として、エントリー定義、価格参照、集計、比較、順位、可視化へ使用しないでください。エントリー後の評価窓が2025-09-08以降へ越えるサンプルも除外理由を記録して除外してください。新しい完全未使用OOS開始日を独断で決定しないでください。

【エントリー定義】
比較するタイミングは、形成期だけで次の規則を事前固定し、検証期へ同じまま適用してください。

1. 基準日当日終値：anchor_close
2. 翌営業日始値：next_session_open
3. 初押し：first_pullback

`first_pullback` は基準日の翌営業日から5営業日以内で、終値が前日終値以下となった最初の日の終値としてください。該当しない場合は未約定として保持してください。日数、価格種別、探索窓、未約定規則を検証期の結果で変更してはいけません。

銘柄ごとの営業日行位置で計算し、暦日シフトは禁止します。売買停止、欠損、上場廃止、分割を勝手に削除・補完しないでください。調整後価格と未調整価格を混在させず、使用価格列と分割調整方針を明記してください。

【評価】
固定済み全2,300組み合わせについて、形成期と検証期を完全分離し、各エントリー定義ごとに最低限以下を保存してください。
・組み合わせID
・エントリー定義
・期間
・シグナル該当数
・価格参照可能数
・約定数、未約定数、約定率
・エントリー日、エントリー価格
・5/10/20/40/60営業日先リターン
・各期間のMFE、MAE
・平均値、中央値、勝率
・形成期から検証期への維持率
・年別件数と方向再現性
・欠損と除外理由

未来リターン、MFE、MAE、将来高値、イベント到達日をエントリー条件や組み合わせ条件へ使用してはいけません。STEP2の目的変数を使う場合も評価専用に限定してください。

今回は取引コスト、出口、同時保有、資金制約、資金管理を適用しないでください。相場環境によるエントリー定義の選別や複数環境の組み合わせも禁止します。結果は参考比較であり、採用判断を行わないでください。

【保存】
旧STEP9を上書きせず、以下へ新規Parquetを保存してください。
data/market_history/analysis/step9_entry_timing_v2/

最低限：
・entry_definitions/
・entry_observations/
・entry_timing_metrics/
・annual_metrics/
・timing_stability/
・reference_summary/
・boundary_exclusions/
・period_assignment_audit/
・old_vs_v2_comparison/

品質レポート：
data/market_history/quality/step9_v2_report.json
data/market_history/quality/STEP9_V2_REPORT.md

【品質検査】
・入力ファイル前後SHA256
・形成期と検証期の分離
・2025-09-08以降の不使用
・銘柄ごとの営業日シフト
・STEP7固定組み合わせ定義との完全一致
・STEP8結果による組み合わせ選別の不存在
・形成期だけで固定したエントリー定義
・検証期でエントリー規則を変更していないこと
・未来情報の条件側への混入
・価格列と分割調整方針の一貫性
・欠損、未約定、除外理由の保存
・主要キー重複
・イベント・対照件数整合
・無限大
・出力Parquet再読込
・同じ入力による2回実行のハッシュ一致

期間越境、汚染期間使用、未来情報の条件側混入、STEP7組み合わせ変更、STEP8結果による候補選別、検証期によるエントリー規則変更、入力変更、主要キー重複、再現性不一致が1件でもあればFAILにしてください。

【完了報告】
最後に必ず以下を報告してください。
・STEP9 V2 完了 / FAIL
・形成期
・検証期
・汚染済み隔離期間
・入力ペア数
・固定候補特徴量数
・固定組み合わせ数
・比較エントリー定義数
・定義別シグナル数、約定数、未約定数、約定率
・形成期と検証期の定義別結果
・欠損・除外件数と理由
・期間越境件数
・未来情報混入件数
・重複件数
・作成ファイル
・保存先
・ファイル容量
・入力変更の有無
・再現性検査結果
・品質判定
・残存リスク

PASSの場合だけ「STEP9 V2で作成した成果物は、同じ入力版では今後再計算不要」と明記してください。

処理完了後、回答の最後にSTEP10 V2「利益を伸ばす出口検証」だけを実行する完全な次回用プロンプトを表示し、同じ内容を `data/market_history/quality/STEP10_V2_PROMPT.md` へ保存してください。STEP11以降を実行させないでください。

今後も各STEP終了時に次のSTEPだけの完全なプロンプトを回答の最後へ表示し、品質フォルダへ保存してください。
"""


def markdown_report(report: dict[str, Any]) -> str:
    m = report["metrics"]
    summary = pd.DataFrame(m["environment_summary"])
    lines = [
        "# STEP8 V2 品質レポート",
        "",
        f"判定：**{report['quality']}**",
        "",
        "- 形成期：データ開始日〜2023-12-31",
        "- 検証期：2024-01-01〜2025-09-07",
        "- 汚染済み隔離期間：2025-09-08以降",
        f"- 入力ペア：{m['input_pairs']:,}（イベント{m['input_events']:,}、対照{m['input_controls']:,}）",
        f"- 対象銘柄：{m['input_tickers']:,}",
        f"- 固定候補：{m['fixed_candidate_features']:,}特徴量",
        f"- 固定組み合わせ：{m['fixed_combinations']:,}",
        f"- 環境変数：{m['environment_variables']:,}",
        f"- 環境観測：{m['environment_sample_rows']:,}、有効{m['environment_valid_rows']:,}、unknown {m['environment_unknown_rows']:,}",
        f"- 組み合わせ×環境×期間：{m['combination_regime_metric_rows']:,}行",
        f"- 年別：{m['annual_regime_metric_rows']:,}行",
        f"- 期間・境界エラー：{m['period_or_boundary_errors']:,}",
        f"- 汚染期間サンプル：{m['contaminated_samples']:,}",
        f"- 未来情報特徴量：{len(m['forbidden_features_used']):,}",
        f"- STEP7定義不一致：{m['combination_definition_mismatches']:,}",
        f"- 主要キー重複：{m['major_key_duplicates']:,}",
        f"- 入力変更：{not report['inputs_unchanged']}",
        f"- 2回実行のParquet一致：{report['reproducibility_passed']}",
        f"- 保存容量：{report['file_size_bytes']:,} bytes",
        "",
        "## 環境別の記述集計",
        "",
        "| 環境 | 期間 | 有効観測 | unknown | リフト中央値 | Q25 | Q75 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.environment_variable} | {row.period} | {row.valid_environment_rows:,} | {row.unknown_rows:,} | {row.median_lift:.4f} | {row.lift_q25:.4f} | {row.lift_q75:.4f} |")
    lines.extend([
        "",
        "環境境界はSTEP6 V2の形成期固定 `coarse_30_70` をそのまま使用し、検証期では再計算していません。STEP7 V2の2,300組み合わせも変更していません。複数環境を結合せず、各環境を単独評価しました。",
        "",
        "ケース・コントロール上のイベント率は実市場の無条件発生確率ではありません。環境別結果は参考記述であり、採用シグナル、環境フィルター、エントリー、出口、資金管理、売買ルールではありません。多数の組み合わせと環境区分を同時に見るため、多重比較と見かけの安定性に注意が必要です。",
        "",
        "旧STEP8〜旧STEP11は再利用禁止です。2025-09-08以降は完全未使用OOSへ戻していません。",
        "",
        report["completion_statement"],
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    options = parse_args()
    repository = Path(__file__).resolve().parents[2]
    root = (repository / "momentum5d" / options.root).resolve()
    report_path = root / "quality/step7_v2_report.json"
    if not report_path.exists():
        raise RuntimeError(f"STEP7 V2 report missing: {report_path}")
    step7_report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_step7(step7_report)
    groups = [
        ("step7_v2", root / "analysis/step7_condition_combinations_v2"),
        ("step7_v2_report", report_path),
        ("step6_analysis_samples", root / "analysis/step6_univariate_v2/analysis_samples"),
        ("step6_quantile_bins", root / "analysis/step6_univariate_v2/quantile_bins"),
    ]
    old_step8 = root / "analysis/step8_regime_analysis"
    if old_step8.exists():
        groups.append(("old_step8_audit_only", old_step8))
    before = manifest(groups)
    target = root / "analysis/step8_regime_analysis_v2"
    quality = root / "quality"
    if target.exists() or (quality / "step8_v2_report.json").exists():
        raise RuntimeError("STEP8 V2 immutable output already exists")

    temporary = Path(tempfile.mkdtemp(prefix="step8_v2_", dir=root.parent))
    first = temporary / "first"
    metrics = build_once(root, first)
    first_hashes = output_manifest(first)
    reproducible = False
    second_hashes: dict[str, str] | None = None
    if options.verify_reproducibility:
        gc.collect()
        second = temporary / "second"
        second_metrics = build_once(root, second)
        second_hashes = output_manifest(second)
        reproducible = first_hashes == second_hashes and metrics == second_metrics
    after = manifest(groups)
    inputs_unchanged = before == after
    hard_failures = {
        "input_sample_id_duplicates": metrics["input_sample_id_duplicates"],
        "input_invalid_pair_rows": metrics["input_invalid_pair_rows"],
        "input_event_control_imbalance": metrics["input_event_control_imbalance"],
        "input_period_event_control_imbalance": metrics["input_period_event_control_imbalance"],
        "regime_sample_event_control_imbalance": metrics["regime_sample_event_control_imbalance"],
        "period_or_boundary_errors": metrics["period_or_boundary_errors"],
        "contaminated_samples": metrics["contaminated_samples"],
        "forbidden_features_used": len(metrics["forbidden_features_used"]),
        "combination_definition_mismatches": metrics["combination_definition_mismatches"],
        "combination_definition_field_mismatches": metrics["combination_definition_field_mismatches"],
        "validation_regime_boundaries_reestimated": int(metrics["validation_regime_boundaries_reestimated"]),
        "combined_regime_definitions": metrics["combined_regime_definitions"],
        "optimized_regime_thresholds": metrics["optimized_regime_thresholds"],
        "major_key_duplicates": metrics["major_key_duplicates"],
        "nonfinite_output_numeric_values": metrics["nonfinite_output_numeric_values"],
    }
    passed = (
        all(value == 0 for value in hard_failures.values())
        and inputs_unchanged
        and options.verify_reproducibility
        and reproducible
        and metrics["parquet_reread_passed"]
        and not metrics["validation_used_for_regime_boundary"]
        and metrics["input_pairs"] == 8611
        and metrics["input_events"] == metrics["input_controls"] == 8611
        and metrics["input_tickers"] == 2637
        and metrics["fixed_candidate_features"] == 24
        and metrics["fixed_combinations"] == 2300
        and metrics["environment_variables"] == len(REGIME_SPECS)
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    first.rename(target)
    report = {
        "step": 8,
        "version": 2,
        "status": "STEP8 V2 complete" if passed else "STEP8 V2 FAIL",
        "quality": "PASS" if passed else "FAIL",
        "created_at_jst": datetime.now(JST).isoformat(),
        "formation": {"start": "data_start", "end": "2023-12-31"},
        "validation": {"start": "2024-01-01", "end": "2025-09-07"},
        "contaminated_holdout_start": "2025-09-08",
        "new_untouched_oos_start": None,
        "steps1_to_7_v2_recalculated": False,
        "old_step8_to_11_reused": False,
        "step7_combination_definitions_changed": False,
        "regime_boundary_source": "STEP6 V2 formation coarse_30_70",
        "validation_used_after_regime_freeze_only": True,
        "signal_entry_exit_money_management_created": False,
        "regime_filter_or_trading_rule_adopted": False,
        "input_manifest_before": before,
        "input_manifest_after": after,
        "inputs_unchanged": inputs_unchanged,
        "reproducibility_requested": options.verify_reproducibility,
        "reproducibility_passed": reproducible,
        "first_output_manifest": first_hashes,
        "second_output_manifest": second_hashes,
        "hard_failures": hard_failures,
        "metrics": metrics,
        "save_path": str(target),
        "file_size_bytes": directory_size(target),
        "completion_statement": "STEP8 V2で作成した成果物は、同じ入力版では今後再計算不要" if passed else "STEP8 V2はFAILのため再計算不要とは認証しない",
        "residual_risks": [
            "case-control event rates are not unconditional market probabilities",
            "2,300 combinations across nine environments create substantial multiple-testing risk",
            "regime measurements are d-5_to_d-1 bucket medians rather than live intraday state",
            "repeated tickers and shared macro dates reduce effective sample independence",
            "formation coarse quantiles are descriptive policy boundaries, not economically optimized thresholds",
            "thin combination-regime cells can produce unstable lift and retention estimates",
            "STEP8 V2 adopts no combination, environment filter, signal, or trading rule",
            "2025-09-08 onward remains contaminated and cannot be reused as untouched OOS",
        ],
    }
    write_json(quality / "step8_v2_report.json", report)
    (quality / "STEP8_V2_REPORT.md").write_text(markdown_report(report), encoding="utf-8")
    (quality / "STEP9_V2_PROMPT.md").write_text(step9_prompt(), encoding="utf-8")
    shutil.rmtree(temporary, ignore_errors=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
