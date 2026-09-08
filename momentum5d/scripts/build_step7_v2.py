# ruff: noqa: E501
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
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
FORMATION_AUC_MIN = 0.55
FORMATION_MISSING_MAX = 0.20
FORMATION_ABS_SMD_MIN = 0.10
FORMATION_YEAR_REPRO_MIN = 2 / 3
CORRELATION_LIMIT = 0.80
CORRELATION_MIN_PAIRS = 200
REFERENCE_SUPPORT_MIN = 100
FORBIDDEN = ("future", "forward", "target", "label", "mfe", "mae", "peak_date", "event_end", "outcome")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build isolated STEP7 V2 condition combinations")
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


def validate_step6(report: dict[str, Any]) -> None:
    expected = {
        "quality": "PASS", "version": 2, "inputs_unchanged": True,
        "reproducibility_passed": True, "direction_source": "formation only",
        "quantile_boundary_source": "formation only", "validation_direction_reestimated": False,
        "validation_quantiles_reestimated": False,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(f"STEP6 V2 precondition mismatch: {key}")
    expected_metrics = {
        "matched_pairs": 8611, "event_rows": 8611, "control_rows": 8611,
        "tickers": 2637, "period_or_boundary_errors": 0, "contaminated_samples": 0,
        "major_key_duplicates": 0, "sample_id_duplicates": 0, "invalid_pair_rows": 0,
        "event_control_imbalance": 0, "period_event_control_imbalance": 0,
        "validation_fixed_direction_mismatches": 0, "quantile_threshold_mismatches": 0,
    }
    for key, value in expected_metrics.items():
        if report.get("metrics", {}).get(key) != value:
            raise RuntimeError(f"STEP6 V2 metric mismatch: {key}")
    if report.get("metrics", {}).get("forbidden_predictor_columns") != []:
        raise RuntimeError("STEP6 V2 contains forbidden predictors")


def formation_candidate_table(
    samples: pd.DataFrame,
    metrics: pd.DataFrame,
    annual: pd.DataFrame,
    quantiles: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    main_metrics = metrics[metrics.bucket == MAIN_BUCKET].copy()
    formation_annual = annual[(annual.bucket == MAIN_BUCKET) & (annual.period == "formation")]
    yearly = formation_annual.groupby("feature", sort=True).direction_reproduced.agg(
        formation_years_evaluable="count",
        formation_years_reproduced="sum",
        formation_year_reproduction_rate="mean",
    ).reset_index()
    coarse = quantiles[(quantiles.scheme == "coarse_30_70") & (quantiles.period == "formation")]
    thresholds = coarse.groupby("feature", sort=True).formation_thresholds_json.first().reset_index()
    candidates = main_metrics.merge(yearly, on="feature", how="left", validate="one_to_one")
    candidates = candidates.merge(thresholds, on="feature", how="left", validate="one_to_one")
    decoded = candidates.formation_thresholds_json.map(
        lambda value: json.loads(value) if isinstance(value, str) else [None, None]
    )
    candidates["q30"] = decoded.map(lambda values: finite(values[0]) if len(values) == 2 else None)
    candidates["q70"] = decoded.map(lambda values: finite(values[1]) if len(values) == 2 else None)
    candidates["condition_threshold"] = np.where(
        candidates.formation_direction.eq("high"), candidates.q70, candidates.q30
    )
    candidates["condition_operator"] = np.where(candidates.formation_direction.eq("high"), ">=", "<=")

    def base_reason(row: Any) -> str | None:
        if finite(row.condition_threshold) is None:
            return "missing_formation_coarse_quantile"
        if row.formation_fixed_direction_auc < FORMATION_AUC_MIN:
            return "formation_fixed_auc_below_0.55"
        if row.formation_missing_rate > FORMATION_MISSING_MAX:
            return "formation_missing_rate_above_0.20"
        if abs(row.formation_standardized_mean_difference) < FORMATION_ABS_SMD_MIN:
            return "formation_abs_smd_below_0.10"
        if row.formation_year_reproduction_rate < FORMATION_YEAR_REPRO_MIN:
            return "formation_year_reproduction_below_two_thirds"
        return None

    candidates["base_exclusion_reason"] = [base_reason(row) for row in candidates.itertuples(index=False)]
    candidates["base_candidate"] = candidates.base_exclusion_reason.isna()
    ordering = candidates[candidates.base_candidate].sort_values(
        ["formation_fixed_direction_auc", "formation_missing_rate", "formation_valid_rows", "feature"],
        ascending=[False, True, False, True], kind="stable",
    )
    base_features = ordering.feature.tolist()
    formation = samples[(samples.bucket == MAIN_BUCKET) & (samples.period == "formation")]
    correlation_matrix = formation[base_features].corr(method="spearman", min_periods=CORRELATION_MIN_PAIRS)
    correlation_rows: list[dict[str, Any]] = []
    for left, right in itertools.combinations(base_features, 2):
        correlation = finite(correlation_matrix.loc[left, right])
        valid_pairs = int(formation[[left, right]].dropna().shape[0])
        correlation_rows.append({
            "feature_a": left, "feature_b": right, "spearman_correlation": correlation,
            "absolute_correlation": abs(correlation) if correlation is not None else None,
            "valid_pairs": valid_pairs,
            "strong_correlation": abs(correlation) >= CORRELATION_LIMIT if correlation is not None else False,
            "calculation_period": "formation", "used_validation": False,
        })
    correlations = pd.DataFrame(correlation_rows)

    selected: list[str] = []
    conflict: dict[str, tuple[str, float]] = {}
    for feature in base_features:
        overlaps = [
            (kept, abs(float(correlation_matrix.loc[feature, kept])))
            for kept in selected
            if pd.notna(correlation_matrix.loc[feature, kept])
            and abs(float(correlation_matrix.loc[feature, kept])) >= CORRELATION_LIMIT
        ]
        if overlaps:
            conflict[feature] = max(overlaps, key=lambda item: item[1])
        else:
            selected.append(feature)
    candidates["correlation_conflict_feature"] = candidates.feature.map(
        lambda feature: conflict.get(feature, (None, None))[0]
    )
    candidates["correlation_with_conflict"] = candidates.feature.map(
        lambda feature: conflict.get(feature, (None, None))[1]
    )
    candidates["selected_candidate"] = candidates.feature.isin(selected)
    candidates["final_exclusion_reason"] = candidates.base_exclusion_reason
    candidates.loc[candidates.feature.isin(conflict), "final_exclusion_reason"] = "absolute_formation_spearman_at_least_0.80"
    candidates["selection_source"] = "formation_only"
    candidates["validation_used_for_selection"] = False
    candidates["candidate_signal_adopted"] = False
    candidates["post_freeze_effect_sign_reproduced"] = (
        np.sign(candidates.formation_standardized_mean_difference)
        == np.sign(candidates.validation_standardized_mean_difference)
    )
    candidates["post_freeze_validation_descriptive_only"] = True
    candidates["selection_order"] = candidates.feature.map({feature: i + 1 for i, feature in enumerate(selected)}).astype("Int64")
    keep = [
        "feature", "formation_direction", "condition_operator", "condition_threshold", "q30", "q70",
        "formation_fixed_direction_auc", "formation_standardized_mean_difference", "formation_missing_rate",
        "formation_valid_rows", "formation_years_evaluable", "formation_years_reproduced",
        "formation_year_reproduction_rate", "base_candidate", "selected_candidate", "selection_order",
        "base_exclusion_reason", "final_exclusion_reason", "correlation_conflict_feature",
        "correlation_with_conflict", "selection_source", "validation_used_for_selection",
        "candidate_signal_adopted", "validation_fixed_direction_auc",
        "direction_reproduced_in_validation", "validation_standardized_mean_difference",
        "post_freeze_effect_sign_reproduced", "post_freeze_validation_descriptive_only",
    ]
    return candidates[keep].sort_values(["selected_candidate", "selection_order", "feature"], ascending=[False, True, True], na_position="last").reset_index(drop=True), correlations, selected


def combination_definitions(candidates: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    lookup = candidates.set_index("feature")
    rows: list[dict[str, Any]] = []
    for count in (2, 3):
        for index, features in enumerate(itertools.combinations(selected, count), start=1):
            padded = [*features, *([None] * (3 - count))]
            rows.append({
                "combination_id": f"C{count}-{index:04d}", "condition_count": count,
                "feature_1": padded[0], "direction_1": lookup.loc[padded[0], "formation_direction"],
                "operator_1": lookup.loc[padded[0], "condition_operator"],
                "threshold_1": finite(lookup.loc[padded[0], "condition_threshold"]),
                "feature_2": padded[1], "direction_2": lookup.loc[padded[1], "formation_direction"],
                "operator_2": lookup.loc[padded[1], "condition_operator"],
                "threshold_2": finite(lookup.loc[padded[1], "condition_threshold"]),
                "feature_3": padded[2],
                "direction_3": lookup.loc[padded[2], "formation_direction"] if padded[2] else None,
                "operator_3": lookup.loc[padded[2], "condition_operator"] if padded[2] else None,
                "threshold_3": finite(lookup.loc[padded[2], "condition_threshold"]) if padded[2] else None,
                "logical_operator": "AND", "threshold_source": "STEP6 V2 formation coarse_30_70",
                "candidate_source": "formation_only", "validation_used_for_definition": False,
                "adopted_signal": False, "adopted_trading_rule": False,
            })
    return pd.DataFrame(rows)


def definition_features(row: Any) -> list[tuple[str, str, float]]:
    result: list[tuple[str, str, float]] = []
    for number in range(1, int(row.condition_count) + 1):
        result.append((
            getattr(row, f"feature_{number}"), getattr(row, f"direction_{number}"),
            float(getattr(row, f"threshold_{number}")),
        ))
    return result


def evaluate_condition(frame: pd.DataFrame, conditions: list[tuple[str, str, float]]) -> dict[str, Any]:
    valid = pd.Series(True, index=frame.index)
    condition = pd.Series(True, index=frame.index)
    for feature, direction, threshold in conditions:
        values = pd.to_numeric(frame[feature], errors="coerce")
        feature_valid = values.notna() & np.isfinite(values)
        valid &= feature_valid
        condition &= feature_valid & ((values >= threshold) if direction == "high" else (values <= threshold))
    selected = frame[valid & condition]
    valid_frame = frame[valid]
    base_rate = finite(valid_frame.outcome.mean()) if len(valid_frame) else None
    event_rate = finite(selected.outcome.mean()) if len(selected) else None
    return {
        "total_rows": int(len(frame)), "valid_rows": int(valid.sum()), "missing_rows": int((~valid).sum()),
        "missing_rate": finite((~valid).mean()), "condition_rows": int(len(selected)),
        "condition_events": int(selected.outcome.sum()),
        "condition_controls": int(len(selected) - selected.outcome.sum()),
        "event_rate": event_rate, "base_event_rate": base_rate,
        "lift": finite(event_rate / base_rate) if event_rate is not None and base_rate else None,
        "event_recall": finite(selected.outcome.sum() / valid_frame.outcome.sum()) if len(valid_frame) and valid_frame.outcome.sum() else None,
        "missing_reason": "one_or_more_component_features_missing; retained without imputation" if (~valid).any() else None,
    }


def evaluate_combinations(samples: pd.DataFrame, definitions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    for definition in definitions.itertuples(index=False):
        conditions = definition_features(definition)
        for period in ("formation", "validation"):
            part = samples[samples.period == period]
            rows.append({
                "combination_id": definition.combination_id, "condition_count": int(definition.condition_count),
                "period": period, **evaluate_condition(part, conditions),
                "definition_frozen_before_validation": True,
            })
        for year in sorted(samples.pair_year.unique()):
            part = samples[samples.pair_year == year]
            evaluated = evaluate_condition(part, conditions)
            annual_rows.append({
                "combination_id": definition.combination_id, "condition_count": int(definition.condition_count),
                "pair_year": int(year), "period": str(part.period.iloc[0]), **evaluated,
                "direction_reproduced": evaluated["lift"] >= 1.0 if evaluated["lift"] is not None else None,
                "definition_frozen_before_validation": True,
            })
    return pd.DataFrame(rows), pd.DataFrame(annual_rows)


def reference_ranking(
    definitions: pd.DataFrame, metrics: pd.DataFrame, annual: pd.DataFrame
) -> pd.DataFrame:
    formation = metrics[metrics.period == "formation"].drop(columns="period").add_prefix("formation_").rename(columns={"formation_combination_id": "combination_id"})
    validation = metrics[metrics.period == "validation"].drop(columns="period").add_prefix("validation_").rename(columns={"validation_combination_id": "combination_id"})
    yearly = annual.groupby("combination_id", sort=True).direction_reproduced.agg(
        years_evaluable="count", years_reproduced="sum", year_direction_reproduction_rate="mean"
    ).reset_index()
    ranking = definitions.merge(formation, on="combination_id", validate="one_to_one")
    ranking = ranking.merge(validation, on="combination_id", validate="one_to_one").merge(yearly, on="combination_id", validate="one_to_one")
    ranking["formation_to_validation_lift_retention"] = ranking.validation_lift / ranking.formation_lift
    ranking["formation_support_adequate"] = ranking.formation_condition_rows >= REFERENCE_SUPPORT_MIN
    ranking["validation_direction_reproduced"] = ranking.validation_lift >= 1.0
    ranking["reference_only"] = True
    ranking["adopted_signal"] = False
    ranking["adopted_threshold"] = False
    ranking["ranking_basis"] = "formation support and lift only; validation appended after frozen definition"
    ranking = ranking.sort_values(
        ["formation_support_adequate", "formation_lift", "formation_condition_rows", "formation_missing_rate", "condition_count", "combination_id"],
        ascending=[False, False, False, True, True, True], na_position="last", kind="stable",
    ).reset_index(drop=True)
    ranking.insert(0, "reference_rank", np.arange(1, len(ranking) + 1))
    return ranking


def old_comparison(root: Path, definitions: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    old = root / "analysis/step7_condition_combinations"
    if not old.exists():
        return pd.DataFrame([{
            "metric": "old_step7_available", "old_value": 0, "v2_value": 1,
            "used_for_v2_selection": False, "note": "old STEP7 unavailable and not required",
        }])
    old_metrics = pd.read_parquet(old / "combination_metrics")
    values = [
        ("combination_count", int(old_metrics.combo_id.nunique()), int(definitions.combination_id.nunique())),
        ("two_condition_count", int(old_metrics.loc[old_metrics.condition_count == 2, "combo_id"].nunique()), int((definitions.condition_count == 2).sum())),
        ("three_condition_count", int(old_metrics.loc[old_metrics.condition_count == 3, "combo_id"].nunique()), int((definitions.condition_count == 3).sum())),
        ("period_metric_rows", int(len(old_metrics)), int(len(metrics))),
    ]
    return pd.DataFrame([
        {"metric": name, "old_value": old_value, "v2_value": v2_value,
         "used_for_v2_selection": False, "note": "old result invalidated; audit comparison only"}
        for name, old_value, v2_value in values
    ])


def build_once(root: Path, stage: Path) -> dict[str, Any]:
    names = (
        "candidate_features", "feature_correlations", "combination_definitions", "combination_metrics",
        "annual_metrics", "reference_ranking", "period_assignment_audit", "old_vs_v2_comparison",
    )
    for name in names:
        (stage / name).mkdir(parents=True, exist_ok=True)
    step6 = root / "analysis/step6_univariate_v2"
    samples_all = pd.read_parquet(step6 / "analysis_samples")
    samples = samples_all[samples_all.bucket == MAIN_BUCKET].copy()
    metrics6 = pd.read_parquet(step6 / "univariate_metrics")
    annual6 = pd.read_parquet(step6 / "annual_metrics")
    quantiles6 = pd.read_parquet(step6 / "quantile_bins")
    candidates, correlations, selected = formation_candidate_table(
        samples, metrics6, annual6, quantiles6
    )
    definitions = combination_definitions(candidates, selected)
    metrics, annual = evaluate_combinations(samples, definitions)
    ranking = reference_ranking(definitions, metrics, annual)
    comparison = old_comparison(root, definitions, metrics)
    period_audit = pd.read_parquet(step6 / "period_assignment_audit").copy()
    period_audit["step7_used_main_bucket_only"] = True
    period_audit["step7_candidate_period"] = "formation_only"
    period_audit["step7_contaminated_holdout_used"] = False

    frames = {
        "candidate_features": candidates, "feature_correlations": correlations,
        "combination_definitions": definitions, "combination_metrics": metrics,
        "annual_metrics": annual, "reference_ranking": ranking,
        "period_assignment_audit": period_audit, "old_vs_v2_comparison": comparison,
    }
    for name, frame in frames.items():
        write_parquet(frame, stage / name / "part.parquet")

    input_balance = samples.groupby("period").outcome.agg(["sum", "count"])
    input_period_imbalance = int((input_balance["sum"] - (input_balance["count"] - input_balance["sum"])).abs().sum())
    selected_correlations = correlations[
        correlations.feature_a.isin(selected) & correlations.feature_b.isin(selected)
    ]
    selected_strong_correlations = int(selected_correlations.strong_correlation.sum())
    key_duplicates = int(
        candidates.duplicated(["feature"]).sum()
        + correlations.duplicated(["feature_a", "feature_b"]).sum()
        + definitions.duplicated(["combination_id"]).sum()
        + metrics.duplicated(["combination_id", "period"]).sum()
        + annual.duplicated(["combination_id", "pair_year"]).sum()
        + ranking.duplicated(["combination_id"]).sum()
        + period_audit.duplicated(["event_id"]).sum()
    )
    numeric_output = pd.concat([
        candidates.select_dtypes(include="number"), correlations.select_dtypes(include="number"),
        definitions.select_dtypes(include="number"), metrics.select_dtypes(include="number"),
        annual.select_dtypes(include="number"), ranking.select_dtypes(include="number"),
    ], axis=1)
    forbidden_candidates = [feature for feature in selected if any(token in feature.lower() for token in FORBIDDEN)]
    threshold_lookup = candidates[candidates.selected_candidate].set_index("feature").condition_threshold.to_dict()
    definition_threshold_mismatches = 0
    for definition in definitions.itertuples(index=False):
        for feature, _, threshold in definition_features(definition):
            if feature not in threshold_lookup or not math.isclose(threshold, float(threshold_lookup[feature]), rel_tol=0, abs_tol=0):
                definition_threshold_mismatches += 1
    period_errors = int(
        (~period_audit.period_equal).sum() + period_audit.contaminated_anchor.sum()
        + period_audit.formation_target_crossing.sum() + period_audit.validation_window_crossing.sum()
        + period_audit.validation_target_crossing.sum()
    )
    contaminated_samples = int(
        ((pd.to_datetime(samples.event_start_date) >= HOLDOUT_START)
         | (pd.to_datetime(samples.control_start_date) >= HOLDOUT_START)).sum()
    )
    condition_missing = metrics.missing_rate.dropna()
    formation_valid = metrics.loc[metrics.period == "formation", "valid_rows"]
    validation_valid = metrics.loc[metrics.period == "validation", "valid_rows"]
    return {
        "input_pairs": int(samples.event_id.nunique()), "input_rows": int(len(samples)),
        "input_events": int((samples.outcome == 1).sum()), "input_controls": int((samples.outcome == 0).sum()),
        "input_tickers": int(samples.Ticker.nunique()), "eligible_step6_features": int(len(candidates)),
        "base_candidate_features": int(candidates.base_candidate.sum()),
        "selected_candidate_features": int(candidates.selected_candidate.sum()),
        "excluded_candidate_features": int((~candidates.selected_candidate).sum()),
        "candidate_exclusions_by_reason": {str(k): int(v) for k, v in candidates.loc[~candidates.selected_candidate, "final_exclusion_reason"].value_counts().sort_index().items()},
        "formation_only_correlation_pairs": int(len(correlations)),
        "selected_strong_correlation_pairs": selected_strong_correlations,
        "two_condition_combinations": int((definitions.condition_count == 2).sum()),
        "three_condition_combinations": int((definitions.condition_count == 3).sum()),
        "total_combinations": int(len(definitions)), "period_metric_rows": int(len(metrics)),
        "annual_metric_rows": int(len(annual)), "formation_valid_rows_min": int(formation_valid.min()),
        "formation_valid_rows_max": int(formation_valid.max()), "validation_valid_rows_min": int(validation_valid.min()),
        "validation_valid_rows_max": int(validation_valid.max()),
        "combination_missing_rate_min": finite(condition_missing.min()),
        "combination_missing_rate_median": finite(condition_missing.median()),
        "combination_missing_rate_max": finite(condition_missing.max()),
        "input_sample_id_duplicates": int(samples.sample_id.duplicated().sum()),
        "input_invalid_pair_rows": int(
            ((samples.groupby("event_id").outcome.agg(["sum", "count"])["sum"] != 1)
             | (samples.groupby("event_id").outcome.agg(["sum", "count"])["count"] != 2)).sum()
        ),
        "input_event_control_imbalance": abs(int((samples.outcome == 1).sum()) - int((samples.outcome == 0).sum())),
        "input_period_event_control_imbalance": input_period_imbalance,
        "period_or_boundary_errors": period_errors, "contaminated_samples": contaminated_samples,
        "forbidden_candidate_features": forbidden_candidates,
        "definition_threshold_mismatches": definition_threshold_mismatches,
        "validation_used_for_candidate_selection": False, "validation_used_for_correlation": False,
        "validation_used_for_definition": False, "validation_thresholds_reestimated": False,
        "major_key_duplicates": key_duplicates,
        "nonfinite_output_numeric_values": int(np.isinf(numeric_output.to_numpy(dtype=float, na_value=np.nan)).sum()),
        "parquet_reread_passed": True,
    }


def step8_prompt() -> str:
    return """目的：
認証済みSTEP7 V2を基準に、STEP8 V2「相場環境別検証」だけを実行してください。

STEP1〜STEP7 V2を再計算・変更しないでください。旧STEP5〜旧STEP11の成果物は再利用禁止です。STEP7 V2で形成期だけから固定した候補特徴量、方向、30%・70%分位境界、2条件・3条件ANDの定義を変更・追加・削除・反転しないでください。今回は固定済み参考組み合わせの相場環境依存性を調べる工程であり、シグナル採用、エントリー、出口、資金管理、売買ルールは作成しないでください。STEP9以降は実行しないでください。

【認証済み入力】
・data/market_history/analysis/step7_condition_combinations_v2/
・data/market_history/quality/step7_v2_report.json

固定済み組み合わせを各サンプルへ適用するために限り、以下を読み取り専用で使用できます。
・data/market_history/analysis/step6_univariate_v2/analysis_samples/
・data/market_history/analysis/step6_univariate_v2/quantile_bins/

分析対象event_id、期間、候補特徴量、方向、境界値、組み合わせは必ずSTEP7 V2を正本としてください。STEP6の順位から候補を選び直してはいけません。

最初にSTEP7 V2がPASS、入力8,611ペア、イベント8,611件、対照8,611件、対象2,637銘柄、期間越境0、汚染期間混入0、未来情報混入0、選択候補間の強相関0、主要キー重複0、形成期のみで候補・相関・組み合わせを固定、再現性PASSであることを確認してください。一致しない場合はSTEP8 V2を実行せずFAILにしてください。

【期間】
・形成期：データ開始日〜2023-12-31
・検証期：2024-01-01〜2025-09-07
・汚染済み隔離期間：2025-09-08以降

2025-09-08以降は `contaminated_holdout` として、環境定義、分位計算、分析、集計、順位、可視化へ使用しないでください。OOS、形成期、検証期として扱わず、新しい完全未使用OOS開始日を独断で決定しないでください。

【相場環境の定義】
相場環境の分類規則と境界は形成期だけで事前固定し、検証期へ同じまま適用してください。STEP6 V2に形成期固定済みの粗い30%・70%分位境界がある場合はそれを使用してください。検証期で境界を再計算したり、組み合わせの成績がよく見えるよう環境区分を変更してはいけません。

最低限、利用可能な既存特徴量から以下を個別に評価してください。
・日経平均：弱い／中立／強い
・米国株（S&P500、NASDAQ、SOX）：弱い／中立／強い
・VIX：低い／中立／高い
・USDJPY：円高方向／中立／円安方向
・米10年金利：低下／中立／上昇
・市場内部：弱い／中立／強い
・市場売買代金：減少／中立／増加

複数環境を組み合わせた細分化、最良環境の探索、環境閾値の最適化は禁止します。欠損した環境値は無条件補完せず `unknown` として件数を保存してください。

【評価】
STEP7 V2の全固定組み合わせについて、形成期と検証期を完全分離し、環境変数×環境区分ごとに以下を保存してください。
・組み合わせID
・環境変数、環境区分、形成期固定境界
・全行数、有効行数、欠損数
・条件該当数、イベント数、対照数
・イベント率、全体イベント率、リフト
・形成期から検証期へのリフト維持率
・環境間の最大差
・年別件数と方向再現性

ケース・コントロール上のイベント率は、実市場の無条件発生確率ではありません。相場環境別結果は参考評価であり、組み合わせや売買ルールの採用判断を今回は行わないでください。

【保存】
旧STEP8を上書きせず、以下へ新規Parquetを保存してください。
data/market_history/analysis/step8_regime_analysis_v2/

最低限：regime_definitions/、sample_regimes/、combination_regime_metrics/、annual_regime_metrics/、regime_stability/、reference_summary/、period_assignment_audit/、old_vs_v2_comparison/

品質レポート：
data/market_history/quality/step8_v2_report.json
data/market_history/quality/STEP8_V2_REPORT.md

【品質検査】
入力前後SHA256、期間分離、汚染期間不使用、未来情報不使用、STEP7定義の完全一致、形成期のみの環境境界、検証期での境界再計算なし、環境欠損件数、主要キー重複、件数整合、無限大、Parquet再読込、同じ入力による2回実行のハッシュ一致を検査してください。

期間越境、汚染期間使用、未来情報混入、STEP7組み合わせ変更、検証期による環境境界変更、入力変更、主要キー重複、再現性不一致が1件でもあればFAILにしてください。

【完了報告】
STEP8 V2 完了 / FAIL、期間、入力ペア数、固定組み合わせ数、環境変数数、環境別有効件数、unknown件数、形成期・検証期の結果、期間越境、未来情報、重複、作成ファイル、保存先、容量、入力変更、再現性、品質、残存リスクを報告してください。

PASSの場合だけ「STEP8 V2で作成した成果物は、同じ入力版では今後再計算不要」と明記してください。

処理完了後、回答の最後にSTEP9 V2「エントリータイミング比較」だけを実行する完全な次回用プロンプトを表示し、同じ内容を `data/market_history/quality/STEP9_V2_PROMPT.md` へ保存してください。STEP10以降を実行させないでください。今後も各STEP終了時に次のSTEPだけの完全なプロンプトを回答の最後へ表示し、品質フォルダへ保存してください。
"""


def markdown_report(report: dict[str, Any]) -> str:
    m = report["metrics"]
    return f"""# STEP7 V2 品質レポート

判定：**{report['quality']}**

- 形成期：データ開始日〜2023-12-31
- 検証期：2024-01-01〜2025-09-07
- 汚染済み隔離期間：2025-09-08以降
- 入力ペア：{m['input_pairs']:,}（イベント{m['input_events']:,}、対照{m['input_controls']:,}）
- 対象銘柄：{m['input_tickers']:,}
- STEP6 V2評価対象：{m['eligible_step6_features']:,}特徴量
- 形成期粗選別通過：{m['base_candidate_features']:,}特徴量
- 強相関除去後候補：{m['selected_candidate_features']:,}特徴量
- 除外候補：{m['excluded_candidate_features']:,}
- 形成期相関ペア：{m['formation_only_correlation_pairs']:,}
- 選択候補間の強相関：{m['selected_strong_correlation_pairs']:,}
- 2条件AND：{m['two_condition_combinations']:,}
- 3条件AND：{m['three_condition_combinations']:,}
- 全組み合わせ：{m['total_combinations']:,}
- 形成期の組み合わせ別有効行：{m['formation_valid_rows_min']:,}〜{m['formation_valid_rows_max']:,}
- 検証期の組み合わせ別有効行：{m['validation_valid_rows_min']:,}〜{m['validation_valid_rows_max']:,}
- 組み合わせ欠損率：中央値{m['combination_missing_rate_median']:.6%}、最大{m['combination_missing_rate_max']:.6%}
- 期間・境界エラー：{m['period_or_boundary_errors']:,}
- 汚染期間サンプル：{m['contaminated_samples']:,}
- 未来情報候補：{len(m['forbidden_candidate_features']):,}
- 主要キー重複：{m['major_key_duplicates']:,}
- 入力変更：{not report['inputs_unchanged']}
- 2回実行のParquet一致：{report['reproducibility_passed']}
- 保存容量：{report['file_size_bytes']:,} bytes

候補、方向、境界、相関除去、組み合わせは形成期だけで固定しました。検証期は固定後の評価にだけ使用しています。参考順位も形成期の支持件数とリフトだけで並べ、検証期成績は固定後の記述値として付記しました。

ケース・コントロール上のイベント率は実市場の無条件発生確率ではありません。今回は参考組み合わせの記述だけであり、採用シグナル、閾値採用、エントリー、出口、資金管理、売買ルールではありません。

旧STEP7〜旧STEP11は再利用禁止です。2025-09-08以降は完全未使用OOSへ戻していません。

{report['completion_statement']}
"""


def main() -> int:
    options = parse_args()
    repository = Path(__file__).resolve().parents[2]
    root = (repository / "momentum5d" / options.root).resolve()
    report_path = root / "quality/step6_v2_report.json"
    if not report_path.exists():
        raise RuntimeError(f"STEP6 V2 report missing: {report_path}")
    step6_report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_step6(step6_report)
    groups = [
        ("step6_v2", root / "analysis/step6_univariate_v2"),
        ("step6_v2_report", report_path),
    ]
    old_step7 = root / "analysis/step7_condition_combinations"
    if old_step7.exists():
        groups.append(("old_step7_audit_only", old_step7))
    before = manifest(groups)
    target = root / "analysis/step7_condition_combinations_v2"
    quality = root / "quality"
    if target.exists() or (quality / "step7_v2_report.json").exists():
        raise RuntimeError("STEP7 V2 immutable output already exists")

    temporary = Path(tempfile.mkdtemp(prefix="step7_v2_", dir=root.parent))
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
        "period_or_boundary_errors": metrics["period_or_boundary_errors"],
        "contaminated_samples": metrics["contaminated_samples"],
        "forbidden_candidate_features": len(metrics["forbidden_candidate_features"]),
        "selected_strong_correlation_pairs": metrics["selected_strong_correlation_pairs"],
        "definition_threshold_mismatches": metrics["definition_threshold_mismatches"],
        "major_key_duplicates": metrics["major_key_duplicates"],
        "nonfinite_output_numeric_values": metrics["nonfinite_output_numeric_values"],
    }
    passed = (
        all(value == 0 for value in hard_failures.values()) and inputs_unchanged
        and options.verify_reproducibility and reproducible and metrics["parquet_reread_passed"]
        and not metrics["validation_used_for_candidate_selection"]
        and not metrics["validation_used_for_correlation"]
        and not metrics["validation_used_for_definition"]
        and not metrics["validation_thresholds_reestimated"]
        and metrics["input_pairs"] == 8611
        and metrics["input_events"] == metrics["input_controls"] == 8611
        and metrics["input_tickers"] == 2637 and metrics["total_combinations"] > 0
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    first.rename(target)
    report = {
        "step": 7, "version": 2, "status": "STEP7 V2 complete" if passed else "STEP7 V2 FAIL",
        "quality": "PASS" if passed else "FAIL", "created_at_jst": datetime.now(JST).isoformat(),
        "formation": {"start": "data_start", "end": "2023-12-31"},
        "validation": {"start": "2024-01-01", "end": "2025-09-07"},
        "contaminated_holdout_start": "2025-09-08", "new_untouched_oos_start": None,
        "steps1_to_6_v2_recalculated": False, "old_step7_used_for_v2_selection": False,
        "entry_exit_money_management_created": False, "signal_or_trading_rule_adopted": False,
        "candidate_selection_source": "formation_only", "correlation_source": "formation_only",
        "threshold_source": "STEP6 V2 formation coarse_30_70",
        "validation_used_after_definition_freeze_only": True,
        "coarse_candidate_policy": {
            "formation_fixed_auc_min": FORMATION_AUC_MIN,
            "formation_missing_rate_max": FORMATION_MISSING_MAX,
            "formation_absolute_smd_min": FORMATION_ABS_SMD_MIN,
            "formation_year_reproduction_rate_min": FORMATION_YEAR_REPRO_MIN,
            "absolute_spearman_pruning_limit": CORRELATION_LIMIT,
            "correlation_min_pairs": CORRELATION_MIN_PAIRS,
        },
        "input_manifest_before": before, "input_manifest_after": after,
        "inputs_unchanged": inputs_unchanged, "reproducibility_requested": options.verify_reproducibility,
        "reproducibility_passed": reproducible, "first_output_manifest": first_hashes,
        "second_output_manifest": second_hashes, "hard_failures": hard_failures,
        "metrics": metrics, "save_path": str(target), "file_size_bytes": directory_size(target),
        "completion_statement": "STEP7 V2で作成した成果物は、同じ入力版では今後再計算不要" if passed else "STEP7 V2はFAILのため再計算不要とは認証しない",
        "residual_risks": [
            "case-control event rates are not unconditional market probabilities",
            "2,300 combinations create substantial multiple-testing risk even without validation tuning",
            "formation thresholds are coarse policy choices and not proof of economic optimality",
            "repeated tickers and shared market regimes reduce effective sample independence",
            "validation results are descriptive only and no combination is adopted in STEP7 V2",
            "2025-09-08 onward remains contaminated and cannot be reused as untouched OOS",
        ],
    }
    write_json(quality / "step7_v2_report.json", report)
    (quality / "STEP7_V2_REPORT.md").write_text(markdown_report(report), encoding="utf-8")
    (quality / "STEP8_V2_PROMPT.md").write_text(step8_prompt(), encoding="utf-8")
    shutil.rmtree(temporary, ignore_errors=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
