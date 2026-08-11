from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FEATURES: tuple[tuple[str, str], ...] = (
    ("po_candle_body_return", "当日ローソク実体騰落率"),
    ("po_close_location", "当日終値位置"),
    ("po_upper_wick_ratio", "上ヒゲ比率"),
    ("po_lower_wick_ratio", "下ヒゲ比率"),
    ("range_rate", "当日値幅率"),
    ("return_1d", "1日騰落率"),
    ("po_return_3d", "3日騰落率"),
    ("return_5d", "5日騰落率"),
    ("return_20d", "20日騰落率"),
    ("rsi_14", "RSI14"),
    ("atr_14_pct", "ATR14率"),
    ("volume_change_1d", "出来高前日比"),
    ("po_volume_change_3d", "出来高3日前比"),
    ("volume_ratio_1_20", "当日出来高/直前20日平均"),
    ("volume_ratio_5_20", "5日出来高/20日出来高"),
    ("up_volume_share_10d", "10日上昇日出来高比率"),
    ("po_ma5_deviation", "5日線乖離率"),
    ("po_ma25_deviation", "25日線乖離率"),
    ("po_ma75_deviation", "75日線乖離率"),
    ("po_ma_5_slope", "5日線の5日傾き"),
    ("po_ma_25_slope", "25日線の5日傾き"),
    ("po_ma_75_slope", "75日線の5日傾き"),
    ("po_order_spread", "5日線/75日線の開き"),
    ("po_spread_change_5d", "移動平均線の開き5日変化"),
    ("po_perfect_order_age", "パーフェクトオーダー継続日数"),
    ("breakout_20d", "直前20日高値からの位置"),
    ("relative_return_20d", "市場比20日騰落"),
)


def analyze_pullback_failures(
    candidates_by_rule: dict[str, pd.DataFrame],
    *,
    validation_start: object,
    horizon_days: int,
    stop_loss: float,
    minimum_threshold_samples: int = 30,
) -> dict[str, Any]:
    """Contrast genuine rebounds with pullbacks that keep declining."""
    events = _combine_candidates(candidates_by_rule)
    if events.empty:
        return _empty_report()
    events = _attach_outcomes(
        events,
        horizon_days=horizon_days,
        stop_loss=stop_loss,
    )
    calibration = events.loc[events["date"] < validation_start]
    validation = events.loc[events["date"] >= validation_start]
    conditions = _discover_thresholds(
        calibration,
        validation,
        minimum_samples=minimum_threshold_samples,
    )
    return {
        "method": "perfect_order_pullback_failure_diagnostics_v1",
        "definitions": {
            "rebound_success": "翌日始値から固定-3%より先に+5%到達",
            "stop_first_failure": "+5%より先に固定-3%到達",
            "persistent_decline": ("7営業日で+5%未達、期間安値-5%以下、7日目終値-3%以下"),
            "straight_decline": ("persistent_declineに加え、最初の3日間の終値が連続切り下げ"),
            "same_day_stop_and_target": "stop_first",
        },
        "event_count": len(events),
        "periods": {
            "calibration_end_exclusive": str(validation_start),
            "validation_start": str(validation_start),
        },
        "summary": {
            "all": _outcome_summary(events),
            "calibration": _outcome_summary(calibration),
            "validation": _outcome_summary(validation),
        },
        "by_pullback_rule": _summarize_by_rule(events),
        "feature_comparison": {
            "all": _compare_features(events),
            "calibration": _compare_features(calibration),
            "validation": _compare_features(validation),
        },
        "validated_risk_conditions": conditions,
        "persistent_decline_examples": _worst_examples(events),
        "interpretation_note": (
            "条件は前半70%だけで発見し、同じ閾値を後半30%へ固定して再検証。"
            "後半でも下落継続率が全体より高い条件だけを掲載"
        ),
    }


def _combine_candidates(
    candidates_by_rule: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    frames = [
        frame.assign(po_matched_rule=rule)
        for rule, frame in candidates_by_rule.items()
        if not frame.empty
    ]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    rule_matches = (
        combined.groupby(["ticker", "date"], sort=False)["po_matched_rule"]
        .agg(lambda values: ",".join(sorted(set(values))))
        .rename("po_matched_rules")
    )
    result = (
        combined.sort_values(
            ["ticker", "date", "po_pullback_score"],
            ascending=[True, True, False],
        )
        .drop_duplicates(["ticker", "date"])
        .drop(columns="po_matched_rule")
        .merge(rule_matches, on=["ticker", "date"], how="left")
    )
    result["po_matched_rule_count"] = result["po_matched_rules"].str.count(",") + 1
    return result.reset_index(drop=True)


def _attach_outcomes(
    events: pd.DataFrame,
    *,
    horizon_days: int,
    stop_loss: float,
) -> pd.DataFrame:
    result = events.copy()
    entry = pd.to_numeric(result["entry_price"], errors="coerce")
    stop = entry * (1.0 - stop_loss)
    target = entry * 1.05
    first_stop_day = pd.Series(pd.NA, index=result.index, dtype="Int8")
    first_target_day = pd.Series(pd.NA, index=result.index, dtype="Int8")
    unresolved = pd.Series(True, index=result.index, dtype="bool")
    all_highs: list[pd.Series] = []
    all_lows: list[pd.Series] = []
    for day in range(1, horizon_days + 1):
        open_ = pd.to_numeric(result[f"future_open_{day}"], errors="coerce")
        high = pd.to_numeric(result[f"future_high_{day}"], errors="coerce")
        low = pd.to_numeric(result[f"future_low_{day}"], errors="coerce")
        all_highs.append(high)
        all_lows.append(low)
        stopped = unresolved & (open_.le(stop) | low.le(stop))
        first_stop_day.loc[stopped] = day
        unresolved &= ~stopped
        targeted = unresolved & high.ge(target)
        first_target_day.loc[targeted] = day
        unresolved &= ~targeted

    maximum_high = pd.concat(all_highs, axis=1).max(axis=1)
    minimum_low = pd.concat(all_lows, axis=1).min(axis=1)
    final_close = pd.to_numeric(result[f"future_close_{horizon_days}"], errors="coerce")
    first_three_closes = [
        pd.to_numeric(result[f"future_close_{day}"], errors="coerce")
        for day in range(1, min(3, horizon_days) + 1)
    ]
    descending_three = pd.Series(False, index=result.index, dtype="bool")
    if len(first_three_closes) == 3:
        descending_three = (
            first_three_closes[0].lt(entry)
            & first_three_closes[1].lt(first_three_closes[0])
            & first_three_closes[2].lt(first_three_closes[1])
        )
    result["po_first_stop_day"] = first_stop_day
    result["po_first_target_day"] = first_target_day
    result["po_maximum_return_7d"] = maximum_high / entry - 1.0
    result["po_minimum_return_7d"] = minimum_low / entry - 1.0
    result["po_final_return_7d"] = final_close / entry - 1.0
    result["po_rebound_success"] = first_target_day.notna()
    result["po_stop_first_failure"] = first_stop_day.notna()
    result["po_persistent_decline"] = (
        maximum_high.lt(target)
        & result["po_minimum_return_7d"].le(-0.05)
        & result["po_final_return_7d"].le(-0.03)
    )
    result["po_straight_decline"] = result["po_persistent_decline"] & descending_three
    return result


def _outcome_summary(events: pd.DataFrame) -> dict[str, Any]:
    if events.empty:
        return {
            "events": 0,
            "rebound_success_rate": None,
            "stop_first_failure_rate": None,
            "persistent_decline_rate": None,
            "straight_decline_rate": None,
        }
    return {
        "events": len(events),
        "rebound_success_count": int(events["po_rebound_success"].sum()),
        "rebound_success_rate": float(events["po_rebound_success"].mean()),
        "stop_first_failure_count": int(events["po_stop_first_failure"].sum()),
        "stop_first_failure_rate": float(events["po_stop_first_failure"].mean()),
        "persistent_decline_count": int(events["po_persistent_decline"].sum()),
        "persistent_decline_rate": float(events["po_persistent_decline"].mean()),
        "straight_decline_count": int(events["po_straight_decline"].sum()),
        "straight_decline_rate": float(events["po_straight_decline"].mean()),
        "median_minimum_return_7d": _median(events["po_minimum_return_7d"]),
        "median_final_return_7d": _median(events["po_final_return_7d"]),
    }


def _summarize_by_rule(events: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule in sorted({rule for value in events["po_matched_rules"] for rule in value.split(",")}):
        subset = events.loc[
            events["po_matched_rules"]
            .str.split(",")
            .apply(lambda values, current_rule=rule: current_rule in values)
        ]
        rows.append({"pullback_rule": rule, **_outcome_summary(subset)})
    return rows


def _compare_features(events: pd.DataFrame) -> list[dict[str, Any]]:
    success = events.loc[events["po_rebound_success"]]
    decline = events.loc[events["po_persistent_decline"]]
    rows: list[dict[str, Any]] = []
    for feature, label in FEATURES:
        if feature not in events:
            continue
        success_values = pd.to_numeric(success[feature], errors="coerce").dropna()
        decline_values = pd.to_numeric(decline[feature], errors="coerce").dropna()
        if success_values.empty or decline_values.empty:
            continue
        pooled_variance = (success_values.var(ddof=1) + decline_values.var(ddof=1)) / 2.0
        standardized_difference = (
            (decline_values.mean() - success_values.mean()) / np.sqrt(pooled_variance)
            if pooled_variance > 0
            else 0.0
        )
        rows.append(
            {
                "feature": feature,
                "label": label,
                "success_samples": len(success_values),
                "persistent_decline_samples": len(decline_values),
                "success_median": float(success_values.median()),
                "persistent_decline_median": float(decline_values.median()),
                "median_difference": float(decline_values.median() - success_values.median()),
                "standardized_difference": float(standardized_difference),
            }
        )
    return sorted(
        rows,
        key=lambda row: abs(row["standardized_difference"]),
        reverse=True,
    )


def _discover_thresholds(
    calibration: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    minimum_samples: int,
) -> list[dict[str, Any]]:
    if calibration.empty or validation.empty:
        return []
    calibration_base = float(calibration["po_persistent_decline"].mean())
    validation_base = float(validation["po_persistent_decline"].mean())
    if calibration_base <= 0 or validation_base <= 0:
        return []
    discoveries: list[dict[str, Any]] = []
    for feature, label in FEATURES:
        if feature not in calibration or feature not in validation:
            continue
        calibration_values = pd.to_numeric(calibration[feature], errors="coerce")
        valid_values = calibration_values.dropna()
        if len(valid_values) < minimum_samples * 2:
            continue
        candidates: list[dict[str, Any]] = []
        for quantile, operator in ((0.2, "le"), (0.3, "le"), (0.7, "ge"), (0.8, "ge")):
            threshold = float(valid_values.quantile(quantile))
            calibration_mask = _threshold_mask(calibration_values, operator, threshold)
            calibration_result = _condition_summary(
                calibration,
                calibration_mask,
                calibration_base,
            )
            if calibration_result["samples"] < minimum_samples:
                continue
            candidates.append(
                {
                    "feature": feature,
                    "label": label,
                    "operator": operator,
                    "threshold": threshold,
                    "calibration": calibration_result,
                }
            )
        if not candidates:
            continue
        best = max(candidates, key=lambda row: row["calibration"]["risk_lift"])
        if best["calibration"]["risk_lift"] <= 1.0:
            continue
        validation_values = pd.to_numeric(validation[feature], errors="coerce")
        validation_mask = _threshold_mask(
            validation_values,
            best["operator"],
            best["threshold"],
        )
        best["validation"] = _condition_summary(
            validation,
            validation_mask,
            validation_base,
        )
        if (
            best["validation"]["samples"] >= minimum_samples
            and best["validation"]["risk_lift"] > 1.0
        ):
            best["validated_strength"] = min(
                best["calibration"]["risk_lift"],
                best["validation"]["risk_lift"],
            )
            discoveries.append(best)
    return sorted(
        discoveries,
        key=lambda row: row["validated_strength"],
        reverse=True,
    )[:12]


def _condition_summary(
    events: pd.DataFrame,
    mask: pd.Series,
    base_rate: float,
) -> dict[str, Any]:
    subset = events.loc[mask.fillna(False)]
    if subset.empty:
        return {
            "samples": 0,
            "persistent_decline_rate": None,
            "straight_decline_rate": None,
            "stop_first_failure_rate": None,
            "rebound_success_rate": None,
            "risk_lift": 0.0,
        }
    decline_rate = float(subset["po_persistent_decline"].mean())
    return {
        "samples": len(subset),
        "persistent_decline_rate": decline_rate,
        "straight_decline_rate": float(subset["po_straight_decline"].mean()),
        "stop_first_failure_rate": float(subset["po_stop_first_failure"].mean()),
        "rebound_success_rate": float(subset["po_rebound_success"].mean()),
        "risk_lift": decline_rate / base_rate if base_rate > 0 else 0.0,
    }


def _threshold_mask(
    values: pd.Series,
    operator: str,
    threshold: float,
) -> pd.Series:
    return values.le(threshold) if operator == "le" else values.ge(threshold)


def _worst_examples(events: pd.DataFrame) -> list[dict[str, Any]]:
    subset = events.loc[events["po_persistent_decline"]].nsmallest(
        min(12, int(events["po_persistent_decline"].sum())),
        "po_minimum_return_7d",
    )
    columns = [
        "date",
        "ticker",
        "code",
        "po_matched_rules",
        "entry_date",
        "entry_price",
        "po_minimum_return_7d",
        "po_final_return_7d",
        "volume_ratio_1_20",
        "po_close_location",
        "po_perfect_order_age",
    ]
    return [
        {key: value.item() if hasattr(value, "item") else value for key, value in row.items()}
        for row in subset[columns].to_dict("records")
    ]


def _median(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.median()) if not numeric.empty else None


def _empty_report() -> dict[str, Any]:
    return {
        "method": "perfect_order_pullback_failure_diagnostics_v1",
        "event_count": 0,
        "summary": {},
        "by_pullback_rule": [],
        "feature_comparison": {},
        "validated_risk_conditions": [],
        "persistent_decline_examples": [],
        "error": "評価可能な押し目候補がありません",
    }
