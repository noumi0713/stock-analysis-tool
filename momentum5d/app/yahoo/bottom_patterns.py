from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

SHAPE_LABELS = {
    "double_bottom": "二番底",
    "capitulation_reversal": "投げ売り反転",
    "compression_base": "底値圏収束",
    "sharp_selloff": "急落継続",
    "rounded_base": "緩やかな底固め",
    "other_swing_low": "その他スイング安値",
}

FEATURE_SPECS = {
    "return_5d": "5日騰落",
    "return_20d": "20日騰落",
    "volume_ratio_5_20": "出来高比(5日/20日)",
    "volatility_10d": "10日ボラ",
    "range_width_10d": "10日値幅",
}


@dataclass(frozen=True)
class BottomPatternConfig:
    lookback_days: int = 20
    horizon_days: int = 5
    swing_radius: int = 2
    dedupe_days: int = 5
    min_rank_samples: int = 30
    prior_strength: float = 20.0


def analyze_bottom_patterns(
    features: pd.DataFrame,
    *,
    config: BottomPatternConfig | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Classify swing lows and estimate historical +5% hit rates by chart shape."""
    config = config or BottomPatternConfig()
    required = {
        "date",
        "ticker",
        "code",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "turnover_value",
        "return_5d",
        "return_20d",
        "volume_ratio_5_20",
        "volatility_10d",
        "range_width_10d",
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"底値パターン分析の必須列がありません: {sorted(missing)}")

    frame = features.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    adjustment_ratio = frame["adjusted_close"] / frame["close"]
    frame["_adjusted_high"] = frame["high"] * adjustment_ratio
    frame["_adjusted_low"] = frame["low"] * adjustment_ratio

    rows: list[dict[str, Any]] = []
    for _, values in frame.groupby("ticker", sort=False):
        values = values.reset_index(drop=True)
        candidate_positions = _swing_low_positions(values, config)
        for position in _dedupe_positions(values, candidate_positions, config.dedupe_days):
            event = _event_row(values, position, config)
            if event is not None:
                rows.append(event)

    events = pd.DataFrame(rows)
    if events.empty:
        return _empty_summary(config), events

    overall_rate = float(events["target_5pct"].mean())
    rankings = _shape_rankings(events, overall_rate, config)
    summary = {
        "method": "confirmed_swing_low_shape_v1",
        "definition": (
            "前後2営業日の安値より低いスイング安値を底値とし、近接する底は5営業日単位で"
            "重複除外。底値日の調整後安値から次の5営業日の調整後高値が+5%以上なら達成。"
        ),
        "lookback_days": config.lookback_days,
        "horizon_days": config.horizon_days,
        "swing_confirmation_days": config.swing_radius,
        "events": int(len(events)),
        "successes": int(events["target_5pct"].sum()),
        "failures": int((~events["target_5pct"]).sum()),
        "overall_success_rate": overall_rate,
        "rankings": rankings,
        "liquidity_shape_cross": _liquidity_shape_cross(events),
    }
    return summary, events


def _swing_low_positions(values: pd.DataFrame, config: BottomPatternConfig) -> list[int]:
    lows = values["_adjusted_low"].to_numpy(dtype="float64")
    highs = values["_adjusted_high"].to_numpy(dtype="float64")
    positions: list[int] = []
    start = max(config.lookback_days, config.swing_radius)
    stop = len(values) - max(config.horizon_days, config.swing_radius)
    for position in range(start, stop):
        current_low = lows[position]
        if not np.isfinite(current_low):
            continue
        neighbors = np.concatenate(
            [
                lows[position - config.swing_radius : position],
                lows[position + 1 : position + config.swing_radius + 1],
            ]
        )
        if neighbors.size == 0 or not np.isfinite(neighbors).all():
            continue
        prior_high = np.nanmax(highs[position - config.lookback_days : position])
        meaningful_drawdown = current_low / prior_high - 1 <= -0.04
        if current_low < neighbors.min() and meaningful_drawdown:
            positions.append(position)
    return positions


def _dedupe_positions(
    values: pd.DataFrame,
    positions: list[int],
    dedupe_days: int,
) -> list[int]:
    if not positions:
        return []
    lows = values["_adjusted_low"].to_numpy(dtype="float64")
    clusters: list[list[int]] = [[positions[0]]]
    for position in positions[1:]:
        if position - clusters[-1][-1] <= dedupe_days:
            clusters[-1].append(position)
        else:
            clusters.append([position])
    return [min(cluster, key=lambda position: lows[position]) for cluster in clusters]


def _event_row(
    values: pd.DataFrame,
    position: int,
    config: BottomPatternConfig,
) -> dict[str, Any] | None:
    row = values.iloc[position]
    current_low = float(row["_adjusted_low"])
    future = values.iloc[position + 1 : position + config.horizon_days + 1]
    if len(future) < config.horizon_days:
        return None
    future_highs = future["_adjusted_high"].to_numpy(dtype="float64")
    future_lows = future["_adjusted_low"].to_numpy(dtype="float64")
    if (
        not np.isfinite(future_highs).all()
        or not np.isfinite(future_lows).all()
        or current_low <= 0
    ):
        return None

    target_price = current_low * 1.05
    hits = np.flatnonzero(future_highs >= target_price)
    max_return = float(np.max(future_highs) / current_low - 1)
    min_return = float(np.min(future_lows) / current_low - 1)
    turnover_value = _number_or_none(row["turnover_value"])
    previous = values.iloc[position - config.lookback_days + 1 : position + 1]
    normalized_shape = (
        previous["adjusted_close"] / float(previous["adjusted_close"].iloc[0]) * 100
    )
    shape_key = _classify_shape(values, position, current_low)
    return {
        "date": str(row["date"]),
        "ticker": str(row["ticker"]),
        "code": str(row["code"]),
        "shape": shape_key,
        "shape_label": SHAPE_LABELS[shape_key],
        "bottom_price": current_low,
        "max_return_5d": max_return,
        "min_return_5d": min_return,
        "turnover_value": turnover_value,
        "broke_bottom_5d": min_return < 0,
        "continued_down_3pct": min_return <= -0.03,
        "continued_down_5pct": min_return <= -0.05,
        "target_5pct": bool(hits.size),
        "days_to_5pct": int(hits[0] + 1) if hits.size else None,
        "return_5d": _number_or_none(row["return_5d"]),
        "return_20d": _number_or_none(row["return_20d"]),
        "volume_ratio_5_20": _number_or_none(row["volume_ratio_5_20"]),
        "volatility_10d": _number_or_none(row["volatility_10d"]),
        "range_width_10d": _number_or_none(row["range_width_10d"]),
        "pre_shape": [round(float(value), 4) for value in normalized_shape],
    }


def _classify_shape(values: pd.DataFrame, position: int, current_low: float) -> str:
    row = values.iloc[position]
    prior = values.iloc[max(0, position - 20) : max(0, position - 4)]
    prior_low = float(prior["_adjusted_low"].min()) if not prior.empty else np.nan
    double_bottom = (
        np.isfinite(prior_low)
        and prior_low > 0
        and abs(current_low / prior_low - 1) <= 0.03
    )
    return_5d = _number(row["return_5d"])
    return_20d = _number(row["return_20d"])
    volume_ratio = _number(row["volume_ratio_5_20"])
    volatility = _number(row["volatility_10d"])
    range_width = _number(row["range_width_10d"])
    candle_span = float(row["high"] - row["low"])
    close_location = (
        float(row["close"] - row["low"]) / candle_span if candle_span > 0 else 0.5
    )

    if double_bottom:
        return "double_bottom"
    if return_20d <= -0.10 and volume_ratio >= 1.20 and close_location >= 0.55:
        return "capitulation_reversal"
    if range_width <= 0.07 and volatility <= 0.02:
        return "compression_base"
    if return_5d <= -0.06:
        return "sharp_selloff"
    if return_20d <= -0.05 and return_5d >= -0.03:
        return "rounded_base"
    return "other_swing_low"


def _shape_rankings(
    events: pd.DataFrame,
    overall_rate: float,
    config: BottomPatternConfig,
) -> list[dict[str, Any]]:
    rankings: list[dict[str, Any]] = []
    for shape, values in events.groupby("shape", sort=False):
        samples = int(len(values))
        successes = int(values["target_5pct"].sum())
        success_rate = successes / samples
        lower, upper = _wilson_interval(successes, samples)
        smoothed = (
            successes + overall_rate * config.prior_strength
        ) / (samples + config.prior_strength)
        reached = values.loc[values["target_5pct"], "days_to_5pct"].dropna()
        rankings.append(
            {
                "shape": shape,
                "shape_label": SHAPE_LABELS[shape],
                "samples": samples,
                "successes": successes,
                "failures": samples - successes,
                "success_rate": success_rate,
                "smoothed_success_rate": float(smoothed),
                "wilson_95_low": lower,
                "wilson_95_high": upper,
                "lift_vs_overall": success_rate / overall_rate if overall_rate > 0 else None,
                "median_max_return_5d": float(values["max_return_5d"].median()),
                "median_days_to_5pct": float(reached.median()) if not reached.empty else None,
                "reliable_sample": samples >= config.min_rank_samples,
                "feature_comparison": _feature_comparison(values),
                "subtypes": _shape_subtypes(values, success_rate, config),
                "success_examples": _examples(values.loc[values["target_5pct"]], True),
                "failure_examples": _examples(values.loc[~values["target_5pct"]], False),
            }
        )
    rankings.sort(
        key=lambda item: (
            not item["reliable_sample"],
            -item["wilson_95_low"],
            -item["smoothed_success_rate"],
            -item["samples"],
        )
    )
    for rank, item in enumerate(rankings, start=1):
        item["rank"] = rank
    return rankings



def _liquidity_shape_cross(events: pd.DataFrame) -> list[dict[str, Any]]:
    """Cross-tab chart shapes by absolute turnover and post-bottom downside."""
    thresholds = (50_000_000.0, 100_000_000.0, 200_000_000.0)
    pullback_shapes = {"double_bottom", "compression_base"}
    groups: list[tuple[str, str, pd.DataFrame]] = [
        (shape, SHAPE_LABELS[shape], values)
        for shape, values in events.groupby("shape", sort=False)
    ]
    groups.append(
        (
            "pullback_candidate",
            "押し目候補（二番底＋底値圏収束）",
            events.loc[events["shape"].isin(pullback_shapes)],
        )
    )

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        for shape, shape_label, values in groups:
            turnover = pd.to_numeric(values["turnover_value"], errors="coerce")
            liquid = values.loc[turnover >= threshold]
            samples = int(len(liquid))
            if samples == 0:
                continue
            successes = int(liquid["target_5pct"].sum())
            failures = samples - successes
            broke_bottom = int(liquid["broke_bottom_5d"].sum())
            down_3pct = int(liquid["continued_down_3pct"].sum())
            down_5pct = int(liquid["continued_down_5pct"].sum())
            misses = liquid.loc[~liquid["target_5pct"]]
            miss_down_3pct = int(misses["continued_down_3pct"].sum())
            miss_down_5pct = int(misses["continued_down_5pct"].sum())
            rows.append(
                {
                    "turnover_threshold": int(threshold),
                    "shape": shape,
                    "shape_label": shape_label,
                    "samples": samples,
                    "successes": successes,
                    "failures": failures,
                    "success_rate": successes / samples,
                    "failure_rate": failures / samples,
                    "broke_bottom_5d": broke_bottom,
                    "broke_bottom_rate": broke_bottom / samples,
                    "continued_down_3pct": down_3pct,
                    "continued_down_3pct_rate": down_3pct / samples,
                    "continued_down_5pct": down_5pct,
                    "continued_down_5pct_rate": down_5pct / samples,
                    "failure_and_down_3pct": miss_down_3pct,
                    "failure_and_down_3pct_rate": (
                        miss_down_3pct / failures if failures else 0.0
                    ),
                    "failure_and_down_5pct": miss_down_5pct,
                    "failure_and_down_5pct_rate": (
                        miss_down_5pct / failures if failures else 0.0
                    ),
                    "median_min_return_5d": float(liquid["min_return_5d"].median()),
                }
            )
    return rows


def _feature_comparison(values: pd.DataFrame) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    hits = values.loc[values["target_5pct"]]
    misses = values.loc[~values["target_5pct"]]
    for feature, label in FEATURE_SPECS.items():
        all_values = pd.to_numeric(values[feature], errors="coerce").dropna()
        hit_values = pd.to_numeric(hits[feature], errors="coerce").dropna()
        miss_values = pd.to_numeric(misses[feature], errors="coerce").dropna()
        if all_values.empty or hit_values.empty or miss_values.empty:
            continue
        hit_median = float(hit_values.median())
        miss_median = float(miss_values.median())
        q25, q75 = all_values.quantile([0.25, 0.75])
        iqr = float(q75 - q25)
        raw_gap = hit_median - miss_median
        effect_score = raw_gap / iqr if iqr > 0 else 0.0
        comparisons.append(
            {
                "feature": feature,
                "label": label,
                "success_median": hit_median,
                "failure_median": miss_median,
                "median_gap": raw_gap,
                "effect_score_iqr": float(effect_score),
                "success_side": "high" if raw_gap >= 0 else "low",
            }
        )
    comparisons.sort(key=lambda item: -abs(item["effect_score_iqr"]))
    for rank, item in enumerate(comparisons, start=1):
        item["difference_rank"] = rank
    return comparisons


def _shape_subtypes(
    values: pd.DataFrame,
    parent_rate: float,
    config: BottomPatternConfig,
) -> list[dict[str, Any]]:
    comparisons = _feature_comparison(values)
    if len(comparisons) < 2:
        return []
    selected = comparisons[:2]
    first, second = selected[0]["feature"], selected[1]["feature"]
    work = values.dropna(subset=[first, second]).copy()
    if work.empty:
        return []
    thresholds = {
        first: float(pd.to_numeric(work[first], errors="coerce").median()),
        second: float(pd.to_numeric(work[second], errors="coerce").median()),
    }
    work["_subtype"] = work.apply(
        lambda row: (
            ("high" if float(row[first]) >= thresholds[first] else "low"),
            ("high" if float(row[second]) >= thresholds[second] else "low"),
        ),
        axis=1,
    )

    subtypes: list[dict[str, Any]] = []
    for (first_side, second_side), group in work.groupby("_subtype", sort=False):
        samples = int(len(group))
        successes = int(group["target_5pct"].sum())
        rate = successes / samples
        lower, upper = _wilson_interval(successes, samples)
        smoothed = (
            successes + parent_rate * config.prior_strength
        ) / (samples + config.prior_strength)
        subtypes.append(
            {
                "samples": samples,
                "successes": successes,
                "failures": samples - successes,
                "success_rate": rate,
                "smoothed_success_rate": float(smoothed),
                "wilson_95_low": lower,
                "wilson_95_high": upper,
                "lift_vs_parent": rate / parent_rate if parent_rate > 0 else None,
                "reliable_sample": samples >= config.min_rank_samples,
                "conditions": [
                    {
                        "feature": first,
                        "label": FEATURE_SPECS[first],
                        "side": first_side,
                        "threshold": thresholds[first],
                    },
                    {
                        "feature": second,
                        "label": FEATURE_SPECS[second],
                        "side": second_side,
                        "threshold": thresholds[second],
                    },
                ],
            }
        )
    subtypes.sort(
        key=lambda item: (
            not item["reliable_sample"],
            -item["wilson_95_low"],
            -item["smoothed_success_rate"],
            -item["samples"],
        )
    )
    for rank, item in enumerate(subtypes, start=1):
        item["rank"] = rank
    return subtypes


def _examples(values: pd.DataFrame, success: bool) -> list[dict[str, Any]]:
    if values.empty:
        return []
    ordered = values.sort_values("max_return_5d", ascending=False).head(3)
    keys = [
        "date",
        "ticker",
        "code",
        "bottom_price",
        "max_return_5d",
        "days_to_5pct",
        "pre_shape",
    ]
    examples: list[dict[str, Any]] = []
    for _, row in ordered.iterrows():
        example = {key: row[key] for key in keys}
        example["bottom_price"] = float(example["bottom_price"])
        example["max_return_5d"] = float(example["max_return_5d"])
        days = example["days_to_5pct"]
        example["days_to_5pct"] = int(days) if pd.notna(days) else None
        examples.append(example)
    return examples


def _wilson_interval(successes: int, samples: int, z: float = 1.96) -> tuple[float, float]:
    if samples == 0:
        return 0.0, 0.0
    rate = successes / samples
    denominator = 1 + z**2 / samples
    center = (rate + z**2 / (2 * samples)) / denominator
    margin = z * sqrt(rate * (1 - rate) / samples + z**2 / (4 * samples**2)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _number(value: Any) -> float:
    number = float(value)
    return number if np.isfinite(number) else 0.0


def _number_or_none(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _empty_summary(config: BottomPatternConfig) -> dict[str, Any]:
    return {
        "method": "confirmed_swing_low_shape_v1",
        "definition": "分析可能なスイング安値がありません。",
        "lookback_days": config.lookback_days,
        "horizon_days": config.horizon_days,
        "swing_confirmation_days": config.swing_radius,
        "events": 0,
        "successes": 0,
        "failures": 0,
        "overall_success_rate": None,
        "rankings": [],
        "liquidity_shape_cross": [],
    }
