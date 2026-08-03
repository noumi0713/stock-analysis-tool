from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

BUY_PATTERNS: tuple[tuple[str, str, float], ...] = (
    ("sakata_rising_three_methods", "上げ三法", 1.00),
    ("sakata_inverse_three_mountains", "逆三山", 0.95),
    ("sakata_morning_star", "三川明けの明星", 0.90),
    ("sakata_three_white_soldiers", "赤三兵", 0.85),
    ("sakata_three_gaps_down", "三空叩き込み", 0.80),
)

SELL_PATTERNS: tuple[tuple[str, str, float], ...] = (
    ("sakata_falling_three_methods", "下げ三法", 1.00),
    ("sakata_three_mountains", "三山", 0.95),
    ("sakata_evening_star", "三川宵の明星", 0.90),
    ("sakata_three_black_crows", "黒三兵", 0.85),
    ("sakata_three_gaps_up", "三空踏み上げ", 0.80),
)

PATTERN_COLUMNS = [column for column, _, _ in (*BUY_PATTERNS, *SELL_PATTERNS)]


def add_sakata_features(features: pd.DataFrame) -> pd.DataFrame:
    """当日までのOHLCだけで酒田五法の買い型・売り型を定量判定する。"""
    required = {"date", "ticker", "open", "high", "low", "close"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"酒田五法判定の必須列がありません: {sorted(missing)}")

    frame = features.sort_values(["ticker", "date"]).copy()
    detected = []
    for _, group in frame.groupby("ticker", sort=False):
        detected.append(_detect_group(group))
    patterns = pd.concat(detected).sort_index() if detected else pd.DataFrame(index=frame.index)
    for column in PATTERN_COLUMNS:
        frame[column] = (
            patterns.get(column, False)
            .reindex(frame.index, fill_value=False)
            .astype(bool)
        )

    buy_strength = _weighted_max(frame, BUY_PATTERNS)
    sell_strength = _weighted_max(frame, SELL_PATTERNS)
    frame["sakata_bullish_count"] = frame[[item[0] for item in BUY_PATTERNS]].sum(axis=1)
    frame["sakata_bearish_count"] = frame[[item[0] for item in SELL_PATTERNS]].sum(axis=1)
    frame["sakata_buy_signal"] = frame["sakata_bullish_count"] > 0
    frame["sakata_sell_signal"] = frame["sakata_bearish_count"] > 0
    frame["sakata_score"] = (
        0.35
        + 0.65 * buy_strength
        + 0.03 * (frame["sakata_bullish_count"] - 1).clip(lower=0)
        - 0.70 * sell_strength
    ).clip(0.0, 1.0)
    frame["sakata_pattern"] = _pattern_labels(frame, (*BUY_PATTERNS, *SELL_PATTERNS))
    return frame.sort_values(["ticker", "date"]).reset_index(drop=True)


def _detect_group(group: pd.DataFrame) -> pd.DataFrame:
    values = group.sort_values("date")
    open_ = pd.to_numeric(values["open"], errors="coerce")
    high = pd.to_numeric(values["high"], errors="coerce")
    low = pd.to_numeric(values["low"], errors="coerce")
    close = pd.to_numeric(values["close"], errors="coerce")
    span = (high - low).replace(0, np.nan)
    body = (close - open_).abs()
    body_ratio = body / span
    bullish = close > open_
    bearish = close < open_
    long_body = body_ratio >= 0.55
    small_body = body_ratio <= 0.35

    prior_down = close.shift(2) / close.shift(7) - 1 <= -0.03
    prior_up = close.shift(2) / close.shift(7) - 1 >= 0.03
    first_midpoint = (open_.shift(2) + close.shift(2)) / 2
    morning_star = (
        prior_down
        & bearish.shift(2, fill_value=False)
        & long_body.shift(2, fill_value=False)
        & small_body.shift(1, fill_value=False)
        & (pd.concat([open_.shift(1), close.shift(1)], axis=1).max(axis=1) <= close.shift(2))
        & bullish
        & long_body
        & (close >= first_midpoint)
    )
    evening_star = (
        prior_up
        & bullish.shift(2, fill_value=False)
        & long_body.shift(2, fill_value=False)
        & small_body.shift(1, fill_value=False)
        & (pd.concat([open_.shift(1), close.shift(1)], axis=1).min(axis=1) >= close.shift(2))
        & bearish
        & long_body
        & (close <= first_midpoint)
    )

    upper_shadow_ratio = (high - pd.concat([open_, close], axis=1).max(axis=1)) / span
    lower_shadow_ratio = (pd.concat([open_, close], axis=1).min(axis=1) - low) / span
    opens_in_previous_body = (
        open_ >= pd.concat([open_.shift(1), close.shift(1)], axis=1).min(axis=1)
    ) & (open_ <= pd.concat([open_.shift(1), close.shift(1)], axis=1).max(axis=1))
    red_soldiers = (
        bullish
        & bullish.shift(1, fill_value=False)
        & bullish.shift(2, fill_value=False)
        & (close > close.shift(1))
        & (close.shift(1) > close.shift(2))
        & opens_in_previous_body
        & opens_in_previous_body.shift(1, fill_value=False)
        & (upper_shadow_ratio <= 0.40)
        & (upper_shadow_ratio.shift(1) <= 0.40)
        & (upper_shadow_ratio.shift(2) <= 0.40)
        & (close.shift(3) / close.shift(8) - 1 <= 0.03)
    )
    black_crows = (
        bearish
        & bearish.shift(1, fill_value=False)
        & bearish.shift(2, fill_value=False)
        & (close < close.shift(1))
        & (close.shift(1) < close.shift(2))
        & opens_in_previous_body
        & opens_in_previous_body.shift(1, fill_value=False)
        & (lower_shadow_ratio <= 0.40)
        & (lower_shadow_ratio.shift(1) <= 0.40)
        & (lower_shadow_ratio.shift(2) <= 0.40)
        & (close.shift(3) / close.shift(8) - 1 >= -0.03)
    )

    gap_up = low > high.shift(1)
    gap_down = high < low.shift(1)
    three_gaps_up = gap_up & gap_up.shift(1, fill_value=False) & gap_up.shift(2, fill_value=False)
    three_gaps_down = (
        gap_down & gap_down.shift(1, fill_value=False) & gap_down.shift(2, fill_value=False)
    )

    first_open = open_.shift(4)
    first_close = close.shift(4)
    first_high = high.shift(4)
    first_low = low.shift(4)
    first_body = (first_close - first_open).abs()
    inside = []
    small_vs_first = []
    for offset in (1, 2, 3):
        inside.append((high.shift(offset) <= first_high) & (low.shift(offset) >= first_low))
        small_vs_first.append(
            (close.shift(offset) - open_.shift(offset)).abs() <= first_body * 0.60
        )
    middle_bearish = sum(
        bearish.shift(offset, fill_value=False).astype(int) for offset in (1, 2, 3)
    )
    middle_bullish = sum(
        bullish.shift(offset, fill_value=False).astype(int) for offset in (1, 2, 3)
    )
    rising_methods = (
        bullish.shift(4, fill_value=False)
        & long_body.shift(4, fill_value=False)
        & inside[0]
        & inside[1]
        & inside[2]
        & small_vs_first[0]
        & small_vs_first[1]
        & small_vs_first[2]
        & (middle_bearish >= 2)
        & bullish
        & long_body
        & (close > first_high)
    )
    falling_methods = (
        bearish.shift(4, fill_value=False)
        & long_body.shift(4, fill_value=False)
        & inside[0]
        & inside[1]
        & inside[2]
        & small_vs_first[0]
        & small_vs_first[1]
        & small_vs_first[2]
        & (middle_bullish >= 2)
        & bearish
        & long_body
        & (close < first_low)
    )

    triple_top, triple_bottom = _triple_formations(high, low, close)
    return pd.DataFrame(
        {
            "sakata_inverse_three_mountains": triple_bottom,
            "sakata_three_mountains": triple_top,
            "sakata_morning_star": morning_star.fillna(False),
            "sakata_evening_star": evening_star.fillna(False),
            "sakata_three_gaps_down": three_gaps_down.fillna(False),
            "sakata_three_gaps_up": three_gaps_up.fillna(False),
            "sakata_three_white_soldiers": red_soldiers.fillna(False),
            "sakata_three_black_crows": black_crows.fillna(False),
            "sakata_rising_three_methods": rising_methods.fillna(False),
            "sakata_falling_three_methods": falling_methods.fillna(False),
        },
        index=values.index,
    )


def _triple_formations(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    lookback: int = 35,
    tolerance: float = 0.035,
) -> tuple[pd.Series, pd.Series]:
    highs = high.to_numpy(dtype="float64")
    lows = low.to_numpy(dtype="float64")
    closes = close.to_numpy(dtype="float64")
    peaks = [
        i
        for i in range(1, len(highs) - 1)
        if highs[i] >= highs[i - 1] and highs[i] > highs[i + 1]
    ]
    valleys = [
        i
        for i in range(1, len(lows) - 1)
        if lows[i] <= lows[i - 1] and lows[i] < lows[i + 1]
    ]
    triple_top = np.zeros(len(highs), dtype=bool)
    triple_bottom = np.zeros(len(highs), dtype=bool)
    for i in range(7, len(highs)):
        recent_peaks = [value for value in peaks if i - lookback <= value <= i - 1]
        if len(recent_peaks) >= 3:
            p1, p2, p3 = recent_peaks[-3:]
            levels = highs[[p1, p2, p3]]
            if p2 - p1 >= 2 and p3 - p2 >= 2 and np.ptp(levels) / np.mean(levels) <= tolerance:
                neckline = min(np.min(lows[p1 : p2 + 1]), np.min(lows[p2 : p3 + 1]))
                triple_top[i] = closes[i] < neckline <= closes[i - 1]
        recent_valleys = [value for value in valleys if i - lookback <= value <= i - 1]
        if len(recent_valleys) >= 3:
            v1, v2, v3 = recent_valleys[-3:]
            levels = lows[[v1, v2, v3]]
            if v2 - v1 >= 2 and v3 - v2 >= 2 and np.ptp(levels) / np.mean(levels) <= tolerance:
                neckline = max(np.max(highs[v1 : v2 + 1]), np.max(highs[v2 : v3 + 1]))
                triple_bottom[i] = closes[i] > neckline >= closes[i - 1]
    return (
        pd.Series(triple_top, index=high.index),
        pd.Series(triple_bottom, index=high.index),
    )


def _weighted_max(frame: pd.DataFrame, patterns: Sequence[tuple[str, str, float]]) -> pd.Series:
    weighted = pd.concat(
        [frame[column].astype(float) * weight for column, _, weight in patterns],
        axis=1,
    )
    return weighted.max(axis=1)


def _pattern_labels(
    frame: pd.DataFrame,
    patterns: Sequence[tuple[str, str, float]],
) -> pd.Series:
    labels = pd.Series("", index=frame.index, dtype="string")
    for column, label, _ in patterns:
        matched = frame[column].fillna(False)
        separator = labels.ne("").map({True: "・", False: ""})
        labels = labels + (separator + label).where(matched, "")
    return labels.mask(labels.eq(""), "該当なし")
