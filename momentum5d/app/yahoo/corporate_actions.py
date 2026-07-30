from __future__ import annotations

import numpy as np
import pandas as pd

COMMON_SPLIT_FACTORS = (0.1, 0.2, 0.25, 1 / 3, 0.5, 2.0, 3.0, 4.0, 5.0, 10.0)


def detect_effective_split_factors(prices: pd.DataFrame) -> pd.Series:
    """前日終値と当日始値の不連続から、価格へ反映された分割比率を推定する。"""
    ordered = prices.sort_values(["ticker", "date"])
    close = pd.to_numeric(ordered["close"], errors="coerce")
    open_price = pd.to_numeric(ordered["open"], errors="coerce")
    previous_close = close.groupby(ordered["ticker"], sort=False).shift(1)
    overnight_ratio = previous_close / open_price
    factors = pd.Series(1.0, index=ordered.index, dtype="float64")
    best_error = (overnight_ratio - 1.0).abs()
    for candidate in COMMON_SPLIT_FACTORS:
        candidate_error = (overnight_ratio / candidate - 1.0).abs()
        matched = (candidate_error <= 0.08) & (candidate_error < best_error)
        factors.loc[matched] = candidate
        best_error.loc[matched] = candidate_error.loc[matched]
    factors.loc[(overnight_ratio - 1.0).abs() < 0.20] = 1.0
    return factors.reindex(prices.index).fillna(1.0)


def normalize_split_adjusted_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """実効分割日を使い、価格と出来高を最新株式数ベースへ連続化する。"""
    if prices.empty:
        return prices.copy()
    frame = prices.sort_values(["ticker", "date"]).copy()
    frame["effective_split_factor"] = detect_effective_split_factors(frame)

    def future_product(values: pd.Series) -> pd.Series:
        reversed_product = values.iloc[::-1].cumprod().iloc[::-1]
        return reversed_product.shift(-1, fill_value=1.0)

    adjustment = frame.groupby("ticker", sort=False)["effective_split_factor"].transform(
        future_product
    )
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce") / adjustment
    frame["adjusted_close"] = frame["close"]
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce") * adjustment
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame.sort_index()
