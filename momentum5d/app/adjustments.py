from __future__ import annotations

import numpy as np
import pandas as pd

COMMON_SPLIT_RATIOS = (0.1, 0.2, 0.25, 1 / 3, 0.5, 2.0, 3.0, 4.0, 5.0, 10.0)


def price_adjustment_factor(
    raw_close: pd.Series, adjusted_close: pd.Series
) -> pd.Series:
    """Return the Yahoo price adjustment factor, defaulting invalid rows to 1."""

    factor = pd.to_numeric(adjusted_close, errors="coerce").div(
        pd.to_numeric(raw_close, errors="coerce").replace(0.0, np.nan)
    )
    return factor.where(factor.gt(0.0)).fillna(1.0)


def split_adjusted_volume(volume: pd.Series, adjustment: pd.Series) -> pd.Series:
    """Express historical volume on the same share-count basis as adjusted prices."""

    raw_volume = pd.to_numeric(volume, errors="coerce")
    safe_adjustment = pd.to_numeric(adjustment, errors="coerce").where(
        lambda values: values.gt(0.0)
    )
    return raw_volume.div(safe_adjustment).replace([np.inf, -np.inf], np.nan)


def _inferred_split_ratio(frame: pd.DataFrame) -> pd.Series:
    """Infer an effective split ratio only when the overnight jump is unambiguous."""

    close = pd.to_numeric(frame["close"], errors="coerce")
    open_price = pd.to_numeric(frame["open"], errors="coerce")
    previous_close = close.groupby(frame["ticker"], sort=False).shift(1)
    overnight_ratio = previous_close.div(open_price.replace(0.0, np.nan))
    result = pd.Series(1.0, index=frame.index, dtype="float64")
    best_error = (overnight_ratio - 1.0).abs()
    for candidate in COMMON_SPLIT_RATIOS:
        candidate_error = (overnight_ratio / candidate - 1.0).abs()
        matched = candidate_error.le(0.08) & candidate_error.lt(best_error)
        result.loc[matched] = candidate
        best_error.loc[matched] = candidate_error.loc[matched]
    result.loc[(overnight_ratio - 1.0).abs().lt(0.20)] = 1.0
    return result


def split_share_factor(prices: pd.DataFrame) -> pd.Series:
    """Return the cumulative share factor needed to express rows on today's basis.

    Yahoo ``adjusted_close / close`` is deliberately not used because it also
    contains dividend adjustments.  Explicit split events are preferred and an
    overnight-price inference is only a fallback for missing corporate actions.
    A 2-for-1 event therefore gives historical rows a factor of 2: prices are
    divided by 2 and volume is multiplied by 2.
    """

    required = {"ticker", "date", "open", "close"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Split adjustment is missing columns: {sorted(missing)}")
    if prices.empty:
        return pd.Series(dtype="float64", index=prices.index)

    frame = prices.sort_values(["ticker", "date"]).copy()
    inferred = _inferred_split_ratio(frame)
    if "stock_splits" in frame.columns:
        explicit = pd.to_numeric(frame["stock_splits"], errors="coerce")
        valid_explicit = explicit.gt(0.0) & ~np.isclose(explicit, 1.0)
        event_ratio = inferred.where(~valid_explicit, explicit)
    else:
        event_ratio = inferred
    event_ratio = event_ratio.where(event_ratio.gt(0.0), 1.0).fillna(1.0)

    def future_product(values: pd.Series) -> pd.Series:
        product = values.iloc[::-1].cumprod().iloc[::-1]
        return product.shift(-1, fill_value=1.0)

    factor = event_ratio.groupby(frame["ticker"], sort=False).transform(future_product)
    return factor.reindex(prices.index).astype("float64")


def normalize_split_adjusted_ohlcv(prices: pd.DataFrame) -> pd.DataFrame:
    """Attach split-only adjusted OHLCV and turnover columns.

    Output columns use an underscore prefix so callers cannot accidentally mix
    raw and adjusted fields.  The transformation is internally consistent:
    adjusted price × adjusted volume preserves raw turnover on every row.
    """

    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Split adjustment is missing columns: {sorted(missing)}")
    result = prices.copy()
    factor = split_share_factor(result)
    result["_split_share_factor"] = factor
    safe_factor = factor.where(factor.gt(0.0))
    for column in ("open", "high", "low", "close"):
        result[f"_{column}"] = pd.to_numeric(
            result[column], errors="coerce"
        ).div(safe_factor)
    result["_volume_raw"] = pd.to_numeric(result["volume"], errors="coerce")
    result["_volume"] = result["_volume_raw"].mul(safe_factor)
    result["_turnover"] = result["_close"].mul(result["_volume"])
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result
