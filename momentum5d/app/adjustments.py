from __future__ import annotations

import numpy as np
import pandas as pd


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
