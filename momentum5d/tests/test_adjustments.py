from __future__ import annotations

import pandas as pd

from app.adjustments import price_adjustment_factor, split_adjusted_volume


def test_split_adjusted_volume_preserves_turnover_basis() -> None:
    raw_close = pd.Series([5_000.0, 2_500.0])
    adjusted_close = pd.Series([2_500.0, 2_500.0])
    raw_volume = pd.Series([100_000.0, 200_000.0])

    adjustment = price_adjustment_factor(raw_close, adjusted_close)
    adjusted_volume = split_adjusted_volume(raw_volume, adjustment)

    assert adjustment.tolist() == [0.5, 1.0]
    assert adjusted_volume.tolist() == [200_000.0, 200_000.0]
    assert (adjusted_close * adjusted_volume).tolist() == [500_000_000.0, 500_000_000.0]
