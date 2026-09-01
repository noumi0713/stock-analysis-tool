from __future__ import annotations

import pandas as pd

from app.adjustments import (
    normalize_split_adjusted_ohlcv,
    price_adjustment_factor,
    split_adjusted_volume,
)


def test_split_adjusted_volume_preserves_turnover_basis() -> None:
    raw_close = pd.Series([5_000.0, 2_500.0])
    adjusted_close = pd.Series([2_500.0, 2_500.0])
    raw_volume = pd.Series([100_000.0, 200_000.0])

    adjustment = price_adjustment_factor(raw_close, adjusted_close)
    adjusted_volume = split_adjusted_volume(raw_volume, adjustment)

    assert adjustment.tolist() == [0.5, 1.0]
    assert adjusted_volume.tolist() == [200_000.0, 200_000.0]
    assert (adjusted_close * adjusted_volume).tolist() == [500_000_000.0, 500_000_000.0]


def test_split_only_adjustment_does_not_treat_dividend_as_split() -> None:
    prices = pd.DataFrame(
        [
            {
                "ticker": "1111.T",
                "date": "2026-08-27",
                "open": 1_000.0,
                "high": 1_010.0,
                "low": 990.0,
                "close": 1_000.0,
                "adjusted_close": 990.0,
                "volume": 100_000.0,
                "stock_splits": 0.0,
            },
            {
                "ticker": "1111.T",
                "date": "2026-08-28",
                "open": 995.0,
                "high": 1_005.0,
                "low": 985.0,
                "close": 995.0,
                "adjusted_close": 995.0,
                "volume": 100_000.0,
                "stock_splits": 0.0,
            },
        ]
    )

    adjusted = normalize_split_adjusted_ohlcv(prices)

    assert adjusted["_split_share_factor"].tolist() == [1.0, 1.0]
    assert adjusted["_close"].tolist() == [1_000.0, 995.0]
    assert adjusted["_volume"].tolist() == [100_000.0, 100_000.0]


def test_explicit_split_adjusts_ohlcv_and_preserves_turnover() -> None:
    prices = pd.DataFrame(
        [
            {
                "ticker": "9279.T",
                "date": "2026-08-27",
                "open": 5_000.0,
                "high": 5_200.0,
                "low": 4_900.0,
                "close": 5_120.0,
                "volume": 100_000.0,
                "stock_splits": 0.0,
            },
            {
                "ticker": "9279.T",
                "date": "2026-08-28",
                "open": 2_500.0,
                "high": 2_600.0,
                "low": 2_450.0,
                "close": 2_560.0,
                "volume": 200_000.0,
                "stock_splits": 2.0,
            },
        ]
    )

    adjusted = normalize_split_adjusted_ohlcv(prices)

    assert adjusted["_split_share_factor"].tolist() == [2.0, 1.0]
    assert adjusted["_close"].tolist() == [2_560.0, 2_560.0]
    assert adjusted["_volume"].tolist() == [200_000.0, 200_000.0]
    assert adjusted["_turnover"].tolist() == [512_000_000.0, 512_000_000.0]
