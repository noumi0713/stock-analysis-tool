from __future__ import annotations

import pandas as pd
import pytest

from app.yahoo.candle_sequence import analyze_three_up_one_down


def _prices(closes: list[float], opens: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=len(closes), freq="B"),
            "ticker": ["1111.T"] * len(closes),
            "open": opens,
            "high": [
                max(open_price, close) * 1.01
                for open_price, close in zip(opens, closes, strict=True)
            ],
            "low": [
                min(open_price, close) * 0.99
                for open_price, close in zip(opens, closes, strict=True)
            ],
            "close": closes,
            "volume": [3_000_000] * len(closes),
        }
    )


def test_three_up_one_down_uses_fourth_close_as_reference() -> None:
    prices = _prices(
        closes=[101, 103, 105, 104, 106, 103, 110, 111, 112, 113, 114, 115, 116, 117],
        opens=[100, 102, 104, 106, 105, 104, 108, 110, 111, 112, 113, 114, 115, 116],
    )

    result = analyze_three_up_one_down(prices)

    assert result["sample_count"] == 1
    assert result["by_horizon"]["1d"]["up_rate"] == pytest.approx(1.0)
    assert result["by_horizon"]["3d"]["down_rate"] == pytest.approx(0.0)
    assert result["liquidity"]["turnover_200m_plus"]["5d"]["samples"] == 1


def test_three_up_one_down_requires_three_strict_bullish_bars() -> None:
    prices = _prices(
        closes=[100, 103, 105, 104, 105, 106, 107, 108, 109, 110, 111],
        opens=[100, 102, 104, 106, 104, 105, 106, 107, 108, 109, 110],
    )

    result = analyze_three_up_one_down(prices)

    assert result["sample_count"] == 0
