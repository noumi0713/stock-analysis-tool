from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_live_strategy_backtest import audit_input_data


def _prices(*, adjusted: bool = True) -> pd.DataFrame:
    row = {
        "ticker": "1111.T",
        "date": "2026-01-05",
        "open": 100,
        "high": 105,
        "low": 99,
        "close": 102,
        "volume": 1_000,
    }
    if adjusted:
        row["adjusted_close"] = 51
    return pd.DataFrame([row])


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "1111.T",
                "valid_from": "2025-01-01",
                "valid_to": None,
                "ticker_reused": False,
            }
        ]
    )


def test_raw_adjusted_input_is_certified() -> None:
    audit = audit_input_data(
        _prices(), _history(), input_mode="raw_ohlcv_with_adjusted_close"
    )

    assert audit["status"] == "certified"
    assert audit["split_adjusted_volume"] is True


def test_legacy_shard_input_is_only_provisional() -> None:
    audit = audit_input_data(
        _prices(adjusted=False),
        _history(),
        input_mode="legacy_preadjusted_prices_raw_volume",
    )

    assert audit["status"] == "provisional"
    assert audit["split_adjusted_volume"] is False
    assert audit["warnings"]


def test_certified_input_requires_adjusted_close() -> None:
    with pytest.raises(ValueError, match="adjusted_close"):
        audit_input_data(
            _prices(adjusted=False),
            _history(),
            input_mode="raw_ohlcv_with_adjusted_close",
        )
