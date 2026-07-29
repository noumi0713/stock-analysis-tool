from __future__ import annotations

import pandas as pd
import pytest

from app.modeling.features import FEATURE_COLUMNS, FeatureBuilder, LabelConfig


def make_calendar(days: int) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-06", periods=days)
    return pd.DataFrame(
        {
            "date": dates.date,
            "holiday_division": ["1"] * len(dates),
        }
    )


def make_equities(days: int, *, spike_offset: int | None = None) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-06", periods=days)
    records: list[dict[str, object]] = []
    for offset, current_date in enumerate(dates):
        high = 106.0 if offset == spike_offset else 101.0
        records.append(
            {
                "date": current_date.date(),
                "code": "13010",
                "adjusted_open": 100.0,
                "adjusted_high": high,
                "adjusted_low": 99.0,
                "adjusted_close": 100.0,
                "adjusted_volume": 1_000_000.0,
                "turnover_value": 100_000_000.0,
            }
        )
    return pd.DataFrame(records)


def test_features_use_only_information_available_by_signal_date() -> None:
    calendar = make_calendar(30)
    normal = FeatureBuilder().build(make_equities(30), calendar)
    spiked = FeatureBuilder().build(
        make_equities(30, spike_offset=23),
        calendar,
    )
    signal_date = calendar.iloc[20]["date"]
    normal_row = normal.loc[normal["date"] == signal_date].iloc[0]
    spiked_row = spiked.loc[spiked["date"] == signal_date].iloc[0]

    assert normal_row[FEATURE_COLUMNS].equals(spiked_row[FEATURE_COLUMNS])
    assert normal_row["target_5d"] == 0
    assert spiked_row["target_5d"] == 1
    assert spiked_row["target_hit_day"] == 3
    assert spiked_row["trade_gross_return"] == pytest.approx(0.05)


def test_last_horizon_days_have_no_label() -> None:
    horizon = 5
    built = FeatureBuilder(LabelConfig(horizon_days=horizon)).build(
        make_equities(30),
        make_calendar(30),
    )

    assert built.tail(horizon)["target_5d"].isna().all()
    assert built.iloc[-horizon - 1]["target_5d"] in (0, 1)


def test_future_lookup_uses_market_days_not_stock_row_shift() -> None:
    calendar = make_calendar(10)
    equities = make_equities(10, spike_offset=6)
    missing_date = calendar.iloc[3]["date"]
    equities = equities.loc[equities["date"] != missing_date]

    built = FeatureBuilder().build(equities, calendar)
    first = built.iloc[0]

    # カレンダー上6営業日先の上昇は、5営業日ラベルに混入しない。
    assert first["target_5d"] == 0
