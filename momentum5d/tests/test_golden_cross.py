from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from app.yahoo.golden_cross import (
    GoldenCrossBacktestConfig,
    backtest_golden_cross_volume,
)


def _cross_features(*, ambiguous_stop: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = date(2026, 1, 1)
    for index in range(100):
        close = 100.0 if index < 20 else 95.0 if index < 25 else 125.0
        open_ = close
        high = close + 1.0
        low = close - 1.0
        if index == 26:
            open_ = 125.0
            close = 128.0
            high = 132.0
            low = 120.0 if ambiguous_stop else 124.0
        volume = 2_000_000.0 if index == 25 else 1_000_000.0
        rows.append(
            {
                "date": start + timedelta(days=index),
                "ticker": "1111.T",
                "code": "1111",
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "adjusted_close": close,
                "volume": volume,
                "turnover_value": close * volume,
            }
        )
    return pd.DataFrame(rows)


def _config() -> GoldenCrossBacktestConfig:
    return GoldenCrossBacktestConfig(
        moving_average_pairs=((5, 25),),
        volume_ratio_thresholds=(1.0, 1.2, 1.5, 2.0),
        minimum_calibration_trades=1,
        calibration_fraction=0.70,
    )


def test_golden_cross_with_double_volume_reaches_five_percent() -> None:
    report = backtest_golden_cross_volume(_cross_features(), _config())

    result = report["strategies"]["gc_5_25_prior20_2x"]["all"]
    assert result["trades"] == 1
    assert result["target_hit_rate"] == pytest.approx(1.0)
    assert result["win_rate"] == pytest.approx(1.0)
    assert result["mean_net_return"] == pytest.approx(0.048)
    assert report["selected_strategy"] is not None


def test_same_day_target_and_stop_is_counted_as_stop_first() -> None:
    report = backtest_golden_cross_volume(
        _cross_features(ambiguous_stop=True),
        _config(),
    )

    result = report["strategies"]["gc_5_25_prior20_2x"]["all"]
    assert result["trades"] == 1
    assert result["target_hit_rate"] == pytest.approx(0.0)
    assert result["stop_hit_rate"] == pytest.approx(1.0)
    assert result["mean_net_return"] == pytest.approx(-0.032)


def test_volume_threshold_uses_prior_twenty_days_without_current_day() -> None:
    report = backtest_golden_cross_volume(_cross_features(), _config())

    assert report["strategies"]["gc_5_25_prior20_2x"]["all"]["trades"] == 1
    assert report["strategies"]["gc_5_25_previous_day"]["all"]["trades"] == 1
    assert report["leakage_control"].startswith("移動平均と出来高平均はtまで")
