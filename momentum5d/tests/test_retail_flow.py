from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.yahoo.retail_flow import RETAIL_DETAIL_COLUMNS, add_retail_flow_features


def _retail_features(days: int = 30) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = date(2026, 1, 1)
    for ticker, strong in (("STRONG.T", True), ("WEAK.T", False)):
        for index in range(days):
            recent = index >= days - 5
            rows.append(
                {
                    "date": start + timedelta(days=index),
                    "ticker": ticker,
                    "adjusted_close": 100 + index,
                    "turnover_value": (
                        80_000_000 if strong and recent else 20_000_000
                    ),
                    "return_1d": 0.02 if strong else -0.01,
                    "return_5d": 0.03 if strong else -0.05,
                    "return_20d": 0.08 if strong else -0.10,
                    "volume_ratio_5_20": 1.6 if strong else 0.7,
                    "intraday_return": 0.015 if strong else -0.02,
                    "breakout_20d": -0.01 if strong else -0.18,
                    "close_to_ma20": 0.02 if strong else -0.10,
                    "volatility_10d": 0.018 if strong else 0.05,
                    "up_volume_share_10d": 0.62 if strong else 0.35,
                    "rsi_14": 60.0 if strong else 35.0,
                    "relative_return_20d": 0.06 if strong else -0.06,
                    "sector_17_median_return_5d": 0.03 if strong else -0.03,
                    "sector_17_breadth_5d": 0.70 if strong else 0.30,
                    "sector_17_trend_score": 0.82 if strong else 0.20,
                    "legacy_setup_score": 0.72 if strong else 0.30,
                }
            )
    return pd.DataFrame(rows)


def test_retail_flow_rewards_attention_expectation_and_action_alignment() -> None:
    scored = add_retail_flow_features(_retail_features())
    latest = scored.loc[scored["date"] == scored["date"].max()].set_index("ticker")

    assert latest.loc["STRONG.T", "retail_flow_score"] > 0.65
    assert (
        latest.loc["STRONG.T", "retail_flow_score"]
        > latest.loc["WEAK.T", "retail_flow_score"]
    )
    assert (
        latest.loc["STRONG.T", "retail_action_score"]
        > latest.loc["WEAK.T", "retail_action_score"]
    )
    assert (
        latest.loc["STRONG.T", "retail_attention_hybrid_score"]
        > latest.loc["WEAK.T", "retail_attention_hybrid_score"]
    )
    expected_hybrid = (
        0.60 * (0.75 * 0.72 + 0.25 * 0.82)
        + 0.40 * latest.loc["STRONG.T", "retail_flow_score"]
    )
    assert latest.loc["STRONG.T", "retail_attention_hybrid_score"] == expected_hybrid


def test_retail_flow_has_no_future_data_dependency() -> None:
    base_input = _retail_features()
    base = add_retail_flow_features(base_input)
    future = base_input.copy()
    future["date"] = future["date"] + timedelta(days=40)
    future["return_1d"] = -0.20
    extended = add_retail_flow_features(pd.concat([base_input, future], ignore_index=True))

    columns = list(RETAIL_DETAIL_COLUMNS)
    earlier = extended.loc[extended["date"] <= base_input["date"].max()]
    pd.testing.assert_frame_equal(
        base[columns].reset_index(drop=True),
        earlier[columns].reset_index(drop=True),
    )
