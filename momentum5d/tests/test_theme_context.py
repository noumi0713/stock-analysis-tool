from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.yahoo.theme_context import calculate_theme_context, latest_context_by_code


def _indicators() -> pd.DataFrame:
    rows: list[dict] = []
    start = date(2026, 1, 1)
    for offset in range(20):
        day = start + timedelta(days=offset)
        rows.extend(
            [
                {
                    "ticker": "1001.T",
                    "date": day,
                    "return_5d": 0.04,
                    "trading_value": 100.0 + offset * 20,
                },
                {
                    "ticker": "1002.T",
                    "date": day,
                    "return_5d": 0.02,
                    "trading_value": 80.0 + offset * 15,
                },
                {
                    "ticker": "2001.T",
                    "date": day,
                    "return_5d": -0.03,
                    "trading_value": 200.0 - offset,
                },
                {
                    "ticker": "2002.T",
                    "date": day,
                    "return_5d": -0.01,
                    "trading_value": 150.0 - offset,
                },
            ]
        )
    return pd.DataFrame(rows)


def _memberships() -> dict[str, list[dict[str, str]]]:
    return {
        "1001": [{"theme": "強いテーマ"}],
        "1002": [{"theme": "強いテーマ"}],
        "2001": [{"theme": "弱いテーマ"}],
        "2002": [{"theme": "弱いテーマ"}],
    }


def test_theme_context_ranks_breadth_return_and_turnover_at_each_close() -> None:
    context = calculate_theme_context(_indicators(), _memberships())
    latest = context.loc[context["date"].eq(context["date"].max())].set_index("theme")

    assert latest.at["強いテーマ", "theme_score"] > latest.at[
        "弱いテーマ", "theme_score"
    ]
    assert bool(latest.at["強いテーマ", "theme_flow_confirmed"])
    assert not bool(latest.at["弱いテーマ", "theme_flow_confirmed"])


def test_latest_context_uses_the_strongest_membership_for_each_stock() -> None:
    memberships = _memberships()
    memberships["1001"].append({"theme": "弱いテーマ"})
    context = calculate_theme_context(_indicators(), memberships)
    result = latest_context_by_code(
        context,
        memberships,
        latest_date=context["date"].max(),
    )

    assert result["1001"]["primary_theme"] == "強いテーマ"
    assert result["1001"]["status"] == "confirmed"
    assert result["2001"]["status"] == "neutral"
