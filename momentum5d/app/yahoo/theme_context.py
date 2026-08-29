from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ThemeContextConfig:
    return_weight: float = 0.40
    breadth_weight: float = 0.30
    turnover_weight: float = 0.30
    confirmed_score: float = 0.60
    confirmed_breadth: float = 0.50
    confirmed_turnover_ratio: float = 0.90
    turnover_lookback: int = 20
    turnover_min_periods: int = 10


def calculate_theme_context(
    indicators: pd.DataFrame,
    theme_memberships: dict[str, list[dict[str, str]]],
    *,
    config: ThemeContextConfig | None = None,
) -> pd.DataFrame:
    """Calculate leakage-safe price/volume context for each theme and date.

    The price features use only information available at each close. Membership
    history is not available, so callers must treat results based on the current
    catalog as research metadata rather than a deployment-quality hard filter.
    """

    settings = config or ThemeContextConfig()
    columns = [
        "date",
        "theme",
        "theme_return_5d",
        "theme_breadth_5d",
        "theme_turnover_ratio_1_20",
        "theme_member_count",
        "theme_score",
        "theme_flow_confirmed",
    ]
    if indicators.empty or not theme_memberships:
        return pd.DataFrame(columns=columns)

    membership_rows = [
        {"ticker": f"{code}.T", "theme": item.get("theme", "")}
        for code, items in theme_memberships.items()
        for item in items
        if item.get("theme")
    ]
    membership_frame = pd.DataFrame(membership_rows).drop_duplicates()
    if membership_frame.empty:
        return pd.DataFrame(columns=columns)

    required = {"ticker", "date", "return_5d", "trading_value"}
    missing = required.difference(indicators.columns)
    if missing:
        raise ValueError(f"Theme context is missing indicator columns: {sorted(missing)}")

    members = indicators[list(required)].merge(
        membership_frame,
        on="ticker",
        how="inner",
        validate="many_to_many",
    )
    if members.empty:
        return pd.DataFrame(columns=columns)

    themes = (
        members.groupby(["date", "theme"], sort=False)
        .agg(
            theme_return_5d=("return_5d", "median"),
            theme_breadth_5d=(
                "return_5d",
                lambda values: float((values.dropna() > 0).mean())
                if values.notna().any()
                else float("nan"),
            ),
            theme_turnover=("trading_value", "sum"),
            theme_member_count=("ticker", "nunique"),
        )
        .reset_index()
        .sort_values(["theme", "date"])
    )
    rolling_turnover = themes.groupby("theme", sort=False)["theme_turnover"].transform(
        lambda values: values.rolling(
            settings.turnover_lookback,
            min_periods=settings.turnover_min_periods,
        ).mean()
    )
    themes["theme_turnover_ratio_1_20"] = themes["theme_turnover"].div(
        rolling_turnover.where(rolling_turnover.gt(0))
    )

    by_date = themes.groupby("date", sort=False)
    return_rank = by_date["theme_return_5d"].rank(pct=True)
    breadth_rank = by_date["theme_breadth_5d"].rank(pct=True)
    turnover_rank = by_date["theme_turnover_ratio_1_20"].rank(pct=True)
    themes["theme_score"] = (
        settings.return_weight * return_rank
        + settings.breadth_weight * breadth_rank
        + settings.turnover_weight * turnover_rank
    )
    themes["theme_flow_confirmed"] = (
        themes["theme_score"].ge(settings.confirmed_score)
        & themes["theme_return_5d"].gt(0)
        & themes["theme_breadth_5d"].ge(settings.confirmed_breadth)
        & themes["theme_turnover_ratio_1_20"].ge(
            settings.confirmed_turnover_ratio
        )
    )
    return themes[columns].reset_index(drop=True)


def latest_context_by_code(
    context: pd.DataFrame,
    theme_memberships: dict[str, list[dict[str, str]]],
    *,
    latest_date: Any,
) -> dict[str, dict[str, Any]]:
    if context.empty:
        return {}

    memberships = pd.DataFrame(
        [
            {"code": code, "theme": item.get("theme", "")}
            for code, items in theme_memberships.items()
            for item in items
            if item.get("theme")
        ]
    ).drop_duplicates()
    if memberships.empty:
        return {}

    latest = context.loc[context["date"].eq(latest_date)].merge(
        memberships,
        on="theme",
        how="inner",
        validate="one_to_many",
    )
    if latest.empty:
        return {}
    best = (
        latest.sort_values(
            ["code", "theme_score", "theme"],
            ascending=[True, False, True],
        )
        .groupby("code", sort=False, as_index=False)
        .first()
    )

    result: dict[str, dict[str, Any]] = {}
    for row in best.itertuples(index=False):
        score = float(row.theme_score) if pd.notna(row.theme_score) else None
        confirmed = bool(row.theme_flow_confirmed)
        if confirmed:
            status = "confirmed"
            label = "資金流入確認"
        elif score is not None and score < 0.40:
            status = "weak"
            label = "テーマ弱め"
        else:
            status = "neutral"
            label = "テーマ中立"
        result[str(row.code)] = {
            "status": status,
            "label": label,
            "primary_theme": str(row.theme),
            "score": None if score is None else round(score, 4),
            "return_5d": (
                None
                if pd.isna(row.theme_return_5d)
                else round(float(row.theme_return_5d), 6)
            ),
            "breadth_5d": (
                None
                if pd.isna(row.theme_breadth_5d)
                else round(float(row.theme_breadth_5d), 6)
            ),
            "turnover_ratio_1_20": (
                None
                if pd.isna(row.theme_turnover_ratio_1_20)
                else round(float(row.theme_turnover_ratio_1_20), 6)
            ),
            "member_count": int(row.theme_member_count),
            "use": "pullback_ranking_aid_only",
        }
    return result
