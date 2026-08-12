from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SECTOR_17_NAMES = {
    "1": "食品",
    "2": "エネルギー資源",
    "3": "建設・資材",
    "4": "素材・化学",
    "5": "医薬品",
    "6": "自動車・輸送機",
    "7": "鉄鋼・非鉄",
    "8": "機械",
    "9": "電機・精密",
    "10": "情報通信・サービスその他",
    "11": "電力・ガス",
    "12": "運輸・物流",
    "13": "商社・卸売",
    "14": "小売",
    "15": "銀行",
    "16": "金融（除く銀行）",
    "17": "不動産",
}

SECTOR_33_NAMES = {
    "0050": "水産・農林業",
    "1050": "鉱業",
    "2050": "建設業",
    "3050": "食料品",
    "3100": "繊維製品",
    "3150": "パルプ・紙",
    "3200": "化学",
    "3250": "医薬品",
    "3300": "石油・石炭製品",
    "3350": "ゴム製品",
    "3400": "ガラス・土石製品",
    "3450": "鉄鋼",
    "3500": "非鉄金属",
    "3550": "金属製品",
    "3600": "機械",
    "3650": "電気機器",
    "3700": "輸送用機器",
    "3750": "精密機器",
    "3800": "その他製品",
    "4050": "電気・ガス業",
    "5050": "陸運業",
    "5100": "海運業",
    "5150": "空運業",
    "5200": "倉庫・運輸関連業",
    "5250": "情報・通信業",
    "6050": "卸売業",
    "6100": "小売業",
    "7050": "銀行業",
    "7100": "証券・商品先物取引業",
    "7150": "保険業",
    "7200": "その他金融業",
    "8050": "不動産業",
    "9050": "サービス業",
}

SECTOR_WEEKLY_HISTORY_START = "2025-08-12"


def load_sector_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "code",
                "sector_17_code",
                "sector_17_name",
                "sector_33_code",
                "sector_33_name",
            ]
        )
    sectors = pd.read_csv(
        path,
        dtype={
            "code": "string",
            "sector_17_code": "string",
            "sector_33_code": "string",
        },
    )
    required = {"code", "sector_17_code", "sector_33_code"}
    missing = required.difference(sectors.columns)
    if missing:
        raise ValueError(f"業種マスターの必須列がありません: {sorted(missing)}")
    sectors = sectors[["code", "sector_17_code", "sector_33_code"]].copy()
    sectors["code"] = sectors["code"].str.strip()
    sectors["sector_17_code"] = sectors["sector_17_code"].str.strip()
    sectors["sector_33_code"] = sectors["sector_33_code"].str.strip().str.zfill(4)
    sectors["sector_17_name"] = sectors["sector_17_code"].map(SECTOR_17_NAMES)
    sectors["sector_33_name"] = sectors["sector_33_code"].map(SECTOR_33_NAMES)
    if sectors[["sector_17_name", "sector_33_name"]].isna().any().any():
        raise ValueError("業種マスターに未定義の業種コードがあります")
    if sectors["code"].duplicated().any():
        raise ValueError("業種マスターの銘柄コードが重複しています")
    return sectors[
        [
            "code",
            "sector_17_code",
            "sector_17_name",
            "sector_33_code",
            "sector_33_name",
        ]
    ].sort_values("code")


def add_trend_features(
    features: pd.DataFrame,
    sectors: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "date",
        "ticker",
        "code",
        "adjusted_close",
        "return_5d",
        "return_20d",
        "setup_score",
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"トレンド計算の必須列がありません: {sorted(missing)}")
    frame = features.copy()
    frame["code"] = frame["code"].astype("string")
    if not sectors.empty:
        frame = frame.merge(
            sectors,
            on="code",
            how="left",
            validate="many_to_one",
        )
    else:
        for column in (
            "sector_17_code",
            "sector_17_name",
            "sector_33_code",
            "sector_33_name",
        ):
            frame[column] = pd.NA

    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    group = frame.groupby("ticker", sort=False)
    frame["return_60d"] = frame["adjusted_close"] / group["adjusted_close"].shift(60) - 1
    frame["ma_20d"] = group["adjusted_close"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    frame["ma_60d"] = group["adjusted_close"].transform(
        lambda values: values.rolling(60, min_periods=60).mean()
    )
    frame["ma_20d_slope_5d"] = (
        frame["ma_20d"] / frame.groupby("ticker", sort=False)["ma_20d"].shift(5) - 1
    )
    frame["above_ma_20d"] = frame["adjusted_close"] > frame["ma_20d"]
    frame["ma_20d_above_60d"] = frame["ma_20d"] > frame["ma_60d"]

    date_group = frame.groupby("date", sort=False)
    frame["market_median_return_20d"] = date_group["return_20d"].transform("median")
    frame["market_median_return_60d"] = date_group["return_60d"].transform("median")
    frame["relative_return_20d"] = (
        frame["return_20d"] - frame["market_median_return_20d"]
    )
    frame["individual_trend_score"] = (
        0.30 * date_group["return_20d"].rank(pct=True)
        + 0.25 * date_group["return_60d"].rank(pct=True)
        + 0.20 * frame["above_ma_20d"].astype(float)
        + 0.15 * frame["ma_20d_above_60d"].astype(float)
        + 0.10 * date_group["ma_20d_slope_5d"].rank(pct=True)
    )

    for sector_level in ("17", "33"):
        frame = _add_sector_trend(frame, sector_level)

    frame["trend_score"] = (
        0.45 * frame["individual_trend_score"]
        + 0.55 * frame["sector_33_trend_score"]
    )
    frame["trend_combined_score"] = (
        0.65 * frame["setup_score"]
        + 0.15 * frame["individual_trend_score"]
        + 0.20 * frame["sector_33_trend_score"]
    )
    sector_17_score = frame["sector_17_trend_score"].fillna(frame["setup_score"])
    frame["trend_ranking_score"] = (
        0.75 * frame["setup_score"] + 0.25 * sector_17_score
    )
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame


def latest_sector_trends(
    features: pd.DataFrame,
    *,
    level: str = "33",
    top_n: int = 10,
) -> list[dict[str, Any]]:
    if level not in {"17", "33"}:
        raise ValueError("levelは17または33で指定してください")
    latest = features.loc[features["date"] == features["date"].max()].copy()
    code_column = f"sector_{level}_code"
    name_column = f"sector_{level}_name"
    columns = [
        code_column,
        name_column,
        f"sector_{level}_median_return_5d",
        f"sector_{level}_median_return_20d",
        f"sector_{level}_median_return_60d",
        f"sector_{level}_breadth_5d",
        f"sector_{level}_trend_score",
        f"sector_{level}_size",
    ]
    ranked = (
        latest.dropna(subset=[code_column])
        .sort_values(f"sector_{level}_trend_score", ascending=False)
        .drop_duplicates(code_column)
        .head(top_n)[columns]
    )
    return [
        {key: _json_value(value) for key, value in row.items()}
        for row in ranked.to_dict("records")
    ]


def weekly_sector_33_returns(
    features: pd.DataFrame,
    *,
    start_date: str = SECTOR_WEEKLY_HISTORY_START,
) -> dict[str, Any]:
    """Return one 33-sector cross-section for each completed market week.

    Each weekly value is the median constituent 5-trading-day return observed on
    the final trading day in that calendar week. The requested start date limits
    the published observations; the 5-day return itself remains the feature
    computed from the available price history.
    """
    required = {
        "date",
        "ticker",
        "return_5d",
        "sector_33_code",
        "sector_33_name",
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"週次業種騰落率の必須列がありません: {sorted(missing)}")

    requested_start = pd.Timestamp(start_date).normalize()
    frame = features[list(required)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["return_5d"] = pd.to_numeric(frame["return_5d"], errors="coerce")
    frame = frame.loc[frame["date"].notna() & frame["date"].ge(requested_start)]
    if frame.empty:
        return _empty_weekly_sector_history(requested_start)

    frame["week"] = frame["date"].dt.to_period("W-FRI")
    weekly_dates = frame.groupby("week", observed=True)["date"].max()
    weeks: list[dict[str, Any]] = []
    for week, as_of in weekly_dates.items():
        snapshot = frame.loc[frame["date"].eq(as_of)].copy()
        valid = snapshot.dropna(subset=["sector_33_code", "return_5d"])
        if valid.empty:
            continue

        grouped = (
            valid.groupby("sector_33_code", observed=True)
            .agg(
                median_return_5d=("return_5d", "median"),
                breadth_5d=("return_5d", lambda values: float((values > 0).mean())),
                stock_count=("ticker", "nunique"),
            )
            .reindex(SECTOR_33_NAMES)
        )
        available = grouped["median_return_5d"].dropna().sort_values(ascending=False)
        ranks = pd.Series(
            range(1, len(available) + 1),
            index=available.index,
            dtype="Int64",
        )
        sectors = []
        for code, name in SECTOR_33_NAMES.items():
            values = grouped.loc[code]
            sectors.append(
                {
                    "rank": _json_value(ranks.get(code, pd.NA)),
                    "sector_33_code": code,
                    "sector_33_name": name,
                    "median_return_5d": _json_value(values["median_return_5d"]),
                    "breadth_5d": _json_value(values["breadth_5d"]),
                    "stock_count": int(values["stock_count"])
                    if pd.notna(values["stock_count"])
                    else 0,
                }
            )
        sectors.sort(
            key=lambda row: (
                row["rank"] is None,
                row["rank"] if row["rank"] is not None else 10_000,
            )
        )
        weeks.append(
            {
                "week_start": str(max(week.start_time.normalize(), requested_start).date()),
                "week_end": str(week.end_time.normalize().date()),
                "as_of": str(as_of.date()),
                "sector_count": int(len(available)),
                "sectors": sectors,
            }
        )

    result = _empty_weekly_sector_history(requested_start)
    result["weeks"] = weeks
    result["week_count"] = len(weeks)
    result["first_as_of"] = weeks[0]["as_of"] if weeks else None
    result["last_as_of"] = weeks[-1]["as_of"] if weeks else None
    return result


def analyze_sector_volume_next_week_returns(
    features: pd.DataFrame,
    *,
    start_date: str = SECTOR_WEEKLY_HISTORY_START,
) -> dict[str, Any]:
    """Measure whether sector volume changes lead the following week's return."""
    required = {
        "date",
        "ticker",
        "adjusted_close",
        "volume",
        "sector_33_code",
        "sector_33_name",
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"業種出来高先行分析の必須列がありません: {sorted(missing)}")

    requested_start = pd.Timestamp(start_date).normalize()
    frame = features[list(required)].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["adjusted_close"] = pd.to_numeric(frame["adjusted_close"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame = frame.dropna(
        subset=[
            "date",
            "ticker",
            "adjusted_close",
            "volume",
            "sector_33_code",
        ]
    )
    frame = frame.loc[frame["adjusted_close"].gt(0) & frame["volume"].gt(0)]
    if frame.empty:
        return _empty_sector_volume_lead_study(requested_start)

    frame = frame.sort_values(["ticker", "date"])
    frame["week"] = frame["date"].dt.to_period("W-FRI")
    weekly_stock = (
        frame.groupby(["ticker", "week"], as_index=False, observed=True)
        .agg(
            as_of=("date", "max"),
            adjusted_close=("adjusted_close", "last"),
            average_daily_volume=("volume", "mean"),
            trading_days=("date", "nunique"),
            sector_33_code=("sector_33_code", "last"),
            sector_33_name=("sector_33_name", "last"),
        )
        .sort_values(["ticker", "week"])
    )
    ticker_group = weekly_stock.groupby("ticker", sort=False)
    weekly_stock["previous_week"] = ticker_group["week"].shift(1)
    weekly_stock["previous_average_daily_volume"] = ticker_group[
        "average_daily_volume"
    ].shift(1)
    weekly_stock["next_week"] = ticker_group["week"].shift(-1)
    weekly_stock["next_as_of"] = ticker_group["as_of"].shift(-1)
    weekly_stock["next_adjusted_close"] = ticker_group["adjusted_close"].shift(-1)
    weekly_stock["weekly_volume_change"] = (
        weekly_stock["average_daily_volume"]
        / weekly_stock["previous_average_daily_volume"]
        - 1
    )
    weekly_stock["next_week_return"] = (
        weekly_stock["next_adjusted_close"] / weekly_stock["adjusted_close"] - 1
    )
    consecutive = (
        weekly_stock["previous_week"].eq(weekly_stock["week"] - 1)
        & weekly_stock["next_week"].eq(weekly_stock["week"] + 1)
    )
    eligible = weekly_stock.loc[
        consecutive
        & weekly_stock["as_of"].ge(requested_start)
        & weekly_stock["weekly_volume_change"].replace([np.inf, -np.inf], np.nan).notna()
        & weekly_stock["next_week_return"].replace([np.inf, -np.inf], np.nan).notna()
    ].copy()
    if eligible.empty:
        return _empty_sector_volume_lead_study(requested_start)

    sector_week = (
        eligible.groupby(["week", "sector_33_code"], as_index=False, observed=True)
        .agg(
            current_as_of=("as_of", "max"),
            next_as_of=("next_as_of", "max"),
            sector_33_name=("sector_33_name", "last"),
            median_weekly_volume_change=("weekly_volume_change", "median"),
            median_next_week_return=("next_week_return", "median"),
            next_week_positive_rate=(
                "next_week_return",
                lambda values: float((values > 0).mean()),
            ),
            stock_count=("ticker", "nunique"),
        )
        .sort_values(["current_as_of", "sector_33_code"])
    )
    sector_week.replace([np.inf, -np.inf], np.nan, inplace=True)
    sector_week.dropna(
        subset=["median_weekly_volume_change", "median_next_week_return"],
        inplace=True,
    )
    if sector_week.empty:
        return _empty_sector_volume_lead_study(requested_start)

    overall = _volume_lead_metrics(sector_week)
    overall["pearson_correlation"] = _safe_correlation(
        sector_week["median_weekly_volume_change"],
        sector_week["median_next_week_return"],
    )
    overall["spearman_correlation"] = _safe_correlation(
        sector_week["median_weekly_volume_change"],
        sector_week["median_next_week_return"],
        rank=True,
    )

    weekly_rank_correlations = []
    for _, values in sector_week.groupby("current_as_of", observed=True):
        correlation = _safe_correlation(
            values["median_weekly_volume_change"],
            values["median_next_week_return"],
            rank=True,
        )
        if correlation is not None:
            weekly_rank_correlations.append(correlation)
    overall["average_weekly_cross_section_spearman"] = (
        float(np.mean(weekly_rank_correlations)) if weekly_rank_correlations else None
    )
    overall["median_weekly_cross_section_spearman"] = (
        float(np.median(weekly_rank_correlations)) if weekly_rank_correlations else None
    )
    overall["positive_weekly_correlation_rate"] = (
        float(np.mean(np.asarray(weekly_rank_correlations) > 0))
        if weekly_rank_correlations
        else None
    )

    direction_groups = []
    for key, label, mask in (
        (
            "volume_down_or_flat",
            "出来高減少・横ばい",
            sector_week["median_weekly_volume_change"].le(0),
        ),
        ("volume_up", "出来高増加", sector_week["median_weekly_volume_change"].gt(0)),
    ):
        values = sector_week.loc[mask]
        metrics = _volume_lead_metrics(values)
        metrics.update({"key": key, "label": label})
        direction_groups.append(metrics)

    ranked_volume = sector_week["median_weekly_volume_change"].rank(method="first")
    sector_week["volume_quintile"] = pd.qcut(
        ranked_volume,
        5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    )
    quintiles = []
    quintile_labels = {
        "Q1": "出来高変化率 下位20%",
        "Q2": "下位20〜40%",
        "Q3": "中位20%",
        "Q4": "上位20〜40%",
        "Q5": "出来高変化率 上位20%",
    }
    for key, values in sector_week.groupby("volume_quintile", observed=True):
        metrics = _volume_lead_metrics(values)
        metrics.update({"key": str(key), "label": quintile_labels[str(key)]})
        quintiles.append(metrics)

    by_sector = []
    for code, values in sector_week.groupby("sector_33_code", observed=True):
        volume_up = values.loc[values["median_weekly_volume_change"].gt(0)]
        volume_down = values.loc[values["median_weekly_volume_change"].le(0)]
        up_return = _series_mean(volume_up["median_next_week_return"])
        down_return = _series_mean(volume_down["median_next_week_return"])
        by_sector.append(
            {
                "sector_33_code": str(code),
                "sector_33_name": str(values["sector_33_name"].iloc[-1]),
                "sample_count": int(len(values)),
                "pearson_correlation": _safe_correlation(
                    values["median_weekly_volume_change"],
                    values["median_next_week_return"],
                ),
                "spearman_correlation": _safe_correlation(
                    values["median_weekly_volume_change"],
                    values["median_next_week_return"],
                    rank=True,
                ),
                "average_next_return_when_volume_up": up_return,
                "average_next_return_when_volume_down_or_flat": down_return,
                "volume_up_return_difference": (
                    up_return - down_return
                    if up_return is not None and down_return is not None
                    else None
                ),
            }
        )
    by_sector.sort(
        key=lambda row: (
            row["spearman_correlation"] is None,
            -(row["spearman_correlation"] or 0),
        )
    )

    return {
        "method": "sector_median_stock_weekly_volume_change_vs_next_week_return",
        "method_label": "業種内銘柄の週間平均出来高前週比中央値と翌週騰落率中央値",
        "requested_start_date": str(requested_start.date()),
        "first_current_as_of": str(sector_week["current_as_of"].min().date()),
        "last_current_as_of": str(sector_week["current_as_of"].max().date()),
        "last_next_as_of": str(sector_week["next_as_of"].max().date()),
        "period_count": int(sector_week["current_as_of"].nunique()),
        "sector_week_observation_count": int(len(sector_week)),
        "overall": overall,
        "direction_groups": direction_groups,
        "volume_change_quintiles": quintiles,
        "by_sector": by_sector,
    }


def _volume_lead_metrics(values: pd.DataFrame) -> dict[str, Any]:
    returns = values["median_next_week_return"].dropna()
    volumes = values["median_weekly_volume_change"].dropna()
    return {
        "sample_count": int(len(values)),
        "median_weekly_volume_change": _series_median(volumes),
        "average_next_week_return": _series_mean(returns),
        "median_next_week_return": _series_median(returns),
        "positive_next_week_rate": float((returns > 0).mean()) if len(returns) else None,
        "plus_5pct_next_week_rate": float((returns >= 0.05).mean()) if len(returns) else None,
    }


def _safe_correlation(
    left: pd.Series,
    right: pd.Series,
    *,
    rank: bool = False,
) -> float | None:
    values = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(values) < 3 or values["left"].nunique() < 2 or values["right"].nunique() < 2:
        return None
    if rank:
        values = values.rank(method="average")
    return float(np.corrcoef(values["left"], values["right"])[0, 1])


def _series_mean(values: pd.Series) -> float | None:
    return float(values.mean()) if len(values) else None


def _series_median(values: pd.Series) -> float | None:
    return float(values.median()) if len(values) else None


def _empty_sector_volume_lead_study(start_date: pd.Timestamp) -> dict[str, Any]:
    return {
        "method": "sector_median_stock_weekly_volume_change_vs_next_week_return",
        "method_label": "業種内銘柄の週間平均出来高前週比中央値と翌週騰落率中央値",
        "requested_start_date": str(start_date.date()),
        "period_count": 0,
        "sector_week_observation_count": 0,
        "overall": {},
        "direction_groups": [],
        "volume_change_quintiles": [],
        "by_sector": [],
    }


def _empty_weekly_sector_history(start_date: pd.Timestamp) -> dict[str, Any]:
    return {
        "method": "constituent_return_5d_median",
        "method_label": "構成銘柄の5営業日騰落率中央値",
        "requested_start_date": str(start_date.date()),
        "return_unit": "decimal",
        "expected_sector_count": len(SECTOR_33_NAMES),
        "week_count": 0,
        "first_as_of": None,
        "last_as_of": None,
        "weeks": [],
    }


def _add_sector_trend(frame: pd.DataFrame, level: str) -> pd.DataFrame:
    code_column = f"sector_{level}_code"
    prefix = f"sector_{level}"
    grouped = (
        frame.dropna(subset=[code_column])
        .groupby(["date", code_column], as_index=False)
        .agg(
            **{
                f"{prefix}_median_return_5d": ("return_5d", "median"),
                f"{prefix}_median_return_20d": ("return_20d", "median"),
                f"{prefix}_median_return_60d": ("return_60d", "median"),
                f"{prefix}_breadth_5d": (
                    "return_5d",
                    lambda values: float((values > 0).mean()),
                ),
                f"{prefix}_size": ("ticker", "nunique"),
            }
        )
    )
    sector_date_group = grouped.groupby("date", sort=False)
    grouped[f"{prefix}_trend_score"] = (
        0.20 * sector_date_group[f"{prefix}_median_return_5d"].rank(pct=True)
        + 0.35 * sector_date_group[f"{prefix}_median_return_20d"].rank(pct=True)
        + 0.25 * sector_date_group[f"{prefix}_median_return_60d"].rank(pct=True)
        + 0.20 * grouped[f"{prefix}_breadth_5d"]
    )
    return frame.merge(
        grouped,
        on=["date", code_column],
        how="left",
        validate="many_to_one",
    )


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
