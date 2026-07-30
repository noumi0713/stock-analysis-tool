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
