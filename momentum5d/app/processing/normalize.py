from __future__ import annotations

import pandas as pd

from app.storage.schemas import DATASET_SCHEMAS

FIELD_MAPS: dict[str, dict[str, str]] = {
    "equities_daily": {
        "Date": "date",
        "Code": "code",
        "O": "open",
        "H": "high",
        "L": "low",
        "C": "close",
        "UL": "upper_limit",
        "LL": "lower_limit",
        "Vo": "volume",
        "Va": "turnover_value",
        "AdjFactor": "adjustment_factor",
        "AdjO": "adjusted_open",
        "AdjH": "adjusted_high",
        "AdjL": "adjusted_low",
        "AdjC": "adjusted_close",
        "AdjVo": "adjusted_volume",
    },
    "listed_master": {
        "Date": "date",
        "Code": "code",
        "CoName": "company_name",
        "CoNameEn": "company_name_english",
        "S17": "sector_17_code",
        "S17Nm": "sector_17_name",
        "S33": "sector_33_code",
        "S33Nm": "sector_33_name",
        "ScaleCat": "scale_category",
        "Mkt": "market_code",
        "MktNm": "market_name",
        "Mrgn": "margin_code",
        "MrgnNm": "margin_name",
    },
    "indices_daily": {
        "Date": "date",
        "Code": "code",
        "O": "open",
        "H": "high",
        "L": "low",
        "C": "close",
    },
    "topix_daily": {
        "Date": "date",
        "O": "open",
        "H": "high",
        "L": "low",
        "C": "close",
    },
    "trading_calendar": {
        "Date": "date",
        "HolDiv": "holiday_division",
    },
}

STRING_COLUMNS = {
    "code",
    "upper_limit",
    "lower_limit",
    "company_name",
    "company_name_english",
    "sector_17_code",
    "sector_17_name",
    "sector_33_code",
    "sector_33_name",
    "scale_category",
    "market_code",
    "market_name",
    "margin_code",
    "margin_name",
    "holiday_division",
}


def normalize(dataset: str, raw: pd.DataFrame) -> pd.DataFrame:
    if dataset not in DATASET_SCHEMAS:
        raise KeyError(f"未定義のデータセットです: {dataset}")
    schema = DATASET_SCHEMAS[dataset]
    expected_columns = [name for name, _ in schema.columns]
    if raw.empty:
        return pd.DataFrame(columns=expected_columns)

    renamed = raw.rename(columns=FIELD_MAPS[dataset]).copy()
    if dataset == "topix_daily":
        renamed["code"] = "TOPIX"

    result = pd.DataFrame(index=renamed.index)
    for column in expected_columns:
        if column in renamed.columns:
            result[column] = renamed[column]
        else:
            result[column] = pd.NA

    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.date
    for column in expected_columns:
        if column == "date":
            continue
        if column in STRING_COLUMNS:
            result[column] = result[column].astype("string")
        else:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    missing_keys = result[list(schema.primary_key)].isna().any(axis=1)
    if missing_keys.any():
        count = int(missing_keys.sum())
        raise ValueError(f"{dataset} に主キー欠損が {count} 件あります")

    result = result.drop_duplicates(list(schema.primary_key), keep="last")
    return result.sort_values(list(schema.primary_key)).reset_index(drop=True)
