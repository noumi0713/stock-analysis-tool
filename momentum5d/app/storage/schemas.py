from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetSchema:
    primary_key: tuple[str, ...]
    columns: tuple[tuple[str, str], ...]


DATASET_SCHEMAS: dict[str, DatasetSchema] = {
    "equities_daily": DatasetSchema(
        primary_key=("code", "date"),
        columns=(
            ("date", "DATE"),
            ("code", "VARCHAR"),
            ("open", "DOUBLE"),
            ("high", "DOUBLE"),
            ("low", "DOUBLE"),
            ("close", "DOUBLE"),
            ("upper_limit", "VARCHAR"),
            ("lower_limit", "VARCHAR"),
            ("volume", "DOUBLE"),
            ("turnover_value", "DOUBLE"),
            ("adjustment_factor", "DOUBLE"),
            ("adjusted_open", "DOUBLE"),
            ("adjusted_high", "DOUBLE"),
            ("adjusted_low", "DOUBLE"),
            ("adjusted_close", "DOUBLE"),
            ("adjusted_volume", "DOUBLE"),
        ),
    ),
    "listed_master": DatasetSchema(
        primary_key=("code", "date"),
        columns=(
            ("date", "DATE"),
            ("code", "VARCHAR"),
            ("company_name", "VARCHAR"),
            ("company_name_english", "VARCHAR"),
            ("sector_17_code", "VARCHAR"),
            ("sector_17_name", "VARCHAR"),
            ("sector_33_code", "VARCHAR"),
            ("sector_33_name", "VARCHAR"),
            ("scale_category", "VARCHAR"),
            ("market_code", "VARCHAR"),
            ("market_name", "VARCHAR"),
            ("margin_code", "VARCHAR"),
            ("margin_name", "VARCHAR"),
        ),
    ),
    "indices_daily": DatasetSchema(
        primary_key=("code", "date"),
        columns=(
            ("date", "DATE"),
            ("code", "VARCHAR"),
            ("open", "DOUBLE"),
            ("high", "DOUBLE"),
            ("low", "DOUBLE"),
            ("close", "DOUBLE"),
        ),
    ),
    "topix_daily": DatasetSchema(
        primary_key=("code", "date"),
        columns=(
            ("date", "DATE"),
            ("code", "VARCHAR"),
            ("open", "DOUBLE"),
            ("high", "DOUBLE"),
            ("low", "DOUBLE"),
            ("close", "DOUBLE"),
        ),
    ),
    "trading_calendar": DatasetSchema(
        primary_key=("date",),
        columns=(
            ("date", "DATE"),
            ("holiday_division", "VARCHAR"),
        ),
    ),
}
