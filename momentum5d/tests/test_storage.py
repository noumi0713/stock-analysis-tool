from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from app.processing.normalize import normalize
from app.storage.catalog import DuckDBCatalog
from app.storage.checkpoints import CheckpointStore
from app.storage.parquet import ParquetStore


def equity_record(code: str, day: str, close: float) -> dict[str, object]:
    return {
        "Date": day,
        "Code": code,
        "O": close - 1,
        "H": close + 2,
        "L": close - 2,
        "C": close,
        "UL": "0",
        "LL": "0",
        "Vo": 1000,
        "Va": 100000,
        "AdjFactor": 1.0,
        "AdjO": close - 1,
        "AdjH": close + 2,
        "AdjL": close - 2,
        "AdjC": close,
        "AdjVo": 1000,
    }


def test_normalize_maps_v2_fields_and_deduplicates() -> None:
    raw = pd.DataFrame(
        [
            equity_record("013010", "2026-01-05", 100),
            equity_record("013010", "2026-01-05", 101),
        ]
    )

    result = normalize("equities_daily", raw)

    assert len(result) == 1
    assert result.iloc[0]["code"] == "013010"
    assert result.iloc[0]["close"] == 101
    assert result.iloc[0]["adjusted_close"] == 101


def test_checkpoint_records_page_cursor_and_completion(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "metadata" / "checkpoints.json")
    store.begin(
        "equities_daily",
        "2026-01-05",
        endpoint="/equities/bars/daily",
        params={"date": "2026-01-05"},
    )
    store.page_saved(
        "equities_daily",
        "2026-01-05",
        next_pagination_key="cursor-2",
        next_page=1,
        added_rows=100,
    )

    resumed = CheckpointStore(store.path).get("equities_daily", "2026-01-05")

    assert resumed is not None
    assert resumed["next_pagination_key"] == "cursor-2"
    assert resumed["next_page"] == 1
    assert resumed["row_count"] == 100

    store.page_saved(
        "equities_daily",
        "2026-01-05",
        next_pagination_key=None,
        next_page=2,
        added_rows=50,
    )
    assert store.get("equities_daily", "2026-01-05")["status"] == "downloaded"
    store.complete("equities_daily", "2026-01-05", processed_rows=150)
    assert store.get("equities_daily", "2026-01-05")["status"] == "complete"


def test_parquet_upsert_enforces_unique_primary_key_and_duckdb_view(
    tmp_path: Path,
) -> None:
    parquet = ParquetStore(tmp_path / "raw", tmp_path / "processed")
    parquet.write_raw_page(
        "equities_daily",
        "2026-01-05",
        endpoint="/equities/bars/daily",
        page_number=0,
        records=[equity_record("13010", "2026-01-05", 100)],
    )
    parquet.process_raw_unit("equities_daily", "2026-01-05")

    parquet.write_raw_page(
        "equities_daily",
        "2026-01-05-revision",
        endpoint="/equities/bars/daily",
        page_number=0,
        records=[equity_record("13010", "2026-01-05", 105)],
    )
    parquet.process_raw_unit("equities_daily", "2026-01-05-revision")

    saved = parquet.read_processed("equities_daily")
    assert len(saved) == 1
    assert saved.iloc[0]["close"] == 105

    catalog = DuckDBCatalog(tmp_path / "metadata" / "market.duckdb", tmp_path / "processed")
    catalog.refresh()
    with duckdb.connect(str(catalog.database_path), read_only=True) as connection:
        row = connection.execute("SELECT code, close FROM equities_daily").fetchone()
    assert row == ("13010", 105.0)
    assert catalog.status()[0]["duplicate_key_groups"] == 0


def test_parquet_upsert_replaces_all_rows_for_incoming_date(tmp_path: Path) -> None:
    parquet = ParquetStore(tmp_path / "raw", tmp_path / "processed")
    initial = normalize(
        "equities_daily",
        pd.DataFrame(
            [
                equity_record("13010", "2026-01-05", 100),
                equity_record("13020", "2026-01-05", 200),
            ]
        ),
    )
    parquet.upsert_processed("equities_daily", initial)

    replacement = normalize(
        "equities_daily",
        pd.DataFrame([equity_record("13010", "2026-01-05", 105)]),
    )
    parquet.upsert_processed("equities_daily", replacement)

    saved = parquet.read_processed("equities_daily")
    assert list(saved["code"]) == ["13010"]
    assert saved.iloc[0]["close"] == 105
