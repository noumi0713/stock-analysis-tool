from __future__ import annotations

import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from app.processing.normalize import normalize
from app.storage.schemas import DATASET_SCHEMAS

LOGGER = logging.getLogger(__name__)


class ParquetStore:
    def __init__(self, raw_dir: Path, processed_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def raw_unit_dir(self, dataset: str, unit_id: str) -> Path:
        safe_unit = re.sub(r"[^0-9A-Za-z_.-]", "_", unit_id)
        return self.raw_dir / dataset / f"unit={safe_unit}"

    def write_raw_page(
        self,
        dataset: str,
        unit_id: str,
        *,
        endpoint: str,
        page_number: int,
        records: list[dict[str, object]],
    ) -> Path:
        directory = self.raw_unit_dir(dataset, unit_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"page-{page_number:05d}.parquet"
        frame = pd.DataFrame.from_records(records)
        frame["__ingested_at"] = datetime.now(UTC).isoformat()
        frame["__endpoint"] = endpoint
        frame["__page_number"] = page_number
        self._atomic_parquet(frame, path)
        return path

    def read_raw_unit(self, dataset: str, unit_id: str) -> pd.DataFrame:
        files = sorted(self.raw_unit_dir(dataset, unit_id).glob("page-*.parquet"))
        if not files:
            return pd.DataFrame()
        frames = [pd.read_parquet(path) for path in files]
        return pd.concat(frames, ignore_index=True, sort=False)

    def filter_raw_unit(
        self,
        dataset: str,
        unit_id: str,
        *,
        column: str,
        allowed_values: set[str] | frozenset[str],
    ) -> int:
        """保存済みrawページを許可値だけに縮約し、残った行数を返す。"""
        total_rows = 0
        files = sorted(self.raw_unit_dir(dataset, unit_id).glob("page-*.parquet"))
        for path in files:
            frame = pd.read_parquet(path)
            if column in frame.columns:
                values = frame[column].astype("string")
                frame = frame.loc[values.isin(allowed_values)].reset_index(drop=True)
                self._atomic_parquet(frame, path)
            total_rows += len(frame)
        return total_rows

    def normalize_raw_unit(self, dataset: str, unit_id: str) -> pd.DataFrame:
        return normalize(dataset, self.read_raw_unit(dataset, unit_id))

    def process_raw_unit(self, dataset: str, unit_id: str) -> pd.DataFrame:
        frame = self.normalize_raw_unit(dataset, unit_id)
        if not frame.empty:
            self.upsert_processed(dataset, frame)
        return frame

    def upsert_processed(self, dataset: str, incoming: pd.DataFrame) -> None:
        schema = DATASET_SCHEMAS[dataset]
        if incoming.empty:
            return
        years = pd.to_datetime(incoming["date"]).dt.year
        for year in sorted(years.unique()):
            selected = incoming.loc[years == year].copy()
            path = self.processed_dir / dataset / f"year={year}" / "data.parquet"
            if path.exists():
                existing = pd.read_parquet(path)
                incoming_dates = set(pd.to_datetime(selected["date"]).dt.date)
                existing_dates = pd.to_datetime(existing["date"]).dt.date
                existing = existing.loc[~existing_dates.isin(incoming_dates)]
                selected = pd.concat([existing, selected], ignore_index=True, sort=False)
            selected = normalize(dataset, selected)
            duplicate_count = int(selected.duplicated(list(schema.primary_key), keep=False).sum())
            if duplicate_count:
                raise ValueError(
                    f"{dataset}/{year} の主キー重複を排除できません: {duplicate_count}件"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_parquet(selected, path)
            LOGGER.info(
                "Processed parquet updated dataset=%s year=%s rows=%s",
                dataset,
                year,
                len(selected),
            )

    def read_processed(
        self,
        dataset: str,
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        files = sorted((self.processed_dir / dataset).glob("year=*/data.parquet"))
        columns = [name for name, _ in DATASET_SCHEMAS[dataset].columns]
        if not files:
            return pd.DataFrame(columns=columns)
        frame = pd.concat(
            [pd.read_parquet(path) for path in files],
            ignore_index=True,
            sort=False,
        )
        dates = pd.to_datetime(frame["date"])
        if start:
            frame = frame.loc[dates >= pd.Timestamp(start)]
            dates = pd.to_datetime(frame["date"])
        if end:
            frame = frame.loc[dates <= pd.Timestamp(end)]
        return frame.reset_index(drop=True)

    def max_date(self, dataset: str) -> object | None:
        frame = self.read_processed(dataset)
        if frame.empty:
            return None
        return pd.to_datetime(frame["date"]).max().date()

    @staticmethod
    def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        os.replace(temporary, path)
