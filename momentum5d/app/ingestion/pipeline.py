from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from app.api.client import APIError, JQuantsClient
from app.config import Settings
from app.storage.catalog import DuckDBCatalog
from app.storage.checkpoints import CheckpointStore
from app.storage.parquet import ParquetStore

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    endpoint: str
    required: bool


CALENDAR = DatasetSpec("trading_calendar", "/markets/calendar", True)
DAILY_DATASETS = (
    DatasetSpec("listed_master", "/equities/master", True),
    DatasetSpec("equities_daily", "/equities/bars/daily", True),
    DatasetSpec("topix_daily", "/indices/bars/daily/topix", False),
    DatasetSpec("indices_daily", "/indices/bars/daily", False),
)


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        client: JQuantsClient,
        *,
        parquet: ParquetStore | None = None,
        checkpoints: CheckpointStore | None = None,
        catalog: DuckDBCatalog | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.parquet = parquet or ParquetStore(settings.raw_dir, settings.processed_dir)
        self.checkpoints = checkpoints or CheckpointStore(settings.checkpoint_path)
        self.catalog = catalog or DuckDBCatalog(settings.duckdb_path, settings.processed_dir)
        self._universe_codes_by_date: dict[str, frozenset[str]] = {}

    def ingest(
        self,
        start: date,
        end: date,
        *,
        datasets: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        if start > end:
            raise ValueError("start は end 以下である必要があります")
        today_jst = datetime.now(ZoneInfo("Asia/Tokyo")).date()
        if end > today_jst:
            raise ValueError(
                f"未来日は取得できません: end={end.isoformat()}, today={today_jst.isoformat()}"
            )

        self.settings.ensure_directories()
        LOGGER.info("Ingestion started start=%s end=%s", start, end)
        calendar_unit = f"{start.isoformat()}_{end.isoformat()}"
        self._ingest_unit(
            CALENDAR,
            calendar_unit,
            {"from": start.isoformat(), "to": end.isoformat()},
            allow_empty=False,
        )

        calendar = self.parquet.read_processed(
            "trading_calendar", start=start.isoformat(), end=end.isoformat()
        )
        # 公式仕様: HolDiv=1 が東証営業日。0は休業日、2はOSE祝日取引日。
        trading_dates = sorted(
            pd.to_datetime(
                calendar.loc[calendar["holiday_division"].astype("string") == "1", "date"]
            )
            .dt.date.drop_duplicates()
            .tolist()
        )
        if not trading_dates:
            raise RuntimeError("指定期間の東証営業日を公式取引カレンダーから取得できません")

        unknown = set(datasets or ()).difference(spec.name for spec in DAILY_DATASETS)
        if unknown:
            raise ValueError(f"未対応のデータセットです: {sorted(unknown)}")
        requested = (
            set(datasets) if datasets is not None else {spec.name for spec in DAILY_DATASETS}
        )
        if "equities_daily" in requested:
            # 日次株価のPrime判定には同日銘柄マスタが必須。
            requested.add("listed_master")
        selected_specs = tuple(spec for spec in DAILY_DATASETS if spec.name in requested)
        if not selected_specs:
            raise ValueError("取得対象データセットがありません")

        unavailable: set[str] = set()
        for position, trading_date in enumerate(trading_dates, start=1):
            LOGGER.info(
                "Processing trading date=%s progress=%s/%s",
                trading_date,
                position,
                len(trading_dates),
            )
            for spec in selected_specs:
                if spec.name in unavailable:
                    continue
                params = self._daily_params(spec, trading_date)
                try:
                    self._ingest_unit(
                        spec,
                        trading_date.isoformat(),
                        params,
                        allow_empty=False,
                    )
                    self.checkpoints.mark_dataset_available(spec.name)
                except APIError as exc:
                    if exc.status_code == 403 and not spec.required:
                        unavailable.add(spec.name)
                        self.checkpoints.skip_unavailable(
                            spec.name, trading_date.isoformat(), str(exc)
                        )
                        self.checkpoints.mark_dataset_unavailable(spec.name, str(exc))
                        LOGGER.warning(
                            "Dataset unavailable for current plan dataset=%s error=%s",
                            spec.name,
                            exc,
                        )
                        continue
                    raise

        self.catalog.refresh()
        snapshot = self.checkpoints.snapshot()
        summary = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "trading_days": len(trading_dates),
            "datasets": [spec.name for spec in selected_specs],
            "unavailable_datasets": sorted(unavailable),
            "checkpoint_units": len(snapshot["units"]),
        }
        LOGGER.info("Ingestion completed summary=%s", summary)
        return summary

    def update(self, end: date | None = None) -> dict[str, Any]:
        last_saved = self.parquet.max_date("equities_daily")
        if last_saved is None:
            raise RuntimeError(
                "保存済み株価がありません。先に ingest --start YYYY-MM-DD を実行してください"
            )
        update_end = end or datetime.now(ZoneInfo("Asia/Tokyo")).date()
        update_start = last_saved + timedelta(days=1)
        pending_start = self._earliest_pending_date()
        if pending_start is not None:
            update_start = min(update_start, pending_start)
        if update_start > update_end:
            self.catalog.refresh()
            return {
                "start": update_start.isoformat(),
                "end": update_end.isoformat(),
                "trading_days": 0,
                "unavailable_datasets": [],
                "message": "更新対象日はありません",
            }
        return self.ingest(update_start, update_end)

    def _earliest_pending_date(self) -> date | None:
        state = self.checkpoints.snapshot()
        pending: list[date] = []
        for unit in state["units"].values():
            if unit.get("status") in {"complete", "skipped_unavailable"}:
                continue
            unit_id = str(unit.get("unit_id", ""))
            try:
                pending.append(date.fromisoformat(unit_id))
            except ValueError:
                continue
        return min(pending) if pending else None

    def _ingest_unit(
        self,
        spec: DatasetSpec,
        unit_id: str,
        params: dict[str, Any],
        *,
        allow_empty: bool,
    ) -> pd.DataFrame:
        checkpoint = self.checkpoints.begin(
            spec.name,
            unit_id,
            endpoint=spec.endpoint,
            params=params,
        )
        if checkpoint["status"] == "complete":
            return self._process_raw_unit(spec, unit_id)

        try:
            if checkpoint["status"] != "downloaded":
                cursor = checkpoint.get("next_pagination_key")
                start_page = int(checkpoint.get("next_page", 0))
                for page in self.client.iter_pages(
                    spec.endpoint,
                    params,
                    pagination_key=cursor,
                    start_page=start_page,
                ):
                    records = self._filter_page_records(spec, unit_id, page.records)
                    self.parquet.write_raw_page(
                        spec.name,
                        unit_id,
                        endpoint=spec.endpoint,
                        page_number=page.page_number,
                        records=records,
                    )
                    self.checkpoints.page_saved(
                        spec.name,
                        unit_id,
                        next_pagination_key=page.next_key,
                        next_page=page.page_number + 1,
                        added_rows=len(records),
                    )

            processed = self._process_raw_unit(spec, unit_id)
            if processed.empty and not allow_empty:
                message = (
                    f"{spec.name}/{unit_id} が空です。公開前または契約期間外の可能性があります"
                )
                self.checkpoints.fail(spec.name, unit_id, message, restart_download=True)
                raise RuntimeError(message)
            self.checkpoints.complete(spec.name, unit_id, processed_rows=len(processed))
            return processed
        except Exception as exc:
            current = self.checkpoints.get(spec.name, unit_id)
            if current and current["status"] != "failed":
                self.checkpoints.fail(spec.name, unit_id, str(exc))
            raise

    def _filter_page_records(
        self,
        spec: DatasetSpec,
        unit_id: str,
        records: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if spec.name == "listed_master":
            allowed = set(self.settings.universe_market_codes)
            return [record for record in records if str(record.get("Mkt", "")) in allowed]
        if spec.name == "equities_daily":
            allowed = self._allowed_equity_codes(unit_id)
            return [record for record in records if str(record.get("Code", "")) in allowed]
        return records

    def _process_raw_unit(self, spec: DatasetSpec, unit_id: str) -> pd.DataFrame:
        if spec.name == "listed_master":
            self.parquet.filter_raw_unit(
                spec.name,
                unit_id,
                column="Mkt",
                allowed_values=frozenset(self.settings.universe_market_codes),
            )
        elif spec.name == "equities_daily":
            self.parquet.filter_raw_unit(
                spec.name,
                unit_id,
                column="Code",
                allowed_values=self._allowed_equity_codes(unit_id),
            )

        processed = self.parquet.normalize_raw_unit(spec.name, unit_id)
        if spec.name == "listed_master":
            processed = processed.loc[
                processed["market_code"].isin(self.settings.universe_market_codes)
            ].reset_index(drop=True)
            self._universe_codes_by_date[unit_id] = frozenset(
                processed["code"].dropna().astype(str)
            )
        elif spec.name == "equities_daily":
            processed = processed.loc[
                processed["code"].isin(self._allowed_equity_codes(unit_id))
            ].reset_index(drop=True)

        if not processed.empty:
            self.parquet.upsert_processed(spec.name, processed)
        return processed

    def _allowed_equity_codes(self, unit_id: str) -> frozenset[str]:
        cached = self._universe_codes_by_date.get(unit_id)
        if cached is not None:
            return cached
        master = self.parquet.read_processed(
            "listed_master",
            start=unit_id,
            end=unit_id,
        )
        allowed = frozenset(
            master.loc[
                master["market_code"].isin(self.settings.universe_market_codes),
                "code",
            ]
            .dropna()
            .astype(str)
        )
        if not allowed:
            raise RuntimeError(
                f"{unit_id} の対象市場銘柄マスタがありません。listed_masterを先に取得してください"
            )
        self._universe_codes_by_date[unit_id] = allowed
        return allowed

    @staticmethod
    def _daily_params(spec: DatasetSpec, trading_date: date) -> dict[str, str]:
        value = trading_date.isoformat()
        if spec.name == "topix_daily":
            return {"from": value, "to": value}
        return {"date": value}
