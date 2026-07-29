from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from app.config import Settings
from app.storage.parquet import ParquetStore

LOGGER = logging.getLogger(__name__)

YahooDownloader = Callable[..., pd.DataFrame]


@dataclass(frozen=True, slots=True)
class YahooConfig:
    retention_days: int = 365
    overlap_days: int = 10
    batch_size: int = 40
    pause_seconds: float = 2.0
    max_retries: int = 3
    timeout_seconds: float = 30.0

    @classmethod
    def load(cls) -> YahooConfig:
        return cls(
            retention_days=int(os.getenv("YAHOO_RETENTION_DAYS", "365")),
            overlap_days=int(os.getenv("YAHOO_OVERLAP_DAYS", "10")),
            batch_size=int(os.getenv("YAHOO_BATCH_SIZE", "40")),
            pause_seconds=float(os.getenv("YAHOO_PAUSE_SECONDS", "2")),
            max_retries=int(os.getenv("YAHOO_MAX_RETRIES", "3")),
            timeout_seconds=float(os.getenv("YAHOO_TIMEOUT_SECONDS", "30")),
        )


@dataclass(frozen=True, slots=True)
class YahooPaths:
    root: Path

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.root / "processed"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def prices_path(self) -> Path:
        return self.processed_dir / "equities_daily.parquet"

    @property
    def universe_path(self) -> Path:
        return self.metadata_dir / "prime_universe.parquet"

    @property
    def status_path(self) -> Path:
        return self.metadata_dir / "ingestion_status.json"

    def ensure(self) -> None:
        for path in (self.raw_dir, self.processed_dir, self.metadata_dir):
            path.mkdir(parents=True, exist_ok=True)


class YahooFinanceIngestion:
    """yfinanceの日足を分割取得し、直近1年のPrimeデータだけを保持する。"""

    def __init__(
        self,
        settings: Settings,
        config: YahooConfig | None = None,
        *,
        downloader: YahooDownloader | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.config = config or YahooConfig.load()
        self.paths = YahooPaths(settings.data_dir / "yahoo")
        self.downloader = downloader or yf.download
        self.sleeper = sleeper
        self.paths.ensure()
        self._validate_config()

    def ingest(
        self,
        *,
        as_of: date,
        tickers_file: Path | None = None,
        full_refresh: bool = False,
    ) -> dict[str, Any]:
        universe = self._load_universe(tickers_file)
        tickers = universe["ticker"].astype(str).tolist()
        existing = self._read_existing()
        existing_tickers = (
            set()
            if full_refresh
            else (set(existing["ticker"].astype(str).unique()) if not existing.empty else set())
        )
        retention_start = as_of - timedelta(days=self.config.retention_days)
        if existing.empty or full_refresh:
            request_start = retention_start
        else:
            latest = pd.to_datetime(existing["date"]).max().date()
            request_start = max(retention_start, latest - timedelta(days=self.config.overlap_days))
        request_end = as_of + timedelta(days=1)

        batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        frames: list[pd.DataFrame] = []
        failures: list[dict[str, str]] = []
        batches = [
            tickers[index : index + self.config.batch_size]
            for index in range(0, len(tickers), self.config.batch_size)
        ]
        for batch_number, batch in enumerate(batches):
            try:
                batch_start = (
                    retention_start
                    if any(ticker not in existing_tickers for ticker in batch)
                    else request_start
                )
                downloaded = self._download_batch(batch, batch_start, request_end)
                normalized = self._normalize_download(downloaded, batch)
                if not normalized.empty:
                    normalized["__batch_id"] = batch_id
                    normalized["__batch_number"] = batch_number
                    frames.append(normalized)
                    raw_path = (
                        self.paths.raw_dir
                        / f"batch={batch_id}"
                        / f"part-{batch_number:05d}.parquet"
                    )
                    raw_path.parent.mkdir(parents=True, exist_ok=True)
                    ParquetStore._atomic_parquet(normalized, raw_path)
            except Exception as exc:
                failures.append({"tickers": ",".join(batch), "error": str(exc)[:500]})
                LOGGER.exception("Yahoo batch failed batch=%s", batch_number)
            if batch_number + 1 < len(batches):
                self.sleeper(self.config.pause_seconds)

        incoming = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        combined = self._merge_and_prune(existing, incoming, retention_start, as_of)
        if combined.empty:
            raise RuntimeError("Yahoo Financeから有効な日足を取得できませんでした")
        ParquetStore._atomic_parquet(combined, self.paths.prices_path)
        ParquetStore._atomic_parquet(universe, self.paths.universe_path)
        from app.yahoo.catalog import YahooDuckDBCatalog

        YahooDuckDBCatalog(self.paths).refresh()

        saved_tickers = set(combined["ticker"].astype(str).unique())
        missing_tickers = sorted(set(tickers) - saved_tickers)
        status = {
            "source": "yfinance",
            "personal_research_only": True,
            "as_of": as_of.isoformat(),
            "request_start": request_start.isoformat(),
            "request_end_exclusive": request_end.isoformat(),
            "retention_start": retention_start.isoformat(),
            "full_refresh": full_refresh,
            "tickers": len(tickers),
            "successful_tickers": int(combined["ticker"].nunique()),
            "missing_tickers": missing_tickers,
            "rows": len(combined),
            "min_date": str(combined["date"].min()),
            "max_date": str(combined["date"].max()),
            "failed_batches": failures,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self._atomic_json(status, self.paths.status_path)
        return status

    def _load_universe(self, tickers_file: Path | None) -> pd.DataFrame:
        if tickers_file is not None:
            values = [
                line.strip()
                for line in tickers_file.read_text(encoding="utf-8-sig").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            if values and values[0].lower() in {"ticker", "symbol"}:
                values = values[1:]
            tickers = sorted({_normalize_ticker(value.split(",")[0]) for value in values})
            return pd.DataFrame({"ticker": tickers, "code": [_ticker_to_code(x) for x in tickers]})

        master_store = ParquetStore(self.settings.raw_dir, self.settings.processed_dir)
        master = master_store.read_processed("listed_master")
        if master.empty:
            raise RuntimeError(
                "Prime銘柄一覧がありません。--tickers-file を指定するか、"
                "保存済みlisted_masterを用意してください"
            )
        latest_date = pd.to_datetime(master["date"]).max().date()
        latest = master.loc[
            (pd.to_datetime(master["date"]).dt.date == latest_date)
            & (master["market_code"].astype("string") == "0111")
        ].copy()
        latest["ticker"] = latest["code"].astype(str).map(_code_to_ticker)
        return (
            latest[["ticker", "code", "company_name"]]
            .drop_duplicates("ticker")
            .sort_values("ticker")
            .reset_index(drop=True)
        )

    def _download_batch(
        self,
        tickers: Sequence[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                result = self.downloader(
                    list(tickers),
                    start=start.isoformat(),
                    end=end.isoformat(),
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=False,
                    actions=True,
                    repair=True,
                    keepna=True,
                    progress=False,
                    threads=min(4, len(tickers)),
                    timeout=self.config.timeout_seconds,
                    multi_level_index=True,
                )
                if result is None or result.empty:
                    raise RuntimeError("空のレスポンス")
                return result
            except Exception as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                self.sleeper(max(5.0, self.config.pause_seconds * (2**attempt)))
        raise RuntimeError(f"Yahoo Finance取得に{self.config.max_retries + 1}回失敗: {last_error}")

    @staticmethod
    def _normalize_download(downloaded: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            selected: pd.DataFrame | None = None
            if isinstance(downloaded.columns, pd.MultiIndex):
                level_zero = downloaded.columns.get_level_values(0)
                level_one = downloaded.columns.get_level_values(1)
                if ticker in level_zero:
                    selected = downloaded[ticker].copy()
                elif ticker in level_one:
                    selected = downloaded.xs(ticker, axis=1, level=1).copy()
            elif len(tickers) == 1:
                selected = downloaded.copy()
            if selected is None or selected.empty:
                continue

            selected = selected.reset_index()
            selected.columns = [
                str(column).strip().lower().replace(" ", "_") for column in selected.columns
            ]
            date_column = "date" if "date" in selected.columns else selected.columns[0]
            selected = selected.rename(columns={date_column: "date", "adj_close": "adjusted_close"})
            for column in ("open", "high", "low", "close", "adjusted_close", "volume"):
                if column not in selected.columns:
                    selected[column] = pd.NA
                selected[column] = pd.to_numeric(selected[column], errors="coerce")
            for column in ("dividends", "stock_splits"):
                if column not in selected.columns:
                    selected[column] = 0.0
                selected[column] = pd.to_numeric(selected[column], errors="coerce").fillna(0.0)
            selected["date"] = pd.to_datetime(selected["date"], errors="coerce").dt.date
            selected["ticker"] = ticker
            selected["code"] = _ticker_to_code(ticker)
            selected["turnover_value"] = selected["close"] * selected["volume"]
            selected["source"] = "yfinance"
            selected = selected.loc[
                selected["date"].notna()
                & selected[["open", "high", "low", "close"]].notna().any(axis=1)
            ]
            frames.append(
                selected[
                    [
                        "date",
                        "ticker",
                        "code",
                        "open",
                        "high",
                        "low",
                        "close",
                        "adjusted_close",
                        "volume",
                        "turnover_value",
                        "dividends",
                        "stock_splits",
                        "source",
                    ]
                ]
            )
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)

    def _read_existing(self) -> pd.DataFrame:
        if not self.paths.prices_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.paths.prices_path)

    @staticmethod
    def _merge_and_prune(
        existing: pd.DataFrame,
        incoming: pd.DataFrame,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        frames = [frame for frame in (existing, incoming) if not frame.empty]
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.date
        combined = combined.loc[(combined["date"] >= start) & (combined["date"] <= end)]
        combined = combined.drop_duplicates(["ticker", "date"], keep="last")
        metadata_columns = [column for column in combined.columns if not column.startswith("__")]
        return combined[metadata_columns].sort_values(["ticker", "date"]).reset_index(drop=True)

    @staticmethod
    def _atomic_json(value: dict[str, Any], path: Path) -> None:
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _validate_config(self) -> None:
        if self.config.retention_days < 30:
            raise ValueError("retention_daysは30以上である必要があります")
        if self.config.batch_size < 1:
            raise ValueError("batch_sizeは1以上である必要があります")
        if self.config.max_retries < 0:
            raise ValueError("max_retriesは0以上である必要があります")


def _normalize_ticker(value: str) -> str:
    ticker = value.strip().upper()
    return ticker if ticker.endswith(".T") else f"{ticker}.T"


def _code_to_ticker(code: str) -> str:
    normalized = str(code).strip().upper()
    base = normalized[:-1] if len(normalized) == 5 else normalized
    return f"{base}.T"


def _ticker_to_code(ticker: str) -> str:
    base = ticker.upper().removesuffix(".T")
    return f"{base}0" if len(base) == 4 else base
