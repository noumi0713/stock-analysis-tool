from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import Settings
from app.storage.parquet import ParquetStore
from app.yahoo.ingestion import YahooPaths

ISSUE_COLUMNS = [
    "checked_at",
    "severity",
    "check_name",
    "dataset",
    "date",
    "ticker",
    "code",
    "message",
    "observed_value",
]


@dataclass(frozen=True, slots=True)
class YahooQualityReport:
    issues: pd.DataFrame
    summary: dict[str, Any]
    issues_path: Path
    summary_path: Path

    @property
    def has_errors(self) -> bool:
        return bool(not self.issues.empty and (self.issues["severity"] == "error").any())


class YahooQualityValidator:
    """Yahoo日足に対する、保存後に再実行できる品質検査。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.paths = YahooPaths(settings.data_dir / "yahoo")
        self._checked_at = ""
        self._issues: list[dict[str, Any]] = []

    def run(self) -> YahooQualityReport:
        if not self.paths.prices_path.exists():
            raise RuntimeError("品質検査対象のYahoo日足がありません")
        self._checked_at = datetime.now(UTC).isoformat()
        self._issues = []
        prices = pd.read_parquet(self.paths.prices_path)
        universe = (
            pd.read_parquet(self.paths.universe_path)
            if self.paths.universe_path.exists()
            else pd.DataFrame()
        )

        self._check_keys(prices)
        self._check_ohlc(prices)
        self._check_volume(prices)
        self._check_returns_and_splits(prices)
        self._check_universe(prices, universe)

        issues = pd.DataFrame(self._issues, columns=ISSUE_COLUMNS)
        if not issues.empty:
            issues["date"] = pd.to_datetime(issues["date"], errors="coerce").dt.date
            issues = issues.sort_values(
                ["severity", "check_name", "date", "ticker"],
                na_position="last",
            ).reset_index(drop=True)
        severity_counts = (
            issues["severity"].value_counts().sort_index().to_dict() if not issues.empty else {}
        )
        check_counts = (
            issues["check_name"].value_counts().sort_index().to_dict() if not issues.empty else {}
        )
        summary = {
            "source": "yfinance",
            "checked_at": self._checked_at,
            "rows": len(prices),
            "tickers": int(prices["ticker"].nunique()),
            "issue_count": len(issues),
            "severity_counts": severity_counts,
            "check_counts": check_counts,
            "thresholds": {
                "abnormal_return": self.settings.abnormal_return_threshold,
                "split_ratio_tolerance": self.settings.split_ratio_tolerance,
            },
        }
        quality_dir = self.paths.processed_dir / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        issues_path = quality_dir / "latest.parquet"
        summary_path = self.paths.metadata_dir / "quality_latest.json"
        ParquetStore._atomic_parquet(issues, issues_path)
        temporary = summary_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, summary_path)
        from app.yahoo.catalog import YahooDuckDBCatalog

        YahooDuckDBCatalog(self.paths).refresh()
        return YahooQualityReport(issues, summary, issues_path, summary_path)

    def _check_keys(self, frame: pd.DataFrame) -> None:
        missing = frame[["ticker", "date"]].isna().any(axis=1)
        self._add_rows(
            frame.loc[missing],
            "error",
            "missing_primary_key",
            "主キー ticker, date に欠損があります",
        )
        duplicate = frame.duplicated(["ticker", "date"], keep=False)
        self._add_rows(
            frame.loc[duplicate],
            "error",
            "duplicate_primary_key",
            "主キー ticker, date が重複しています",
        )

    def _check_ohlc(self, frame: pd.DataFrame) -> None:
        columns = ["open", "high", "low", "close"]
        values = frame[columns].apply(pd.to_numeric, errors="coerce")
        partial = values.isna().any(axis=1) & ~values.isna().all(axis=1)
        self._add_rows(
            frame.loc[partial],
            "error",
            "partial_ohlc_missing",
            "OHLCの一部が欠損しています",
        )
        complete = ~values.isna().any(axis=1)
        invalid = complete & (
            (values["high"] < values[["open", "low", "close"]].max(axis=1))
            | (values["low"] > values[["open", "high", "close"]].min(axis=1))
        )
        observed = values.astype(str).agg(",".join, axis=1)
        self._add_rows(
            frame.loc[invalid],
            "error",
            "ohlc_inconsistent",
            "high/lowがopen/closeを包含していません",
            observed.loc[invalid],
        )

    def _check_volume(self, frame: pd.DataFrame) -> None:
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        self._add_rows(
            frame.loc[volume == 0],
            "warning",
            "zero_volume",
            "出来高が0です",
            volume.loc[volume == 0],
        )
        self._add_rows(
            frame.loc[volume < 0],
            "error",
            "negative_volume",
            "出来高が負です",
            volume.loc[volume < 0],
        )
        missing = volume.isna() & frame[["open", "high", "low", "close"]].notna().any(axis=1)
        self._add_rows(
            frame.loc[missing],
            "error",
            "volume_missing_with_price",
            "価格が存在するのに出来高が欠損しています",
        )

    def _check_returns_and_splits(self, frame: pd.DataFrame) -> None:
        ordered = frame.sort_values(["ticker", "date"]).copy()
        adjusted_close = pd.to_numeric(ordered["adjusted_close"], errors="coerce")
        previous_adjusted = adjusted_close.groupby(ordered["ticker"]).shift()
        adjusted_return = adjusted_close / previous_adjusted - 1.0
        abnormal = adjusted_return.abs() > self.settings.abnormal_return_threshold
        self._add_rows(
            ordered.loc[abnormal],
            "warning",
            "abnormal_adjusted_return",
            f"調整済み日次リターンが{self.settings.abnormal_return_threshold:.0%}を超えています",
            adjusted_return.loc[abnormal],
        )

        if "stock_splits" in ordered:
            stock_splits = pd.to_numeric(ordered["stock_splits"], errors="coerce").fillna(0.0)
            split_event = (stock_splits > 0) & ~np.isclose(stock_splits, 1.0)
            inferred_factor = 1.0 / stock_splits.where(stock_splits > 0)
        else:
            close = pd.to_numeric(ordered["close"], errors="coerce")
            adjustment_ratio = adjusted_close / close
            previous_ratio = adjustment_ratio.groupby(ordered["ticker"]).shift()
            ratio_change = adjustment_ratio / previous_ratio
            split_event = (
                ratio_change.notna()
                & np.isfinite(ratio_change)
                & ((ratio_change < 0.80) | (ratio_change > 1.25))
            )
            inferred_factor = previous_ratio / adjustment_ratio
        self._add_rows(
            ordered.loc[split_event],
            "info",
            "stock_split_event",
            "Yahooの企業アクションから株式分割・併合を検出しました",
            inferred_factor.loc[split_event],
        )

        self._add_rows(
            ordered.loc[split_event & abnormal],
            "warning",
            "stock_split_adjusted_price_jump",
            "分割・併合日に調整済み終値が大きく変動しています",
            adjusted_return.loc[split_event & abnormal],
        )

    def _check_universe(self, prices: pd.DataFrame, universe: pd.DataFrame) -> None:
        if universe.empty:
            self._add_dataset_issue(
                "error",
                "prime_universe_missing",
                "Prime銘柄一覧が保存されていません",
            )
            return
        saved = set(prices["ticker"].astype(str).unique())
        for ticker in sorted(set(universe["ticker"].astype(str)) - saved):
            row = universe.loc[universe["ticker"].astype(str) == ticker].iloc[0]
            self._issues.append(
                self._issue(
                    severity="warning",
                    check_name="universe_ticker_without_price",
                    message="Prime銘柄一覧にありますがYahoo日足を取得できません",
                    ticker=ticker,
                    code=row.get("code"),
                )
            )

    def _add_rows(
        self,
        frame: pd.DataFrame,
        severity: str,
        check_name: str,
        message: str,
        observed: pd.Series | None = None,
    ) -> None:
        for index, row in frame.iterrows():
            self._issues.append(
                self._issue(
                    severity=severity,
                    check_name=check_name,
                    message=message,
                    date=row.get("date"),
                    ticker=row.get("ticker"),
                    code=row.get("code"),
                    observed_value=observed.loc[index] if observed is not None else None,
                )
            )

    def _add_dataset_issue(self, severity: str, check_name: str, message: str) -> None:
        self._issues.append(self._issue(severity=severity, check_name=check_name, message=message))

    def _issue(
        self,
        *,
        severity: str,
        check_name: str,
        message: str,
        date: object = None,
        ticker: object = None,
        code: object = None,
        observed_value: object = None,
    ) -> dict[str, Any]:
        return {
            "checked_at": self._checked_at,
            "severity": severity,
            "check_name": check_name,
            "dataset": "yahoo_equities_daily",
            "date": date,
            "ticker": ticker,
            "code": code,
            "message": message,
            "observed_value": (
                None if observed_value is None or pd.isna(observed_value) else str(observed_value)
            ),
        }
