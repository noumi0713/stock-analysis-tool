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
from app.storage.catalog import DuckDBCatalog
from app.storage.parquet import ParquetStore
from app.storage.schemas import DATASET_SCHEMAS

ISSUE_COLUMNS = [
    "checked_at",
    "severity",
    "check_name",
    "dataset",
    "date",
    "code",
    "message",
    "observed_value",
]


@dataclass(frozen=True, slots=True)
class QualityReport:
    issues: pd.DataFrame
    summary: dict[str, Any]
    issues_path: Path
    summary_path: Path

    @property
    def has_errors(self) -> bool:
        return bool((self.issues["severity"] == "error").any())


class QualityValidator:
    def __init__(
        self,
        settings: Settings,
        *,
        parquet: ParquetStore | None = None,
        catalog: DuckDBCatalog | None = None,
    ) -> None:
        self.settings = settings
        self.parquet = parquet or ParquetStore(settings.raw_dir, settings.processed_dir)
        self.catalog = catalog or DuckDBCatalog(settings.duckdb_path, settings.processed_dir)
        self._checked_at = datetime.now(UTC).isoformat()
        self._issues: list[dict[str, Any]] = []

    def run(self) -> QualityReport:
        self._checked_at = datetime.now(UTC).isoformat()
        self._issues = []
        equities = self.parquet.read_processed("equities_daily")
        master = self.parquet.read_processed("listed_master")
        calendar = self.parquet.read_processed("trading_calendar")
        if equities.empty:
            raise RuntimeError("品質検査対象の equities_daily がありません")

        self._check_primary_keys(equities, "equities_daily")
        self._check_primary_keys(master, "listed_master")
        self._check_ohlc(equities, adjusted=False)
        self._check_ohlc(equities, adjusted=True)
        self._check_volume(equities)
        self._check_returns_and_splits(equities)
        self._check_historical_membership(equities, master)
        self._check_trading_days(equities, calendar)

        issues = pd.DataFrame(self._issues, columns=ISSUE_COLUMNS)
        if not issues.empty:
            issues["date"] = pd.to_datetime(issues["date"], errors="coerce").dt.date
            issues = issues.sort_values(
                ["severity", "check_name", "date", "code"],
                na_position="last",
            ).reset_index(drop=True)

        severity_counts = (
            issues["severity"].value_counts().sort_index().to_dict() if not issues.empty else {}
        )
        check_counts = (
            issues["check_name"].value_counts().sort_index().to_dict() if not issues.empty else {}
        )
        summary = {
            "checked_at": self._checked_at,
            "equities_rows": len(equities),
            "master_snapshot_rows": len(master),
            "issue_count": len(issues),
            "severity_counts": severity_counts,
            "check_counts": check_counts,
            "thresholds": {
                "abnormal_return": self.settings.abnormal_return_threshold,
                "split_ratio_tolerance": self.settings.split_ratio_tolerance,
            },
        }
        issues_path, summary_path = self._save(issues, summary)
        self.catalog.refresh()
        return QualityReport(issues, summary, issues_path, summary_path)

    def _check_primary_keys(self, frame: pd.DataFrame, dataset: str) -> None:
        if frame.empty:
            return
        keys = list(DATASET_SCHEMAS[dataset].primary_key)
        missing = frame[keys].isna().any(axis=1)
        self._add_frame(
            frame.loc[missing],
            severity="error",
            check_name="missing_primary_key",
            dataset=dataset,
            message=f"主キー {keys} に欠損があります",
        )
        duplicate = frame.duplicated(keys, keep=False)
        self._add_frame(
            frame.loc[duplicate],
            severity="error",
            check_name="duplicate_primary_key",
            dataset=dataset,
            message=f"主キー {keys} が重複しています",
        )

    def _check_ohlc(self, frame: pd.DataFrame, *, adjusted: bool) -> None:
        prefix = "adjusted_" if adjusted else ""
        columns = [f"{prefix}{name}" for name in ("open", "high", "low", "close")]
        label = "adjusted" if adjusted else "unadjusted"
        values = frame[columns]
        all_missing = values.isna().all(axis=1)
        partial_missing = values.isna().any(axis=1) & ~all_missing
        known_outage = pd.to_datetime(frame["date"]).dt.date == pd.Timestamp("2020-10-01").date()

        self._add_frame(
            frame.loc[partial_missing],
            severity="error",
            check_name=f"{label}_partial_ohlc_missing",
            dataset="equities_daily",
            message=f"{label} OHLCの一部だけが欠損しています",
        )
        self._add_frame(
            frame.loc[all_missing & known_outage],
            severity="info",
            check_name=f"{label}_known_market_outage",
            dataset="equities_daily",
            message="2020-10-01の東証終日売買停止による公式欠損です",
        )
        self._add_frame(
            frame.loc[all_missing & ~known_outage],
            severity="warning",
            check_name=f"{label}_ohlc_missing",
            dataset="equities_daily",
            message="売買不成立、売買停止またはデータ欠損の可能性があります",
        )

        complete = ~values.isna().any(axis=1)
        maximum = values[[columns[0], columns[3], columns[2]]].max(axis=1)
        minimum = values[[columns[0], columns[3], columns[1]]].min(axis=1)
        invalid = complete & ((values[columns[1]] < maximum) | (values[columns[2]] > minimum))
        observed = values.apply(
            lambda row: ",".join("" if pd.isna(value) else str(value) for value in row),
            axis=1,
        )
        self._add_frame(
            frame.loc[invalid],
            severity="error",
            check_name=f"{label}_ohlc_inconsistent",
            dataset="equities_daily",
            message="high/lowがopen/closeを包含していません",
            observed=observed.loc[invalid],
        )

    def _check_volume(self, frame: pd.DataFrame) -> None:
        volume = pd.to_numeric(frame["volume"], errors="coerce")
        turnover = pd.to_numeric(frame["turnover_value"], errors="coerce")
        self._add_frame(
            frame.loc[volume == 0],
            severity="warning",
            check_name="zero_volume",
            dataset="equities_daily",
            message="出来高が0です",
            observed=volume.loc[volume == 0],
        )
        self._add_frame(
            frame.loc[volume < 0],
            severity="error",
            check_name="negative_volume",
            dataset="equities_daily",
            message="出来高が負です",
            observed=volume.loc[volume < 0],
        )
        self._add_frame(
            frame.loc[turnover < 0],
            severity="error",
            check_name="negative_turnover_value",
            dataset="equities_daily",
            message="売買代金が負です",
            observed=turnover.loc[turnover < 0],
        )
        traded = frame[["open", "high", "low", "close"]].notna().any(axis=1)
        missing_volume = traded & volume.isna()
        self._add_frame(
            frame.loc[missing_volume],
            severity="error",
            check_name="volume_missing_with_price",
            dataset="equities_daily",
            message="価格が存在するのに出来高が欠損しています",
        )

    def _check_returns_and_splits(self, frame: pd.DataFrame) -> None:
        ordered = frame.sort_values(["code", "date"]).copy()
        adjusted_close = pd.to_numeric(ordered["adjusted_close"], errors="coerce")
        previous_adjusted_close = adjusted_close.groupby(ordered["code"]).shift(1)
        adjusted_return = adjusted_close / previous_adjusted_close - 1.0
        abnormal = (
            adjusted_return.abs() > self.settings.abnormal_return_threshold
        ) & adjusted_return.notna()
        self._add_frame(
            ordered.loc[abnormal],
            severity="warning",
            check_name="abnormal_adjusted_return",
            dataset="equities_daily",
            message=(
                "調整済み終値の日次リターンが閾値 "
                f"{self.settings.abnormal_return_threshold:.0%} を超えています"
            ),
            observed=adjusted_return.loc[abnormal],
        )

        factor = pd.to_numeric(ordered["adjustment_factor"], errors="coerce")
        invalid_factor = factor.notna() & (factor <= 0)
        self._add_frame(
            ordered.loc[invalid_factor],
            severity="error",
            check_name="invalid_adjustment_factor",
            dataset="equities_daily",
            message="調整係数が0以下です",
            observed=factor.loc[invalid_factor],
        )

        split_event = factor.notna() & ~np.isclose(factor, 1.0) & ~invalid_factor
        self._add_frame(
            ordered.loc[split_event],
            severity="info",
            check_name="stock_split_event",
            dataset="equities_daily",
            message="株式分割・併合の調整係数を検出しました",
            observed=factor.loc[split_event],
        )

        previous_close = (
            pd.to_numeric(ordered["close"], errors="coerce").groupby(ordered["code"]).shift(1)
        )
        current_open = pd.to_numeric(ordered["open"], errors="coerce")
        raw_gap_ratio = current_open / previous_close
        split_deviation = (raw_gap_ratio / factor - 1.0).abs()
        inconsistent_split = (
            split_event
            & split_deviation.notna()
            & (split_deviation > self.settings.split_ratio_tolerance)
        )
        self._add_frame(
            ordered.loc[inconsistent_split],
            severity="warning",
            check_name="stock_split_raw_price_mismatch",
            dataset="equities_daily",
            message=(
                "分割日の未調整価格ギャップが調整係数と整合しません "
                f"(許容差 {self.settings.split_ratio_tolerance:.0%})"
            ),
            observed=split_deviation.loc[inconsistent_split],
        )

        split_adjusted_jump = split_event & abnormal
        self._add_frame(
            ordered.loc[split_adjusted_jump],
            severity="warning",
            check_name="stock_split_adjusted_price_jump",
            dataset="equities_daily",
            message="分割日前後の調整済み終値が大きく不連続です",
            observed=adjusted_return.loc[split_adjusted_jump],
        )

    def _check_historical_membership(self, equities: pd.DataFrame, master: pd.DataFrame) -> None:
        if master.empty:
            self._add_dataset_issue(
                severity="error",
                check_name="historical_master_missing",
                dataset="listed_master",
                message="過去時点の銘柄一覧がありません",
            )
            return
        members = master[["date", "code"]].drop_duplicates().assign(_listed=True)
        joined = equities[["date", "code"]].merge(members, how="left", on=["date", "code"])
        missing = joined["_listed"].isna()
        self._add_frame(
            joined.loc[missing],
            severity="error",
            check_name="price_without_historical_membership",
            dataset="equities_daily",
            message="株価日に対応する過去時点の銘柄一覧に存在しません",
        )

    def _check_trading_days(self, equities: pd.DataFrame, calendar: pd.DataFrame) -> None:
        if calendar.empty:
            self._add_dataset_issue(
                severity="error",
                check_name="trading_calendar_missing",
                dataset="trading_calendar",
                message="公式取引カレンダーがありません",
            )
            return
        business_days = set(
            pd.to_datetime(
                calendar.loc[calendar["holiday_division"].astype("string") == "1", "date"]
            ).dt.date
        )
        dates = pd.to_datetime(equities["date"]).dt.date
        invalid = ~dates.isin(business_days)
        self._add_frame(
            equities.loc[invalid],
            severity="error",
            check_name="price_on_non_trading_day",
            dataset="equities_daily",
            message="公式取引カレンダー上の東証営業日ではありません",
        )

    def _add_frame(
        self,
        frame: pd.DataFrame,
        *,
        severity: str,
        check_name: str,
        dataset: str,
        message: str,
        observed: pd.Series | None = None,
    ) -> None:
        for index, row in frame.iterrows():
            value = observed.loc[index] if observed is not None else None
            self._issues.append(
                {
                    "checked_at": self._checked_at,
                    "severity": severity,
                    "check_name": check_name,
                    "dataset": dataset,
                    "date": row.get("date"),
                    "code": row.get("code"),
                    "message": message,
                    "observed_value": None if pd.isna(value) else str(value),
                }
            )

    def _add_dataset_issue(
        self, *, severity: str, check_name: str, dataset: str, message: str
    ) -> None:
        self._issues.append(
            {
                "checked_at": self._checked_at,
                "severity": severity,
                "check_name": check_name,
                "dataset": dataset,
                "date": None,
                "code": None,
                "message": message,
                "observed_value": None,
            }
        )

    def _save(self, issues: pd.DataFrame, summary: dict[str, Any]) -> tuple[Path, Path]:
        quality_dir = self.settings.processed_dir / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        issues_path = quality_dir / "latest.parquet"
        temporary_issues = issues_path.with_suffix(".parquet.tmp")
        issues.to_parquet(temporary_issues, index=False, engine="pyarrow")
        os.replace(temporary_issues, issues_path)

        summary_path = self.settings.metadata_dir / "quality_latest.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_summary = summary_path.with_suffix(".json.tmp")
        with temporary_summary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_summary, summary_path)
        return issues_path, summary_path
