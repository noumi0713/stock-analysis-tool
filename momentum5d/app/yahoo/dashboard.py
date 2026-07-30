from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import Settings
from app.yahoo.ingestion import YahooPaths


class DashboardExporter:
    """分析結果と候補銘柄の短期チャートを表示用JSONへ変換する。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.paths = YahooPaths(settings.data_dir / "yahoo")

    def export(self, output: Path) -> dict[str, Any]:
        analysis_path = self.paths.metadata_dir / "analysis_latest.json"
        quality_path = self.paths.metadata_dir / "quality_latest.json"
        candidates_path = self.paths.processed_dir / "analysis" / "latest_candidates.parquet"
        if (
            not analysis_path.exists()
            or not candidates_path.exists()
            or not self.paths.prices_path.exists()
        ):
            raise RuntimeError("分析結果がありません。先に yahoo-analyze を実行してください")

        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        quality = (
            json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
        )
        candidates = pd.read_parquet(candidates_path)
        company_names = self._load_company_names()
        prices = pd.read_parquet(
            self.paths.prices_path,
            columns=[
                "date",
                "code",
                "open",
                "high",
                "low",
                "close",
                "adjusted_close",
                "volume",
            ],
        )
        candidate_columns = [
            "ticker",
            "code",
            "close",
            "adjusted_close",
            "return_1d",
            "return_5d",
            "return_20d",
            "volume_change_1d",
            "volume_ratio_5_20",
            "breakout_20d",
            "volatility_10d",
            "range_width_10d",
            "up_volume_share_10d",
            "sector_17_code",
            "sector_17_name",
            "sector_33_code",
            "sector_33_name",
            "sector_17_median_return_5d",
            "sector_17_median_return_20d",
            "sector_17_breadth_5d",
            "sector_17_trend_score",
            "individual_trend_score",
            "setup_reasons",
            "setup_score",
            "trend_ranking_score",
            "signal_score",
        ]
        records: list[dict[str, Any]] = []
        for rank, row in enumerate(candidates[candidate_columns].to_dict("records"), start=1):
            code = str(row["code"])
            records.append(
                {
                    "rank": rank,
                    "company_name": company_names.get(code),
                    **{key: _json_scalar(value) for key, value in row.items()},
                }
            )

        candidate_codes = {str(record["code"]) for record in records}
        chart_prices = prices[prices["code"].astype(str).isin(candidate_codes)].copy()
        chart_prices["code"] = chart_prices["code"].astype(str)
        chart_prices = chart_prices.sort_values(["code", "date"])
        charts: dict[str, list[dict[str, Any]]] = {}
        chart_columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
        ]
        for code, rows in chart_prices.groupby("code", sort=False):
            chart_records: list[dict[str, Any]] = []
            for row in rows.tail(60)[chart_columns].to_dict("records"):
                chart_records.append(
                    {
                        "date": pd.Timestamp(row["date"]).date().isoformat(),
                        **{key: _json_scalar(value) for key, value in row.items() if key != "date"},
                    }
                )
            charts[code] = chart_records

        payload = {
            "schema_version": 4,
            "source": "yfinance",
            "personal_research_only": True,
            "generated_at": datetime.now(UTC).isoformat(),
            "latest_date": analysis["latest_date"],
            "metrics": {
                "rows": analysis["rows"],
                "tickers": analysis["tickers"],
                "positive_rate": analysis["positive_rate"],
                "excluded_rows": analysis.get("excluded_non_trading_or_invalid_rows", 0),
                "quality_warnings": quality.get("severity_counts", {}).get("warning", 0),
            },
            "market_regime": analysis.get("market_regime"),
            "industry_trends": analysis.get("industry_trends", {}),
            "patterns": analysis["patterns"],
            "candidates": records,
            "charts": charts,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(f"{output.suffix}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return {
            "output": str(output.resolve()),
            "latest_date": payload["latest_date"],
            "candidate_count": len(records),
        }

    def _load_company_names(self) -> dict[str, str]:
        """銘柄一覧に名称がない場合も、リポジトリ同梱マスターで補完する。"""
        names: dict[str, str] = {}
        fallback_path = self.settings.data_dir.parent / "config" / "prime_names.csv"
        if fallback_path.exists():
            fallback = pd.read_csv(fallback_path, dtype={"code": "string"})
            if {"code", "company_name"}.issubset(fallback.columns):
                names.update(_company_name_map(fallback))

        if self.paths.universe_path.exists():
            universe = pd.read_parquet(self.paths.universe_path)
            if {"code", "company_name"}.issubset(universe.columns):
                names.update(_company_name_map(universe))
        return names


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _company_name_map(frame: pd.DataFrame) -> dict[str, str]:
    records = frame[["code", "company_name"]].dropna().to_dict("records")
    return {
        str(row["code"]).strip(): str(row["company_name"]).strip()
        for row in records
        if str(row["company_name"]).strip()
    }
