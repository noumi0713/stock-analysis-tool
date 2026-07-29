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
    """分析済みの最小限の集計結果だけをサイト送信用JSONへ変換する。"""

    def __init__(self, settings: Settings) -> None:
        self.paths = YahooPaths(settings.data_dir / "yahoo")

    def export(self, output: Path) -> dict[str, Any]:
        analysis_path = self.paths.metadata_dir / "analysis_latest.json"
        quality_path = self.paths.metadata_dir / "quality_latest.json"
        candidates_path = self.paths.processed_dir / "analysis" / "latest_candidates.parquet"
        if not analysis_path.exists() or not candidates_path.exists():
            raise RuntimeError("分析結果がありません。先に yahoo-analyze を実行してください")

        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        quality = (
            json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
        )
        candidates = pd.read_parquet(candidates_path)
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
            "signal_score",
        ]
        records: list[dict[str, Any]] = []
        for rank, row in enumerate(candidates[candidate_columns].to_dict("records"), start=1):
            records.append(
                {
                    "rank": rank,
                    **{key: _json_scalar(value) for key, value in row.items()},
                }
            )

        payload = {
            "schema_version": 1,
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
            "patterns": analysis["patterns"],
            "candidates": records,
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


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
