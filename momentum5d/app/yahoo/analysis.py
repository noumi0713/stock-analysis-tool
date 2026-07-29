from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from app.config import Settings
from app.storage.parquet import ParquetStore
from app.yahoo.ingestion import YahooPaths


class YahooPatternAnalyzer:
    """+5%到達前の値動き・出来高を記述し、最新日の候補をルール順位化する。"""

    def __init__(self, settings: Settings) -> None:
        self.paths = YahooPaths(settings.data_dir / "yahoo")
        self.paths.ensure()

    def run(self, *, top_n: int = 20) -> dict[str, Any]:
        if not self.paths.prices_path.exists():
            raise RuntimeError(
                "Yahoo Finance日足がありません。先に yahoo-ingest を実行してください"
            )
        prices = pd.read_parquet(self.paths.prices_path)
        valid_prices = prices.loc[
            (pd.to_numeric(prices["volume"], errors="coerce") > 0)
            & prices[["open", "high", "low", "close", "adjusted_close"]]
            .apply(pd.to_numeric, errors="coerce")
            .gt(0)
            .all(axis=1)
        ].copy()
        features = self._build_features(valid_prices)
        historical = features.loc[features["horizon_complete"]].copy()
        candidates = self._latest_candidates(features, top_n)
        patterns = self._pattern_summary(historical)

        analysis_dir = self.paths.processed_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = analysis_dir / "latest_candidates.parquet"
        feature_path = analysis_dir / "historical_patterns.parquet"
        ParquetStore._atomic_parquet(candidates, candidate_path)
        ParquetStore._atomic_parquet(historical, feature_path)
        from app.yahoo.catalog import YahooDuckDBCatalog

        YahooDuckDBCatalog(self.paths).refresh()

        summary = {
            "source": "yfinance",
            "personal_research_only": True,
            "analyzed_at": datetime.now(UTC).isoformat(),
            "rows": len(features),
            "excluded_non_trading_or_invalid_rows": len(prices) - len(valid_prices),
            "tickers": int(features["ticker"].nunique()),
            "latest_date": str(features["date"].max()),
            "historical_complete_rows": len(historical),
            "positive_5d_rows": int(historical["target_5d"].sum()),
            "positive_rate": float(historical["target_5d"].mean()),
            "candidate_count": len(candidates),
            "patterns": patterns,
            "candidate_path": str(candidate_path),
            "historical_path": str(feature_path),
        }
        output = self.paths.metadata_dir / "analysis_latest.json"
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return summary

    @staticmethod
    def _build_features(prices: pd.DataFrame) -> pd.DataFrame:
        required = {
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
        }
        missing = required.difference(prices.columns)
        if missing:
            raise ValueError(f"Yahoo日足の必須列がありません: {sorted(missing)}")
        frame = prices.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
        group = frame.groupby("ticker", sort=False)

        frame["return_1d"] = frame["adjusted_close"] / group["adjusted_close"].shift(1) - 1
        frame["return_5d"] = frame["adjusted_close"] / group["adjusted_close"].shift(5) - 1
        frame["return_20d"] = frame["adjusted_close"] / group["adjusted_close"].shift(20) - 1
        frame["volume_change_1d"] = frame["volume"] / group["volume"].shift(1) - 1
        volume_ma5 = group["volume"].transform(lambda x: x.rolling(5, min_periods=5).mean())
        volume_ma20 = group["volume"].transform(lambda x: x.rolling(20, min_periods=20).mean())
        close_ma20 = group["adjusted_close"].transform(
            lambda x: x.rolling(20, min_periods=20).mean()
        )
        prior_high20 = group["adjusted_close"].transform(
            lambda x: x.shift(1).rolling(20, min_periods=20).max()
        )
        frame["volume_ratio_5_20"] = volume_ma5 / volume_ma20
        frame["close_to_ma20"] = frame["adjusted_close"] / close_ma20 - 1
        frame["breakout_20d"] = frame["adjusted_close"] / prior_high20 - 1
        frame["intraday_return"] = frame["close"] / frame["open"] - 1
        frame["range_rate"] = frame["high"] / frame["low"] - 1

        adjustment_ratio = frame["adjusted_close"] / frame["close"]
        adjusted_high = frame["high"] * adjustment_ratio
        future_highs = [
            adjusted_high.groupby(frame["ticker"], sort=False).shift(-offset)
            for offset in range(1, 6)
        ]
        future = pd.concat(future_highs, axis=1)
        frame["future_max_return"] = future.max(axis=1) / frame["adjusted_close"] - 1
        frame["horizon_complete"] = future.notna().all(axis=1)
        frame["target_5d"] = (
            (frame["future_max_return"] >= 0.05) & frame["horizon_complete"]
        ).astype("int8")

        rank_features = {
            "return_5d_rank": "return_5d",
            "return_20d_rank": "return_20d",
            "volume_change_rank": "volume_change_1d",
            "volume_ratio_rank": "volume_ratio_5_20",
            "breakout_rank": "breakout_20d",
            "ma20_rank": "close_to_ma20",
            "range_rank": "range_rate",
        }
        for output, source in rank_features.items():
            frame[output] = frame.groupby("date")[source].rank(pct=True)
        frame["signal_score"] = (
            0.10 * frame["return_5d_rank"]
            + 0.15 * frame["return_20d_rank"]
            + 0.15 * frame["volume_change_rank"]
            + 0.25 * frame["volume_ratio_rank"]
            + 0.15 * frame["ma20_rank"]
            + 0.10 * (1.0 - frame["breakout_rank"])
            + 0.10 * frame["range_rank"]
        )
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        return frame

    @staticmethod
    def _latest_candidates(features: pd.DataFrame, top_n: int) -> pd.DataFrame:
        latest = features.loc[features["date"] == features["date"].max()].copy()
        latest = latest.loc[
            latest["signal_score"].notna()
            & (latest["volume_ratio_5_20"] >= 1.0)
            & (latest["turnover_value"] >= 10_000_000)
        ]
        columns = [
            "date",
            "ticker",
            "code",
            "close",
            "adjusted_close",
            "volume",
            "turnover_value",
            "return_1d",
            "return_5d",
            "return_20d",
            "volume_change_1d",
            "volume_ratio_5_20",
            "close_to_ma20",
            "breakout_20d",
            "signal_score",
        ]
        return (
            latest.sort_values("signal_score", ascending=False)[columns]
            .head(top_n)
            .reset_index(drop=True)
        )

    @staticmethod
    def _pattern_summary(historical: pd.DataFrame) -> dict[str, dict[str, float | None]]:
        metrics = [
            "return_1d",
            "return_5d",
            "return_20d",
            "volume_change_1d",
            "volume_ratio_5_20",
            "close_to_ma20",
            "breakout_20d",
            "intraday_return",
            "range_rate",
        ]
        result: dict[str, dict[str, float | None]] = {}
        positive = historical["target_5d"] == 1
        for metric in metrics:
            positive_median = historical.loc[positive, metric].median()
            other_median = historical.loc[~positive, metric].median()
            result[metric] = {
                "positive_median": _finite_or_none(positive_median),
                "other_median": _finite_or_none(other_median),
                "difference": _finite_or_none(positive_median - other_median),
            }
        return result


def _finite_or_none(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None
