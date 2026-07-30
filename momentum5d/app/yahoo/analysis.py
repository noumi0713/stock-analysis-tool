from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from app.config import Settings
from app.storage.parquet import ParquetStore
from app.yahoo.corporate_actions import normalize_split_adjusted_prices
from app.yahoo.ingestion import YahooPaths


class YahooPatternAnalyzer:
    """+5%到達前の値動きを記述し、上昇前の仕込み候補をルール順位化する。"""

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
        valid_prices = normalize_split_adjusted_prices(valid_prices)
        features = self._build_features(valid_prices)
        historical = features.loc[features["horizon_complete"]].copy()
        market_regime = self._market_regime(features)
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
            "market_regime": market_regime,
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
        frame["volatility_10d"] = frame.groupby("ticker", sort=False)["return_1d"].transform(
            lambda x: x.rolling(10, min_periods=10).std()
        )
        frame["volatility_20d"] = frame.groupby("ticker", sort=False)["return_1d"].transform(
            lambda x: x.rolling(20, min_periods=20).std()
        )
        close_high10 = group["adjusted_close"].transform(
            lambda x: x.rolling(10, min_periods=10).max()
        )
        close_low10 = group["adjusted_close"].transform(
            lambda x: x.rolling(10, min_periods=10).min()
        )
        frame["range_width_10d"] = (close_high10 - close_low10) / frame["adjusted_close"]
        up_volume = frame["volume"].where(frame["return_1d"] > 0, 0.0)
        rolling_up_volume = up_volume.groupby(frame["ticker"], sort=False).transform(
            lambda x: x.rolling(10, min_periods=10).sum()
        )
        rolling_total_volume = frame.groupby("ticker", sort=False)["volume"].transform(
            lambda x: x.rolling(10, min_periods=10).sum()
        )
        frame["up_volume_share_10d"] = rolling_up_volume / rolling_total_volume

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

        frame["setup_compression_score"] = 0.60 * (1.0 - frame["volatility_10d"] / 0.035).clip(
            0.0, 1.0
        ) + 0.40 * (1.0 - frame["range_width_10d"] / 0.12).clip(0.0, 1.0)
        frame["setup_accumulation_score"] = ((frame["up_volume_share_10d"] - 0.42) / 0.25).clip(
            0.0, 1.0
        )
        frame["setup_quiet_volume_score"] = (
            1.0 - (frame["volume_ratio_5_20"] - 1.0).abs() / 0.75
        ).clip(0.0, 1.0)
        frame["setup_position_score"] = (1.0 - (frame["breakout_20d"] + 0.04).abs() / 0.08).clip(
            0.0, 1.0
        )
        frame["setup_calm_score"] = (1.0 - frame["return_5d"].abs() / 0.06).clip(0.0, 1.0)
        frame["setup_trend_score"] = (1.0 - (frame["return_20d"] - 0.03).abs() / 0.15).clip(
            0.0, 1.0
        )
        frame["setup_volume_onset_score"] = (
            1.0 - (frame["volume_ratio_5_20"] - 1.30).abs() / 0.70
        ).clip(0.0, 1.0)
        frame["setup_active_volatility_score"] = (
            1.0 - (frame["volatility_10d"] - 0.020).abs() / 0.020
        ).clip(0.0, 1.0)
        frame["setup_early_momentum_score"] = (
            1.0 - (frame["return_5d"] - 0.015).abs() / 0.055
        ).clip(0.0, 1.0)
        frame["setup_score"] = (
            0.28 * frame["setup_volume_onset_score"]
            + 0.18 * frame["setup_active_volatility_score"]
            + 0.18 * frame["setup_position_score"]
            + 0.14 * frame["setup_accumulation_score"]
            + 0.12 * frame["setup_early_momentum_score"]
            + 0.10 * frame["setup_trend_score"]
        )
        frame["signal_score"] = frame["setup_score"]
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        return frame

    @staticmethod
    def _latest_candidates(features: pd.DataFrame, top_n: int) -> pd.DataFrame:
        latest = features.loc[features["date"] == features["date"].max()].copy()
        favorable_regime = YahooPatternAnalyzer._market_regime(features)["favorable"]
        latest = latest.loc[
            favorable_regime
            & latest["setup_score"].ge(0.55)
            & latest["return_1d"].between(-0.03, 0.025)
            & latest["return_5d"].between(-0.03, 0.04)
            & latest["return_20d"].between(-0.08, 0.15)
            & latest["breakout_20d"].between(-0.10, 0.0)
            & latest["close_to_ma20"].between(-0.04, 0.08)
            & latest["volatility_10d"].between(0.012, 0.035)
            & latest["volume_ratio_5_20"].between(0.85, 1.65)
            & latest["up_volume_share_10d"].ge(0.48)
            & (latest["turnover_value"] >= 10_000_000)
        ].copy()
        latest["setup_reasons"] = latest.apply(_setup_reasons, axis=1)
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
            "volatility_10d",
            "volatility_20d",
            "range_width_10d",
            "up_volume_share_10d",
            "setup_compression_score",
            "setup_accumulation_score",
            "setup_position_score",
            "setup_reasons",
            "setup_score",
            "signal_score",
        ]
        return (
            latest.sort_values("setup_score", ascending=False)[columns]
            .head(top_n)
            .reset_index(drop=True)
        )

    @staticmethod
    def _market_regime(features: pd.DataFrame) -> dict[str, Any]:
        latest_date = features["date"].max()
        latest = features.loc[features["date"] == latest_date]
        breadth_5d = float((latest["return_5d"] > 0).mean())
        median_return_20d = float(latest["return_20d"].median())
        favorable = breadth_5d > 0.50 and median_return_20d > 0.0
        return {
            "date": str(latest_date),
            "favorable": favorable,
            "breadth_5d": breadth_5d,
            "median_return_20d": median_return_20d,
            "rule": "5日上昇銘柄比率>50%かつ20日リターン中央値>0",
        }

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
            "volatility_10d",
            "range_width_10d",
            "up_volume_share_10d",
            "setup_score",
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


def _setup_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    if 1.05 <= row["volume_ratio_5_20"] <= 1.55:
        reasons.append("出来高立ち上がり")
    if 0.015 <= row["volatility_10d"] <= 0.030:
        reasons.append("初動の値動き")
    if row["up_volume_share_10d"] >= 0.54:
        reasons.append("上昇日の出来高優勢")
    if -0.08 <= row["breakout_20d"] <= -0.01:
        reasons.append("20日高値の手前")
    if 0.0 <= row["return_5d"] <= 0.03:
        reasons.append("緩やかな上向き")
    return "・".join(reasons[:3]) or "出来高を伴う初動"
