from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from app.config import Settings
from app.storage.parquet import ParquetStore
from app.yahoo.bottom_patterns import analyze_bottom_patterns
from app.yahoo.candle_sequence import analyze_three_up_one_down
from app.yahoo.corporate_actions import normalize_split_adjusted_prices
from app.yahoo.demand_supply import analyze_demand_supply_timing
from app.yahoo.events import add_event_risk_controls, load_event_calendars
from app.yahoo.golden_cross import backtest_golden_cross_volume
from app.yahoo.ingestion import YahooPaths
from app.yahoo.perfect_order import backtest_perfect_order_pullbacks
from app.yahoo.retail_flow import (
    RETAIL_DETAIL_COLUMNS,
    add_retail_flow_features,
    observed_inflow_reasons,
    retail_flow_reasons,
)
from app.yahoo.rise_pattern import (
    add_latest_ml_sharp_selloff_signals,
    add_latest_rise_pattern_signals,
    backtest_rise_pattern_signals,
)
from app.yahoo.sakata import PATTERN_COLUMNS, add_sakata_features
from app.yahoo.trend import (
    add_trend_features,
    latest_sector_trends,
    load_sector_map,
    weekly_sector_33_returns,
)


class YahooPatternAnalyzer:
    """+5%到達前を集計し、個人投資家の注意・追随行動で候補を順位化する。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
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
        sectors = load_sector_map(self.settings.data_dir.parent / "config" / "prime_sectors.csv")
        features = add_trend_features(features, sectors)
        features = add_retail_flow_features(features)
        features["setup_score"] = features["observed_inflow_score"]
        features["trend_ranking_score"] = features["observed_inflow_score"]
        features["signal_score"] = features["observed_inflow_score"]
        historical = features.loc[features["horizon_complete"]].copy()
        three_up_one_down_study = analyze_three_up_one_down(valid_prices)
        bottom_pattern_study, bottom_events = analyze_bottom_patterns(features)
        rise_pattern_backtest = backtest_rise_pattern_signals(features, bottom_events)
        demand_supply_study = analyze_demand_supply_timing(features)
        golden_cross_volume_study = backtest_golden_cross_volume(features)
        perfect_order_pullback_study = backtest_perfect_order_pullbacks(features)
        features = add_latest_rise_pattern_signals(features, bottom_events)
        features = add_latest_ml_sharp_selloff_signals(features)
        earnings_calendar, important_events = load_event_calendars(self.settings)
        features, event_risk_summary = add_event_risk_controls(
            features,
            earnings_calendar,
            important_events,
        )
        features["signal_score"] = features[
            [
                "observed_inflow_score",
                "rise_pattern_probability",
                "ml_sharp_probability",
            ]
        ].max(axis=1)
        features["setup_score"] = features["signal_score"]
        features["trend_ranking_score"] = features["signal_score"]
        market_regime = self._market_regime(features)
        candidates = self._latest_candidates(features, top_n)
        latest_scores = self._latest_scores(features, candidates)
        patterns = self._pattern_summary(historical)
        industry_trends = {
            "sector_17": latest_sector_trends(features, level="17", top_n=10),
            "sector_33": latest_sector_trends(features, level="33", top_n=10),
            "sector_33_weekly": weekly_sector_33_returns(features),
        }

        analysis_dir = self.paths.processed_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = analysis_dir / "latest_candidates.parquet"
        score_path = analysis_dir / "latest_scores.parquet"
        feature_path = analysis_dir / "historical_patterns.parquet"
        bottom_event_path = analysis_dir / "bottom_pattern_events.parquet"
        ParquetStore._atomic_parquet(candidates, candidate_path)
        ParquetStore._atomic_parquet(latest_scores, score_path)
        ParquetStore._atomic_parquet(historical, feature_path)
        ParquetStore._atomic_parquet(bottom_events, bottom_event_path)
        from app.yahoo.catalog import YahooDuckDBCatalog

        YahooDuckDBCatalog(self.paths).refresh()

        summary = {
            "source": "yfinance",
            "personal_research_only": True,
            "technical_method": "observed_inflow_plus_sharp_selloff_ml_v2",
            "technical_method_label": "資金流入観測＋急落継続ML",
            "analyzed_at": datetime.now(UTC).isoformat(),
            "rows": len(features),
            "excluded_non_trading_or_invalid_rows": len(prices) - len(valid_prices),
            "tickers": int(features["ticker"].nunique()),
            "latest_date": str(features["date"].max()),
            "historical_complete_rows": len(historical),
            "positive_5d_rows": int(historical["target_5d"].sum()),
            "positive_rate": float(historical["target_5d"].mean()),
            "candidate_count": len(candidates),
            "sector_map_coverage": float(features["sector_17_code"].notna().mean()),
            "market_regime": market_regime,
            "industry_trends": industry_trends,
            "patterns": patterns,
            "bottom_pattern_study": bottom_pattern_study,
            "rise_pattern_backtest": rise_pattern_backtest,
            "demand_supply_study": demand_supply_study,
            "golden_cross_volume_study": golden_cross_volume_study,
            "perfect_order_pullback_study": perfect_order_pullback_study,
            "three_up_one_down_study": three_up_one_down_study,
            "event_risk_summary": event_risk_summary,
            "candidate_path": str(candidate_path),
            "score_path": str(score_path),
            "historical_path": str(feature_path),
            "bottom_event_path": str(bottom_event_path),
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

        delta = group["adjusted_close"].diff()
        gains = delta.clip(lower=0.0)
        losses = -delta.clip(upper=0.0)
        average_gain = gains.groupby(frame["ticker"], sort=False).transform(
            lambda x: x.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        )
        average_loss = losses.groupby(frame["ticker"], sort=False).transform(
            lambda x: x.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        )
        relative_strength = average_gain / average_loss
        frame["rsi_14"] = 100.0 - 100.0 / (1.0 + relative_strength)
        frame.loc[(average_loss == 0) & (average_gain > 0), "rsi_14"] = 100.0
        frame.loc[(average_loss == 0) & (average_gain == 0), "rsi_14"] = 50.0

        adjustment_ratio = frame["adjusted_close"] / frame["close"]
        adjusted_high = frame["high"] * adjustment_ratio
        adjusted_low = frame["low"] * adjustment_ratio
        previous_close = group["adjusted_close"].shift(1)
        true_range = pd.concat(
            [
                adjusted_high - adjusted_low,
                (adjusted_high - previous_close).abs(),
                (adjusted_low - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        frame["atr_14"] = true_range.groupby(frame["ticker"], sort=False).transform(
            lambda x: x.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        )
        frame["atr_14_pct"] = frame["atr_14"] / frame["adjusted_close"]
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
        frame["legacy_setup_score"] = (
            0.28 * frame["setup_volume_onset_score"]
            + 0.18 * frame["setup_active_volatility_score"]
            + 0.18 * frame["setup_position_score"]
            + 0.14 * frame["setup_accumulation_score"]
            + 0.12 * frame["setup_early_momentum_score"]
            + 0.10 * frame["setup_trend_score"]
        )
        frame = add_sakata_features(frame)
        # 既存JSONとの互換性を保ちつつ、中身を酒田五法スコアへ切り替える。
        frame["setup_score"] = frame["sakata_score"]
        frame["signal_score"] = frame["sakata_score"]
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        return frame

    @staticmethod
    def _latest_candidates(features: pd.DataFrame, top_n: int) -> pd.DataFrame:
        latest = features.loc[features["date"] == features["date"].max()].copy()
        trend_defaults: dict[str, Any] = {
            "sector_17_code": pd.NA,
            "sector_17_name": pd.NA,
            "sector_33_code": pd.NA,
            "sector_33_name": pd.NA,
            "sector_17_median_return_5d": np.nan,
            "sector_17_median_return_20d": np.nan,
            "sector_17_breadth_5d": np.nan,
            "sector_17_trend_score": np.nan,
            "individual_trend_score": np.nan,
            "return_60d": np.nan,
            "relative_return_20d": np.nan,
            "rsi_14": np.nan,
            "atr_14_pct": np.nan,
            "retail_flow_score": 0.0,
            "retail_attention_hybrid_score": 0.0,
            "retail_discovery_score": 0.0,
            "retail_understanding_proxy_score": 0.0,
            "retail_expectation_score": 0.0,
            "retail_safety_score": 0.0,
            "retail_action_score": 0.0,
            "retail_overheat_penalty": 0.0,
            "retail_loss_anxiety_penalty": 0.0,
            "sakata_pattern": "該当なし",
            "sakata_score": 0.0,
            "sakata_buy_signal": False,
            "sakata_sell_signal": False,
            "sakata_bullish_count": 0,
            "sakata_bearish_count": 0,
            "rise_pattern_probability": 0.0,
            "rise_pattern_samples": 0,
            "rise_pattern_signal": False,
            "rise_pattern_shape": pd.NA,
            "rise_pattern_reason": "",
            "ml_sharp_probability": 0.0,
            "ml_sharp_down_5pct_probability": 1.0,
            "ml_sharp_down_8pct_probability": 1.0,
            "ml_sharp_expected_net_return": -1.0,
            "ml_sharp_model_samples": 0,
            "ml_sharp_signal": False,
            "ml_sharp_rank": pd.NA,
            "ml_sharp_reason": "",
            "ml_sharp_entry_rule": "翌営業日寄付きが前日終値以下の場合のみ有効",
            "earnings_calendar_covered": False,
            "next_earnings_date": pd.NaT,
            "earnings_days_ahead": pd.NA,
            "earnings_crossing_risk": False,
            "earnings_exit_date": pd.NaT,
            "important_event_nearby": False,
            "important_event_name": "",
            "event_position_scale": 1.0,
            "event_entry_allowed": False,
            "event_trade_action": "CHECK_EARNINGS_CALENDAR",
            "event_risk_reason": "決算予定未取得のため新規買い不可",
            "earnings_gap_down": np.nan,
            "earnings_gd_reversal_signal": False,
            "earnings_gd_entry_price": np.nan,
            "earnings_gd_stop_price": np.nan,
            "earnings_gd_take_profit": np.nan,
            "earnings_gd_reason": "",
        }
        for column in RETAIL_DETAIL_COLUMNS:
            trend_defaults.setdefault(column, 0.0)
        for column, default in trend_defaults.items():
            if column not in latest:
                latest[column] = default
        latest["rise_pattern_probability"] = pd.to_numeric(
            latest["rise_pattern_probability"], errors="coerce"
        ).fillna(0.0)
        latest["rise_pattern_samples"] = pd.to_numeric(
            latest["rise_pattern_samples"], errors="coerce"
        ).fillna(0)
        latest["rise_pattern_signal"] = latest["rise_pattern_signal"].map(
            lambda value: bool(value) if pd.notna(value) else False
        )
        latest["rise_pattern_reason"] = latest["rise_pattern_reason"].fillna("")
        if "trend_ranking_score" not in latest:
            latest["trend_ranking_score"] = latest["setup_score"]
        latest["signal_score"] = latest[
            [
                "observed_inflow_score",
                "rise_pattern_probability",
                "ml_sharp_probability",
            ]
        ].max(axis=1)
        latest["trend_ranking_score"] = latest["signal_score"]
        latest["setup_score"] = latest["signal_score"]
        observed_signal = (
            latest["observed_inflow_confirmed"].fillna(False).astype(bool)
            & latest["return_1d"].between(0.002, 0.10)
            & latest["return_5d"].between(-0.05, 0.18)
        )
        rise_signal = latest["rise_pattern_signal"].fillna(False).astype(bool)
        ml_sharp_signal = latest["ml_sharp_signal"].fillna(False).astype(bool)
        earnings_gd_signal = latest["earnings_gd_reversal_signal"].fillna(False).astype(bool)
        latest = latest.loc[
            (observed_signal | rise_signal | ml_sharp_signal | earnings_gd_signal)
            & latest["rsi_14"].le(82.0)
            & (latest["turnover_value"] >= 10_000_000)
        ].copy()
        latest["sakata_reasons"] = latest.apply(_sakata_reasons, axis=1)
        latest["retail_flow_reasons"] = latest.apply(retail_flow_reasons, axis=1)
        latest["observed_inflow_reasons"] = latest.apply(observed_inflow_reasons, axis=1)
        latest["setup_reasons"] = latest.apply(_combined_signal_reasons, axis=1)
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
            "intraday_return",
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
            "sector_17_code",
            "sector_17_name",
            "sector_33_code",
            "sector_33_name",
            "sector_17_median_return_5d",
            "sector_17_median_return_20d",
            "sector_17_breadth_5d",
            "sector_17_trend_score",
            "individual_trend_score",
            "return_60d",
            "relative_return_20d",
            "rsi_14",
            "atr_14_pct",
            *RETAIL_DETAIL_COLUMNS,
            "retail_flow_reasons",
            "observed_inflow_reasons",
            "sakata_pattern",
            "sakata_reasons",
            "sakata_score",
            "sakata_buy_signal",
            "sakata_sell_signal",
            "sakata_bullish_count",
            "sakata_bearish_count",
            "rise_pattern_probability",
            "rise_pattern_samples",
            "rise_pattern_signal",
            "rise_pattern_shape",
            "rise_pattern_reason",
            "ml_sharp_probability",
            "ml_sharp_down_5pct_probability",
            "ml_sharp_down_8pct_probability",
            "ml_sharp_expected_net_return",
            "ml_sharp_model_samples",
            "ml_sharp_signal",
            "ml_sharp_rank",
            "ml_sharp_reason",
            "ml_sharp_entry_rule",
            "earnings_calendar_covered",
            "next_earnings_date",
            "earnings_days_ahead",
            "earnings_crossing_risk",
            "earnings_exit_date",
            "important_event_nearby",
            "important_event_name",
            "event_position_scale",
            "event_entry_allowed",
            "event_trade_action",
            "event_risk_reason",
            "earnings_gap_down",
            "earnings_gd_reversal_signal",
            "earnings_gd_entry_price",
            "earnings_gd_stop_price",
            "earnings_gd_take_profit",
            "earnings_gd_reason",
            "setup_reasons",
            "setup_score",
            "trend_ranking_score",
            "signal_score",
        ]
        return (
            latest.sort_values(
                [
                    "earnings_gd_reversal_signal",
                    "event_entry_allowed",
                    "ml_sharp_signal",
                    "signal_score",
                    "code",
                ],
                ascending=[False, False, False, False, True],
            )[columns]
            .head(top_n)
            .reset_index(drop=True)
        )

    @staticmethod
    def _latest_scores(
        features: pd.DataFrame,
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        latest = features.loc[features["date"] == features["date"].max()].copy()
        latest["sakata_reasons"] = latest.apply(_sakata_reasons, axis=1)
        latest["retail_flow_reasons"] = latest.apply(retail_flow_reasons, axis=1)
        latest["observed_inflow_reasons"] = latest.apply(observed_inflow_reasons, axis=1)
        latest["setup_reasons"] = latest.apply(_combined_signal_reasons, axis=1)
        latest["score_rank"] = latest["trend_ranking_score"].rank(method="min", ascending=False)
        latest["score_percentile"] = latest["trend_ranking_score"].rank(
            method="average",
            pct=True,
        )
        candidate_codes = set(candidates["code"].astype(str))
        latest["is_ranked_candidate"] = latest["code"].astype(str).isin(candidate_codes)
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
            "return_60d",
            "intraday_return",
            "volume_change_1d",
            "volume_ratio_5_20",
            "close_to_ma20",
            "breakout_20d",
            "volatility_10d",
            "range_width_10d",
            "up_volume_share_10d",
            "rsi_14",
            "atr_14_pct",
            "relative_return_20d",
            "individual_trend_score",
            "sector_17_code",
            "sector_17_name",
            "sector_33_code",
            "sector_33_name",
            "sector_17_trend_score",
            *RETAIL_DETAIL_COLUMNS,
            "retail_flow_reasons",
            "observed_inflow_reasons",
            "sakata_pattern",
            "sakata_reasons",
            "sakata_score",
            "sakata_buy_signal",
            "sakata_sell_signal",
            "sakata_bullish_count",
            "sakata_bearish_count",
            "rise_pattern_probability",
            "rise_pattern_samples",
            "rise_pattern_signal",
            "rise_pattern_shape",
            "rise_pattern_reason",
            "ml_sharp_probability",
            "ml_sharp_down_5pct_probability",
            "ml_sharp_down_8pct_probability",
            "ml_sharp_expected_net_return",
            "ml_sharp_model_samples",
            "ml_sharp_signal",
            "ml_sharp_rank",
            "ml_sharp_reason",
            "ml_sharp_entry_rule",
            "earnings_calendar_covered",
            "next_earnings_date",
            "earnings_days_ahead",
            "earnings_crossing_risk",
            "earnings_exit_date",
            "important_event_nearby",
            "important_event_name",
            "event_position_scale",
            "event_entry_allowed",
            "event_trade_action",
            "event_risk_reason",
            "earnings_gap_down",
            "earnings_gd_reversal_signal",
            "earnings_gd_entry_price",
            "earnings_gd_stop_price",
            "earnings_gd_take_profit",
            "earnings_gd_reason",
            "setup_reasons",
            "setup_score",
            "trend_ranking_score",
            "signal_score",
            "score_rank",
            "score_percentile",
            "is_ranked_candidate",
        ]
        return (
            latest[columns]
            .sort_values(
                ["trend_ranking_score", "code"],
                ascending=[False, True],
            )
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
            "sakata_score",
            "sakata_bullish_count",
            "sakata_bearish_count",
            "individual_trend_score",
            "sector_17_trend_score",
            "sector_33_trend_score",
            "trend_ranking_score",
            *RETAIL_DETAIL_COLUMNS,
            *PATTERN_COLUMNS,
        ]
        metrics = [metric for metric in metrics if metric in historical]
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


def _sakata_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    pattern = str(row.get("sakata_pattern", "該当なし"))
    if pattern != "該当なし":
        reasons.append(pattern)
    sector_name = row.get("sector_17_name")
    sector_score = row.get("sector_17_trend_score")
    if pd.notna(sector_name) and pd.notna(sector_score) and float(sector_score) >= 0.67:
        reasons.append(f"{sector_name}トレンド")
    if row.get("sakata_bullish_count", 0) >= 2:
        reasons.append("買い型が複数一致")
    if pd.notna(row.get("rsi_14")) and float(row["rsi_14"]) <= 70:
        reasons.append("過熱圏未満")
    return "・".join(reasons[:3]) or "酒田五法の明確な型なし"


def _combined_signal_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    gd_signal = row.get("earnings_gd_reversal_signal", False)
    if pd.notna(gd_signal) and bool(gd_signal) and row.get("earnings_gd_reason"):
        reasons.append(str(row["earnings_gd_reason"]))
    event_reason = str(row.get("event_risk_reason", "")).strip()
    if event_reason and event_reason != "イベント制約なし":
        reasons.append(event_reason)
    ml_signal = row.get("ml_sharp_signal", False)
    if pd.notna(ml_signal) and bool(ml_signal) and row.get("ml_sharp_reason"):
        reasons.append(str(row["ml_sharp_reason"]))
    rise_signal = row.get("rise_pattern_signal", False)
    if pd.notna(rise_signal) and bool(rise_signal) and row.get("rise_pattern_reason"):
        reasons.append(str(row["rise_pattern_reason"]))
    observed_signal = row.get("observed_inflow_confirmed", False)
    if pd.notna(observed_signal) and bool(observed_signal):
        observed = str(row.get("observed_inflow_reasons", "")).strip()
        if observed:
            reasons.append(observed)
    return "・".join(reasons) or str(row.get("observed_inflow_reasons", ""))
