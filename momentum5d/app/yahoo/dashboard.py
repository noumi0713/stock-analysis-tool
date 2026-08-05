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
        scores_path = self.paths.processed_dir / "analysis" / "latest_scores.parquet"
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
        ingestion = (
            json.loads(self.paths.status_path.read_text(encoding="utf-8"))
            if self.paths.status_path.exists()
            else {}
        )
        candidates = pd.read_parquet(candidates_path)
        scores = pd.read_parquet(scores_path) if scores_path.exists() else candidates.copy()
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
            "intraday_return",
            "volume_change_1d",
            "volume_ratio_5_20",
            "volume_ratio_1_20",
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
            "relative_return_20d",
            "rsi_14",
            "atr_14_pct",
            "turnover_ratio_5_20",
            "turnover_ratio_1_20",
            "observed_volume_ratio_rank",
            "observed_turnover_ratio_rank",
            "observed_volume_intensity_score",
            "observed_turnover_intensity_score",
            "observed_price_confirmation_score",
            "observed_inflow_score",
            "observed_inflow_confirmed",
            "retail_volume_attention_rank",
            "retail_return_attention_rank",
            "retail_turnover_rank",
            "retail_relative_strength_rank",
            "retail_attention_acceleration_score",
            "retail_discovery_score",
            "retail_understanding_proxy_score",
            "retail_expectation_score",
            "retail_safety_score",
            "retail_action_score",
            "retail_overheat_penalty",
            "retail_loss_anxiety_penalty",
            "retail_flow_score",
            "retail_attention_hybrid_score",
            "retail_flow_reasons",
            "observed_inflow_reasons",
            "sakata_pattern",
            "sakata_reasons",
            "sakata_score",
            "sakata_buy_signal",
            "sakata_sell_signal",
            "sakata_bullish_count",
            "sakata_bearish_count",
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

        score_columns = [
            "ticker",
            "code",
            "close",
            "adjusted_close",
            "return_1d",
            "return_5d",
            "return_20d",
            "intraday_return",
            "volume_change_1d",
            "volume_ratio_5_20",
            "volume_ratio_1_20",
            "breakout_20d",
            "volatility_10d",
            "range_width_10d",
            "up_volume_share_10d",
            "sector_17_code",
            "sector_17_name",
            "sector_33_code",
            "sector_33_name",
            "sector_17_trend_score",
            "individual_trend_score",
            "relative_return_20d",
            "rsi_14",
            "atr_14_pct",
            "turnover_ratio_5_20",
            "turnover_ratio_1_20",
            "observed_volume_ratio_rank",
            "observed_turnover_ratio_rank",
            "observed_volume_intensity_score",
            "observed_turnover_intensity_score",
            "observed_price_confirmation_score",
            "observed_inflow_score",
            "observed_inflow_confirmed",
            "retail_volume_attention_rank",
            "retail_return_attention_rank",
            "retail_turnover_rank",
            "retail_relative_strength_rank",
            "retail_attention_acceleration_score",
            "retail_discovery_score",
            "retail_understanding_proxy_score",
            "retail_expectation_score",
            "retail_safety_score",
            "retail_action_score",
            "retail_overheat_penalty",
            "retail_loss_anxiety_penalty",
            "retail_flow_score",
            "retail_attention_hybrid_score",
            "retail_flow_reasons",
            "observed_inflow_reasons",
            "sakata_pattern",
            "sakata_reasons",
            "sakata_score",
            "sakata_buy_signal",
            "sakata_sell_signal",
            "sakata_bullish_count",
            "sakata_bearish_count",
            "setup_reasons",
            "setup_score",
            "trend_ranking_score",
            "signal_score",
            "score_rank",
            "score_percentile",
            "is_ranked_candidate",
        ]
        available_score_columns = [
            column for column in score_columns if column in scores.columns
        ]
        stock_records = []
        for row in scores[available_score_columns].to_dict("records"):
            code = str(row["code"])
            stock_records.append(
                {
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

        generated_at = datetime.now(UTC).isoformat()
        intraday = ingestion.get("intraday")
        if isinstance(intraday, dict) and intraday.get("status") == "complete":
            update = {
                "status": "complete",
                "session": intraday["session"],
                "session_label": intraday["session_label"],
                "market_date": intraday["market_date"],
                "data_through": intraday["data_through"],
                "interval": intraday["interval"],
                "successful_tickers": intraday["successful_tickers"],
                "coverage": intraday["coverage"],
                "generated_at": generated_at,
            }
        else:
            update = {
                "status": "complete",
                "session": "daily",
                "session_label": "日足",
                "market_date": analysis["latest_date"],
                "data_through": None,
                "interval": "1d",
                "successful_tickers": analysis["tickers"],
                "coverage": 1.0,
                "generated_at": generated_at,
            }

        payload = {
            "schema_version": 9,
            "source": "yfinance",
            "personal_research_only": True,
            "generated_at": generated_at,
            "latest_date": analysis["latest_date"],
            "update": update,
            "metrics": {
                "rows": analysis["rows"],
                "tickers": analysis["tickers"],
                "positive_rate": analysis["positive_rate"],
                "excluded_rows": analysis.get("excluded_non_trading_or_invalid_rows", 0),
                "quality_warnings": quality.get("severity_counts", {}).get("warning", 0),
            },
            "market_regime": analysis.get("market_regime"),
            "industry_trends": analysis.get("industry_trends", {}),
            "technical_method": {
                "key": "observed_inflow_v1",
                "label": "資金流入観測",
                "stages": [
                    "発見",
                    "理解（業種連動の代理）",
                    "期待",
                    "安心",
                    "行動",
                ],
                "available_inputs": [
                    "株価",
                    "出来高",
                    "売買代金",
                    "17業種トレンド",
                ],
                "unavailable_inputs": [
                    "ニュース件数",
                    "SNS言及数",
                    "決算・上方修正",
                    "信用残",
                ],
                "note": (
                    "当日の出来高・売買代金を各銘柄の過去20日平均と"
                    "Prime内順位で比較し、株価上昇と陽線の一致を確認する代理スコア"
                ),
            },
            "patterns": analysis["patterns"],
            "indicator_notes": [
                {
                    "key": "observed_inflow_score",
                    "label": "資金流入観測スコア",
                    "reason": (
                        "当日出来高・売買代金の増加と株価上昇が"
                        "同時に確認できた銘柄を評価する"
                    ),
                },
                {
                    "key": "volume_ratio_1_20",
                    "label": "当日出来高倍率",
                    "reason": (
                        "当日の出来高を直前20営業日の平均と比較し、"
                        "通常時を上回る参加を確認する"
                    ),
                },
                {
                    "key": "turnover_ratio_1_20",
                    "label": "当日売買代金倍率",
                    "reason": "株価水準の違いを含めた実際の取引金額の増加を確認する",
                },
                {
                    "key": "retail_overheat_penalty",
                    "label": "高値警戒",
                    "reason": "RSI、短期上昇率、出来高過熱からFOMO後の高値づかみを減点する",
                },
                {
                    "key": "retail_loss_anxiety_penalty",
                    "label": "損失不安",
                    "reason": "移動平均割れ、下落トレンド、高ボラティリティを減点する",
                },
            ],
            "candidates": records,
            "stocks": stock_records,
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
            "stock_count": len(stock_records),
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
