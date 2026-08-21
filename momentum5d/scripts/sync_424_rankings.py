from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

MAX_RANK = 3
REFERENCE_MINIMUM_TURNOVER_YEN = 300_000_000
REFERENCE_SHAPE = "capitulation_reversal"
REFERENCE_SHAPE_LABEL = "投げ売り反転"
REFERENCE_ENTRY_LIMIT_OFFSET = 0.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any], *, compact: bool = False) -> None:
    if compact:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    else:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
            allow_nan=False,
        )
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _top_three(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda row: (
            str(row.get("signal_date") or ""),
            int(row.get("rank") or 9999),
            str(row.get("ticker") or ""),
        ),
    )
    return [row for row in ordered if int(row.get("rank") or 9999) <= MAX_RANK]


def _reference_parameters(study: dict[str, Any]) -> dict[str, Any]:
    parameters = dict(study.get("parameters") or {})
    parameters.update(
        {
            "method": "deterministic_rsi14_rank1",
            "technical_profile": "rsi14_exhaustive_rank1",
            "rsi_period": 14,
            "rsi_min": 25.0,
            "rsi_max": 33.0,
            "return_1d_min": -0.03,
            "return_1d_max": 0.0,
            "return_5d_min": -0.15,
            "return_5d_max": 0.02,
            "volume_ratio_min": 1.8,
            "minimum_turnover_yen": REFERENCE_MINIMUM_TURNOVER_YEN,
            "atr_14_pct_min": 0.02,
            "atr_14_pct_max": 0.10,
            "ma25": "below",
            "bullish": True,
            "top_n_per_day": MAX_RANK,
        }
    )
    return parameters


def _normalize_demo_study(study: dict[str, Any]) -> dict[str, Any]:
    live = _top_three(list(study.get("live_signals") or []))
    historical = _top_three(list(study.get("historical_signals") or []))
    study["minimum_turnover_yen"] = REFERENCE_MINIMUM_TURNOVER_YEN
    for key in (
        "probability_threshold",
        "maximum_down_5pct_probability",
        "maximum_down_8pct_probability",
        "minimum_expected_net_return",
    ):
        study.pop(key, None)
    study["technical_profile"] = "rsi14_exhaustive_rank1"
    study["maximum_signals_per_day"] = MAX_RANK
    study["entry_limit_offset_from_previous_close"] = REFERENCE_ENTRY_LIMIT_OFFSET
    study["parameters"] = _reference_parameters(study)
    study["live_signals"] = live
    study["live_signal_count"] = len(live)
    study["historical_signals"] = historical
    study["historical_signal_count"] = len(historical)
    study["note"] = (
        "RSI14固定の全3億2006万160通り総当たり1位を固定。RSI25〜33、"
        "前日比-3〜0%、5日騰落-15〜+2%、出来高比1.8倍以上、売買代金3億円以上、"
        "ATR2〜10%、25日線下の陽線を出来高比順に1日最大3銘柄。"
    )
    return study


def _sync_parquet_rankings(
    analysis: dict[str, Any],
    scores_path: Path,
    candidates_path: Path,
) -> int:
    ten_day = analysis.get("ten_day_signal_study") or {}
    study = ten_day.get("demo_trade_signal_study") or {}
    live = list(study.get("live_signals") or [])
    scores = pd.read_parquet(scores_path)

    if "ticker" not in scores.columns:
        raise RuntimeError("latest_scores.parquet has no ticker column")

    scores["ticker"] = scores["ticker"].astype(str)
    for column, value in (
        ("ml_ten_day_signal", False),
        ("ml_ten_day_rank", pd.NA),
        ("is_ranked_candidate", False),
    ):
        if column in scores.columns:
            scores[column] = value

    rank_by_ticker = {str(row["ticker"]): int(row["rank"]) for row in live}
    missing = sorted(set(rank_by_ticker).difference(set(scores["ticker"])))
    if missing:
        raise RuntimeError(f"RSI14 rank-1 live signals are missing from latest scores: {missing}")

    for record in live:
        ticker = str(record["ticker"])
        mask = scores["ticker"].eq(ticker)
        overlays = {
            "ml_ten_day_signal": True,
            "ml_ten_day_rank": int(record["rank"]),
            "ml_ten_day_probability": record.get("target_probability"),
            "ml_ten_day_down_5pct_probability": record.get("down_5pct_probability"),
            "ml_ten_day_down_8pct_probability": record.get("down_8pct_probability"),
            "ml_ten_day_expected_net_return": record.get("expected_net_return"),
            "ml_ten_day_reason": record.get("reason") or (
                f"{REFERENCE_SHAPE_LABEL}・RSI14全件テスト1位・10営業日+5%候補"
            ),
            "ml_ten_day_entry_rule": "翌営業日始値でエントリー",
            "rise_pattern_shape": record.get("shape") or REFERENCE_SHAPE,
            "is_ranked_candidate": True,
        }
        for column, value in overlays.items():
            if column in scores.columns:
                scores.loc[mask, column] = value
        probability = record.get("target_probability")
        if probability is not None:
            for column in ("signal_score", "setup_score", "trend_ranking_score"):
                if column in scores.columns:
                    scores.loc[mask, column] = float(probability)
        if "score_rank" in scores.columns:
            scores.loc[mask, "score_rank"] = int(record["rank"])

    if live:
        candidates = scores.loc[scores["ticker"].isin(rank_by_ticker)].copy()
        candidates["_shared_424_rank"] = candidates["ticker"].map(rank_by_ticker)
        candidates = candidates.sort_values(["_shared_424_rank", "ticker"])
        candidates = candidates.drop(columns=["_shared_424_rank"])
    else:
        candidates = scores.iloc[0:0].copy()

    candidates = candidates.head(MAX_RANK).reset_index(drop=True)
    scores.to_parquet(scores_path, index=False)
    candidates.to_parquet(candidates_path, index=False)

    analysis["candidate_count"] = len(candidates)
    ten_day["minimum_turnover_yen"] = REFERENCE_MINIMUM_TURNOVER_YEN
    ten_day["maximum_candidates_per_day"] = MAX_RANK
    ten_day["chosen_parameters"] = dict(study.get("parameters") or {})
    ten_day["live_signal_count"] = len(candidates)
    ten_day["live_candidate"] = live
    analysis["ten_day_signal_study"] = ten_day
    return len(candidates)


def sync_analysis(data_dir: Path) -> int:
    analysis_path = data_dir / "metadata" / "analysis_latest.json"
    scores_path = data_dir / "processed" / "analysis" / "latest_scores.parquet"
    candidates_path = data_dir / "processed" / "analysis" / "latest_candidates.parquet"
    analysis = _load_json(analysis_path)
    ten_day = analysis.get("ten_day_signal_study") or {}
    study = ten_day.get("demo_trade_signal_study")
    if not isinstance(study, dict) or study.get("status") != "completed":
        raise RuntimeError("RSI14 rank-1 signal study is unavailable")
    ten_day["demo_trade_signal_study"] = _normalize_demo_study(study)
    analysis["ten_day_signal_study"] = ten_day
    count = _sync_parquet_rankings(analysis, scores_path, candidates_path)
    _write_json(analysis_path, analysis)
    return count


def patch_dashboard(path: Path) -> None:
    payload = _load_json(path)
    ten_day = payload.get("ten_day_signal_study") or {}
    study = ten_day.get("demo_trade_signal_study") or {}
    live = _top_three(list(study.get("live_signals") or []))
    study["live_signals"] = live
    study["live_signal_count"] = len(live)
    study["maximum_signals_per_day"] = MAX_RANK
    study["parameters"] = _reference_parameters(study)
    ten_day["demo_trade_signal_study"] = study
    payload["ten_day_signal_study"] = ten_day

    technical = payload.get("technical_method") or {}
    technical["key"] = "rsi14_exhaustive_rank1_10d_v1"
    technical["label"] = "RSI14全件テスト1位・投げ売り反転10D"
    technical["note"] = (
        "RSI14固定の全件総当たり1位を抽出エンジンとして使用。"
        "RSI25〜33、前日比-3〜0%、5日騰落-15〜+2%、出来高比1.8倍以上、"
        "売買代金3億円以上、ATR2〜10%、25日線下の陽線。"
    )
    payload["technical_method"] = technical

    signal_model = payload.get("signal_model") or {}
    signal_model["key"] = "rsi14_exhaustive_rank1_10d_v1"
    signal_model["label"] = "RSI14全件テスト1位ランキング"
    signal_model["conditions"] = {
        "target_holding_days": 10,
        "target_return": 0.05,
        "method": "deterministic_rsi14_rank1",
        "shape": REFERENCE_SHAPE,
        "minimum_turnover_yen": REFERENCE_MINIMUM_TURNOVER_YEN,
        "rsi_period": 14,
        "rsi_min": 25.0,
        "rsi_max": 33.0,
        "return_1d_min": -0.03,
        "return_1d_max": 0.0,
        "return_5d_min": -0.15,
        "return_5d_max": 0.02,
        "volume_ratio_min": 1.8,
        "atr_14_pct_min": 0.02,
        "atr_14_pct_max": 0.10,
        "ma25": "below",
        "bullish": True,
        "maximum_candidates_per_day": MAX_RANK,
        "entry_limit_offset_from_previous_close": REFERENCE_ENTRY_LIMIT_OFFSET,
        "entry_rule": "翌営業日始値",
        "take_profit_pct": 0.05,
        "stop_loss_pct": -0.12,
        "holding_days": 10,
    }
    reference = study.get("reference_result") or {}
    signal_model["historical_results"] = {
        "signals": reference.get("completed_trades"),
        "trade_win_rate": reference.get("trade_win_rate"),
        "mean_trade_net_return": reference.get("mean_trade_net_return"),
        "profit_factor": reference.get("profit_factor"),
        "reference_only": True,
    }
    signal_model["live_signal_count"] = min(
        MAX_RANK,
        sum(bool(record.get("ml_ten_day_signal")) for record in payload.get("candidates", [])),
    )
    signal_model["note"] = (
        "全3億2006万160通りを同一3年間で比較した探索結果。"
        "完全な未使用期間による前向き確認は継続して必要。"
    )
    payload["signal_model"] = signal_model
    payload["candidates"] = list(payload.get("candidates") or [])[:MAX_RANK]
    _write_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/yahoo"))
    parser.add_argument("--dashboard", type=Path)
    args = parser.parse_args()

    if args.dashboard is not None:
        patch_dashboard(args.dashboard)
        print(f"Patched dashboard to RSI14 rank-1 ranking: {args.dashboard}")
        return

    count = sync_analysis(args.data_dir)
    print(f"RSI14 rank-1 ranking prepared: {count} candidate(s), max={MAX_RANK}")


if __name__ == "__main__":
    main()
