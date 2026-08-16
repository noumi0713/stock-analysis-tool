from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

MAX_RANK = 3
REFERENCE_MINIMUM_TURNOVER_YEN = 150_000_000
REFERENCE_PROBABILITY_THRESHOLD = 0.55
REFERENCE_MAX_DOWN_5PCT_PROBABILITY = 0.50
REFERENCE_MAX_DOWN_8PCT_PROBABILITY = 0.30
REFERENCE_MIN_EXPECTED_NET_RETURN = -0.01
REFERENCE_SHAPE = "capitulation_reversal"
REFERENCE_SHAPE_LABEL = "投げ売り反転"
REFERENCE_ENTRY_LIMIT_OFFSET = 0.015


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
            "shape_profile": REFERENCE_SHAPE,
            "allowed_shapes": [REFERENCE_SHAPE],
            "regime_profile": "all_regimes",
            "technical_profile": "all_technical",
            "model": "logistic",
            "probability_threshold": REFERENCE_PROBABILITY_THRESHOLD,
            "max_down_5pct_probability": REFERENCE_MAX_DOWN_5PCT_PROBABILITY,
            "max_down_8pct_probability": REFERENCE_MAX_DOWN_8PCT_PROBABILITY,
            "min_expected_net_return": REFERENCE_MIN_EXPECTED_NET_RETURN,
            "top_n_per_day": MAX_RANK,
        }
    )
    return parameters


def _normalize_demo_study(study: dict[str, Any]) -> dict[str, Any]:
    live = _top_three(list(study.get("live_signals") or []))
    historical = _top_three(list(study.get("historical_signals") or []))
    study["minimum_turnover_yen"] = REFERENCE_MINIMUM_TURNOVER_YEN
    study["probability_threshold"] = REFERENCE_PROBABILITY_THRESHOLD
    study["maximum_down_5pct_probability"] = REFERENCE_MAX_DOWN_5PCT_PROBABILITY
    study["maximum_down_8pct_probability"] = REFERENCE_MAX_DOWN_8PCT_PROBABILITY
    study["minimum_expected_net_return"] = REFERENCE_MIN_EXPECTED_NET_RETURN
    study["technical_profile"] = "all_technical"
    study["maximum_signals_per_day"] = MAX_RANK
    study["entry_limit_offset_from_previous_close"] = REFERENCE_ENTRY_LIMIT_OFFSET
    study["parameters"] = _reference_parameters(study)
    study["live_signals"] = live
    study["live_signal_count"] = len(live)
    study["historical_signals"] = historical
    study["historical_signal_count"] = len(historical)
    study["note"] = (
        "424万6,171円の参考結果で採用した投げ売り反転・売買代金1.5億円以上・"
        "Logistic確率55%以上・損失確率条件を共通抽出条件として固定。"
        "モメンタム10Dとカブトレは同じ候補を使い、ランキング表示は1日最大3銘柄。"
        "参考成績はランキング上限変更前の基準値。"
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
        raise RuntimeError(f"424 live signals are missing from latest scores: {missing}")

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
                f"{REFERENCE_SHAPE_LABEL}・売買代金1.5億円以上・10営業日+5%候補"
            ),
            "ml_ten_day_entry_rule": "翌営業日の指値条件を満たす場合のみ有効",
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
        raise RuntimeError("424 demo-trade signal study is unavailable")
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
    technical["key"] = "portfolio_424_capitulation_10d_v1"
    technical["label"] = "424万円基準・投げ売り反転10D"
    technical["note"] = (
        "424万6,171円の参考バックテスト条件を共通抽出エンジンとして使用。"
        "投げ売り反転、売買代金1.5億円以上、Logistic確率55%以上、"
        "-5%確率50%以下、-8%確率30%以下、期待ネットリターン-1%以上。"
        "モメンタム10Dとカブトレは同一候補を順位順に最大3銘柄表示する。"
    )
    payload["technical_method"] = technical

    signal_model = payload.get("signal_model") or {}
    signal_model["key"] = "portfolio_424_capitulation_10d_v1"
    signal_model["label"] = "424万円基準・共通ランキング"
    signal_model["conditions"] = {
        "target_holding_days": 10,
        "target_return": 0.05,
        "shapes": [REFERENCE_SHAPE],
        "minimum_turnover_yen": REFERENCE_MINIMUM_TURNOVER_YEN,
        "model": "logistic",
        "minimum_probability": REFERENCE_PROBABILITY_THRESHOLD,
        "maximum_down_5pct_probability": REFERENCE_MAX_DOWN_5PCT_PROBABILITY,
        "maximum_down_8pct_probability": REFERENCE_MAX_DOWN_8PCT_PROBABILITY,
        "minimum_expected_net_return": REFERENCE_MIN_EXPECTED_NET_RETURN,
        "maximum_candidates_per_day": MAX_RANK,
        "entry_limit_offset_from_previous_close": REFERENCE_ENTRY_LIMIT_OFFSET,
        "entry_rule": "翌営業日の指値条件を満たす場合のみ有効",
    }
    reference = study.get("reference_result") or {}
    signal_model["historical_results"] = {
        "signals": reference.get("completed_trades"),
        "trade_win_rate": reference.get("trade_win_rate"),
        "ending_equity_yen": reference.get("ending_equity_yen"),
        "total_return": reference.get("total_return"),
        "reference_only": True,
    }
    signal_model["live_signal_count"] = min(
        MAX_RANK,
        sum(bool(record.get("ml_ten_day_signal")) for record in payload.get("candidates", [])),
    )
    signal_model["note"] = (
        "424万円の成績は条件固定時の参考バックテスト。今回の変更は"
        "モメンタム10Dとカブトレの抽出元を共通化し、表示上限を3銘柄に統一するもの。"
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
        print(f"Patched dashboard to shared 424 ranking: {args.dashboard}")
        return

    count = sync_analysis(args.data_dir)
    print(f"Shared 424 ranking prepared: {count} candidate(s), max={MAX_RANK}")


if __name__ == "__main__":
    main()
