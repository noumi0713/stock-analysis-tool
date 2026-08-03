from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

RETAIL_STAGE_COLUMNS = (
    "retail_discovery_score",
    "retail_understanding_proxy_score",
    "retail_expectation_score",
    "retail_safety_score",
    "retail_action_score",
)

RETAIL_DETAIL_COLUMNS = (
    "turnover_ratio_5_20",
    "retail_volume_attention_rank",
    "retail_return_attention_rank",
    "retail_turnover_rank",
    "retail_relative_strength_rank",
    "retail_attention_acceleration_score",
    "retail_overheat_penalty",
    "retail_loss_anxiety_penalty",
    "retail_flow_score",
    "retail_attention_hybrid_score",
    *RETAIL_STAGE_COLUMNS,
)

_STAGE_LABELS = {
    "retail_discovery_score": "発見が増加",
    "retail_understanding_proxy_score": "業種の物語が明瞭",
    "retail_expectation_score": "上昇期待が初動",
    "retail_safety_score": "流動性と下値安定",
    "retail_action_score": "買い行動を確認",
}


def add_retail_flow_features(features: pd.DataFrame) -> pd.DataFrame:
    """当日までの価格・出来高・業種から個人投資家の資金流入代理値を作る。"""
    required = {
        "date",
        "ticker",
        "adjusted_close",
        "turnover_value",
        "return_1d",
        "return_5d",
        "return_20d",
        "volume_ratio_5_20",
        "intraday_return",
        "breakout_20d",
        "close_to_ma20",
        "volatility_10d",
        "up_volume_share_10d",
        "rsi_14",
        "relative_return_20d",
        "sector_17_median_return_5d",
        "sector_17_breadth_5d",
        "sector_17_trend_score",
        "legacy_setup_score",
    }
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"個人投資家フロー計算の必須列がありません: {sorted(missing)}")

    frame = features.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    ticker_group = frame.groupby("ticker", sort=False)
    turnover_ma5 = ticker_group["turnover_value"].transform(
        lambda values: values.rolling(5, min_periods=5).mean()
    )
    turnover_ma20 = ticker_group["turnover_value"].transform(
        lambda values: values.rolling(20, min_periods=20).mean()
    )
    frame["turnover_ratio_5_20"] = turnover_ma5 / turnover_ma20
    frame["downside_return_10d"] = ticker_group["return_1d"].transform(
        lambda values: values.rolling(10, min_periods=10).min()
    )

    frame["_absolute_return_1d"] = frame["return_1d"].abs()
    date_group = frame.groupby("date", sort=False)
    frame["retail_volume_attention_rank"] = date_group["volume_ratio_5_20"].rank(
        pct=True
    )
    frame["retail_return_attention_rank"] = date_group["_absolute_return_1d"].rank(
        pct=True
    )
    frame["retail_turnover_rank"] = date_group["turnover_value"].rank(pct=True)
    frame["retail_turnover_onset_rank"] = date_group["turnover_ratio_5_20"].rank(
        pct=True
    )
    frame["retail_relative_strength_rank"] = date_group["relative_return_20d"].rank(
        pct=True
    )
    frame["retail_sector_momentum_rank"] = date_group[
        "sector_17_median_return_5d"
    ].rank(pct=True)
    volume_attention_change = (
        frame["volume_ratio_5_20"]
        / ticker_group["volume_ratio_5_20"].shift(3)
        - 1
    )
    turnover_attention_change = (
        frame["turnover_ratio_5_20"]
        / frame.groupby("ticker", sort=False)["turnover_ratio_5_20"].shift(3)
        - 1
    )
    frame["_volume_attention_change"] = volume_attention_change
    frame["_turnover_attention_change"] = turnover_attention_change
    change_group = frame.groupby("date", sort=False)
    frame["retail_attention_acceleration_score"] = (
        0.50 * change_group["_volume_attention_change"].rank(pct=True)
        + 0.50 * change_group["_turnover_attention_change"].rank(pct=True)
    )

    high_visibility = ((frame["breakout_20d"] + 0.12) / 0.14).clip(0.0, 1.0)
    frame["retail_discovery_score"] = (
        0.40 * frame["retail_attention_acceleration_score"]
        + 0.25 * frame["retail_volume_attention_rank"]
        + 0.20 * frame["retail_return_attention_rank"]
        + 0.15 * high_visibility
    )

    frame["retail_understanding_proxy_score"] = (
        0.55 * frame["sector_17_trend_score"]
        + 0.25 * frame["sector_17_breadth_5d"]
        + 0.20 * frame["retail_sector_momentum_rank"]
    )

    early_momentum = (1.0 - (frame["return_5d"] - 0.025).abs() / 0.11).clip(
        0.0, 1.0
    )
    frame["retail_expectation_score"] = (
        0.30 * early_momentum
        + 0.25 * frame["retail_relative_strength_rank"]
        + 0.25 * frame["sector_17_trend_score"]
        + 0.20 * high_visibility
    )

    volatility_comfort = (
        1.0 - (frame["volatility_10d"] - 0.018).abs() / 0.030
    ).clip(0.0, 1.0)
    support_comfort = (1.0 - frame["close_to_ma20"].abs() / 0.10).clip(0.0, 1.0)
    downside_stability = (1.0 - (-frame["downside_return_10d"]) / 0.10).clip(
        0.0, 1.0
    )
    frame["retail_safety_score"] = (
        0.35 * frame["retail_turnover_rank"]
        + 0.25 * volatility_comfort
        + 0.20 * support_comfort
        + 0.20 * downside_stability
    )

    volume_onset = (
        1.0 - (frame["volume_ratio_5_20"] - 1.50).abs() / 1.50
    ).clip(0.0, 1.0)
    turnover_onset = (
        1.0 - (frame["turnover_ratio_5_20"] - 1.45).abs() / 1.45
    ).clip(0.0, 1.0)
    positive_candle = ((frame["intraday_return"] + 0.01) / 0.05).clip(0.0, 1.0)
    positive_start = ((frame["return_1d"] + 0.01) / 0.05).clip(0.0, 1.0)
    up_volume = ((frame["up_volume_share_10d"] - 0.40) / 0.35).clip(0.0, 1.0)
    frame["retail_action_score"] = (
        0.25 * volume_onset
        + 0.20 * turnover_onset
        + 0.20 * frame["retail_attention_acceleration_score"]
        + 0.15 * up_volume
        + 0.10 * positive_candle
        + 0.10 * positive_start
    )

    frame["retail_overheat_penalty"] = (
        0.35 * ((frame["rsi_14"] - 70.0) / 20.0).clip(0.0, 1.0)
        + 0.30 * ((frame["return_5d"] - 0.08) / 0.12).clip(0.0, 1.0)
        + 0.20 * ((frame["return_1d"] - 0.04) / 0.08).clip(0.0, 1.0)
        + 0.15 * ((frame["volume_ratio_5_20"] - 3.0) / 4.0).clip(0.0, 1.0)
    )
    frame["retail_loss_anxiety_penalty"] = (
        0.35 * ((-frame["return_20d"] - 0.08) / 0.20).clip(0.0, 1.0)
        + 0.25 * ((-frame["close_to_ma20"] - 0.05) / 0.12).clip(0.0, 1.0)
        + 0.25 * ((frame["volatility_10d"] - 0.04) / 0.04).clip(0.0, 1.0)
        + 0.15 * ((-frame["intraday_return"] - 0.04) / 0.08).clip(0.0, 1.0)
    )

    weights = np.array([0.25, 0.10, 0.25, 0.15, 0.25], dtype="float64")
    stages = frame[list(RETAIL_STAGE_COLUMNS)].clip(0.0, 1.0)
    weighted_mean = stages.mul(weights, axis=1).sum(axis=1)
    geometric_sequence = [
        np.power(stages[column].clip(lower=0.05), weight)
        for column, weight in zip(RETAIL_STAGE_COLUMNS, weights, strict=True)
    ]
    geometric_mean = pd.concat(geometric_sequence, axis=1).prod(axis=1)
    frame["retail_flow_score"] = (
        0.55 * geometric_mean
        + 0.45 * weighted_mean
        - 0.20 * frame["retail_overheat_penalty"]
        - 0.12 * frame["retail_loss_anxiety_penalty"]
    ).clip(0.0, 1.0)
    # 価格の仕込み条件を土台にし、投資家の注意・追随行動を順位の補助に使う。
    # 40%は独立した過去期間と評価期間の双方で検証した固定比率。
    sector_setup = frame["sector_17_trend_score"].fillna(frame["legacy_setup_score"])
    setup_with_sector = 0.75 * frame["legacy_setup_score"] + 0.25 * sector_setup
    frame["retail_attention_hybrid_score"] = (
        0.60 * setup_with_sector + 0.40 * frame["retail_flow_score"]
    ).clip(0.0, 1.0)
    frame.drop(
        columns=[
            "_absolute_return_1d",
            "_volume_attention_change",
            "_turnover_attention_change",
        ],
        inplace=True,
    )
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    return frame


def retail_flow_reasons(row: pd.Series, *, limit: int = 3) -> str:
    """画面とコピー用に、スコア上位の段階を日本語で返す。"""
    stages: Sequence[tuple[str, float]] = sorted(
        (
            (label, float(row.get(column, 0.0)))
            for column, label in _STAGE_LABELS.items()
            if pd.notna(row.get(column))
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    reasons = [label for label, score in stages if score >= 0.55][:limit]
    sector_name = row.get("sector_17_name")
    sector_score = row.get("sector_17_trend_score")
    if (
        len(reasons) < limit
        and pd.notna(sector_name)
        and pd.notna(sector_score)
        and float(sector_score) >= 0.60
    ):
        reasons.append(f"{sector_name}へ資金回転")
    return "・".join(reasons[:limit]) or "注意・期待・行動の一致待ち"
