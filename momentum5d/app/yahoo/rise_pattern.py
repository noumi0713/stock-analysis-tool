from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.yahoo.bottom_patterns import FEATURE_SPECS, SHAPE_LABELS
from app.yahoo.selective_ml import (
    LIVE_STRONG_SIGNAL_COLUMNS,
    LIVE_TEN_DAY_SIGNAL_COLUMNS,
    ML_FEATURES,
    evaluate_frozen_ml_strategy,
    score_latest_strong_shape_candidates,
    score_latest_ten_day_candidates,
    tune_and_select_ml_strategy,
    walk_forward_ml_scores,
)

STRONG_SHAPES = frozenset({"sharp_selloff", "capitulation_reversal", "rounded_base"})
TEN_DAY_ADOPTED_SHAPE = "capitulation_reversal"
TEN_DAY_ADOPTED_SHAPE_LABEL = "投げ売り反転"
STRONG_SHAPE_MIN_TURNOVER = 200_000_000.0
SELECTIVE_SHAPE_FEATURES = (
    "_rise_market_favorable",
    "_rise_theme_flow",
    "_rise_volume_continuation",
    "rise_pattern_reversal_confirmed",
    "_rise_observed_inflow",
)


@dataclass(frozen=True)
class RisePatternConfig:
    min_signal_rate: float = 0.80
    min_samples: int = 30
    prior_strength: float = 20.0
    horizon_days: int = 5
    test_days: int = 60
    top_n: int = 20
    transaction_cost_bps: float = 20.0
    fixed_stop_pcts: tuple[float, ...] = (0.03, 0.04, 0.05)
    atr_stop_multipliers: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    exit_target_pcts: tuple[float, ...] = (0.03, 0.04)
    exit_holding_days: tuple[int, ...] = (2, 3, 4)
    follow_through_checks: tuple[tuple[int, float], ...] = (
        (1, 0.00),
        (1, 0.01),
        (1, 0.02),
        (2, 0.00),
        (2, 0.01),
        (2, 0.02),
    )
    selective_test_days: int = 120
    selective_top_n: int = 3
    selective_probability_threshold: float = 0.70
    selective_probability_grid: tuple[float, ...] = (0.60, 0.65, 0.70, 0.75)
    selective_min_samples: int = 40
    selective_prior_strength: float = 50.0
    selective_max_gap_up: float = 0.02
    selective_max_down_5pct_probability: float = 0.18
    selective_max_down_8pct_probability: float = 0.10
    selective_min_expected_net_return: float = 0.0
    selective_validation_samples: int = 300
    ml_test_days: int = 180
    ml_refit_days: int = 30
    ml_minimum_shape_samples: int = 180
    ml_minimum_development_signals: int = 40


def add_latest_rise_pattern_signals(
    features: pd.DataFrame,
    bottom_events: pd.DataFrame,
    *,
    config: RisePatternConfig | None = None,
) -> pd.DataFrame:
    """Attach actionable pattern signals using information known by the latest close."""
    config = config or RisePatternConfig()
    frame = _add_live_bottom_features(features)
    frame["rise_pattern_probability"] = 0.0
    frame["rise_pattern_samples"] = 0
    frame["rise_pattern_signal"] = False
    frame["rise_pattern_shape"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame["rise_pattern_reason"] = ""
    if frame.empty or bottom_events.empty:
        return frame

    profiles = _calibrate_profiles(bottom_events, config)
    latest_date = frame["date"].max()
    latest_indexes = frame.index[frame["date"] == latest_date]
    for index in latest_indexes:
        row = frame.loc[index]
        if not bool(row["rise_pattern_live_bottom"]):
            continue
        profile = profiles.get(str(row["_rise_shape"]))
        if profile is None:
            continue
        subtype = _match_subtype(row, profile)
        if subtype is None:
            continue
        probability = float(subtype["smoothed_success_rate"])
        samples = int(subtype["samples"])
        signal = probability >= config.min_signal_rate and samples >= config.min_samples
        frame.at[index, "rise_pattern_probability"] = probability
        frame.at[index, "rise_pattern_samples"] = samples
        frame.at[index, "rise_pattern_signal"] = signal
        frame.at[index, "rise_pattern_shape"] = str(row["_rise_shape"])
        frame.at[index, "rise_pattern_reason"] = _signal_reason(
            str(row["_rise_shape"]), subtype, probability
        )
    return frame


def add_latest_ml_sharp_selloff_signals(
    features: pd.DataFrame,
    *,
    config: RisePatternConfig | None = None,
) -> pd.DataFrame:
    """Attach the frozen three-strong-shape ML candidate to the latest date."""
    config = config or RisePatternConfig()
    frame = features.copy()
    defaults: dict[str, Any] = {
        "ml_sharp_probability": 0.0,
        "ml_sharp_down_5pct_probability": 1.0,
        "ml_sharp_down_8pct_probability": 1.0,
        "ml_sharp_expected_net_return": -1.0,
        "ml_sharp_model_samples": 0,
        "ml_sharp_signal": False,
        "ml_sharp_rank": pd.NA,
        "ml_sharp_reason": "",
        "ml_sharp_entry_rule": "翌営業日寄付きが前日終値比+3%以下の場合のみ有効",
    }
    for column, default in defaults.items():
        frame[column] = default
    frame["ml_sharp_rank"] = frame["ml_sharp_rank"].astype("Int64")
    if frame.empty:
        return frame

    outcome_frame = _attach_trade_outcomes(frame, config)
    latest_scores = score_latest_strong_shape_candidates(
        outcome_frame,
        minimum_shape_samples=config.ml_minimum_shape_samples,
        minimum_turnover=STRONG_SHAPE_MIN_TURNOVER,
    )
    for column in LIVE_STRONG_SIGNAL_COLUMNS:
        frame.loc[latest_scores.index, column] = latest_scores[column]
    return frame


def add_ten_day_signal_and_study(
    features: pd.DataFrame,
    *,
    minimum_turnover: float = STRONG_SHAPE_MIN_TURNOVER,
    evaluation_days: int = 240,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach and validate a separate +5% within 10 trading days signal.

    Model and threshold selection use only the first half of the walk-forward
    period.  The second half is reported as untouched validation.  At most one
    live candidate is emitted.
    """
    frame = features.copy()
    defaults: dict[str, Any] = {
        "ml_ten_day_probability": 0.0,
        "ml_ten_day_down_5pct_probability": 1.0,
        "ml_ten_day_down_8pct_probability": 1.0,
        "ml_ten_day_expected_net_return": -1.0,
        "ml_ten_day_model_samples": 0,
        "ml_ten_day_signal": False,
        "ml_ten_day_rank": pd.NA,
        "ml_ten_day_reason": "",
        "ml_ten_day_entry_rule": (
            "翌営業日寄付きが開発期間で選択したギャップ上限以下の場合のみ有効"
        ),
    }
    for column, default in defaults.items():
        frame[column] = default
    frame["ml_ten_day_rank"] = frame["ml_ten_day_rank"].astype("Int64")
    if frame.empty:
        return frame, {"status": "no_features"}

    config = RisePatternConfig(
        horizon_days=10,
        ml_test_days=evaluation_days,
        ml_refit_days=30,
        ml_minimum_shape_samples=180,
        ml_minimum_development_signals=40,
    )
    outcome_frame = _attach_ml_trade_outcomes(
        _add_live_bottom_features(features, include_signature=False),
        config,
    )
    outcome_frame["date"] = pd.to_datetime(outcome_frame["date"]).dt.date
    all_dates = sorted(outcome_frame["date"].drop_duplicates())
    date_position = {value: position for position, value in enumerate(all_dates)}
    complete_dates = [
        value
        for value in all_dates
        if outcome_frame.loc[
            outcome_frame["date"].eq(value), "trade_outcome_available"
        ].any()
    ]
    evaluation_dates = complete_dates[-config.ml_test_days :]
    pool = outcome_frame.loc[
        outcome_frame["rise_pattern_live_bottom"].fillna(False).astype(bool)
        & outcome_frame["_rise_shape"].eq(TEN_DAY_ADOPTED_SHAPE)
        & pd.to_numeric(outcome_frame["turnover_value"], errors="coerce").ge(
            minimum_turnover
        )
        & pd.to_numeric(outcome_frame["rsi_14"], errors="coerce").le(82.0)
    ].copy()
    if not evaluation_dates or pool.empty:
        return frame, {"status": "no_completed_ten_day_pool"}

    scored = walk_forward_ml_scores(
        pool,
        evaluation_dates,
        date_position,
        horizon_days=config.horizon_days,
        refit_days=config.ml_refit_days,
        minimum_shape_samples=config.ml_minimum_shape_samples,
    )
    validation_trades, diagnostics = tune_and_select_ml_strategy(
        scored,
        evaluation_dates,
        minimum_development_signals=20,
        probability_thresholds=(0.45, 0.55, 0.65, 0.70),
        gap_limits=(0.00, 0.03),
        top_n_options=(1,),
        allowed_shape_profiles=(TEN_DAY_ADOPTED_SHAPE,),
        technical_profiles=(
            ("all_technical", {}),
            ("high_atr", {"min_atr_14_pct": 0.05}),
            ("wide_range", {"min_range_width_10d": 0.18}),
            ("trend_strength", {"min_individual_trend_score": 0.45}),
            ("deep_pullback", {"max_return_5d": -0.08}),
            (
                "volatile_pullback",
                {"min_atr_14_pct": 0.05, "max_return_5d": -0.08},
            ),
            (
                "wide_trend",
                {
                    "min_range_width_10d": 0.18,
                    "min_individual_trend_score": 0.45,
                },
            ),
        ),
        allowed_regime_profiles=("all_regimes",),
        split_on_scored_dates=True,
    )
    parameters = diagnostics.get("chosen_parameters")
    if not parameters:
        return frame, {
            "status": diagnostics.get("status", "no_parameters"),
            "diagnostics": diagnostics,
        }
    validation_summary = diagnostics.get("validation", {})
    deployment_approved = bool(
        int(validation_summary.get("selected_signals") or 0) >= 30
        and float(validation_summary.get("target_hit_rate") or 0.0) >= 0.60
        and float(validation_summary.get("mean_trade_net_return") or -1.0) > 0.0
    )
    latest_scores = outcome_frame.loc[
        outcome_frame["date"].eq(outcome_frame["date"].max())
    ].copy()
    if deployment_approved:
        latest_scores = score_latest_ten_day_candidates(
            outcome_frame,
            parameters,
            minimum_shape_samples=config.ml_minimum_shape_samples,
            minimum_turnover=minimum_turnover,
        )
        for column in LIVE_TEN_DAY_SIGNAL_COLUMNS:
            frame.loc[latest_scores.index, column] = latest_scores[column]

    development_end = diagnostics.get("development_end")
    development_dates = [
        value for value in evaluation_dates if development_end and str(value) <= development_end
    ]
    development_scored = scored.loc[scored["date"].isin(development_dates)].copy()
    validation_start = diagnostics.get("validation_start")
    validation_end = diagnostics.get("validation_end")
    comparison_dates = [
        value
        for value in evaluation_dates
        if validation_start
        and validation_end
        and validation_start <= str(value) <= validation_end
    ]
    turnover_thresholds = (
        200_000_000.0,
        150_000_000.0,
        100_000_000.0,
        50_000_000.0,
    )
    turnover_sensitivity_rows: list[dict[str, Any]] = []
    limit_order_validation_trades = validation_trades
    portfolio_scored = scored
    for threshold in turnover_thresholds:
        threshold_pool = outcome_frame.loc[
            outcome_frame["rise_pattern_live_bottom"].fillna(False).astype(bool)
            & outcome_frame["_rise_shape"].eq(TEN_DAY_ADOPTED_SHAPE)
            & pd.to_numeric(outcome_frame["turnover_value"], errors="coerce").ge(
                threshold
            )
            & pd.to_numeric(outcome_frame["rsi_14"], errors="coerce").le(82.0)
        ].copy()
        threshold_scored = (
            scored
            if threshold == minimum_turnover
            else walk_forward_ml_scores(
                threshold_pool,
                evaluation_dates,
                date_position,
                horizon_days=config.horizon_days,
                refit_days=config.ml_refit_days,
                minimum_shape_samples=config.ml_minimum_shape_samples,
            )
        )
        threshold_trades, threshold_evaluation = evaluate_frozen_ml_strategy(
            threshold_scored,
            comparison_dates,
            parameters,
        )
        if threshold == 150_000_000.0:
            limit_order_validation_trades = threshold_trades
            portfolio_scored = threshold_scored
        turnover_sensitivity_rows.append(
            {
                "minimum_turnover_yen": int(threshold),
                "scored_candidates": int(
                    threshold_scored["date"].isin(comparison_dates).sum()
                )
                if not threshold_scored.empty
                else 0,
                "validation": threshold_evaluation["validation"],
                "validation_folds": threshold_evaluation["validation_folds"],
            }
        )
    limit_order_study = calculate_ten_day_limit_order_study(
        outcome_frame,
        limit_order_validation_trades,
        config,
        minimum_turnover=150_000_000.0,
    )
    portfolio_parameters = {**parameters, "top_n_per_day": 2}
    portfolio_candidates, _ = evaluate_frozen_ml_strategy(
        portfolio_scored,
        evaluation_dates,
        portfolio_parameters,
    )
    portfolio_study = calculate_ten_day_portfolio_study(
        outcome_frame,
        portfolio_candidates,
        all_dates,
        config,
        initial_cash=1_000_000.0,
        minimum_turnover=150_000_000.0,
        limit_offset=0.015,
        maximum_daily_buys=2,
        maximum_positions=3,
        lot_size=100,
    )
    live_candidate_records = (
        latest_scores.loc[
            latest_scores["ml_ten_day_signal"],
            [
                "ticker",
                "_rise_shape",
                "ml_ten_day_probability",
                "ml_ten_day_down_5pct_probability",
                "ml_ten_day_down_8pct_probability",
                "ml_ten_day_expected_net_return",
                "ml_ten_day_reason",
                "ml_ten_day_entry_rule",
            ],
        ].to_dict(orient="records")
        if deployment_approved
        else []
    )
    study = {
        "status": "completed",
        "method": "walk_forward_capitulation_reversal_10d_ml_v3",
        "adopted_shape": TEN_DAY_ADOPTED_SHAPE,
        "adopted_shape_label": TEN_DAY_ADOPTED_SHAPE_LABEL,
        "target": "翌営業日始値から10営業日以内に日中高値+5%",
        "entry": "翌営業日始値（ギャップ上限は注文時に確認）",
        "holding_days": 10,
        "minimum_turnover_yen": int(minimum_turnover),
        "maximum_candidates_per_day": 1,
        "development_start": diagnostics.get("development_start"),
        "development_end": diagnostics.get("development_end"),
        "validation_start": diagnostics.get("validation_start"),
        "validation_end": diagnostics.get("validation_end"),
        "chosen_parameters": parameters,
        "deployment_approved": deployment_approved,
        "deployment_rule": (
            "未使用後半30件以上・10営業日+5%達成率60%以上・平均損益プラス"
        ),
        "development": diagnostics.get("development", {}),
        "validation": validation_summary,
        "validation_folds": diagnostics.get("validation_folds", []),
        "validation_by_shape": diagnostics.get("validation_by_shape", {}),
        "limit_order_study": limit_order_study,
        "portfolio_study": portfolio_study,
        "turnover_sensitivity": {
            "comparison_mode": (
                "frozen_200m_rule_with_threshold_specific_walk_forward_refits"
            ),
            "validation_start": validation_start,
            "validation_end": validation_end,
            "threshold_step_yen": 50_000_000,
            "thresholds": turnover_sensitivity_rows,
            "deployed_threshold_changed": False,
            "note": (
                "投げ売り反転・予測閾値・損失確率条件・1日最大1銘柄を固定し、"
                "最低売買代金だけを5000万円刻みで変更。各流動性母集団で"
                "ウォークフォワードモデルを再学習し、同じ検証期間で比較"
            ),
        },
        "additional_validation": {
            "type": "post_selection_chronological_robustness",
            "shape_locked": True,
            "shape": TEN_DAY_ADOPTED_SHAPE,
            "shape_label": TEN_DAY_ADOPTED_SHAPE_LABEL,
            "fully_untouched": False,
            "forward_confirmation_required_signals": 30,
            "note": (
                "投げ売り反転は前回の検証結果を確認後に固定したため、"
                "この再検証は頑健性確認。完全な未使用検証は今後のシグナルで行う"
            ),
        },
        "feature_lifts": _ten_day_feature_lifts(development_scored),
        "live_signal_count": (
            int(latest_scores["ml_ten_day_signal"].sum())
            if "ml_ten_day_signal" in latest_scores
            else 0
        ),
        "live_candidate": live_candidate_records,
        "leakage_control": (
            "各評価ブロックの10営業日前までに結果が確定したデータだけで再学習。"
            "条件選択は前半、成績評価は未使用の後半で実施"
        ),
        "caveat": (
            "投げ売り反転だけを採用対象に固定した10営業日の新規シグナル。"
            "過去3年の再検証は形状選択後の頑健性確認であり、"
            "完全な未使用検証は今後30件の新規シグナルで継続する。"
            "決算跨ぎと重要指標前の既存制限を適用する"
        ),
    }
    if not validation_trades.empty:
        study["validation"]["median_trade_net_return"] = float(
            validation_trades["rise_trade_net_return"].median()
        )
    return frame, study


def calculate_ten_day_limit_order_study(
    outcome_frame: pd.DataFrame,
    validation_trades: pd.DataFrame,
    config: RisePatternConfig,
    *,
    minimum_turnover: float,
) -> dict[str, Any]:
    """Compare next-day-only limit orders for the frozen validation signals."""
    offsets = tuple(-step / 200.0 for step in range(21)) + tuple(
        step / 200.0 for step in range(1, 7)
    )
    minimum_robust_fills = min(15, int(len(validation_trades)))
    base = {
        "status": "completed",
        "signal_count": int(len(validation_trades)),
        "minimum_turnover_yen": int(minimum_turnover),
        "order_validity": "next_trading_day_only",
        "unfilled_return": 0.0,
        "transaction_cost_bps": config.transaction_cost_bps,
        "target_return": 0.05,
        "holding_days": config.horizon_days,
        "intraday_ambiguity_rule": (
            "when an intraday low first reaches the limit, day-one high is excluded "
            "from target evaluation; gap-at-open fills may use day-one high"
        ),
        "efficiency_metric": (
            "expected_net_return_per_signal = fill_rate * mean_filled_trade_net_return"
        ),
        "minimum_fills_for_recommendation": minimum_robust_fills,
    }
    if validation_trades.empty:
        return {**base, "status": "no_validation_signals", "levels": []}

    keys = validation_trades[["ticker", "date"]].drop_duplicates()
    selected_tickers = keys["ticker"].dropna().unique().tolist()
    prices = outcome_frame.loc[
        outcome_frame["ticker"].isin(selected_tickers),
        ["ticker", "date", "open", "high", "low", "close", "adjusted_close"],
    ].copy()
    prices = prices.sort_values(["ticker", "date"])
    ratio = prices["adjusted_close"] / prices["close"]
    adjusted_open = prices["open"] * ratio
    adjusted_high = prices["high"] * ratio
    adjusted_low = prices["low"] * ratio
    group_key = prices["ticker"]
    future_open = pd.concat(
        [
            adjusted_open.groupby(group_key, sort=False).shift(-offset)
            for offset in range(1, config.horizon_days + 1)
        ],
        axis=1,
    )
    future_high = pd.concat(
        [
            adjusted_high.groupby(group_key, sort=False).shift(-offset)
            for offset in range(1, config.horizon_days + 1)
        ],
        axis=1,
    )
    future_low = pd.concat(
        [
            adjusted_low.groupby(group_key, sort=False).shift(-offset)
            for offset in range(1, config.horizon_days + 1)
        ],
        axis=1,
    )
    future_close = pd.concat(
        [
            prices["adjusted_close"].groupby(group_key, sort=False).shift(-offset)
            for offset in range(1, config.horizon_days + 1)
        ],
        axis=1,
    )
    selected_mask = pd.MultiIndex.from_frame(prices[["ticker", "date"]]).isin(
        pd.MultiIndex.from_frame(keys)
    )
    signal_close = prices.loc[selected_mask, "adjusted_close"]
    next_opens = future_open.loc[selected_mask]
    next_highs = future_high.loc[selected_mask]
    next_lows = future_low.loc[selected_mask]
    next_closes = future_close.loc[selected_mask]
    complete = (
        signal_close.notna()
        & next_opens.notna().all(axis=1)
        & next_highs.notna().all(axis=1)
        & next_lows.notna().all(axis=1)
        & next_closes.notna().all(axis=1)
    )
    signal_count = int(complete.sum())
    levels: list[dict[str, Any]] = []
    for limit_offset in offsets:
        limit_price = signal_close * (1.0 + limit_offset)
        open_fill = complete & next_opens.iloc[:, 0].le(limit_price)
        intraday_fill = (
            complete
            & ~open_fill
            & next_lows.iloc[:, 0].le(limit_price)
        )
        filled = open_fill | intraday_fill
        entry = pd.Series(np.nan, index=signal_close.index, dtype=float)
        entry.loc[open_fill] = next_opens.loc[open_fill].iloc[:, 0]
        entry.loc[intraday_fill] = limit_price.loc[intraday_fill]
        target_price = entry * 1.05
        target_matrix = next_highs.ge(target_price, axis=0)
        target_matrix.loc[intraday_fill, target_matrix.columns[0]] = False
        target_hit = target_matrix.any(axis=1) & filled
        gross_return = pd.Series(np.nan, index=signal_close.index, dtype=float)
        gross_return.loc[target_hit] = 0.05
        unresolved = filled & ~target_hit
        gross_return.loc[unresolved] = (
            next_closes.loc[unresolved].iloc[:, -1] / entry.loc[unresolved] - 1.0
        )
        net_return = gross_return.loc[filled] - config.transaction_cost_bps / 10_000.0
        filled_count = int(filled.sum())
        hits = int(target_hit.sum())
        fill_rate = filled_count / signal_count if signal_count else 0.0
        mean_net_return = float(net_return.mean()) if filled_count else None
        expected_per_signal = (
            float(net_return.sum() / signal_count) if signal_count else None
        )
        future_min_return = next_lows.loc[filled].min(axis=1) / entry.loc[filled] - 1.0
        levels.append(
            {
                "limit_offset_from_previous_close": limit_offset,
                "filled_orders": filled_count,
                "fill_rate": fill_rate,
                "open_fill_rate": (
                    float(open_fill.sum() / signal_count) if signal_count else 0.0
                ),
                "target_hit_rate": hits / filled_count if filled_count else None,
                "mean_filled_trade_net_return": mean_net_return,
                "median_filled_trade_net_return": (
                    float(net_return.median()) if filled_count else None
                ),
                "expected_net_return_per_signal": expected_per_signal,
                "trade_win_rate": (
                    float(net_return.gt(0.0).mean()) if filled_count else None
                ),
                "loss_5pct_rate": (
                    float(future_min_return.le(-0.05).mean()) if filled_count else None
                ),
                "loss_8pct_rate": (
                    float(future_min_return.le(-0.08).mean()) if filled_count else None
                ),
                "worst_trade_net_return": (
                    float(net_return.min()) if filled_count else None
                ),
            }
        )
    eligible_levels = [
        level for level in levels if level["filled_orders"] >= minimum_robust_fills
    ]
    best_raw = max(
        levels,
        key=lambda level: (
            float(level["expected_net_return_per_signal"])
            if level["expected_net_return_per_signal"] is not None
            else float("-inf")
        ),
    )
    best_recommended = (
        max(
            eligible_levels,
            key=lambda level: (
                float(level["expected_net_return_per_signal"])
                if level["expected_net_return_per_signal"] is not None
                else float("-inf")
            ),
        )
        if eligible_levels
        else best_raw
    )
    return {
        **base,
        "signal_count": signal_count,
        "levels": levels,
        "best_raw": best_raw,
        "recommended": best_recommended,
        "caveat": (
            f"Frozen validation signals from the {int(minimum_turnover)} yen "
            "minimum-turnover universe are reused. "
            "This is a post-selection execution study, so forward confirmation is required."
        ),
    }


def calculate_ten_day_portfolio_study(
    outcome_frame: pd.DataFrame,
    candidates: pd.DataFrame,
    study_dates: list[Any],
    config: RisePatternConfig,
    *,
    initial_cash: float,
    minimum_turnover: float,
    limit_offset: float,
    maximum_daily_buys: int,
    maximum_positions: int,
    lot_size: int,
) -> dict[str, Any]:
    """Simulate overlapping positions with cash, lot, and capacity constraints."""
    base = {
        "status": "completed",
        "initial_cash_yen": initial_cash,
        "minimum_turnover_yen": int(minimum_turnover),
        "limit_offset_from_previous_close": limit_offset,
        "maximum_daily_buys": maximum_daily_buys,
        "maximum_positions": maximum_positions,
        "lot_size": lot_size,
        "position_budget": "all available cash",
        "exit_rule": "+5% target or tenth-trading-day close",
        "same_day_sequence": "scheduled exits, then entries; entry-day targets exit after entry",
        "transaction_cost_bps": config.transaction_cost_bps,
    }
    if not study_dates:
        return {**base, "status": "no_study_dates"}

    calendar = sorted(set(study_dates))
    study_start = calendar[0]
    study_end = calendar[-1]
    if candidates.empty:
        return {
            **base,
            "status": "no_candidates",
            "study_start": str(study_start),
            "study_end": str(study_end),
            "trading_days": len(calendar),
            "days_without_positions": len(calendar),
        }

    candidate_keys = candidates[["ticker", "date"]].drop_duplicates()
    tickers = candidate_keys["ticker"].dropna().unique().tolist()
    prices = outcome_frame.loc[
        outcome_frame["ticker"].isin(tickers),
        ["ticker", "date", "open", "high", "low", "close", "adjusted_close"],
    ].copy()
    prices = prices.sort_values(["ticker", "date"])
    ratio = prices["adjusted_close"] / prices["close"]
    prices["_open"] = prices["open"] * ratio
    prices["_high"] = prices["high"] * ratio
    prices["_low"] = prices["low"] * ratio
    price_groups = {
        str(ticker): values.reset_index(drop=True)
        for ticker, values in prices.groupby("ticker", sort=False)
    }
    trade_plans: list[dict[str, Any]] = []
    unfilled_orders = 0
    incomplete_orders = 0
    score_column = "_ml_rank_score"
    for candidate in candidates.to_dict(orient="records"):
        ticker = str(candidate["ticker"])
        ticker_prices = price_groups.get(ticker)
        if ticker_prices is None:
            incomplete_orders += 1
            continue
        matches = ticker_prices.index[ticker_prices["date"].eq(candidate["date"])]
        if len(matches) != 1:
            incomplete_orders += 1
            continue
        signal_position = int(matches[0])
        future = ticker_prices.iloc[
            signal_position + 1 : signal_position + config.horizon_days + 1
        ]
        if len(future) < config.horizon_days:
            incomplete_orders += 1
            continue
        signal_close = float(ticker_prices.at[signal_position, "adjusted_close"])
        limit_price = signal_close * (1.0 + limit_offset)
        first = future.iloc[0]
        if float(first["_open"]) <= limit_price:
            entry_price = float(first["_open"])
            fill_mode = "open"
        elif float(first["_low"]) <= limit_price:
            entry_price = limit_price
            fill_mode = "intraday"
        else:
            unfilled_orders += 1
            continue
        target_price = entry_price * 1.05
        exit_date = future.iloc[-1]["date"]
        exit_price = float(future.iloc[-1]["adjusted_close"])
        target_hit = False
        holding_trading_days = config.horizon_days
        for holding_index, (_, day) in enumerate(future.iterrows()):
            if holding_index == 0 and fill_mode == "intraday":
                continue
            if float(day["_high"]) >= target_price:
                exit_date = day["date"]
                exit_price = target_price
                target_hit = True
                holding_trading_days = holding_index + 1
                break
        trade_plans.append(
            {
                "ticker": ticker,
                "signal_date": candidate["date"],
                "entry_date": first["date"],
                "entry_price": entry_price,
                "entry_fill_mode": fill_mode,
                "exit_date": exit_date,
                "exit_price": exit_price,
                "target_hit": target_hit,
                "holding_trading_days": holding_trading_days,
                "rank_score": float(candidate.get(score_column, 0.0)),
            }
        )

    plans_by_entry: dict[Any, list[dict[str, Any]]] = {}
    for plan in trade_plans:
        plans_by_entry.setdefault(plan["entry_date"], []).append(plan)
    close_lookup = prices.set_index(["ticker", "date"])["adjusted_close"]
    half_cost = config.transaction_cost_bps / 20_000.0
    cash = float(initial_cash)
    positions: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    days_without_positions = 0
    days_with_positions = 0
    maximum_positions_used = 0
    skipped_capacity = 0
    skipped_lot_cost = 0

    def marked_equity(value_date: Any) -> float:
        market_value = 0.0
        for position in positions:
            key = (position["ticker"], value_date)
            close_value = close_lookup.get(key, position["entry_price"])
            market_value += position["shares"] * float(close_value)
        return cash + market_value

    for value_date in calendar:
        exiting = [position for position in positions if position["exit_date"] == value_date]
        for position in exiting:
            proceeds = position["shares"] * position["exit_price"] * (1.0 - half_cost)
            cash += proceeds
            position["net_profit_yen"] = proceeds - position["entry_cost_yen"]
            completed.append(position)
            positions.remove(position)

        daily_buys = 0
        entry_plans = sorted(
            plans_by_entry.get(value_date, []),
            key=lambda plan: plan["rank_score"],
            reverse=True,
        )
        for plan in entry_plans:
            if daily_buys >= maximum_daily_buys or len(positions) >= maximum_positions:
                skipped_capacity += 1
                continue
            if any(position["ticker"] == plan["ticker"] for position in positions):
                skipped_capacity += 1
                continue
            budget = cash
            lot_cost = plan["entry_price"] * lot_size * (1.0 + half_cost)
            lots = int(budget // lot_cost)
            if lots < 1:
                skipped_lot_cost += 1
                continue
            shares = lots * lot_size
            entry_cost = shares * plan["entry_price"] * (1.0 + half_cost)
            cash -= entry_cost
            position = {**plan, "shares": shares, "entry_cost_yen": entry_cost}
            positions.append(position)
            daily_buys += 1
            maximum_positions_used = max(maximum_positions_used, len(positions))
            if plan["exit_date"] == value_date:
                proceeds = shares * plan["exit_price"] * (1.0 - half_cost)
                cash += proceeds
                position["net_profit_yen"] = proceeds - entry_cost
                completed.append(position)
                positions.remove(position)

        equity = marked_equity(value_date)
        equity_curve.append(
            {"date": value_date, "equity": equity, "positions": len(positions)}
        )
        if positions:
            days_with_positions += 1
        else:
            days_without_positions += 1

    curve = pd.DataFrame(equity_curve)
    curve["peak"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["peak"] - 1.0
    ending_equity = float(curve.iloc[-1]["equity"])
    trade_profits = pd.Series(
        [trade["net_profit_yen"] for trade in completed],
        dtype=float,
    )
    first_entry_date = min(
        (trade["entry_date"] for trade in completed),
        default=None,
    )
    last_exit_date = max(
        (trade["exit_date"] for trade in completed),
        default=None,
    )
    active_window = (
        curve.loc[curve["date"].between(first_entry_date, last_exit_date)]
        if first_entry_date is not None and last_exit_date is not None
        else pd.DataFrame()
    )
    trade_records: list[dict[str, Any]] = []
    for sequence, trade in enumerate(
        sorted(completed, key=lambda item: (item["entry_date"], item["ticker"])),
        start=1,
    ):
        gross_return = trade["exit_price"] / trade["entry_price"] - 1.0
        net_return = trade["net_profit_yen"] / trade["entry_cost_yen"]
        if trade["target_hit"]:
            exit_reason = "take_profit_5pct"
            exit_reason_label = "+5%利確"
        elif net_return > 0.0:
            exit_reason = "time_exit_profit"
            exit_reason_label = "10営業日目決済（利益）"
        elif net_return < 0.0:
            exit_reason = "time_exit_loss"
            exit_reason_label = "10営業日目決済（損失）"
        else:
            exit_reason = "time_exit_flat"
            exit_reason_label = "10営業日目決済（同値）"
        trade_records.append(
            {
                "sequence": sequence,
                "ticker": trade["ticker"],
                "signal_date": str(trade["signal_date"]),
                "entry_date": str(trade["entry_date"]),
                "exit_date": str(trade["exit_date"]),
                "holding_trading_days": int(trade["holding_trading_days"]),
                "shares": int(trade["shares"]),
                "entry_price": float(trade["entry_price"]),
                "exit_price": float(trade["exit_price"]),
                "entry_fill_mode": trade["entry_fill_mode"],
                "exit_reason": exit_reason,
                "exit_reason_label": exit_reason_label,
                "gross_return": float(gross_return),
                "net_return": float(net_return),
                "net_profit_yen": float(trade["net_profit_yen"]),
            }
        )
    return {
        **base,
        "study_start": str(study_start),
        "study_end": str(study_end),
        "trading_days": len(calendar),
        "candidate_signals": int(len(candidates)),
        "fillable_orders": len(trade_plans),
        "unfilled_orders": unfilled_orders,
        "incomplete_orders": incomplete_orders,
        "completed_trades": len(completed),
        "target_hit_rate": (
            float(np.mean([trade["target_hit"] for trade in completed]))
            if completed
            else None
        ),
        "trade_win_rate": (
            float(trade_profits.gt(0.0).mean()) if not trade_profits.empty else None
        ),
        "ending_equity_yen": ending_equity,
        "total_return": ending_equity / initial_cash - 1.0,
        "annualized_return": (ending_equity / initial_cash) ** (252 / len(calendar)) - 1.0,
        "maximum_drawdown": float(curve["drawdown"].min()),
        "first_entry_date": str(first_entry_date) if first_entry_date else None,
        "last_exit_date": str(last_exit_date) if last_exit_date else None,
        "days_without_positions": days_without_positions,
        "days_without_positions_during_active_window": (
            int(active_window["positions"].eq(0).sum())
            if not active_window.empty
            else len(calendar)
        ),
        "active_window_trading_days": int(len(active_window)),
        "days_with_positions": days_with_positions,
        "position_day_rate": days_with_positions / len(calendar),
        "maximum_positions_used": maximum_positions_used,
        "skipped_for_capacity": skipped_capacity,
        "skipped_for_lot_cost": skipped_lot_cost,
        "open_positions_at_end": len(positions),
        "trades": trade_records,
        "caveat": (
            "Walk-forward model scores avoid future-outcome leakage, but the frozen signal "
            "parameters were selected using part of this same three-year history."
        ),
    }


def _ten_day_feature_lifts(scored: pd.DataFrame) -> list[dict[str, Any]]:
    """Rank technical feature quartiles by development-period +5% lift."""
    if scored.empty:
        return []
    target = scored["rise_trade_target_hit"].fillna(False).astype(bool)
    baseline = float(target.mean())
    results: list[dict[str, Any]] = []
    for feature in ML_FEATURES:
        if feature not in scored:
            continue
        values = pd.to_numeric(scored[feature], errors="coerce")
        valid = values.notna() & target.notna()
        if int(valid.sum()) < 200:
            continue
        valid_values = values.loc[valid]
        lower = float(valid_values.quantile(0.25))
        upper = float(valid_values.quantile(0.75))
        sides = (
            ("low", valid & values.le(lower), lower),
            ("high", valid & values.ge(upper), upper),
        )
        side_results: list[tuple[str, pd.Series, float, float]] = []
        for direction, mask, threshold in sides:
            samples = int(mask.sum())
            if samples < 50:
                continue
            rate = float(target.loc[mask].mean())
            side_results.append((direction, mask, threshold, rate))
        if not side_results:
            continue
        direction, mask, threshold, rate = max(side_results, key=lambda item: item[3])
        results.append(
            {
                "feature": feature,
                "direction": direction,
                "threshold": threshold,
                "samples": int(mask.sum()),
                "target_hit_rate": rate,
                "baseline_hit_rate": baseline,
                "lift": rate - baseline,
            }
        )
    return sorted(results, key=lambda item: item["lift"], reverse=True)[:12]


def backtest_rise_pattern_signals(
    features: pd.DataFrame,
    bottom_events: pd.DataFrame,
    *,
    config: RisePatternConfig | None = None,
) -> dict[str, Any]:
    """Walk forward the pattern detector and compare it with the prior inflow signal."""
    config = config or RisePatternConfig()
    if features.empty or bottom_events.empty:
        return {"method": "walk_forward_rise_pattern_v1", "strategies": {}}

    frame = _attach_trade_outcomes(_add_live_bottom_features(features), config)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    all_dates = sorted(frame["date"].drop_duplicates())
    date_position = {value: position for position, value in enumerate(all_dates)}
    complete_dates = [
        value
        for value in all_dates
        if frame.loc[frame["date"] == value, "trade_outcome_available"].any()
    ]
    test_dates = complete_dates[-config.test_days :]
    if not test_dates:
        return {"method": "walk_forward_rise_pattern_v1", "strategies": {}}

    events = bottom_events.copy()
    events["date"] = pd.to_datetime(events["date"]).dt.date
    events["_date_position"] = events["date"].map(date_position)
    events = events.loc[events["_date_position"].notna()].copy()

    strategy_rows: dict[str, list[pd.DataFrame]] = {
        "observed_inflow": [],
        "rise_pattern": [],
        "rise_pattern_reversal": [],
        "rise_pattern_with_inflow": [],
        "strong_shape_200m": [],
        "strong_shape_200m_signal": [],
        "combined": [],
    }
    for test_date in test_dates:
        position = date_position[test_date]
        training = events.loc[events["_date_position"] <= position - config.horizon_days]
        profiles = _calibrate_profiles(training, config)
        day = frame.loc[frame["date"] == test_date].copy()
        day = _score_day_patterns(day, profiles, config)
        eligible = day.loc[
            day["trade_outcome_available"].fillna(False)
            & (day["turnover_value"].fillna(0) >= 10_000_000)
            & day["rsi_14"].fillna(100).le(82.0)
        ].copy()
        if eligible.empty:
            continue

        old_mask = (
            eligible["observed_inflow_confirmed"].fillna(False).astype(bool)
            & eligible["return_1d"].between(0.002, 0.10)
            & eligible["return_5d"].between(-0.05, 0.18)
        )
        pattern_mask = eligible["rise_pattern_signal"].fillna(False).astype(bool)
        reversal_mask = pattern_mask & eligible["rise_pattern_reversal_confirmed"].fillna(
            False
        ).astype(bool)
        pattern_with_inflow_mask = pattern_mask & eligible["observed_inflow_confirmed"].fillna(
            False
        ).astype(bool)
        strong_shape_mask = (
            eligible["rise_pattern_live_bottom"].fillna(False).astype(bool)
            & eligible["_rise_shape"].isin(STRONG_SHAPES)
            & eligible["turnover_value"].fillna(0).ge(STRONG_SHAPE_MIN_TURNOVER)
        )
        strong_shape_signal_mask = strong_shape_mask & pattern_mask
        old = _select_top(eligible.loc[old_mask], "observed_inflow_score", config.top_n)
        pattern = _select_top(eligible.loc[pattern_mask], "rise_pattern_probability", config.top_n)
        reversal = _select_top(
            eligible.loc[reversal_mask], "rise_pattern_probability", config.top_n
        )
        pattern_with_inflow = _select_top(
            eligible.loc[pattern_with_inflow_mask],
            "rise_pattern_probability",
            config.top_n,
        )
        strong_shape = eligible.loc[strong_shape_mask].copy()
        strong_shape["_strong_shape_score"] = strong_shape["rise_pattern_probability"].where(
            strong_shape["rise_pattern_probability"].gt(0),
            strong_shape["observed_inflow_score"].fillna(0),
        )
        strong_shape = _select_top(strong_shape, "_strong_shape_score", config.top_n)
        strong_shape_signal = _select_top(
            eligible.loc[strong_shape_signal_mask],
            "rise_pattern_probability",
            config.top_n,
        )
        combined = eligible.loc[old_mask | pattern_mask].copy()
        combined["_combined_score"] = combined[
            ["observed_inflow_score", "rise_pattern_probability"]
        ].max(axis=1)
        combined = _select_top(combined, "_combined_score", config.top_n)
        strategy_rows["observed_inflow"].append(old)
        strategy_rows["rise_pattern"].append(pattern)
        strategy_rows["rise_pattern_reversal"].append(reversal)
        strategy_rows["rise_pattern_with_inflow"].append(pattern_with_inflow)
        strategy_rows["strong_shape_200m"].append(strong_shape)
        strategy_rows["strong_shape_200m_signal"].append(strong_shape_signal)
        strategy_rows["combined"].append(combined)

    selective_dates = complete_dates[-config.selective_test_days :]
    selective_pool = frame.loc[_selective_candidate_mask(frame)].copy()
    selective_pool["_date_position"] = selective_pool["date"].map(date_position)
    selective_scored_rows: list[pd.DataFrame] = []
    for test_date in selective_dates:
        position = date_position[test_date]
        training = selective_pool.loc[
            selective_pool["_date_position"] <= position - config.horizon_days
        ]
        day = selective_pool.loc[
            (selective_pool["date"] == test_date)
            & selective_pool["trade_outcome_available"].fillna(False).astype(bool)
        ]
        if day.empty:
            continue
        selective_scored_rows.append(_score_selective_day(day, training, config))
    selective_scored = (
        pd.concat(selective_scored_rows, ignore_index=True)
        if selective_scored_rows
        else pd.DataFrame()
    )
    selective_trades = _select_selective_trades(
        selective_scored,
        config,
        probability_threshold=config.selective_probability_threshold,
        top_n=config.selective_top_n,
    )

    summaries: dict[str, Any] = {}
    strategy_trades: dict[str, pd.DataFrame] = {}
    for name, rows in strategy_rows.items():
        trades = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        strategy_trades[name] = trades
        summaries[name] = _strategy_summary(trades, test_dates, config)
    strategy_trades["selective_70"] = selective_trades
    summaries["selective_70"] = _strategy_summary(
        selective_trades,
        selective_dates,
        config,
    )

    selective_threshold_grid: dict[str, Any] = {}
    for probability_threshold in config.selective_probability_grid:
        for top_n in range(1, config.selective_top_n + 1):
            key = f"p{int(round(probability_threshold * 100))}_top{top_n}"
            grid_trades = _select_selective_trades(
                selective_scored,
                config,
                probability_threshold=probability_threshold,
                top_n=top_n,
            )
            summary = _strategy_summary(grid_trades, selective_dates, config)
            summary["probability_threshold"] = probability_threshold
            summary["top_n_per_day"] = top_n
            selective_threshold_grid[key] = summary

    selective_by_shape = (
        {
            shape: _strategy_summary(
                selective_trades.loc[selective_trades["_rise_shape"] == shape],
                selective_dates,
                config,
            )
            for shape in sorted(STRONG_SHAPES)
        }
        if not selective_trades.empty
        else {
            shape: _strategy_summary(pd.DataFrame(), selective_dates, config)
            for shape in sorted(STRONG_SHAPES)
        }
    )

    ml_dates = complete_dates[-config.ml_test_days :]
    ml_scored = walk_forward_ml_scores(
        selective_pool,
        ml_dates,
        date_position,
        horizon_days=config.horizon_days,
        refit_days=config.ml_refit_days,
        minimum_shape_samples=config.ml_minimum_shape_samples,
    )
    ml_validation_trades, ml_diagnostics = tune_and_select_ml_strategy(
        ml_scored,
        ml_dates,
        minimum_development_signals=config.ml_minimum_development_signals,
    )
    ml_validation_dates = ml_dates[len(ml_dates) // 2 :]
    strategy_trades["ml_selective_60"] = ml_validation_trades
    summaries["ml_selective_60"] = _strategy_summary(
        ml_validation_trades,
        ml_validation_dates,
        config,
    )

    pattern_trades = strategy_trades["rise_pattern"]
    risk_management = {
        "assumptions": {
            "take_profit_pct": 0.05,
            "holding_days": config.horizon_days,
            "transaction_cost_bps": config.transaction_cost_bps,
            "same_day_stop_and_target": "stop_first",
            "gap_through_stop": "exit_at_open",
            "atr_period": 14,
        },
        "no_stop": _risk_summary(
            pattern_trades,
            test_dates,
            config,
            return_column="rise_trade_net_return",
            target_column="rise_trade_target_hit",
        ),
    }
    for stop_pct in config.fixed_stop_pcts:
        key = _fixed_stop_key(stop_pct)
        prefix = f"rise_trade_{key}"
        risk_management[key] = _risk_summary(
            pattern_trades,
            test_dates,
            config,
            return_column=f"{prefix}_net_return",
            target_column=f"{prefix}_target_hit",
            stop_column=f"{prefix}_stop_hit",
            ambiguous_column=f"{prefix}_ambiguous_both_hit",
            stop_distance_column=f"{prefix}_stop_pct",
        )
    for multiplier in config.atr_stop_multipliers:
        key = _atr_stop_key(multiplier)
        prefix = f"rise_trade_{key}"
        risk_management[key] = _risk_summary(
            pattern_trades,
            test_dates,
            config,
            return_column=f"{prefix}_net_return",
            target_column=f"{prefix}_target_hit",
            stop_column=f"{prefix}_stop_hit",
            ambiguous_column=f"{prefix}_ambiguous_both_hit",
            stop_distance_column=f"{prefix}_stop_pct",
        )

    strong_shape_trades = strategy_trades["strong_shape_200m"]
    strong_shape_risk_management = {
        "assumptions": risk_management["assumptions"],
        "no_stop": _risk_summary(
            strong_shape_trades,
            test_dates,
            config,
            return_column="rise_trade_net_return",
            target_column="rise_trade_target_hit",
        ),
    }
    for stop_pct in config.fixed_stop_pcts:
        key = _fixed_stop_key(stop_pct)
        prefix = f"rise_trade_{key}"
        strong_shape_risk_management[key] = _risk_summary(
            strong_shape_trades,
            test_dates,
            config,
            return_column=f"{prefix}_net_return",
            target_column=f"{prefix}_target_hit",
            stop_column=f"{prefix}_stop_hit",
            ambiguous_column=f"{prefix}_ambiguous_both_hit",
            stop_distance_column=f"{prefix}_stop_pct",
        )

    strong_shape_exit_management = {
        "assumptions": {
            "entry": "next_trading_day_open",
            "transaction_cost_bps": config.transaction_cost_bps,
            "intraday_target_and_close_exit": "target_first",
            "baseline_target_pct": 0.05,
            "baseline_holding_days": config.horizon_days,
        },
        "baseline_5d_target_5pct": _exit_summary(
            strong_shape_trades,
            test_dates,
            config,
            return_column="rise_trade_net_return",
            target_column="rise_trade_target_hit",
        ),
    }
    for target_pct in config.exit_target_pcts:
        key = f"target_{int(round(target_pct * 100))}pct"
        prefix = f"rise_trade_exit_{key}"
        summary = _exit_summary(
            strong_shape_trades,
            test_dates,
            config,
            return_column=f"{prefix}_net_return",
            target_column=f"{prefix}_target_hit",
            early_exit_column=f"{prefix}_early_exit",
        )
        summary["target_pct"] = target_pct
        summary["holding_days"] = config.horizon_days
        strong_shape_exit_management[key] = summary
    for holding_days in config.exit_holding_days:
        key = f"time_{holding_days}d"
        prefix = f"rise_trade_exit_{key}"
        summary = _exit_summary(
            strong_shape_trades,
            test_dates,
            config,
            return_column=f"{prefix}_net_return",
            target_column=f"{prefix}_target_hit",
            early_exit_column=f"{prefix}_early_exit",
        )
        summary["target_pct"] = 0.05
        summary["holding_days"] = holding_days
        strong_shape_exit_management[key] = summary
    for check_day, minimum_return in config.follow_through_checks:
        threshold = int(round(minimum_return * 100))
        key = f"day{check_day}_min_{threshold}pct"
        prefix = f"rise_trade_exit_{key}"
        summary = _exit_summary(
            strong_shape_trades,
            test_dates,
            config,
            return_column=f"{prefix}_net_return",
            target_column=f"{prefix}_target_hit",
            early_exit_column=f"{prefix}_early_exit",
        )
        summary["target_pct"] = 0.05
        summary["holding_days"] = config.horizon_days
        summary["follow_through_check_day"] = check_day
        summary["minimum_close_return"] = minimum_return
        strong_shape_exit_management[key] = summary
    for multiplier in config.atr_stop_multipliers:
        key = _atr_stop_key(multiplier)
        prefix = f"rise_trade_{key}"
        strong_shape_risk_management[key] = _risk_summary(
            strong_shape_trades,
            test_dates,
            config,
            return_column=f"{prefix}_net_return",
            target_column=f"{prefix}_target_hit",
            stop_column=f"{prefix}_stop_hit",
            ambiguous_column=f"{prefix}_ambiguous_both_hit",
            stop_distance_column=f"{prefix}_stop_pct",
        )

    return {
        "method": "walk_forward_rise_pattern_v1",
        "signal_timing": "当日引けで検知し翌営業日始値でエントリー",
        "leakage_control": (
            "各評価日の5営業日前までに結果確定した底値イベントだけで特徴差と閾値を再学習"
        ),
        "config": asdict(config),
        "test_start": test_dates[0].isoformat(),
        "test_end": test_dates[-1].isoformat(),
        "test_days": len(test_dates),
        "strategies": summaries,
        "strong_shape_200m_definition": {
            "shapes": sorted(STRONG_SHAPES),
            "min_turnover": int(STRONG_SHAPE_MIN_TURNOVER),
            "entry": "next_trading_day_open",
            "top_n_per_day": config.top_n,
            "raw_ranking": "walk-forward subtype probability, fallback observed inflow score",
            "signal_overlay": "existing walk-forward probability >= 80% and samples >= 30",
        },
        "selective_70_definition": {
            "shapes": sorted(STRONG_SHAPES),
            "min_turnover": int(STRONG_SHAPE_MIN_TURNOVER),
            "entry": "next_trading_day_open_if_gap_up_is_within_limit",
            "max_gap_up": config.selective_max_gap_up,
            "top_n_per_day": config.selective_top_n,
            "allows_zero_trades": True,
            "probability_threshold": config.selective_probability_threshold,
            "min_expected_net_return": config.selective_min_expected_net_return,
            "max_down_5pct_probability": (config.selective_max_down_5pct_probability),
            "max_down_8pct_probability": (config.selective_max_down_8pct_probability),
            "training_population": (
                "prior live-detectable strong-shape signals with completed "
                "next-open five-day outcomes"
            ),
            "shape_model": "separate empirical Bayes model for each strong shape",
            "features": {
                "market_regime": "market breadth 5d > 50% and median return 20d > 0",
                "theme_flow_proxy": ("TSE-33/TSE-17 sector trend score >= 60% and breadth >= 50%"),
                "volume_continuation": ("5d/20d volume and turnover ratios are both >= 1"),
                "candlestick_reversal": "close location >= 55% of daily range",
                "observed_inflow": "confirmed or observed inflow score >= 50%",
            },
            "note": (
                "The repository has no historical Kabutan-theme membership series, "
                "so leakage-safe sector flow is used as the theme-flow proxy."
            ),
        },
        "selective_70_validation": {
            "test_start": selective_dates[0].isoformat(),
            "test_end": selective_dates[-1].isoformat(),
            "test_days": len(selective_dates),
            "minimum_required_signals": config.selective_validation_samples,
            "sample_requirement_met": (
                len(selective_trades) >= config.selective_validation_samples
            ),
            "target_rate_requirement": 0.70,
            "target_rate_requirement_met": (
                not selective_trades.empty
                and float(selective_trades["rise_trade_target_hit"].mean()) >= 0.70
            ),
            "positive_expectancy_requirement_met": (
                not selective_trades.empty
                and float(selective_trades["rise_trade_net_return"].mean()) > 0.0
            ),
            "by_shape": selective_by_shape,
            "threshold_grid": selective_threshold_grid,
        },
        "ml_selective_60_study": {
            "goal": {
                "validation_target_hit_rate": 0.60,
                "validation_minimum_signals": 30,
                "combined_minimum_signals": 100,
                "mean_trade_net_return_must_be_positive": True,
            },
            "leakage_control": (
                "shape-specific models are refit in chronological blocks using only "
                "outcomes completed before each block; configuration is tuned on the "
                "first half and frozen for the final half"
            ),
            "models": ["logistic", "hist_gradient_boosting"],
            "features": (
                "price, volume, turnover, volatility, RSI, ATR, candle close location, "
                "market breadth, TSE-17/TSE-33 flow proxies, observed inflow, retail flow, "
                "Sakata score, and next-open gap known at order time"
            ),
            **ml_diagnostics,
        },
        "rise_pattern_risk_management": risk_management,
        "strong_shape_200m_risk_management": strong_shape_risk_management,
        "strong_shape_200m_exit_management": strong_shape_exit_management,
    }


def _add_live_bottom_features(
    features: pd.DataFrame,
    *,
    include_signature: bool = True,
) -> pd.DataFrame:
    frame = features.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    ratio = frame["adjusted_close"] / frame["close"]
    frame["_rise_adjusted_low"] = frame["low"] * ratio
    frame["_rise_adjusted_high"] = frame["high"] * ratio
    low_group = frame.groupby("ticker", sort=False)["_rise_adjusted_low"]
    high_group = frame.groupby("ticker", sort=False)["_rise_adjusted_high"]
    prior_low_1 = low_group.shift(1)
    prior_low_2 = low_group.shift(2)
    prior_high_20 = high_group.transform(
        lambda values: values.shift(1).rolling(20, min_periods=20).max()
    )
    prior_bottom = low_group.transform(
        lambda values: values.shift(4).rolling(17, min_periods=17).min()
    )
    current_low = frame["_rise_adjusted_low"]
    frame["rise_pattern_live_bottom"] = current_low.lt(
        pd.concat([prior_low_1, prior_low_2], axis=1).min(axis=1)
    ) & current_low.div(prior_high_20).sub(1).le(-0.04)

    double_bottom = prior_bottom.gt(0) & current_low.div(prior_bottom).sub(1).abs().le(0.03)
    candle_span = frame["high"] - frame["low"]
    close_location = (frame["close"] - frame["low"]).div(candle_span.where(candle_span > 0))
    frame["rise_pattern_reversal_confirmed"] = close_location.fillna(0.5).ge(0.55)
    capitulation = (
        frame["return_20d"].le(-0.10)
        & frame["volume_ratio_5_20"].ge(1.20)
        & close_location.fillna(0.5).ge(0.55)
    )
    compression = frame["range_width_10d"].le(0.07) & frame["volatility_10d"].le(0.02)
    sharp = frame["return_5d"].le(-0.06)
    rounded = frame["return_20d"].le(-0.05) & frame["return_5d"].ge(-0.03)
    frame["_rise_shape"] = np.select(
        [double_bottom, capitulation, compression, sharp, rounded],
        [
            "double_bottom",
            "capitulation_reversal",
            "compression_base",
            "sharp_selloff",
            "rounded_base",
        ],
        default="other_swing_low",
    )

    date_group = frame.groupby("date", sort=False)
    market_breadth = date_group["return_5d"].transform(lambda values: float((values > 0).mean()))
    market_median_return = date_group["return_20d"].transform("median")
    frame["_rise_close_location"] = close_location.fillna(0.5)
    frame["_rise_market_breadth_5d"] = market_breadth
    frame["_rise_market_median_return_20d"] = market_median_return
    frame["_rise_market_favorable"] = market_breadth.gt(0.50) & market_median_return.gt(0.0)

    sector_score = _first_numeric_column(
        frame,
        ("sector_33_trend_score", "sector_17_trend_score"),
    )
    sector_breadth = _first_numeric_column(
        frame,
        ("sector_33_breadth_5d", "sector_17_breadth_5d"),
    )
    frame["_rise_theme_flow"] = sector_score.ge(0.60) & sector_breadth.ge(0.50)
    volume_ratio = _numeric_column(frame, "volume_ratio_5_20")
    turnover_ratio = _numeric_column(frame, "turnover_ratio_5_20")
    frame["_rise_volume_continuation"] = volume_ratio.ge(1.0) & turnover_ratio.ge(1.0)
    observed_score = _numeric_column(frame, "observed_inflow_score", default=0.0)
    observed_confirmed = (
        frame.get(
            "observed_inflow_confirmed",
            pd.Series(False, index=frame.index),
        )
        .fillna(False)
        .astype(bool)
    )
    frame["_rise_observed_inflow"] = observed_confirmed | observed_score.ge(0.50)
    frame["_rise_quality_score"] = sum(
        frame[column].fillna(False).astype("int8") for column in SELECTIVE_SHAPE_FEATURES
    )
    if include_signature:
        frame["_rise_feature_signature"] = frame[list(SELECTIVE_SHAPE_FEATURES)].apply(
            lambda row: "".join("1" if bool(value) else "0" for value in row),
            axis=1,
        )
    return frame


def _numeric_column(
    frame: pd.DataFrame,
    column: str,
    *,
    default: float = np.nan,
) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _first_numeric_column(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for column in columns:
        values = _numeric_column(frame, column)
        result = result.fillna(values)
    return result


def _calibrate_profiles(
    events: pd.DataFrame,
    config: RisePatternConfig,
) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    if events.empty:
        return profiles
    for shape, values in events.groupby("shape", sort=False):
        if len(values) < config.min_samples or values["target_5pct"].nunique() < 2:
            continue
        comparisons = _feature_comparison(values)
        if len(comparisons) < 2:
            continue
        selected = [comparisons[0]["feature"], comparisons[1]["feature"]]
        work = values.dropna(subset=selected).copy()
        if len(work) < config.min_samples:
            continue
        thresholds = {feature: float(work[feature].median()) for feature in selected}
        parent_rate = float(work["target_5pct"].mean())
        work["_side_1"] = np.where(work[selected[0]] >= thresholds[selected[0]], "high", "low")
        work["_side_2"] = np.where(work[selected[1]] >= thresholds[selected[1]], "high", "low")
        subtypes: dict[tuple[str, str], dict[str, Any]] = {}
        for sides, group in work.groupby(["_side_1", "_side_2"], sort=False):
            samples = int(len(group))
            successes = int(group["target_5pct"].sum())
            rate = successes / samples
            smoothed = (successes + parent_rate * config.prior_strength) / (
                samples + config.prior_strength
            )
            subtypes[(str(sides[0]), str(sides[1]))] = {
                "samples": samples,
                "successes": successes,
                "success_rate": rate,
                "smoothed_success_rate": float(smoothed),
                "sides": [str(sides[0]), str(sides[1])],
            }
        profiles[str(shape)] = {
            "features": selected,
            "thresholds": thresholds,
            "subtypes": subtypes,
        }
    return profiles


def _feature_comparison(values: pd.DataFrame) -> list[dict[str, Any]]:
    hits = values.loc[values["target_5pct"]]
    misses = values.loc[~values["target_5pct"]]
    comparisons: list[dict[str, Any]] = []
    for feature in FEATURE_SPECS:
        all_values = pd.to_numeric(values[feature], errors="coerce").dropna()
        hit_values = pd.to_numeric(hits[feature], errors="coerce").dropna()
        miss_values = pd.to_numeric(misses[feature], errors="coerce").dropna()
        if all_values.empty or hit_values.empty or miss_values.empty:
            continue
        q25, q75 = all_values.quantile([0.25, 0.75])
        iqr = float(q75 - q25)
        gap = float(hit_values.median() - miss_values.median())
        comparisons.append({"feature": feature, "effect": gap / iqr if iqr > 0 else 0.0})
    return sorted(comparisons, key=lambda item: -abs(item["effect"]))


def _score_day_patterns(
    day: pd.DataFrame,
    profiles: dict[str, dict[str, Any]],
    config: RisePatternConfig,
) -> pd.DataFrame:
    result = day.copy()
    result["rise_pattern_probability"] = 0.0
    result["rise_pattern_samples"] = 0
    result["rise_pattern_signal"] = False
    for index in result.index[result["rise_pattern_live_bottom"].fillna(False)]:
        row = result.loc[index]
        profile = profiles.get(str(row["_rise_shape"]))
        if profile is None:
            continue
        subtype = _match_subtype(row, profile)
        if subtype is None:
            continue
        probability = float(subtype["smoothed_success_rate"])
        samples = int(subtype["samples"])
        result.at[index, "rise_pattern_probability"] = probability
        result.at[index, "rise_pattern_samples"] = samples
        result.at[index, "rise_pattern_signal"] = (
            probability >= config.min_signal_rate and samples >= config.min_samples
        )
    return result


def _match_subtype(
    row: pd.Series,
    profile: dict[str, Any],
) -> dict[str, Any] | None:
    features = profile["features"]
    if any(pd.isna(row.get(feature)) for feature in features):
        return None
    sides = tuple(
        "high" if float(row[feature]) >= profile["thresholds"][feature] else "low"
        for feature in features
    )
    return profile["subtypes"].get(sides)


def _signal_reason(shape: str, subtype: dict[str, Any], probability: float) -> str:
    return (
        f"{SHAPE_LABELS.get(shape, shape)}・過去類似{subtype['samples']}件・"
        f"補正+5%率{probability:.0%}"
    )


def _attach_trade_outcomes(
    features: pd.DataFrame,
    config: RisePatternConfig,
) -> pd.DataFrame:
    frame = features.copy()
    ratio = frame["adjusted_close"] / frame["close"]
    adjusted_open = frame["open"] * ratio
    adjusted_high = frame["high"] * ratio
    adjusted_low = frame["low"] * ratio
    group_key = frame["ticker"]
    entry = adjusted_open.groupby(group_key, sort=False).shift(-1)
    future_opens = [
        adjusted_open.groupby(group_key, sort=False).shift(-offset)
        for offset in range(1, config.horizon_days + 1)
    ]
    future_highs = [
        adjusted_high.groupby(group_key, sort=False).shift(-offset)
        for offset in range(1, config.horizon_days + 1)
    ]
    future_lows = [
        adjusted_low.groupby(group_key, sort=False).shift(-offset)
        for offset in range(1, config.horizon_days + 1)
    ]
    future_closes = [
        frame["adjusted_close"].groupby(group_key, sort=False).shift(-offset)
        for offset in range(1, config.horizon_days + 1)
    ]
    future_open = pd.concat(future_opens, axis=1)
    future_high = pd.concat(future_highs, axis=1)
    future_low = pd.concat(future_lows, axis=1)
    future_close = pd.concat(future_closes, axis=1)
    exit_close = frame["adjusted_close"].groupby(group_key, sort=False).shift(-config.horizon_days)
    complete = (
        entry.notna()
        & exit_close.notna()
        & future_open.notna().all(axis=1)
        & future_high.notna().all(axis=1)
        & future_low.notna().all(axis=1)
        & future_close.notna().all(axis=1)
    )
    target = entry * 1.05
    hit = future_high.ge(target, axis=0).any(axis=1) & complete
    frame["trade_outcome_available"] = complete
    frame["rise_trade_entry_gap_return"] = entry / frame["adjusted_close"] - 1
    frame["rise_trade_target_hit"] = hit
    frame["rise_trade_future_max_return"] = future_high.max(axis=1) / entry - 1
    frame["rise_trade_future_min_return"] = future_low.min(axis=1) / entry - 1
    frame["rise_trade_down_5pct"] = frame["rise_trade_future_min_return"].le(-0.05) & complete
    frame["rise_trade_down_8pct"] = frame["rise_trade_future_min_return"].le(-0.08) & complete
    frame["rise_trade_gross_return"] = np.where(
        hit,
        0.05,
        exit_close / entry - 1,
    )
    frame["rise_trade_net_return"] = (
        frame["rise_trade_gross_return"] - config.transaction_cost_bps / 10_000.0
    )

    for stop_pct in config.fixed_stop_pcts:
        key = _fixed_stop_key(stop_pct)
        prefix = f"rise_trade_{key}"
        fixed_stop_return = pd.Series(-stop_pct, index=frame.index)
        fixed = _stopped_trade_outcome(
            entry,
            future_open,
            future_high,
            future_low,
            exit_close,
            complete,
            fixed_stop_return,
        )
        frame[f"{prefix}_stop_pct"] = stop_pct
        _assign_stop_outcome(frame, prefix, fixed, config)

    for multiplier in config.atr_stop_multipliers:
        key = _atr_stop_key(multiplier)
        prefix = f"rise_trade_{key}"
        atr_stop_return = -(multiplier * frame["atr_14"] / entry)
        atr = _stopped_trade_outcome(
            entry,
            future_open,
            future_high,
            future_low,
            exit_close,
            complete & atr_stop_return.notna(),
            atr_stop_return,
        )
        frame[f"{prefix}_stop_pct"] = -atr_stop_return
        _assign_stop_outcome(frame, prefix, atr, config)

    for target_pct in config.exit_target_pcts:
        key = f"target_{int(round(target_pct * 100))}pct"
        prefix = f"rise_trade_exit_{key}"
        outcome = _exit_trade_outcome(
            entry,
            future_high,
            future_close,
            complete,
            target_return=target_pct,
            exit_day=config.horizon_days,
        )
        _assign_exit_outcome(frame, prefix, outcome, config)

    for holding_days in config.exit_holding_days:
        if holding_days >= config.horizon_days:
            continue
        key = f"time_{holding_days}d"
        prefix = f"rise_trade_exit_{key}"
        outcome = _exit_trade_outcome(
            entry,
            future_high,
            future_close,
            complete,
            target_return=0.05,
            exit_day=holding_days,
        )
        _assign_exit_outcome(frame, prefix, outcome, config)

    for check_day, minimum_return in config.follow_through_checks:
        if check_day > config.horizon_days:
            continue
        threshold = int(round(minimum_return * 100))
        key = f"day{check_day}_min_{threshold}pct"
        prefix = f"rise_trade_exit_{key}"
        outcome = _exit_trade_outcome(
            entry,
            future_high,
            future_close,
            complete,
            target_return=0.05,
            exit_day=config.horizon_days,
            follow_through_day=check_day,
            minimum_close_return=minimum_return,
        )
        _assign_exit_outcome(frame, prefix, outcome, config)
    return frame


def _attach_ml_trade_outcomes(
    features: pd.DataFrame,
    config: RisePatternConfig,
) -> pd.DataFrame:
    """Attach only the outcomes required by the selective ML study."""
    frame = features.copy()
    ratio = frame["adjusted_close"] / frame["close"]
    adjusted_open = frame["open"] * ratio
    adjusted_high = frame["high"] * ratio
    adjusted_low = frame["low"] * ratio
    group_key = frame["ticker"]
    entry = adjusted_open.groupby(group_key, sort=False).shift(-1)
    future_high = pd.concat(
        [
            adjusted_high.groupby(group_key, sort=False).shift(-offset)
            for offset in range(1, config.horizon_days + 1)
        ],
        axis=1,
    )
    future_low = pd.concat(
        [
            adjusted_low.groupby(group_key, sort=False).shift(-offset)
            for offset in range(1, config.horizon_days + 1)
        ],
        axis=1,
    )
    exit_close = frame["adjusted_close"].groupby(group_key, sort=False).shift(
        -config.horizon_days
    )
    complete = (
        entry.notna()
        & exit_close.notna()
        & future_high.notna().all(axis=1)
        & future_low.notna().all(axis=1)
    )
    hit = future_high.ge(entry * 1.05, axis=0).any(axis=1) & complete
    frame["trade_outcome_available"] = complete
    frame["rise_trade_entry_gap_return"] = entry / frame["adjusted_close"] - 1
    frame["rise_trade_target_hit"] = hit
    frame["rise_trade_future_max_return"] = future_high.max(axis=1) / entry - 1
    frame["rise_trade_future_min_return"] = future_low.min(axis=1) / entry - 1
    frame["rise_trade_down_5pct"] = frame["rise_trade_future_min_return"].le(-0.05) & complete
    frame["rise_trade_down_8pct"] = frame["rise_trade_future_min_return"].le(-0.08) & complete
    frame["rise_trade_gross_return"] = np.where(hit, 0.05, exit_close / entry - 1)
    frame["rise_trade_net_return"] = (
        frame["rise_trade_gross_return"] - config.transaction_cost_bps / 10_000.0
    )
    return frame


def _exit_trade_outcome(
    entry: pd.Series,
    future_high: pd.DataFrame,
    future_close: pd.DataFrame,
    complete: pd.Series,
    *,
    target_return: float,
    exit_day: int,
    follow_through_day: int | None = None,
    minimum_close_return: float | None = None,
) -> dict[str, pd.Series]:
    """Resolve target, time exit, and close-confirmed follow-through exits."""
    valid = complete & entry.notna()
    gross_return = pd.Series(np.nan, index=entry.index, dtype=float)
    target_hit = pd.Series(False, index=entry.index, dtype=bool)
    early_exit = pd.Series(False, index=entry.index, dtype=bool)
    unresolved = valid.copy()
    target_price = entry * (1.0 + target_return)

    for day in range(exit_day):
        day_high = future_high.iloc[:, day]
        day_close = future_close.iloc[:, day]
        target = unresolved & day_high.ge(target_price)
        gross_return.loc[target] = target_return
        target_hit.loc[target] = True
        unresolved &= ~target

        day_number = day + 1
        if follow_through_day == day_number and minimum_close_return is not None:
            close_return = day_close / entry - 1
            weak = unresolved & close_return.lt(minimum_close_return)
            gross_return.loc[weak] = close_return.loc[weak]
            early_exit.loc[weak] = True
            unresolved &= ~weak

        if day_number == exit_day:
            gross_return.loc[unresolved] = day_close.loc[unresolved] / entry.loc[unresolved] - 1
            early_exit.loc[unresolved] = exit_day < future_high.shape[1]
            unresolved.loc[:] = False

    return {
        "gross_return": gross_return,
        "target_hit": target_hit,
        "early_exit": early_exit,
    }


def _assign_exit_outcome(
    frame: pd.DataFrame,
    prefix: str,
    outcome: dict[str, pd.Series],
    config: RisePatternConfig,
) -> None:
    frame[f"{prefix}_gross_return"] = outcome["gross_return"]
    frame[f"{prefix}_net_return"] = outcome["gross_return"] - config.transaction_cost_bps / 10_000.0
    frame[f"{prefix}_target_hit"] = outcome["target_hit"]
    frame[f"{prefix}_early_exit"] = outcome["early_exit"]


def _fixed_stop_key(stop_pct: float) -> str:
    return f"fixed_{stop_pct * 100:g}pct".replace(".", "_")


def _atr_stop_key(multiplier: float) -> str:
    return f"atr_{multiplier:g}x".replace(".", "_")


def _stopped_trade_outcome(
    entry: pd.Series,
    future_open: pd.DataFrame,
    future_high: pd.DataFrame,
    future_low: pd.DataFrame,
    exit_close: pd.Series,
    complete: pd.Series,
    stop_return: pd.Series,
) -> dict[str, pd.Series]:
    """Resolve target/stop order conservatively from daily OHLC bars."""
    valid = complete & stop_return.notna() & stop_return.lt(0)
    gross_return = pd.Series(np.nan, index=entry.index, dtype=float)
    target_hit = pd.Series(False, index=entry.index, dtype=bool)
    stop_hit = pd.Series(False, index=entry.index, dtype=bool)
    ambiguous_both_hit = pd.Series(False, index=entry.index, dtype=bool)
    unresolved = valid.copy()
    target_price = entry * 1.05
    stop_price = entry * (1.0 + stop_return)

    for day in range(future_high.shape[1]):
        day_open = future_open.iloc[:, day]
        day_high = future_high.iloc[:, day]
        day_low = future_low.iloc[:, day]
        touches_stop = day_low.le(stop_price)
        touches_target = day_high.ge(target_price)
        both = unresolved & touches_stop & touches_target
        ambiguous_both_hit.loc[both] = True

        gap_stop = unresolved & day_open.le(stop_price)
        intraday_stop = unresolved & ~gap_stop & touches_stop
        target = unresolved & ~gap_stop & ~intraday_stop & touches_target

        gross_return.loc[gap_stop] = day_open.loc[gap_stop] / entry.loc[gap_stop] - 1
        gross_return.loc[intraday_stop] = stop_return.loc[intraday_stop]
        gross_return.loc[target] = 0.05
        stop_hit.loc[gap_stop | intraday_stop] = True
        target_hit.loc[target] = True
        unresolved &= ~(gap_stop | intraday_stop | target)

    gross_return.loc[unresolved] = exit_close.loc[unresolved] / entry.loc[unresolved] - 1
    return {
        "gross_return": gross_return,
        "target_hit": target_hit,
        "stop_hit": stop_hit,
        "ambiguous_both_hit": ambiguous_both_hit,
    }


def _assign_stop_outcome(
    frame: pd.DataFrame,
    prefix: str,
    outcome: dict[str, pd.Series],
    config: RisePatternConfig,
) -> None:
    frame[f"{prefix}_gross_return"] = outcome["gross_return"]
    frame[f"{prefix}_net_return"] = outcome["gross_return"] - config.transaction_cost_bps / 10_000.0
    frame[f"{prefix}_target_hit"] = outcome["target_hit"]
    frame[f"{prefix}_stop_hit"] = outcome["stop_hit"]
    frame[f"{prefix}_ambiguous_both_hit"] = outcome["ambiguous_both_hit"]


def _select_top(frame: pd.DataFrame, score_column: str, top_n: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.sort_values(score_column, ascending=False).head(top_n).copy()


def _selective_candidate_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["rise_pattern_live_bottom"].fillna(False).astype(bool)
        & frame["_rise_shape"].isin(STRONG_SHAPES)
        & _numeric_column(frame, "turnover_value", default=0.0).ge(STRONG_SHAPE_MIN_TURNOVER)
        & _numeric_column(frame, "rsi_14", default=100.0).le(82.0)
    )


def _score_selective_day(
    day: pd.DataFrame,
    training: pd.DataFrame,
    config: RisePatternConfig,
) -> pd.DataFrame:
    """Estimate live-entry outcomes from prior live signals of the same shape."""
    result = day.copy()
    defaults: dict[str, Any] = {
        "selective_70_probability": 0.0,
        "selective_70_down_5pct_probability": 1.0,
        "selective_70_down_8pct_probability": 1.0,
        "selective_70_expected_net_return": -1.0,
        "selective_70_samples": 0,
        "selective_70_probability_lower_95": 0.0,
        "selective_70_peer_level": "none",
    }
    for column, default in defaults.items():
        result[column] = default
    if result.empty or training.empty:
        return result

    valid_training = training.loc[
        training["trade_outcome_available"].fillna(False).astype(bool)
        & training["rise_trade_entry_gap_return"].le(config.selective_max_gap_up)
    ].copy()
    if valid_training.empty:
        return result

    for index, row in result.iterrows():
        shape = str(row["_rise_shape"])
        shape_training = valid_training.loc[valid_training["_rise_shape"] == shape]
        if len(shape_training) < config.selective_min_samples:
            continue

        signature_training = shape_training.loc[
            shape_training["_rise_feature_signature"] == str(row["_rise_feature_signature"])
        ]
        peers = signature_training
        peer_level = "shape_signature"
        if len(peers) < config.selective_min_samples:
            quality = int(row["_rise_quality_score"])
            peers = shape_training.loc[
                shape_training["_rise_quality_score"].sub(quality).abs().le(1)
            ]
            peer_level = "shape_quality_band"
        if len(peers) < config.selective_min_samples:
            peers = shape_training
            peer_level = "shape"

        sample_count = int(len(peers))
        target_probability = _shrunk_binary_rate(
            peers["rise_trade_target_hit"],
            shape_training["rise_trade_target_hit"],
            config.selective_prior_strength,
        )
        down_5_probability = _shrunk_binary_rate(
            peers["rise_trade_down_5pct"],
            shape_training["rise_trade_down_5pct"],
            config.selective_prior_strength,
        )
        down_8_probability = _shrunk_binary_rate(
            peers["rise_trade_down_8pct"],
            shape_training["rise_trade_down_8pct"],
            config.selective_prior_strength,
        )
        expected_return = _shrunk_mean(
            peers["rise_trade_net_return"],
            shape_training["rise_trade_net_return"],
            config.selective_prior_strength,
        )
        successes = int(peers["rise_trade_target_hit"].sum())
        result.at[index, "selective_70_probability"] = target_probability
        result.at[index, "selective_70_down_5pct_probability"] = down_5_probability
        result.at[index, "selective_70_down_8pct_probability"] = down_8_probability
        result.at[index, "selective_70_expected_net_return"] = expected_return
        result.at[index, "selective_70_samples"] = sample_count
        result.at[index, "selective_70_probability_lower_95"] = _wilson_lower_bound(
            successes,
            sample_count,
        )
        result.at[index, "selective_70_peer_level"] = peer_level
    return result


def _shrunk_binary_rate(
    values: pd.Series,
    prior_values: pd.Series,
    prior_strength: float,
) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    prior = pd.to_numeric(prior_values, errors="coerce").dropna()
    if numeric.empty or prior.empty:
        return 0.0
    return float((numeric.sum() + prior.mean() * prior_strength) / (len(numeric) + prior_strength))


def _shrunk_mean(
    values: pd.Series,
    prior_values: pd.Series,
    prior_strength: float,
) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    prior = pd.to_numeric(prior_values, errors="coerce").dropna()
    if numeric.empty or prior.empty:
        return -1.0
    return float((numeric.sum() + prior.mean() * prior_strength) / (len(numeric) + prior_strength))


def _wilson_lower_bound(successes: int, samples: int, z: float = 1.96) -> float:
    if samples <= 0:
        return 0.0
    rate = successes / samples
    denominator = 1.0 + z**2 / samples
    center = rate + z**2 / (2.0 * samples)
    margin = z * np.sqrt(rate * (1.0 - rate) / samples + z**2 / (4.0 * samples**2))
    return float((center - margin) / denominator)


def _select_selective_trades(
    scored: pd.DataFrame,
    config: RisePatternConfig,
    *,
    probability_threshold: float,
    top_n: int,
) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    eligible = scored.loc[
        scored["rise_trade_entry_gap_return"].le(config.selective_max_gap_up)
        & scored["selective_70_probability"].ge(probability_threshold)
        & scored["selective_70_down_5pct_probability"].le(
            config.selective_max_down_5pct_probability
        )
        & scored["selective_70_down_8pct_probability"].le(
            config.selective_max_down_8pct_probability
        )
        & scored["selective_70_expected_net_return"].gt(config.selective_min_expected_net_return)
        & scored["selective_70_samples"].ge(config.selective_min_samples)
    ].copy()
    if eligible.empty:
        return eligible
    eligible["_selective_rank_score"] = (
        eligible["selective_70_expected_net_return"]
        + 0.05 * eligible["selective_70_probability"]
        - 0.05 * eligible["selective_70_down_5pct_probability"]
        - 0.08 * eligible["selective_70_down_8pct_probability"]
    )
    return (
        eligible.sort_values(
            ["date", "_selective_rank_score"],
            ascending=[True, False],
        )
        .groupby("date", sort=False, as_index=False)
        .head(top_n)
        .copy()
    )


def _strategy_summary(
    trades: pd.DataFrame,
    test_dates: list[Any],
    config: RisePatternConfig,
) -> dict[str, Any]:
    if trades.empty:
        return {
            "selected_signals": 0,
            "target_hit_rate": None,
            "mean_trade_net_return": None,
            "median_trade_net_return": None,
            "trade_win_rate": None,
            "ending_equity": 1.0,
            "max_drawdown": 0.0,
        }
    daily = trades.groupby("date", as_index=False)["rise_trade_net_return"].mean()
    daily = daily.set_index("date").reindex(test_dates, fill_value=0.0).reset_index()
    daily["portfolio_return"] = daily["rise_trade_net_return"] / config.horizon_days
    daily["equity"] = (1 + daily["portfolio_return"]).cumprod()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1
    result = {
        "selected_signals": int(len(trades)),
        "active_days": int(trades["date"].nunique()),
        "average_signals_per_test_day": float(len(trades) / len(test_dates)),
        "target_hit_rate": float(trades["rise_trade_target_hit"].mean()),
        "mean_future_max_return": float(trades["rise_trade_future_max_return"].mean()),
        "mean_trade_net_return": float(trades["rise_trade_net_return"].mean()),
        "median_trade_net_return": float(trades["rise_trade_net_return"].median()),
        "trade_win_rate": float((trades["rise_trade_net_return"] > 0).mean()),
        "ending_equity": float(daily["equity"].iloc[-1]),
        "max_drawdown": float(daily["drawdown"].min()),
    }
    if "selective_70_probability" in trades:
        result.update(
            {
                "mean_predicted_target_rate": float(trades["selective_70_probability"].mean()),
                "prediction_calibration_gap": float(
                    trades["rise_trade_target_hit"].mean()
                    - trades["selective_70_probability"].mean()
                ),
                "realized_down_5pct_rate": float(trades["rise_trade_down_5pct"].mean()),
                "realized_down_8pct_rate": float(trades["rise_trade_down_8pct"].mean()),
                "mean_entry_gap_return": float(trades["rise_trade_entry_gap_return"].mean()),
            }
        )
    return result


def _risk_summary(
    trades: pd.DataFrame,
    test_dates: list[Any],
    config: RisePatternConfig,
    *,
    return_column: str,
    target_column: str,
    stop_column: str | None = None,
    ambiguous_column: str | None = None,
    stop_distance_column: str | None = None,
) -> dict[str, Any]:
    if trades.empty or return_column not in trades:
        return {
            "selected_signals": 0,
            "target_hit_rate": None,
            "stop_hit_rate": None,
            "mean_trade_net_return": None,
            "median_trade_net_return": None,
            "trade_win_rate": None,
            "ending_equity": 1.0,
            "max_drawdown": 0.0,
        }

    valid = trades.loc[trades[return_column].notna()].copy()
    if valid.empty:
        return {
            "selected_signals": 0,
            "target_hit_rate": None,
            "stop_hit_rate": None,
            "mean_trade_net_return": None,
            "median_trade_net_return": None,
            "trade_win_rate": None,
            "ending_equity": 1.0,
            "max_drawdown": 0.0,
        }

    daily = valid.groupby("date", as_index=False)[return_column].mean()
    daily = daily.set_index("date").reindex(test_dates, fill_value=0.0).reset_index()
    daily["portfolio_return"] = daily[return_column] / config.horizon_days
    daily["equity"] = (1 + daily["portfolio_return"]).cumprod()
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1
    result: dict[str, Any] = {
        "selected_signals": int(len(valid)),
        "active_days": int(valid["date"].nunique()),
        "target_hit_rate": float(valid[target_column].mean()),
        "stop_hit_rate": (float(valid[stop_column].mean()) if stop_column is not None else 0.0),
        "mean_trade_net_return": float(valid[return_column].mean()),
        "median_trade_net_return": float(valid[return_column].median()),
        "trade_win_rate": float((valid[return_column] > 0).mean()),
        "ending_equity": float(daily["equity"].iloc[-1]),
        "max_drawdown": float(daily["drawdown"].min()),
    }
    if ambiguous_column is not None:
        result["ambiguous_both_hit_rate"] = float(valid[ambiguous_column].mean())
    if stop_distance_column is not None:
        result["mean_stop_distance_pct"] = float(valid[stop_distance_column].mean())
        result["median_stop_distance_pct"] = float(valid[stop_distance_column].median())
    return result


def _exit_summary(
    trades: pd.DataFrame,
    test_dates: list[Any],
    config: RisePatternConfig,
    *,
    return_column: str,
    target_column: str,
    early_exit_column: str | None = None,
) -> dict[str, Any]:
    result = _risk_summary(
        trades,
        test_dates,
        config,
        return_column=return_column,
        target_column=target_column,
    )
    if early_exit_column is not None and not trades.empty:
        valid = trades.loc[trades[return_column].notna()]
        result["early_exit_rate"] = (
            float(valid[early_exit_column].mean()) if not valid.empty else None
        )
    else:
        result["early_exit_rate"] = 0.0
    return result
