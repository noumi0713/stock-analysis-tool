from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ML_FEATURES = (
    "return_1d",
    "return_5d",
    "return_20d",
    "intraday_return",
    "volume_change_1d",
    "volume_ratio_5_20",
    "volume_ratio_1_20",
    "turnover_ratio_5_20",
    "turnover_ratio_1_20",
    "volatility_10d",
    "range_width_10d",
    "up_volume_share_10d",
    "rsi_14",
    "atr_14_pct",
    "observed_inflow_score",
    "individual_trend_score",
    "relative_return_20d",
    "sector_17_trend_score",
    "sector_17_breadth_5d",
    "sector_33_trend_score",
    "sector_33_breadth_5d",
    "sakata_score",
    "sakata_bullish_count",
    "sakata_bearish_count",
    "retail_safety_score",
    "retail_action_score",
    "retail_flow_score",
    "_rise_close_location",
    "_rise_market_breadth_5d",
    "_rise_market_median_return_20d",
    "_rise_quality_score",
    "rise_trade_entry_gap_return",
)

STRONG_SHAPES = frozenset(
    {
        "capitulation_reversal",
        "rounded_base",
        "sharp_selloff",
    }
)

STRONG_SHAPE_LABELS = {
    "capitulation_reversal": "投げ売り反転",
    "rounded_base": "緩やかな底固め",
    "sharp_selloff": "急落継続",
}

# Keep the existing JSON field names so the published site remains compatible.
LIVE_STRONG_SIGNAL_COLUMNS = (
    "ml_sharp_probability",
    "ml_sharp_down_5pct_probability",
    "ml_sharp_down_8pct_probability",
    "ml_sharp_expected_net_return",
    "ml_sharp_model_samples",
    "ml_sharp_signal",
    "ml_sharp_rank",
    "ml_sharp_reason",
    "ml_sharp_entry_rule",
)

LIVE_SHARP_SIGNAL_COLUMNS = LIVE_STRONG_SIGNAL_COLUMNS

LIVE_TEN_DAY_SIGNAL_COLUMNS = (
    "ml_ten_day_probability",
    "ml_ten_day_down_5pct_probability",
    "ml_ten_day_down_8pct_probability",
    "ml_ten_day_expected_net_return",
    "ml_ten_day_model_samples",
    "ml_ten_day_signal",
    "ml_ten_day_rank",
    "ml_ten_day_reason",
    "ml_ten_day_entry_rule",
)


@dataclass
class _BinaryPredictor:
    model: Any | None
    calibrator: IsotonicRegression | None
    constant: float

    def predict(self, values: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(values), self.constant, dtype=float)
        probability = self.model.predict_proba(values)[:, 1]
        if self.calibrator is not None:
            probability = self.calibrator.predict(probability)
        return np.clip(probability, 0.0, 1.0)


@dataclass
class _ModelBundle:
    target: _BinaryPredictor
    down_5pct: _BinaryPredictor
    down_8pct: _BinaryPredictor
    expected_return: Any


def score_latest_strong_shape_candidates(
    frame: pd.DataFrame,
    *,
    minimum_shape_samples: int = 180,
    minimum_turnover: float = 200_000_000.0,
) -> pd.DataFrame:
    """Score the three strong shapes with the frozen walk-forward ML rule.

    The next open is not known at the close, so the model is evaluated at a
    neutral 0% gap. Execution remains conditional on the actual next-open gap
    being no greater than +3% versus the signal-day close.
    """
    if frame.empty:
        return pd.DataFrame(columns=[*frame.columns, *LIVE_STRONG_SIGNAL_COLUMNS])
    work = frame.copy()
    latest_date = work["date"].max()
    latest = work.loc[work["date"] == latest_date].copy()
    latest["ml_sharp_probability"] = 0.0
    latest["ml_sharp_down_5pct_probability"] = 1.0
    latest["ml_sharp_down_8pct_probability"] = 1.0
    latest["ml_sharp_expected_net_return"] = -1.0
    latest["ml_sharp_model_samples"] = 0
    latest["ml_sharp_signal"] = False
    latest["ml_sharp_rank"] = pd.Series(pd.NA, index=latest.index, dtype="Int64")
    latest["ml_sharp_reason"] = ""
    latest["ml_sharp_entry_rule"] = "翌営業日寄付きが前日終値比+3%以下の場合のみ有効"

    training = work.loc[
        work["trade_outcome_available"].fillna(False).astype(bool)
        & work["_rise_shape"].isin(STRONG_SHAPES)
        & work["rise_pattern_live_bottom"].fillna(False).astype(bool)
        & pd.to_numeric(work["turnover_value"], errors="coerce").ge(minimum_turnover)
        & pd.to_numeric(work["rsi_14"], errors="coerce").le(82.0)
        & pd.to_numeric(work["rise_trade_entry_gap_return"], errors="coerce").between(-0.10, 0.04)
    ].sort_values(["date", "ticker"])
    if training.empty:
        return latest

    eligible = latest.loc[
        latest["_rise_shape"].isin(STRONG_SHAPES)
        & latest["rise_pattern_live_bottom"].fillna(False).astype(bool)
        & pd.to_numeric(latest["turnover_value"], errors="coerce").ge(minimum_turnover)
        & pd.to_numeric(latest["rsi_14"], errors="coerce").le(82.0)
    ].copy()
    if eligible.empty:
        return latest

    eligible["rise_trade_entry_gap_return"] = 0.0
    scored_indexes: list[Any] = []
    for shape, shape_eligible in eligible.groupby("_rise_shape", sort=False):
        shape_training = training.loc[training["_rise_shape"].eq(shape)]
        if len(shape_training) < minimum_shape_samples:
            continue
        bundle = _fit_bundle(shape_training, "hist_gradient_boosting")
        feature_values = _feature_frame(shape_eligible)
        indexes = shape_eligible.index
        eligible.loc[indexes, "ml_sharp_probability"] = bundle.target.predict(feature_values)
        eligible.loc[indexes, "ml_sharp_down_5pct_probability"] = bundle.down_5pct.predict(
            feature_values
        )
        eligible.loc[indexes, "ml_sharp_down_8pct_probability"] = bundle.down_8pct.predict(
            feature_values
        )
        eligible.loc[indexes, "ml_sharp_expected_net_return"] = bundle.expected_return.predict(
            feature_values
        )
        eligible.loc[indexes, "ml_sharp_model_samples"] = len(shape_training)
        scored_indexes.extend(indexes.tolist())
    if not scored_indexes:
        return latest
    scored = eligible.loc[scored_indexes].copy()
    scored["_ml_sharp_rank_score"] = (
        scored["ml_sharp_probability"]
        + 4.0 * scored["ml_sharp_expected_net_return"]
        - 0.10 * scored["ml_sharp_down_5pct_probability"]
        - 0.15 * scored["ml_sharp_down_8pct_probability"]
    )
    candidates = scored.loc[
        scored["ml_sharp_probability"].ge(0.55)
        & scored["ml_sharp_down_5pct_probability"].le(0.50)
        & scored["ml_sharp_down_8pct_probability"].le(0.30)
        & scored["ml_sharp_expected_net_return"].gt(-0.01)
    ].sort_values("_ml_sharp_rank_score", ascending=False)
    if not candidates.empty:
        winner_index = candidates.index[0]
        eligible.at[winner_index, "ml_sharp_signal"] = True
        eligible.at[winner_index, "ml_sharp_rank"] = 1
        probability = float(eligible.at[winner_index, "ml_sharp_probability"])
        shape = str(eligible.at[winner_index, "_rise_shape"])
        shape_label = STRONG_SHAPE_LABELS.get(shape, shape)
        eligible.at[winner_index, "ml_sharp_reason"] = (
            f"{shape_label}・売買代金2億円以上・強形状ML参考率{probability:.0%}・"
            "翌日寄付き条件待ち"
        )

    for column in LIVE_STRONG_SIGNAL_COLUMNS:
        latest.loc[eligible.index, column] = eligible[column]
    return latest


def score_latest_sharp_selloff_candidates(
    frame: pd.DataFrame,
    *,
    minimum_shape_samples: int = 180,
    minimum_turnover: float = 200_000_000.0,
) -> pd.DataFrame:
    """Backward-compatible alias for the public strong-shape scorer."""
    return score_latest_strong_shape_candidates(
        frame,
        minimum_shape_samples=minimum_shape_samples,
        minimum_turnover=minimum_turnover,
    )


def score_latest_ten_day_candidates(
    frame: pd.DataFrame,
    parameters: dict[str, Any],
    *,
    minimum_shape_samples: int = 180,
    minimum_turnover: float = 200_000_000.0,
) -> pd.DataFrame:
    """Score the latest strong-shape rows for a 10-day +5% target.

    ``parameters`` must come from the development half of the walk-forward
    study.  The actual next-open gap remains an execution-time condition; the
    close-time score uses a neutral 0% gap.
    """
    if frame.empty:
        return pd.DataFrame(columns=[*frame.columns, *LIVE_TEN_DAY_SIGNAL_COLUMNS])
    work = frame.copy()
    latest_date = work["date"].max()
    latest = work.loc[work["date"] == latest_date].copy()
    gap_limit = float(parameters.get("max_gap_up", 0.03))
    gap_label = f"{gap_limit:+.0%}"
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
            f"翌営業日寄付きが前日終値比{gap_label}以下の場合のみ有効"
        ),
    }
    for column, default in defaults.items():
        latest[column] = default
    latest["ml_ten_day_rank"] = latest["ml_ten_day_rank"].astype("Int64")

    allowed_shapes = parameters.get("allowed_shapes") or list(STRONG_SHAPES)
    training = work.loc[
        work["trade_outcome_available"].fillna(False).astype(bool)
        & work["_rise_shape"].isin(allowed_shapes)
        & work["rise_pattern_live_bottom"].fillna(False).astype(bool)
        & pd.to_numeric(work["turnover_value"], errors="coerce").ge(minimum_turnover)
        & pd.to_numeric(work["rsi_14"], errors="coerce").le(82.0)
        & pd.to_numeric(work["rise_trade_entry_gap_return"], errors="coerce").between(-0.10, 0.04)
    ].sort_values(["date", "ticker"])
    eligible = latest.loc[
        latest["_rise_shape"].isin(allowed_shapes)
        & latest["rise_pattern_live_bottom"].fillna(False).astype(bool)
        & pd.to_numeric(latest["turnover_value"], errors="coerce").ge(minimum_turnover)
        & pd.to_numeric(latest["rsi_14"], errors="coerce").le(82.0)
    ].copy()
    if training.empty or eligible.empty:
        return latest

    minimum_breadth = parameters.get("min_market_breadth_5d")
    if minimum_breadth is not None:
        eligible = eligible.loc[
            pd.to_numeric(eligible["_rise_market_breadth_5d"], errors="coerce").ge(
                float(minimum_breadth)
            )
        ]
    minimum_market_return = parameters.get("min_market_median_return_20d")
    if minimum_market_return is not None:
        eligible = eligible.loc[
            pd.to_numeric(
                eligible["_rise_market_median_return_20d"], errors="coerce"
            ).ge(float(minimum_market_return))
        ]
    if eligible.empty:
        return latest

    eligible["rise_trade_entry_gap_return"] = 0.0
    model_name = str(parameters.get("model", "hist_gradient_boosting"))
    scored_indexes: list[Any] = []
    for shape, shape_eligible in eligible.groupby("_rise_shape", sort=False):
        shape_training = training.loc[training["_rise_shape"].eq(shape)]
        if len(shape_training) < minimum_shape_samples:
            continue
        bundle = _fit_bundle(shape_training, model_name)
        feature_values = _feature_frame(shape_eligible)
        indexes = shape_eligible.index
        eligible.loc[indexes, "ml_ten_day_probability"] = bundle.target.predict(
            feature_values
        )
        eligible.loc[indexes, "ml_ten_day_down_5pct_probability"] = (
            bundle.down_5pct.predict(feature_values)
        )
        eligible.loc[indexes, "ml_ten_day_down_8pct_probability"] = (
            bundle.down_8pct.predict(feature_values)
        )
        eligible.loc[indexes, "ml_ten_day_expected_net_return"] = (
            bundle.expected_return.predict(feature_values)
        )
        eligible.loc[indexes, "ml_ten_day_model_samples"] = len(shape_training)
        scored_indexes.extend(indexes.tolist())
    if not scored_indexes:
        return latest

    scored = eligible.loc[scored_indexes].copy()
    scored["_ml_ten_day_rank_score"] = (
        scored["ml_ten_day_probability"]
        + 4.0 * scored["ml_ten_day_expected_net_return"]
        - 0.10 * scored["ml_ten_day_down_5pct_probability"]
        - 0.15 * scored["ml_ten_day_down_8pct_probability"]
    )
    candidates = scored.loc[
        scored["ml_ten_day_probability"].ge(
            float(parameters.get("probability_threshold", 0.60))
        )
        & scored["ml_ten_day_down_5pct_probability"].le(
            float(parameters.get("max_down_5pct_probability", 0.50))
        )
        & scored["ml_ten_day_down_8pct_probability"].le(
            float(parameters.get("max_down_8pct_probability", 0.30))
        )
        & scored["ml_ten_day_expected_net_return"].gt(
            float(parameters.get("min_expected_net_return", 0.0))
        )
    ].sort_values("_ml_ten_day_rank_score", ascending=False)
    if not candidates.empty:
        winner_index = candidates.index[0]
        eligible.at[winner_index, "ml_ten_day_signal"] = True
        eligible.at[winner_index, "ml_ten_day_rank"] = 1
        probability = float(eligible.at[winner_index, "ml_ten_day_probability"])
        shape = str(eligible.at[winner_index, "_rise_shape"])
        shape_label = STRONG_SHAPE_LABELS.get(shape, shape)
        eligible.at[winner_index, "ml_ten_day_reason"] = (
            f"{shape_label}・売買代金2億円以上・10営業日+5%参考率"
            f"{probability:.0%}・翌日寄付き条件待ち"
        )

    for column in LIVE_TEN_DAY_SIGNAL_COLUMNS:
        latest.loc[eligible.index, column] = eligible[column]
    return latest


def walk_forward_ml_scores(
    pool: pd.DataFrame,
    evaluation_dates: list[Any],
    date_position: dict[Any, int],
    *,
    horizon_days: int,
    refit_days: int = 30,
    minimum_shape_samples: int = 180,
) -> pd.DataFrame:
    """Score candidates with shape-specific models fitted only on prior outcomes."""
    if pool.empty or not evaluation_dates:
        return pd.DataFrame()
    work = pool.copy()
    work["_date_position"] = work["date"].map(date_position)
    scored_blocks: list[pd.DataFrame] = []
    for start in range(0, len(evaluation_dates), refit_days):
        block_dates = evaluation_dates[start : start + refit_days]
        cutoff = date_position[block_dates[0]] - horizon_days
        training = work.loc[
            work["_date_position"].le(cutoff)
            & work["trade_outcome_available"].fillna(False).astype(bool)
            & work["rise_trade_entry_gap_return"].between(-0.10, 0.04)
        ].sort_values(["date", "ticker"])
        block = work.loc[
            work["date"].isin(block_dates)
            & work["trade_outcome_available"].fillna(False).astype(bool)
            & work["rise_trade_entry_gap_return"].between(-0.10, 0.04)
        ].copy()
        if training.empty or block.empty:
            continue
        for model_name in ("logistic", "hist_gradient_boosting"):
            for shape, shape_block in block.groupby("_rise_shape", sort=False):
                shape_training = training.loc[training["_rise_shape"] == shape]
                if len(shape_training) < minimum_shape_samples:
                    continue
                bundle = _fit_bundle(shape_training, model_name)
                feature_values = _feature_frame(shape_block)
                indexes = shape_block.index
                block.loc[indexes, f"ml_{model_name}_target_probability"] = bundle.target.predict(
                    feature_values
                )
                block.loc[indexes, f"ml_{model_name}_down_5pct_probability"] = (
                    bundle.down_5pct.predict(feature_values)
                )
                block.loc[indexes, f"ml_{model_name}_down_8pct_probability"] = (
                    bundle.down_8pct.predict(feature_values)
                )
                block.loc[indexes, f"ml_{model_name}_expected_net_return"] = (
                    bundle.expected_return.predict(feature_values)
                )
        scored_blocks.append(block)
    if not scored_blocks:
        return pd.DataFrame()
    return pd.concat(scored_blocks, ignore_index=True)


def tune_and_select_ml_strategy(
    scored: pd.DataFrame,
    evaluation_dates: list[Any],
    *,
    minimum_development_signals: int = 40,
    probability_thresholds: tuple[float, ...] = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
    gap_limits: tuple[float, ...] = (0.00, 0.01, 0.02, 0.03),
    top_n_options: tuple[int, ...] = (1, 2, 3),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tune on the first half and evaluate the unchanged rule on the final half."""
    if scored.empty or len(evaluation_dates) < 2:
        return pd.DataFrame(), {"status": "no_scored_candidates"}
    split = len(evaluation_dates) // 2
    development_dates = evaluation_dates[:split]
    validation_dates = evaluation_dates[split:]
    development = scored.loc[scored["date"].isin(development_dates)]
    validation = scored.loc[scored["date"].isin(validation_dates)]
    experiments: list[dict[str, Any]] = []
    risk_profiles = (
        ("loose", 0.50, 0.30, -0.01),
        ("balanced", 0.35, 0.20, 0.00),
        ("strict", 0.25, 0.12, 0.003),
    )
    shape_profiles = (
        ("all_strong_shapes", ()),
        ("sharp_selloff", ("sharp_selloff",)),
        ("capitulation_reversal", ("capitulation_reversal",)),
        ("rounded_base", ("rounded_base",)),
    )
    regime_profiles = (
        ("all_regimes", None, None),
        ("supportive_breadth", 0.50, -0.02),
        ("positive_market_trend", 0.50, 0.00),
    )
    for shape_profile, allowed_shapes in shape_profiles:
        for regime_profile, minimum_breadth, minimum_market_return in regime_profiles:
            for model_name in ("logistic", "hist_gradient_boosting"):
                probability_column = f"ml_{model_name}_target_probability"
                if probability_column not in development:
                    continue
                for probability_threshold in probability_thresholds:
                    for gap_limit in gap_limits:
                        for (
                            risk_name,
                            down_5_limit,
                            down_8_limit,
                            expected_return_min,
                        ) in risk_profiles:
                            for top_n in top_n_options:
                                parameters = {
                                    "shape_profile": shape_profile,
                                    "allowed_shapes": list(allowed_shapes),
                                    "regime_profile": regime_profile,
                                    "min_market_breadth_5d": minimum_breadth,
                                    "min_market_median_return_20d": (minimum_market_return),
                                    "model": model_name,
                                    "probability_threshold": probability_threshold,
                                    "max_gap_up": gap_limit,
                                    "risk_profile": risk_name,
                                    "max_down_5pct_probability": down_5_limit,
                                    "max_down_8pct_probability": down_8_limit,
                                    "min_expected_net_return": expected_return_min,
                                    "top_n_per_day": top_n,
                                }
                                trades = _select_with_parameters(
                                    development,
                                    parameters,
                                )
                                summary = _compact_summary(trades)
                                fold_summaries = _chronological_fold_summaries(
                                    development,
                                    development_dates,
                                    parameters,
                                )
                                objective = _development_objective(
                                    summary,
                                    fold_summaries,
                                    minimum_development_signals,
                                    model_name=model_name,
                                )
                                experiments.append(
                                    {
                                        **parameters,
                                        **summary,
                                        "development_folds": fold_summaries,
                                        "objective": objective,
                                    }
                                )
    if not experiments:
        return pd.DataFrame(), {"status": "no_model_predictions"}
    ranked = sorted(experiments, key=lambda item: item["objective"], reverse=True)
    chosen = ranked[0]
    parameters = {
        key: chosen[key]
        for key in (
            "shape_profile",
            "allowed_shapes",
            "regime_profile",
            "min_market_breadth_5d",
            "min_market_median_return_20d",
            "model",
            "probability_threshold",
            "max_gap_up",
            "risk_profile",
            "max_down_5pct_probability",
            "max_down_8pct_probability",
            "min_expected_net_return",
            "top_n_per_day",
        )
    }
    validation_trades = _select_with_parameters(validation, parameters)
    development_trades = _select_with_parameters(development, parameters)
    combined_trades = pd.concat(
        [development_trades, validation_trades],
        ignore_index=True,
    )
    validation_summary = _compact_summary(validation_trades)
    combined_summary = _compact_summary(combined_trades)
    validation_folds = _chronological_fold_summaries(
        validation,
        validation_dates,
        parameters,
    )
    sharp_candidate_parameters = {
        **parameters,
        "shape_profile": "sharp_selloff_confirmation",
        "allowed_shapes": ["sharp_selloff"],
    }
    sharp_development_trades = _select_with_parameters(
        development,
        sharp_candidate_parameters,
    )
    sharp_validation_trades = _select_with_parameters(
        validation,
        sharp_candidate_parameters,
    )
    confirmation_dates = validation_dates[(len(validation_dates) * 2) // 3 :]
    sharp_confirmation_trades = _select_with_parameters(
        validation.loc[validation["date"].isin(confirmation_dates)],
        sharp_candidate_parameters,
    )
    sharp_combined_trades = pd.concat(
        [sharp_development_trades, sharp_validation_trades],
        ignore_index=True,
    )
    sharp_validation_summary = _compact_summary(sharp_validation_trades)
    sharp_confirmation_summary = _compact_summary(sharp_confirmation_trades)
    diagnostics = {
        "status": "completed",
        "selection_rule": (
            "configuration chosen only from three chronological development folds; "
            "validation half was untouched"
        ),
        "development_start": str(development_dates[0]),
        "development_end": str(development_dates[-1]),
        "validation_start": str(validation_dates[0]),
        "validation_end": str(validation_dates[-1]),
        "chosen_parameters": parameters,
        "development": _compact_summary(development_trades),
        "development_folds": chosen["development_folds"],
        "development_by_shape": _summaries_by_shape(development_trades),
        "validation": validation_summary,
        "validation_folds": validation_folds,
        "validation_by_shape": _summaries_by_shape(validation_trades),
        "sharp_selloff_candidate": {
            "status": "exploratory_post_selection_candidate",
            "parameters": sharp_candidate_parameters,
            "development": _compact_summary(sharp_development_trades),
            "validation": sharp_validation_summary,
            "confirmation_start": (str(confirmation_dates[0]) if confirmation_dates else None),
            "confirmation_end": (str(confirmation_dates[-1]) if confirmation_dates else None),
            "latest_period_confirmation": sharp_confirmation_summary,
            "combined": _compact_summary(sharp_combined_trades),
            "historical_60pct_candidate_met": bool(
                sharp_validation_summary["selected_signals"] >= 30
                and (sharp_validation_summary["target_hit_rate"] or 0.0) >= 0.60
                and (sharp_validation_summary["mean_trade_net_return"] or -1.0) > 0.0
                and sharp_confirmation_summary["selected_signals"] >= 15
                and (sharp_confirmation_summary["target_hit_rate"] or 0.0) >= 0.60
                and (sharp_confirmation_summary["mean_trade_net_return"] or -1.0) > 0.0
            ),
            "caveat": (
                "shape restriction was identified during validation analysis; freeze it "
                "before collecting new dates for a fully untouched confirmation"
            ),
        },
        "combined": combined_summary,
        "validation_goal_met": bool(
            validation_summary["selected_signals"] >= 30
            and (validation_summary["target_hit_rate"] or 0.0) >= 0.60
            and (validation_summary["mean_trade_net_return"] or -1.0) > 0.0
        ),
        "combined_100_signal_goal_met": bool(
            combined_summary["selected_signals"] >= 100
            and (combined_summary["target_hit_rate"] or 0.0) >= 0.60
            and (combined_summary["mean_trade_net_return"] or -1.0) > 0.0
        ),
        "top_development_configurations": ranked[:20],
    }
    return validation_trades, diagnostics


def _fit_bundle(training: pd.DataFrame, model_name: str) -> _ModelBundle:
    features = _feature_frame(training)
    return _ModelBundle(
        target=_fit_binary_predictor(
            features,
            training["rise_trade_target_hit"],
            model_name,
        ),
        down_5pct=_fit_binary_predictor(
            features,
            training["rise_trade_down_5pct"],
            model_name,
        ),
        down_8pct=_fit_binary_predictor(
            features,
            training["rise_trade_down_8pct"],
            model_name,
        ),
        expected_return=_fit_return_model(
            features,
            training["rise_trade_net_return"],
            model_name,
        ),
    )


def _fit_binary_predictor(
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str,
) -> _BinaryPredictor:
    numeric_target = pd.to_numeric(target, errors="coerce").fillna(0).astype(int)
    constant = float(numeric_target.mean())
    if numeric_target.nunique() < 2 or len(numeric_target) < 100:
        return _BinaryPredictor(None, None, constant)
    split = max(100, int(len(numeric_target) * 0.80))
    split = min(split, len(numeric_target) - 50)
    fit_x = features.iloc[:split]
    fit_y = numeric_target.iloc[:split]
    calibration_x = features.iloc[split:]
    calibration_y = numeric_target.iloc[split:]
    if fit_y.nunique() < 2:
        return _BinaryPredictor(None, None, constant)
    model = _classifier(model_name)
    model.fit(fit_x, fit_y)
    calibrator = None
    raw_probability = model.predict_proba(calibration_x)[:, 1]
    if calibration_y.nunique() >= 2 and np.unique(raw_probability).size >= 3:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw_probability, calibration_y)
    return _BinaryPredictor(model, calibrator, constant)


def _classifier(model_name: str) -> Any:
    if model_name == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.25, max_iter=600),
        )
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=80,
            max_leaf_nodes=7,
            min_samples_leaf=50,
            l2_regularization=2.0,
            random_state=42,
        ),
    )


def _fit_return_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str,
) -> Any:
    numeric_target = pd.to_numeric(target, errors="coerce").fillna(0.0)
    if model_name == "logistic":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=10.0),
        )
    else:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=80,
                max_leaf_nodes=7,
                min_samples_leaf=50,
                l2_regularization=2.0,
                random_state=42,
            ),
        )
    model.fit(features, numeric_target)
    return model


def _feature_frame(values: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=values.index)
    for column in ML_FEATURES:
        result[column] = (
            pd.to_numeric(values[column], errors="coerce") if column in values else np.nan
        )
    return result.replace([np.inf, -np.inf], np.nan)


def _select_with_parameters(
    scored: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    model_name = parameters["model"]
    target = f"ml_{model_name}_target_probability"
    down_5 = f"ml_{model_name}_down_5pct_probability"
    down_8 = f"ml_{model_name}_down_8pct_probability"
    expected = f"ml_{model_name}_expected_net_return"
    if any(column not in scored for column in (target, down_5, down_8, expected)):
        return pd.DataFrame(columns=scored.columns)
    eligible_mask = (
        scored[target].ge(parameters["probability_threshold"])
        & scored[down_5].le(parameters["max_down_5pct_probability"])
        & scored[down_8].le(parameters["max_down_8pct_probability"])
        & scored[expected].gt(parameters["min_expected_net_return"])
        & scored["rise_trade_entry_gap_return"].le(parameters["max_gap_up"])
    )
    allowed_shapes = parameters.get("allowed_shapes") or []
    if allowed_shapes:
        if "_rise_shape" not in scored:
            return pd.DataFrame(columns=scored.columns)
        eligible_mask &= scored["_rise_shape"].isin(allowed_shapes)
    minimum_breadth = parameters.get("min_market_breadth_5d")
    if minimum_breadth is not None:
        eligible_mask &= pd.to_numeric(scored["_rise_market_breadth_5d"], errors="coerce").ge(
            float(minimum_breadth)
        )
    minimum_market_return = parameters.get("min_market_median_return_20d")
    if minimum_market_return is not None:
        eligible_mask &= pd.to_numeric(
            scored["_rise_market_median_return_20d"], errors="coerce"
        ).ge(float(minimum_market_return))
    eligible = scored.loc[eligible_mask].copy()
    if eligible.empty:
        return eligible
    eligible["_ml_rank_score"] = (
        eligible[target]
        + 4.0 * eligible[expected]
        - 0.10 * eligible[down_5]
        - 0.15 * eligible[down_8]
    )
    return (
        eligible.sort_values(["date", "_ml_rank_score"], ascending=[True, False])
        .groupby("date", sort=False, as_index=False)
        .head(int(parameters["top_n_per_day"]))
        .copy()
    )


def _compact_summary(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "selected_signals": 0,
            "active_days": 0,
            "target_hit_rate": None,
            "mean_trade_net_return": None,
            "trade_win_rate": None,
            "target_rate_lower_95": None,
        }
    successes = int(trades["rise_trade_target_hit"].sum())
    samples = int(len(trades))
    return {
        "selected_signals": samples,
        "active_days": int(trades["date"].nunique()),
        "target_hit_rate": float(successes / samples),
        "mean_trade_net_return": float(trades["rise_trade_net_return"].mean()),
        "trade_win_rate": float((trades["rise_trade_net_return"] > 0).mean()),
        "target_rate_lower_95": _wilson_lower_bound(successes, samples),
    }


def _chronological_fold_summaries(
    scored: pd.DataFrame,
    dates: list[Any],
    parameters: dict[str, Any],
    *,
    fold_count: int = 3,
) -> list[dict[str, Any]]:
    if not dates:
        return []
    summaries: list[dict[str, Any]] = []
    for fold_number, fold_dates in enumerate(
        np.array_split(np.asarray(dates, dtype=object), fold_count),
        start=1,
    ):
        date_values = fold_dates.tolist()
        fold_trades = _select_with_parameters(
            scored.loc[scored["date"].isin(date_values)],
            parameters,
        )
        summaries.append(
            {
                "fold": fold_number,
                "start": str(date_values[0]) if date_values else None,
                "end": str(date_values[-1]) if date_values else None,
                **_compact_summary(fold_trades),
            }
        )
    return summaries


def _summaries_by_shape(trades: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if trades.empty or "_rise_shape" not in trades:
        return {}
    return {
        str(shape): _compact_summary(shape_trades)
        for shape, shape_trades in trades.groupby("_rise_shape", sort=True)
    }


def _development_objective(
    summary: dict[str, Any],
    fold_summaries: list[dict[str, Any]],
    minimum_signals: int,
    *,
    model_name: str,
) -> float:
    samples = int(summary["selected_signals"])
    target_rate = float(summary["target_hit_rate"] or 0.0)
    expected_return = float(summary["mean_trade_net_return"] or -1.0)
    lower_bound = float(summary["target_rate_lower_95"] or 0.0)
    fold_counts = [int(item["selected_signals"]) for item in fold_summaries]
    fold_hit_rates = [float(item["target_hit_rate"] or 0.0) for item in fold_summaries]
    fold_returns = [float(item["mean_trade_net_return"] or -1.0) for item in fold_summaries]
    minimum_fold_signals = max(5, minimum_signals // 6)
    if (
        samples < minimum_signals
        or expected_return <= 0
        or not fold_counts
        or min(fold_counts) < minimum_fold_signals
    ):
        return (
            -10.0
            + samples / max(minimum_signals, 1)
            + expected_return
            + (min(fold_counts) if fold_counts else 0) / minimum_fold_signals
        )

    minimum_fold_hit_rate = min(fold_hit_rates)
    minimum_fold_return = min(fold_returns)
    positive_return_folds = sum(value > 0 for value in fold_returns)
    target_stability = float(np.std(fold_hit_rates))
    goal_bonus = 4.0 if target_rate >= 0.60 else 0.0
    fold_floor_bonus = 2.0 if minimum_fold_hit_rate >= 0.50 else 0.0
    expectancy_bonus = 1.0 if positive_return_folds == len(fold_returns) else 0.0
    sample_bonus = min(samples, 120) / 500.0
    complexity_penalty = 0.03 if model_name == "hist_gradient_boosting" else 0.0
    return (
        goal_bonus
        + fold_floor_bonus
        + expectancy_bonus
        + 1.5 * minimum_fold_hit_rate
        + lower_bound
        + 2.0 * minimum_fold_return
        + 2.0 * expected_return
        + sample_bonus
        - target_stability
        - complexity_penalty
    )


def _wilson_lower_bound(successes: int, samples: int, z: float = 1.96) -> float:
    if samples <= 0:
        return 0.0
    rate = successes / samples
    denominator = 1.0 + z**2 / samples
    center = rate + z**2 / (2.0 * samples)
    margin = z * np.sqrt(rate * (1.0 - rate) / samples + z**2 / (4.0 * samples**2))
    return float((center - margin) / denominator)
