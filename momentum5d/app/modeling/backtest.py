from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.modeling.features import FEATURE_COLUMNS


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    start: date | None = None
    end: date | None = None
    min_train_days: int = 252
    retrain_every_days: int = 20
    horizon_days: int = 5
    top_n: int = 20
    min_turnover: float = 10_000_000.0
    transaction_cost_bps: float = 20.0
    random_state: int = 42

    def __post_init__(self) -> None:
        for name in (
            "min_train_days",
            "retrain_every_days",
            "horizon_days",
            "top_n",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} は1以上である必要があります")
        if self.min_turnover < 0 or self.transaction_cost_bps < 0:
            raise ValueError("min_turnover と transaction_cost_bps は0以上です")


@dataclass(frozen=True, slots=True)
class BacktestResult:
    predictions: pd.DataFrame
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    summary: dict[str, Any]


class WalkForwardBacktester:
    """5営業日のラベル期間をパージするウォークフォワード検証。"""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(self, dataset: pd.DataFrame) -> BacktestResult:
        if dataset.empty:
            raise ValueError("バックテスト対象が空です")
        missing = set(FEATURE_COLUMNS + ["date", "code", "target_5d"]).difference(dataset.columns)
        if missing:
            raise ValueError(f"バックテスト必須列がありません: {sorted(missing)}")

        frame = dataset.copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        frame = frame.loc[frame["target_5d"].notna()].copy()
        all_dates = sorted(frame["date"].drop_duplicates())
        if len(all_dates) < self.config.min_train_days + self.config.horizon_days + 1:
            raise ValueError(
                "学習期間が不足しています: "
                f"available_days={len(all_dates)}, "
                f"required>{self.config.min_train_days + self.config.horizon_days}"
            )

        date_position = {value: position for position, value in enumerate(all_dates)}
        frame["_backtest_date_position"] = frame["date"].map(date_position)
        first_test_position = self.config.min_train_days + self.config.horizon_days
        requested_test_dates = [
            value
            for position, value in enumerate(all_dates)
            if position >= first_test_position
            and (self.config.start is None or value >= self.config.start)
            and (self.config.end is None or value <= self.config.end)
        ]
        if not requested_test_dates:
            raise ValueError("指定期間に評価可能なテスト日がありません")

        predictions: list[pd.DataFrame] = []
        for fold, block_start in enumerate(
            range(0, len(requested_test_dates), self.config.retrain_every_days),
            start=1,
        ):
            block_dates = requested_test_dates[
                block_start : block_start + self.config.retrain_every_days
            ]
            test_start_position = date_position[block_dates[0]]
            train_cutoff_position = test_start_position - self.config.horizon_days
            train = frame.loc[frame["_backtest_date_position"] <= train_cutoff_position]
            test = frame.loc[frame["date"].isin(block_dates)].copy()
            train = self._eligible_rows(train, require_outcome=False)
            test = self._eligible_rows(test, require_outcome=False)
            if train["target_5d"].nunique() < 2 or test.empty:
                continue

            model = self._new_model()
            model.fit(train[FEATURE_COLUMNS], train["target_5d"].astype("int8"))
            test["probability"] = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
            test["fold"] = fold
            predictions.append(test)

        if not predictions:
            raise ValueError("有効なウォークフォワードfoldを作成できません")
        scored = pd.concat(predictions, ignore_index=True, sort=False)
        scored["rank"] = (
            scored.groupby("date")["probability"]
            .rank(method="first", ascending=False)
            .astype("int32")
        )
        scored["selected"] = scored["rank"] <= self.config.top_n
        scored["trade_net_return"] = (
            scored["trade_gross_return"] - self.config.transaction_cost_bps / 10_000.0
        )

        trades = scored.loc[
            scored["selected"] & scored["trade_outcome_available"].fillna(False)
        ].copy()
        equity_curve = self._equity_curve(trades)
        summary = self._summary(scored, trades, equity_curve)
        prediction_columns = [
            "date",
            "code",
            "fold",
            "probability",
            "rank",
            "selected",
            "target_5d",
            "future_max_return",
            "turnover_mean_20",
        ]
        trade_columns = prediction_columns + [
            "entry_price",
            "exit_price",
            "exit_date",
            "target_hit_day",
            "trade_gross_return",
            "trade_net_return",
        ]
        return BacktestResult(
            predictions=scored[prediction_columns].sort_values(["date", "rank"]),
            trades=trades[trade_columns].sort_values(["date", "rank"]),
            equity_curve=equity_curve,
            summary=summary,
        )

    def _eligible_rows(self, frame: pd.DataFrame, *, require_outcome: bool) -> pd.DataFrame:
        eligible = frame.loc[frame["turnover_mean_20"].fillna(0) >= self.config.min_turnover].copy()
        if require_outcome:
            eligible = eligible.loc[eligible["trade_outcome_available"].fillna(False)]
        return eligible

    def _new_model(self) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=self.config.random_state,
                    ),
                ),
            ]
        )

    def _equity_curve(self, trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return pd.DataFrame(
                columns=[
                    "exit_date",
                    "cohort_return",
                    "portfolio_return",
                    "equity",
                    "drawdown",
                ]
            )
        curve = (
            trades.groupby("exit_date", as_index=False)["trade_net_return"]
            .mean()
            .rename(columns={"trade_net_return": "cohort_return"})
            .sort_values("exit_date")
        )
        # 毎営業日に資金の1/horizonを新規シグナルへ配分する近似。
        curve["portfolio_return"] = curve["cohort_return"] / self.config.horizon_days
        curve["equity"] = (1.0 + curve["portfolio_return"]).cumprod()
        curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1.0
        return curve

    def _summary(
        self,
        scored: pd.DataFrame,
        trades: pd.DataFrame,
        equity_curve: pd.DataFrame,
    ) -> dict[str, Any]:
        selected = scored.loc[scored["selected"]]
        base_rate = float(scored["target_5d"].astype(float).mean())
        precision = float(selected["target_5d"].astype(float).mean())
        daily_precision = selected.groupby("date")["target_5d"].mean().astype(float)
        target = scored["target_5d"].astype("int8")
        rank_correlations = [
            group["probability"].corr(group["future_max_return"], method="spearman")
            for _, group in scored.groupby("date")
        ]
        return {
            "config": {
                key: value.isoformat() if isinstance(value, date) else value
                for key, value in asdict(self.config).items()
            },
            "feature_columns": FEATURE_COLUMNS,
            "test_start": min(scored["date"]).isoformat(),
            "test_end": max(scored["date"]).isoformat(),
            "test_days": int(scored["date"].nunique()),
            "folds": int(scored["fold"].nunique()),
            "eligible_predictions": len(scored),
            "selected_signals": len(selected),
            "base_positive_rate": base_rate,
            "roc_auc": float(roc_auc_score(target, scored["probability"]))
            if target.nunique() > 1
            else None,
            "average_precision": float(average_precision_score(target, scored["probability"]))
            if target.nunique() > 1
            else None,
            "precision_at_n": precision,
            "mean_daily_precision_at_n": float(daily_precision.mean()),
            "lift_at_n": precision / base_rate if base_rate > 0 else None,
            "mean_daily_rank_ic": float(pd.Series(rank_correlations, dtype="float64").mean()),
            "mean_future_max_return": float(selected["future_max_return"].mean()),
            "completed_trades": len(trades),
            "target_hit_rate": float(trades["target_hit_day"].notna().mean())
            if not trades.empty
            else None,
            "mean_trade_net_return": float(trades["trade_net_return"].mean())
            if not trades.empty
            else None,
            "median_trade_net_return": float(trades["trade_net_return"].median())
            if not trades.empty
            else None,
            "trade_win_rate": float((trades["trade_net_return"] > 0).mean())
            if not trades.empty
            else None,
            "ending_equity": float(equity_curve["equity"].iloc[-1])
            if not equity_curve.empty
            else None,
            "max_drawdown": float(equity_curve["drawdown"].min())
            if not equity_curve.empty
            else None,
        }
