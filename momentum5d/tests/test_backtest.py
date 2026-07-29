from __future__ import annotations

from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from app.modeling.backtest import BacktestConfig, WalkForwardBacktester
from app.modeling.features import FEATURE_COLUMNS
from app.modeling.store import BacktestStore
from app.storage.catalog import DuckDBCatalog


def make_modeling_dataset(days: int = 100, codes: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-06", periods=days)
    records: list[dict[str, object]] = []
    for date_index, current_date in enumerate(dates):
        for code_index in range(codes):
            signal = code_index / codes
            row: dict[str, object] = {
                "date": current_date.date(),
                "code": f"{13000 + code_index}",
                "target_5d": int(signal >= 0.8),
                "future_max_return": 0.08 if signal >= 0.8 else 0.01,
                "turnover_mean_20": 100_000_000.0,
                "trade_outcome_available": True,
                "trade_gross_return": 0.05 if signal >= 0.8 else -0.02,
                "entry_price": 100.0,
                "exit_price": 105.0 if signal >= 0.8 else 98.0,
                "exit_date": current_date.date() + timedelta(days=7),
                "target_hit_day": 2 if signal >= 0.8 else pd.NA,
            }
            for feature_index, feature in enumerate(FEATURE_COLUMNS):
                row[feature] = (
                    signal if feature_index == 0 else rng.normal(0, 0.1) + date_index * 0.0001
                )
            records.append(row)
    return pd.DataFrame(records)


def test_walk_forward_backtest_ranks_predictive_signal() -> None:
    config = BacktestConfig(
        min_train_days=40,
        retrain_every_days=10,
        horizon_days=5,
        top_n=5,
        min_turnover=0,
        transaction_cost_bps=20,
    )

    result = WalkForwardBacktester(config).run(make_modeling_dataset())

    assert result.summary["folds"] >= 2
    assert result.summary["precision_at_n"] > result.summary["base_positive_rate"]
    assert result.summary["lift_at_n"] > 1
    assert result.summary["completed_trades"] == len(result.trades)
    assert result.summary["ending_equity"] > 1
    assert result.predictions.groupby("date")["selected"].sum().eq(5).all()


def test_backtest_respects_requested_test_dates() -> None:
    config = BacktestConfig(
        start=date(2025, 4, 1),
        end=date(2025, 4, 30),
        min_train_days=40,
        retrain_every_days=5,
        top_n=3,
        min_turnover=0,
    )

    result = WalkForwardBacktester(config).run(make_modeling_dataset())

    assert result.predictions["date"].min() >= config.start
    assert result.predictions["date"].max() <= config.end


def test_backtest_store_writes_all_artifacts(tmp_path) -> None:
    config = BacktestConfig(
        min_train_days=40,
        retrain_every_days=20,
        top_n=5,
        min_turnover=0,
    )
    result = WalkForwardBacktester(config).run(make_modeling_dataset())

    paths = BacktestStore(
        tmp_path / "processed",
        tmp_path / "metadata",
    ).save(result)

    assert all(path.exists() for path in paths.values())
    assert len(pd.read_parquet(paths["trades"])) == len(result.trades)

    catalog = DuckDBCatalog(
        tmp_path / "metadata" / "market.duckdb",
        tmp_path / "processed",
    )
    catalog.refresh()

    with duckdb.connect(str(catalog.database_path), read_only=True) as connection:
        saved_trades = connection.execute("SELECT COUNT(*) FROM backtest_trades").fetchone()[0]
    assert saved_trades == len(result.trades)
