from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import Settings
from app.yahoo.analysis import YahooPatternAnalyzer
from app.yahoo.corporate_actions import normalize_split_adjusted_prices
from app.yahoo.ingestion import YahooPaths
from app.yahoo.retail_flow import RETAIL_STAGE_COLUMNS, add_retail_flow_features
from app.yahoo.trend import add_trend_features, load_sector_map


@dataclass(frozen=True)
class SakataBacktestConfig:
    start: date = date(2026, 4, 1)
    end: date | None = None
    initial_capital: float = 1_000_000.0
    horizon_days: int = 5
    take_profit: float = 0.05
    top_n: int = 10
    min_turnover: float = 10_000_000.0
    gap_down_limit: float = -0.03
    gap_up_limit: float = 0.015
    transaction_cost_bps: float = 20.0


class SakataBacktester:
    """資金流入観測、個人投資家フロー、酒田五法、従来方式を比較する。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.paths = YahooPaths(settings.data_dir / "yahoo")

    def run(
        self,
        config: SakataBacktestConfig,
        *,
        output_dir: Path,
    ) -> dict[str, Any]:
        prices = pd.read_parquet(self.paths.prices_path)
        valid = prices.loc[
            (pd.to_numeric(prices["volume"], errors="coerce") > 0)
            & prices[["open", "high", "low", "close", "adjusted_close"]]
            .apply(pd.to_numeric, errors="coerce")
            .gt(0)
            .all(axis=1)
        ].copy()
        features = YahooPatternAnalyzer._build_features(
            normalize_split_adjusted_prices(valid)
        )
        sectors = load_sector_map(
            self.settings.data_dir.parent / "config" / "prime_sectors.csv"
        )
        features = add_trend_features(features, sectors)
        features = add_retail_flow_features(features)
        evaluated = _attach_outcomes(features, config)
        eligible_end = evaluated.loc[evaluated["trade_available"], "date"].max()
        if pd.isna(eligible_end):
            raise RuntimeError("5営業日先まで評価できるデータがありません")
        end = min(config.end, eligible_end) if config.end else eligible_end
        if end < config.start:
            raise ValueError("バックテスト終了日は開始日以降にしてください")

        trading_dates = sorted(
            evaluated.loc[
                evaluated["date"].between(config.start, end), "date"
            ].drop_duplicates()
        )
        summaries: dict[str, Any] = {}
        all_trades: list[pd.DataFrame] = []
        all_equity: list[pd.DataFrame] = []
        strategies = (
            "observed_inflow",
            "legacy_setup",
            "sakata_five_methods",
            "retail_attention_flow",
            "retail_attention_hybrid",
        )
        for strategy in strategies:
            selected = _select_candidates(
                evaluated,
                config=config,
                start=config.start,
                end=end,
                strategy=strategy,
            )
            equity = _portfolio(selected, trading_dates, config)
            summaries[strategy] = _summarize(selected, equity, config)
            selected["strategy"] = strategy
            equity["strategy"] = strategy
            all_trades.append(selected)
            all_equity.append(equity)

        trades_by_strategy = {
            strategy: trades for strategy, trades in zip(strategies, all_trades, strict=True)
        }
        sakata_trades = trades_by_strategy["sakata_five_methods"]
        retail_trades = trades_by_strategy["retail_attention_hybrid"]
        observed_trades = trades_by_strategy["observed_inflow"]
        summary = {
            "technical_method": "observed_inflow_v1",
            "period": {"start": str(config.start), "end": str(end)},
            "rules": {
                "entry": "シグナル翌営業日始値",
                "take_profit": config.take_profit,
                "time_exit": f"{config.horizon_days}営業日目終値",
                "stop_loss": None,
                "gap_filter": [config.gap_down_limit, config.gap_up_limit],
                "transaction_cost_bps": config.transaction_cost_bps,
                "top_n_per_day": config.top_n,
                "capital_model": f"資金を{config.horizon_days}スリーブへ等分",
            },
            "strategies": summaries,
            "sakata_pattern_analysis": _pattern_analysis(sakata_trades),
            "retail_stage_analysis": _retail_stage_analysis(retail_trades),
            "observed_inflow_analysis": _observed_inflow_analysis(observed_trades),
        }
        _save_results(
            output_dir,
            summary,
            pd.concat(all_trades, ignore_index=True),
            pd.concat(all_equity, ignore_index=True),
        )
        return summary


def _attach_outcomes(frame: pd.DataFrame, config: SakataBacktestConfig) -> pd.DataFrame:
    result = frame.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    trading_dates = sorted(result["date"].drop_duplicates())
    date_position = {value: index for index, value in enumerate(trading_dates)}
    result["calendar_position"] = result["date"].map(date_position)
    group = result.groupby("ticker", sort=False)
    result["entry_price"] = group["open"].shift(-1)
    result["entry_date"] = group["date"].shift(-1)
    result["entry_gap"] = result["entry_price"] / result["close"] - 1
    complete = result["entry_price"].gt(0)
    for offset in range(1, config.horizon_days + 1):
        result[f"future_date_{offset}"] = group["date"].shift(-offset)
        result[f"future_high_{offset}"] = group["high"].shift(-offset)
        result[f"future_low_{offset}"] = group["low"].shift(-offset)
        result[f"future_close_{offset}"] = group["close"].shift(-offset)
        future_position = group["calendar_position"].shift(-offset)
        complete &= future_position.eq(result["calendar_position"] + offset)
        complete &= result[f"future_high_{offset}"].notna()
    result["trade_available"] = complete

    target_price = result["entry_price"] * (1 + config.take_profit)
    hit_day = pd.Series(pd.NA, index=result.index, dtype="Int8")
    for offset in range(1, config.horizon_days + 1):
        hit = (
            hit_day.isna()
            & result["trade_available"]
            & result[f"future_high_{offset}"].ge(target_price)
        )
        hit_day.loc[hit] = offset
    result["target_hit_day"] = hit_day
    result["target_hit"] = hit_day.notna()
    result["exit_price"] = target_price.where(
        result["target_hit"], result[f"future_close_{config.horizon_days}"]
    )
    result["exit_date"] = result[f"future_date_{config.horizon_days}"]
    for offset in range(1, config.horizon_days + 1):
        mask = result["target_hit_day"].eq(offset)
        result.loc[mask, "exit_date"] = result.loc[mask, f"future_date_{offset}"]
    result["gross_return"] = result["exit_price"] / result["entry_price"] - 1
    result["net_return"] = (
        result["gross_return"] - config.transaction_cost_bps / 10_000
    )
    result["trade_win"] = result["net_return"] > 0
    future_highs = result[
        [f"future_high_{offset}" for offset in range(1, config.horizon_days + 1)]
    ]
    future_lows = result[
        [f"future_low_{offset}" for offset in range(1, config.horizon_days + 1)]
    ]
    result["max_favorable_excursion"] = future_highs.max(axis=1) / result["entry_price"] - 1
    result["max_adverse_excursion"] = future_lows.min(axis=1) / result["entry_price"] - 1
    return result


def _market_favorable(frame: pd.DataFrame) -> pd.Series:
    regime = frame.groupby("date", sort=False).agg(
        breadth_5d=("return_5d", lambda values: float((values > 0).mean())),
        median_return_20d=("return_20d", "median"),
    )
    favorable = (regime["breadth_5d"] > 0.50) & (regime["median_return_20d"] > 0)
    return frame["date"].map(favorable).fillna(False)


def _select_candidates(
    frame: pd.DataFrame,
    *,
    config: SakataBacktestConfig,
    start: date,
    end: date,
    strategy: str,
) -> pd.DataFrame:
    common = (
        frame["date"].between(start, end)
        & frame["trade_available"]
        & frame["entry_gap"].between(config.gap_down_limit, config.gap_up_limit)
        & frame["turnover_value"].ge(config.min_turnover)
    )
    favorable_market = _market_favorable(frame)
    if strategy == "observed_inflow":
        mask = (
            common
            & frame["observed_inflow_confirmed"].fillna(False).astype(bool)
            & frame["return_1d"].between(0.002, 0.10)
            & frame["return_5d"].between(-0.05, 0.18)
            & frame["rsi_14"].le(82.0)
        )
        score = frame["observed_inflow_score"]
    elif strategy == "sakata_five_methods":
        mask = (
            common
            & favorable_market
            & frame["sakata_buy_signal"]
            & ~frame["sakata_sell_signal"]
            & frame["sakata_score"].ge(0.65)
            & frame["return_5d"].between(-0.30, 0.15)
            & frame["return_20d"].between(-0.35, 0.30)
            & frame["rsi_14"].le(80)
        )
        score = frame["trend_ranking_score"]
    elif strategy == "legacy_setup":
        mask = (
            common
            & favorable_market
            & frame["legacy_setup_score"].ge(0.55)
            & frame["return_1d"].between(-0.03, 0.025)
            & frame["return_5d"].between(-0.03, 0.04)
            & frame["return_20d"].between(-0.08, 0.15)
            & frame["breakout_20d"].between(-0.10, 0.0)
            & frame["close_to_ma20"].between(-0.04, 0.08)
            & frame["volatility_10d"].between(0.012, 0.035)
            & frame["volume_ratio_5_20"].between(0.85, 1.65)
            & frame["up_volume_share_10d"].ge(0.48)
        )
        sector_score = frame["sector_17_trend_score"].fillna(frame["legacy_setup_score"])
        score = 0.75 * frame["legacy_setup_score"] + 0.25 * sector_score
    elif strategy == "retail_attention_flow":
        local_rotation = (
            frame["sector_17_trend_score"].ge(0.60)
            & frame["sector_17_breadth_5d"].ge(0.50)
        )
        attention_breakout = (
            frame["retail_discovery_score"].ge(0.82)
            & frame["retail_action_score"].ge(0.72)
        )
        mask = (
            common
            & (favorable_market | local_rotation | attention_breakout)
            & frame["retail_flow_score"].ge(0.72)
            & frame["retail_discovery_score"].ge(0.72)
            & frame["retail_attention_acceleration_score"].ge(0.55)
            & frame["retail_understanding_proxy_score"].ge(0.45)
            & frame["retail_expectation_score"].ge(0.55)
            & frame["retail_safety_score"].ge(0.50)
            & frame["retail_action_score"].ge(0.65)
            & frame["retail_overheat_penalty"].le(0.35)
            & frame["retail_loss_anxiety_penalty"].le(0.55)
            & frame["return_1d"].between(-0.06, 0.06)
            & frame["return_5d"].between(-0.10, 0.12)
            & frame["return_20d"].between(-0.20, 0.30)
            & frame["rsi_14"].le(85)
        )
        score = frame["retail_flow_score"]
    elif strategy == "retail_attention_hybrid":
        mask = (
            common
            & favorable_market
            & frame["legacy_setup_score"].ge(0.55)
            & frame["return_1d"].between(-0.03, 0.025)
            & frame["return_5d"].between(-0.03, 0.04)
            & frame["return_20d"].between(-0.08, 0.15)
            & frame["breakout_20d"].between(-0.10, 0.0)
            & frame["close_to_ma20"].between(-0.04, 0.08)
            & frame["volatility_10d"].between(0.012, 0.035)
            & frame["volume_ratio_5_20"].between(0.85, 1.65)
            & frame["up_volume_share_10d"].ge(0.48)
        )
        score = frame["retail_attention_hybrid_score"]
    else:
        raise ValueError(f"未知の戦略です: {strategy}")
    selected = frame.loc[mask].copy()
    selected["ranking_score"] = score.loc[selected.index]
    selected["rank"] = selected.groupby("date")["ranking_score"].rank(
        method="first", ascending=False
    )
    selected = selected.loc[selected["rank"] <= config.top_n].copy()
    columns = [
        "date", "entry_date", "exit_date", "ticker", "code", "rank",
        "ranking_score", "sakata_pattern", "sakata_score", "legacy_setup_score",
        "return_1d", "return_5d", "return_20d", "intraday_return", "rsi_14",
        "volume_ratio_5_20", "up_volume_share_10d",
        "retail_flow_score", *RETAIL_STAGE_COLUMNS, "retail_overheat_penalty",
        "retail_loss_anxiety_penalty",
        "volume_ratio_1_20", "turnover_ratio_1_20",
        "observed_volume_ratio_rank", "observed_turnover_ratio_rank",
        "observed_price_confirmation_score", "observed_inflow_score",
        "observed_inflow_confirmed",
        "sector_17_name", "sector_17_trend_score", "entry_price", "exit_price",
        "entry_gap", "target_hit_day", "target_hit", "gross_return", "net_return",
        "trade_win", "max_favorable_excursion", "max_adverse_excursion",
    ]
    return selected[columns].sort_values(["date", "rank"]).reset_index(drop=True)


def _portfolio(
    trades: pd.DataFrame,
    signal_dates: list[date],
    config: SakataBacktestConfig,
) -> pd.DataFrame:
    sleeves = [config.initial_capital / config.horizon_days] * config.horizon_days
    rows = [{"date": config.start, "capital": config.initial_capital, "drawdown": 0.0}]
    peak = config.initial_capital
    for index, signal_date in enumerate(signal_dates):
        selected = trades.loc[trades["date"] == signal_date]
        cohort_return = float(selected["net_return"].mean()) if not selected.empty else 0.0
        sleeve = index % config.horizon_days
        sleeves[sleeve] *= 1 + cohort_return
        capital = sum(sleeves)
        peak = max(peak, capital)
        rows.append(
            {
                "date": signal_date,
                "capital": capital,
                "drawdown": capital / peak - 1,
                "positions": len(selected),
                "cohort_return": cohort_return,
            }
        )
    return pd.DataFrame(rows)


def _summarize(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    config: SakataBacktestConfig,
) -> dict[str, Any]:
    ending = float(equity.iloc[-1]["capital"])
    return {
        "initial_capital": config.initial_capital,
        "ending_capital": ending,
        "profit": ending - config.initial_capital,
        "total_return": ending / config.initial_capital - 1,
        "max_drawdown": float(equity["drawdown"].min()),
        "selected_signals": len(trades),
        "entry_days": int(trades["date"].nunique()) if not trades.empty else 0,
        "target_hit_rate": _mean_or_none(trades, "target_hit"),
        "win_rate": _mean_or_none(trades, "trade_win"),
        "mean_net_return": _mean_or_none(trades, "net_return"),
        "median_net_return": _median_or_none(trades, "net_return"),
    }


def _pattern_analysis(trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    expanded = trades.assign(pattern=trades["sakata_pattern"].str.split("・")).explode("pattern")
    grouped = expanded.groupby("pattern", dropna=False)
    rows = []
    for pattern, values in grouped:
        rows.append(
            {
                "pattern": pattern,
                "signals": len(values),
                "target_hit_rate": float(values["target_hit"].mean()),
                "win_rate": float(values["trade_win"].mean()),
                "mean_net_return": float(values["net_return"].mean()),
            }
        )
    return sorted(rows, key=lambda row: (-row["signals"], row["pattern"]))


def _retail_stage_analysis(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {"signals": 0, "winning_trade_medians": {}, "losing_trade_medians": {}}
    winners = trades["trade_win"]
    return {
        "signals": len(trades),
        "winning_trade_medians": {
            column: _median_or_none(trades.loc[winners], column)
            for column in RETAIL_STAGE_COLUMNS
        },
        "losing_trade_medians": {
            column: _median_or_none(trades.loc[~winners], column)
            for column in RETAIL_STAGE_COLUMNS
        },
    }


def _observed_inflow_analysis(trades: pd.DataFrame) -> dict[str, Any]:
    columns = (
        "volume_ratio_1_20",
        "turnover_ratio_1_20",
        "observed_price_confirmation_score",
        "observed_inflow_score",
    )
    if trades.empty:
        return {"signals": 0, "winning_trade_medians": {}, "losing_trade_medians": {}}
    winners = trades["trade_win"]
    return {
        "signals": len(trades),
        "winning_trade_medians": {
            column: _median_or_none(trades.loc[winners], column) for column in columns
        },
        "losing_trade_medians": {
            column: _median_or_none(trades.loc[~winners], column) for column in columns
        },
    }


def _mean_or_none(frame: pd.DataFrame, column: str) -> float | None:
    return None if frame.empty else float(frame[column].mean())


def _median_or_none(frame: pd.DataFrame, column: str) -> float | None:
    return None if frame.empty else float(frame[column].median())


def _save_results(
    output_dir: Path,
    summary: dict[str, Any],
    trades: pd.DataFrame,
    equity: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "retail_flow_backtest_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, summary_path)
    serialized_summary = summary_path.read_text(encoding="utf-8")
    (output_dir / "sakata_backtest_summary.json").write_text(
        serialized_summary,
        encoding="utf-8",
    )
    for prefix in ("retail_flow", "sakata"):
        trades.to_csv(
            output_dir / f"{prefix}_backtest_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )
        equity.to_csv(
            output_dir / f"{prefix}_backtest_equity.csv",
            index=False,
            encoding="utf-8-sig",
        )
    strategies = summary["strategies"]
    report = [
        "# 資金流入観測方式 比較バックテスト",
        "",
        f"期間: {summary['period']['start']}〜{summary['period']['end']}",
        "",
        "| 方式 | 最終資金 | 収益率 | 最大DD | シグナル | +5%到達率 | 勝率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in (
        "observed_inflow",
        "legacy_setup",
        "sakata_five_methods",
        "retail_attention_flow",
        "retail_attention_hybrid",
    ):
        values = strategies[key]
        report.append(
            f"| {key} | ¥{values['ending_capital']:,.0f} | "
            f"{values['total_return']:.2%} | {values['max_drawdown']:.2%} | "
            f"{values['selected_signals']} | {_percent(values['target_hit_rate'])} | "
            f"{_percent(values['win_rate'])} |"
        )
    report_text = "\n".join(report) + "\n"
    for prefix in ("retail_flow", "sakata"):
        (output_dir / f"{prefix}_backtest_report.md").write_text(
            report_text,
            encoding="utf-8",
        )


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value:.2%}"
