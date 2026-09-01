from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from app.live_strategy import FROZEN_STRATEGY_SHA256, load_frozen_strategy

POLICIES = {
    "A_only": {"A"},
    "A_plus_B": {"A", "B"},
    "A_plus_B_plus_C": {"A", "B", "C"},
    "all_primary": {"A", "B", "C", "SKIP"},
}
MIN_TOTAL_SAMPLE = 50
TARGET_SAMPLE = 100
OUTCOME_DEFAULTS: dict[str, Any] = {
    "status": None,
    "entry_rule_eligible": False,
    "strategy_net_return": None,
    "strategy_entry_price": None,
    "strategy_exit_price": None,
    "strategy_exit_date": None,
    "entry_date": None,
    "hit_plus_5pct": False,
    "hit_minus_5pct": False,
    "hit_minus_8pct": False,
    "maximum_favorable_excursion_official_horizon": None,
    "daily_mark_path": None,
}


def _finite(value: float | None) -> float | None:
    return float(value) if value is not None and math.isfinite(value) else None


def _wilson(successes: int, count: int) -> list[float] | None:
    if count == 0:
        return None
    z = 1.959963984540054
    probability = successes / count
    denominator = 1 + z * z / count
    centre = (probability + z * z / (2 * count)) / denominator
    margin = z * math.sqrt(
        probability * (1 - probability) / count + z * z / (4 * count * count)
    ) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def _mean_ci(values: pd.Series) -> list[float] | None:
    if len(values) < 2:
        return None
    margin = 1.959963984540054 * float(values.std(ddof=1)) / math.sqrt(len(values))
    mean = float(values.mean())
    return [mean - margin, mean + margin]


def _trade_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    equity = (1.0 + values).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def _maximum_losing_streak(values: pd.Series) -> int:
    maximum = current = 0
    for losing in values.lt(0):
        current = current + 1 if losing else 0
        maximum = max(maximum, current)
    return maximum


def _sample_status(count: int) -> str:
    if count < 20:
        return "insufficient"
    if count < MIN_TOTAL_SAMPLE:
        return "limited"
    if count < TARGET_SAMPLE:
        return "provisional"
    return "adequate"


def trade_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    completed = rows.loc[
        rows["status"].eq("completed")
        & rows["entry_rule_eligible"].eq(True)  # noqa: E712
        & rows["strategy_net_return"].notna()
    ].sort_values(["entry_date", "primary_rank", "ticker"])
    values = completed["strategy_net_return"].astype(float)
    gains = float(values.loc[values.gt(0)].sum())
    losses = float(-values.loc[values.lt(0)].sum())
    wins = int(values.gt(0).sum())
    return {
        "decisions": int(len(rows)),
        "completed_entry_eligible": int(len(completed)),
        "pending_or_entry_excluded": int(len(rows) - len(completed)),
        "sample_status": _sample_status(len(completed)),
        "win_rate": _finite(float(values.gt(0).mean())) if len(values) else None,
        "win_rate_95pct_ci": _wilson(wins, len(values)),
        "mean_net_return": _finite(float(values.mean())) if len(values) else None,
        "mean_net_return_95pct_ci": _mean_ci(values),
        "median_net_return": _finite(float(values.median())) if len(values) else None,
        "expected_value_per_trade": _finite(float(values.mean()))
        if len(values)
        else None,
        "profit_factor": _finite(gains / losses) if losses else None,
        "trade_sequence_maximum_drawdown": _trade_drawdown(values),
        "maximum_losing_streak": _maximum_losing_streak(values),
        "hit_plus_5pct_rate": _finite(float(completed["hit_plus_5pct"].mean()))
        if len(completed)
        else None,
        "hit_minus_5pct_rate": _finite(float(completed["hit_minus_5pct"].mean()))
        if len(completed)
        else None,
        "hit_minus_8pct_rate": _finite(float(completed["hit_minus_8pct"].mean()))
        if len(completed)
        else None,
    }


def _simulate_portfolio(rows: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    eligible = rows.loc[
        rows["status"].eq("completed")
        & rows["entry_rule_eligible"].eq(True)  # noqa: E712
        & rows["strategy_exit_date"].notna()
    ].copy()
    initial = float(spec["portfolio"]["initial_capital_yen"])
    if eligible.empty:
        return {
            "starting_equity_yen": initial,
            "ending_equity_yen": initial,
            "total_profit_yen": 0.0,
            "total_return": 0.0,
            "maximum_drawdown": 0.0,
            "completed_trades": 0,
            "mean_gross_exposure": 0.0,
            "invested_session_ratio": 0.0,
            "capital_efficiency": None,
            "profitable_trade_profit_yen": 0.0,
            "losing_trade_loss_yen": 0.0,
        }
    one_way_cost = float(spec["portfolio"]["round_trip_transaction_cost_bps"]) / 20_000
    lot = int(spec["portfolio"]["lot_size"])
    max_positions = int(spec["portfolio"]["maximum_open_positions"])
    max_new = int(spec["portfolio"]["maximum_new_positions_per_day"])
    entries: dict[str, list[dict[str, Any]]] = {}
    marks: dict[tuple[str, str], float] = {}
    dates: set[str] = set()
    for record in eligible.to_dict("records"):
        entries.setdefault(str(record["entry_date"]), []).append(record)
        for mark in record.get("daily_mark_path") or []:
            day = str(mark["date"])
            dates.add(day)
            marks[(day, str(record["ticker"]))] = float(mark["close"])
    cash = initial
    positions: dict[str, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    curve: list[float] = []
    exposures: list[float] = []
    for day in sorted(dates):
        exiting = [
            ticker
            for ticker, item in positions.items()
            if item["strategy_exit_date"] == day
        ]
        for ticker in exiting:
            item = positions.pop(ticker)
            proceeds = item["shares"] * float(item["strategy_exit_price"]) * (1 - one_way_cost)
            cash += proceeds
            completed.append({**item, "net_profit_yen": proceeds - item["cash_out"]})
        new_count = 0
        ordered_entries = sorted(
            entries.get(day, []),
            key=lambda item: (item["primary_rank"], item["ticker"]),
        )
        for record in ordered_entries:
            ticker = str(record["ticker"])
            if new_count >= max_new or len(positions) >= max_positions or ticker in positions:
                continue
            slots = max_positions - len(positions)
            unit = float(record["strategy_entry_price"]) * lot * (1 + one_way_cost)
            lots = int(np.floor((cash / slots) / unit))
            if lots < 1:
                continue
            shares = lots * lot
            cash_out = shares * float(record["strategy_entry_price"]) * (1 + one_way_cost)
            cash -= cash_out
            positions[ticker] = {**record, "shares": shares, "cash_out": cash_out}
            new_count += 1
        market_value = sum(
            item["shares"] * marks.get((day, ticker), float(item["strategy_entry_price"]))
            for ticker, item in positions.items()
        )
        equity = cash + market_value
        curve.append(equity)
        exposures.append(market_value / equity if equity else 0.0)
    ending = curve[-1] if curve else initial
    equity_series = pd.Series(curve, dtype=float)
    maximum_drawdown = float((equity_series / equity_series.cummax() - 1).min())
    mean_exposure = float(np.mean(exposures)) if exposures else 0.0
    total_return = ending / initial - 1.0
    return {
        "starting_equity_yen": initial,
        "ending_equity_yen": ending,
        "total_profit_yen": ending - initial,
        "total_return": total_return,
        "maximum_drawdown": maximum_drawdown,
        "completed_trades": len(completed),
        "mean_gross_exposure": mean_exposure,
        "invested_session_ratio": float(np.mean(np.array(exposures) > 0)),
        "capital_efficiency": total_return / mean_exposure if mean_exposure else None,
        "profitable_trade_profit_yen": float(
            sum(max(0.0, item["net_profit_yen"]) for item in completed)
        ),
        "losing_trade_loss_yen": float(
            sum(max(0.0, -item["net_profit_yen"]) for item in completed)
        ),
    }


def _opportunity(
    rows: pd.DataFrame,
    selected: set[str],
    baseline_trades: int,
    spec: dict[str, Any],
) -> dict[str, Any]:
    excluded = rows.loc[
        ~rows["classification"].isin(selected)
        & rows["status"].eq("completed")
        & rows["entry_rule_eligible"].eq(True)  # noqa: E712
        & rows["strategy_net_return"].notna()
    ]
    returns = excluded["strategy_net_return"].astype(float)
    excluded_portfolio = _simulate_portfolio(excluded, spec)
    return {
        "excluded_completed_trades": int(len(excluded)),
        "missed_plus_5pct_candidates": int(excluded["hit_plus_5pct"].sum()),
        "missed_plus_10pct_candidates": int(
            excluded["maximum_favorable_excursion_official_horizon"].ge(0.10).sum()
        ),
        "missed_plus_20pct_candidates": int(
            excluded["maximum_favorable_excursion_official_horizon"].ge(0.20).sum()
        ),
        "excluded_profitable_return_sum": float(returns.loc[returns.gt(0)].sum()),
        "avoided_loss_return_sum": float(-returns.loc[returns.lt(0)].sum()),
        "excluded_profit_yen": excluded_portfolio["profitable_trade_profit_yen"],
        "avoided_loss_yen": excluded_portfolio["losing_trade_loss_yen"],
        "trade_opportunity_retained": (baseline_trades - len(excluded)) / baseline_trades
        if baseline_trades
        else None,
    }


def _verdict(policy_results: dict[str, dict[str, Any]]) -> dict[str, str]:
    baseline = policy_results["all_primary"]
    if baseline["trade_metrics"]["completed_entry_eligible"] < MIN_TOTAL_SAMPLE:
        return {
            "decision": "サンプル不足で判断保留",
            "reason": f"一次シグナル完了件数が{MIN_TOTAL_SAMPLE}件未満",
        }
    qualifying: list[str] = []
    for name in ("A_only", "A_plus_B", "A_plus_B_plus_C"):
        result = policy_results[name]
        if result["trade_metrics"]["completed_entry_eligible"] < MIN_TOTAL_SAMPLE:
            continue
        expectation = result["trade_metrics"]["expected_value_per_trade"]
        baseline_expectation = baseline["trade_metrics"]["expected_value_per_trade"]
        if (
            expectation is not None
            and baseline_expectation is not None
            and expectation >= baseline_expectation
            and result["portfolio"]["maximum_drawdown"]
            > baseline["portfolio"]["maximum_drawdown"]
        ):
            qualifying.append(name)
    if "A_only" in qualifying:
        return {"decision": "A/B/C選別を採用", "reason": "期待値を維持し最大DDを低減"}
    if qualifying:
        return {"decision": "一部のみ採用", "reason": f"条件を満たした比較={qualifying}"}
    alternatives = [policy_results[name] for name in POLICIES if name != "all_primary"]
    if all(
        item["trade_metrics"]["expected_value_per_trade"] is None
        or item["trade_metrics"]["expected_value_per_trade"]
        < baseline["trade_metrics"]["expected_value_per_trade"]
        for item in alternatives
    ):
        return {"decision": "一次シグナル全件の方が優秀", "reason": "選別後の期待値が基準未満"}
    return {
        "decision": "サンプル不足で判断保留",
        "reason": "期待値維持と最大DD低減を同時確認できない",
    }


def evaluate_abc_effectiveness(
    decisions: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    git_commit: str,
    calculated_at: str | None = None,
) -> dict[str, Any]:
    spec = load_frozen_strategy()
    stamp = calculated_at or datetime.now(UTC).isoformat()
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "collecting" if decisions.empty else "completed",
        "calculation_executed_at": stamp,
        "git_commit_id": git_commit,
        "strategy_version": spec["strategy_version"],
        "strategy_config_sha256": FROZEN_STRATEGY_SHA256,
        "evaluation_policy": {
            "no_hindsight_classification": True,
            "same_frozen_exit_rules_for_all_groups": True,
            "minimum_completed_sample": MIN_TOTAL_SAMPLE,
            "target_completed_sample": TARGET_SAMPLE,
            "adoption_rule": "expectancy_not_lower_and_portfolio_maximum_drawdown_lower",
        },
        "strategies": {},
    }
    if decisions.empty:
        for signal_type in spec["signals"]:
            result["strategies"][signal_type] = {
                "status": "no_forward_abc_records",
                "verdict": {"decision": "サンプル不足で判断保留", "reason": "A/B/C判定記録が0件"},
            }
        return result
    audit_fields = {
        "evaluated_at",
        "information_cutoff_at",
        "entry_session_open_at",
        "decision_context_sha256",
    }
    missing_audit = sorted(audit_fields.difference(decisions.columns))
    if not missing_audit:
        missing_audit = sorted(
            field for field in audit_fields if decisions[field].isna().any()
        )
    if missing_audit:
        result["status"] = "unauditable_decision_records"
        result["audit_error"] = f"Missing mandatory no-hindsight fields: {missing_audit}"
        for signal_type in spec["signals"]:
            result["strategies"][signal_type] = {
                "status": "unauditable_decision_records",
                "verdict": {
                    "decision": "サンプル不足で判断保留",
                    "reason": "判定時点情報の監査項目が不足",
                },
            }
        return result
    outcomes = outcomes.copy()
    for key in ("signal_date", "ticker", "signal_type"):
        if key not in outcomes:
            outcomes[key] = pd.Series(dtype="object")
    for key, default in OUTCOME_DEFAULTS.items():
        if key not in outcomes:
            outcomes[key] = default
    merged = decisions.merge(
        outcomes,
        on=["signal_date", "ticker", "signal_type"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_outcome"),
    )
    for signal_type in spec["signals"]:
        strategy_rows = merged.loc[merged["signal_type"].eq(signal_type)].copy()
        by_classification = {
            label: trade_metrics(strategy_rows.loc[strategy_rows["classification"].eq(label)])
            for label in sorted(CLASSIFICATIONS_PRESENT(strategy_rows))
        }
        policy_results: dict[str, dict[str, Any]] = {}
        baseline_trade_count = trade_metrics(strategy_rows)["completed_entry_eligible"]
        for name, labels in POLICIES.items():
            selected_rows = strategy_rows.loc[strategy_rows["classification"].isin(labels)]
            portfolio = _simulate_portfolio(selected_rows, spec)
            policy_results[name] = {
                "included_classifications": sorted(labels),
                "trade_metrics": trade_metrics(selected_rows),
                "portfolio": portfolio,
                "opportunity_cost": _opportunity(
                    strategy_rows, labels, baseline_trade_count, spec
                ),
            }
        baseline_count = policy_results["all_primary"]["trade_metrics"]["completed_entry_eligible"]
        for item in policy_results.values():
            count = item["trade_metrics"]["completed_entry_eligible"]
            item["trade_opportunity_reduction"] = (
                1 - count / baseline_count if baseline_count else None
            )
        result["strategies"][signal_type] = {
            "status": "completed" if len(strategy_rows) else "no_forward_abc_records",
            "decision_count": int(len(strategy_rows)),
            "by_classification": by_classification,
            "policy_comparison": policy_results,
            "verdict": _verdict(policy_results),
        }
    return result


def CLASSIFICATIONS_PRESENT(rows: pd.DataFrame) -> set[str]:
    present = set(rows["classification"].dropna().astype(str)) if not rows.empty else set()
    return {"A", "B", "C", "SKIP"}.union(present)
