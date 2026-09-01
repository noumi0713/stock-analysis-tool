from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from app.adjustments import normalize_split_adjusted_ohlcv
from app.audit_metadata import resolve_git_commit
from app.live_strategy import load_frozen_strategy

OBSERVATION_SESSIONS = 10


def _primary_close(row: pd.Series) -> float:
    snapshot = row.get("primary_signal_snapshot")
    if isinstance(snapshot, dict) and snapshot.get("close") is not None:
        return float(snapshot["close"])
    if row.get("signal_close") is not None:
        return float(row["signal_close"])
    raise ValueError(f"Decision has no primary signal close: {row.get('ticker')}")


def track_ten_session_outcomes(
    decisions: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    strategy: dict[str, Any] | None = None,
    computed_at: str | None = None,
    git_commit: str | None = None,
    confirmed_buy_ids: dict[tuple[object, str, str], list[str]] | None = None,
) -> dict[str, Any]:
    spec = strategy or load_frozen_strategy()
    stamp = computed_at or datetime.now(UTC).isoformat()
    if decisions.empty:
        return {
            "schema_version": 1,
            "status": "no_secondary_decisions",
            "strategy_version": spec["strategy_version"],
            "observation_sessions": OBSERVATION_SESSIONS,
            "calculation_executed_at": stamp,
            "git_commit_id": resolve_git_commit(git_commit),
            "outcomes": [],
        }

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame = normalize_split_adjusted_ohlcv(frame)
    market_dates = sorted(frame["date"].dropna().unique())
    date_position = {day: index for index, day in enumerate(market_dates)}
    bars = {
        str(ticker): stock.sort_values("date").set_index("date")
        for ticker, stock in frame.groupby("ticker", sort=False)
    }
    one_way_slippage = float(spec["portfolio"]["slippage_bps_per_side"]) / 10_000.0
    round_trip_cost = (
        float(spec["portfolio"]["round_trip_transaction_cost_bps"]) / 10_000.0
    )
    fills = confirmed_buy_ids or {}
    output: list[dict[str, Any]] = []
    work = decisions.copy()
    work["signal_date"] = pd.to_datetime(work["signal_date"]).dt.date
    for _, decision in work.sort_values(
        ["signal_date", "signal_type", "primary_rank", "ticker"]
    ).iterrows():
        signal_date = decision["signal_date"]
        ticker = str(decision["ticker"])
        signal_type = str(decision["signal_type"])
        base = {
            "signal_date": signal_date.isoformat(),
            "ticker": ticker,
            "signal_type": signal_type,
            "primary_rank": int(decision["primary_rank"]),
            "classification": str(decision["classification"]),
            "secondary_action": str(decision.get("secondary_action") or "unknown"),
            "decision_audit_id": decision.get("decision_audit_id"),
            "observation_sessions": OBSERVATION_SESSIONS,
            "confirmed_buy_execution_ids": [],
            "all_primary_buy_net_return_10_sessions": None,
        }
        start_index = date_position.get(signal_date)
        if start_index is None or start_index + OBSERVATION_SESSIONS >= len(market_dates):
            output.append({**base, "status": "pending_10_sessions"})
            continue
        observation_dates = market_dates[
            start_index + 1 : start_index + 1 + OBSERVATION_SESSIONS
        ]
        base["confirmed_buy_execution_ids"] = fills.get(
            (observation_dates[0], ticker, signal_type), []
        )
        base["actual_trade_status"] = (
            "confirmed_bought"
            if base["confirmed_buy_execution_ids"]
            else "not_confirmed_bought"
        )
        stock = bars.get(ticker)
        if stock is None or any(day not in stock.index for day in observation_dates):
            output.append(
                {
                    **base,
                    "status": "data_insufficient_for_10_sessions",
                    "expected_observation_end": observation_dates[-1].isoformat(),
                }
            )
            continue
        future = stock.loc[observation_dates]
        signal_close = _primary_close(decision)
        raw_entry = float(future.iloc[0]["_open"])
        entry_gap = raw_entry / signal_close - 1.0
        execution = spec["signals"][signal_type]["execution"]
        entry_rule_eligible = True
        if "entry_gap_min" in execution:
            entry_rule_eligible = (
                float(execution["entry_gap_min"])
                <= entry_gap
                <= float(execution["entry_gap_max"])
            )

        raw_close_10 = float(future.iloc[-1]["_close"])
        raw_return_10 = raw_close_10 / raw_entry - 1.0
        mfe = float(future["_high"].max()) / raw_entry - 1.0
        mae = float(future["_low"].min()) / raw_entry - 1.0
        result = {
            **base,
            "status": "completed",
            "entry_date": observation_dates[0].isoformat(),
            "observation_end_date": observation_dates[-1].isoformat(),
            "signal_close": signal_close,
            "next_open": raw_entry,
            "entry_gap": entry_gap,
            "entry_rule_eligible": entry_rule_eligible,
            "close_10_sessions": raw_close_10,
            "unfiltered_return_10_sessions": raw_return_10,
            "maximum_favorable_excursion_10_sessions": mfe,
            "maximum_adverse_excursion_10_sessions": mae,
            "all_primary_buy_comparison_eligible": entry_rule_eligible,
            "all_primary_buy_net_return_10_sessions": None,
            "all_primary_buy_exit_reason": "entry_rule_excluded",
        }
        if entry_rule_eligible:
            entry_price = raw_entry * (1.0 + one_way_slippage)
            target = entry_price * (1.0 + float(execution["take_profit_pct"]))
            stop = entry_price * (1.0 + float(execution["stop_loss_pct"]))
            exit_price = raw_close_10 * (1.0 - one_way_slippage)
            exit_reason = "mark_at_10_sessions"
            exit_date = observation_dates[-1]
            for day, bar in future.iterrows():
                if float(bar["_low"]) <= stop:
                    exit_price = stop * (1.0 - one_way_slippage)
                    exit_reason = "stop_loss"
                    exit_date = day
                    break
                if float(bar["_high"]) >= target:
                    exit_price = target * (1.0 - one_way_slippage)
                    exit_reason = "take_profit"
                    exit_date = day
                    break
            result.update(
                {
                    "all_primary_buy_net_return_10_sessions": (
                        exit_price / entry_price - 1.0 - round_trip_cost
                    ),
                    "all_primary_buy_exit_reason": exit_reason,
                    "all_primary_buy_exit_date": exit_date.isoformat(),
                }
            )
        output.append(result)

    completed = [row for row in output if row["status"] == "completed"]
    return {
        "schema_version": 1,
        "status": "completed" if completed else "collecting",
        "strategy_version": spec["strategy_version"],
        "observation_sessions": OBSERVATION_SESSIONS,
        "policy": "track_every_primary_candidate_regardless_of_ABC_or_actual_purchase",
        "calculation_executed_at": stamp,
        "git_commit_id": resolve_git_commit(git_commit),
        "decision_count": len(output),
        "completed_count": len(completed),
        "outcomes": output,
    }
