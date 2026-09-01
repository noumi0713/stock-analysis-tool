from __future__ import annotations

import pandas as pd

from app.abc_evaluation import evaluate_abc_effectiveness, trade_metrics


def _outcome(index: int, signal_type: str, classification: str, net_return: float) -> dict:
    day = pd.Timestamp("2026-01-01") + pd.offsets.BDay(index)
    exit_day = day + pd.offsets.BDay(1)
    return {
        "signal_date": day.date().isoformat(),
        "ticker": f"{1000 + index}.T",
        "signal_type": signal_type,
        "primary_rank": 1,
        "classification": classification,
        "status": "completed",
        "entry_rule_eligible": True,
        "entry_date": day.date().isoformat(),
        "strategy_entry_price": 100.0,
        "strategy_exit_price": 100.0 * (1 + net_return),
        "strategy_exit_date": exit_day.date().isoformat(),
        "strategy_net_return": net_return,
        "hit_plus_5pct": net_return >= 0.05,
        "hit_minus_5pct": net_return <= -0.05,
        "hit_minus_8pct": net_return <= -0.08,
        "maximum_favorable_excursion_official_horizon": max(net_return, 0.0),
        "daily_mark_path": [
            {"date": day.date().isoformat(), "close": 100.0},
            {"date": exit_day.date().isoformat(), "close": 100.0 * (1 + net_return)},
        ],
    }


def test_trade_metrics_include_requested_risk_and_confidence_fields() -> None:
    frame = pd.DataFrame(
        [
            _outcome(0, "first_pullback", "A", 0.10),
            _outcome(1, "first_pullback", "A", -0.08),
            _outcome(2, "first_pullback", "A", 0.06),
        ]
    )

    metrics = trade_metrics(frame)

    assert metrics["completed_entry_eligible"] == 3
    assert metrics["win_rate"] == 2 / 3
    assert metrics["profit_factor"] == 2.0
    assert metrics["maximum_losing_streak"] == 1
    assert metrics["hit_plus_5pct_rate"] == 2 / 3
    assert metrics["hit_minus_8pct_rate"] == 1 / 3
    assert len(metrics["mean_net_return_95pct_ci"]) == 2
    assert len(metrics["win_rate_95pct_ci"]) == 2


def test_evaluation_separates_strategies_and_reports_all_policies() -> None:
    outcomes = pd.DataFrame(
        [
            _outcome(0, "first_pullback", "A", 0.08),
            _outcome(1, "first_pullback", "SKIP", -0.08),
            _outcome(2, "capitulation_reversal", "B", 0.03),
            _outcome(3, "capitulation_reversal", "C", -0.04),
        ]
    )
    decisions = outcomes[
        ["signal_date", "ticker", "signal_type", "primary_rank", "classification"]
    ].copy()
    decisions["evaluated_at"] = "2025-12-31T08:00:00+09:00"
    decisions["information_cutoff_at"] = "2025-12-31T07:55:00+09:00"
    decisions["entry_session_open_at"] = "2026-01-01T09:00:00+09:00"
    decisions["decision_context_sha256"] = "a" * 64

    result = evaluate_abc_effectiveness(
        decisions,
        outcomes,
        git_commit="abc123",
        calculated_at="2026-09-01T08:00:00+00:00",
    )

    pullback = result["strategies"]["first_pullback"]
    reversal = result["strategies"]["capitulation_reversal"]
    assert pullback["decision_count"] == 2
    assert reversal["decision_count"] == 2
    assert set(pullback["policy_comparison"]) == {
        "A_only",
        "A_plus_B",
        "A_plus_B_plus_C",
        "all_primary",
    }
    assert pullback["policy_comparison"]["A_only"]["opportunity_cost"][
        "avoided_loss_return_sum"
    ] == 0.08
    assert pullback["verdict"]["decision"] == "サンプル不足で判断保留"


def test_no_forward_decisions_is_not_backfilled_with_hindsight() -> None:
    result = evaluate_abc_effectiveness(
        pd.DataFrame(),
        pd.DataFrame(),
        git_commit="abc123",
    )

    assert result["status"] == "collecting"
    assert all(
        item["verdict"]["decision"] == "サンプル不足で判断保留"
        for item in result["strategies"].values()
    )
