from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from app.forward_validation import assess_forward_readiness, load_forward_gate


def _observations(count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = []
    paths = []
    start = date(2026, 9, 1)
    for index in range(count):
        signal_type = (
            "capitulation_reversal" if index % 2 == 0 else "first_pullback"
        )
        signal_date = start + timedelta(days=index)
        ticker = f"{1000 + index}.T"
        decisions.append(
            {
                "signal_date": signal_date,
                "ticker": ticker,
                "signal_type": signal_type,
                "classification": "A" if index % 3 == 0 else "B",
                "strategy_version": "live_v1_2026-08-31",
            }
        )
        paths.append(
            {
                "signal_date": signal_date,
                "ticker": ticker,
                "signal_type": signal_type,
                "net_return": 0.01 if index % 2 == 0 else -0.005,
            }
        )
    return pd.DataFrame(decisions), pd.DataFrame(paths)


def test_forward_gate_requires_at_least_fifty_completed_records() -> None:
    decisions, paths = _observations(49)

    result = assess_forward_readiness(
        decisions, paths, as_of=date(2027, 8, 31)
    )

    assert result["status"] == "not_ready"
    assert not result["checks"]["minimum_records"]


def test_passing_gate_never_authorizes_automatic_live_trading() -> None:
    decisions, paths = _observations(50)

    result = assess_forward_readiness(
        decisions, paths, as_of=date(2027, 8, 31)
    )

    assert result["status"] == "ready_for_manual_risk_review"
    assert result["checks"]["minimum_per_signal"]
    assert result["automated_live_trading_authorized"] is False


def test_gate_remains_closed_before_final_oos_unlock() -> None:
    decisions, paths = _observations(50)

    result = assess_forward_readiness(
        decisions, paths, as_of=date(2027, 3, 1)
    )

    assert result["status"] == "not_ready"
    assert not result["checks"]["final_oos_unlocked"]


def test_forward_gate_configuration_is_frozen_at_fifty_to_one_hundred() -> None:
    gate = load_forward_gate()

    assert gate["minimum_completed_records"] == 50
    assert gate["target_completed_records"] == 100
