from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from app.yahoo.pullback_failure import (
    add_pullback_risk_flags,
    analyze_pullback_failures,
)


def _event(
    index: int,
    *,
    outcome: str,
    volume_ratio: float,
) -> dict[str, object]:
    entry = 100.0
    rows = {
        "success": {
            "highs": [106.0, 106.0, 106.0, 106.0, 106.0, 106.0, 106.0],
            "lows": [99.0, 100.0, 101.0, 101.0, 101.0, 101.0, 101.0],
            "closes": [105.0, 104.0, 103.0, 103.0, 103.0, 103.0, 103.0],
        },
        "persistent": {
            "highs": [101.0, 100.0, 99.0, 98.0, 98.0, 98.0, 98.0],
            "lows": [98.0, 96.0, 94.0, 93.0, 94.0, 94.0, 95.0],
            "closes": [99.0, 97.0, 95.0, 94.0, 95.0, 95.0, 96.0],
        },
        "recovered": {
            "highs": [101.0, 103.0, 106.0, 106.0, 106.0, 106.0, 106.0],
            "lows": [96.0, 98.0, 100.0, 101.0, 102.0, 102.0, 102.0],
            "closes": [98.0, 102.0, 105.0, 104.0, 104.0, 104.0, 104.0],
        },
        "ambiguous": {
            "highs": [106.0, 103.0, 103.0, 103.0, 103.0, 103.0, 103.0],
            "lows": [96.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
            "closes": [101.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0],
        },
    }[outcome]
    event: dict[str, object] = {
        "date": date(2026, 1, 1) + timedelta(days=index),
        "ticker": f"{1000 + index}.T",
        "code": str(1000 + index),
        "entry_date": date(2026, 1, 2) + timedelta(days=index),
        "entry_price": entry,
        "po_pullback_score": 0.5,
        "volume_ratio_1_20": volume_ratio,
        "po_close_location": 0.25 if outcome == "persistent" else 0.70,
        "po_perfect_order_age": 4 if outcome == "persistent" else 15,
    }
    for day in range(1, 8):
        event[f"future_open_{day}"] = entry
        event[f"future_high_{day}"] = rows["highs"][day - 1]
        event[f"future_low_{day}"] = rows["lows"][day - 1]
        event[f"future_close_{day}"] = rows["closes"][day - 1]
    return event


def test_failure_analysis_separates_persistent_decline_from_recovery() -> None:
    candidates = pd.DataFrame(
        [
            _event(0, outcome="success", volume_ratio=1.2),
            _event(1, outcome="persistent", volume_ratio=0.4),
            _event(2, outcome="recovered", volume_ratio=0.8),
            _event(3, outcome="ambiguous", volume_ratio=1.0),
        ]
    )
    report = analyze_pullback_failures(
        {
            "ma5_touch": candidates,
            "three_day_pullback": candidates.iloc[[1]],
        },
        validation_start=date(2026, 1, 3),
        horizon_days=7,
        stop_loss=0.03,
        minimum_threshold_samples=1,
    )

    result = report["summary"]["all"]
    assert report["event_count"] == 4
    assert result["rebound_success_count"] == 1
    assert result["stop_first_failure_count"] == 3
    assert result["persistent_decline_count"] == 1
    assert result["straight_decline_count"] == 1
    assert result["persistent_decline_rate"] == pytest.approx(0.25)
    assert len(report["persistent_decline_examples"]) == 1
    assert report["persistent_decline_examples"][0]["po_matched_rules"] == (
        "ma5_touch,three_day_pullback"
    )


def test_same_day_target_and_stop_is_not_rebound_success() -> None:
    candidates = pd.DataFrame([_event(0, outcome="ambiguous", volume_ratio=1.0)])
    report = analyze_pullback_failures(
        {"ma5_touch": candidates},
        validation_start=date(2026, 2, 1),
        horizon_days=7,
        stop_loss=0.03,
    )

    result = report["summary"]["all"]
    assert result["rebound_success_count"] == 0
    assert result["stop_first_failure_count"] == 1
    assert result["persistent_decline_count"] == 0


def test_two_or_more_frozen_risk_flags_are_counted_for_exclusion() -> None:
    frame = pd.DataFrame(
        {
            "atr_14_pct": [0.04, 0.02],
            "range_rate": [0.05, 0.02],
            "breakout_20d": [-0.02, -0.02],
            "return_20d": [0.03, 0.03],
            "volume_ratio_1_20": [1.0, 1.0],
            "po_return_3d": [-0.01, -0.01],
            "po_ma25_deviation": [0.01, 0.01],
        }
    )

    result = add_pullback_risk_flags(frame)

    assert result["po_risk_flag_count"].tolist() == [2, 0]
    assert result["po_risk_atr_14_pct"].tolist() == [True, False]
    assert result["po_risk_range_rate"].tolist() == [True, False]
