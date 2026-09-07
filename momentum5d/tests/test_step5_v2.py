# ruff: noqa: E501
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_step5_v2.py"
SPEC = importlib.util.spec_from_file_location("build_step5_v2", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_period_classification_has_quarantined_holdout() -> None:
    dates = pd.Series(pd.to_datetime(["2023-12-31", "2024-01-01", "2025-09-07", "2025-09-08"]))
    assert MODULE.classify_period(dates).tolist() == [
        "formation",
        "validation",
        "validation",
        "contaminated_holdout",
    ]


def test_deterministic_matches_stay_in_period_and_never_reuse() -> None:
    events = pd.DataFrame(
        {
            "event_id": ["e2", "e1"],
            "Ticker": ["1", "1"],
            "event_start_date": pd.to_datetime(["2024-02-01", "2024-01-10"]),
            "event_period": ["validation", "validation"],
            "event_window_start_date": pd.to_datetime(["2024-01-04", "2024-01-01"]),
            "event_target_end_date": pd.to_datetime(["2024-04-01", "2024-03-01"]),
            "event_row_position": [40, 20],
        }
    )
    controls = pd.DataFrame(
        {
            "Ticker": ["1", "1", "1"],
            "control_start_date": pd.to_datetime(["2024-01-08", "2024-02-02", "2023-12-20"]),
            "control_period": ["validation", "validation", "formation"],
            "control_window_start_date": pd.to_datetime(["2024-01-01", "2024-01-05", "2023-11-20"]),
            "control_target_end_date": pd.to_datetime(["2024-03-01", "2024-04-02", "2023-12-29"]),
            "control_row_position": [18, 41, 5],
        }
    )
    first, unmatched = MODULE.deterministic_matches(events, controls)
    second, _ = MODULE.deterministic_matches(
        events.sample(frac=1, random_state=7), controls.sample(frac=1, random_state=8)
    )
    assert unmatched.empty
    assert first[["event_id", "control_start_date"]].equals(
        second[["event_id", "control_start_date"]]
    )
    assert not first.duplicated(["Ticker", "control_start_date"]).any()
    assert (first.event_period == first.control_period).all()
    assert set(first.control_start_date) == set(pd.to_datetime(["2024-01-08", "2024-02-02"]))


def test_boundary_reason_rejects_target_crossing() -> None:
    frame = pd.DataFrame(
        {
            "event_period": ["validation"],
            "event_window_start_date": pd.to_datetime(["2025-06-01"]),
            "event_target_end_date": pd.to_datetime(["2025-09-08"]),
            "event_window_period": ["validation"],
            "event_target_period": ["contaminated_holdout"],
        }
    )
    assert MODULE.reason_for_boundary(next(frame.itertuples(index=False)), "event") == (
        "target_horizon_crosses_period_boundary"
    )
