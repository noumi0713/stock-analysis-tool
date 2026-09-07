from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_steps5_to10.py"
SPEC = importlib.util.spec_from_file_location("build_steps5_to10", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_greedy_matching_never_reuses_a_control() -> None:
    events = pd.DataFrame({"event_id": ["e1", "e2"], "Ticker": ["1", "1"], "event_start_date": pd.to_datetime(["2023-01-10", "2023-01-20"]), "row_position": [10, 20]})
    pool = pd.DataFrame({"Date": pd.to_datetime(["2023-01-01", "2023-02-01"]), "Ticker": ["1", "1"], "row_position": [1, 30]})
    result = MODULE.greedy_control_matches(events, pool)
    assert len(result) == 2
    assert not result.duplicated(["Ticker", "control_start_date"]).any()


def test_auc_is_oriented() -> None:
    y = pd.Series([0] * 10 + [1] * 10)
    auc, direction, count = MODULE.auc_metric(y, pd.Series(range(20, 0, -1)))
    assert auc == 1.0
    assert direction == "low"
    assert count == 20


def test_cooldown_is_per_ticker() -> None:
    frame = pd.DataFrame({"Ticker": ["1", "1", "1", "2"], "row_position": [1, 10, 23, 2]})
    selected = MODULE.cooldown_signals(frame, pd.Series(True, index=frame.index), 20)
    assert selected.index.tolist() == [0, 2, 3]


def test_equal_notional_drawdown_does_not_underflow() -> None:
    result = MODULE.max_equal_notional_drawdown_points(pd.Series([0.1, -0.2, 0.05]))
    assert abs(result - (-0.2)) < 1e-12
