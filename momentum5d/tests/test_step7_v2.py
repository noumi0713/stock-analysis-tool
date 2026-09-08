# ruff: noqa: E501
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_step7_v2.py"
SPEC = importlib.util.spec_from_file_location("build_step7_v2", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_and_condition_respects_fixed_direction_and_missingness() -> None:
    frame = pd.DataFrame({
        "outcome": [1, 0, 1, 0],
        "high_feature": [3.0, 3.0, 1.0, None],
        "low_feature": [1.0, 3.0, 1.0, 1.0],
    })
    result = MODULE.evaluate_condition(
        frame, [("high_feature", "high", 2.0), ("low_feature", "low", 2.0)]
    )
    assert result["valid_rows"] == 3
    assert result["missing_rows"] == 1
    assert result["condition_rows"] == 1
    assert result["condition_events"] == 1
    assert result["condition_controls"] == 0


def test_combination_definitions_are_only_two_and_three_way_and() -> None:
    candidates = pd.DataFrame({
        "feature": ["a", "b", "c"],
        "formation_direction": ["high", "low", "high"],
        "condition_operator": [">=", "<=", ">="],
        "condition_threshold": [1.0, 2.0, 3.0],
    })
    definitions = MODULE.combination_definitions(candidates, ["a", "b", "c"])
    assert len(definitions) == 4
    assert (definitions.logical_operator == "AND").all()
    assert set(definitions.condition_count) == {2, 3}
    assert not definitions.validation_used_for_definition.any()


def test_candidate_policy_constants_are_coarse_and_fixed() -> None:
    assert MODULE.FORMATION_AUC_MIN == 0.55
    assert MODULE.FORMATION_MISSING_MAX == 0.20
    assert MODULE.FORMATION_ABS_SMD_MIN == 0.10
    assert MODULE.FORMATION_YEAR_REPRO_MIN == 2 / 3
    assert MODULE.CORRELATION_LIMIT == 0.80
