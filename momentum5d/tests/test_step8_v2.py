# ruff: noqa: E501
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_step8_v2.py"
SPEC = importlib.util.spec_from_file_location("build_step8_v2", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_regime_classification_keeps_unknown_without_imputation() -> None:
    values = pd.Series([-2.0, 0.0, 2.0, None, float("inf")])
    result = MODULE.classify(values, -1.0, 1.0, ("weak", "neutral", "strong"))
    assert result.tolist() == ["weak", "neutral", "strong", "unknown", "unknown"]


def test_regime_specs_are_individual_and_complete() -> None:
    assert len(MODULE.REGIME_SPECS) == 9
    assert len({item[0] for item in MODULE.REGIME_SPECS}) == 9
    assert all(item[1] for item in MODULE.REGIME_SPECS)


def test_summary_uses_valid_combination_rows_and_case_control_base() -> None:
    outcomes = pd.Series([1, 0, 1, 0]).to_numpy()
    group = pd.Series([True, True, True, False]).to_numpy()
    valid = pd.Series([True, True, False, True]).to_numpy()
    selected = pd.Series([True, False, False, True]).to_numpy()
    result = MODULE.summarize_mask(outcomes, group, valid, selected)
    assert result["total_rows"] == 3
    assert result["valid_rows"] == 2
    assert result["condition_rows"] == 1
    assert result["condition_events"] == 1
    assert result["condition_controls"] == 0
    assert result["base_event_rate"] == 0.5


def test_no_regime_optimization_or_cross_environment_definition() -> None:
    assert MODULE.MAIN_BUCKET == "d-5_to_d-1"
    assert MODULE.HOLDOUT_START == pd.Timestamp("2025-09-08")


def test_zero_formation_lift_is_not_an_infinite_retention() -> None:
    formation = pd.Series([0.0, 1.0])
    validation = pd.Series([1.0, 2.0])
    retention = pd.Series(
        MODULE.np.where(
            formation.notna() & validation.notna() & (formation != 0),
            validation / formation,
            MODULE.np.nan,
        )
    )
    assert pd.isna(retention.iloc[0])
    assert retention.iloc[1] == 2.0
