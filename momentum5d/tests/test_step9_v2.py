# ruff: noqa: E501
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_step9_v2.py"
SPEC = importlib.util.spec_from_file_location("build_step9_v2", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "sample_id": "e-event",
            "event_id": "e",
            "Ticker": "1111.T",
            "period": "validation",
            "observation_type": "event",
            "outcome": 1,
            "anchor_date": pd.Timestamp("2024-01-02"),
            "anchor_year": 2024,
        }
    ])


def price_panel(closes: list[float]) -> dict[str, dict[str, np.ndarray]]:
    dates = pd.bdate_range("2024-01-02", periods=len(closes)).to_numpy(dtype="datetime64[ns]")
    close = np.asarray(closes, dtype=float)
    return {
        "1111.T": {
            "date": dates,
            "adj_open": close + 1.0,
            "adj_high": close + 2.0,
            "adj_low": close - 2.0,
            "adj_close": close,
            "volume": np.full(len(close), 1000.0),
            "price_values_loaded": np.ones(len(close), dtype=bool),
        }
    }


def test_entry_definitions_are_exact_and_frozen() -> None:
    definitions = MODULE.make_entry_definitions()
    assert definitions.entry_definition.tolist() == list(MODULE.ENTRY_NAMES)
    assert definitions.fixed_before_validation.all()
    assert not definitions.validation_used_to_change_rule.any()
    assert not definitions.step8_used_to_select_rule.any()


def test_first_pullback_uses_first_qualifying_ticker_row() -> None:
    entries = MODULE.make_entry_observations(sample_frame(), price_panel([100, 102, 101, 99, 103, 104, 105]))
    pullback = entries[entries.entry_definition.eq("first_pullback")].iloc[0]
    assert pullback.filled
    assert pullback.entry_date == pd.Timestamp("2024-01-04")
    assert pullback.entry_price == 101.0


def test_first_pullback_preserves_no_fill() -> None:
    entries = MODULE.make_entry_observations(sample_frame(), price_panel([100, 101, 102, 103, 104, 105, 106]))
    pullback = entries[entries.entry_definition.eq("first_pullback")].iloc[0]
    assert not pullback.filled
    assert pullback.entry_rule_reason == "no_pullback_within_5_sessions"


def test_next_open_is_next_ticker_row_not_calendar_day() -> None:
    entries = MODULE.make_entry_observations(sample_frame(), price_panel([100, 101, 102, 103, 104, 105, 106]))
    next_open = entries[entries.entry_definition.eq("next_session_open")].iloc[0]
    assert next_open.entry_date == pd.Timestamp("2024-01-03")
    assert next_open.entry_price == 102.0


def test_attach_targets_rebases_open_and_keeps_close_triplet(tmp_path: Path) -> None:
    samples = sample_frame()
    prices = price_panel([100, 101, 100, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162])
    entries = MODULE.make_entry_observations(samples, prices)
    target_rows = []
    for date in pd.bdate_range("2024-01-02", periods=63):
        row = {"Date": date.date(), "Ticker": "1111.T"}
        for horizon in MODULE.HORIZONS:
            row[f"forward_return_{horizon}d"] = 0.10
            row[f"mfe_{horizon}d"] = 0.20
            row[f"mae_{horizon}d"] = -0.05
        target_rows.append(row)
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    pd.DataFrame(target_rows).to_parquet(target_dir / "part.parquet", index=False)
    result = MODULE.attach_targets(entries, prices, target_dir)
    anchor = result[result.entry_definition.eq("anchor_close")].iloc[0]
    assert anchor.forward_return_5d == 0.10
    assert anchor.mfe_5d == 0.20
    assert anchor.mae_5d == -0.05
    next_open = result[result.entry_definition.eq("next_session_open")].iloc[0]
    expected_return = 101.0 * 1.10 / 102.0 - 1.0
    expected_mfe = max(103.0, 101.0 * 1.20) / 102.0 - 1.0
    expected_mae = min(99.0, 101.0 * 0.95) / 102.0 - 1.0
    assert np.isclose(next_open.forward_return_5d, expected_return)
    assert np.isclose(next_open.mfe_5d, expected_mfe)
    assert np.isclose(next_open.mae_5d, expected_mae)


def test_validation_boundary_is_excluded_without_value() -> None:
    dates = pd.bdate_range("2025-08-25", periods=20).to_numpy(dtype="datetime64[ns]")
    close = np.arange(100, 120, dtype=float)
    prices = {"1111.T": {"date": dates, "adj_open": close, "adj_high": close + 1, "adj_low": close - 1, "adj_close": close, "volume": np.full(len(close), 1000.0), "price_values_loaded": dates < np.datetime64("2025-09-08")}}
    sample = sample_frame()
    sample.loc[0, "anchor_date"] = pd.Timestamp("2025-09-05")
    sample.loc[0, "anchor_year"] = 2025
    entries = MODULE.make_entry_observations(sample, prices)
    assert not entries[entries.entry_definition.eq("next_session_open")].iloc[0].filled
    assert entries[entries.entry_definition.eq("next_session_open")].iloc[0].entry_rule_reason == "next_session_crosses_period_boundary"


def test_condition_mask_uses_frozen_and_thresholds() -> None:
    samples = pd.DataFrame({"a": [0.0, 2.0, np.nan], "b": [3.0, 1.0, 3.0]})
    definitions = pd.DataFrame([{
        "combination_id": "C2-0001", "condition_count": 2,
        "feature_1": "a", "direction_1": "high", "threshold_1": 1.0,
        "feature_2": "b", "direction_2": "low", "threshold_2": 2.0,
        "feature_3": None, "direction_3": None, "threshold_3": np.nan,
    }])
    assert MODULE.condition_masks(samples, definitions)["C2-0001"].tolist() == [False, True, False]


def test_zero_volume_is_indeterminate_not_a_hypothetical_fill() -> None:
    prices = price_panel([100, 99, 101, 102, 103, 104, 105])
    prices["1111.T"]["volume"][1] = 0.0
    entries = MODULE.make_entry_observations(sample_frame(), prices)
    next_open = entries[entries.entry_definition.eq("next_session_open")].iloc[0]
    assert next_open.price_reference_available
    assert next_open.entry_status == "indeterminate"
    assert not next_open.hypothetical_filled
    assert next_open.entry_rule_reason == "entry_tradability_unconfirmed_zero_or_missing_volume"


def test_missing_pullback_search_row_stops_as_indeterminate() -> None:
    prices = price_panel([100, np.nan, 99, 98, 97, 96, 95])
    entries = MODULE.make_entry_observations(sample_frame(), prices)
    pullback = entries[entries.entry_definition.eq("first_pullback")].iloc[0]
    assert pullback.entry_status == "indeterminate"
    assert pullback.entry_rule_reason == "entry_search_source_missing"


def test_holdout_price_values_are_not_loaded(tmp_path: Path) -> None:
    source = tmp_path / "features"
    source.mkdir()
    frame = pd.DataFrame({
        "Date": pd.to_datetime(["2025-09-05", "2025-09-08"]),
        "Ticker": ["1111.T", "1111.T"],
        "Open": [100.0, 999.0], "High": [101.0, 999.0], "Low": [99.0, 999.0],
        "Close": [100.0, 999.0], "Adj Close": [100.0, 999.0], "Volume": [1000.0, 9999.0],
    })
    frame.to_parquet(source / "part.parquet", index=False)
    loaded = MODULE.load_prices(source, ["1111.T"])["1111.T"]
    assert loaded["date"].tolist() == frame.Date.to_numpy(dtype="datetime64[ns]").tolist()
    assert loaded["adj_close"][0] == 100.0
    assert np.isnan(loaded["adj_close"][1])
    assert np.isnan(loaded["volume"][1])
    assert loaded["price_values_loaded"].tolist() == [True, False]
