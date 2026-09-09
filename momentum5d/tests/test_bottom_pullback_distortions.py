import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("distortions", Path(__file__).parents[1]/"scripts/bottom_pullback_distortions.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def synthetic():
    dates = pd.bdate_range("2021-08-19", periods=180)
    p = pd.DataFrame({"Ticker": "TEST", "Date": dates, "Close": 100., "Adj Close": 100.,
                      "volume_ratio_20d": 1., "trading_value_ratio_20d": 1., "rsi_14": np.arange(180)/3,
                      "lower_wick_ratio": .3, "upper_wick_ratio": .2, "volatility_contraction_ratio": .8})
    rows = []
    for kind, i in (("bottom", 130), ("pullback", 150)):
        rows.append({"signal_id": kind, "Ticker": "TEST", "kind": kind, "candidate_date": dates[i-5],
                     "signal_date": dates[i], "entry_date": dates[i+1], "eval_end_date": dates[i+10],
                     "status": "complete", "reason": None})
    meta = pd.DataFrame(rows)
    labels = pd.DataFrame({"signal_id": ["bottom", "pullback"], "hit_5": [1, 0], "hit_10": [0, 0],
                           "mae": [-.02, -.04], "close_return": [.03, -.02], "net_close_return": [.026, -.024]})
    resets = pd.DataFrame(columns=["Ticker", "Date", "reason"])
    return p, meta, labels, resets


def test_existing_feature_join_causal_and_recursive_dependency():
    p, meta, labels, resets = synthetic()
    a, exclusions = m.build_samples(p, meta, labels, resets)
    assert len(a) == 2 and not exclusions
    assert a.iloc[0].candidate_dependency_start == p.Date.iloc[40]
    assert a.iloc[1].candidate_dependency_start == p.Date.iloc[0]
    assert a.iloc[0].rsi_change5 == pytest.approx(5/3)
    altered = p.copy()
    altered.loc[altered.Date > meta.signal_date.max(), "rsi_14"] = 9999
    altered_labels = labels.copy()
    altered_labels.hit_5 = 1-altered_labels.hit_5
    b, _ = m.build_samples(altered, meta, altered_labels, resets)
    pd.testing.assert_frame_equal(a[list(m.FEATURES)], b[list(m.FEATURES)])


def test_adjustment_change_masks_multiday_features_not_wicks():
    p, meta, labels, resets = synthetic()
    p.loc[120:, "Adj Close"] = 50.
    a, _ = m.build_samples(p, meta, labels, resets)
    assert pd.isna(a.iloc[0].volume20)
    assert a.iloc[0].volume20__reason == "adjustment_not_constant_over_dependency"
    assert a.iloc[0].lower_wick == .3
    assert a.relative_sector20.isna().all() and a.relative_topix20.isna().all()


def test_incomplete_labels_not_failures():
    p, meta, labels, resets = synthetic()
    meta.loc[1, "status"], meta.loc[1, "reason"] = "unknown", "source_gap"
    a, excluded = m.build_samples(p, meta, labels.iloc[:1], resets)
    assert len(a) == 1 and excluded[0]["reason"] == "source_gap"


def test_purge_full_dependency_and_target_end():
    p, meta, labels, resets = synthetic()
    a, _ = m.build_samples(p, meta, labels, resets)
    assert m.interval_mask(a, p.Date.iloc[20], p.Date.iloc[-1], "lower_wick").tolist() == [True, False]
    assert not m.interval_mask(a, p.Date.iloc[0], meta.signal_date.min(), "lower_wick").any()
    a.loc[0, "lower_wick__start"] = p.Date.iloc[0]
    assert not m.interval_mask(a, p.Date.iloc[20], p.Date.iloc[-1], "lower_wick").any()


def test_train_only_direction_and_quantiles():
    train = pd.DataFrame({"x": [0., 1., 2., 3., 4., np.nan], "hit_5": [1, 1, 0, 0, 0, 1]})
    fixed = m.fit(train, "x")
    assert fixed["direction"] == "low" and fixed["q30"] == pytest.approx(1.2)
    test = pd.DataFrame({"x": [1., 4., np.nan], "hit_5": [0, 1, 1]})
    mask = m.apply_filter(test, "x", fixed)
    test.hit_5 = 1-test.hit_5
    assert mask.tolist() == [True, False, False]
    assert m.apply_filter(test, "x", fixed).equals(mask)


def test_auc_ties_and_single_class():
    assert m.auc([1, 1, 1], [1, 0, 1]) == .5
    assert np.isnan(m.auc([1, 2], [1, 1]))


def test_evaluation_denominator_includes_feature_missing():
    df = pd.DataFrame({"Ticker": ["A", "B", "C", "D"], "signal_date": pd.to_datetime(["2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01"]),
                       "x": [1., 2., 3., np.nan], "hit_5": [1, 0, 1, 1], "hit_10": [0, 0, 1, 0],
                       "mae": [-.01, -.02, -.03, -.04], "close_return": [.02, -.01, .05, .03], "net_close_return": [.016, -.014, .046, .026]})
    result, _ = m.evaluation(df, "x", {"direction": "high", "q70": 2.5, "q30": 1.5})
    assert result["baseline_n"] == 4 and result["retained_n"] == 1
    assert result["missing"] == 1 and result["missed_success_rate5"] == pytest.approx(2/3)


def test_independent_parquet_serialization_and_duplicates(tmp_path):
    for name in ("a", "b"):
        m.save(tmp_path/name, "data", [{"id": "2", "x": np.nan}, {"id": "1", "x": .4}], ["id"])
    assert m.manifest(tmp_path/"a") == m.manifest(tmp_path/"b")
    with pytest.raises(ValueError, match="duplicate"):
        m.save(tmp_path/"c", "data", [{"id": 1}, {"id": 1}], ["id"])


def test_no_future_frame_permitted():
    p, meta, labels, resets = synthetic()
    p.loc[len(p)-1, "Date"] = pd.Timestamp("2024-01-01")
    with pytest.raises(AssertionError):
        m.build_samples(p, meta, labels, resets)
