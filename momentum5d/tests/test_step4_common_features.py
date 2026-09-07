from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_step4_common_features.py"
SPEC = importlib.util.spec_from_file_location("build_step4_common_features", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_baselines_are_documented_and_conservative() -> None:
    assert MODULE.baseline_for("return_20d", False) == 0.0
    assert MODULE.baseline_for("volume_ratio_20d", False) == 1.0
    assert MODULE.baseline_for("market_advancer_ratio", False) == 0.5
    assert MODULE.baseline_for("Close", False) is None


def test_bucket_summary_uses_one_median_per_event() -> None:
    frame = pd.DataFrame({
        "event_id": ["a", "a", "b", "b"],
        "relative_day": [-5, -4, -5, -4],
        "return_20d": [0.1, 0.2, -0.1, -0.2],
    })
    summary, medians = MODULE.summarize_buckets(frame, ["return_20d"], set())
    selected = summary.query("bucket == 'd-5_to_d-1'").iloc[0]
    assert selected["events_with_value"] == 2
    assert len(medians.query("bucket == 'd-5_to_d-1'")) == 2
    assert selected["directional_consistency_rate"] == 0.5
