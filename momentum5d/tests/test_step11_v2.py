from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts/build_step11_v2.py"
FREEZE_SCRIPT = Path(__file__).parents[1] / "scripts/freeze_step11_v2_candidate.py"
spec = importlib.util.spec_from_file_location("build_step11_v2", SCRIPT)
assert spec and spec.loader
step11 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step11)


def frozen_row() -> pd.Series:
    conditions = [
        {"feature": "return_60d", "operator": "<=", "threshold": -0.067149, "direction": "low"},
        {
            "feature": "external_vix_change_1d",
            "operator": ">=",
            "threshold": 0.013187,
            "direction": "high",
        },
        {
            "feature": "external_sox_change_5d",
            "operator": "<=",
            "threshold": -0.026231,
            "direction": "low",
        },
    ]
    regimes = [
        {"feature": "index_nikkei225_change_5d", "q30": -0.01, "q70": 0.01},
        {"feature": "external_vix_change_1d", "q30": -0.01, "q70": 0.01},
        {"feature": "market_decliner_ratio", "q30": 0.4, "q70": 0.55},
    ]
    return pd.Series(
        {
            "candidate_id": "STEP11V2-FROZEN-001",
            "combination_id": "C3-0184",
            "entry_definition": "anchor_close",
            "exit_definition": "time_40_close",
            "features_json": json.dumps(conditions),
            "regime_definitions_json": json.dumps(regimes),
            "same_day_priority": "ascending sha256(date|ticker|candidate_id)",
            "analysis_label": "contaminated_validation",
            "untouched_oos": False,
        }
    )


def synthetic_prices(tickers: int = 1, sessions: int = 65) -> pd.DataFrame:
    dates = pd.bdate_range("2025-08-25", periods=sessions)
    rows = []
    for ticker_number in range(tickers):
        ticker = f"{1000 + ticker_number}.T"
        for position, date in enumerate(dates):
            price = 100.0 + position
            rows.append(
                {
                    "Date": date,
                    "Ticker": ticker,
                    "Open": price,
                    "High": price * 1.01,
                    "Low": price * 0.99,
                    "Close": price,
                    "Adj Close": price,
                    "Volume": 100_000.0,
                    "trading_value": price * 100_000.0,
                    "return_60d": -0.10,
                    "external_vix_change_1d": 0.02,
                    "external_sox_change_5d": -0.03,
                    "index_nikkei225_change_5d": 0.0,
                    "market_decliner_ratio": 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_analysis_is_explicitly_not_oos() -> None:
    assert step11.ANALYSIS_LABEL == "contaminated_validation"
    assert "not untouched OOS" in step11.__doc__
    assert "完全未使用OOSではありません" in step11.step12_prompt()


def test_conditions_use_five_prior_rows() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.bdate_range("2025-09-08", periods=2),
            "bucket__x": [1.0, 1.0],
            "history_count__x": [4, 5],
        }
    )
    mask = step11.condition_mask(frame, [{"feature": "x", "operator": ">=", "threshold": 1.0}])
    assert mask.tolist() == [False, True]


def test_same_day_priority_is_deterministic() -> None:
    date = pd.Timestamp("2025-10-01")
    first = step11.priority_hash(date, "7203.T", "candidate")
    assert first == step11.priority_hash(date, "7203.T", "candidate")
    assert first != step11.priority_hash(date, "6758.T", "candidate")


def test_build_once_and_portfolio_constraints(tmp_path: Path) -> None:
    root = tmp_path / "market_history"
    step1 = root / "features/equity_daily_features"
    step1.mkdir(parents=True)
    prices = synthetic_prices(tickers=12, sessions=70)
    prices.to_parquet(step1 / "part.parquet", index=False)
    prior = root / "analysis/step10_exit_analysis_v2/exit_stability"
    prior.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "combination_id": "C3-0184",
                "entry_definition": "anchor_close",
                "exit_definition": "time_40_close",
                "formation_mean_net_return": 0.10,
                "validation_mean_net_return": 0.12,
                "formation_profit_factor": 2.0,
                "validation_profit_factor": 2.2,
            }
        ]
    ).to_parquet(prior / "part.parquet", index=False)
    freeze = frozen_row()
    conditions = json.loads(freeze.features_json)
    regimes = json.loads(freeze.regime_definitions_json)
    destination = tmp_path / "output"
    step11.build_once(root, destination, freeze, conditions, regimes)
    policy = pd.read_parquet(destination / "validation_policy")
    signals = pd.read_parquet(destination / "signals")
    decisions = pd.read_parquet(destination / "portfolio_decisions")
    daily = pd.read_parquet(destination / "portfolio_daily")
    assert not bool(policy.iloc[0].untouched_oos)
    assert signals.signal_date.min() >= pd.Timestamp("2025-09-08")
    evaluated = signals[signals.exit_evaluable]
    assert evaluated.lookback_start_date.min() >= pd.Timestamp("2025-09-08")
    assert signals.exclusion_reason.eq("lookback_crosses_validation_start").any()
    assert int(daily.open_positions.max()) <= 10
    assert decisions.accepted.any()
    assert decisions.rejection_reason.eq("maximum_concurrent_positions_reached").any()


def test_freeze_selects_expected_candidate_from_certified_inputs(tmp_path: Path) -> None:
    workspace = Path(__file__).parents[3]
    available_step7 = workspace / "step7-v2-production/analysis/step7_condition_combinations_v2"
    available_step10 = workspace / "step10-restored/analysis/step10_exit_analysis_v2"
    available_quality7 = workspace / "step7-v2-production/quality/step7_v2_report.json"
    available_quality10 = workspace / "step10-restored/quality/step10_v2_report.json"
    available = (available_step7, available_step10, available_quality7, available_quality10)
    if not all(path.exists() for path in available):
        return
    root = tmp_path / "market_history"
    import shutil

    shutil.copytree(available_step7, root / "analysis/step7_condition_combinations_v2")
    shutil.copytree(available_step10, root / "analysis/step10_exit_analysis_v2")
    (root / "quality").mkdir(parents=True)
    shutil.copy2(available_quality7, root / "quality/step7_v2_report.json")
    shutil.copy2(available_quality10, root / "quality/step10_v2_report.json")
    (root / "quality/step11_input_provenance.json").write_text(
        json.dumps({"untouched_oos_claimed": False}), encoding="utf-8"
    )
    subprocess.run([sys.executable, str(FREEZE_SCRIPT), "--root", str(root)], check=True)
    frozen = pd.read_parquet(root / "frozen/step11_v2_candidate")
    selected = (
        frozen.iloc[0].combination_id,
        frozen.iloc[0].entry_definition,
        frozen.iloc[0].exit_definition,
    )
    assert selected == (
        "C3-0184",
        "anchor_close",
        "time_40_close",
    )
