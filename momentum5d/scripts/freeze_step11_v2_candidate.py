"""Freeze one STEP10 candidate before any 2025-09-08+ value is loaded."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

EXPECTED_STEP7_REPORT = "5b32a250ac673bbfa60c7e6dcd818ada28832f4fa7a2381c8f906775a3f2f14a"
EXPECTED_STEP10_REPORT = "02fca13eef0542f065dac66d10ccc51e50a6eeeb4daf3be4c8482fc4bdd10a0d"
EXPECTED_SELECTION = ("C3-0184", "anchor_close", "time_40_close")


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def manifest(path: Path) -> dict[str, str]:
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    return {str(item): sha256(item) for item in files}


def finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/market_history")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    step7 = root / "analysis/step7_condition_combinations_v2"
    step10 = root / "analysis/step10_exit_analysis_v2"
    step7_report_path = root / "quality/step7_v2_report.json"
    step10_report_path = root / "quality/step10_v2_report.json"
    provenance_path = root / "quality/step11_input_provenance.json"

    required = [
        step7 / "combination_definitions/part.parquet",
        step7 / "candidate_features/part.parquet",
        step10 / "exit_stability/part.parquet",
        step7_report_path,
        step10_report_path,
        provenance_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"STEP11 freeze inputs missing: {missing}")
    if sha256(step7_report_path) != EXPECTED_STEP7_REPORT:
        raise RuntimeError("STEP7 V2 report hash mismatch")
    if sha256(step10_report_path) != EXPECTED_STEP10_REPORT:
        raise RuntimeError("STEP10 V2 report hash mismatch")

    step7_report = json.loads(step7_report_path.read_text(encoding="utf-8"))
    step10_report = json.loads(step10_report_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if step7_report.get("quality") != "PASS" or step10_report.get("quality") != "PASS":
        raise RuntimeError("STEP7 V2 and STEP10 V2 must both be PASS")
    if not step10_report.get("reproducibility_passed") or not step10_report.get("inputs_unchanged"):
        raise RuntimeError("STEP10 V2 reproducibility/input-integrity prerequisite failed")
    if any(step10_report.get("hard_failures", {}).values()):
        raise RuntimeError("STEP10 V2 contains a hard failure")
    if provenance.get("untouched_oos_claimed") is not False:
        raise RuntimeError("STEP11 provenance must explicitly reject untouched-OOS status")

    input_before = {}
    for path in required:
        if path.suffix == ".parquet":
            input_before.update(manifest(path))
        else:
            input_before[str(path)] = sha256(path)

    stability = pd.read_parquet(step10 / "exit_stability")
    definitions = pd.read_parquet(step7 / "combination_definitions")
    candidates = pd.read_parquet(step7 / "candidate_features")
    eligible = stability[
        stability.formation_exit_evaluable_count.ge(100)
        & stability.validation_exit_evaluable_count.ge(100)
        & stability.formation_mean_net_return.gt(0)
        & stability.validation_mean_net_return.gt(0)
        & stability.formation_profit_factor.gt(1)
        & stability.validation_profit_factor.gt(1)
        & stability.mean_return_direction_reproduced
        & stability.positive_mean_return_years.eq(stability.annual_periods)
    ].copy()
    eligible["worst_period_mean_net_return"] = eligible[
        ["formation_mean_net_return", "validation_mean_net_return"]
    ].min(axis=1)
    eligible["worst_period_profit_factor"] = eligible[
        ["formation_profit_factor", "validation_profit_factor"]
    ].min(axis=1)
    eligible["minimum_period_evaluable_count"] = eligible[
        ["formation_exit_evaluable_count", "validation_exit_evaluable_count"]
    ].min(axis=1)
    eligible["pre_holdout_score"] = (
        eligible.worst_period_mean_net_return
        * np.log1p(eligible.worst_period_profit_factor)
        * np.log1p(eligible.minimum_period_evaluable_count)
    )
    eligible = eligible.sort_values(
        [
            "pre_holdout_score",
            "worst_period_mean_net_return",
            "worst_period_profit_factor",
            "combination_id",
            "entry_definition",
            "exit_definition",
        ],
        ascending=[False, False, False, True, True, True],
        kind="stable",
    )
    if eligible.empty:
        raise RuntimeError("No eligible STEP10 candidate can be frozen")
    selected = eligible.iloc[0]
    selected_key = (
        str(selected.combination_id),
        str(selected.entry_definition),
        str(selected.exit_definition),
    )
    if selected_key != EXPECTED_SELECTION:
        raise RuntimeError(f"Deterministic freeze drifted: {selected_key} != {EXPECTED_SELECTION}")

    definition = definitions[definitions.combination_id.eq(selected.combination_id)]
    if len(definition) != 1 or bool(definition.iloc[0].adopted_signal):
        raise RuntimeError("STEP7 definition mismatch or previously adopted signal")
    definition = definition.iloc[0]
    features: list[dict[str, object]] = []
    for number in range(1, int(definition.condition_count) + 1):
        features.append(
            {
                "feature": str(definition[f"feature_{number}"]),
                "direction": str(definition[f"direction_{number}"]),
                "operator": str(definition[f"operator_{number}"]),
                "threshold": float(definition[f"threshold_{number}"]),
            }
        )
    selected_features = candidates[candidates.selected_candidate.fillna(False)]
    regimes = []
    for feature in (
        "index_nikkei225_change_5d",
        "external_vix_change_1d",
        "market_decliner_ratio",
    ):
        row = selected_features[selected_features.feature.eq(feature)]
        if len(row) != 1:
            raise RuntimeError(f"Fixed regime feature missing from STEP7 candidates: {feature}")
        regimes.append(
            {
                "feature": feature,
                "q30": float(row.iloc[0].q30),
                "q70": float(row.iloc[0].q70),
                "source": "STEP7 formation-only coarse_30_70",
            }
        )

    freeze = pd.DataFrame(
        [
            {
                "candidate_id": "STEP11V2-FROZEN-001",
                "combination_id": selected.combination_id,
                "entry_definition": selected.entry_definition,
                "exit_definition": selected.exit_definition,
                "condition_count": int(selected.condition_count),
                "features_json": json.dumps(features, ensure_ascii=False, sort_keys=True),
                "regime_definitions_json": json.dumps(regimes, ensure_ascii=False, sort_keys=True),
                "selection_rule": (
                    "max deterministic pre_holdout_score among stable STEP10 variants"
                ),
                "selection_inputs": "STEP7 V2 and STEP10 V2 formation+validation only",
                "pre_holdout_score": float(selected.pre_holdout_score),
                "formation_evaluable_count": int(selected.formation_exit_evaluable_count),
                "validation_evaluable_count": int(selected.validation_exit_evaluable_count),
                "formation_mean_net_return": float(selected.formation_mean_net_return),
                "validation_mean_net_return": float(selected.validation_mean_net_return),
                "formation_profit_factor": float(selected.formation_profit_factor),
                "validation_profit_factor": float(selected.validation_profit_factor),
                "round_trip_cost_rate": 0.004,
                "maximum_positions": 10,
                "target_position_fraction": 0.10,
                "same_ticker_concurrent_positions": 1,
                "same_day_priority": "ascending sha256(date|ticker|STEP11V2-FROZEN-001)",
                "validation_slice_start": pd.Timestamp("2025-09-08"),
                "analysis_label": "contaminated_validation",
                "untouched_oos": False,
                "post_2025_09_08_values_read_during_freeze": False,
                "candidate_reoptimization_allowed": False,
            }
        ]
    )
    freeze_root = root / "frozen/step11_v2_candidate"
    if freeze_root.exists():
        raise RuntimeError(f"Refusing to overwrite candidate freeze: {freeze_root}")
    freeze_root.mkdir(parents=True)
    output = freeze_root / "part.parquet"
    pq.write_table(pa.Table.from_pandas(freeze, preserve_index=False), output, compression="zstd")
    reread = pd.read_parquet(output)
    if not reread.equals(freeze):
        raise RuntimeError("Frozen candidate Parquet reread mismatch")

    input_after = {}
    for path in required:
        if path.suffix == ".parquet":
            input_after.update(manifest(path))
        else:
            input_after[str(path)] = sha256(path)
    if input_before != input_after:
        raise RuntimeError("Inputs changed while freezing STEP11 candidate")
    report = {
        "quality": "PASS",
        "analysis_label": "contaminated_validation",
        "untouched_oos": False,
        "freeze_completed_before_validation_values_loaded": True,
        "selected_key": list(selected_key),
        "expected_selected_key": list(EXPECTED_SELECTION),
        "selection_rule": freeze.iloc[0].selection_rule,
        "eligible_variant_count": int(len(eligible)),
        "frozen_candidate_count": 1,
        "freeze_parquet_sha256": sha256(output),
        "input_manifest_before": input_before,
        "input_manifest_after": input_after,
        "inputs_unchanged": True,
    }
    report_path = root / "quality/step11_candidate_freeze.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
