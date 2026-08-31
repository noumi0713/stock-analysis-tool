from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from app.evaluation_protocol import load_evaluation_protocol, oos_access_status

DEFAULT_FORWARD_GATE_PATH = (
    Path(__file__).resolve().parent
    / "resources"
    / "evaluation_protocols"
    / "forward_gate_v1_2026-08-31.json"
)
FROZEN_FORWARD_GATE_SHA256 = (
    "a796a1af12605b9e989bee1109826958b1a52a8ae46bdf43f724d3ac3766cf64"
)


def load_forward_gate(path: Path | None = None) -> dict[str, Any]:
    gate_path = path or DEFAULT_FORWARD_GATE_PATH
    if gate_path.resolve() == DEFAULT_FORWARD_GATE_PATH.resolve():
        digest = hashlib.sha256(gate_path.read_bytes()).hexdigest()
        if digest != FROZEN_FORWARD_GATE_SHA256:
            raise ValueError(
                "Frozen forward gate changed without a new version: "
                f"expected={FROZEN_FORWARD_GATE_SHA256} actual={digest}"
            )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "frozen":
        raise ValueError("Forward-validation gate must be frozen")
    if gate.get("strategy_version") != "live_v1_2026-08-31":
        raise ValueError("Forward-validation strategy version mismatch")
    if int(gate.get("minimum_completed_records", 0)) < 50:
        raise ValueError("Forward-validation minimum cannot be below 50")
    if int(gate.get("target_completed_records", 0)) < int(
        gate["minimum_completed_records"]
    ):
        raise ValueError("Forward-validation target is below its minimum")
    return gate


def assess_forward_readiness(
    decisions: pd.DataFrame,
    candidate_paths: pd.DataFrame,
    *,
    as_of: date,
    gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rules = gate or load_forward_gate()
    oos = oos_access_status(as_of=as_of, protocol=load_evaluation_protocol())
    if decisions.empty:
        return {
            "status": "not_ready",
            "completed_records": 0,
            "target_completed_records": rules["target_completed_records"],
            "checks": {"minimum_records": False, "final_oos_unlocked": False},
        }

    left = decisions.copy()
    right = candidate_paths.copy()
    left["signal_date"] = pd.to_datetime(left["signal_date"]).dt.date
    right["signal_date"] = pd.to_datetime(right["signal_date"]).dt.date
    start = date.fromisoformat(rules["collection_start"])
    left = left.loc[left["signal_date"].ge(start)].copy()
    if "strategy_version" not in left:
        left["strategy_version"] = None
    merged = left.merge(
        right,
        on=["signal_date", "ticker", "signal_type"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_path"),
    )
    completed = merged.loc[merged["net_return"].notna()].copy()
    by_signal = {
        str(signal): int(count)
        for signal, count in completed.groupby("signal_type").size().items()
    }
    required_signals = {"capitulation_reversal", "first_pullback"}
    minimum_per_signal = int(rules["minimum_completed_records_per_signal"])
    checks = {
        "strategy_unchanged": bool(
            not left.empty
            and left["strategy_version"].eq(rules["strategy_version"]).all()
        ),
        "minimum_records": len(completed)
        >= int(rules["minimum_completed_records"]),
        "minimum_per_signal": all(
            by_signal.get(signal, 0) >= minimum_per_signal
            for signal in required_signals
        ),
        "final_oos_unlocked": bool(oos["performance_metrics_visible"]),
    }
    passed = all(checks.values())
    return {
        "status": "ready_for_manual_risk_review" if passed else "not_ready",
        "automated_live_trading_authorized": False,
        "decision_records": len(left),
        "completed_records": len(completed),
        "minimum_completed_records": rules["minimum_completed_records"],
        "target_completed_records": rules["target_completed_records"],
        "completed_by_signal": by_signal,
        "checks": checks,
        "oos_access": oos,
    }
