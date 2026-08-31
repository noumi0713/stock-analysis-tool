from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.live_strategy import load_frozen_strategy

SIGNAL_ARRAYS = {
    "capitulation_reversal": "signals",
    "first_pullback": "pullback_signals",
}
CLASSIFICATIONS = {"A", "B", "C"}
ASSESSMENTS = {"supportive", "neutral", "adverse", "unknown"}


class DecisionLogError(ValueError):
    """Raised when a secondary-selection decision is not auditable."""


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _primary_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for signal_type, key in SIGNAL_ARRAYS.items():
        for row in payload.get(key) or []:
            candidates.append(
                {
                    "ticker": str(row["ticker"]),
                    "signal_type": signal_type,
                    "rank": int(row["rank"]),
                    "primary_signal": row,
                }
            )
    return candidates


def build_decision_record(
    primary_payload: dict[str, Any],
    analysis: dict[str, Any],
    *,
    strategy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = strategy or load_frozen_strategy()
    signal_date = str(primary_payload.get("date") or "")
    update = primary_payload.get("update") or {}
    if update.get("status") != "complete" or update.get("session") != "close":
        raise DecisionLogError("Primary signals must be completed close-session data")
    if str(update.get("market_date") or "") != signal_date:
        raise DecisionLogError("Primary signal date and market date do not match")
    if primary_payload.get("strategy_version") != spec["strategy_version"]:
        raise DecisionLogError("Primary signals do not use the frozen strategy version")
    if (primary_payload.get("data_quality") or {}).get("status") != "certified":
        raise DecisionLogError("Primary signal data quality is not certified")
    if str(analysis.get("signal_date") or "") != signal_date:
        raise DecisionLogError("Analysis signal_date does not match primary signals")

    candidates = _primary_candidates(primary_payload)
    expected = {(row["ticker"], row["signal_type"]): row for row in candidates}
    supplied = analysis.get("decisions") or []
    supplied_keys = [
        (str(row.get("ticker") or ""), str(row.get("signal_type") or ""))
        for row in supplied
    ]
    counts = Counter(supplied_keys)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise DecisionLogError(f"Duplicate decisions: {duplicates}")
    if set(supplied_keys) != set(expected):
        missing = sorted(set(expected).difference(supplied_keys))
        extra = sorted(set(supplied_keys).difference(expected))
        raise DecisionLogError(f"Decision coverage mismatch missing={missing} extra={extra}")

    evaluated_at = str(analysis.get("evaluated_at") or "")
    if not evaluated_at:
        raise DecisionLogError("evaluated_at is required")
    source_snapshot = analysis.get("source_snapshot") or {}
    if not source_snapshot.get("checked_at") or not source_snapshot.get("sources"):
        raise DecisionLogError("Timestamped market/news sources are required")

    normalized: list[dict[str, Any]] = []
    a_count = 0
    for supplied_row in supplied:
        key = (str(supplied_row["ticker"]), str(supplied_row["signal_type"]))
        primary = expected[key]
        label = str(supplied_row.get("classification") or "")
        if label not in CLASSIFICATIONS:
            raise DecisionLogError(f"Invalid classification for {key}: {label}")
        assessments = {
            name: str(supplied_row.get(name) or "unknown")
            for name in (
                "market_assessment",
                "sector_theme_assessment",
                "news_assessment",
            )
        }
        invalid = set(assessments.values()).difference(ASSESSMENTS)
        if invalid:
            raise DecisionLogError(f"Invalid assessment for {key}: {sorted(invalid)}")
        reasons = supplied_row.get("reasons") or []
        if len(reasons) < 3:
            raise DecisionLogError(f"At least three reasons are required for {key}")
        if not supplied_row.get("maximum_risk"):
            raise DecisionLogError(f"maximum_risk is required for {key}")
        if not supplied_row.get("next_session_entry_condition"):
            raise DecisionLogError(f"next-session entry condition is required for {key}")

        if label == "A":
            a_count += 1
            blockers = [
                not bool(supplied_row.get("data_fresh")),
                not bool(supplied_row.get("signal_shape_intact")),
                bool(supplied_row.get("major_event_risk")),
                any(value in {"adverse", "unknown"} for value in assessments.values()),
            ]
            if any(blockers):
                raise DecisionLogError(f"A classification has an unresolved blocker: {key}")

        normalized.append(
            {
                "ticker": key[0],
                "signal_type": key[1],
                "primary_rank": primary["rank"],
                "classification": label,
                **assessments,
                "data_fresh": bool(supplied_row.get("data_fresh")),
                "signal_shape_intact": bool(
                    supplied_row.get("signal_shape_intact")
                ),
                "major_event_risk": bool(supplied_row.get("major_event_risk")),
                "reasons": [str(reason) for reason in reasons],
                "maximum_risk": str(supplied_row["maximum_risk"]),
                "next_session_entry_condition": str(
                    supplied_row["next_session_entry_condition"]
                ),
            }
        )
    maximum_a = int(spec["secondary_selection"]["maximum_a_candidates"])
    if a_count > maximum_a:
        raise DecisionLogError(f"A candidates exceed frozen maximum: {a_count}>{maximum_a}")

    return {
        "schema_version": 1,
        "strategy_version": spec["strategy_version"],
        "signal_date": signal_date,
        "evaluated_at": evaluated_at,
        "primary_signal_sha256": _canonical_digest(primary_payload),
        "primary_data_quality": primary_payload["data_quality"],
        "source_snapshot": source_snapshot,
        "decision_policy": "secondary_only_does_not_change_primary_signal",
        "decisions": sorted(
            normalized,
            key=lambda row: (row["signal_type"], row["primary_rank"], row["ticker"]),
        ),
    }


def write_decision_record(record: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{record['signal_date']}.json"
    encoded = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != encoded:
            raise DecisionLogError(f"Decision log is append-only and already exists: {output}")
        return output
    output.write_text(encoded, encoding="utf-8")
    return output


def load_decisions(log_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for decision in payload.get("decisions") or []:
            rows.append(
                {
                    "signal_date": payload["signal_date"],
                    "strategy_version": payload["strategy_version"],
                    **decision,
                }
            )
    return pd.DataFrame(rows)


def evaluate_decisions(
    decisions: pd.DataFrame, candidate_paths: pd.DataFrame
) -> dict[str, Any]:
    if decisions.empty:
        return {"status": "insufficient_forward_records", "decision_count": 0}
    left = decisions.copy()
    right = candidate_paths.copy()
    left["signal_date"] = pd.to_datetime(left["signal_date"]).dt.date
    right["signal_date"] = pd.to_datetime(right["signal_date"]).dt.date
    merged = left.merge(
        right,
        on=["signal_date", "ticker", "signal_type"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_path"),
    )
    by_class: dict[str, dict[str, Any]] = {}
    for label, group in merged.groupby("classification", sort=True):
        eligible = group.loc[group["net_return"].notna()]
        by_class[str(label)] = {
            "decisions": len(group),
            "entry_eligible": len(eligible),
            "mean_net_return": float(eligible["net_return"].mean())
            if not eligible.empty
            else None,
            "win_rate": float(eligible["net_return"].gt(0).mean())
            if not eligible.empty
            else None,
        }
    return {
        "status": "completed",
        "decision_count": len(merged),
        "entry_eligible_count": int(merged["net_return"].notna().sum()),
        "by_classification": by_class,
        "evaluated_at": datetime.now(UTC).isoformat(),
    }
