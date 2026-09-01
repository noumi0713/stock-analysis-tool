from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _row_metrics(row: pd.Series, metric_keys: list[str]) -> dict[str, object]:
    return {key: _value(row.get(key)) for key in metric_keys}


def build_strategy_audit(
    latest: pd.DataFrame,
    *,
    signal_type: str,
    strategy_name: str,
    condition_results: pd.DataFrame,
    condition_actuals: dict[str, pd.Series],
    condition_requirements: dict[str, str],
    selected_tickers: list[str],
    ranking: pd.DataFrame,
    ranking_definition: dict[str, Any],
    metric_keys: list[str],
    audit_context: dict[str, Any],
    missing_tickers: list[str] | None = None,
    company_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    condition_order = list(condition_results.columns)
    if set(condition_order) != set(condition_actuals):
        raise ValueError(f"Condition actuals mismatch for {signal_type}")
    if not latest.index.equals(condition_results.index):
        raise ValueError(f"Condition result index mismatch for {signal_type}")

    missing_by_condition = pd.DataFrame(
        {key: condition_actuals[key].isna() for key in condition_order},
        index=latest.index,
    )
    valid_condition_results = condition_results.astype(bool) & ~missing_by_condition
    any_missing = missing_by_condition.any(axis=1)
    eligible = valid_condition_results.all(axis=1) & ~any_missing

    independent = []
    stages = []
    remaining = pd.Series(True, index=latest.index)
    for key in condition_order:
        missing = missing_by_condition[key]
        passed = valid_condition_results[key]
        independent.append(
            {
                "condition": key,
                "requirement": condition_requirements[key],
                "evaluated": int((~missing).sum()),
                "passed": int((passed & ~missing).sum()),
                "failed": int((~passed & ~missing).sum()),
                "data_missing": int(missing.sum()),
            }
        )
        stage_input = remaining.copy()
        stage_missing = stage_input & missing
        stage_passed = stage_input & ~missing & passed
        stage_failed = stage_input & ~missing & ~passed
        stages.append(
            {
                "condition": key,
                "requirement": condition_requirements[key],
                "input": int(stage_input.sum()),
                "passed": int(stage_passed.sum()),
                "failed": int(stage_failed.sum()),
                "data_missing": int(stage_missing.sum()),
            }
        )
        remaining = stage_passed

    rank_by_ticker = {
        str(row.ticker): {
            key: _value(getattr(row, key))
            for key in ranking.columns
            if key != "ticker"
        }
        for row in ranking.itertuples(index=False)
    }
    selected = set(selected_tickers)
    names = company_names or {}
    rows: list[dict[str, Any]] = []
    for index, row in latest.iterrows():
        ticker = str(row["ticker"])
        code = str(row.get("code") or ticker.removesuffix(".T"))[:4]
        conditions = []
        failed_reasons: list[str] = []
        missing_reasons: list[str] = []
        for key in condition_order:
            missing = bool(missing_by_condition.at[index, key])
            passed = bool(valid_condition_results.at[index, key]) if not missing else None
            if missing:
                missing_reasons.append(f"missing:{key}")
            elif not passed:
                failed_reasons.append(key)
            conditions.append(
                {
                    "key": key,
                    "requirement": condition_requirements[key],
                    "actual": _value(condition_actuals[key].at[index]),
                    "passed": passed,
                }
            )
        is_eligible = bool(eligible.at[index])
        if ticker in selected:
            final_decision = "detected"
            exclusion_reasons: list[str] = []
        elif missing_reasons:
            final_decision = "excluded_data_insufficient"
            exclusion_reasons = missing_reasons
        elif failed_reasons:
            final_decision = "excluded_conditions"
            exclusion_reasons = failed_reasons
        elif is_eligible:
            final_decision = "excluded_ranking_limit"
            exclusion_reasons = ["rank_below_maximum_candidates_per_day"]
        else:
            raise AssertionError(f"Unclassified audit row: {signal_type} {ticker}")
        rows.append(
            {
                "audit_id": audit_context["audit_id"],
                "strategy_name": strategy_name,
                "signal_type": signal_type,
                "ticker": ticker,
                "code": code,
                "name": names.get(code) or ticker,
                "data_status": "insufficient" if missing_reasons else "complete",
                "prices_and_indicators": _row_metrics(row, metric_keys),
                "conditions": conditions,
                "passed_condition_count": sum(item["passed"] is True for item in conditions),
                "total_condition_count": len(conditions),
                "eligible_before_ranking": is_eligible,
                "ranking": rank_by_ticker.get(ticker),
                "exclusion_reasons": exclusion_reasons,
                "final_decision": final_decision,
            }
        )

    for ticker in sorted(missing_tickers or []):
        code = ticker.removesuffix(".T")[:4]
        rows.append(
            {
                "audit_id": audit_context["audit_id"],
                "strategy_name": strategy_name,
                "signal_type": signal_type,
                "ticker": ticker,
                "code": code,
                "name": names.get(code) or ticker,
                "data_status": "missing_latest_daily_bar",
                "prices_and_indicators": {key: None for key in metric_keys},
                "conditions": [
                    {
                        "key": key,
                        "requirement": condition_requirements[key],
                        "actual": None,
                        "passed": None,
                    }
                    for key in condition_order
                ],
                "passed_condition_count": 0,
                "total_condition_count": len(condition_order),
                "eligible_before_ranking": False,
                "ranking": None,
                "exclusion_reasons": ["missing_latest_daily_bar"],
                "final_decision": "excluded_data_insufficient",
            }
        )

    missing_count = int(any_missing.sum()) + len(missing_tickers or [])
    selected_count = len(selected)
    if selected_count:
        detection_status = "matches_found"
    elif missing_count:
        detection_status = "no_matches_with_data_gaps"
    else:
        detection_status = "no_matches_complete_universe"
    return {
        "strategy_name": strategy_name,
        "signal_type": signal_type,
        "detection_status": detection_status,
        "evaluated_tickers": len(latest),
        "missing_or_insufficient_tickers": missing_count,
        "eligible_before_ranking": int(eligible.sum()),
        "selected_count": selected_count,
        "condition_order": condition_order,
        "condition_counts_independent": independent,
        "condition_funnel_sequential": stages,
        "ranking_definition": ranking_definition,
        "candidates": sorted(rows, key=lambda item: item["ticker"]),
    }


def build_audit_bundle(
    *,
    audit_context: dict[str, Any],
    certification: dict[str, Any],
    strategies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "audit_context": audit_context,
        "universe": {
            "expected_tickers": int(certification["expected_tickers"]),
            "successful_tickers": int(certification["successful_tickers"]),
            "coverage": float(certification["coverage"]),
            "missing_tickers": sorted(certification.get("missing_tickers") or []),
        },
        "strategies": strategies,
    }


def write_gzip_json(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as output:
            output.write(encoded)
    temporary.replace(path)
    return {
        "format": "json.gz",
        "path": path.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "uncompressed_bytes": len(encoded),
        "compressed_bytes": path.stat().st_size,
    }
