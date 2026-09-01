from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from app.signal_audit import build_strategy_audit, write_gzip_json


def test_condition_funnel_distinguishes_failures_missing_and_ranking_limit() -> None:
    latest = pd.DataFrame(
        [
            {"ticker": "1001.T", "score": 3.0, "metric": 10.0},
            {"ticker": "1002.T", "score": 2.0, "metric": 9.0},
            {"ticker": "1003.T", "score": 1.0, "metric": 8.0},
            {"ticker": "1004.T", "score": 0.0, "metric": float("nan")},
        ]
    )
    conditions = pd.DataFrame(
        {
            "first": [True, True, False, False],
            "second": [True, True, True, False],
        },
        index=latest.index,
    )
    ranking = pd.DataFrame(
        [
            {"ticker": "1001.T", "rank": 1, "ranking_value": 3.0},
            {"ticker": "1002.T", "rank": 2, "ranking_value": 2.0},
        ]
    )
    audit = build_strategy_audit(
        latest,
        signal_type="test",
        strategy_name="test strategy",
        condition_results=conditions,
        condition_actuals={
            "first": latest["metric"],
            "second": latest["score"],
        },
        condition_requirements={"first": "metric > 0", "second": "score > 0"},
        selected_tickers=["1001.T"],
        ranking=ranking,
        ranking_definition={"meaning": "ordinal_selection_only"},
        metric_keys=["metric", "score"],
        audit_context={"audit_id": "audit-1"},
        missing_tickers=["1005.T"],
    )

    assert audit["eligible_before_ranking"] == 2
    assert audit["selected_count"] == 1
    assert audit["detection_status"] == "matches_found"
    rows = {row["ticker"]: row for row in audit["candidates"]}
    assert rows["1001.T"]["final_decision"] == "detected"
    assert rows["1002.T"]["final_decision"] == "excluded_ranking_limit"
    assert rows["1003.T"]["exclusion_reasons"] == ["first"]
    assert rows["1004.T"]["final_decision"] == "excluded_data_insufficient"
    assert rows["1005.T"]["data_status"] == "missing_latest_daily_bar"
    assert audit["condition_funnel_sequential"][1]["input"] == 2


def test_gzip_audit_output_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json.gz"
    second = tmp_path / "second.json.gz"
    payload = {"schema_version": 1, "rows": [{"ticker": "1001.T"}]}

    first_metadata = write_gzip_json(payload, first)
    second_metadata = write_gzip_json(payload, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_metadata["sha256"] == second_metadata["sha256"]
    with gzip.open(first, "rt", encoding="utf-8") as source:
        assert json.load(source) == payload
