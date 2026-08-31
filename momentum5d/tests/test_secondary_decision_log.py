from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from app.secondary_decision_log import (
    DecisionLogError,
    build_decision_record,
    evaluate_decisions,
    write_decision_record,
)


def _primary() -> dict:
    return {
        "strategy_version": "live_v1_2026-08-31",
        "data_quality": {"status": "certified", "split_adjusted_volume": True},
        "date": "2026-08-28",
        "update": {
            "status": "complete",
            "session": "close",
            "market_date": "2026-08-28",
        },
        "signals": [
            {"ticker": "1111.T", "rank": 1, "close": 1_000},
        ],
        "pullback_signals": [
            {"ticker": "2222.T", "rank": 1, "close": 2_000},
        ],
    }


def _decision(ticker: str, signal_type: str, classification: str) -> dict:
    return {
        "ticker": ticker,
        "signal_type": signal_type,
        "classification": classification,
        "market_assessment": "neutral",
        "sector_theme_assessment": "supportive",
        "news_assessment": "neutral",
        "data_fresh": True,
        "signal_shape_intact": True,
        "major_event_risk": False,
        "reasons": ["地合い", "テーマ", "ニュース"],
        "maximum_risk": "翌日に出来高を伴って安値を割ること",
        "next_session_entry_condition": "寄り付きギャップ条件内でのみ執行",
    }


def _analysis() -> dict:
    return {
        "signal_date": "2026-08-28",
        "evaluated_at": "2026-08-31T08:00:00+09:00",
        "source_snapshot": {
            "checked_at": "2026-08-31T07:55:00+09:00",
            "sources": ["https://example.test/market"],
        },
        "decisions": [
            _decision("1111.T", "capitulation_reversal", "B"),
            _decision("2222.T", "first_pullback", "A"),
        ],
    }


def test_build_decision_record_covers_every_primary_candidate() -> None:
    record = build_decision_record(_primary(), _analysis())

    assert record["signal_date"] == "2026-08-28"
    assert len(record["decisions"]) == 2
    assert len(record["primary_signal_sha256"]) == 64


def test_a_classification_rejects_unknown_or_adverse_inputs() -> None:
    analysis = _analysis()
    analysis["decisions"][1]["news_assessment"] = "unknown"

    with pytest.raises(DecisionLogError, match="unresolved blocker"):
        build_decision_record(_primary(), analysis)


def test_decision_log_is_append_only(tmp_path) -> None:
    record = build_decision_record(_primary(), _analysis())
    path = write_decision_record(record, tmp_path)
    changed = deepcopy(record)
    changed["decisions"][0]["classification"] = "C"

    assert write_decision_record(record, tmp_path) == path
    with pytest.raises(DecisionLogError, match="append-only"):
        write_decision_record(changed, tmp_path)


def test_decision_evaluation_keeps_non_executable_candidates() -> None:
    decisions = pd.DataFrame(
        [
            {
                "signal_date": "2026-08-28",
                "ticker": "1111.T",
                "signal_type": "capitulation_reversal",
                "classification": "A",
            },
            {
                "signal_date": "2026-08-28",
                "ticker": "2222.T",
                "signal_type": "first_pullback",
                "classification": "C",
            },
        ]
    )
    paths = pd.DataFrame(
        [
            {
                "signal_date": "2026-08-28",
                "ticker": "1111.T",
                "signal_type": "capitulation_reversal",
                "net_return": 0.1,
            }
        ]
    )

    result = evaluate_decisions(decisions, paths)

    assert result["decision_count"] == 2
    assert result["entry_eligible_count"] == 1
    assert result["by_classification"]["A"]["mean_net_return"] == pytest.approx(
        0.1
    )
    assert result["by_classification"]["C"]["mean_net_return"] is None
