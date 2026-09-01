from __future__ import annotations

import pandas as pd

from app.candidate_outcomes import track_ten_session_outcomes


def _prices(periods: int = 12) -> pd.DataFrame:
    rows = []
    for index, day in enumerate(pd.date_range("2026-09-01", periods=periods, freq="B")):
        price = 100.0 + index
        rows.append(
            {
                "ticker": "1001.T",
                "date": day.date(),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": 100_000,
                "stock_splits": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _decisions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_date": "2026-09-01",
                "ticker": "1001.T",
                "signal_type": "capitulation_reversal",
                "primary_rank": 1,
                "classification": "C",
                "secondary_action": "skip",
                "decision_audit_id": "audit-1",
                "primary_signal_snapshot": {"close": 100.0},
            }
        ]
    )


def test_tracks_skipped_candidate_for_ten_sessions() -> None:
    result = track_ten_session_outcomes(
        _decisions(),
        _prices(),
        computed_at="2026-09-20T00:00:00+00:00",
        git_commit="abc123",
    )

    outcome = result["outcomes"][0]
    assert result["completed_count"] == 1
    assert outcome["classification"] == "C"
    assert outcome["status"] == "completed"
    assert outcome["entry_date"] == "2026-09-02"
    assert outcome["observation_end_date"] == "2026-09-15"
    assert outcome["all_primary_buy_comparison_eligible"] is True
    assert outcome["unfiltered_return_10_sessions"] > 0


def test_outcome_remains_pending_until_ten_sessions_exist() -> None:
    result = track_ten_session_outcomes(_decisions(), _prices(periods=5))

    assert result["completed_count"] == 0
    assert result["outcomes"][0]["status"] == "pending_10_sessions"
