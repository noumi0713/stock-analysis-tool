from __future__ import annotations

from datetime import date

from app.evaluation_protocol import load_evaluation_protocol, oos_access_status


def test_frozen_oos_protocol_rejects_old_history_as_true_oos() -> None:
    protocol = load_evaluation_protocol()

    assert protocol["historical_contamination"]["used_during_rule_development"]
    assert not protocol["historical_contamination"]["eligible_as_true_out_of_sample"]


def test_oos_metrics_remain_sealed_until_final_unlock() -> None:
    protocol = load_evaluation_protocol()

    collecting = oos_access_status(as_of=date(2026, 12, 1), protocol=protocol)
    interim = oos_access_status(as_of=date(2027, 3, 1), protocol=protocol)
    final = oos_access_status(as_of=date(2027, 8, 31), protocol=protocol)

    assert collecting["status"] == "sealed_collecting"
    assert interim["status"] == "interim_sample_check_only"
    assert not interim["performance_metrics_visible"]
    assert final["status"] == "unlocked_for_final_evaluation"
    assert final["performance_metrics_visible"]
