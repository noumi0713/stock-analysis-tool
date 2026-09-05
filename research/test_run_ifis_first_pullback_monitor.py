from run_ifis_first_pullback_monitor import discovery_index


def rows():
    return [
        {"date": "2026-09-02"},
        {"date": "2026-09-03"},
        {"date": "2026-09-04"},
    ]


def test_preclose_snapshot_uses_prior_market_day():
    assert discovery_index(rows(), "2026-09-04T01:20:00+09:00") == 1


def test_afterclose_snapshot_can_use_same_market_day():
    assert discovery_index(rows(), "2026-09-04T16:20:00+09:00") == 2
