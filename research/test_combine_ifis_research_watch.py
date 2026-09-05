from combine_ifis_research_watch import disposition


def test_pass_plus_forming_is_watch_forming():
    assert disposition("PASS", "PULLBACK_FORMING") == "WATCH_FORMING"


def test_pass_plus_signal_is_priority_review():
    assert disposition("PASS", "FIRST_PULLBACK_SIGNAL") == "PRIORITY_REVIEW"


def test_caution_never_becomes_priority():
    assert disposition("CAUTION", "PULLBACK_FORMING") == "SECONDARY_FORMING"
    assert disposition("CAUTION", "WAIT_FIRST_PULLBACK") == "SECONDARY_WATCH"


def test_technical_reject_overrides_catalyst():
    assert disposition("PASS", "INVALIDATED") == "TECHNICAL_REJECT"
    assert disposition("PASS", "EXCLUDE_RAN_AWAY") == "TECHNICAL_REJECT"


def test_failed_catalyst_holds_for_evidence():
    assert disposition("FAIL", "WAIT_FIRST_PULLBACK") == "HOLD_FOR_EVIDENCE"
