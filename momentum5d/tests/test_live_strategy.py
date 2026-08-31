from __future__ import annotations

import json

import pytest

from app.live_strategy import (
    DEFAULT_STRATEGY_PATH,
    StrategySpecError,
    load_frozen_strategy,
)


def test_frozen_live_strategy_is_valid() -> None:
    spec = load_frozen_strategy()

    assert spec["status"] == "frozen"
    assert spec["portfolio"]["fixed_loss_yen_limit_enabled"] is False
    assert set(spec["signals"]) == {"capitulation_reversal", "first_pullback"}


def test_modified_frozen_strategy_is_rejected(tmp_path) -> None:
    spec = json.loads(DEFAULT_STRATEGY_PATH.read_text(encoding="utf-8"))
    spec["status"] = "draft"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(StrategySpecError, match="status=frozen"):
        load_frozen_strategy(changed)


def test_fixed_yen_loss_limit_cannot_be_reenabled(tmp_path) -> None:
    spec = json.loads(DEFAULT_STRATEGY_PATH.read_text(encoding="utf-8"))
    spec["portfolio"]["fixed_loss_yen_limit_enabled"] = True
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(StrategySpecError, match="fixed yen loss limit"):
        load_frozen_strategy(changed)
