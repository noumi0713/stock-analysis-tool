from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FROZEN_STRATEGY_VERSION = "live_v1_2026-08-31"
FROZEN_STRATEGY_SHA256 = "d505dc664986c4ad9741bb3354e87403b0ebadc7f4906f8a04c88cc72e0d834b"
DEFAULT_STRATEGY_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "strategy_specs"
    / "live_v1_2026-08-31.json"
)


class StrategySpecError(ValueError):
    """Raised when the frozen live strategy specification is invalid."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_strategy_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != 1:
        raise StrategySpecError("Unsupported strategy schema_version")
    if spec.get("strategy_version") != FROZEN_STRATEGY_VERSION:
        raise StrategySpecError("The strategy version is not the frozen live version")
    if spec.get("status") != "frozen":
        raise StrategySpecError("The live strategy must have status=frozen")

    portfolio = spec.get("portfolio") or {}
    if portfolio.get("fixed_loss_yen_limit_enabled") is not False:
        raise StrategySpecError("A fixed yen loss limit must remain disabled")
    if not 0 < float(portfolio.get("maximum_gross_exposure", 0)) <= 1:
        raise StrategySpecError("maximum_gross_exposure must be in (0, 1]")
    if int(portfolio.get("maximum_open_positions", 0)) < 1:
        raise StrategySpecError("maximum_open_positions must be positive")

    required_signals = {"capitulation_reversal", "first_pullback"}
    signals = spec.get("signals") or {}
    if required_signals.difference(signals):
        raise StrategySpecError("Both production signal definitions are required")
    for signal_name in required_signals:
        signal = signals[signal_name]
        execution = signal.get("execution") or {}
        if execution.get("entry") != "next_session_open":
            raise StrategySpecError(f"{signal_name}: entry must be next_session_open")
        if execution.get("same_bar_priority") != "stop_loss":
            raise StrategySpecError(f"{signal_name}: same-bar priority must be stop_loss")
        if int(signal.get("maximum_candidates_per_day", 0)) < 1:
            raise StrategySpecError(f"{signal_name}: maximum candidates must be positive")


def load_frozen_strategy(path: Path | None = None) -> dict[str, Any]:
    strategy_path = path or DEFAULT_STRATEGY_PATH
    if not strategy_path.exists():
        raise StrategySpecError(f"Strategy specification is missing: {strategy_path}")
    if strategy_path.resolve() == DEFAULT_STRATEGY_PATH.resolve():
        digest = _sha256(strategy_path)
        if digest != FROZEN_STRATEGY_SHA256:
            raise StrategySpecError(
                "Frozen strategy specification changed without a new version: "
                f"expected={FROZEN_STRATEGY_SHA256} actual={digest}"
            )
    spec = json.loads(strategy_path.read_text(encoding="utf-8"))
    validate_strategy_spec(spec)
    return spec
