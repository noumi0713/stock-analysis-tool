from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

FROZEN_PROTOCOL_VERSION = "oos_v1_2026-08-31"
FROZEN_PROTOCOL_SHA256 = "1c1347e70c276fa707ea6201bdb40e73d16e008e25cd3d2f6a409395db97ab82"
DEFAULT_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "evaluation_protocols"
    / "oos_v1_2026-08-31.json"
)


class EvaluationProtocolError(ValueError):
    """Raised when the sealed OOS protocol is invalid or opened too early."""


def load_evaluation_protocol(path: Path | None = None) -> dict[str, Any]:
    protocol_path = path or DEFAULT_PROTOCOL_PATH
    if not protocol_path.exists():
        raise EvaluationProtocolError(f"Evaluation protocol is missing: {protocol_path}")
    if protocol_path.resolve() == DEFAULT_PROTOCOL_PATH.resolve():
        digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        if digest != FROZEN_PROTOCOL_SHA256:
            raise EvaluationProtocolError(
                "Frozen evaluation protocol changed without a new version: "
                f"expected={FROZEN_PROTOCOL_SHA256} actual={digest}"
            )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != FROZEN_PROTOCOL_VERSION:
        raise EvaluationProtocolError("Unexpected evaluation protocol version")
    if protocol.get("status") != "frozen":
        raise EvaluationProtocolError("Evaluation protocol must be frozen")
    if protocol.get("strategy_version") != "live_v1_2026-08-31":
        raise EvaluationProtocolError("Evaluation protocol strategy mismatch")
    if protocol["historical_contamination"].get("eligible_as_true_out_of_sample"):
        raise EvaluationProtocolError("Previously tuned history cannot be true OOS")
    return protocol


def oos_access_status(*, as_of: date, protocol: dict[str, Any]) -> dict[str, Any]:
    rules = protocol["prospective_out_of_sample"]
    interim = date.fromisoformat(rules["minimum_interim_date"])
    unlock = date.fromisoformat(rules["final_unlock_date"])
    if as_of < interim:
        status = "sealed_collecting"
    elif as_of < unlock:
        status = "interim_sample_check_only"
    else:
        status = "unlocked_for_final_evaluation"
    return {
        "status": status,
        "as_of": as_of.isoformat(),
        "oos_start": rules["start_date"],
        "minimum_interim_date": interim.isoformat(),
        "final_unlock_date": unlock.isoformat(),
        "performance_metrics_visible": as_of >= unlock,
    }
