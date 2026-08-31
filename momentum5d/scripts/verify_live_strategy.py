from __future__ import annotations

from app.evaluation_protocol import (
    FROZEN_PROTOCOL_SHA256,
    load_evaluation_protocol,
)
from app.forward_validation import FROZEN_FORWARD_GATE_SHA256, load_forward_gate
from app.live_strategy import FROZEN_STRATEGY_SHA256, load_frozen_strategy


def main() -> int:
    spec = load_frozen_strategy()
    protocol = load_evaluation_protocol()
    forward_gate = load_forward_gate()
    print(
        "Frozen live strategy verified: "
        f"version={spec['strategy_version']} sha256={FROZEN_STRATEGY_SHA256}"
    )
    print(
        "Frozen evaluation controls verified: "
        f"oos={protocol['protocol_version']} sha256={FROZEN_PROTOCOL_SHA256} "
        f"forward={forward_gate['protocol_version']} "
        f"sha256={FROZEN_FORWARD_GATE_SHA256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
