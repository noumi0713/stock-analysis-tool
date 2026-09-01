from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import UTC, datetime
from typing import Any

from app.live_strategy import FROZEN_STRATEGY_SHA256


def resolve_git_commit(explicit: str | None = None) -> str:
    value = (explicit or os.environ.get("GITHUB_SHA") or "").strip()
    if value:
        return value
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def build_audit_context(
    *,
    market_date: str,
    acquired_at: str,
    strategy_version: str,
    strategy_names: list[str],
    snapshot_fingerprint: str,
    computed_at: str | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    stamp = computed_at or datetime.now(UTC).isoformat()
    core = {
        "data_final_market_date": market_date,
        "data_acquired_at": acquired_at,
        "strategy_config_version": strategy_version,
        "strategy_config_sha256": FROZEN_STRATEGY_SHA256,
        "git_commit_id": resolve_git_commit(git_commit),
        "calculation_executed_at": stamp,
        "strategy_names": sorted(strategy_names),
        "market_snapshot_fingerprint": snapshot_fingerprint,
    }
    canonical = "|".join(str(core[key]) for key in sorted(core))
    core["audit_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return core
