from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.production_readiness import atomic_write_json


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify rollback without changing production")
    parser.add_argument("--dashboard-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = _read(args.manifest)
    current = manifest.get("current") or {}
    snapshot = args.dashboard_root / str(current.get("snapshot_path") or "")
    errors: list[str] = []
    if manifest.get("rollback_supported") is not True:
        errors.append("rollback_supported is not true")
    if not snapshot.is_file():
        errors.append(f"snapshot missing: {snapshot}")
        payload: dict[str, Any] = {}
    else:
        payload = _read(snapshot)
    if payload and payload.get("date") != current.get("market_date"):
        errors.append("snapshot market_date mismatch")
    if payload and payload.get("strategy_version") != current.get("strategy_config_version"):
        errors.append("snapshot strategy version mismatch")
    if payload and payload.get("schema_version") != current.get("data_schema_version"):
        errors.append("snapshot schema version mismatch")
    if not current.get("git_commit_id") or not current.get("strategy_config_sha256"):
        errors.append("rollback identity is incomplete")
    result = {
        **manifest,
        "dry_run": {
            "status": "PASS" if not errors else "FAIL",
            "snapshot_resolved": str(snapshot),
            "production_mutated": False,
            "errors": errors,
            "procedure": "deploy recorded git commit and restore recorded snapshot/config",
        },
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result["dry_run"], ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
