from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.production_readiness import atomic_write_json


def _read(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two certified backtest runs")
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = _read(args.first)
    second = _read(args.second)
    first_hash = first.get("deterministic_fingerprint")
    second_hash = second.get("deterministic_fingerprint")
    matched = bool(first_hash) and first_hash == second_hash
    first["reproducibility"] = {
        "status": "PASS" if matched else "FAIL",
        "deterministic_fingerprint": first_hash,
        "comparison_fingerprint": second_hash,
        "same_input_sha256": (
            (first.get("audit_context") or {}).get("input_file_sha256")
            == (second.get("audit_context") or {}).get("input_file_sha256")
        ),
        "same_strategy_sha256": (
            (first.get("audit_context") or {}).get("strategy_config_sha256")
            == (second.get("audit_context") or {}).get("strategy_config_sha256")
        ),
        "same_git_commit": (
            (first.get("audit_context") or {}).get("git_commit_id")
            == (second.get("audit_context") or {}).get("git_commit_id")
        ),
    }
    atomic_write_json(args.output, first)
    print(json.dumps(first["reproducibility"], ensure_ascii=False))
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
