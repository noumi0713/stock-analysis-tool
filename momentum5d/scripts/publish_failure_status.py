from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an explicit unavailable payload instead of serving stale signals"
    )
    parser.add_argument("--market-date", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--error", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    errors = args.error or ["production pipeline failed before publication"]
    stamp = datetime.now(UTC).isoformat()
    payload = {
        "schema_version": 4,
        "date": args.market_date,
        "latest_date": args.market_date,
        "generated_at": stamp,
        "update": {
            "status": "failed",
            "session": "close",
            "market_date": args.market_date,
            "generated_at": stamp,
            "message": "更新失敗・本日データ未確定",
        },
        "data_quality": {"status": "rejected"},
        "git_commit_id": args.git_commit,
        "errors": errors,
        "total_signal_count": 0,
        "signal_count": 0,
        "signals": [],
        "pullback_signal_count": 0,
        "pullback_signals": [],
        "near_miss_count": 0,
        "near_misses": [],
        "stale_result_fallback_allowed": False,
    }
    _atomic_write_json(args.output_dir / "latest_signals.json", payload)
    _atomic_write_json(
        args.output_dir / "production-status.json",
        {
            "schema_version": 1,
            "status": "failed",
            "complete": False,
            "market_date": args.market_date,
            "calculation_completed_at": stamp,
            "git_commit_id": args.git_commit,
            "errors": errors,
            "publication": {
                "status": "blocked",
                "stale_result_fallback_allowed": False,
            },
        },
    )
    _atomic_write_json(
        args.output_dir / "monitoring" / "latest.json",
        {
            "schema_version": 1,
            "market_date": args.market_date,
            "status": "failed",
            "data_download": "unknown",
            "signal_counts": {
                "capitulation_reversal": 0,
                "first_pullback": 0,
            },
            "zero_signal_reason": {
                "capitulation_reversal": "processing_failed",
                "first_pullback": "processing_failed",
            },
            "anomalies": errors,
            "publication": "blocked",
            "stale_result_fallback_allowed": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
