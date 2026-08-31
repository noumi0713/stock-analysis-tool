from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.secondary_decision_log import build_decision_record, write_decision_record


def main() -> int:
    parser = argparse.ArgumentParser(description="Record an immutable A/B/C decision log")
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    primary = json.loads(args.signals.read_text(encoding="utf-8"))
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    record = build_decision_record(primary, analysis)
    output = write_decision_record(record, args.output_dir)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
