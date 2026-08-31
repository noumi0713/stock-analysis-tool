from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from app.forward_validation import assess_forward_readiness
from app.secondary_decision_log import load_decisions


def main() -> int:
    parser = argparse.ArgumentParser(description="Check forward-validation readiness")
    parser.add_argument("--decision-log-dir", type=Path, required=True)
    parser.add_argument("--candidate-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    decisions = load_decisions(args.decision_log_dir)
    paths = pd.concat(
        [pd.read_csv(path) for path in args.candidate_paths], ignore_index=True
    )
    result = assess_forward_readiness(decisions, paths, as_of=args.as_of)
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
