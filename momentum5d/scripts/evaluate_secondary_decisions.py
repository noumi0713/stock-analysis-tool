from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.secondary_decision_log import evaluate_decisions, load_decisions


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate logged A/B/C decisions")
    parser.add_argument("--decision-log-dir", type=Path, required=True)
    parser.add_argument("--candidate-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    decisions = load_decisions(args.decision_log_dir)
    paths = pd.concat(
        [pd.read_csv(path) for path in args.candidate_paths], ignore_index=True
    )
    result = evaluate_decisions(decisions, paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
