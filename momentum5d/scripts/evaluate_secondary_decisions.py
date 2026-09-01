from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.abc_evaluation import evaluate_abc_effectiveness
from app.audit_metadata import resolve_git_commit
from app.secondary_decision_log import load_decisions


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate logged A/B/C decisions")
    parser.add_argument("--decision-log-dir", type=Path, required=True)
    parser.add_argument("--candidate-outcomes", type=Path, required=True)
    parser.add_argument("--git-commit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    decisions = load_decisions(args.decision_log_dir)
    payload = json.loads(args.candidate_outcomes.read_text(encoding="utf-8"))
    outcomes = pd.DataFrame(payload.get("outcomes") or [])
    result = evaluate_abc_effectiveness(
        decisions,
        outcomes,
        git_commit=resolve_git_commit(args.git_commit),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
