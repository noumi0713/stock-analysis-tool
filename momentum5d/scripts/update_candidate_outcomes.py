from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.candidate_outcomes import track_ten_session_outcomes
from app.position_ledger import load_fills
from app.secondary_decision_log import load_decisions


def _confirmed_buys(path: Path | None) -> dict[tuple[object, str, str], list[str]]:
    if path is None or not path.exists():
        return {}
    grouped: dict[tuple[object, str, str], list[str]] = {}
    for fill in load_fills(path):
        if fill.side != "buy" or not fill.strategy:
            continue
        executed = datetime.fromisoformat(fill.executed_at)
        key = (executed.date(), fill.ticker, fill.strategy)
        grouped.setdefault(key, []).append(fill.execution_id)
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Track every A/B/C candidate for ten subsequent TSE sessions"
    )
    parser.add_argument("--decision-log-dir", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--git-commit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = track_ten_session_outcomes(
        load_decisions(args.decision_log_dir),
        pd.read_parquet(args.prices),
        git_commit=args.git_commit,
        confirmed_buy_ids=_confirmed_buys(args.ledger),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "outcomes"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
