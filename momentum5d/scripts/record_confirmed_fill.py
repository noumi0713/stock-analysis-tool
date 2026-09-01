from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.position_ledger import ConfirmedFill, record_confirmed_fill


def main() -> int:
    parser = argparse.ArgumentParser(
        description="証券会社で約定確認済みの取引だけを保有台帳へ記録"
    )
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--side", choices=("buy", "sell"), required=True)
    parser.add_argument("--quantity", type=int, required=True)
    parser.add_argument("--price", type=float, required=True)
    parser.add_argument("--executed-at", required=True)
    parser.add_argument("--strategy")
    args = parser.parse_args()
    result = record_confirmed_fill(
        args.ledger,
        ConfirmedFill(
            execution_id=args.execution_id,
            ticker=args.ticker,
            side=args.side,
            quantity=args.quantity,
            price=args.price,
            executed_at=args.executed_at,
            strategy=args.strategy,
        ),
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
