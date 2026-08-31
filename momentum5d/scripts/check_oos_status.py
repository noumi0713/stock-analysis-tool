from __future__ import annotations

import argparse
import json
from datetime import date

from app.evaluation_protocol import load_evaluation_protocol, oos_access_status


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the sealed OOS evaluation status")
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    args = parser.parse_args()

    protocol = load_evaluation_protocol()
    status = oos_access_status(as_of=args.as_of, protocol=protocol)
    print(json.dumps(status, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
