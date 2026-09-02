from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from app.live_strategy import StrategySpecError, load_frozen_strategy
from app.market_data_contract import daily_snapshot_fingerprint
from app.production_readiness import (
    atomic_write_json,
    summarize_checks,
    validate_price_quality,
    validate_signal_payload,
)
from scripts.export_close_signals import build_payload


def _prices() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker, offset in (("1001.T", 0.0), ("1002.T", 20.0)):
        for index, day in enumerate(pd.bdate_range("2026-04-29", periods=90)):
            close = 100.0 + offset + index * 0.2
            rows.append(
                {
                    "date": day.date(),
                    "ticker": ticker,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "adjusted_close": close,
                    "volume": 1_000_000 + index,
                    "source": "yfinance",
                    "stock_splits": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _certification(prices: pd.DataFrame) -> dict[str, Any]:
    market_date = pd.to_datetime(prices["date"]).max().date().isoformat()
    return {
        "status": "certified",
        "market_date": market_date,
        "source": "yfinance",
        "session": "close",
        "interval": "1d",
        "market_timezone": "Asia/Tokyo",
        "data_through": f"{market_date}T15:30:00+09:00",
        "acquired_at": f"{market_date}T07:00:00+00:00",
        "successful_tickers": 2,
        "expected_tickers": 2,
        "coverage": 1.0,
        "minimum_coverage": 0.95,
        "missing_tickers": [],
        "rejected_sources": [],
        "snapshot_fingerprint": daily_snapshot_fingerprint(prices, market_date),
    }


def _blocked(action: Callable[[], object]) -> tuple[bool, str]:
    try:
        result = action()
    except (KeyError, OSError, StrategySpecError, TypeError, ValueError) as exc:
        return True, str(exc)
    if isinstance(result, list):
        summary = summarize_checks(result)
        return summary["status"] == "FAIL", "; ".join(summary["errors"]) or "unexpected pass"
    return False, "unexpected pass"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated production fail-safe injections")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prices = _prices()
    certification = _certification(prices)
    market_date = pd.to_datetime(prices["date"]).max().date()
    payload = build_payload(
        prices,
        certification=certification,
        generated_at="2026-09-02T07:00:00+00:00",
        git_commit="failsafe-test",
    )

    scenarios: dict[str, dict[str, object]] = {}
    blocked, detail = _blocked(
        lambda: validate_price_quality(
            prices, certification, expected_date=market_date + timedelta(days=1)
        )
    )
    scenarios["latest_date_mismatch"] = {"publication_blocked": blocked, "detail": detail}

    short = dict(certification)
    short.update({"successful_tickers": 1, "coverage": 0.5, "missing_tickers": ["1002.T"]})
    blocked, detail = _blocked(
        lambda: validate_price_quality(prices, short, expected_date=market_date)
    )
    scenarios["ticker_shortfall"] = {"publication_blocked": blocked, "detail": detail}

    incomplete = dict(payload)
    incomplete["update"] = {**payload["update"], "status": "running"}
    blocked, detail = _blocked(
        lambda: validate_signal_payload(incomplete, certification, expected_date=market_date)
    )
    scenarios["complete_false"] = {"publication_blocked": blocked, "detail": detail}

    rejected = dict(certification)
    rejected["status"] = "rejected"
    blocked, detail = _blocked(
        lambda: validate_price_quality(prices, rejected, expected_date=market_date)
    )
    scenarios["close_certification_failure"] = {"publication_blocked": blocked, "detail": detail}

    invalid_spec = dict(load_frozen_strategy())
    invalid_spec["status"] = "draft"
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "strategy.json"
        path.write_text(json.dumps(invalid_spec), encoding="utf-8")
        blocked, detail = _blocked(lambda: load_frozen_strategy(path))
    scenarios["invalid_strategy_config"] = {"publication_blocked": blocked, "detail": detail}

    passed = all(bool(item["publication_blocked"]) for item in scenarios.values())
    report = {"schema_version": 1, "status": "PASS" if passed else "FAIL", "scenarios": scenarios}
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
