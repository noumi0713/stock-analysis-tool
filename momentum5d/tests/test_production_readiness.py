from __future__ import annotations

import json
from datetime import date

import pandas as pd

from app.market_data_contract import daily_snapshot_fingerprint
from app.production_readiness import (
    decision_fingerprint,
    summarize_checks,
    validate_indicator_quality,
    validate_price_quality,
    validate_signal_payload,
)
from scripts.evaluate_production_readiness import evaluate
from scripts.export_close_signals import build_payload, calculate_indicators
from scripts.prepare_production_release import prepare_release


def _prices() -> pd.DataFrame:
    rows = []
    sessions = pd.bdate_range("2026-04-29", periods=90)
    for ticker, offset in (("1001.T", 0.0), ("1002.T", 20.0)):
        for index, day in enumerate(sessions):
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


def _certification(prices: pd.DataFrame) -> dict[str, object]:
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


def test_production_quality_gate_accepts_certified_adjusted_data() -> None:
    prices = _prices()
    certification = _certification(prices)
    market_date = pd.to_datetime(prices["date"]).max().date()
    payload = build_payload(
        prices,
        certification=certification,
        generated_at="2026-09-02T07:00:00+00:00",
        git_commit="abc123",
    )

    checks = validate_price_quality(prices, certification, expected_date=market_date)
    checks.extend(
        validate_indicator_quality(calculate_indicators(prices), expected_date=market_date)
    )
    checks.extend(validate_signal_payload(payload, certification, expected_date=market_date))

    assert summarize_checks(checks)["status"] == "PASS"
    assert decision_fingerprint(payload)


def test_production_quality_gate_rejects_invalid_ohlc() -> None:
    prices = _prices()
    certification = _certification(prices)
    market_date = pd.to_datetime(prices["date"]).max().date()
    latest_index = prices.index[prices["date"].eq(market_date)][0]
    prices.loc[latest_index, "high"] = prices.loc[latest_index, "low"] - 1
    certification["snapshot_fingerprint"] = daily_snapshot_fingerprint(prices, market_date)

    checks = validate_price_quality(prices, certification, expected_date=market_date)

    assert summarize_checks(checks)["status"] == "FAIL"
    assert any(check.name == "ohlcv_invariants" and check.status == "FAIL" for check in checks)


def test_prepare_release_creates_atomic_snapshot_and_strategy_files(tmp_path) -> None:
    prices = _prices()
    certification = _certification(prices)
    market_date = pd.to_datetime(prices["date"]).max().date()
    payload = build_payload(
        prices,
        certification=certification,
        generated_at="2026-09-02T07:00:00+00:00",
        git_commit="abc123",
    )
    prices_path = tmp_path / "prices.parquet"
    certification_path = tmp_path / "certification.json"
    signals_path = tmp_path / "signals.json"
    output_dir = tmp_path / "release"
    prices.to_parquet(prices_path, index=False)
    certification_path.write_text(json.dumps(certification), encoding="utf-8")
    signals_path.write_text(json.dumps(payload), encoding="utf-8")

    audit = prepare_release(
        prices_path=prices_path,
        certification_path=certification_path,
        signals_path=signals_path,
        output_dir=output_dir,
        expected_date=market_date,
        names_path=None,
        themes_path=None,
        git_commit="abc123",
        started_at="2026-09-02T07:00:00+00:00",
    )

    assert audit["complete"] is True
    assert (output_dir / "production-status.json").exists()
    assert (
        output_dir / "production-snapshots" / market_date.isoformat() / "latest_signals.json"
    ).exists()
    assert (
        output_dir / "strategies" / market_date.isoformat() / "capitulation_reversal.json"
    ).exists()
    assert (output_dir / "rollback-manifest.json").exists()


def test_system_gate_can_pass_while_full_capital_waits_for_oos() -> None:
    effective = {
        "strategy_version": "live_v1_2026-08-31",
        "portfolio": {"maximum_open_positions": 3},
    }
    daily = {
        "complete": True,
        "status": "complete",
        "market_date": "2026-09-01",
        "successful_tickers": 3700,
        "expected_tickers": 3707,
        "missing_tickers": 7,
        "processing_seconds": 10.0,
        "quality_gate": {
            "status": "PASS",
            "checks": [{"name": "split_adjusted_ohlcv", "status": "PASS"}],
            "errors": [],
        },
        "logic_contract": {
            "effective_values": effective,
            "effective_values_sha256": "same",
            "strategies_evaluated_separately": True,
        },
        "reproducibility": {"status": "PASS", "decision_fingerprint": "abc"},
        "publication": {"stale_result_fallback_allowed": False},
    }
    metrics = {
        "completed_trades": 10,
        "trade_win_rate": 0.5,
        "mean_trade_net_return": 0.01,
        "profit_factor": 1.2,
        "maximum_drawdown": -0.1,
        "maximum_consecutive_losses": 2,
        "expected_net_return_per_trade": 0.01,
        "total_return": 0.1,
        "annualized_return": 0.1,
        "annual_returns": {"2026": 0.1},
        "costs_included": True,
    }
    backtest = {
        "status": "completed_certified",
        "input_data_audit": {
            "status": "certified",
            "split_adjusted_volume": True,
            "point_in_time_universe": True,
        },
        "signals_evaluated_separately": True,
        "logic_contract": {
            "effective_values": effective,
            "effective_values_sha256": "same",
        },
        "reproducibility": {"status": "PASS", "deterministic_fingerprint": "bt"},
        "summaries": {
            "capitulation_reversal": metrics,
            "first_pullback": metrics,
        },
    }
    rollback = {
        "rollback_supported": True,
        "current": {"snapshot_path": "snapshot"},
        "dry_run": {"status": "PASS", "production_mutated": False},
    }
    rollback["current"].update({"git_commit_id": "abc", "strategy_config_sha256": "same"})
    failsafe = {
        "scenarios": {
            name: {"publication_blocked": True, "detail": "blocked"}
            for name in (
                "latest_date_mismatch",
                "ticker_shortfall",
                "complete_false",
                "invalid_strategy_config",
                "close_certification_failure",
            )
        }
    }
    monitoring = {
        "market_date": "2026-09-02",
        "status": "complete",
        "successful_tickers": 3700,
        "expected_tickers": 3707,
        "missing_tickers": 7,
        "signal_counts": {},
        "zero_signal_reason": {},
        "processing_seconds": 10.0,
        "publication": "success",
    }
    daily.update(
        {
            "market_date": "2026-09-02",
            "data_acquired_at": "2026-09-02T08:00:00+09:00",
            "calculation_started_at": "2026-09-02T08:01:00+09:00",
            "calculation_completed_at": "2026-09-02T08:02:00+09:00",
            "strategy_config_version": "live_v1_2026-08-31",
            "strategy_config_sha256": "same",
            "git_commit_id": "abc",
            "strategy_names": ["capitulation_reversal", "first_pullback"],
            "signal_counts": {},
            "publication": {
                "status": "ready_for_atomic_publish",
                "stale_result_fallback_allowed": False,
            },
        }
    )

    result = evaluate(
        daily=daily,
        backtest=backtest,
        rollback=rollback,
        failsafe=failsafe,
        monitoring=monitoring,
        as_of=date(2026, 9, 2),
    )

    assert result["SYSTEM_PRODUCTION_READY"] == "PASS"
    assert result["FULL_CAPITAL_STRATEGY_READY"] == "WAIT_OOS"
    assert result["OOS"] == "WAIT"
