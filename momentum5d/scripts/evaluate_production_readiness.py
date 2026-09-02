from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from app.production_readiness import atomic_write_json

STRATEGIES = ("capitulation_reversal", "first_pullback")
SYSTEM_CATEGORIES = (
    "DATA",
    "LIVE_PIPELINE",
    "BACKTEST",
    "LIVE_BT_CONSISTENCY",
    "REPRODUCIBILITY",
    "FAILSAFE",
    "AUDIT",
    "MONITORING",
    "ROLLBACK",
)
REQUIRED_BACKTEST_METRICS = {
    "completed_trades",
    "trade_win_rate",
    "mean_trade_net_return",
    "expected_net_return_per_trade",
    "profit_factor",
    "maximum_drawdown",
    "maximum_consecutive_losses",
    "total_return",
    "annualized_return",
    "annual_returns",
    "costs_included",
}


def _read(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _check(status: bool, detail: str) -> dict[str, str]:
    return {"status": "PASS" if status else "FAIL", "detail": detail}


def _category(checks: dict[str, dict[str, str]]) -> dict[str, Any]:
    return {
        "status": "PASS" if all(v["status"] == "PASS" for v in checks.values()) else "FAIL",
        "checks": checks,
    }


def evaluate(
    *,
    daily: dict[str, Any],
    backtest: dict[str, Any],
    rollback: dict[str, Any],
    failsafe: dict[str, Any] | None = None,
    monitoring: dict[str, Any] | None = None,
    as_of: date,
) -> dict[str, Any]:
    quality = daily.get("quality_gate") or {}
    summaries = backtest.get("summaries") or {}
    input_audit = backtest.get("input_data_audit") or {}
    live_contract = daily.get("logic_contract") or {}
    bt_contract = backtest.get("logic_contract") or {}
    monitoring = monitoring or {}
    failsafe = failsafe or {}
    expected_date = as_of.isoformat()

    categories: dict[str, dict[str, Any]] = {}
    categories["DATA"] = _category(
        {
            "latest_market_date": _check(
                daily.get("market_date") == expected_date,
                f"expected={expected_date} actual={daily.get('market_date')}",
            ),
            "coverage": _check(
                int(daily.get("successful_tickers") or 0) > 0
                and int(daily.get("expected_tickers") or 0)
                >= int(daily.get("successful_tickers") or 0),
                f"{daily.get('successful_tickers', 0)}/{daily.get('expected_tickers', 0)}",
            ),
            "certified_quality": _check(
                quality.get("status") == "PASS",
                "; ".join(quality.get("errors") or []) or "certified",
            ),
        }
    )
    categories["LIVE_PIPELINE"] = _category(
        {
            "complete": _check(daily.get("complete") is True, str(daily.get("status"))),
            "atomic_publication": _check(
                (daily.get("publication") or {}).get("status") == "ready_for_atomic_publish",
                str((daily.get("publication") or {}).get("status") or "missing"),
            ),
            "stale_fallback_disabled": _check(
                (daily.get("publication") or {}).get("stale_result_fallback_allowed") is False,
                "stale result fallback must be false",
            ),
        }
    )
    bt_checks: dict[str, dict[str, str]] = {
        "certified_input": _check(
            backtest.get("status") == "completed_certified"
            and input_audit.get("status") == "certified"
            and input_audit.get("split_adjusted_volume") is True
            and input_audit.get("point_in_time_universe") is True,
            str(backtest.get("status") or "TEST_NOT_COMPLETED"),
        ),
        "strategies_separated": _check(
            backtest.get("signals_evaluated_separately") is True,
            str(backtest.get("signals_evaluated_separately")),
        ),
    }
    for strategy in STRATEGIES:
        summary = summaries.get(strategy) or {}
        missing = sorted(REQUIRED_BACKTEST_METRICS.difference(summary))
        bt_checks[f"{strategy}_metrics"] = _check(
            not missing and summary.get("costs_included") is True,
            "complete" if not missing else f"missing={','.join(missing)}",
        )
    categories["BACKTEST"] = _category(bt_checks)

    live_values = live_contract.get("effective_values")
    bt_values = bt_contract.get("effective_values")
    live_hash = live_contract.get("effective_values_sha256")
    bt_hash = bt_contract.get("effective_values_sha256")
    categories["LIVE_BT_CONSISTENCY"] = _category(
        {
            "effective_values": _check(
                bool(live_values) and live_values == bt_values,
                "identical" if live_values and live_values == bt_values else "mismatch/missing",
            ),
            "effective_values_sha256": _check(
                bool(live_hash) and live_hash == bt_hash,
                f"live={live_hash or 'missing'} bt={bt_hash or 'missing'}",
            ),
        }
    )
    bt_repro = backtest.get("reproducibility") or {}
    categories["REPRODUCIBILITY"] = _category(
        {
            "live_signal_rebuild": _check(
                (daily.get("reproducibility") or {}).get("status") == "PASS",
                str((daily.get("reproducibility") or {}).get("decision_fingerprint") or "missing"),
            ),
            "backtest_repeat": _check(
                bt_repro.get("status") == "PASS",
                str(bt_repro.get("deterministic_fingerprint") or "TEST_NOT_COMPLETED"),
            ),
        }
    )
    scenario_names = (
        "latest_date_mismatch",
        "ticker_shortfall",
        "complete_false",
        "invalid_strategy_config",
        "close_certification_failure",
    )
    scenarios = failsafe.get("scenarios") or {}
    categories["FAILSAFE"] = _category(
        {
            name: _check(
                (scenarios.get(name) or {}).get("publication_blocked") is True,
                str((scenarios.get(name) or {}).get("detail") or "TEST_NOT_COMPLETED"),
            )
            for name in scenario_names
        }
    )
    audit_fields = {
        "market_date",
        "data_acquired_at",
        "calculation_started_at",
        "calculation_completed_at",
        "successful_tickers",
        "expected_tickers",
        "strategy_config_version",
        "strategy_config_sha256",
        "git_commit_id",
        "strategy_names",
        "signal_counts",
        "complete",
    }
    missing_audit = sorted(audit_fields.difference(daily))
    categories["AUDIT"] = _category(
        {
            "required_fields": _check(
                not missing_audit,
                "complete" if not missing_audit else f"missing={','.join(missing_audit)}",
            )
        }
    )
    monitoring_fields = {
        "market_date",
        "status",
        "successful_tickers",
        "expected_tickers",
        "missing_tickers",
        "signal_counts",
        "zero_signal_reason",
        "processing_seconds",
        "publication",
    }
    missing_monitoring = sorted(monitoring_fields.difference(monitoring))
    categories["MONITORING"] = _category(
        {
            "required_fields": _check(
                not missing_monitoring,
                "complete" if not missing_monitoring else f"missing={','.join(missing_monitoring)}",
            ),
            "publication_success": _check(
                monitoring.get("publication") == "success",
                str(monitoring.get("publication") or "missing"),
            ),
        }
    )
    current = rollback.get("current") or {}
    categories["ROLLBACK"] = _category(
        {
            "manifest": _check(
                rollback.get("rollback_supported") is True
                and bool(current.get("snapshot_path"))
                and bool(current.get("git_commit_id"))
                and bool(current.get("strategy_config_sha256")),
                str(current.get("snapshot_path") or "missing"),
            ),
            "dry_run": _check(
                (rollback.get("dry_run") or {}).get("status") == "PASS"
                and (rollback.get("dry_run") or {}).get("production_mutated") is False,
                str(
                    (rollback.get("dry_run") or {}).get("status")
                    or "TEST_NOT_COMPLETED"
                ),
            ),
        }
    )

    blockers = [
        f"{category}.{name}"
        for category in SYSTEM_CATEGORIES
        for name, result in categories[category]["checks"].items()
        if result["status"] != "PASS"
    ]
    ready = not blockers
    return {
        "schema_version": 2,
        "as_of": expected_date,
        "SYSTEM_PRODUCTION_READY": "PASS" if ready else "FAIL",
        "FULL_CAPITAL_STRATEGY_READY": "WAIT_OOS",
        "categories": categories,
        "OOS": "WAIT",
        "FORWARD_RECORDS": "0 / 50 / 100",
        "blocking_items": blockers,
        "production_migration_allowed": ready,
        "rule": "Unexecuted or unconfirmed checks cannot pass.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate final production acceptance")
    parser.add_argument("--daily-audit", type=Path, required=True)
    parser.add_argument("--backtest-summary", type=Path, required=True)
    parser.add_argument("--rollback-manifest", type=Path, required=True)
    parser.add_argument("--failsafe-report", type=Path, required=True)
    parser.add_argument("--monitoring", type=Path, required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        daily=_read(args.daily_audit),
        backtest=_read(args.backtest_summary),
        rollback=_read(args.rollback_manifest),
        failsafe=_read(args.failsafe_report),
        monitoring=_read(args.monitoring),
        as_of=args.as_of,
    )
    atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["production_migration_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
