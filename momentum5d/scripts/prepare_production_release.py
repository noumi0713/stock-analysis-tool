from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.market_data_contract import load_market_certification
from app.production_readiness import (
    ProductionGateError,
    atomic_write_json,
    build_production_audit,
    decision_fingerprint,
    failure_payload,
    validate_indicator_quality,
    validate_price_quality,
    validate_signal_payload,
)
from scripts.export_close_signals import (
    _load_names,
    _load_theme_memberships,
    build_payload,
    calculate_indicators,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _strategy_release(payload: dict[str, Any], strategy_name: str) -> dict[str, Any]:
    if strategy_name == "capitulation_reversal":
        rows = payload.get("signals") or []
        model = payload.get("signal_model") or {}
    elif strategy_name == "first_pullback":
        rows = payload.get("pullback_signals") or []
        model = payload.get("pullback_signal_model") or {}
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return {
        "schema_version": 1,
        "market_date": payload["date"],
        "strategy_version": payload["strategy_version"],
        "strategy_name": strategy_name,
        "strategy_model": model,
        "portfolio_rules": payload["portfolio_rules"],
        "signal_count": len(rows),
        "signals": rows,
        "audit_context": payload.get("audit_context") or {},
    }


def _rebuild_fingerprint(
    prices: pd.DataFrame,
    certification: dict[str, Any],
    *,
    names: Path | None,
    themes: Path | None,
    generated_at: str,
    git_commit: str,
) -> tuple[str, str]:
    kwargs = {
        "certification": certification,
        "names": _load_names(names),
        "theme_memberships": _load_theme_memberships(themes),
        "generated_at": generated_at,
        "git_commit": git_commit,
    }
    first = build_payload(prices, **kwargs)
    second = build_payload(prices, **kwargs)
    return decision_fingerprint(first), decision_fingerprint(second)


def prepare_release(
    *,
    prices_path: Path,
    certification_path: Path,
    signals_path: Path,
    output_dir: Path,
    expected_date: date,
    names_path: Path | None,
    themes_path: Path | None,
    git_commit: str,
    previous_status_path: Path | None = None,
    started_at: str | None = None,
) -> dict[str, Any]:
    start = started_at or datetime.now(UTC).isoformat()
    prices = pd.read_parquet(prices_path)
    certification = load_market_certification(certification_path)
    payload = _read_json(signals_path)
    checks = validate_price_quality(prices, certification, expected_date=expected_date)
    checks.extend(
        validate_indicator_quality(calculate_indicators(prices), expected_date=expected_date)
    )
    checks.extend(validate_signal_payload(payload, certification, expected_date=expected_date))

    generated_at = str(payload.get("generated_at") or start)
    first_hash, second_hash = _rebuild_fingerprint(
        prices,
        certification,
        names=names_path,
        themes=themes_path,
        generated_at=generated_at,
        git_commit=git_commit,
    )
    payload_hash = decision_fingerprint(payload)
    reproducible = first_hash == second_hash == payload_hash
    completed = datetime.now(UTC).isoformat()
    counts = {
        "capitulation_reversal": int(payload.get("signal_count") or 0),
        "first_pullback": int(payload.get("pullback_signal_count") or 0),
    }
    audit = build_production_audit(
        market_date=expected_date.isoformat(),
        acquired_at=str(certification["acquired_at"]),
        calculation_started_at=start,
        calculation_completed_at=completed,
        successful_tickers=int(certification["successful_tickers"]),
        expected_tickers=int(certification["expected_tickers"]),
        git_commit=git_commit,
        signal_counts=counts,
        checks=checks,
        decision_hash=payload_hash,
        reproducible=reproducible,
    )
    if not audit["complete"]:
        errors = list(audit["quality_gate"]["errors"])
        if not reproducible:
            errors.append(
                "reproducibility: repeated signal calculation did not match staged output"
            )
        atomic_write_json(
            output_dir / "latest_signals.json",
            failure_payload(
                market_date=expected_date.isoformat(),
                errors=errors,
                git_commit=git_commit,
            ),
        )
        atomic_write_json(output_dir / "production-status.json", audit)
        raise ProductionGateError("; ".join(errors))

    release_payload = dict(payload)
    release_payload["production"] = {
        # This file is only exposed by the same Git commit as the complete release bundle.
        # If the push fails, the file never becomes current and the failure path clears it.
        "status": "published",
        "pipeline_version": audit["pipeline_version"],
        "audit_path": f"production-audit/{expected_date.isoformat()}.json",
        "decision_fingerprint": payload_hash,
        "stale_result_fallback_allowed": False,
    }
    atomic_write_json(output_dir / "latest_signals.json", release_payload)
    atomic_write_json(output_dir / "production-status.json", audit)
    atomic_write_json(
        output_dir / "production-audit" / f"{expected_date.isoformat()}.json",
        audit,
    )
    atomic_write_json(
        output_dir / "production-snapshots" / expected_date.isoformat() / "latest_signals.json",
        release_payload,
    )
    for strategy_name in counts:
        atomic_write_json(
            output_dir / "strategies" / expected_date.isoformat() / f"{strategy_name}.json",
            _strategy_release(release_payload, strategy_name),
        )

    previous: dict[str, Any] | None = None
    if previous_status_path and previous_status_path.exists():
        previous = _read_json(previous_status_path)
    rollback = {
        "schema_version": 1,
        "current": {
            "market_date": expected_date.isoformat(),
            "git_commit_id": git_commit,
            "strategy_config_version": audit["strategy_config_version"],
            "strategy_config_sha256": audit["strategy_config_sha256"],
            "data_schema_version": release_payload["schema_version"],
            "snapshot_path": (
                f"production-snapshots/{expected_date.isoformat()}/latest_signals.json"
            ),
        },
        "previous_known_good": previous,
        "rollback_supported": True,
    }
    atomic_write_json(output_dir / "rollback-manifest.json", rollback)
    monitoring = {
        "schema_version": 1,
        "market_date": expected_date.isoformat(),
        "status": "complete",
        "data_download": "success",
        "successful_tickers": audit["successful_tickers"],
        "expected_tickers": audit["expected_tickers"],
        "missing_tickers": audit["missing_tickers"],
        "coverage": audit["coverage"],
        "signal_counts": counts,
        "zero_signal_reason": {
            strategy: "normal_zero" if count == 0 else "signals_found"
            for strategy, count in counts.items()
        },
        "anomalies": audit["quality_gate"]["errors"],
        "processing_seconds": audit["processing_seconds"],
        "publication": "success",
        "audit_path": f"production-audit/{expected_date.isoformat()}.json",
    }
    atomic_write_json(output_dir / "monitoring" / "latest.json", monitoring)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and stage an atomic production signal release"
    )
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--certification", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-date", type=date.fromisoformat, required=True)
    parser.add_argument("--names", type=Path)
    parser.add_argument("--themes", type=Path)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--previous-status", type=Path)
    parser.add_argument("--started-at")
    args = parser.parse_args()
    try:
        audit = prepare_release(
            prices_path=args.prices,
            certification_path=args.certification,
            signals_path=args.signals,
            output_dir=args.output_dir,
            expected_date=args.expected_date,
            names_path=args.names,
            themes_path=args.themes,
            git_commit=args.git_commit,
            previous_status_path=args.previous_status,
            started_at=args.started_at,
        )
    except (ProductionGateError, OSError, ValueError) as exc:
        print(f"Production publication blocked: {exc}")
        return 1
    print(
        "Production release staged: "
        f"date={audit['market_date']} signals={audit['signal_counts']} "
        f"fingerprint={audit['reproducibility']['decision_fingerprint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
