from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.adjustments import normalize_split_adjusted_ohlcv
from app.execution_contract import EXECUTION_ENGINE_ID
from app.live_strategy import FROZEN_STRATEGY_SHA256, load_frozen_strategy
from app.market_data_contract import DAILY_SOURCE, validate_market_certification

PRODUCTION_SCHEMA_VERSION = 1
PRODUCTION_PIPELINE_VERSION = "production_gate_v1"
REQUIRED_STRATEGIES = ("capitulation_reversal", "first_pullback")


class ProductionGateError(ValueError):
    """Raised when a release is not safe to expose as the current result."""


@dataclass(frozen=True, slots=True)
class GateCheck:
    name: str
    status: str
    detail: str


def _pass(name: str, detail: str) -> GateCheck:
    return GateCheck(name=name, status="PASS", detail=detail)


def _fail(name: str, detail: str) -> GateCheck:
    return GateCheck(name=name, status="FAIL", detail=detail)


def _finite_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan)


def validate_price_quality(
    prices: pd.DataFrame,
    certification: dict[str, Any],
    *,
    expected_date: date,
) -> list[GateCheck]:
    """Validate the certified close and split-adjusted values before signals run."""

    checks: list[GateCheck] = []
    try:
        validate_market_certification(
            certification,
            prices=prices,
            expected_date=expected_date,
        )
        checks.append(_pass("latest_market_date", expected_date.isoformat()))
        checks.append(_pass("tse_daily_only", DAILY_SOURCE))
        checks.append(
            _pass(
                "ticker_coverage",
                f"{certification['successful_tickers']}/{certification['expected_tickers']}",
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        checks.append(_fail("market_certification", str(exc)))
        return checks

    required = {
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
        "stock_splits",
    }
    missing = sorted(required.difference(prices.columns))
    if missing:
        checks.append(_fail("required_price_columns", ",".join(missing)))
        return checks
    checks.append(_pass("required_price_columns", "complete"))

    work = prices.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.date
    latest = work.loc[work["date"].eq(expected_date)].copy()
    if latest.empty:
        checks.append(_fail("latest_rows", "no rows for expected market date"))
        return checks

    numeric_columns = ["open", "high", "low", "close", "volume", "stock_splits"]
    numeric = _finite_numeric(latest, numeric_columns)
    if numeric.isna().any(axis=None):
        bad = latest.loc[numeric.isna().any(axis=1), "ticker"].astype(str).head(10)
        checks.append(_fail("finite_ohlcv", ",".join(bad)))
    else:
        checks.append(_pass("finite_ohlcv", f"rows={len(latest)}"))

    invalid_price = numeric[["open", "high", "low", "close"]].le(0).any(axis=1)
    invalid_volume = numeric["volume"].lt(0)
    invalid_ohlc = numeric["high"].lt(numeric[["open", "close", "low"]].max(axis=1)) | numeric[
        "low"
    ].gt(numeric[["open", "close", "high"]].min(axis=1))
    invalid = invalid_price | invalid_volume | invalid_ohlc
    if invalid.any():
        bad = latest.loc[invalid, "ticker"].astype(str).head(10)
        checks.append(_fail("ohlcv_invariants", ",".join(bad)))
    else:
        checks.append(_pass("ohlcv_invariants", "positive prices, nonnegative volume"))

    split_values = numeric["stock_splits"]
    # Yahoo uses 0 for no split, so only negative and non-finite values are invalid.
    invalid_splits = split_values.lt(0) | split_values.isna()
    if invalid_splits.any():
        bad = latest.loc[invalid_splits, "ticker"].astype(str).head(10)
        checks.append(_fail("split_events", ",".join(bad)))
    else:
        checks.append(_pass("split_events", "explicit events are finite and nonnegative"))

    try:
        adjusted = normalize_split_adjusted_ohlcv(work)
        latest_adjusted = adjusted.loc[adjusted["date"].eq(expected_date)]
        adjusted_columns = [
            "_open",
            "_high",
            "_low",
            "_close",
            "_volume",
            "_turnover",
            "_split_share_factor",
        ]
        values = _finite_numeric(latest_adjusted, adjusted_columns)
        invalid_adjusted = (
            values[["_open", "_high", "_low", "_close", "_split_share_factor"]].le(0).any(axis=1)
            | values[["_volume", "_turnover"]].lt(0).any(axis=1)
            | values.isna().any(axis=1)
        )
        if invalid_adjusted.any():
            bad = latest_adjusted.loc[invalid_adjusted, "ticker"].astype(str).head(10)
            checks.append(_fail("split_adjusted_ohlcv", ",".join(bad)))
        else:
            checks.append(
                _pass("split_adjusted_ohlcv", "price, volume and turnover are consistent")
            )
    except (KeyError, TypeError, ValueError) as exc:
        checks.append(_fail("split_adjusted_ohlcv", str(exc)))

    return checks


def validate_signal_payload(
    payload: dict[str, Any],
    certification: dict[str, Any],
    *,
    expected_date: date,
) -> list[GateCheck]:
    checks: list[GateCheck] = []
    market_date = expected_date.isoformat()
    update = payload.get("update") or {}
    if (
        update.get("status") == "complete"
        and update.get("session") == "close"
        and update.get("market_date") == market_date
        and payload.get("date") == market_date
    ):
        checks.append(_pass("complete_close_payload", market_date))
    else:
        checks.append(_fail("complete_close_payload", "payload date/session/status mismatch"))

    if payload.get("strategy_version") == load_frozen_strategy()["strategy_version"]:
        checks.append(_pass("strategy_config", str(payload.get("strategy_version"))))
    else:
        checks.append(_fail("strategy_config", "payload does not use frozen strategy"))

    quality = payload.get("data_quality") or {}
    if (
        quality.get("status") == "certified"
        and quality.get("snapshot_fingerprint") == certification.get("snapshot_fingerprint")
        and quality.get("adjusted_ohlc") is True
        and quality.get("split_adjusted_volume") is True
    ):
        checks.append(_pass("payload_data_quality", "certification fingerprint matches"))
    else:
        checks.append(_fail("payload_data_quality", "certification or split adjustment mismatch"))

    collections = {
        "capitulation_reversal": ("signals", "signal_count"),
        "first_pullback": ("pullback_signals", "pullback_signal_count"),
    }
    for strategy_name, (list_key, count_key) in collections.items():
        rows = payload.get(list_key)
        count = payload.get(count_key)
        if isinstance(rows, list) and count == len(rows):
            checks.append(_pass(f"{strategy_name}_count", f"signals={len(rows)}"))
        else:
            checks.append(_fail(f"{strategy_name}_count", "count does not match rows"))
            continue
        invalid_rows: list[str] = []
        for row in rows:
            required_numeric = ["close", "trading_value", "ATR"]
            for key in required_numeric:
                value = row.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    invalid_rows.append(f"{row.get('ticker')}:{key}")
            if float(row.get("close") or 0) <= 0:
                invalid_rows.append(f"{row.get('ticker')}:close")
            if float(row.get("trading_value") or -1) < 0:
                invalid_rows.append(f"{row.get('ticker')}:trading_value")
            atr = float(row.get("ATR") or -1)
            if not 0 <= atr <= 1:
                invalid_rows.append(f"{row.get('ticker')}:ATR")
        if invalid_rows:
            checks.append(_fail(f"{strategy_name}_metrics", ",".join(invalid_rows[:10])))
        else:
            checks.append(_pass(f"{strategy_name}_metrics", "finite signal metrics"))

    total = int(payload.get("total_signal_count") or 0)
    expected_total = int(payload.get("signal_count") or 0) + int(
        payload.get("pullback_signal_count") or 0
    )
    if total == expected_total:
        checks.append(_pass("total_signal_count", str(total)))
    else:
        checks.append(_fail("total_signal_count", f"expected={expected_total} actual={total}"))
    return checks


def validate_indicator_quality(
    indicators: pd.DataFrame,
    *,
    expected_date: date,
) -> list[GateCheck]:
    """Reject abnormal calculated values before a release can become current."""

    required = {
        "date",
        "ticker",
        "_close",
        "_volume",
        "trading_value",
        "ATR",
        "ma25",
        "ma75",
        "ma25_slope_5d",
        "ma75_slope_10d",
    }
    missing = sorted(required.difference(indicators.columns))
    if missing:
        return [_fail("indicator_columns", ",".join(missing))]

    work = indicators.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.date
    latest = work.loc[work["date"].eq(expected_date)].copy()
    if latest.empty:
        return [_fail("indicator_latest_date", "no calculated rows for expected date")]

    history_counts = work.groupby("ticker", sort=False)["date"].transform("count")
    mature_tickers = set(work.loc[history_counts.ge(85), "ticker"].astype(str))
    mature = latest.loc[latest["ticker"].astype(str).isin(mature_tickers)].copy()
    if mature.empty:
        return [_fail("indicator_history", "no ticker has the required 85 sessions")]

    columns = [
        "_close",
        "_volume",
        "trading_value",
        "ATR",
        "ma25",
        "ma75",
        "ma25_slope_5d",
        "ma75_slope_10d",
    ]
    values = _finite_numeric(mature, columns)
    invalid = (
        values.isna().any(axis=1)
        | values[["_close", "ma25", "ma75"]].le(0).any(axis=1)
        | values[["_volume", "trading_value"]].lt(0).any(axis=1)
        | values["ATR"].le(0)
        | values["ATR"].gt(1.0)
    )
    if invalid.any():
        bad = mature.loc[invalid, "ticker"].astype(str).head(10)
        return [_fail("indicator_values", ",".join(bad))]
    return [
        _pass("indicator_columns", "complete"),
        _pass("indicator_values", f"finite mature_rows={len(mature)}"),
    ]


def decision_fingerprint(payload: dict[str, Any]) -> str:
    """Hash only deterministic signal decisions, excluding run timestamps."""

    decision = {
        "schema_version": payload.get("schema_version"),
        "strategy_version": payload.get("strategy_version"),
        "portfolio_rules": payload.get("portfolio_rules"),
        "date": payload.get("date"),
        "signal_model": payload.get("signal_model"),
        "pullback_signal_model": payload.get("pullback_signal_model"),
        "signals": payload.get("signals"),
        "pullback_signals": payload.get("pullback_signals"),
    }
    canonical = json.dumps(
        decision,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def frozen_logic_contract() -> dict[str, Any]:
    spec = load_frozen_strategy()
    effective = {
        "strategy_version": spec["strategy_version"],
        "strategy_sha256": FROZEN_STRATEGY_SHA256,
        "data": spec["data"],
        "portfolio": spec["portfolio"],
        "signals": spec["signals"],
        "secondary_selection": spec["secondary_selection"],
        "execution_engine_id": EXECUTION_ENGINE_ID,
    }
    canonical = json.dumps(
        effective,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return {
        "consumer": "live",
        "effective_values": effective,
        "effective_values_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "signal_module": "scripts.export_close_signals",
        "strategy_names": list(REQUIRED_STRATEGIES),
        "strategies_evaluated_separately": True,
    }


def summarize_checks(checks: list[GateCheck]) -> dict[str, Any]:
    failed = [check for check in checks if check.status != "PASS"]
    return {
        "status": "PASS" if not failed else "FAIL",
        "checks": [asdict(check) for check in checks],
        "error_count": len(failed),
        "errors": [f"{check.name}: {check.detail}" for check in failed],
    }


def build_production_audit(
    *,
    market_date: str,
    acquired_at: str,
    calculation_started_at: str,
    calculation_completed_at: str,
    successful_tickers: int,
    expected_tickers: int,
    git_commit: str,
    signal_counts: dict[str, int],
    checks: list[GateCheck],
    decision_hash: str,
    reproducible: bool,
) -> dict[str, Any]:
    summary = summarize_checks(checks)
    started = datetime.fromisoformat(calculation_started_at)
    completed = datetime.fromisoformat(calculation_completed_at)
    return {
        "schema_version": PRODUCTION_SCHEMA_VERSION,
        "pipeline_version": PRODUCTION_PIPELINE_VERSION,
        "status": "complete" if summary["status"] == "PASS" and reproducible else "failed",
        "market_date": market_date,
        "latest_market_date": market_date,
        "expected_market_date": market_date,
        "data_acquired_at": acquired_at,
        "calculation_started_at": calculation_started_at,
        "calculation_completed_at": calculation_completed_at,
        "processing_seconds": max((completed - started).total_seconds(), 0.0),
        "successful_tickers": successful_tickers,
        "expected_tickers": expected_tickers,
        "missing_tickers": expected_tickers - successful_tickers,
        "coverage": successful_tickers / expected_tickers if expected_tickers else 0.0,
        "data_quality_status": "certified" if summary["status"] == "PASS" else "rejected",
        "generated_at": calculation_completed_at,
        "strategy_config_version": load_frozen_strategy()["strategy_version"],
        "strategy_config_sha256": FROZEN_STRATEGY_SHA256,
        "git_commit_id": git_commit,
        "strategy_names": list(REQUIRED_STRATEGIES),
        "signal_counts": signal_counts,
        "complete": summary["status"] == "PASS" and reproducible,
        "reproducibility": {
            "status": "PASS" if reproducible else "FAIL",
            "decision_fingerprint": decision_hash,
        },
        "logic_contract": frozen_logic_contract(),
        "quality_gate": summary,
        "publication": {
            "status": (
                "ready_for_atomic_publish"
                if summary["status"] == "PASS" and reproducible
                else "blocked"
            ),
            "atomic_release_required": True,
            "stale_result_fallback_allowed": False,
        },
    }


def failure_payload(*, market_date: str, errors: list[str], git_commit: str) -> dict[str, Any]:
    stamp = datetime.now(UTC).isoformat()
    return {
        "schema_version": 4,
        "production_schema_version": PRODUCTION_SCHEMA_VERSION,
        "strategy_version": load_frozen_strategy()["strategy_version"],
        "date": market_date,
        "latest_date": market_date,
        "generated_at": stamp,
        "update": {
            "status": "failed",
            "session": "close",
            "market_date": market_date,
            "generated_at": stamp,
            "message": "更新失敗・本日データ未確定",
        },
        "data_quality": {"status": "rejected"},
        "git_commit_id": git_commit,
        "errors": errors,
        "total_signal_count": 0,
        "signal_count": 0,
        "signals": [],
        "pullback_signal_count": 0,
        "pullback_signals": [],
        "near_miss_count": 0,
        "near_misses": [],
        "stale_result_fallback_allowed": False,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)
