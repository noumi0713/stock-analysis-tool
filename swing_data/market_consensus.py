"""Collect and classify Yahoo Finance consensus for the daily BBS top 100.

Consensus data are provider snapshots, not trading signals or point-in-time backtest
data. Missing observations never become neutral. When analyst price targets are
unusable, three prices are ATR scenarios and are labelled as such.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time
from zoneinfo import ZoneInfo
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from swing_data.bbs_ranking import atomic_csv
from swing_data.collector import atomic_json, read_json

JST = ZoneInfo("Asia/Tokyo")
COLUMNS = [
    "date", "rank", "stock_code", "stock_name", "market",
    "current_price", "price_source", "provider_price", "provider_price_date",
    "target_low", "target_mean", "target_high", "target_upside_pct",
    "recommendation_key", "recommendation_mean", "analyst_count",
    "current_year_end", "current_year_eps", "current_year_eps_30d",
    "current_year_eps_change_pct", "next_year_end", "next_year_eps",
    "classification", "classification_score", "data_quality", "data_note",
    "bear_price", "base_price", "bull_price", "scenario_price_method",
    "source_url", "data_reference_date", "retrieved_at",
]
SOURCE_TEMPLATE = "https://finance.yahoo.com/quote/{code}.T/analysis/"


def finite(value):
    value = value.get("raw") if isinstance(value, dict) else value
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def market_date(epoch):
    value = finite(epoch)
    if value is None:
        return None
    return datetime.fromtimestamp(value, JST).date().isoformat()


def period(data, key):
    for item in data.get("earningsTrend", {}).get("trend", []):
        if item.get("period") == key:
            estimate = item.get("earningsEstimate", {})
            trend = item.get("epsTrend", {})
            return {
                "end_date": item.get("endDate"),
                "analyst_count": finite(estimate.get("numberOfAnalysts")),
                "eps": finite(trend.get("current")),
                "eps_30d": finite(trend.get("30daysAgo")),
            }
    return {"end_date": None, "analyst_count": None, "eps": None, "eps_30d": None}


def parse_snapshot(payload, code):
    result = payload.get("quoteSummary", {}).get("result")
    if not result:
        raise ValueError("No quoteSummary data")
    data = result[0]
    price = data.get("price", {})
    financial = data.get("financialData", {})
    if price.get("symbol") != code + ".T":
        raise ValueError("Provider symbol mismatch")
    current = finite(financial.get("currentPrice"))
    if current is None:
        current = finite(price.get("regularMarketPrice"))
    current_year, next_year = period(data, "0y"), period(data, "+1y")
    counts = [
        finite(financial.get("numberOfAnalystOpinions")),
        current_year["analyst_count"],
        next_year["analyst_count"],
    ]
    counts = [x for x in counts if x is not None]
    return {
        "provider_price": current,
        "provider_price_date": market_date(price.get("regularMarketTime")),
        "target_low": finite(financial.get("targetLowPrice")),
        "target_mean": finite(financial.get("targetMeanPrice")),
        "target_high": finite(financial.get("targetHighPrice")),
        "recommendation_key": financial.get("recommendationKey"),
        "recommendation_mean": finite(financial.get("recommendationMean")),
        "analyst_count": max(counts) if counts else None,
        "current_year": current_year,
        "next_year": next_year,
        "source_url": SOURCE_TEMPLATE.format(code=code),
    }


def yahoo_snapshot(code):
    import yfinance as yf
    from yfinance.scrapers.quote import _QUOTE_SUMMARY_URL_
    obj = yf.Ticker(code + ".T")
    payload = obj._data.get_raw_json(
        _QUOTE_SUMMARY_URL_ + "/" + code + ".T",
        params={
            "modules": "price,financialData,earningsTrend",
            "formatted": "false",
        },
        timeout=20,
    )
    return parse_snapshot(payload, code)


def local_market_data(target, code, expected_date):
    path = Path(target) / "stocks" / (code + ".csv")
    if not path.exists():
        return {"date": None, "close": None, "atr14": None, "recent_split": False}
    frame = pd.read_csv(path)
    if frame.empty:
        return {"date": None, "close": None, "atr14": None, "recent_split": False}
    frame = frame.sort_values("date")
    latest = frame.iloc[-1]
    close_column = "adj_close" if "adj_close" in frame else "close"
    high_column = "adj_high" if "adj_high" in frame else "high"
    low_column = "adj_low" if "adj_low" in frame else "low"
    work = frame.tail(30).copy()
    prior = work[close_column].shift(1)
    true_range = pd.concat([
        work[high_column] - work[low_column],
        (work[high_column] - prior).abs(),
        (work[low_column] - prior).abs(),
    ], axis=1).max(axis=1)
    atr14 = finite(true_range.tail(14).mean()) if len(work) >= 15 else None
    recent_split = bool(
        "stock_splits" in work and
        pd.to_numeric(work["stock_splits"], errors="coerce").fillna(0).tail(30).ne(0).any()
    )
    return {
        "date": str(latest["date"]),
        "close": finite(latest[close_column]),
        "atr14": atr14,
        "recent_split": recent_split,
        "date_matches": str(latest["date"]) == expected_date,
    }


def eps_change(current, prior):
    if current is None or prior is None or prior == 0:
        return None
    return (current - prior) / abs(prior) * 100


def inspect(snapshot, local, expected_date):
    issues = []
    insufficient = []
    price = local.get("close") if local.get("date") == expected_date else snapshot.get("provider_price")
    if price is None or price <= 0:
        issues.append("current_price_missing")
    if snapshot.get("provider_price_date") != expected_date:
        issues.append("provider_price_date_mismatch")
    provider_price = snapshot.get("provider_price")
    if price and provider_price and abs(provider_price / price - 1) > 0.05:
        issues.append("provider_and_close_price_mismatch")
    targets = [snapshot.get(k) for k in ("target_low", "target_mean", "target_high")]
    if all(x is not None for x in targets):
        if not (0 < targets[0] <= targets[1] <= targets[2]):
            issues.append("target_price_order_invalid")
        elif price and any(not 0.25 <= x / price <= 4 for x in targets):
            issues.append("target_price_outlier")
    else:
        insufficient.append("target_prices_missing")
    if local.get("recent_split"):
        issues.append("recent_stock_split")
    current_year, next_year = snapshot["current_year"], snapshot["next_year"]
    if current_year["end_date"] and current_year["end_date"] < expected_date:
        issues.append("current_year_period_expired")
    if current_year["end_date"] and next_year["end_date"] and next_year["end_date"] <= current_year["end_date"]:
        issues.append("forecast_period_order_invalid")
    analysts = snapshot.get("analyst_count")
    if analysts is None or analysts < 3:
        insufficient.append("analyst_count_below_3")
    if not snapshot.get("recommendation_key") and snapshot.get("recommendation_mean") is None:
        insufficient.append("recommendation_missing")
    if current_year["eps"] is None or current_year["eps_30d"] is None:
        insufficient.append("eps_30d_comparison_missing")
    return price, issues, insufficient


def classify(snapshot, price, issues, insufficient):
    if issues or insufficient or price is None:
        return "判定不能", None
    target_upside = (snapshot["target_mean"] / price - 1) * 100
    eps_move = eps_change(snapshot["current_year"]["eps"], snapshot["current_year"]["eps_30d"])
    signals = []
    signals.append(1 if target_upside >= 10 else -1 if target_upside <= -10 else 0)
    key = str(snapshot.get("recommendation_key") or "").lower()
    mean = snapshot.get("recommendation_mean")
    signals.append(1 if key in {"strong_buy", "buy"} or mean is not None and mean <= 2.5
                   else -1 if key in {"underperform", "sell"} or mean is not None and mean >= 3.5 else 0)
    signals.append(1 if eps_move is not None and eps_move >= 5
                   else -1 if eps_move is not None and eps_move <= -5 else 0)
    score = sum(signals)
    return ("強気" if score >= 2 else "弱気" if score <= -2 else "基本"), score


def scenario_prices(snapshot, price, atr14, usable_targets):
    if usable_targets:
        return (
            snapshot["target_low"], snapshot["target_mean"], snapshot["target_high"],
            "market_consensus_low_mean_high",
        )
    if price is None:
        return None, None, None, "unavailable"
    width = 2 * atr14 if atr14 is not None and atr14 > 0 else price * 0.10
    method = "atr14_plus_minus_2atr" if atr14 is not None and atr14 > 0 else "price_plus_minus_10pct"
    return (
        round(max(0, price - width), 2),
        round(price, 2),
        round(price + width, 2),
        method + "; not analyst targets",
    )


def unavailable_snapshot(code):
    return {
        "provider_price": None, "provider_price_date": None,
        "target_low": None, "target_mean": None, "target_high": None,
        "recommendation_key": None, "recommendation_mean": None,
        "analyst_count": None,
        "current_year": {"end_date": None, "analyst_count": None, "eps": None, "eps_30d": None},
        "next_year": {"end_date": None, "analyst_count": None, "eps": None, "eps_30d": None},
        "source_url": SOURCE_TEMPLATE.format(code=code),
    }


def collect(target, *, now=None, fetcher=yahoo_snapshot, max_workers=3):
    target = Path(target)
    now = (now or datetime.now(JST)).astimezone(JST)
    retrieved_at = now.isoformat()
    status_path = target / "market_consensus_status.json"
    status = read_json(target / "bbs_ranking_status.json")
    manifest = read_json(target / "manifest.json")
    # "Analysis day" is the completed equity session, not the wall-clock day.
    # This matters for delayed/manual runs after midnight JST and still forbids
    # substituting a ranking from a different market session.
    expected_date = manifest.get("expected_equity_date") or now.date().isoformat()
    if status.get("status") != "success" or status.get("ranking_date") != expected_date:
        result = {
            "status": "failed", "reason": "当日の掲示板ランキング取得失敗",
            "attempted_at": retrieved_at, "ranking_status": status,
        }
        atomic_json(status_path, result)
        (target / "market_consensus_latest.csv").unlink(missing_ok=True)
        raise RuntimeError(result["reason"])
    ranking = pd.read_csv(target / "bbs_ranking_latest.csv", dtype={"stock_code": str})
    ranking = ranking.sort_values("rank").head(100)
    if ranking.empty or ranking["date"].astype(str).nunique() != 1 or str(ranking.iloc[0]["date"]) != expected_date:
        raise RuntimeError("当日の掲示板ランキング取得失敗")
    codes = ranking["stock_code"].tolist()
    started = time.monotonic()
    errors = {}
    def one(code):
        try:
            return fetcher(code)
        except Exception as exc:
            errors[code] = (type(exc).__name__ + ": " + str(exc))[:200]
            return unavailable_snapshot(code)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        snapshots = list(pool.map(one, codes))
    rows = []
    for source, snapshot in zip(ranking.to_dict("records"), snapshots):
        code = source["stock_code"]
        local = local_market_data(target, code, expected_date)
        price, issues, insufficient = inspect(snapshot, local, expected_date)
        if code in errors:
            insufficient.append("provider_fetch_failed")
        classification, score = classify(snapshot, price, issues, insufficient)
        valid_targets = (
            not issues and all(snapshot.get(k) is not None for k in ("target_low", "target_mean", "target_high"))
        )
        bear, base, bull, method = scenario_prices(snapshot, price, local.get("atr14"), valid_targets)
        current_year, next_year = snapshot["current_year"], snapshot["next_year"]
        eps_move = eps_change(current_year["eps"], current_year["eps_30d"])
        target_upside = (
            (snapshot["target_mean"] / price - 1) * 100
            if price and snapshot.get("target_mean") is not None and not issues else None
        )
        notes = issues + insufficient
        rows.append({
            "date": expected_date, "rank": int(source["rank"]), "stock_code": code,
            "stock_name": source["stock_name"], "market": source["market"],
            "current_price": price,
            "price_source": "published_stock_close" if local.get("date") == expected_date else "provider_snapshot",
            "provider_price": snapshot.get("provider_price"),
            "provider_price_date": snapshot.get("provider_price_date"),
            "target_low": snapshot.get("target_low"), "target_mean": snapshot.get("target_mean"),
            "target_high": snapshot.get("target_high"), "target_upside_pct": target_upside,
            "recommendation_key": snapshot.get("recommendation_key"),
            "recommendation_mean": snapshot.get("recommendation_mean"),
            "analyst_count": snapshot.get("analyst_count"),
            "current_year_end": current_year["end_date"], "current_year_eps": current_year["eps"],
            "current_year_eps_30d": current_year["eps_30d"],
            "current_year_eps_change_pct": eps_move,
            "next_year_end": next_year["end_date"], "next_year_eps": next_year["eps"],
            "classification": classification, "classification_score": score,
            "data_quality": "invalid" if issues else "insufficient" if insufficient else "complete",
            "data_note": ";".join(notes) if notes else "complete",
            "bear_price": bear, "base_price": base, "bull_price": bull,
            "scenario_price_method": method, "source_url": snapshot["source_url"],
            "data_reference_date": expected_date, "retrieved_at": retrieved_at,
        })
    latest = pd.DataFrame(rows, columns=COLUMNS)
    daily = target / "market_consensus" / "daily" / (expected_date + ".csv")
    atomic_csv(daily, latest)
    history_path = target / "market_consensus_history.csv"
    old = pd.read_csv(history_path, dtype={"stock_code": str}) if history_path.exists() else pd.DataFrame(columns=COLUMNS)
    history = pd.concat([old, latest], ignore_index=True)
    history = history.drop_duplicates(["date", "stock_code"], keep="last").sort_values(["date", "rank"])
    atomic_csv(target / "market_consensus_latest.csv", latest)
    atomic_csv(history_path, history[COLUMNS])
    counts = latest["classification"].value_counts().to_dict()
    quality_counts = latest["data_quality"].value_counts().to_dict()
    result = {
        "status": "success", "ranking_date": expected_date, "target_count": len(latest),
        "fetch_success_count": len(latest) - len(errors), "fetch_failure_count": len(errors),
        "classification_counts": counts, "quality_counts": quality_counts,
        "unclassifiable_count": counts.get("判定不能", 0),
        "failed_codes": [{"stock_code": code, "error": error} for code, error in sorted(errors.items())],
        "retrieved_at": retrieved_at, "elapsed_seconds": round(time.monotonic() - started, 2),
        "classification_rule": {
            "minimum_analysts": 3,
            "bullish": "at least 2 of target upside >=10%, buy rating, EPS 30d change >=5%",
            "bearish": "at least 2 of target upside <=-10%, sell rating, EPS 30d change <=-5%",
            "base": "otherwise, only when required data pass",
            "missing_or_suspicious": "判定不能",
        },
        "scenario_rule": "Analyst low/mean/high when valid; otherwise current close +/-2 ATR14 (or +/-10%) labelled non-consensus.",
    }
    atomic_json(status_path, result)
    artifact_names = [
        "market_consensus_latest.csv",
        "market_consensus_history.csv",
        "market_consensus_status.json",
    ]
    hash_path = target / "sha256.json"
    if hash_path.exists():
        hashes = read_json(hash_path)
        for name in artifact_names:
            hashes[name] = hashlib.sha256((target / name).read_bytes()).hexdigest()
        atomic_json(hash_path, hashes)
    bundle = target / "chatgpt_120d.zip"
    if bundle.exists():
        temp = target / "chatgpt_120d.consensus.tmp.zip"
        with ZipFile(bundle) as old_zip, ZipFile(temp, "w", ZIP_DEFLATED) as new_zip:
            for entry in old_zip.infolist():
                if entry.filename not in artifact_names:
                    new_zip.writestr(entry, old_zip.read(entry.filename))
            for name in artifact_names:
                new_zip.write(target / name, name)
        temp.replace(bundle)
    return result


def main():
    parser = argparse.ArgumentParser(description="掲示板ランキング上位100銘柄の市場コンセンサスを保存")
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(collect(args.target), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
