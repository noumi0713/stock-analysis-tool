from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

JST = ZoneInfo("Asia/Tokyo")
TICKER_RE = re.compile(r"^[0-9A-Z]{4}\.T$")
PRICE_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
ACTION_COLS = ["Dividends", "Stock Splits"]
BATCH_SIZE = 150
MAX_RETRIES = 3


@dataclass
class QualityResult:
    dataset: str
    rows: int
    tickers: int
    success_count: int
    failure_count: int
    missing_rate: float
    oldest: str | None
    latest: str | None
    duplicate_rows: int
    ohlc_errors: int
    abnormal_rows: int
    split_anomalies: int
    future_rows: int
    quality: str
    risks: list[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build isolated market-environment history dataset")
    p.add_argument("--root", default="data/market_history")
    p.add_argument("--mode", choices=["bootstrap", "update"], default="bootstrap")
    p.add_argument("--universe-file", default="")
    p.add_argument("--asof", default="")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--min-success-rate", type=float, default=0.95)
    return p.parse_args()


def iso_day(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def atomic_replace(staged: Path, production: Path) -> None:
    production.parent.mkdir(parents=True, exist_ok=True)
    backup = production.with_name(production.name + ".backup")
    if backup.exists():
        shutil.rmtree(backup)
    if production.exists():
        production.rename(backup)
    staged.rename(production)
    if backup.exists():
        shutil.rmtree(backup)


def discover_universe(repo_root: Path, explicit: str = "") -> tuple[pd.DataFrame, list[str]]:
    risks: list[str] = []
    rows: list[dict[str, str]] = []
    if explicit:
        p = Path(explicit)
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p, dtype=str)
            ticker_col = next((c for c in df.columns if c.lower() in {"ticker", "symbol", "code"}), None)
            if ticker_col is None:
                raise ValueError("universe CSV requires ticker/symbol/code column")
            for _, r in df.iterrows():
                raw = str(r[ticker_col]).strip().upper()
                ticker = raw if raw.endswith(".T") else raw + ".T"
                rows.append({
                    "ticker": ticker,
                    "name": str(r.get("name", "")),
                    "sector": str(r.get("sector", r.get("industry", ""))),
                    "source": str(p),
                })
        else:
            for line in p.read_text(encoding="utf-8").splitlines():
                raw = line.strip().upper()
                if raw:
                    rows.append({"ticker": raw if raw.endswith(".T") else raw + ".T", "name": "", "sector": "", "source": str(p)})
    else:
        latest = repo_root / "dashboard-data" / "latest.json"
        if latest.exists():
            obj = json.loads(latest.read_text(encoding="utf-8"))

            def walk(x: Any) -> None:
                if isinstance(x, dict):
                    vals = {str(v).strip().upper() for v in x.values() if isinstance(v, str)}
                    ticker = next((v for v in vals if TICKER_RE.match(v)), None)
                    if ticker:
                        sector = ""
                        name = ""
                        for k, v in x.items():
                            lk = str(k).lower()
                            if isinstance(v, str) and lk in {"sector", "industry", "sector_name", "industry_name", "業種"}:
                                sector = v
                            if isinstance(v, str) and lk in {"name", "company_name", "銘柄名"}:
                                name = v
                        rows.append({"ticker": ticker, "name": name, "sector": sector, "source": str(latest)})
                    for v in x.values():
                        walk(v)
                elif isinstance(x, list):
                    for v in x:
                        walk(v)
                elif isinstance(x, str):
                    s = x.strip().upper()
                    if TICKER_RE.match(s):
                        rows.append({"ticker": s, "name": "", "sector": "", "source": str(latest)})
            walk(obj)

        prime = repo_root / "momentum5d" / "config" / "prime_tickers.txt"
        if prime.exists():
            for raw in prime.read_text(encoding="utf-8").splitlines():
                raw = raw.strip().upper()
                if raw:
                    rows.append({"ticker": raw if raw.endswith(".T") else raw + ".T", "name": "", "sector": "", "source": str(prime)})

    df = pd.DataFrame(rows).drop_duplicates("ticker") if rows else pd.DataFrame(columns=["ticker", "name", "sector", "source"])
    df = df[df["ticker"].astype(str).str.match(TICKER_RE)].sort_values("ticker").reset_index(drop=True)
    if len(df) < 3000:
        risks.append(f"current universe has only {len(df)} TSE-style tickers; full-TSE coverage is not proven")
    risks.append("historical delisted issues beyond available universe sources may be absent; survivor bias remains")
    return df, risks


def normalize_download(raw: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    failures: list[str] = []
    frames: list[pd.DataFrame] = []
    if raw.empty:
        return pd.DataFrame(), tickers
    multi = isinstance(raw.columns, pd.MultiIndex)
    for ticker in tickers:
        try:
            if multi:
                if ticker in raw.columns.get_level_values(0):
                    part = raw[ticker].copy()
                elif ticker in raw.columns.get_level_values(1):
                    part = raw.xs(ticker, axis=1, level=1).copy()
                else:
                    failures.append(ticker)
                    continue
            else:
                if len(tickers) != 1:
                    failures.append(ticker)
                    continue
                part = raw.copy()
            part = part.reset_index()
            date_col = next((c for c in part.columns if str(c).lower() in {"date", "datetime"}), part.columns[0])
            part = part.rename(columns={date_col: "Date"})
            part["Date"] = pd.to_datetime(part["Date"], errors="coerce").dt.tz_localize(None)
            part = part.dropna(subset=["Date"])
            if part.empty or "Close" not in part or part["Close"].dropna().empty:
                failures.append(ticker)
                continue
            for c in PRICE_COLS + ACTION_COLS:
                if c not in part:
                    part[c] = np.nan if c != "Stock Splits" else 0.0
            # yf.download aligns every ticker in a batch to the union of all
            # trading dates.  Rows on which this ticker did not trade are
            # therefore all-NA padding, not observations.  Keeping them made
            # the dataset appear to have ~5% missing prices and failed the
            # publication gate even though the downloads themselves succeeded.
            part = part.dropna(subset=["Close"])
            part["Ticker"] = ticker
            frames.append(part[["Date", "Ticker"] + PRICE_COLS + ACTION_COLS])
        except Exception:
            failures.append(ticker)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), failures


def download_batches(tickers: list[str], start: date, end: date, batch_size: int) -> tuple[pd.DataFrame, list[str]]:
    all_frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for offset in range(0, len(tickers), batch_size):
        batch = tickers[offset : offset + batch_size]
        last_fail = batch
        for attempt in range(MAX_RETRIES):
            try:
                raw = yf.download(
                    batch,
                    start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(),
                    auto_adjust=False,
                    actions=True,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                    timeout=30,
                )
                norm, last_fail = normalize_download(raw, batch)
                if not norm.empty:
                    all_frames.append(norm)
                if len(last_fail) < len(batch):
                    break
            except Exception:
                last_fail = batch
            time.sleep(2**attempt)
        # retry individual failures to distinguish batch/rate-limit failures
        for ticker in list(last_fail):
            got = False
            for attempt in range(MAX_RETRIES):
                try:
                    raw = yf.download(
                        ticker,
                        start=start.isoformat(),
                        end=(end + timedelta(days=1)).isoformat(),
                        auto_adjust=False,
                        actions=True,
                        progress=False,
                        timeout=30,
                    )
                    norm, miss = normalize_download(raw, [ticker])
                    if not norm.empty and not miss:
                        all_frames.append(norm)
                        got = True
                        break
                except Exception:
                    pass
                time.sleep(2**attempt)
            if not got:
                failed.append(ticker)
    out = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    if not out.empty:
        out = out.drop_duplicates(["Ticker", "Date"], keep="last").sort_values(["Ticker", "Date"])
    return out, sorted(set(failed))


def quarantine_invalid_ohlc(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove internally impossible OHLC rows without fabricating prices."""
    if df.empty:
        return df, pd.DataFrame(columns=df.columns)
    required = ["Open", "High", "Low", "Close"]
    complete = df[required].notna().all(axis=1)
    upper = df[["Open", "Close"]].max(axis=1)
    lower = df[["Open", "Close"]].min(axis=1)
    invalid = complete & ((df["High"] < upper) | (df["Low"] > lower) | (df["High"] < df["Low"]))
    quarantine = df.loc[invalid].copy()
    clean = df.loc[~invalid].copy()
    return clean, quarantine


def merge_existing(new: pd.DataFrame, production_raw: Path) -> pd.DataFrame:
    parts = list(production_raw.rglob("*.parquet")) if production_raw.exists() else []
    if not parts:
        return new
    old = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    if new.empty:
        return old
    return pd.concat([old, new], ignore_index=True).drop_duplicates(["Ticker", "Date"], keep="last")


def write_year_partitions(df: pd.DataFrame, root: Path, dataset: str) -> None:
    target = root / dataset
    target.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    temp = df.copy()
    temp["year"] = pd.to_datetime(temp["Date"]).dt.year
    for year, part in temp.groupby("year"):
        out = target / f"year={int(year)}"
        out.mkdir(parents=True, exist_ok=True)
        part.drop(columns="year").to_parquet(out / "part.parquet", index=False, compression="zstd")


def load_manifest(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "momentum5d" / "config" / "market_history_tickers.json"
    return json.loads(path.read_text(encoding="utf-8"))


def download_named(manifest: dict[str, Any], group: str, start: date, end: date) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    mapping: dict[str, Any] = {}
    for key, spec in manifest[group].items():
        ticker = spec.get("ticker")
        mapping[key] = spec
        if spec.get("disabled") or not ticker:
            failures.append(key)
            continue
        df, miss = download_batches([ticker], start, end, 1)
        if miss or df.empty:
            failures.append(key)
            continue
        df["Series"] = key
        df["FormalName"] = spec.get("name", "")
        frames.append(df)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), failures, mapping


def compute_market_features(equities: pd.DataFrame, universe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    risks: list[str] = []
    if equities.empty:
        return pd.DataFrame(), pd.DataFrame(), ["equities empty: market internals unavailable"]
    x = equities.copy().sort_values(["Ticker", "Date"])
    x["prev_close"] = x.groupby("Ticker")["Close"].shift(1)
    x["ret"] = x["Close"] / x["prev_close"] - 1.0
    x["trading_value"] = x["Close"] * x["Volume"]
    x["high20_prev"] = x.groupby("Ticker")["High"].transform(lambda s: s.shift(1).rolling(20, min_periods=20).max())
    x["low20_prev"] = x.groupby("Ticker")["Low"].transform(lambda s: s.shift(1).rolling(20, min_periods=20).min())
    x["high252_prev"] = x.groupby("Ticker")["High"].transform(lambda s: s.shift(1).rolling(252, min_periods=200).max())
    x["low252_prev"] = x.groupby("Ticker")["Low"].transform(lambda s: s.shift(1).rolling(252, min_periods=200).min())
    x["up"] = x["ret"] > 0
    x["down"] = x["ret"] < 0
    x["flat"] = x["ret"] == 0
    x["new_high20"] = x["High"] > x["high20_prev"]
    x["new_low20"] = x["Low"] < x["low20_prev"]
    x["new_high52w"] = x["High"] > x["high252_prev"]
    x["new_low52w"] = x["Low"] < x["low252_prev"]
    daily = x.groupby("Date").agg(
        advancers=("up", "sum"), decliners=("down", "sum"), unchanged=("flat", "sum"),
        new_high_20d=("new_high20", "sum"), new_low_20d=("new_low20", "sum"),
        new_high_52w=("new_high52w", "sum"), new_low_52w=("new_low52w", "sum"),
        market_trading_value=("trading_value", "sum"),
        advancing_trading_value=("trading_value", lambda s: s[x.loc[s.index, "up"]].sum()),
    ).reset_index()
    daily["advancing_trading_value_ratio"] = daily["advancing_trading_value"] / daily["market_trading_value"].replace(0, np.nan)
    daily["trading_value_change"] = daily["market_trading_value"].pct_change()
    # 25-day advance/decline ratio, a common Japan-market implementation; definition is recorded in metadata.
    adv25 = daily["advancers"].rolling(25, min_periods=25).sum()
    dec25 = daily["decliners"].rolling(25, min_periods=25).sum()
    daily["advance_decline_ratio_25d"] = 100.0 * adv25 / dec25.replace(0, np.nan)

    sector_map = universe.set_index("ticker")["sector"].to_dict() if "sector" in universe else {}
    x["sector"] = x["Ticker"].map(sector_map).fillna("")
    known = x[x["sector"].astype(str).str.len() > 0].copy()
    if known.empty:
        risks.append("sector classification unavailable; sector aggregates not produced")
        sector = pd.DataFrame()
    else:
        sector = known.groupby(["Date", "sector"]).agg(
            equal_weight_return=("ret", "mean"),
            trading_value=("trading_value", "sum"),
            constituents=("Ticker", "nunique"),
        ).reset_index()
        sector["trading_value_change"] = sector.groupby("sector")["trading_value"].pct_change()
        risks.append("sector returns are equal-weighted using available classification; historical membership may contain look-ahead/survivorship bias")
    return daily, sector, risks


def quality_check(df: pd.DataFrame, dataset: str, requested: int, failures: list[str], asof: date, extra_risks: list[str], min_success_rate: float, enforce_ohlc: bool = True) -> QualityResult:
    risks = list(extra_risks)
    if df.empty:
        return QualityResult(dataset, 0, 0, 0, len(failures), 1.0, None, None, 0, 0, 0, 0, 0, "FAIL", risks + ["dataset empty"])
    d = df.copy()
    dates = pd.to_datetime(d["Date"], errors="coerce")
    dup = int(d.duplicated(["Ticker", "Date"]).sum()) if "Ticker" in d else int(d.duplicated(["Date"]).sum())
    missing = float(d[PRICE_COLS].isna().mean().mean()) if all(c in d for c in PRICE_COLS) else float(d.isna().mean().mean())
    ohlc = 0
    abnormal = 0
    split_anomalies = 0
    if all(c in d for c in ["Open", "High", "Low", "Close"]):
        ohlc_mask = (d["High"] < d[["Open", "Close", "Low"]].max(axis=1)) | (d["Low"] > d[["Open", "Close", "High"]].min(axis=1))
        ohlc = int(ohlc_mask.fillna(False).sum())
        abnormal = int(((d["Close"] <= 0) | (d["Volume"] < 0)).fillna(False).sum())
    if "Stock Splits" in d and "Ticker" in d:
        tmp = d.sort_values(["Ticker", "Date"]).copy()
        tmp["ret_abs"] = tmp.groupby("Ticker")["Close"].pct_change().abs()
        split_anomalies = int(((tmp["Stock Splits"].fillna(0) != 0) & (tmp["ret_abs"] < 0.10)).sum())
    future = int((dates.dt.date > asof).sum())
    success = max(0, requested - len(failures)) if requested else int(d["Ticker"].nunique() if "Ticker" in d else 1)
    success_rate = success / requested if requested else 1.0
    quality = "PASS"
    if success_rate < min_success_rate or dup or (enforce_ohlc and ohlc) or abnormal or future:
        quality = "FAIL"
    if missing > 0.05:
        quality = "FAIL"
    if split_anomalies:
        risks.append(f"{split_anomalies} split-event rows require manual inspection")
    return QualityResult(
        dataset=dataset, rows=len(d), tickers=int(d["Ticker"].nunique() if "Ticker" in d else 1),
        success_count=success, failure_count=len(failures), missing_rate=missing,
        oldest=iso_day(dates.min()), latest=iso_day(dates.max()), duplicate_rows=dup,
        ohlc_errors=ohlc, abnormal_rows=abnormal, split_anomalies=split_anomalies,
        future_rows=future, quality=quality, risks=risks,
    )


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    root = (repo_root / "momentum5d" / args.root).resolve() if not Path(args.root).is_absolute() else Path(args.root)
    asof = date.fromisoformat(args.asof) if args.asof else datetime.now(JST).date()
    universe, universe_risks = discover_universe(repo_root, args.universe_file)
    if universe.empty:
        raise RuntimeError("No TSE universe discovered. Supply --universe-file; existing strategy files are not modified.")

    raw_prod = root / "raw"
    feat_prod = root / "features"
    quality_prod = root / "quality"
    stage_parent = root.parent
    stage = Path(tempfile.mkdtemp(prefix="market_history_stage_", dir=stage_parent))
    stage_raw = stage / "raw"
    stage_feat = stage / "features"
    stage_quality = stage / "quality"

    eq_start = asof - timedelta(days=366 * 5 + 15)
    long_start = asof - timedelta(days=366 * 10 + 15)
    if args.mode == "update" and raw_prod.exists():
        existing = list((raw_prod / "equities").rglob("*.parquet"))
        if existing:
            old_max = max(pd.read_parquet(p, columns=["Date"])["Date"].max() for p in existing)
            eq_start = pd.Timestamp(old_max).date() - timedelta(days=7)

    equities_new, eq_fail = download_batches(universe["ticker"].tolist(), eq_start, asof, args.batch_size)
    equities = merge_existing(equities_new, raw_prod / "equities") if args.mode == "update" else equities_new
    manifest = load_manifest(repo_root)
    indexes, index_fail, index_map = download_named(manifest, "indexes", long_start, asof)
    external, ext_fail, ext_map = download_named(manifest, "external", long_start, asof)

    # Yahoo occasionally returns a close outside the reported daily high/low
    # for a very small number of Japanese equity rows.  Do not "repair" these
    # by inventing a high/low: isolate the source rows and certify only the
    # internally consistent observations.
    equities, equity_quarantine = quarantine_invalid_ohlc(equities)
    indexes, index_quarantine = quarantine_invalid_ohlc(indexes)

    write_year_partitions(equities, stage_raw, "equities")
    write_year_partitions(indexes, stage_raw, "indexes")
    write_year_partitions(external, stage_raw, "external")
    universe.to_parquet(stage_raw / "universe.parquet", index=False, compression="zstd")
    (stage_raw / "ticker_mapping.json").write_text(json.dumps({"indexes": index_map, "external": ext_map}, ensure_ascii=False, indent=2), encoding="utf-8")
    quarantine_root = stage_raw / "quarantine"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    if not equity_quarantine.empty:
        equity_quarantine.to_parquet(quarantine_root / "equities_invalid_ohlc.parquet", index=False, compression="zstd")
    if not index_quarantine.empty:
        index_quarantine.to_parquet(quarantine_root / "indexes_invalid_ohlc.parquet", index=False, compression="zstd")

    internals, sectors, feature_risks = compute_market_features(equities, universe)
    write_year_partitions(internals.rename(columns={"Date": "Date"}), stage_feat, "market_internals")
    if not sectors.empty:
        write_year_partitions(sectors.rename(columns={"Date": "Date"}), stage_feat, "sectors")
    metadata = {
        "created_at_jst": datetime.now(JST).isoformat(),
        "asof": asof.isoformat(),
        "equity_source": "yfinance",
        "batch_size": args.batch_size,
        "retry_policy": "exponential backoff, minimum 3 attempts; failed batch members retried individually",
        "timezone": "daily dates normalized timezone-naive; operational timezone Asia/Tokyo",
        "trading_value_formula": "unadjusted Close * Volume",
        "advance_decline_ratio": "25-day sum(advancers) / sum(decliners) * 100",
        "sector_return_method": "equal-weight return where sector classification is available",
        "staging_policy": "write temporary tree; replace production only if all critical quality gates PASS",
        "delisted_policy": "record observed delisted symbols if present in supplied/discovered universe; no fabricated historical constituents",
        "survivorship_bias": "historical delisted coverage cannot be guaranteed from current-universe sources; explicitly reported",
        "strategy_frozen": True,
        "strategy_files_modified": [],
    }
    stage_feat.mkdir(parents=True, exist_ok=True)
    (stage_feat / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    equity_quality_risks = list(universe_risks)
    if not equity_quarantine.empty:
        equity_quality_risks.append(f"{len(equity_quarantine)} source OHLC rows quarantined; no price was fabricated")
    q_eq = quality_check(equities, "TSE equities", len(universe), eq_fail, asof, equity_quality_risks, args.min_success_rate)
    q_idx = quality_check(indexes, "indexes", len(manifest["indexes"]), index_fail, asof, ["unavailable requested indexes are reported; no substitution is used"], 0.50)
    enabled_external = [k for k, v in manifest["external"].items() if not v.get("disabled")]
    q_ext = quality_check(
        external, "external", len(enabled_external),
        [x for x in ext_fail if x in enabled_external], asof,
        ["exact requested series only; disabled/unavailable series are not proxied",
         "FX daily open/close can fall marginally outside Yahoo high/low because of session-boundary conventions; reported but not rejected"],
        0.85, enforce_ohlc=False,
    )

    critical_pass = q_eq.quality == "PASS" and q_ext.quality == "PASS"
    # Index group may have explicit unavailable items. It fails publication only if configured exact tickers themselves are materially missing.
    configured_indexes = [
        k for k, v in manifest["indexes"].items()
        if v.get("ticker") and not v.get("allow_unavailable")
    ]
    configured_index_fail = [x for x in index_fail if x in configured_indexes]
    if configured_indexes and len(configured_index_fail) / len(configured_indexes) > 0.5:
        critical_pass = False

    stage_quality.mkdir(parents=True, exist_ok=True)
    (stage_quality / "failed_equities.txt").write_text("\n".join(eq_fail), encoding="utf-8")
    report = {
        "status": "PASS" if critical_pass else "FAIL",
        "published": bool(critical_pass),
        "quality": [q_eq.__dict__, q_idx.__dict__, q_ext.__dict__],
        "residual_risks": sorted(set(universe_risks + feature_risks + q_idx.risks + q_ext.risks)),
        "strategy_frozen": True,
    }
    (stage_quality / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_rows = []
    for q, period, target in [
        (q_eq, "5y", "all discovered TSE tickers"),
        (q_idx, "10y", "configured requested indexes"),
        (q_ext, "10y", "configured external series"),
    ]:
        summary_rows.append({
            "data_category": q.dataset, "period": period, "target": target,
            "success_count": q.success_count, "failure_count": q.failure_count,
            "missing_rate": q.missing_rate, "oldest": q.oldest, "latest": q.latest,
            "file_size_bytes": 0, "save_path": str(root), "quality": q.quality,
            "residual_risk": "; ".join(q.risks),
        })
    pd.DataFrame(summary_rows).to_csv(stage_quality / "completion_report.csv", index=False)

    if critical_pass:
        # Keep raw and features isolated from strategy/backtest data. Replace only after successful checks.
        if raw_prod.exists():
            shutil.rmtree(raw_prod)
        if feat_prod.exists():
            shutil.rmtree(feat_prod)
        if quality_prod.exists():
            shutil.rmtree(quality_prod)
        root.mkdir(parents=True, exist_ok=True)
        stage_raw.rename(raw_prod)
        stage_feat.rename(feat_prod)
        stage_quality.rename(quality_prod)
        shutil.rmtree(stage, ignore_errors=True)
        summary_path = quality_prod / "completion_report.csv"
        summary = pd.read_csv(summary_path)
        summary["file_size_bytes"] = [dir_size(raw_prod / "equities"), dir_size(raw_prod / "indexes"), dir_size(raw_prod / "external")]
        summary.to_csv(summary_path, index=False)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    failed_root = root / "failed_staging" / datetime.now(JST).strftime("%Y%m%dT%H%M%S")
    failed_root.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(failed_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
