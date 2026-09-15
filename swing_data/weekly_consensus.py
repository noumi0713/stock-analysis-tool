"""Weekly analyst-consensus snapshots for the full JPX domestic common-stock universe.

The automated provider is Yahoo Finance via the existing yfinance quoteSummary
adapter. MarketScreener is intentionally not scraped in bulk because its terms
restrict downloading/reuse of site contents without prior permission.

This module supports sharded collection for GitHub Actions and writes one dated
CSV snapshot plus a latest CSV and status JSON. Historical records are retained
as weekly files to avoid an ever-growing monolithic CSV.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import pandas as pd

from swing_data.bbs_ranking import atomic_csv
from swing_data.collector import atomic_json
from swing_data.market_consensus import eps_change, unavailable_snapshot, yahoo_snapshot

JST = ZoneInfo("Asia/Tokyo")
PROVIDER = "Yahoo Finance"
PROVIDER_METHOD = "yfinance quoteSummary"
COLUMNS = [
    "snapshot_date", "stock_code", "stock_name", "market", "sector17", "sector33",
    "provider", "provider_method", "provider_symbol", "provider_price",
    "provider_price_date", "target_low", "target_mean", "target_high",
    "target_upside_pct", "recommendation_key", "recommendation_mean",
    "analyst_count", "current_year_end", "current_year_eps", "current_year_eps_30d",
    "current_year_eps_change_pct", "next_year_end", "next_year_eps",
    "next_year_eps_30d", "next_year_eps_change_pct", "target_mean_wow_pct",
    "current_year_eps_wow_pct", "next_year_eps_wow_pct", "analyst_count_change",
    "fetch_status", "data_quality", "error", "source_url", "retrieved_at",
]


def load_universe(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if "stock_code" not in frame:
        raise ValueError("universe.csv に stock_code がありません")
    frame["stock_code"] = frame["stock_code"].str.upper().str.strip()
    frame = frame[frame["stock_code"].str.fullmatch(r"[0-9A-Z]{4}")].copy()
    if frame.empty or frame["stock_code"].duplicated().any():
        raise ValueError("universe.csv の銘柄コードが不正です")
    frame = frame.rename(columns={"company_name": "stock_name"})
    for col in ["stock_name", "market", "sector17", "sector33"]:
        if col not in frame:
            frame[col] = ""
    return frame[["stock_code", "stock_name", "market", "sector17", "sector33"]].sort_values("stock_code")


def _quality(snapshot: dict) -> str:
    targets = [snapshot.get("target_low"), snapshot.get("target_mean"), snapshot.get("target_high")]
    analysts = snapshot.get("analyst_count")
    if all(v is not None for v in targets) and analysts is not None and analysts >= 1:
        return "complete_targets"
    if any(v is not None for v in targets) or analysts is not None:
        return "partial"
    return "no_consensus"


def _row(meta: dict, snapshot: dict, *, snapshot_date: str, retrieved_at: str, error: str = "") -> dict:
    current_year = snapshot["current_year"]
    next_year = snapshot["next_year"]
    price = snapshot.get("provider_price")
    mean = snapshot.get("target_mean")
    upside = (mean / price - 1) * 100 if price and mean is not None else None
    return {
        "snapshot_date": snapshot_date,
        "stock_code": meta["stock_code"],
        "stock_name": meta["stock_name"],
        "market": meta["market"],
        "sector17": meta["sector17"],
        "sector33": meta["sector33"],
        "provider": PROVIDER,
        "provider_method": PROVIDER_METHOD,
        "provider_symbol": meta["stock_code"] + ".T",
        "provider_price": price,
        "provider_price_date": snapshot.get("provider_price_date"),
        "target_low": snapshot.get("target_low"),
        "target_mean": mean,
        "target_high": snapshot.get("target_high"),
        "target_upside_pct": upside,
        "recommendation_key": snapshot.get("recommendation_key"),
        "recommendation_mean": snapshot.get("recommendation_mean"),
        "analyst_count": snapshot.get("analyst_count"),
        "current_year_end": current_year.get("end_date"),
        "current_year_eps": current_year.get("eps"),
        "current_year_eps_30d": current_year.get("eps_30d"),
        "current_year_eps_change_pct": eps_change(current_year.get("eps"), current_year.get("eps_30d")),
        "next_year_end": next_year.get("end_date"),
        "next_year_eps": next_year.get("eps"),
        "next_year_eps_30d": next_year.get("eps_30d"),
        "next_year_eps_change_pct": eps_change(next_year.get("eps"), next_year.get("eps_30d")),
        "target_mean_wow_pct": None,
        "current_year_eps_wow_pct": None,
        "next_year_eps_wow_pct": None,
        "analyst_count_change": None,
        "fetch_status": "failed" if error else "success",
        "data_quality": "fetch_failed" if error else _quality(snapshot),
        "error": error,
        "source_url": snapshot.get("source_url"),
        "retrieved_at": retrieved_at,
    }


def collect_shard(
    universe_path: Path,
    output_path: Path,
    *,
    shard_index: int,
    shard_count: int,
    max_workers: int = 2,
    fetcher=yahoo_snapshot,
    now: datetime | None = None,
) -> dict:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index/shard_count が不正です")
    now = (now or datetime.now(JST)).astimezone(JST)
    snapshot_date = now.date().isoformat()
    retrieved_at = now.isoformat()
    universe = load_universe(universe_path).reset_index(drop=True)
    shard = universe.iloc[[i for i in range(len(universe)) if i % shard_count == shard_index]].copy()

    def one(meta: dict) -> dict:
        code = meta["stock_code"]
        try:
            snap = fetcher(code)
            return _row(meta, snap, snapshot_date=snapshot_date, retrieved_at=retrieved_at)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:300]
            return _row(meta, unavailable_snapshot(code), snapshot_date=snapshot_date,
                        retrieved_at=retrieved_at, error=error)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        rows = list(pool.map(one, shard.to_dict("records")))
    result = pd.DataFrame(rows, columns=COLUMNS)
    atomic_csv(output_path, result)
    return {
        "status": "success",
        "snapshot_date": snapshot_date,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "row_count": len(result),
        "success_count": int((result["fetch_status"] == "success").sum()) if len(result) else 0,
        "failure_count": int((result["fetch_status"] == "failed").sum()) if len(result) else 0,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def _pct_change(current, prior):
    try:
        current = float(current)
        prior = float(prior)
    except (TypeError, ValueError):
        return None
    if pd.isna(current) or pd.isna(prior) or prior == 0:
        return None
    return (current - prior) / abs(prior) * 100


def apply_week_over_week(latest: pd.DataFrame, previous: pd.DataFrame | None) -> pd.DataFrame:
    if previous is None or previous.empty:
        return latest
    fields = {
        "target_mean": "target_mean_wow_pct",
        "current_year_eps": "current_year_eps_wow_pct",
        "next_year_eps": "next_year_eps_wow_pct",
    }
    prev = previous.set_index("stock_code")
    for i, row in latest.iterrows():
        code = row["stock_code"]
        if code not in prev.index:
            continue
        prior = prev.loc[code]
        for source, dest in fields.items():
            latest.at[i, dest] = _pct_change(row.get(source), prior.get(source))
        try:
            a = float(row.get("analyst_count"))
            b = float(prior.get("analyst_count"))
            if not pd.isna(a) and not pd.isna(b):
                latest.at[i, "analyst_count_change"] = a - b
        except (TypeError, ValueError):
            pass
    return latest


def merge_shards(universe_path: Path, shards_dir: Path, target: Path, *, now: datetime | None = None) -> dict:
    now = (now or datetime.now(JST)).astimezone(JST)
    snapshot_date = now.date().isoformat()
    retrieved_at = now.isoformat()
    universe = load_universe(universe_path)
    files = sorted(shards_dir.rglob("*.csv"))
    if not files:
        raise RuntimeError("shard CSV がありません")
    frames = [pd.read_csv(path, dtype={"stock_code": str}) for path in files]
    latest = pd.concat(frames, ignore_index=True)
    latest["stock_code"] = latest["stock_code"].astype(str).str.zfill(4)
    latest = latest.drop_duplicates("stock_code", keep="last").sort_values("stock_code").reset_index(drop=True)

    expected = set(universe["stock_code"])
    actual = set(latest["stock_code"])
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"週次コンセンサスの銘柄集合が不一致 missing={len(missing)} unexpected={len(unexpected)}"
        )

    latest["snapshot_date"] = snapshot_date
    latest["retrieved_at"] = latest["retrieved_at"].fillna(retrieved_at)

    target = Path(target)
    snapshots = target / "weekly_consensus"
    snapshots.mkdir(parents=True, exist_ok=True)
    previous_files = [p for p in snapshots.glob("*.csv") if p.stem < snapshot_date]
    previous = None
    if previous_files:
        previous = pd.read_csv(sorted(previous_files)[-1], dtype={"stock_code": str})

    latest = apply_week_over_week(latest, previous)
    atomic_csv(snapshots / f"{snapshot_date}.csv", latest[COLUMNS])
    atomic_csv(target / "weekly_consensus_latest.csv", latest[COLUMNS])

    result = {
        "status": "success",
        "snapshot_date": snapshot_date,
        "provider": PROVIDER,
        "provider_method": PROVIDER_METHOD,
        "universe_count": len(universe),
        "success_count": int((latest["fetch_status"] == "success").sum()),
        "failure_count": int((latest["fetch_status"] == "failed").sum()),
        "complete_target_count": int((latest["data_quality"] == "complete_targets").sum()),
        "quality_counts": latest["data_quality"].value_counts().to_dict(),
        "previous_snapshot_date": sorted(previous_files)[-1].stem if previous_files else None,
        "retrieved_at": retrieved_at,
        "market_screener_bulk_scraping": "disabled_due_to_terms",
        "notes": [
            "MarketScreener は全銘柄の自動大量取得には使用しない。",
            "目標株価は1年程度のアナリストコンセンサスであり、5〜10営業日の利確目標ではない。",
            "欠損は0や中立に置換しない。",
        ],
    }
    atomic_json(target / "weekly_consensus_status.json", result)
    atomic_json(snapshots / "index.json", {
        "latest": "../weekly_consensus_latest.csv",
        "status": "../weekly_consensus_status.json",
        "snapshot_date": snapshot_date,
        "snapshots": [p.name for p in sorted(snapshots.glob("*.csv"), reverse=True)],
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect")
    collect.add_argument("--universe", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--shard-index", type=int, required=True)
    collect.add_argument("--shard-count", type=int, required=True)
    collect.add_argument("--max-workers", type=int, default=2)

    merge = sub.add_parser("merge")
    merge.add_argument("--universe", required=True)
    merge.add_argument("--shards", required=True)
    merge.add_argument("--target", required=True)

    args = parser.parse_args()
    if args.command == "collect":
        result = collect_shard(
            Path(args.universe), Path(args.output),
            shard_index=args.shard_index, shard_count=args.shard_count,
            max_workers=args.max_workers,
        )
    else:
        result = merge_shards(Path(args.universe), Path(args.shards), Path(args.target))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
