"""Collect Yahoo! Finance Japan's daily message-board post ranking.

The collector stores one immutable snapshot per Japan calendar date.  It never
substitutes an older snapshot when the current fetch fails.
"""
from __future__ import annotations

import argparse
from datetime import date as date_type, datetime, timedelta
import json
import math
import os
from pathlib import Path
import re
import time
import uuid
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from swing_data.collector import atomic_json

JST = ZoneInfo("Asia/Tokyo")
SOURCE_URL = "https://finance.yahoo.co.jp/stocks/ranking/bbs"
HISTORY_COLUMNS = [
    "date", "rank", "stock_code", "stock_name", "market", "price",
    "source_updated_at", "collected_at",
]
RISING_COLUMNS = HISTORY_COLUMNS + ["surge_score", "derivation_method"]
TREND_FIELDS = [
    "rank", "previous_rank", "rank_change", "new_entry",
    "rank_history_3d", "rank_history_5d", "consecutive_days", "best_rank",
]
UNIVERSE_COLUMNS = [
    "date", "rank", "stock_code", "stock_name", "market", "price",
    "popular_rank", "popular_previous_rank", "popular_rank_change",
    "popular_new_entry", "popular_rank_history_3d", "popular_rank_history_5d",
    "popular_consecutive_days", "popular_best_rank",
    "rising_rank", "rising_previous_rank", "rising_rank_change",
    "rising_new_entry", "rising_rank_history_3d", "rising_rank_history_5d",
    "rising_consecutive_days", "rising_best_rank", "ranking_sources",
    "source_updated_at", "collected_at",
]
RISING_METHOD = "derived_from_popular_rank_change; new_entry_baseline=list_size_plus_1"


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def parse_ranking_page(document: str) -> tuple[list[dict], dict]:
    marker = "window.__PRELOADED_STATE__"
    marker_at = document.find(marker)
    if marker_at < 0:
        raise ValueError("Yahooランキングの状態データが見つかりません")
    start = document.find("=", marker_at) + 1
    end = document.find("</script>", start)
    if start <= 0 or end < 0:
        raise ValueError("Yahooランキングの状態データを切り出せません")
    try:
        state = json.loads(document[start:end].strip())
    except json.JSONDecodeError as exc:
        raise ValueError("Yahooランキングの状態データがJSONではありません") from exc
    ranking = state.get("mainRankingList") or {}
    results = ranking.get("results") or []
    paging = ranking.get("paging") or {}
    if not results or not paging:
        raise ValueError("Yahooランキングに順位データがありません")
    rows = []
    for item in results:
        update_text = (((item.get("rankingResult") or {}).get("bbsContents") or {}).get("updateDateTime"))
        try:
            updated = datetime.strptime(update_text, "%Y/%m/%d %H:%M").replace(tzinfo=JST)
            rank = int(item["rank"])
            code = str(item["stockCode"]).strip().upper()
            price = float(str(item["savePrice"]).replace(",", ""))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Yahooランキングの行形式が不正です") from exc
        if not re.fullmatch(r"[0-9A-Z]{4}", code) or rank < 1 or not math.isfinite(price) or price <= 0:
            raise ValueError("Yahooランキングに不正な順位・銘柄コード・株価があります")
        rows.append({
            "rank": rank,
            "stock_code": code,
            "stock_name": str(item.get("stockName") or "").strip(),
            "market": str(item.get("marketName") or "").strip(),
            "price": price,
            "source_updated_at": updated.isoformat(),
        })
    return rows, paging


def fetch_snapshot(*, session: requests.Session | None = None, now: datetime | None = None,
                   attempts: int = 3, sleeper=time.sleep) -> tuple[pd.DataFrame, dict]:
    session = session or requests.Session()
    now = now or datetime.now(JST)
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; stock-analysis-tool/1.0; daily ranking research)",
        "Accept-Language": "ja-JP,ja;q=0.9,en;q=0.5",
    }
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            all_rows: list[dict] = []
            page_meta: dict | None = None
            for page in range(1, 100):
                response = session.get(SOURCE_URL, params={"market":"all", "term":"daily", "page":page},
                                       headers=headers, timeout=30)
                response.raise_for_status()
                rows, paging = parse_ranking_page(response.text)
                if page_meta is None:
                    page_meta = paging
                all_rows.extend(rows)
                if page >= int(paging.get("totalPage", 1)):
                    break
                sleeper(1)
            if not page_meta:
                raise ValueError("Yahooランキングのページ情報がありません")
            update_times = {row["source_updated_at"] for row in all_rows}
            total = int(page_meta.get("totalSize", 0))
            ranks = [row["rank"] for row in all_rows]
            codes = [row["stock_code"] for row in all_rows]
            if len(update_times) != 1:
                raise ValueError("ページ間でYahooランキングの更新日時が一致しません")
            if len(all_rows) != total or sorted(ranks) != list(range(1, total + 1)):
                raise ValueError("Yahooランキングの全順位を取得できません")
            if len(codes) != len(set(codes)):
                raise ValueError("Yahooランキングに銘柄コードの重複があります")
            source_updated_at = next(iter(update_times))
            ranking_date = datetime.fromisoformat(source_updated_at).astimezone(JST).date().isoformat()
            frame = pd.DataFrame(all_rows)
            frame.insert(0, "date", ranking_date)
            frame["collected_at"] = now.astimezone(JST).isoformat()
            return frame[HISTORY_COLUMNS], {
                "ranking_date": ranking_date,
                "source_updated_at": source_updated_at,
                "row_count": len(frame),
                "total_pages": int(page_meta.get("totalPage", 1)),
            }
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                sleeper(5 * (attempt + 1))
    raise RuntimeError(f"Yahoo掲示板投稿ランキング取得失敗: {last_error}") from last_error


def _rank_path(points: pd.DataFrame, dates: list[str], observed_dates: set[str]) -> str:
    values = []
    by_date = points.set_index("date")["rank"].to_dict() if not points.empty else {}
    for date in dates:
        rank = by_date.get(date)
        if date not in observed_dates:
            values.append(f"{date}:未取得")
        else:
            values.append(f"{date}:{int(rank)}" if pd.notna(rank) else f"{date}:圏外")
    return " > ".join(values)


def build_trends(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    trend_columns = HISTORY_COLUMNS + ["previous_rank", "rank_change", "new_entry",
        "rank_history_3d", "rank_history_5d", "consecutive_days", "best_rank"]
    exit_columns = ["date", "stock_code", "stock_name", "market", "previous_rank",
                    "consecutive_days_before_exit", "best_rank", "exited_ranking"]
    if history.empty:
        return pd.DataFrame(columns=trend_columns), pd.DataFrame(columns=exit_columns)
    work = history.copy()
    work["date"] = work.date.astype(str)
    work["rank"] = pd.to_numeric(work["rank"], errors="raise").astype(int)
    dates = sorted(work.date.unique())
    latest_date = dates[-1]
    previous_candidate = (date_type.fromisoformat(latest_date) - timedelta(days=1)).isoformat()
    previous_date = previous_candidate if previous_candidate in dates else None
    latest = work[work.date == latest_date].sort_values("rank").copy()
    previous = work[work.date == previous_date] if previous_date else work.iloc[0:0]
    previous_ranks = previous.set_index("stock_code")["rank"].to_dict()
    date_codes = {date:set(work.loc[work.date == date, "stock_code"]) for date in dates}
    all_best = work.groupby("stock_code")["rank"].min().to_dict()
    output = []
    for row in latest.to_dict("records"):
        code = row["stock_code"]
        prior = previous_ranks.get(code)
        consecutive = 0
        check_date = date_type.fromisoformat(latest_date)
        while True:
            date = check_date.isoformat()
            if date not in date_codes or code not in date_codes[date]:
                break
            consecutive += 1
            check_date -= timedelta(days=1)
        points = work[work.stock_code == code][["date", "rank"]]
        latest_day = date_type.fromisoformat(latest_date)
        history_3d = [(latest_day - timedelta(days=offset)).isoformat() for offset in reversed(range(3))]
        history_5d = [(latest_day - timedelta(days=offset)).isoformat() for offset in reversed(range(5))]
        row.update({
            "previous_rank": int(prior) if prior is not None else pd.NA,
            "rank_change": int(prior - row["rank"]) if prior is not None else pd.NA,
            "new_entry": bool(previous_date and prior is None),
            "rank_history_3d": _rank_path(points, history_3d, set(dates)),
            "rank_history_5d": _rank_path(points, history_5d, set(dates)),
            "consecutive_days": consecutive,
            "best_rank": int(all_best[code]),
        })
        output.append(row)
    exits = []
    for index in range(1, len(dates)):
        date, prior_date = dates[index], dates[index - 1]
        if date_type.fromisoformat(date) - date_type.fromisoformat(prior_date) != timedelta(days=1):
            continue
        current_codes = date_codes[date]
        prior_rows = work[work.date == prior_date]
        for row in prior_rows[~prior_rows.stock_code.isin(current_codes)].to_dict("records"):
            code = row["stock_code"]
            consecutive = 0
            check_day = date_type.fromisoformat(prior_date)
            while True:
                check_date = check_day.isoformat()
                if check_date not in date_codes or code not in date_codes[check_date]:
                    break
                consecutive += 1
                check_day -= timedelta(days=1)
            exits.append({
                "date": date,
                "stock_code": code,
                "stock_name": row["stock_name"],
                "market": row["market"],
                "previous_rank": int(row["rank"]),
                "consecutive_days_before_exit": consecutive,
                "best_rank": int(work[(work.stock_code == code) & (work.date <= prior_date)]["rank"].min()),
                "exited_ranking": True,
            })
    return pd.DataFrame(output, columns=trend_columns), pd.DataFrame(exits, columns=exit_columns)


def derive_rising(popular_trends: pd.DataFrame) -> pd.DataFrame:
    """Build a reproducible rising list from changes in the official web ranking.

    Yahoo's official post-count rising list is app-only. This derived list is
    deliberately labelled and never represented as the official app ranking.
    """
    if popular_trends.empty:
        return pd.DataFrame(columns=RISING_COLUMNS)
    comparable = (
        popular_trends["previous_rank"].notna().any()
        or popular_trends["new_entry"].fillna(False).astype(bool).any()
    )
    if not comparable:
        return pd.DataFrame(columns=RISING_COLUMNS)
    baseline = int(popular_trends["rank"].max()) + 1
    rows = []
    for row in popular_trends.to_dict("records"):
        prior = row.get("previous_rank")
        if pd.notna(prior):
            score = int(prior) - int(row["rank"])
        elif bool(row.get("new_entry")):
            score = baseline - int(row["rank"])
        else:
            score = None
        rows.append({
            key: row.get(key) for key in HISTORY_COLUMNS if key != "rank"
        } | {
            "rank": 0,
            "surge_score": score,
            "derivation_method": RISING_METHOD,
        })
    frame = pd.DataFrame(rows)
    frame["_missing"] = frame["surge_score"].isna()
    frame = frame.sort_values(
        ["_missing", "surge_score", "stock_code"],
        ascending=[True, False, True],
        kind="stable",
    ).drop(columns="_missing").reset_index(drop=True)
    frame["rank"] = range(1, len(frame) + 1)
    return frame[RISING_COLUMNS]


def build_union(popular_trends: pd.DataFrame, rising_trends: pd.DataFrame) -> pd.DataFrame:
    records: dict[str, dict] = {}
    for prefix, frame in (("popular", popular_trends), ("rising", rising_trends)):
        for row in frame.to_dict("records"):
            code = str(row["stock_code"])
            item = records.setdefault(code, {
                "date": row["date"], "stock_code": code,
                "stock_name": row["stock_name"], "market": row["market"],
                "price": row["price"], "source_updated_at": row["source_updated_at"],
                "collected_at": row["collected_at"],
            })
            for field in TREND_FIELDS:
                value = row.get(field)
                item[f"{prefix}_{field}"] = value
    output = []
    for item in records.values():
        sources = []
        if pd.notna(item.get("popular_rank")):
            sources.append("popular")
        if pd.notna(item.get("rising_rank")):
            sources.append("derived_rising")
        item["ranking_sources"] = "+".join(sources)
        output.append(item)
    output.sort(key=lambda row: (
        int(row["popular_rank"]) if pd.notna(row.get("popular_rank")) else 10**9,
        int(row["rising_rank"]) if pd.notna(row.get("rising_rank")) else 10**9,
        row["stock_code"],
    ))
    for index, row in enumerate(output, 1):
        row["rank"] = index
    return pd.DataFrame(output, columns=UNIVERSE_COLUMNS)


def collect(target: Path, *, now: datetime | None = None, fetcher=fetch_snapshot) -> dict:
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    now = (now or datetime.now(JST)).astimezone(JST)
    status_path = target / "bbs_ranking_status.json"
    atomic_json(status_path, {"status":"running", "attempted_at":now.isoformat(), "source_url":SOURCE_URL})
    try:
        snapshot, meta = fetcher(now=now)
        if meta["ranking_date"] != now.date().isoformat():
            raise RuntimeError(f"Yahooランキングが当日更新ではありません: {meta['ranking_date']}")

        # The legacy daily directory remains the immutable source of the official
        # web post-count (popular) ranking.
        popular_daily = target / "bbs_ranking" / "daily" / f"{meta['ranking_date']}.csv"
        if not popular_daily.exists():
            atomic_csv(popular_daily, snapshot)
        popular_saved = pd.read_csv(popular_daily, dtype={"stock_code":str})
        saved_source_updated_at = str(popular_saved.source_updated_at.iloc[0])
        popular_files = sorted((target / "bbs_ranking" / "daily").glob("*.csv"))
        popular_history = pd.concat(
            [pd.read_csv(path, dtype={"stock_code":str}) for path in popular_files],
            ignore_index=True,
        )
        popular_history = popular_history.sort_values(["date", "rank"]).drop_duplicates(
            ["date", "stock_code"], keep="first"
        )
        popular_trends, popular_exits = build_trends(popular_history)

        # Yahoo documents the official rising ranking as app-only. Until an
        # authorized machine-readable source exists, record a separately named
        # derived ranking based only on changes in the official popular list.
        rising_today = derive_rising(popular_trends)
        rising_daily = target / "bbs_ranking" / "rising" / "daily" / f"{meta['ranking_date']}.csv"
        if not rising_daily.exists() and not rising_today.empty:
            atomic_csv(rising_daily, rising_today)
        rising_files = sorted((target / "bbs_ranking" / "rising" / "daily").glob("*.csv"))
        if rising_files:
            rising_history = pd.concat(
                [pd.read_csv(path, dtype={"stock_code":str}) for path in rising_files],
                ignore_index=True,
            )
            rising_history = rising_history.sort_values(["date", "rank"]).drop_duplicates(
                ["date", "stock_code"], keep="first"
            )
            rising_trends, rising_exits = build_trends(rising_history[HISTORY_COLUMNS])
            latest_extra = rising_history[
                rising_history.date.astype(str) == str(rising_trends.date.iloc[0])
            ][["date", "stock_code", "surge_score", "derivation_method"]] if not rising_trends.empty else pd.DataFrame()
            if not latest_extra.empty:
                rising_trends = rising_trends.merge(
                    latest_extra, on=["date", "stock_code"], how="left"
                )
        else:
            rising_history = pd.DataFrame(columns=RISING_COLUMNS)
            rising_trends, rising_exits = build_trends(
                pd.DataFrame(columns=HISTORY_COLUMNS)
            )

        universe = build_union(popular_trends, rising_trends)
        universe_daily = target / "bbs_ranking" / "universe" / "daily" / f"{meta['ranking_date']}.csv"
        if not universe_daily.exists():
            atomic_csv(universe_daily, universe)
        universe_history_path = target / "bbs_ranking_universe_history.csv"
        old_universe = (
            pd.read_csv(universe_history_path, dtype={"stock_code":str})
            if universe_history_path.exists()
            else pd.DataFrame(columns=UNIVERSE_COLUMNS)
        )
        universe_history = pd.concat([old_universe, universe], ignore_index=True)
        universe_history = universe_history.drop_duplicates(
            ["date", "stock_code"], keep="last"
        ).sort_values(["date", "rank"])

        # Backward-compatible names continue to mean the official popular list.
        for name, frame in {
            "bbs_ranking_history.csv": popular_history[HISTORY_COLUMNS],
            "bbs_ranking_latest.csv": popular_saved[HISTORY_COLUMNS],
            "bbs_ranking_trends.csv": popular_trends,
            "bbs_ranking_exits.csv": popular_exits,
            "bbs_ranking_popular_history.csv": popular_history[HISTORY_COLUMNS],
            "bbs_ranking_popular_latest.csv": popular_saved[HISTORY_COLUMNS],
            "bbs_ranking_popular_trends.csv": popular_trends,
            "bbs_ranking_popular_exits.csv": popular_exits,
            "bbs_ranking_rising_history.csv": rising_history[RISING_COLUMNS],
            "bbs_ranking_rising_latest.csv": rising_today[RISING_COLUMNS],
            "bbs_ranking_rising_trends.csv": rising_trends,
            "bbs_ranking_rising_exits.csv": rising_exits,
            "bbs_ranking_universe_latest.csv": universe,
            "bbs_ranking_universe_history.csv": universe_history[UNIVERSE_COLUMNS],
        }.items():
            atomic_csv(target / name, frame)

        result = {
            "status":"success",
            "ranking_date":meta["ranking_date"],
            "source_updated_at":saved_source_updated_at,
            "collected_at":now.isoformat(),
            "row_count":len(popular_saved),
            "universe_count":len(universe),
            "ranking_types":["popular", "derived_rising"],
            "popular":{
                "status":"success", "row_count":len(popular_saved),
                "total_pages":meta["total_pages"],
                "source_url":SOURCE_URL + "?market=all&term=daily",
            },
            "rising":{
                "status":"success" if not rising_today.empty else "insufficient_history",
                "row_count":len(rising_today),
                "source_kind":"derived",
                "derivation_method":RISING_METHOD,
                "official_app_ranking_collected":False,
                "note":"Yahoo official rising ranking is app-only; this is a derived rank, not the official app list.",
            },
            "total_pages":meta["total_pages"],
            "history_days":int(popular_history.date.nunique()),
            "source_url":SOURCE_URL + "?market=all&term=daily",
            "used_previous_day":False,
        }
        atomic_json(status_path, result)
        return result
    except Exception as exc:
        for name in [
            "bbs_ranking_latest.csv", "bbs_ranking_trends.csv",
            "bbs_ranking_popular_latest.csv", "bbs_ranking_popular_trends.csv",
            "bbs_ranking_rising_latest.csv", "bbs_ranking_rising_trends.csv",
            "bbs_ranking_universe_latest.csv",
        ]:
            (target / name).unlink(missing_ok=True)
        result = {
            "status":"failed", "attempted_at":now.isoformat(), "error":str(exc)[:500],
            "source_url":SOURCE_URL + "?market=all&term=daily",
            "used_previous_day":False,
        }
        atomic_json(status_path, result)
        raise

def main() -> int:
    parser = argparse.ArgumentParser(description="Yahoo掲示板投稿ランキングを日次保存")
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(collect(args.target), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status":"failed", "error":str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
