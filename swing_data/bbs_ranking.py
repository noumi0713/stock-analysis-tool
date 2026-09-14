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
        daily = target / "bbs_ranking" / "daily" / f"{meta['ranking_date']}.csv"
        if not daily.exists():
            atomic_csv(daily, snapshot)
        saved = pd.read_csv(daily, dtype={"stock_code":str})
        saved_source_updated_at = str(saved.source_updated_at.iloc[0])
        files = sorted((target / "bbs_ranking" / "daily").glob("*.csv"))
        history = pd.concat([pd.read_csv(path, dtype={"stock_code":str}) for path in files], ignore_index=True)
        history = history.sort_values(["date", "rank"]).drop_duplicates(["date", "stock_code"], keep="first")
        trends, exits = build_trends(history)
        atomic_csv(target / "bbs_ranking_history.csv", history[HISTORY_COLUMNS])
        atomic_csv(target / "bbs_ranking_latest.csv", saved[HISTORY_COLUMNS])
        atomic_csv(target / "bbs_ranking_trends.csv", trends)
        atomic_csv(target / "bbs_ranking_exits.csv", exits)
        result = {
            "status":"success",
            "ranking_date":meta["ranking_date"],
            "source_updated_at":saved_source_updated_at,
            "collected_at":now.isoformat(),
            "row_count":len(saved),
            "total_pages":meta["total_pages"],
            "history_days":int(history.date.nunique()),
            "source_url":SOURCE_URL + "?market=all&term=daily",
            "used_previous_day":False,
        }
        atomic_json(status_path, result)
        return result
    except Exception as exc:
        for name in ["bbs_ranking_latest.csv", "bbs_ranking_trends.csv"]:
            (target / name).unlink(missing_ok=True)
        result = {"status":"failed", "attempted_at":now.isoformat(), "error":str(exc)[:500],
                  "source_url":SOURCE_URL + "?market=all&term=daily", "used_previous_day":False}
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
