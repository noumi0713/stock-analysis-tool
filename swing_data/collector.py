from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import time
import uuid
import zipfile
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
JST = ZoneInfo("Asia/Tokyo")
JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
WINDOW = 120
PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume",
                 "adj_close", "adj_open", "adj_high", "adj_low", "dividends", "stock_splits"]


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@contextmanager
def update_lock(root: Path):
    """OS lock is released even if a worker dies; no stale PID lock."""
    import fcntl
    root.mkdir(parents=True, exist_ok=True)
    with (root / "update.lock").open("a+") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("既に更新中です") from exc
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def session_window(now: datetime, calendar: str = "XTKS") -> list[str]:
    if now.tzinfo is None:
        raise ValueError("Timezone-aware timestamp required")
    stamp = pd.Timestamp(now).tz_convert("UTC")
    if calendar == "FX":
        # Never incorporate today's unfinished FX candle. Source daily boundary is UTC.
        end = stamp.normalize() - pd.Timedelta(days=1)
        return [x.date().isoformat() for x in pd.bdate_range(end=end.tz_localize(None), periods=WINDOW)]
    cal = xcals.get_calendar(calendar)
    dates = cal.sessions_in_range((stamp - pd.Timedelta(days=450)).date(), stamp.date())
    completed = [d for d in dates if cal.session_close(d) + pd.Timedelta(minutes=30) <= stamp]
    if len(completed) < WINDOW:
        raise ValueError("取引所カレンダーの営業日が120日未満です")
    return [d.date().isoformat() for d in completed[-WINDOW:]]


def load_universe(config: Path, output: Path, tickers: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    if tickers:
        codes = sorted(set(t.upper().removesuffix(".T") for t in tickers))
        if any(not __import__("re").fullmatch(r"[0-9A-Z]{4}", c) for c in codes):
            raise ValueError("銘柄コードは4桁（英字を含む場合あり）です")
        return pd.DataFrame({"ticker": [c + ".T" for c in codes], "stock_code": codes,
                             "company_name": codes, "sector17": "", "sector33": ""}), {"status":"selected_subset", "scope":"selected_subset"}
    cache = output / "universe_cache.csv"
    try:
        response = requests.get(JPX_URL, timeout=45)
        response.raise_for_status()
        data = pd.read_excel(BytesIO(response.content), dtype=str, engine="openpyxl")
        market = data["市場・商品区分"].fillna("")
        use = market.str.startswith(("プライム", "スタンダード", "グロース")) & market.str.contains("内国株式", regex=False)
        data = data.loc[use].copy()
        codes = data["コード"].str.upper().str.replace(r"\.0$", "", regex=True)
        excluded_classes = data.loc[codes.str.fullmatch(r"[0-9A-Z]{5}"), ["コード", "銘柄名"]].to_dict("records")
        # JPX also lists five-character preferred/share-class issues under domestic equities.
        # They are not ordinary stocks and do not have the same Yahoo ticker convention.
        ordinary = ~codes.str.fullmatch(r"[0-9A-Z]{5}")
        data, codes = data.loc[ordinary], codes.loc[ordinary]
        if not codes.str.fullmatch(r"[0-9A-Z]{4}").all() or codes.duplicated().any() or len(data) < 3000:
            raise ValueError("JPX銘柄一覧の形式・件数が不正です")
        frame = pd.DataFrame({"stock_code":codes, "ticker": codes + ".T", "company_name":data["銘柄名"],
                              "sector17":data["17業種区分"], "sector33":data["33業種区分"]}).sort_values("ticker")
        frame.to_csv(cache.with_suffix(".tmp"), index=False)
        os.replace(cache.with_suffix(".tmp"), cache)
        meta = {"status":"fetched", "scope":"tse_domestic_common", "source":JPX_URL,
                "source_date":str(data["日付"].iloc[0]), "downloaded_at":datetime.now(JST).isoformat(),
                "excluded_share_classes":excluded_classes}
        atomic_json(output / "universe_cache.json", meta)
        return frame, meta
    except Exception as exc:
        if not cache.exists():
            raise RuntimeError("JPX全銘柄一覧を取得できません。全銘柄としてテーマ所属だけを代用しません。") from exc
        meta = read_json(output / "universe_cache.json")
        meta.update(status="cached_unverified", error=str(exc)[:300])
        return pd.read_csv(cache, dtype=str).fillna(""), meta


def load_themes(config: Path, universe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    catalog = pd.read_csv(config / "theme_catalog.csv", dtype=str)
    if catalog.theme_name.nunique() != 124 or catalog.cluster.nunique() != 13:
        raise ValueError("既存の124テーマ・13分類と一致しません")
    provenance = read_json(config / "theme_provenance.json")
    for source in provenance["sources"]:
        if hashlib.sha256((config / source["file"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("テーマ定義のハッシュが不一致です。由来情報も更新してください")
    members = pd.read_csv(config / "theme_members.csv", dtype=str).fillna("")
    relevance = pd.read_csv(config / "theme_relevance.csv", dtype=str).fillna("")
    keys = ["stock_code", "theme_name"]
    if members.duplicated(keys).any() or relevance.duplicated(keys).any():
        raise ValueError("テーマ所属または関連度に重複があります")
    if not set(members.theme_name).issubset(set(catalog.theme_name)):
        raise ValueError("未定義のテーマがあります")
    members = members.merge(relevance[keys + ["final_relevance_score", "final_confidence", "decision_source", "decision_reason", "source_url"]]
                            .rename(columns={"source_url":"relevance_source_url"}), on=keys, how="left", validate="one_to_one")
    members = members[members.stock_code.isin(universe.stock_code)].copy()
    members["membership_weight"] = 1 / members.groupby("stock_code")["theme_name"].transform("count")
    provenance["classification_note"] = "現在の所属を使った参考集計。過去時点の所属や検証済み予測力を意味しない。関連度は既存値をそのまま掲載し、選別・重みに使わない。"
    return members, provenance


def extract_ticker(download: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if download is None or download.empty:
        return pd.DataFrame()
    if not isinstance(download.columns, pd.MultiIndex):
        return download.copy()
    for level in range(download.columns.nlevels):
        if ticker in download.columns.get_level_values(level):
            return download.xs(ticker, axis=1, level=level).copy()
    return pd.DataFrame()


def normalize(frame: pd.DataFrame, ticker: str, sessions: list[str], *, equity: bool = True) -> tuple[pd.DataFrame, dict]:
    report = {"ticker":ticker, "status":"fetch_failed", "rows":0, "expected_date":sessions[-1],
              "latest_date":None, "missing_sessions":len(sessions), "invalid_rows":0, "zero_volume_rows":0}
    if frame.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS), report
    work = frame.rename(columns={"Open":"open", "High":"high", "Low":"low", "Close":"close", "Volume":"volume",
                                  "Adj Close":"adj_close", "Dividends":"dividends", "Stock Splits":"stock_splits"}).copy()
    if any(c not in work for c in ["open", "high", "low", "close", "volume"]):
        report["status"] = "invalid_schema"
        return pd.DataFrame(columns=PRICE_COLUMNS), report
    work["date"] = [pd.Timestamp(x).date().isoformat() for x in work.index]
    work = work[work.date.isin(sessions)].copy()
    work = work.dropna(subset=["open", "high", "low", "close", "volume"], how="all")
    duplicates = work.date.duplicated(keep=False)
    numeric = ["open", "high", "low", "close", "volume"]
    for col in numeric + ["adj_close", "dividends", "stock_splits"]:
        if col not in work:
            work[col] = work["close"] if col == "adj_close" and not equity else (np.nan if col == "adj_close" else 0.0)
        work[col] = pd.to_numeric(work[col], errors="coerce")
    valid = np.isfinite(work[numeric + ["adj_close"]]).all(axis=1)
    valid &= (work[["open", "high", "low", "close", "adj_close"]] > 0).all(axis=1) & (work.volume >= 0)
    valid &= (work.high >= work[["open", "close", "low"]].max(axis=1)) & (work.low <= work[["open", "close", "high"]].min(axis=1))
    valid &= ~duplicates
    report["invalid_rows"] = int((~valid).sum())
    work = work.loc[valid].sort_values("date")
    report["zero_volume_rows"] = int((work.volume == 0).sum()) if equity else 0
    factor = work.adj_close / work.close
    for col in ["open", "high", "low"]:
        work["adj_" + col] = work[col] * factor
    work["ticker"] = ticker
    count = len(work)
    latest = work.date.iloc[-1] if count else None
    status = ("invalid_rows" if report["invalid_rows"] else "fetch_failed" if not count else
              "stale" if latest != sessions[-1] else "insufficient_history" if count < WINDOW else
              "zero_volume" if report["zero_volume_rows"] else "ready")
    report.update(status=status, rows=count, latest_date=latest, missing_sessions=WINDOW-count)
    return work[PRICE_COLUMNS].reset_index(drop=True), report


def fetch_group(tickers: list[str], sessions: list[str], *, equity: bool = True, downloader=None, sleeper=time.sleep):
    if downloader is None:
        import yfinance as yf
        downloader = yf.download
    results, reports = {}, {}
    pending = list(tickers)
    # Request all 120 sessions anew so corporate-action revisions cannot leave mixed vintages.
    for attempt in range(3):
        if not pending:
            break
        try:
            downloaded = downloader(pending, start=sessions[0], end=(pd.Timestamp(sessions[-1])+pd.Timedelta(days=1)).date().isoformat(),
                                    interval="1d", auto_adjust=False, actions=True, repair=False,
                                    keepna=True, progress=False, threads=4, timeout=25, group_by="ticker")
            for ticker in pending:
                frame, report = normalize(extract_ticker(downloaded, ticker), ticker, sessions, equity=equity)
                results[ticker], reports[ticker] = frame, report
        except Exception as exc:
            for ticker in pending:
                reports[ticker] = {"ticker":ticker, "status":"fetch_failed", "rows":0, "expected_date":sessions[-1],
                                   "latest_date":None, "missing_sessions":WINDOW, "error":str(exc)[:300]}
        pending = [t for t in tickers if reports[t]["status"] in {"fetch_failed", "stale", "invalid_schema", "invalid_rows"}]
        if pending and attempt < 2:
            sleeper(2 ** (attempt + 1))
    return results, reports


def theme_daily(prices: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "theme_name", "cluster", "return_pct", "member_count", "observed_count", "return_count", "coverage_pct", "allocated_turnover_proxy"]
    if prices.empty or members.empty:
        return pd.DataFrame(columns=columns)
    work = prices.sort_values(["ticker", "date"]).copy()
    work["return_pct"] = work.groupby("ticker").adj_close.pct_change(fill_method=None) * 100
    work["stock_code"] = work.ticker.str.removesuffix(".T")
    work["turnover_proxy"] = work.close * work.volume
    joined = work.merge(members[["stock_code", "theme_name", "cluster", "membership_weight"]], on="stock_code", how="inner", validate="many_to_many")
    joined["allocated_turnover_proxy"] = joined.turnover_proxy * joined.membership_weight
    grouped = joined.groupby(["date", "theme_name", "cluster"], as_index=False).agg(return_pct=("return_pct", "mean"),
                        observed_count=("ticker", "nunique"), return_count=("return_pct", "count"), allocated_turnover_proxy=("allocated_turnover_proxy", "sum"))
    counts = members.groupby("theme_name").stock_code.nunique().rename("member_count")
    grouped = grouped.merge(counts, on="theme_name")
    grouped["coverage_pct"] = grouped.observed_count / grouped.member_count * 100
    return grouped[columns]


def export_bundle(run_dir: Path, prices: pd.DataFrame, partial: pd.DataFrame, reports: list[dict],
                  universe: pd.DataFrame, members: pd.DataFrame, markets: pd.DataFrame, market_reports: list[dict], meta: dict) -> None:
    prices.to_csv(run_dir / "equities_120d.csv", index=False)
    prices.to_parquet(run_dir / "equities_120d.parquet", index=False)
    partial.to_csv(run_dir / "equities_incomplete.csv", index=False)
    pd.DataFrame(reports).to_csv(run_dir / "stock_status.csv", index=False)
    universe.to_csv(run_dir / "universe.csv", index=False)
    members.to_csv(run_dir / "theme_members.csv", index=False)
    theme_daily(prices, members).to_csv(run_dir / "themes_120d.csv", index=False)
    markets.to_csv(run_dir / "markets_120d.csv", index=False)
    atomic_json(run_dir / "market_status.json", {"items":market_reports})
    atomic_json(run_dir / "manifest.json", meta)
    (run_dir / "READ_ME.txt").write_text(
        "チャッピーにはこのZIPと毎日の分析プロンプトを渡してください。\n"
        "これは価格データの取得結果です。売買候補の選別・シグナル生成はしていません。\n"
        "equities_120d: 対象となる直近120営業日が揃い、品質確認を通過した銘柄のみ。\n"
        "stock_status: 全対象の結果。equities_incomplete: 欠損・古い・120日未満など分析対象外の参考データ。\n"
        "close/open/high/low: Yahooのauto_adjust=Falseの値（分割補正を含む場合あり）。adj_*: 配当・分割調整系列。\n"
        "異なる系列を混ぜて比較しない。volumeは提供元の値。売買代金はclose×volumeの概算で実際の売買代金ではない。\n"
        "テーマは現在の所属を過去120日に適用した参考集計。生存者・所属の先読みがあり、バックテスト用ではない。\n"
        "テーマリターンは取得できた適格銘柄の等ウェイト平均。売買代金概算のみ複数テーマ数で按分。\n"
        "関連度スコアは以前の評価で、最新の再調査や成功確率ではない。信頼度・根拠も併記。\n"
        "海外データは各市場の終了済みセッションまで。欠損系列はmarket_statusで必ず確認。\n"
        "市場データはfield_mode・unit・sourceを確認。close_onlyは日次値のみでOHLC/ATR計算不可。空欄を補完しない。\n"
        "ドル円は提供元の日次終値。OHLCの不整合と原データを別記録し、終値の検証と分離する。\n"
        "日本10年金利は財務省のコンスタントマチュリティー金利（%）、翌営業日9:30頃公表。新発債利回りとは別定義。\n"
        "利回りの差分はpercentage point、100倍でbp。価格リターンと混同しない。\n"
        "120日の入力から計算できる日次騰落率は119日分。初日は騰落率未計算。\n"
        "判断・検索の基準日時はmanifestのanalysis_as_ofを使用。ニュース・IR・財務の自動取得は本システムの対象外。\n",
        encoding="utf-8")
    files = sorted(p for p in run_dir.iterdir() if p.is_file())
    atomic_json(run_dir / "sha256.json", {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    with zipfile.ZipFile(run_dir / "chatgpt_120d.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(run_dir.iterdir()):
            if p.name != "chatgpt_120d.zip" and p.is_file():
                archive.write(p, p.name)


def collect(output: Path, *, config: Path = ROOT / "config", tickers: list[str] | None = None,
            now: datetime | None = None, batch_size: int = 40, downloader=None, market_fetcher=None) -> dict:
    if not 1 <= batch_size <= 100:
        raise ValueError("batch_size must be between 1 and 100")
    output = Path(output)
    with update_lock(output):
        now = now or datetime.now(JST)
        run_id = now.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        status = {"state":"running", "run_id":run_id, "started_at":now.isoformat(), "processed":0, "total":0}
        atomic_json(output / "status.json", status)
        run_dir = output / "runs" / run_id
        run_dir.mkdir(parents=True)
        try:
            sessions = session_window(now)
            universe, universe_meta = load_universe(config, output, tickers)
            members, provenance = load_themes(config, universe)
            status.update(total=len(universe), expected_date=sessions[-1], phase="個別株")
            ready, incomplete, reports = [], [], []
            targets = universe.ticker.tolist()
            empty_batches = 0
            for start in range(0, len(targets), batch_size):
                chunk = targets[start:start+batch_size]
                frames, checks = fetch_group(chunk, sessions, downloader=downloader)
                for ticker in chunk:
                    reports.append(checks[ticker])
                    frame = frames.get(ticker, pd.DataFrame(columns=PRICE_COLUMNS))
                    (ready if checks[ticker]["status"] == "ready" else incomplete).append(frame)
                empty_batches = empty_batches + 1 if all(checks[t]["status"] == "fetch_failed" for t in chunk) else 0
                if empty_batches >= 3:
                    for ticker in targets[start+batch_size:]:
                        reports.append({"ticker":ticker, "status":"not_fetched_provider_outage", "rows":0,
                                        "expected_date":sessions[-1], "latest_date":None, "missing_sessions":WINDOW})
                    status.update(provider_outage=True)
                status.update(processed=min(start+batch_size, len(targets)), ready=sum(r["status"] == "ready" for r in reports),
                              heartbeat_at=datetime.now(JST).isoformat())
                atomic_json(output / "status.json", status)
                print(json.dumps({k:status[k] for k in ["processed", "total", "ready"]}), flush=True)
                if empty_batches >= 3:
                    break
                if start + batch_size < len(targets):
                    time.sleep(1)
            status.update(phase="市場・テーマ集計")
            atomic_json(output / "status.json", status)
            market_frames, market_reports = [], []
            from swing_data.market_sources import fetch_market, mof_sessions
            market_fetcher = market_fetcher or fetch_market
            for item in read_json_list(config / "markets.json"):
                market_sessions = mof_sessions(now) if item["source"] == "mof" else session_window(now, item["calendar"])
                frame, report, raw = market_fetcher(item, market_sessions, downloader=downloader)
                if not raw.empty:
                    filename = "market_" + __import__("re").sub(r"[^A-Za-z0-9]", "_", item["ticker"]) + ".raw.csv"
                    raw.to_csv(run_dir / filename, index=True)
                    report["raw_file"] = filename
                    report["raw_sha256"] = hashlib.sha256((run_dir / filename).read_bytes()).hexdigest()
                market_reports.append(report)
                frame = frame.copy()
                frame["name"], frame["kind"] = item["name"], item["kind"]
                market_frames.append(frame)
                print(json.dumps({"market":item["name"], "status":report["status"], "rows":report["rows"]}, ensure_ascii=False), flush=True)
            prices = pd.concat(ready, ignore_index=True) if ready else pd.DataFrame(columns=PRICE_COLUMNS)
            partial = pd.concat(incomplete, ignore_index=True) if incomplete else pd.DataFrame(columns=PRICE_COLUMNS)
            markets = pd.concat(market_frames, ignore_index=True) if market_frames else pd.DataFrame(columns=PRICE_COLUMNS+["name", "kind"])
            counts = dict(Counter(r["status"] for r in reports))
            stock_quality = "PASS" if counts.get("ready", 0) == len(universe) and universe_meta["status"] == "fetched" else "PARTIAL" if not prices.empty else "FAIL"
            quality = "PASS" if stock_quality == "PASS" and all(x["status"] == "ready" for x in market_reports if x["required"]) else "PARTIAL" if not prices.empty else "FAIL"
            meta = {"schema_version":1, "run_id":run_id, "analysis_as_of":now.isoformat(), "completed_at":datetime.now(JST).isoformat(),
                    "expected_equity_date":sessions[-1], "window_start":sessions[0], "window_sessions":WINDOW,
                    "quality":quality, "stock_quality":stock_quality, "universe":universe_meta, "target_count":len(universe),
                    "ready_count":counts.get("ready", 0), "stock_status_counts":counts, "rows":len(prices),
                    "theme_count":124, "cluster_count":13, "theme_provenance":provenance,
                    "market_quality":"PASS" if all(x["status"] == "ready" for x in market_reports if x["required"]) else "FAIL",
                    "required_market_failures":[x["name"] for x in market_reports if x["required"] and x["status"] != "ready"],
                    "provider":"Equities: Yahoo Finance via yfinance; markets: source recorded per series", "selection":"none: all current TSE domestic common stocks",
                    "missing_markets":[x["name"] for x in market_reports if x["status"] != "ready"]}
            export_bundle(run_dir, prices, partial, reports, universe, members, markets, market_reports, meta)
            # An entirely failed attempt never replaces the last usable snapshot.
            if not prices.empty:
                atomic_json(output / "latest.json", meta)
                if quality == "PASS":
                    atomic_json(output / "last_complete.json", meta)
            status.update(state="complete" if quality == "PASS" else "partial" if not prices.empty else "failed",
                          completed_at=meta["completed_at"], quality=quality, ready=meta["ready_count"], error=None)
            atomic_json(output / "status.json", status)
            return meta
        except Exception as exc:
            status.update(state="failed", error=str(exc)[:500], completed_at=datetime.now(JST).isoformat())
            atomic_json(output / "status.json", status)
            raise


def read_json_list(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="全東証・直近120営業日データを取得（売買選別なし）")
    parser.add_argument("--output", type=Path, default=ROOT / "runtime")
    parser.add_argument("--tickers", nargs="+", help="動作確認用の一部銘柄。全銘柄取得とは区別されます")
    parser.add_argument("--batch-size", type=int, default=40)
    args = parser.parse_args()
    result = collect(args.output, tickers=args.tickers, batch_size=args.batch_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["quality"] == "FAIL" or result.get("required_market_failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
