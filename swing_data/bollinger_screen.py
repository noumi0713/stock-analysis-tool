"""Daily, after-close Bollinger -2 sigma watchlist from the existing equity bundle."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from swing_data.collector import JST, atomic_json, read_json, session_window

VERSION = "daily-bollinger-minus2-v1"
PUBLIC = "https://github.com/noumi0713/stock-analysis-tool/blob/swing-data-120d-latest"
WATCHLIST_FILES = ("bollinger_minus2.md", "bollinger_minus2_latest.csv",
                   "bollinger_minus2_latest.json", "bollinger_minus2_status.json")
COLUMNS = ["date", "stock_code", "company_name", "close", "low", "ma20",
           "lower_band", "close_sigma", "close_below_band", "new_touch",
           "turnover_proxy", "themes"]
RULES = {
    "period": 20,
    "sigma_multiplier": 2,
    "std_ddof": 0,
    "formula": "SMA20(adj_close) - 2 * population_std20(adj_close)",
    "match": "adj_low <= today's final lower band (including equality)",
    "close_below_band": "adj_close <= today's final lower band",
    "new_touch": "today matches and previous completed session did not match its own final band",
    "price_units": "MA and lower band converted to latest close units by close/adj_close",
    "timing": "引け後の日足判定。当日の終値を含む確定バンドと当日安値の比較。場中の動的バンドへの到達時刻は判定しない。",
    "scope": "東証プライム・スタンダード・グロース内国普通株式。ETF・ETN・REITを除外。掲示板順位・テーマ・RSIの追加絞り込みなし。",
    "note": "条件到達の監視一覧。買い推奨や反発確率ではありません。",
}


def evaluate(frame: pd.DataFrame, sessions: list[str]) -> dict:
    """Validate the same 120 completed sessions; never fill gaps or mix price scales."""
    required = ["open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close", "volume"]
    if not {"date", *required}.issubset(frame.columns):
        raise ValueError("invalid_schema")
    f = frame.sort_values("date").copy()
    if f.date.astype(str).tolist() != sessions:
        raise ValueError("incomplete_or_stale_sessions")
    f[required] = f[required].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(f[required].to_numpy()).all():
        raise ValueError("invalid_numeric")
    if (f[required[:-1]] <= 0).any().any() or (f.volume <= 0).any():
        raise ValueError("invalid_price_or_volume")
    for prefix in ("", "adj_"):
        if ((f[prefix + "high"] < f[[prefix + k for k in ("open", "close", "low")]].max(axis=1)) |
            (f[prefix + "low"] > f[[prefix + k for k in ("open", "close", "high")]].min(axis=1))).any():
            raise ValueError("invalid_ohlc")
    ma = f.adj_close.rolling(20).mean()
    std = f.adj_close.rolling(20).std(ddof=0)
    lower = ma - 2 * std
    # An unchanging price series has no measurable sigma; it is not a -2 sigma fall.
    touch = (std > 1e-12) & (f.adj_low <= lower)
    factor = float(f.close.iloc[-1] / f.adj_close.iloc[-1])
    return {
        "date": sessions[-1], "close": float(f.close.iloc[-1]), "low": float(f.low.iloc[-1]),
        "ma20": float(ma.iloc[-1] * factor), "lower_band": float(lower.iloc[-1] * factor),
        "close_sigma": float((f.adj_close.iloc[-1] - ma.iloc[-1]) / std.iloc[-1]) if std.iloc[-1] > 1e-12 else None,
        "matched": bool(touch.iloc[-1]),
        "close_below_band": bool(touch.iloc[-1] and f.adj_close.iloc[-1] <= lower.iloc[-1]),
        "new_touch": bool(touch.iloc[-1] and not touch.iloc[-2]),
        "turnover_proxy": float(f.close.iloc[-1] * f.volume.iloc[-1]),
    }


def markdown(status: dict, matches: list[dict]) -> str:
    lines = ["# 日足 −2σ 到達銘柄", "",
             f"状態: **{status['result']}** / 対象営業日: **{status['expected_date']}**", "",
             f"一覧生成完了: {status['completed_at']} / 株価取得完了: {status['collection_completed_at'] or '未確認'}", "",
             f"株価取得成功（品質適格） {status['acquired_count']} 銘柄 / 取得失敗・品質除外 {status['acquisition_failed_count']} 銘柄", "",
             f"全対象 {status['target_count']} 銘柄 / 判定済み {status['evaluated_count']} 銘柄 / "
             f"対象外・未取得 {status['excluded_count']} 銘柄 / 到達 **{len(matches)}** 銘柄", "",
             "条件: 調整後終値の20日単純平均 − 2×母標準偏差（ddof=0）に当日安値が到達。等値を含む。",
             "引け後の確定バンドを使用。場中の即時通知ではありません。価格は最新終値と同じ円の尺度に換算。",
             "新規到達＝前営業日は非該当。新規到達を先に、その中で当日の推計売買代金が大きい順。", "",
             f"[CSV]({PUBLIC}/bollinger_minus2_latest.csv) / [JSON]({PUBLIC}/bollinger_minus2_latest.json) / "
             f"[取得状態・除外理由]({PUBLIC}/bollinger_minus2_status.json)", ""]
    if status.get("reason"):
        lines += [f"**{status['reason']}**", ""]
    if status["status"] == "not_ready":
        lines += ["最新一覧は生成できませんでした。過去の一覧を当日分として代用していません。", ""]
        if status.get("last_good_date"):
            lines += [f"[前回の判定一覧（{status['last_good_date']}）]({PUBLIC}/bollinger_last_good/bollinger_minus2.md)", ""]
    elif not matches:
        lines += ["判定済み銘柄の中に該当銘柄はありません。", ""]
    else:
        lines += ["| コード | 銘柄 | 終値 | 安値 | −2σ | 終値も下限以下 | 新規到達 |",
                  "|---|---|---:|---:|---:|:---:|:---:|"]
        for item in matches:
            name = str(item["company_name"]).replace("|", "／").replace("\n", " ")
            code = item["stock_code"]
            lines.append(f"| [{code}](https://finance.yahoo.co.jp/quote/{code}.T/chart) | {name} | "
                         f"{item['close']:,.2f} | {item['low']:,.2f} | {item['lower_band']:,.2f} | "
                         f"{'○' if item['close_below_band'] else '—'} | {'○' if item['new_touch'] else '—'} |")
    lines += ["", RULES["note"], ""]
    return "\n".join(lines)


def build_watchlist(target: Path, *, now: datetime | None = None) -> dict:
    """Publish an explicitly dated list or a not_ready result, replacing stale matches."""
    target.mkdir(parents=True, exist_ok=True)
    now = now or datetime.now(JST)
    sessions = session_window(now)
    manifest = read_json(target / "manifest.json")
    update = read_json(target / "update_status.json")
    archive = target / "bollinger_last_good"
    previous = read_json(target / WATCHLIST_FILES[3])
    if previous.get("source_date") == previous.get("expected_date") and previous.get("status") in {"success", "partial"} and all(
        (target / name).is_file() for name in WATCHLIST_FILES
    ):
        archive.mkdir(exist_ok=True)
        for name in WATCHLIST_FILES:
            shutil.copyfile(target / name, archive / name)
    last_good = read_json(archive / WATCHLIST_FILES[3])
    attempt_is_current = update.get("expected_date") == sessions[-1]
    target_count = int(update.get("total") or manifest.get("target_count", 0))
    acquired = int(update.get("ready", 0)) if attempt_is_current else 0
    status = {"schema_version": 1, "rule_version": VERSION, "generated_at": now.isoformat(),
              "expected_date": sessions[-1], "source_date": manifest.get("expected_equity_date"),
              "source_run_id": manifest.get("run_id"), "status": "not_ready",
              "target_count": target_count, "evaluated_count": 0, "excluded_count": target_count,
              "acquired_count": acquired, "acquisition_failed_count": target_count - acquired, "matched_count": 0,
              "collection_completed_at": update.get("completed_at"),
              "collection_started_at": update.get("started_at"),
              "attempt_run_id": update.get("run_id"), "last_good_date": last_good.get("source_date"),
              "attempt_stock_status_counts": update.get("stock_status_counts", {}),
              "rules": RULES, "scope": manifest.get("universe", {}).get("scope", "unknown"),
              "source_stock_status_counts": manifest.get("stock_status_counts", {})}
    matches = []
    try:
        if update.get("state") == "failed" and update.get("error"):
            raise ValueError(f"株価更新失敗: {update['error']}")
        if manifest.get("expected_equity_date") != sessions[-1]:
            raise ValueError("株価が最新の終了済み営業日と一致しません。株価更新を確認してください。")
        if update.get("run_id") != manifest.get("run_id") or update.get("state") not in {"complete", "partial"}:
            raise ValueError("最新の収集が失敗・未完了です。過去データでは抽出しません。")
        universe = pd.read_csv(target / "universe.csv", dtype=str).fillna("")
        if universe.ticker.duplicated().any() or len(universe) != status["target_count"]:
            raise ValueError("銘柄一覧の件数または重複に異常があります。")
        names = universe.set_index("ticker").company_name.to_dict()
        prices = pd.read_csv(target / "equities_120d.csv", dtype={"date": str, "ticker": str})
        members = pd.read_csv(target / "theme_members.csv", dtype=str).fillna("")
        themes = members.groupby("stock_code").theme_name.agg(lambda x: " / ".join(sorted(set(x)))).to_dict()
        excluded = Counter()
        evaluated = 0
        for ticker, frame in prices.groupby("ticker", sort=True):
            if ticker not in names:
                raise ValueError("価格に銘柄一覧外のコードが含まれています。")
            try:
                item = evaluate(frame, sessions)
            except ValueError as exc:
                excluded[str(exc)] += 1
                continue
            evaluated += 1
            if item.pop("matched"):
                code = ticker.removesuffix(".T")
                item.update(stock_code=code, company_name=names[ticker], themes=themes.get(code, ""))
                matches.append(item)
        matches.sort(key=lambda x: (not x["new_touch"], -x["turnover_proxy"], x["stock_code"]))
        acquired = prices.ticker.nunique()
        missing = len(universe) - acquired
        if missing:
            excluded["not_in_eligible_120d_bundle"] = missing
        if not evaluated:
            raise ValueError("判定可能な最新120営業日データがありません。")
        status.update(status="success" if evaluated == len(universe) and status["scope"] == "tse_domestic_common"
                      and manifest.get("universe", {}).get("status") == "fetched" else "partial",
                      acquired_count=acquired, acquisition_failed_count=missing,
                      evaluated_count=evaluated, excluded_count=len(universe) - evaluated,
                      matched_count=len(matches), excluded_reasons=dict(excluded))
    except (ValueError, KeyError, OSError, pd.errors.ParserError) as exc:
        matches = []
        status["reason"] = str(exc)[:500]
    status["result"] = {"success": "complete", "partial": "partial", "not_ready": "failed"}[status["status"]]
    status["completed_at"] = datetime.now(JST).isoformat()
    atomic_json(target / "bollinger_minus2_status.json", status)
    atomic_json(target / "bollinger_minus2_latest.json", {**status, "matches": matches})
    pd.DataFrame(matches, columns=COLUMNS).to_csv(target / "bollinger_minus2_latest.csv", index=False, encoding="utf-8-sig")
    (target / "bollinger_minus2.md").write_text(markdown(status, matches), encoding="utf-8")
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="日足ボリンジャーバンド−2σ到達銘柄を一覧化")
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    result = build_watchlist(args.target)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(result["status"] == "not_ready")
