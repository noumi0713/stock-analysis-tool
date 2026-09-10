"""Make the latest run readable by a browser/assistant. Does not select stocks."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import shutil

import pandas as pd

from swing_data.collector import atomic_json, read_json

PUBLIC = "https://raw.githubusercontent.com/noumi0713/stock-analysis-tool/swing-data-120d-latest"


def publish(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    status = read_json(source / "status.json")
    latest = read_json(source / "latest.json")
    if status:
        atomic_json(target / "update_status.json", status)
    # On a failed collection publish its failure status, preserving all prior usable data.
    if not latest or latest["run_id"] != status.get("run_id"):
        if not (target / "manifest.json").exists():
            atomic_json(target / "manifest.json", {"quality":"NOT_READY", "ready_count":0,
                "target_count":status.get("total", 0), "expected_equity_date":status.get("expected_date"),
                "missing_markets":[], "note":"初回取得は未完了です。update_status.jsonを確認してください。"})
            (target / "index.md").write_text("# 日本株120日データ\n\n初回取得は未完了です。\n\n"
                f"[更新結果]({PUBLIC}/update_status.json)\n", encoding="utf-8")
        return
    run = source / "runs" / latest["run_id"]
    # Files are made visible together by a single data-branch commit.
    for name in ["manifest.json", "stock_status.csv", "universe.csv", "theme_members.csv", "themes_120d.csv",
                 "market_status.json", "markets_120d.csv", "equities_120d.csv", "equities_120d.parquet",
                 "equities_incomplete.csv", "READ_ME.txt", "sha256.json", "chatgpt_120d.zip"]:
        shutil.copy2(run / name, target / name)
    for old in target.glob("market_*.raw.csv"):
        old.unlink()
    for raw in run.glob("market_*.raw.csv"):
        shutil.copy2(raw, target / raw.name)
    stocks = target / "stocks"
    if stocks.exists():
        shutil.rmtree(stocks)
    stocks.mkdir()
    prices = pd.read_csv(run / "equities_120d.csv")
    universe = pd.read_csv(run / "universe.csv", dtype=str).fillna("")
    members = pd.read_csv(run / "theme_members.csv", dtype=str).fillna("")
    names = universe.set_index("ticker").company_name.to_dict()
    themes = members.groupby("stock_code").theme_name.agg(list).to_dict()
    summary = []
    links = []
    for ticker, frame in prices.groupby("ticker", sort=True):
        code = ticker.removesuffix(".T")
        frame = frame.sort_values("date")
        close = frame.adj_close
        summary.append({"ticker":ticker, "code":code, "name":names.get(ticker,ticker), "date":frame.date.iloc[-1],
                        "close":float(frame.close.iloc[-1]), "rows":len(frame), "themes":themes.get(code,[]),
                        "return_5d_pct":float((close.iloc[-1]/close.iloc[-6]-1)*100),
                        "return_20d_pct":float((close.iloc[-1]/close.iloc[-21]-1)*100),
                        "ohlcv_url":f"{PUBLIC}/stocks/{code}.csv"})
        frame.to_csv(stocks / f"{code}.csv", index=False)
        links.append(f"- {code} {names.get(ticker,ticker)} — [120営業日OHLCV]({PUBLIC}/stocks/{code}.csv)")
    atomic_json(target / "stocks.json", {"run_id":latest["run_id"], "stocks":summary, "selection":"none"})
    text = ("# 日本株120営業日・チャッピー分析用\n\n"
            f"分析基準日時: {latest['analysis_as_of']}\n\n"
            f"株価対象日: {latest['expected_equity_date']} / 品質: {latest['quality']} / 必須市場系列: {latest.get('market_quality', '未評価')}\n\n"
            f"全対象 {latest['target_count']} 銘柄、120日適格 {latest['ready_count']} 銘柄。売買条件による選別なし。\n\n"
            "## 読む順番\n\n"
            f"1. [最新の更新結果]({PUBLIC}/update_status.json)と[データ定義・品質]({PUBLIC}/manifest.json)を確認する。\n"
            f"2. [全対象の取得結果]({PUBLIC}/stock_status.csv)と[市場系列の欠損]({PUBLIC}/market_status.json)を確認する。\n"
            f"3. [全銘柄の120日CSV]({PUBLIC}/equities_120d.csv)または[分析一式ZIP]({PUBLIC}/chatgpt_120d.zip)を取得し数値処理する。\n"
            f"4. [テーマ所属と関連度]({PUBLIC}/theme_members.csv)、[テーマ推移]({PUBLIC}/themes_120d.csv)、[市場データ]({PUBLIC}/markets_120d.csv)を参照する。\n"
            "5. 5〜10営業日のスイング候補を裁量で選び、最新IR・ニュースを別途検索する。固定パターンで絞り込まない。\n\n"
            "ファイルをユーザーに毎日添付してもらう必要はありません。このURLから取得してください。\n"
            "ページの一部分や一覧の要約だけを読んで、全銘柄の120日分を確認したと表現しないこと。\n"
            "取得・処理した銘柄数と未確認数を必ず明記すること。ブラウザーの省略表示に注意すること。\n"
            "ニュース、IR、理論株価、売買判断は未生成です。データ収集と分析済みを混同しないこと。\n\n"
            f"[価格調整・集計の注意]({PUBLIC}/READ_ME.txt) / [一覧JSON]({PUBLIC}/stocks.json)\n\n"
            "## 各銘柄の全120日\n\n" + "\n".join(links) + "\n")
    (target / "index.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    publish(args.source, args.target)
