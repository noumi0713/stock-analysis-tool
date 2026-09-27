"""Publish the day's frozen pick or an actionable failure through a GitHub issue."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import exchange_calendars as xcals


def api(method: str, path: str, payload: dict | None = None):
    base = f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}"
    raw = json.dumps(payload).encode() if payload is not None else None
    request = Request(base + path, data=raw, method=method, headers={
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json", "Content-Type": "application/json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def issue_text(trade_date: str, picks: dict | None) -> str:
    if picks is None:
        return (f"{trade_date} の08:15 JSTまでに保存済みの予想を確認できません。"
                "5分足DB、品質レポート、朝のスナップショットとActionsログを確認してください。"
                "未保存の予想を事後に当日分として扱わないでください。")
    if picks.get("trade_date") != trade_date:
        raise ValueError("Pick date mismatch")
    if picks.get("status") == "NO_TRADE":
        return (f"{trade_date}: 見送り。理由: {picks.get('reason', '選定条件を満たす銘柄なし')}\n\n"
                "これは朝に固定された予想で、発注ではありません。")
    if picks.get("status") != "PROVISIONAL_PICKS":
        raise ValueError("Unexpected forecast status")
    lines = [f"{trade_date}: 朝に固定した暫定予想。発注ではありません。", ""]
    for recommendation in picks["recommendations"]:
        lines.append(f"- {recommendation['ticker']} {recommendation['shares']}株: "
                     f"{recommendation['buy_time_jst']}買い、{recommendation['sell_time_jst']}売り (JST)")
    lines.append("\n引け後の結果はフォワード検証台帳で確認してください。")
    return "\n".join(lines)


def publish(trade_date: str, picks_path: Path | None) -> str:
    if not xcals.get_calendar("XTKS").is_session(date.fromisoformat(trade_date)):
        return "SKIP_NON_SESSION"
    picks = json.loads(picks_path.read_text(encoding="utf-8")) if picks_path and picks_path.is_file() else None
    title = f"Intraday forecast {trade_date}"
    # Search both open and closed issues so a rerun cannot create another notice.
    for issue in api("GET", "/issues?" + urlencode({"state": "all", "per_page": 100})):
        if issue["title"] == title:
            return "ALREADY_POSTED"
    body = issue_text(trade_date, picks)
    owner = os.environ["GITHUB_REPOSITORY"].split("/", 1)[0]
    api("POST", "/issues", {"title": title, "body": body, "assignees": [owner]})
    return "POSTED_READY" if picks is not None else "POSTED_MISSING"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--picks", type=Path)
    args = parser.parse_args()
    print(publish(args.trade_date, args.picks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
