"""Publish the day's frozen pick or an actionable failure through a GitHub issue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date, datetime
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


def verified_picks(trade_date: str, picks_path: Path | None,
                   snapshot_path: Path | None) -> dict | None:
    if not picks_path or not picks_path.is_file() or not snapshot_path or not snapshot_path.is_file():
        return None
    raw = snapshot_path.read_bytes()
    snapshot, picks = json.loads(raw), json.loads(picks_path.read_bytes())
    if snapshot.get("trading_date") != trade_date or picks.get("trade_date") != trade_date:
        return None
    if snapshot.get("status") != "READY" or picks.get("snapshot_sha256") != hashlib.sha256(raw).hexdigest():
        return None
    try:
        cutoff = datetime.fromisoformat(snapshot["cutoff_jst"])
        issued = datetime.fromisoformat(picks["generated_at_jst"])
        completed = datetime.fromisoformat(snapshot["completed_at_jst"])
        if (cutoff.utcoffset() is None or issued.utcoffset() is None or completed.utcoffset() is None or
            issued > cutoff or completed > cutoff):
            return None
    except (KeyError, ValueError, TypeError):
        return None
    return picks


def publish(trade_date: str, picks_path: Path | None,
            snapshot_path: Path | None = None) -> str:
    if not xcals.get_calendar("XTKS").is_session(date.fromisoformat(trade_date)):
        return "SKIP_NON_SESSION"
    picks = verified_picks(trade_date, picks_path, snapshot_path)
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
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    print(publish(args.trade_date, args.picks, args.snapshot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
