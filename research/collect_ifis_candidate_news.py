from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

UA = "Mozilla/5.0"

CATEGORY_PATTERNS = {
    "EARNINGS": [r"決算", r"業績", r"上方修正", r"下方修正", r"増益", r"減益", r"最高益", r"黒字", r"赤字"],
    "ORDER_ADOPTION": [r"受注", r"採用", r"契約", r"導入", r"落札"],
    "PARTNERSHIP_MA": [r"提携", r"協業", r"資本業務", r"買収", r"子会社化", r"M&A", r"TOB"],
    "PRODUCT_TECH": [r"新製品", r"新商品", r"開発", r"サービス開始", r"発売", r"実証"],
    "CAPITAL_RETURN": [r"自社株", r"自己株", r"増配", r"配当", r"株式分割", r"優待"],
    "POLICY_REGULATORY": [r"政府", r"補助金", r"規制", r"認可", r"承認", r"制度"],
}


def parse_gdelt_date(value: str) -> datetime | None:
    s = str(value or "").strip()
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def parse_snapshot(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def categories(title: str) -> list[str]:
    found = []
    for cat, patterns in CATEGORY_PATTERNS.items():
        if any(re.search(p, title or "", re.I) for p in patterns):
            found.append(cat)
    return found or ["OTHER"]


def fetch_news(company_name: str, stock_code: str, maxrecords: int = 20) -> list[dict]:
    query = f'"{company_name}"'
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": str(maxrecords),
        "format": "json",
        "timespan": "14d",
        "sort": "HybridRel",
    }
    url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.load(r)

    seen = set()
    out = []
    for a in data.get("articles", []):
        title = str(a.get("title") or "").strip()
        link = str(a.get("url") or "").strip()
        key = (title, link)
        if not title or key in seen:
            continue
        seen.add(key)
        out.append({
            "title": title,
            "url": link,
            "domain": a.get("domain"),
            "seen_date": a.get("seendate"),
            "categories": categories(title),
            "query_company": company_name,
            "stock_code": stock_code,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    watch = json.loads(Path(args.watch).read_text(encoding="utf-8"))
    results = []
    for i, c in enumerate(watch.get("primary_watch", [])):
        snapshot = parse_snapshot(c["snapshot_at"])
        try:
            articles = fetch_news(c["company_name"], c["stock_code"])
            error = ""
        except Exception as e:
            articles = []
            error = str(e)

        pre, post, undated = [], [], []
        for a in articles:
            dt = parse_gdelt_date(a.get("seen_date"))
            if dt is None:
                undated.append(a)
            elif dt <= snapshot.astimezone(timezone.utc):
                pre.append(a)
            else:
                post.append(a)

        results.append({
            "ifis_rank": c["ifis_rank"],
            "stock_code": c["stock_code"],
            "ticker": c["ticker"],
            "company_name": c["company_name"],
            "snapshot_at": c["snapshot_at"],
            "best_theme": c["best_theme"],
            "theme_relevance_score": c["theme_relevance_score"],
            "pre_discovery_articles": pre,
            "post_discovery_articles": post,
            "undated_articles": undated,
            "pre_discovery_count": len(pre),
            "post_discovery_count": len(post),
            "collector_error": error,
            "semantic_catalyst_status": "PENDING_CHATGPT_REVIEW",
        })
        if i + 1 < len(watch.get("primary_watch", [])):
            time.sleep(0.25)

    payload = {
        "status": "complete",
        "source": "GDELT DOC 2.0",
        "rule": "News is split at the IFIS snapshot timestamp. Automated keyword categories are metadata only; catalyst validity requires semantic review.",
        "candidate_count": len(results),
        "candidates": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "candidate_count": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
