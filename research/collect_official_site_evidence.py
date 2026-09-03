from __future__ import annotations

import argparse
import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

UA = "stock-analysis-theme-relevance-research/1.0"
LINK_HINTS = (
    "事業", "サービス", "製品", "商品", "会社概要", "企業情報",
    "business", "service", "product", "company", "about",
)


def clean_text(html: str) -> tuple[str, BeautifulSoup]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    text = " ".join(soup.stripped_strings)
    text = re.sub(r"\s+", " ", text)
    return text, soup


def same_host(a: str, b: str) -> bool:
    ha = urlparse(a).netloc.lower().removeprefix("www.")
    hb = urlparse(b).netloc.lower().removeprefix("www.")
    return ha == hb


def load_robots(session: requests.Session, website: str) -> tuple[RobotFileParser | None, bool]:
    p = urlparse(website)
    robots_url = f"{p.scheme}://{p.netloc}/robots.txt"
    try:
        r = session.get(robots_url, timeout=5, headers={"User-Agent": UA})
        if r.status_code >= 400:
            return None, False
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.parse(r.text.splitlines())
        return rp, True
    except Exception:
        return None, False


def fetch_page(session: requests.Session, url: str, rp: RobotFileParser | None, robots_known: bool, is_home: bool = False) -> tuple[str, BeautifulSoup] | None:
    if robots_known and rp is not None and not rp.can_fetch(UA, url):
        return None
    # If robots could not be read, keep the fallback conservative: homepage only.
    if not robots_known and not is_home:
        return None
    try:
        r = session.get(url, timeout=8, headers={"User-Agent": UA}, allow_redirects=True)
        if r.status_code != 200:
            return None
        ctype = (r.headers.get("content-type") or "").lower()
        if "html" not in ctype:
            return None
        return clean_text(r.text)
    except Exception:
        return None


def collect_one(row: dict[str, str]) -> dict[str, str]:
    website = (row.get("website") or "").strip()
    base = dict(row)
    if not website:
        return {**base, "official_status": "no_website", "official_urls": "", "official_text": ""}
    if not website.startswith(("http://", "https://")):
        website = "https://" + website

    session = requests.Session()
    rp, robots_known = load_robots(session, website)
    home = fetch_page(session, website, rp, robots_known, is_home=True)
    if not home:
        return {**base, "official_status": "home_unavailable", "official_urls": website, "official_text": ""}
    home_text, soup = home
    texts = [home_text]
    urls = [website]

    links: list[str] = []
    if robots_known:
        for a in soup.find_all("a", href=True):
            label = " ".join(a.stripped_strings).lower()
            href = urljoin(website, a.get("href"))
            if not href.startswith(("http://", "https://")) or not same_host(website, href):
                continue
            if any(h.lower() in label for h in LINK_HINTS):
                href = href.split("#", 1)[0]
                if href not in links and href != website:
                    links.append(href)
            if len(links) >= 8:
                break

    for url in links[:2]:
        time.sleep(0.15)
        page = fetch_page(session, url, rp, robots_known)
        if page:
            text, _ = page
            texts.append(text)
            urls.append(url)

    combined = re.sub(r"\s+", " ", " ".join(texts))[:30000]
    return {
        **base,
        "official_status": "ok",
        "official_urls": " | ".join(urls),
        "official_text": combined,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dir", default="research/results/theme_relevance_batches")
    args = ap.parse_args()
    root = Path(args.dir)
    src = root / f"batch_{args.batch:03d}_profiles.csv"
    dst = root / f"batch_{args.batch:03d}_official_evidence.csv"
    with src.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(collect_one, row): row for row in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            print(i, r.get("stock_code"), r.get("official_status"), len(r.get("official_text", "")))

    order = {r["stock_code"]: i for i, r in enumerate(rows)}
    results.sort(key=lambda r: order.get(r.get("stock_code", ""), 10**9))
    fields = list(results[0].keys()) if results else []
    with dst.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(dst)


if __name__ == "__main__":
    main()
