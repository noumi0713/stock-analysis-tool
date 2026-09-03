from __future__ import annotations

import argparse
import csv
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

try:
    from score_theme_relevance_v3 import ALIASES
except Exception:
    ALIASES = {}

UA = "stock-analysis-theme-relevance-research/1.1"
IR_HINTS = (
    "ir", "investor", "投資家", "株主", "決算", "財務", "業績", "開示", "ir情報",
)
PDF_HINTS = (
    "決算説明", "決算短信", "有価証券報告", "統合報告", "annual report", "financial results",
    "earnings", "fact book", "factbook", "中期経営", "事業報告",
)
SALES_WORDS = (
    "売上高", "売上収益", "営業収益", "セグメント収益", "外部顧客への売上高", "収益",
    "revenue", "net sales", "sales",
)
SEGMENT_WORDS = ("セグメント", "事業", "segment", "business")
STRATEGY_WORDS = ("重点", "成長戦略", "集中投資", "中期経営", "拡大", "成長分野", "growth", "strategy", "investment")
MONEY_RE = re.compile(r"(?P<num>[0-9][0-9,]*(?:\.[0-9]+)?)\s*(?P<unit>兆円|億円|百万円|千円|円|trillion|billion|million)", re.I)
PCT_RE = re.compile(r"(?<![0-9])(?P<pct>[0-9]{1,3}(?:\.[0-9]+)?)\s*%")


def norm(s: str) -> str:
    s = (s or "").replace("\u3000", " ")
    return re.sub(r"\s+", " ", s).strip()


def same_host(a: str, b: str) -> bool:
    ha = urlparse(a).netloc.lower().removeprefix("www.")
    hb = urlparse(b).netloc.lower().removeprefix("www.")
    return ha == hb


def robots_allows(session: requests.Session, url: str) -> bool:
    p = urlparse(url)
    if not p.scheme or not p.netloc:
        return False
    robots = f"{p.scheme}://{p.netloc}/robots.txt"
    try:
        r = session.get(robots, timeout=5, headers={"User-Agent": UA})
        if r.status_code >= 400:
            return True
        rp = RobotFileParser()
        rp.set_url(robots)
        rp.parse(r.text.splitlines())
        return rp.can_fetch(UA, url)
    except Exception:
        return True


def fetch_html(session: requests.Session, url: str) -> tuple[str, BeautifulSoup, str] | None:
    if not robots_allows(session, url):
        return None
    try:
        r = session.get(url, timeout=10, headers={"User-Agent": UA}, allow_redirects=True)
        if r.status_code != 200:
            return None
        if "html" not in (r.headers.get("content-type") or "").lower():
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        return norm(" ".join(soup.stripped_strings)), soup, r.url
    except Exception:
        return None


def fetch_pdf_text(session: requests.Session, url: str, max_pages: int = 80) -> str:
    if not robots_allows(session, url):
        return ""
    try:
        r = session.get(url, timeout=20, headers={"User-Agent": UA}, allow_redirects=True)
        if r.status_code != 200 or len(r.content) > 25_000_000:
            return ""
        reader = PdfReader(io.BytesIO(r.content))
        chunks = []
        for page in reader.pages[:max_pages]:
            try:
                chunks.append(page.extract_text() or "")
            except Exception:
                continue
        return norm(" ".join(chunks))
    except Exception:
        return ""


def theme_terms(theme: str) -> list[str]:
    vals = list(ALIASES.get(theme, [])) + [theme]
    if theme.endswith("関連"):
        vals.append(theme[:-2])
    out = []
    for v in vals:
        v = norm(v)
        if len(v) >= 2 and v.lower() not in {x.lower() for x in out}:
            out.append(v)
    return out


def snippets(text: str, terms: list[str], radius: int = 650, limit: int = 10) -> list[str]:
    low = text.lower()
    found = []
    for term in terms:
        pos = 0
        tl = term.lower()
        while len(found) < limit:
            idx = low.find(tl, pos)
            if idx < 0:
                break
            s = max(0, idx - radius)
            e = min(len(text), idx + len(term) + radius)
            sn = norm(text[s:e])
            if sn not in found:
                found.append(sn)
            pos = idx + len(tl)
    return found


def score_ratio_candidate(snippet: str) -> tuple[float | None, str]:
    low = snippet.lower()
    if not any(w.lower() in low for w in SALES_WORDS):
        return None, ""
    pcts = []
    for m in PCT_RE.finditer(snippet):
        pct = float(m.group("pct"))
        if 0 <= pct <= 100:
            pcts.append((pct, m.start()))
    if not pcts:
        return None, ""
    # Prefer percentages near explicit composition/share wording.
    share_words = ("構成比", "売上比率", "売上構成", "占め", "割合", "share", "composition")
    best = None
    for pct, pos in pcts:
        local = low[max(0, pos - 120): pos + 120]
        rank = 2 if any(w in local for w in share_words) else 1
        cand = (rank, pct)
        if best is None or cand > best:
            best = cand
    return (best[1], "explicit_percent") if best else (None, "")


def money_to_million(num: float, unit: str) -> float:
    u = unit.lower()
    if unit == "兆円" or u == "trillion":
        return num * 1_000_000
    if unit == "億円":
        return num * 100
    if unit == "百万円" or u == "million":
        return num
    if unit == "千円":
        return num / 1000
    if unit == "円":
        return num / 1_000_000
    if u == "billion":
        return num * 1000
    return num


def calculated_ratio(snippet: str) -> tuple[float | None, str]:
    low = snippet.lower()
    if not any(w.lower() in low for w in SALES_WORDS):
        return None, ""
    amounts = []
    for m in MONEY_RE.finditer(snippet):
        try:
            val = money_to_million(float(m.group("num").replace(",", "")), m.group("unit"))
            if val > 0:
                amounts.append((val, m.start()))
        except Exception:
            pass
    if len(amounts) < 2:
        return None, ""
    # Conservative: only calculate if the same snippet contains segment and total/consolidated cues.
    if not any(w.lower() in low for w in SEGMENT_WORDS):
        return None, ""
    if not any(w in low for w in ("合計", "全社", "連結", "total", "consolidated")):
        return None, ""
    vals = sorted({round(v, 6) for v, _ in amounts})
    if len(vals) < 2:
        return None, ""
    total = max(vals)
    possible = [v for v in vals if v < total]
    if not possible:
        return None, ""
    # Use the closest smaller amount only as a candidate; mark calculated and lower confidence later.
    seg = max(possible)
    pct = seg / total * 100
    if 0 < pct < 100:
        return round(pct, 2), "calculated_same_snippet"
    return None, ""


def discover_documents(session: requests.Session, website: str) -> tuple[list[tuple[str, str]], list[str]]:
    html_docs: list[tuple[str, str]] = []
    pdfs: list[str] = []
    home = fetch_html(session, website)
    if not home:
        return html_docs, pdfs
    text, soup, final_url = home
    html_docs.append((final_url, text))
    ir_links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(final_url, a.get("href"))
        label = norm(" ".join(a.stripped_strings))
        key = (label + " " + href).lower()
        if same_host(final_url, href) and any(h.lower() in key for h in IR_HINTS):
            href = href.split("#", 1)[0]
            if href not in ir_links:
                ir_links.append(href)
        if href.lower().endswith(".pdf") and any(h.lower() in key for h in PDF_HINTS):
            if href not in pdfs:
                pdfs.append(href)
    for url in ir_links[:5]:
        time.sleep(0.10)
        page = fetch_html(session, url)
        if not page:
            continue
        ptext, psoup, purl = page
        html_docs.append((purl, ptext))
        for a in psoup.find_all("a", href=True):
            href = urljoin(purl, a.get("href"))
            label = norm(" ".join(a.stripped_strings))
            key = (label + " " + href).lower()
            if href.lower().endswith(".pdf") and any(h.lower() in key for h in PDF_HINTS):
                if href not in pdfs:
                    pdfs.append(href)
    return html_docs[:6], pdfs[:5]


def collect_one(stock: dict[str, str], themes: list[str]) -> list[dict[str, str]]:
    website = norm(stock.get("website", ""))
    code = stock.get("stock_code", "")
    quote_type = (stock.get("quote_type") or "").upper()
    if quote_type in {"ETF", "MUTUALFUND"}:
        return [{
            "stock_code": code, "theme_name": t, "revenue_share_pct": "", "share_status": "non_operating_security",
            "share_basis": "", "source_url": "", "evidence": "事業会社ではないため売上構成比の対象外", "document_count": "0",
        } for t in themes]
    if not website:
        return [{
            "stock_code": code, "theme_name": t, "revenue_share_pct": "", "share_status": "unknown",
            "share_basis": "", "source_url": "", "evidence": "会社公式サイトURL未取得", "document_count": "0",
        } for t in themes]
    if not website.startswith(("http://", "https://")):
        website = "https://" + website

    session = requests.Session()
    html_docs, pdf_urls = discover_documents(session, website)
    docs: list[tuple[str, str, str]] = [(u, txt, "html") for u, txt in html_docs]
    for u in pdf_urls:
        time.sleep(0.12)
        txt = fetch_pdf_text(session, u)
        if txt:
            docs.append((u, txt, "pdf"))

    rows = []
    for theme in themes:
        terms = theme_terms(theme)
        candidates = []
        for url, text, dtype in docs:
            for sn in snippets(text, terms):
                pct, basis = score_ratio_candidate(sn)
                if pct is None:
                    pct, basis = calculated_ratio(sn)
                if pct is not None:
                    quality = 3 if basis == "explicit_percent" else 2
                    if dtype == "pdf":
                        quality += 1
                    if any(w in sn for w in STRATEGY_WORDS):
                        quality += 0.1
                    candidates.append((quality, pct, basis, url, sn))
        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            quality, pct, basis, url, sn = candidates[0]
            rows.append({
                "stock_code": code,
                "theme_name": theme,
                "revenue_share_pct": f"{pct:.2f}",
                "share_status": "verified_candidate",
                "share_basis": basis,
                "source_url": url,
                "evidence": sn[:1200],
                "document_count": str(len(docs)),
            })
        else:
            # Preserve evidence that the theme exists in official documents, but do not invent a ratio.
            evid = ""
            src = ""
            for url, text, _ in docs:
                ss = snippets(text, terms, radius=400, limit=1)
                if ss:
                    src, evid = url, ss[0]
                    break
            rows.append({
                "stock_code": code,
                "theme_name": theme,
                "revenue_share_pct": "",
                "share_status": "unknown",
                "share_basis": "",
                "source_url": src,
                "evidence": (evid or "売上構成比を公式資料から数値で確認できず")[:1200],
                "document_count": str(len(docs)),
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dir", default="research/results/theme_relevance_batches")
    args = ap.parse_args()
    root = Path(args.dir)
    profiles_path = root / f"batch_{args.batch:03d}_profiles.csv"
    memberships_path = root / f"batch_{args.batch:03d}_memberships.csv"
    out_path = root / f"batch_{args.batch:03d}_revenue_mix_v4.csv"
    summary_path = root / f"batch_{args.batch:03d}_revenue_mix_v4_summary.json"

    with profiles_path.open(encoding="utf-8-sig") as f:
        profiles = {r["stock_code"]: r for r in csv.DictReader(f)}
    themes_by_stock: dict[str, list[str]] = {}
    with memberships_path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            themes_by_stock.setdefault(r["stock_code"], []).append(r["theme_name"])

    all_rows: list[dict[str, str]] = []
    codes = list(themes_by_stock)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(collect_one, profiles.get(c, {"stock_code": c}), themes_by_stock[c]): c for c in codes}
        for i, fut in enumerate(as_completed(futs), 1):
            code = futs[fut]
            try:
                rows = fut.result()
            except Exception as exc:
                rows = [{
                    "stock_code": code, "theme_name": t, "revenue_share_pct": "", "share_status": "error",
                    "share_basis": "", "source_url": "", "evidence": f"{type(exc).__name__}: {exc}"[:1200], "document_count": "0",
                } for t in themes_by_stock[code]]
            all_rows.extend(rows)
            print(i, code, CounterLike(rows))

    order = {(c, t): (i, j) for i, c in enumerate(codes) for j, t in enumerate(themes_by_stock[c])}
    all_rows.sort(key=lambda r: order.get((r["stock_code"], r["theme_name"]), (10**9, 10**9)))
    fields = ["stock_code", "theme_name", "revenue_share_pct", "share_status", "share_basis", "source_url", "evidence", "document_count"]
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)

    counts: dict[str, int] = {}
    for r in all_rows:
        counts[r["share_status"]] = counts.get(r["share_status"], 0) + 1
    summary = {
        "batch": args.batch,
        "stocks": len(codes),
        "stock_theme_pairs": len(all_rows),
        "status_counts": counts,
        "rule": "company website -> IR HTML/PDF -> explicit share or same-document calculation; otherwise unknown",
        "warning": "verified_candidate is still an extraction candidate and must pass arithmetic/source QA before scoring as final.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def CounterLike(rows: list[dict[str, str]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r["share_status"]] = out.get(r["share_status"], 0) + 1
    return out


if __name__ == "__main__":
    main()
