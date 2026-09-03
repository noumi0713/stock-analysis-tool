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

UA = "stock-analysis-theme-relevance-research/1.3"
IR_HINTS = ("ir", "investor", "投資家", "株主", "決算", "財務", "業績", "開示", "ir情報")
PDF_HINTS = (
    "決算説明", "決算短信", "有価証券報告", "統合報告", "annual report", "financial results",
    "earnings", "fact book", "factbook", "中期経営", "事業報告",
)
PERCENT_CONTEXT_MARKERS = (
    "セグメント別売上高構成比", "売上高構成比", "売上構成比", "売上高割合", "売上構成割合",
    "事業別連結売上高", "事業別売上高", "revenue mix", "sales mix", "revenue composition",
)
AMOUNT_MARKERS = ("外部顧客への売上高", "外部顧客に対する売上高", "revenue from external customers")
BAD_LABEL_WORDS = (
    "前年", "前年差", "前期", "増減", "利益率", "営業利益率", "自己資本", "roe", "roic", "pbr", "wacc",
    "進捗", "取得", "来場", "顧客別", "地域別", "金利", "ltv", "受注", "受注高", "構成比", "比率",
    "官公庁", "民間", "その他", "首都圏", "北海道", "東北", "関東", "中部", "関西", "中国", "四国", "九州", "沖縄",
)
GENERIC_LABEL_CUES = (
    "事業", "部門", "工事", "設備", "サービス", "食品", "介護", "婚礼", "ホテル", "不動産", "リース",
    "土木", "建築", "発電", "電力", "金融", "証券", "保険", "銀行", "物流", "倉庫", "鉄道", "航空",
    "システム", "クラウド", "ai", "dx", "it", "半導体", "医薬", "創薬", "バイオ", "自動車", "機械",
)
NUM_RE = re.compile(r"(?<![\d.])([0-9][0-9,]*(?:\.[0-9]+)?)(?![\d.])")
PCT_RE = re.compile(r"(?<![0-9])([0-9]{1,3}(?:\.[0-9]+)?)\s*%")
PAIR_RE = re.compile(
    r"([一-龥ぁ-んァ-ンA-Za-z0-9][一-龥ぁ-んァ-ンA-Za-z0-9・＆&／/+＋ー\-\s]{1,34}?)\s*[（(]?\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%",
    re.I,
)


def norm(s: str) -> str:
    s = (s or "").replace("\u3000", " ").replace("％", "%")
    return re.sub(r"[ \t]+", " ", s).strip()


def same_host(a: str, b: str) -> bool:
    ha = urlparse(a).netloc.lower().removeprefix("www.")
    hb = urlparse(b).netloc.lower().removeprefix("www.")
    return ha == hb


def robots_allows(session: requests.Session, url: str) -> bool:
    p = urlparse(url)
    if not p.scheme or not p.netloc:
        return False
    try:
        r = session.get(f"{p.scheme}://{p.netloc}/robots.txt", timeout=5, headers={"User-Agent": UA})
        if r.status_code >= 400:
            return True
        rp = RobotFileParser()
        rp.parse(r.text.splitlines())
        return rp.can_fetch(UA, url)
    except Exception:
        return True


def fetch_html(session: requests.Session, url: str) -> tuple[str, BeautifulSoup, str] | None:
    if not robots_allows(session, url):
        return None
    try:
        r = session.get(url, timeout=10, headers={"User-Agent": UA}, allow_redirects=True)
        if r.status_code != 200 or "html" not in (r.headers.get("content-type") or "").lower():
            return None
        r.encoding = r.apparent_encoding or r.encoding
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        text = "\n".join(norm(x) for x in soup.stripped_strings if norm(x))
        return text, soup, r.url
    except Exception:
        return None


def fetch_pdf_text(session: requests.Session, url: str, max_pages: int = 90) -> str:
    if not robots_allows(session, url):
        return ""
    try:
        r = session.get(url, timeout=25, headers={"User-Agent": UA}, allow_redirects=True)
        if r.status_code != 200 or len(r.content) > 30_000_000:
            return ""
        reader = PdfReader(io.BytesIO(r.content))
        pages = []
        for page in reader.pages[:max_pages]:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pass
        return "\n".join(pages)
    except Exception:
        return ""


def discover_documents(session: requests.Session, website: str) -> list[tuple[str, str, str]]:
    docs: list[tuple[str, str, str]] = []
    home = fetch_html(session, website)
    if not home:
        return docs
    text, soup, final_url = home
    docs.append((final_url, text, "html"))
    ir_links: list[str] = []
    pdfs: list[str] = []
    for a in soup.find_all("a", href=True):
        href = urljoin(final_url, a.get("href"))
        label = norm(" ".join(a.stripped_strings))
        key = (label + " " + href).lower()
        if same_host(final_url, href) and any(h in key for h in IR_HINTS):
            href = href.split("#", 1)[0]
            if href not in ir_links:
                ir_links.append(href)
        if href.lower().endswith(".pdf") and any(h.lower() in key for h in PDF_HINTS) and href not in pdfs:
            pdfs.append(href)
    for u in ir_links[:6]:
        time.sleep(0.08)
        page = fetch_html(session, u)
        if not page:
            continue
        ptext, psoup, purl = page
        docs.append((purl, ptext, "html"))
        for a in psoup.find_all("a", href=True):
            href = urljoin(purl, a.get("href"))
            label = norm(" ".join(a.stripped_strings))
            key = (label + " " + href).lower()
            if href.lower().endswith(".pdf") and any(h.lower() in key for h in PDF_HINTS) and href not in pdfs:
                pdfs.append(href)
    for u in pdfs[:6]:
        time.sleep(0.10)
        txt = fetch_pdf_text(session, u)
        if txt:
            docs.append((u, txt, "pdf"))
    return docs[:13]


def clean_label(raw: str) -> str:
    s = norm(raw).strip("()（）[]【】:-：・ ")
    s = re.sub(r"^(?:売上高|売上収益|連結売上高|事業別|セグメント別|構成比|割合)\s*", "", s)
    # Keep only tail after strong separators; tables often flatten preceding headings into the same match.
    for sep in ("。", "：", ":", "\n", "│", "|", "■", "●", "▶", "→"):
        if sep in s:
            s = s.split(sep)[-1].strip()
    # If too long, keep a tail beginning at the last plausible segment cue.
    if len(s) > 28:
        parts = re.split(r"\s{2,}|　", s)
        if parts:
            s = parts[-1].strip()
        if len(s) > 28:
            s = s[-28:].strip()
    s = re.sub(r"^[-–—・,，.]+", "", s).strip()
    return s


def plausible_segment_label(label: str) -> bool:
    low = label.lower()
    if not 2 <= len(label) <= 30:
        return False
    if any(w.lower() in low for w in BAD_LABEL_WORDS):
        return False
    if re.fullmatch(r"[0-9.,% ]+", label):
        return False
    return any(c.lower() in low for c in GENERIC_LABEL_CUES)


def context_windows(text: str, markers: tuple[str, ...], radius: int = 900) -> list[str]:
    low = text.lower()
    out: list[str] = []
    for marker in markers:
        mlow = marker.lower()
        pos = 0
        while True:
            idx = low.find(mlow, pos)
            if idx < 0:
                break
            win = text[max(0, idx - radius): min(len(text), idx + len(marker) + radius)]
            if win not in out:
                out.append(win)
            pos = idx + len(marker)
    return out[:12]


def parse_explicit_percent_segments(text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for win in context_windows(text, PERCENT_CONTEXT_MARKERS):
        compact = re.sub(r"\s+", " ", win.replace("％", "%"))
        for m in PAIR_RE.finditer(compact):
            label = clean_label(m.group(1))
            pct = float(m.group(2))
            if not 0 <= pct <= 100 or not plausible_segment_label(label):
                continue
            rows.append({"segment_name": label, "share_pct": pct, "basis": "explicit_segment_percent", "evidence": compact[:1800]})
    # Also allow very clear direct label/value sequences even when the heading was lost in PDF extraction.
    direct_pattern = re.compile(
        r"((?:[一-龥ぁ-んァ-ンA-Za-z0-9・＆&／/+＋ー\-]{2,22})(?:事業|部門|工事|設備|発電|食品|介護|婚礼|不動産|リース))\s*[（(]?\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%",
        re.I,
    )
    for m in direct_pattern.finditer(text.replace("％", "%")):
        label = clean_label(m.group(1)); pct = float(m.group(2))
        local = text[max(0, m.start()-300): min(len(text), m.end()+300)]
        if any(x in local for x in ("売上", "セグメント", "事業概要", "事業別")) and plausible_segment_label(label):
            rows.append({"segment_name": label, "share_pct": pct, "basis": "explicit_segment_percent", "evidence": norm(local)[:1800]})
    return dedupe_segments(rows)


def parse_external_customer_amount_segments(text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for marker in AMOUNT_MARKERS:
        pos = 0
        while True:
            idx = text.lower().find(marker.lower(), pos)
            if idx < 0:
                break
            before = norm(text[max(0, idx-420):idx])
            after = norm(text[idx+len(marker): min(len(text), idx+len(marker)+360)])
            labels = re.findall(r"([一-龥ぁ-んァ-ンA-Za-z0-9・＆&／/+＋ー\-]{1,28}事業)", before)
            # preserve order and drop headings
            clean_labels: list[str] = []
            for lab in labels:
                lab = clean_label(lab)
                if lab in {"報告セグメント", "連結事業"} or lab in clean_labels or not plausible_segment_label(lab):
                    continue
                clean_labels.append(lab)
            clean_labels = clean_labels[-8:]
            values = []
            for raw in NUM_RE.findall(after):
                try:
                    values.append(float(raw.replace(",", "")))
                except Exception:
                    pass
            n = len(clean_labels)
            if n >= 2 and len(values) >= n:
                seg_vals = values[:n]
                total = values[n] if len(values) > n and values[n] >= max(seg_vals) else sum(seg_vals)
                if total > 0 and 0.97 <= sum(seg_vals)/total <= 1.03:
                    for lab, val in zip(clean_labels, seg_vals):
                        rows.append({
                            "segment_name": lab,
                            "share_pct": round(val/total*100, 2),
                            "basis": "external_customer_sales_table",
                            "segment_sales": val,
                            "total_sales": total,
                            "evidence": norm(before + " " + marker + " " + after)[:1800],
                        })
            pos = idx + len(marker)
    return dedupe_segments(rows)


def dedupe_segments(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    rank = {"external_customer_sales_table": 3, "explicit_segment_percent": 2}
    for r in rows:
        name = str(r["segment_name"]).strip()
        key = re.sub(r"\s+", "", name).lower()
        old = best.get(key)
        if old is None or rank.get(str(r.get("basis")), 0) > rank.get(str(old.get("basis")), 0):
            best[key] = r
    return list(best.values())


def normalize_match(s: str) -> str:
    return re.sub(r"[\s・＆&／/+＋ー\-()（）]", "", (s or "")).lower()


def theme_terms(theme: str) -> list[str]:
    vals = [theme] + list(ALIASES.get(theme, []))
    if theme.endswith("関連"):
        vals.append(theme[:-2])
    out: list[str] = []
    for v in vals:
        n = normalize_match(v)
        if len(n) >= 2 and n not in out:
            out.append(n)
    return out


BROAD_SEGMENT_RULES: dict[str, list[str]] = {
    "建設": ["建設", "土木", "建築"],
    "食品": ["食品", "食材", "加工食品"],
    "住宅関連": ["住宅"],
    "不動産関連": ["不動産"],
    "介護関連": ["介護"],
    "医薬品関連": ["医薬", "製薬"],
    "IT関連": ["it", "情報", "システム", "ソフトウェア"],
    "リース": ["リース"],
    "外食": ["外食", "レストラン", "飲食"],
    "小売り": ["小売"],
    "物流": ["物流"],
    "倉庫": ["倉庫"],
    "ホテル": ["ホテル"],
    "海運": ["海運"],
    "航空": ["航空"],
}


def segment_matches_theme(segment_name: str, theme: str) -> bool:
    seg = normalize_match(segment_name)
    if theme in BROAD_SEGMENT_RULES:
        return any(normalize_match(x) in seg for x in BROAD_SEGMENT_RULES[theme])
    return any(t in seg for t in theme_terms(theme))


def map_theme_share(theme: str, segments: list[dict[str, object]]) -> tuple[float | None, list[dict[str, object]]]:
    matched = [s for s in segments if segment_matches_theme(str(s["segment_name"]), theme)]
    if not matched:
        return None, []
    # Avoid double counting duplicate labels from multiple docs; prefer newest/first extraction per normalized name.
    uniq: dict[str, dict[str, object]] = {}
    for s in matched:
        key = normalize_match(str(s["segment_name"]))
        if key not in uniq:
            uniq[key] = s
    vals = list(uniq.values())
    total = sum(float(s["share_pct"]) for s in vals)
    if total > 100.5:
        return None, vals
    return round(min(100.0, total), 2), vals


def collect_one(profile: dict[str, str], themes: list[str]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    code = profile.get("stock_code", "")
    quote_type = (profile.get("quote_type") or "").upper()
    website = norm(profile.get("website", ""))
    if quote_type in {"ETF", "MUTUALFUND"}:
        theme_rows = [{"stock_code": code, "theme_name": t, "revenue_share_pct": "", "share_status": "non_operating_security", "share_basis": "", "segment_names": "", "source_url": "", "evidence": "事業会社ではないため対象外"} for t in themes]
        return theme_rows, []
    if not website:
        theme_rows = [{"stock_code": code, "theme_name": t, "revenue_share_pct": "", "share_status": "unknown", "share_basis": "", "segment_names": "", "source_url": "", "evidence": "会社公式サイトURL未取得"} for t in themes]
        return theme_rows, []
    if not website.startswith(("http://", "https://")):
        website = "https://" + website

    session = requests.Session()
    docs = discover_documents(session, website)
    segment_rows: list[dict[str, object]] = []
    for url, text, dtype in docs:
        parsed = parse_external_customer_amount_segments(text) + parse_explicit_percent_segments(text)
        for p in parsed:
            p = dict(p)
            p.update({"stock_code": code, "source_url": url, "document_type": dtype})
            segment_rows.append(p)
    segment_rows = dedupe_segments(segment_rows)

    theme_rows: list[dict[str, str]] = []
    for theme in themes:
        share, matched = map_theme_share(theme, segment_rows)
        if share is None:
            theme_rows.append({
                "stock_code": code, "theme_name": theme, "revenue_share_pct": "", "share_status": "unknown",
                "share_basis": "", "segment_names": "", "source_url": "", "evidence": "売上セグメントとテーマを直接対応できず",
            })
            continue
        basis = "+".join(sorted({str(s.get("basis", "")) for s in matched}))
        sources = [str(s.get("source_url", "")) for s in matched if s.get("source_url")]
        evidence = " || ".join(str(s.get("evidence", ""))[:500] for s in matched)[:1600]
        theme_rows.append({
            "stock_code": code, "theme_name": theme, "revenue_share_pct": f"{share:.2f}",
            "share_status": "structured_segment_candidate", "share_basis": basis,
            "segment_names": " | ".join(str(s["segment_name"]) for s in matched),
            "source_url": sources[0] if sources else "", "evidence": evidence,
        })
    serial_segments = []
    for s in segment_rows:
        serial_segments.append({
            "stock_code": code,
            "segment_name": str(s.get("segment_name", "")),
            "share_pct": f"{float(s.get('share_pct', 0)):.2f}",
            "basis": str(s.get("basis", "")),
            "segment_sales": str(s.get("segment_sales", "")),
            "total_sales": str(s.get("total_sales", "")),
            "source_url": str(s.get("source_url", "")),
            "document_type": str(s.get("document_type", "")),
            "evidence": str(s.get("evidence", ""))[:1600],
        })
    return theme_rows, serial_segments


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--dir", default="research/results/theme_relevance_batches")
    args = ap.parse_args()
    root = Path(args.dir)
    profiles_path = root / f"batch_{args.batch:03d}_profiles.csv"
    memberships_path = root / f"batch_{args.batch:03d}_memberships.csv"
    out_themes = root / f"batch_{args.batch:03d}_revenue_mix_v5.csv"
    out_segments = root / f"batch_{args.batch:03d}_segments_v5.csv"
    out_summary = root / f"batch_{args.batch:03d}_revenue_mix_v5_summary.json"

    with profiles_path.open(encoding="utf-8-sig") as f:
        profiles = {r["stock_code"]: r for r in csv.DictReader(f)}
    themes_by_stock: dict[str, list[str]] = {}
    with memberships_path.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            themes_by_stock.setdefault(r["stock_code"], []).append(r["theme_name"])

    theme_rows: list[dict[str, str]] = []
    segment_rows: list[dict[str, str]] = []
    codes = list(themes_by_stock)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(collect_one, profiles.get(code, {"stock_code": code}), themes_by_stock[code]): code for code in codes}
        for i, fut in enumerate(as_completed(futures), 1):
            code = futures[fut]
            try:
                tr, sr = fut.result()
            except Exception as exc:
                tr = [{"stock_code": code, "theme_name": t, "revenue_share_pct": "", "share_status": "error", "share_basis": "", "segment_names": "", "source_url": "", "evidence": f"{type(exc).__name__}: {exc}"} for t in themes_by_stock[code]]
                sr = []
            theme_rows.extend(tr); segment_rows.extend(sr)
            print(i, code, "segments", len(sr), "mapped", sum(1 for r in tr if r["revenue_share_pct"]))

    order = {(c, t): (i, j) for i, c in enumerate(codes) for j, t in enumerate(themes_by_stock[c])}
    theme_rows.sort(key=lambda r: order.get((r["stock_code"], r["theme_name"]), (10**9, 10**9)))
    segment_rows.sort(key=lambda r: (codes.index(r["stock_code"]) if r["stock_code"] in codes else 10**9, r["segment_name"]))

    tf = ["stock_code", "theme_name", "revenue_share_pct", "share_status", "share_basis", "segment_names", "source_url", "evidence"]
    sf = ["stock_code", "segment_name", "share_pct", "basis", "segment_sales", "total_sales", "source_url", "document_type", "evidence"]
    with out_themes.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=tf); w.writeheader(); w.writerows(theme_rows)
    with out_segments.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=sf); w.writeheader(); w.writerows(segment_rows)

    mapped = sum(1 for r in theme_rows if r["share_status"] == "structured_segment_candidate")
    summary = {
        "batch": args.batch, "stocks": len(codes), "stock_theme_pairs": len(theme_rows),
        "structured_segments": len(segment_rows), "mapped_theme_pairs": mapped,
        "unknown_theme_pairs": sum(1 for r in theme_rows if r["share_status"] == "unknown"),
        "rule": "parse disclosed segment labels and sales shares first, then conservatively map themes to segment labels",
        "critical_rule": "specific themes never inherit a broad segment share (e.g. 下水道 != 土木).",
        "status": "research_candidate_v5",
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
