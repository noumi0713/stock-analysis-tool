from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from collect_revenue_mix_v5 import (
    NUM_RE,
    PCT_RE,
    clean_label,
    discover_documents,
    norm,
    normalize_match,
    plausible_segment_label,
)

# v6 principle:
# 1) reconstruct a coherent revenue-mix table from one official document,
# 2) validate that its component shares approximately sum to 100%,
# 3) prefer the newest coherent table,
# 4) only then map 124 themes to the disclosed segments.
# Specific themes never inherit a broad segment share.

STRONG_PERCENT_MARKERS = (
    "セグメント別売上高構成比",
    "売上高構成比",
    "売上構成比",
    "売上高割合",
    "売上構成割合",
    "事業別連結売上高",
    "事業別売上高",
    "revenue mix",
    "sales mix",
    "revenue composition",
)
EXTERNAL_MARKERS = ("外部顧客への売上高", "外部顧客に対する売上高", "revenue from external customers")
SEG_ENDINGS = ("事業", "部門", "サービス")
LABEL_RE = re.compile(r"([一-龥ぁ-んァ-ンA-Za-z0-9・＆&／/+＋ー\-\s]{1,40}?(?:事業|部門|サービス))")
PAIR_RE = re.compile(
    r"([一-龥ぁ-んァ-ンA-Za-z0-9・＆&／/+＋ー\-\s]{1,36}?(?:事業|部門|工事|設備|発電|食品|介護|婚礼|不動産|リース|サービス))\s*[（(]?\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%",
    re.I,
)

BROAD_RULES: dict[str, list[str]] = {
    "建設": ["建設", "土木", "建築", "鉄構建設"],
    "食品": ["食品", "食材", "加工食品", "食肉", "製粉", "製菓", "菓子", "パン"],
    "住宅関連": ["住宅"],
    "不動産関連": ["不動産"],
    "介護関連": ["介護"],
    "医薬品関連": ["医薬", "製薬"],
    "IT関連": ["it", "情報", "システム", "ソフトウェア", "クラウド", "dx"],
    "リース": ["リース"],
    "外食": ["外食", "レストラン", "飲食"],
    "小売り": ["小売"],
    "物流": ["物流"],
    "倉庫": ["倉庫"],
    "ホテル": ["ホテル"],
    "海運": ["海運"],
    "航空": ["航空"],
    "eコマース": ["ec事業", "eコマース", "電子商取引"],
    "クラウドコンピューティング": ["クラウド"],
    "デジタルトランスフォーメーション": ["dx"],
    "SaaS": ["saas"],
}


def extract_year(url: str, evidence: str) -> int:
    vals = [int(x) for x in re.findall(r"20(?:2[0-9]|1[8-9])", (url or "") + " " + (evidence or ""))]
    return max(vals) if vals else 0


def clean_table_label(raw: str) -> str:
    s = clean_label(raw)
    s = re.sub(r"\s+", "", s)
    # Flattened PDF headers can retain leading table words.
    for prefix in ("報告セグメント", "売上高", "外部顧客への売上高", "合計"):
        if s.startswith(prefix) and len(s) > len(prefix) + 1:
            s = s[len(prefix):]
    return s


def extract_header_labels(before: str) -> list[str]:
    labels: list[str] = []
    for m in LABEL_RE.finditer(before):
        lab = clean_table_label(m.group(1))
        if not plausible_segment_label(lab):
            continue
        if lab not in labels:
            labels.append(lab)
    return labels[-10:]


def parse_external_tables(text: str, url: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    low = text.lower()
    for marker in EXTERNAL_MARKERS:
        pos = 0
        while True:
            idx = low.find(marker.lower(), pos)
            if idx < 0:
                break
            before_raw = text[max(0, idx - 750):idx]
            after_raw = text[idx + len(marker): min(len(text), idx + len(marker) + 650)]
            labels = extract_header_labels(norm(before_raw))
            values: list[float] = []
            for raw in NUM_RE.findall(norm(after_raw)):
                try:
                    values.append(float(raw.replace(",", "")))
                except Exception:
                    pass
            if len(labels) >= 2 and len(values) >= len(labels) + 1:
                # The consolidated/total figure is normally the largest value shortly after segment values.
                probe = values[: min(len(values), len(labels) + 4)]
                total = max(probe) if probe else 0.0
                seg_vals = values[: len(labels)]
                if total > 0 and total not in seg_vals[: max(1, len(seg_vals) - 1)]:
                    coverage = sum(v for v in seg_vals if 0 <= v <= total) / total
                    # Missing a tiny "other" column is acceptable; material omissions are not.
                    if 0.94 <= coverage <= 1.03:
                        ev = norm(before_raw + " " + marker + " " + after_raw)[:2200]
                        for lab, val in zip(labels, seg_vals):
                            if 0 <= val <= total:
                                rows.append({
                                    "segment_name": lab,
                                    "share_pct": round(val / total * 100, 2),
                                    "basis": "external_customer_sales_table_v6",
                                    "segment_sales": val,
                                    "total_sales": total,
                                    "source_url": url,
                                    "evidence": ev,
                                    "year": extract_year(url, ev),
                                })
            pos = idx + len(marker)
    return rows


def percent_windows(text: str, radius: int = 1200) -> list[str]:
    low = text.lower()
    out: list[str] = []
    for marker in STRONG_PERCENT_MARKERS:
        pos = 0
        while True:
            idx = low.find(marker.lower(), pos)
            if idx < 0:
                break
            win = text[max(0, idx - radius): min(len(text), idx + len(marker) + radius)]
            if win not in out:
                out.append(win)
            pos = idx + len(marker)
    return out[:16]


def parse_explicit_mix(text: str, url: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for win in percent_windows(text):
        compact = re.sub(r"\s+", " ", win.replace("％", "%"))
        local: list[dict[str, object]] = []
        for m in PAIR_RE.finditer(compact):
            lab = clean_table_label(m.group(1))
            pct = float(m.group(2))
            if not (0 <= pct <= 100) or not plausible_segment_label(lab):
                continue
            local.append({
                "segment_name": lab,
                "share_pct": pct,
                "basis": "explicit_sales_mix_v6",
                "segment_sales": "",
                "total_sales": "",
                "source_url": url,
                "evidence": compact[:2200],
                "year": extract_year(url, compact),
            })
        # A coherent published mix should substantially account for total sales.
        ded: dict[str, dict[str, object]] = {}
        for r in local:
            ded[normalize_match(str(r["segment_name"]))] = r
        vals = list(ded.values())
        total = sum(float(r["share_pct"]) for r in vals)
        if len(vals) >= 2 and 94 <= total <= 103:
            rows.extend(vals)
    return rows


def group_tables(candidates: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for r in candidates:
        groups[(str(r["source_url"]), str(r["basis"]), int(r.get("year") or 0))].append(r)
    valid: list[list[dict[str, object]]] = []
    for _, rows in groups.items():
        ded: dict[str, dict[str, object]] = {}
        # Later row wins: financial documents usually show prior period before current period.
        for r in rows:
            ded[normalize_match(str(r["segment_name"]))] = r
        vals = list(ded.values())
        s = sum(float(r["share_pct"]) for r in vals)
        if len(vals) >= 2 and 94 <= s <= 103:
            valid.append(vals)
    return valid


def choose_best_table(tables: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    if not tables:
        return []
    def score(rows: list[dict[str, object]]) -> tuple[int, int, float, int]:
        year = max(int(r.get("year") or 0) for r in rows)
        basis = 2 if any(str(r["basis"]).startswith("external") for r in rows) else 1
        closeness = -abs(100 - sum(float(r["share_pct"]) for r in rows))
        return year, basis, closeness, len(rows)
    return max(tables, key=score)


def segment_matches_theme(label: str, theme: str) -> bool:
    seg = normalize_match(label)
    if theme in BROAD_RULES:
        return any(normalize_match(x) in seg for x in BROAD_RULES[theme])
    # Specific themes: exact/direct wording only. No broad inheritance.
    t = normalize_match(theme[:-2] if theme.endswith("関連") else theme)
    aliases = {
        "太陽光発電関連": ["太陽光発電", "太陽光"],
        "再生可能エネルギー": ["再生可能エネルギー", "再エネ"],
        "人工知能": ["人工知能", "ai"],
        "生成AI": ["生成ai"],
        "フィンテック": ["フィンテック", "fintech"],
    }
    terms = [normalize_match(x) for x in aliases.get(theme, [])] or [t]
    return any(x and x in seg for x in terms)


def map_themes(table: list[dict[str, object]], themes: list[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for theme in themes:
        matched = [r for r in table if segment_matches_theme(str(r["segment_name"]), theme)]
        share = sum(float(r["share_pct"]) for r in matched)
        if not matched or share > 100.5:
            out.append({"theme_name": theme, "revenue_share_pct": "", "share_status": "unknown", "share_basis": "", "segment_names": "", "source_url": "", "evidence": "売上セグメントとテーマを直接対応できず"})
        else:
            out.append({
                "theme_name": theme,
                "revenue_share_pct": f"{share:.2f}",
                "share_status": "structured_segment_candidate_v6",
                "share_basis": "+".join(sorted({str(r["basis"]) for r in matched})),
                "segment_names": " | ".join(str(r["segment_name"]) for r in matched),
                "source_url": str(matched[0]["source_url"]),
                "evidence": " || ".join(str(r["evidence"])[:500] for r in matched)[:1800],
            })
    return out


def collect_one(profile: dict[str, str], themes: list[str]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    code = profile.get("stock_code", "")
    quote_type = (profile.get("quote_type") or "").upper()
    website = norm(profile.get("website", ""))
    if quote_type in {"ETF", "MUTUALFUND"}:
        return ([{"stock_code": code, "theme_name": t, "revenue_share_pct": "", "share_status": "non_operating_security", "share_basis": "", "segment_names": "", "source_url": "", "evidence": "事業会社ではないため対象外"} for t in themes], [])
    if not website:
        return ([{"stock_code": code, "theme_name": t, "revenue_share_pct": "", "share_status": "unknown", "share_basis": "", "segment_names": "", "source_url": "", "evidence": "会社公式サイトURL未取得"} for t in themes], [])
    if not website.startswith(("http://", "https://")):
        website = "https://" + website

    import requests
    docs = discover_documents(requests.Session(), website)
    candidates: list[dict[str, object]] = []
    for url, text, dtype in docs:
        candidates.extend(parse_external_tables(text, url))
        candidates.extend(parse_explicit_mix(text, url))
    table = choose_best_table(group_tables(candidates))
    mapped = map_themes(table, themes)
    for r in mapped:
        r["stock_code"] = code
    serial = [{
        "stock_code": code,
        "segment_name": str(r["segment_name"]),
        "share_pct": f"{float(r['share_pct']):.2f}",
        "basis": str(r["basis"]),
        "segment_sales": str(r.get("segment_sales", "")),
        "total_sales": str(r.get("total_sales", "")),
        "source_url": str(r["source_url"]),
        "year": str(r.get("year", "")),
        "evidence": str(r["evidence"])[:1800],
    } for r in table]
    return mapped, serial


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--dir", default="research/results/theme_relevance_batches")
    args = ap.parse_args()
    root = Path(args.dir)
    with (root / f"batch_{args.batch:03d}_profiles.csv").open(encoding="utf-8-sig") as f:
        profiles = {r["stock_code"]: r for r in csv.DictReader(f)}
    themes_by_stock: dict[str, list[str]] = defaultdict(list)
    with (root / f"batch_{args.batch:03d}_memberships.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            themes_by_stock[r["stock_code"]].append(r["theme_name"])
    codes = list(themes_by_stock)
    theme_rows: list[dict[str, str]] = []
    seg_rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(collect_one, profiles.get(c, {"stock_code": c}), themes_by_stock[c]): c for c in codes}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                tr, sr = fut.result()
            except Exception as exc:
                tr = [{"stock_code": c, "theme_name": t, "revenue_share_pct": "", "share_status": "error", "share_basis": "", "segment_names": "", "source_url": "", "evidence": f"{type(exc).__name__}: {exc}"} for t in themes_by_stock[c]]
                sr = []
            theme_rows.extend(tr); seg_rows.extend(sr)
            print(i, c, "segments", len(sr), "mapped", sum(bool(r["revenue_share_pct"]) for r in tr))
    order = {(c, t): (i, j) for i, c in enumerate(codes) for j, t in enumerate(themes_by_stock[c])}
    theme_rows.sort(key=lambda r: order.get((r["stock_code"], r["theme_name"]), (10**9, 10**9)))
    seg_rows.sort(key=lambda r: (codes.index(r["stock_code"]) if r["stock_code"] in codes else 10**9, r["segment_name"]))
    tf = ["stock_code", "theme_name", "revenue_share_pct", "share_status", "share_basis", "segment_names", "source_url", "evidence"]
    sf = ["stock_code", "segment_name", "share_pct", "basis", "segment_sales", "total_sales", "source_url", "year", "evidence"]
    out_t = root / f"batch_{args.batch:03d}_revenue_mix_v6.csv"
    out_s = root / f"batch_{args.batch:03d}_segments_v6.csv"
    with out_t.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=tf); w.writeheader(); w.writerows(theme_rows)
    with out_s.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=sf); w.writeheader(); w.writerows(seg_rows)
    mapped = sum(r["share_status"] == "structured_segment_candidate_v6" for r in theme_rows)
    summary = {
        "batch": args.batch,
        "stocks": len(codes),
        "stock_theme_pairs": len(theme_rows),
        "validated_segment_rows": len(seg_rows),
        "mapped_theme_pairs": mapped,
        "unknown_theme_pairs": sum(r["share_status"] == "unknown" for r in theme_rows),
        "rule": "one coherent official revenue table per stock; newest valid table; broad themes aggregate direct segments; specific themes never inherit broad shares",
        "status": "research_candidate_v6",
    }
    (root / f"batch_{args.batch:03d}_revenue_mix_v6_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
