from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from score_theme_relevance import (
    GROWTH_WORDS,
    INDUSTRY_BOOST,
    MACRO_THEMES,
    RULES,
    band,
    hits,
    norm_text,
)


def earliest_ratio(summary: str, terms: list[str]) -> float | None:
    s = (summary or "").lower()
    if not s:
        return None
    positions = [s.find(t.lower()) for t in terms if s.find(t.lower()) >= 0]
    if not positions:
        return None
    return min(positions) / max(1, len(s))


def score_pair(theme: str, profile: dict[str, str]) -> dict[str, object]:
    rule = RULES.get(theme, {"strong": [], "weak": []})
    summary = profile.get("business_summary", "") or ""
    text = norm_text(
        profile.get("long_name", ""),
        profile.get("sector", ""),
        profile.get("industry", ""),
        summary,
    )
    strong = hits(text, list(rule.get("strong", [])))
    weak = hits(text, list(rule.get("weak", [])))
    industry = (profile.get("industry") or "").lower()
    quote_type = (profile.get("quote_type") or "").upper()
    industry_hit = any(x in industry for x in INDUSTRY_BOOST.get(theme, []))

    if quote_type in {"ETF", "MUTUALFUND"} or " etn" in text:
        if strong or weak or theme.lower() in text:
            return {
                "current": 100,
                "growth": 100,
                "score": 100,
                "confidence": "A",
                "evidence": "投資商品としてテーマ価格へ直接連動",
                "review": "non_operating_security",
                "strong_hits": strong,
                "weak_hits": weak,
                "position": None,
            }
        return {
            "current": 0,
            "growth": 0,
            "score": 0,
            "confidence": "C",
            "evidence": "事業会社ではないため事業関連度の対象外",
            "review": "non_operating_security",
            "strong_hits": strong,
            "weak_hits": weak,
            "position": None,
        }

    pos = earliest_ratio(summary, strong)
    current = 8

    # Industry evidence is strong for broad themes such as food, construction and banks.
    if industry_hit:
        current = 90

    if strong:
        # One direct term is meaningful, but where it appears matters. A hit in the
        # opening/core-business portion is very different from one item in a long product list.
        direct = 82 + min(12, 5 * (len(strong) - 1))
        if pos is not None:
            if pos <= 0.20:
                direct += 10
            elif pos <= 0.40:
                direct += 4
            elif pos >= 0.65:
                direct = min(direct, 58)
            elif pos >= 0.50:
                direct = min(direct, 66)
        current = max(current, min(100, direct))

    if weak and not strong:
        weak_score = 42 + min(12, 4 * (len(weak) - 1))
        current = max(current, weak_score)

    if industry_hit and strong:
        current = min(100, max(current, 94))

    # No textual/industry evidence means Kabutan membership alone is not enough.
    if not strong and not weak and not industry_hit:
        current = 8

    # Macro-benefit themes need P&L sensitivity evidence; cap generic text inference.
    if theme in MACRO_THEMES:
        current = min(current, 66 if strong else 50)

    growth_hits = hits(text, GROWTH_WORDS)
    if current >= 80:
        growth = max(65, current - 8)
    elif current >= 60:
        growth = max(48, current - 10)
    elif current >= 40:
        growth = max(30, current - 12)
    else:
        growth = max(5, current - 3)

    if growth_hits and (strong or weak):
        growth = min(100, growth + 8 + min(10, 2 * len(growth_hits)))
    if theme in MACRO_THEMES:
        growth = min(growth, 58)

    final = int(round(current * 0.70 + growth * 0.30))

    if final >= 85 and (strong or industry_hit):
        confidence = "A"
    elif final >= 60 and (strong or weak or industry_hit):
        confidence = "B"
    else:
        confidence = "C"
    if theme in MACRO_THEMES:
        confidence = "B" if final >= 60 else "C"

    evidence_parts = []
    if industry_hit:
        evidence_parts.append(f"industry={profile.get('industry','')}")
    if strong:
        evidence_parts.append("direct=" + "/".join(strong[:3]))
        if pos is not None:
            evidence_parts.append(f"position={pos:.2f}")
    elif weak:
        evidence_parts.append("adjacent=" + "/".join(weak[:3]))
    if growth_hits and (strong or weak):
        evidence_parts.append("growth=" + "/".join(growth_hits[:2]))
    if not evidence_parts:
        evidence_parts.append("事業概要に直接根拠を確認できず")

    review = "ok"
    if not summary:
        review = "needs_official_ir"
    elif theme in MACRO_THEMES:
        review = "needs_review_macro"
    elif confidence == "C":
        review = "needs_review"
    elif 55 <= final < 75:
        review = "qa_borderline"
    elif pos is not None and pos >= 0.50:
        review = "qa_peripheral"

    return {
        "current": current,
        "growth": growth,
        "score": final,
        "confidence": confidence,
        "evidence": "; ".join(evidence_parts),
        "review": review,
        "strong_hits": strong,
        "weak_hits": weak,
        "position": pos,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dir", default="research/results/theme_relevance_batches")
    args = ap.parse_args()
    root = Path(args.dir)
    profiles_path = root / f"batch_{args.batch:03d}_profiles.csv"
    memberships_path = root / f"batch_{args.batch:03d}_memberships.csv"
    out_path = root / f"batch_{args.batch:03d}_scores_v2.csv"
    summary_path = root / f"batch_{args.batch:03d}_score_summary_v2.json"

    with profiles_path.open(encoding="utf-8-sig") as f:
        profiles = {r["stock_code"]: r for r in csv.DictReader(f)}
    with memberships_path.open(encoding="utf-8-sig") as f:
        memberships = list(csv.DictReader(f))

    fields = [
        "batch", "stock_code", "long_name", "theme_name", "cluster",
        "current_business_score", "growth_relevance_score", "relevance_score",
        "band", "confidence", "review_flag", "evidence", "strong_hits", "weak_hits",
        "evidence_position", "quote_type", "sector", "industry", "website",
    ]
    rows_out: list[dict[str, object]] = []
    for m in memberships:
        code = m["stock_code"]
        p = profiles.get(code, {})
        s = score_pair(m["theme_name"], p)
        rows_out.append({
            "batch": args.batch,
            "stock_code": code,
            "long_name": p.get("long_name", m.get("company_name", "")),
            "theme_name": m["theme_name"],
            "cluster": m["cluster"],
            "current_business_score": s["current"],
            "growth_relevance_score": s["growth"],
            "relevance_score": s["score"],
            "band": band(int(s["score"])),
            "confidence": s["confidence"],
            "review_flag": s["review"],
            "evidence": s["evidence"],
            "strong_hits": " | ".join(s["strong_hits"]),
            "weak_hits": " | ".join(s["weak_hits"]),
            "evidence_position": "" if s["position"] is None else f"{s['position']:.4f}",
            "quote_type": p.get("quote_type", ""),
            "sector": p.get("sector", ""),
            "industry": p.get("industry", ""),
            "website": p.get("website", ""),
        })

    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    band_counts = defaultdict(int)
    confidence_counts = defaultdict(int)
    review_counts = defaultdict(int)
    for r in rows_out:
        band_counts[r["band"]] += 1
        confidence_counts[r["confidence"]] += 1
        review_counts[r["review_flag"]] += 1
    summary = {
        "batch": args.batch,
        "stocks": len(profiles),
        "stock_theme_pairs": len(rows_out),
        "band_counts": dict(band_counts),
        "confidence_counts": dict(confidence_counts),
        "review_counts": dict(review_counts),
        "scoring": "v2: 0.70*current_business + 0.30*growth_relevance; direct-term position penalty",
        "note": "Automated first pass. Official-IR QA is required for macro, missing-profile and sampled borderline/peripheral rows.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
