from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--dir", default="research/results/theme_relevance_batches")
    ap.add_argument("--official-text-limit", type=int, default=5000)
    args = ap.parse_args()

    root = Path(args.dir)
    p = f"{args.batch:03d}"
    scores = load_rows(root / f"batch_{p}_scores_v6.csv")
    profiles = {r.get("stock_code", ""): r for r in load_rows(root / f"batch_{p}_profiles.csv")}
    official = {r.get("stock_code", ""): r for r in load_rows(root / f"batch_{p}_official_evidence.csv")}

    queue: list[dict[str, str]] = []
    for r in scores:
        if r.get("review_flag") != "revenue_share_unknown":
            continue
        code = r.get("stock_code", "")
        pr = profiles.get(code, {})
        ev = official.get(code, {})
        queue.append({
            "batch": str(args.batch),
            "stock_code": code,
            "long_name": r.get("long_name") or pr.get("long_name") or pr.get("company_name", ""),
            "theme_name": r.get("theme_name", ""),
            "cluster": r.get("cluster", ""),
            "revenue_share_pct": r.get("revenue_share_pct", ""),
            "revenue_share_status": "unknown",
            "directness_score_prior": r.get("directness_score", ""),
            "growth_score_prior": r.get("growth_relevance_score", ""),
            "sector": pr.get("sector", ""),
            "industry": pr.get("industry", ""),
            "website": pr.get("website", ""),
            "business_summary": pr.get("business_summary", ""),
            "official_status": ev.get("official_status", ""),
            "official_urls": ev.get("official_urls", ""),
            "official_text": (ev.get("official_text", "") or "")[: args.official_text_limit],
            "discretion_status": "pending_chatgpt_review",
            "discretion_relevance_score": "",
            "discretion_confidence": "",
            "discretion_reason": "",
            "discretion_sources": "",
            "final_relevance_score": "",
            "final_score_source": "",
        })

    dst = root / f"batch_{p}_discretion_queue.csv"
    fields = list(queue[0].keys()) if queue else [
        "batch","stock_code","long_name","theme_name","cluster","revenue_share_pct",
        "revenue_share_status","directness_score_prior","growth_score_prior","sector","industry",
        "website","business_summary","official_status","official_urls","official_text",
        "discretion_status","discretion_relevance_score","discretion_confidence","discretion_reason",
        "discretion_sources","final_relevance_score","final_score_source",
    ]
    with dst.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(queue)
    print("discretion_queue_rows", len(queue), dst)


if __name__ == "__main__":
    main()
