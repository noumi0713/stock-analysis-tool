from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path("research/results/theme_relevance_batches")
LIMIT = 1000

rows = []
for batch in (1, 2):
    v6 = ROOT / f"batch_{batch:03d}_scores_v6.csv"
    v3 = ROOT / f"batch_{batch:03d}_scores_v3.csv"
    with v3.open(encoding="utf-8-sig") as f:
        prior = {(r["stock_code"], r["theme_name"]): r for r in csv.DictReader(f)}
    with v6.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("review_flag") != "revenue_share_unknown":
                continue
            p = prior.get((r["stock_code"], r["theme_name"]), {})
            rows.append({
                "batch": batch,
                "stock_code": r["stock_code"],
                "long_name": r.get("long_name", "") or p.get("long_name", ""),
                "theme_name": r["theme_name"],
                "cluster": r.get("cluster", "") or p.get("cluster", ""),
                "current_business_score": p.get("current_business_score", ""),
                "growth_relevance_score": p.get("growth_relevance_score", ""),
                "prior_relevance_score": p.get("relevance_score", ""),
                "prior_band": p.get("band", ""),
                "prior_confidence": p.get("confidence", ""),
                "prior_review_flag": p.get("review_flag", ""),
                "evidence": p.get("evidence", ""),
                "official_hits": p.get("official_hits", ""),
                "official_hit_count": p.get("official_hit_count", ""),
                "official_urls": p.get("official_urls", ""),
                "sector": p.get("sector", ""),
                "industry": p.get("industry", ""),
                "website": p.get("website", ""),
            })

rows = rows[:LIMIT]
out = ROOT / "discretionary_review_1000_context.csv"
with out.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

summary = {
    "rows": len(rows),
    "batch_counts": dict(Counter(str(r["batch"]) for r in rows)),
    "confidence": dict(Counter(r["prior_confidence"] for r in rows)),
    "review_flags": dict(Counter(r["prior_review_flag"] for r in rows)),
    "bands": dict(Counter(r["prior_band"] for r in rows)),
}
(ROOT / "discretionary_review_1000_context_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False))
