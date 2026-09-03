from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def score_band(score: int) -> str:
    if score >= 80:
        return "主力テーマ"
    if score >= 60:
        return "有力関連"
    if score >= 40:
        return "補助関連"
    return "ノイズ候補"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dir", default="research/results/theme_relevance_batches")
    args = ap.parse_args()
    root = Path(args.dir)
    auto_path = root / f"batch_{args.batch:03d}_scores_v2.csv"
    override_path = root / f"batch_{args.batch:03d}_manual_overrides.csv"
    final_path = root / f"batch_{args.batch:03d}_scores_reviewed.csv"
    summary_path = root / f"batch_{args.batch:03d}_reviewed_summary.json"

    with auto_path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    overrides: dict[tuple[str, str], dict[str, str]] = {}
    if override_path.exists():
        with override_path.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                overrides[(r["stock_code"], r["theme_name"])] = r

    output = []
    applied = 0
    for row in rows:
        key = (row["stock_code"], row["theme_name"])
        ov = overrides.get(key)
        out = dict(row)
        out["score_source"] = "auto_v2"
        out["official_source_url"] = ""
        if ov:
            applied += 1
            out["current_business_score"] = ov["current_business_score"]
            out["growth_relevance_score"] = ov["growth_relevance_score"]
            out["relevance_score"] = ov["relevance_score"]
            out["band"] = score_band(int(ov["relevance_score"]))
            out["confidence"] = ov["confidence"]
            out["review_flag"] = ov["review_status"]
            out["evidence"] = ov["evidence"]
            out["score_source"] = "official_ir_override"
            out["official_source_url"] = ov["source_url"]
        output.append(out)

    fields = list(output[0].keys()) if output else []
    with final_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(output)

    band_counts = Counter(r["band"] for r in output)
    confidence_counts = Counter(r["confidence"] for r in output)
    source_counts = Counter(r["score_source"] for r in output)
    review_counts = Counter(r["review_flag"] for r in output)
    summary = {
        "batch": args.batch,
        "stocks": len({r["stock_code"] for r in output}),
        "stock_theme_pairs": len(output),
        "official_ir_overrides": applied,
        "band_counts": dict(band_counts),
        "confidence_counts": dict(confidence_counts),
        "score_source_counts": dict(source_counts),
        "review_counts": dict(review_counts),
        "status": "research_provisional",
        "warning": "Only official_ir_override rows are directly verified against official sources. auto_v2 rows are current-business-profile first-pass and are not suitable for historical PIT backtests.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(final_path)


if __name__ == "__main__":
    main()
