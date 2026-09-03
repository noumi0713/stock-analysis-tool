from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--per-group", type=int, default=5)
    ap.add_argument("--dir", default="research/results/theme_relevance_batches")
    args = ap.parse_args()
    root = Path(args.dir)
    src = root / f"batch_{args.batch:03d}_scores_v2.csv"
    dst = root / f"batch_{args.batch:03d}_qa_sample.csv"
    with src.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)
    for r in rows:
        flag = r.get("review_flag", "")
        if flag in {"qa_borderline", "qa_peripheral", "needs_official_ir", "needs_review_macro"}:
            group = flag
        elif r.get("band") == "主力テーマ" and r.get("confidence") == "A":
            group = "core_A"
        elif r.get("band") == "ノイズ候補":
            group = "noise"
        else:
            group = "other"
        groups[group].append(r)

    selected = []
    seen_stocks = set()
    order = ["core_A", "qa_borderline", "qa_peripheral", "needs_official_ir", "needs_review_macro", "noise", "other"]
    for group in order:
        candidates = sorted(groups.get(group, []), key=lambda r: (r["stock_code"], r["theme_name"]))
        picked = 0
        for r in candidates:
            # Prefer different companies so the sample covers more business models.
            if r["stock_code"] in seen_stocks and len(candidates) > args.per_group:
                continue
            out = dict(r)
            out["qa_group"] = group
            selected.append(out)
            seen_stocks.add(r["stock_code"])
            picked += 1
            if picked >= args.per_group:
                break

    fields = ["qa_group"] + [x for x in selected[0].keys() if x != "qa_group"] if selected else []
    with dst.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(selected)
    print(f"sample={len(selected)}")
    for r in selected:
        print(r["qa_group"], r["stock_code"], r["long_name"], r["theme_name"], r["relevance_score"], r["review_flag"])
    print(dst)


if __name__ == "__main__":
    main()
