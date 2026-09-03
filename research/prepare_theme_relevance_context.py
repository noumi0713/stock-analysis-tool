from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dir", default="research/results/theme_relevance_batches")
    args = ap.parse_args()

    root = Path(args.dir)
    profiles_path = root / f"batch_{args.batch:03d}_profiles.csv"
    memberships_path = root / f"batch_{args.batch:03d}_memberships.csv"
    out_path = root / f"batch_{args.batch:03d}_context.csv"

    with profiles_path.open(encoding="utf-8-sig") as f:
        profiles = list(csv.DictReader(f))
    with memberships_path.open(encoding="utf-8-sig") as f:
        memberships = list(csv.DictReader(f))

    themes: dict[str, list[str]] = defaultdict(list)
    clusters: dict[str, list[str]] = defaultdict(list)
    for row in memberships:
        code = row["stock_code"]
        if row["theme_name"] not in themes[code]:
            themes[code].append(row["theme_name"])
        if row["cluster"] not in clusters[code]:
            clusters[code].append(row["cluster"])

    fields = [
        "batch", "stock_code", "yahoo_ticker", "company_name", "long_name",
        "quote_type", "sector", "industry", "website", "themes", "clusters",
        "business_summary", "profile_status", "profile_error",
    ]
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in profiles:
            code = p["stock_code"]
            w.writerow({
                "batch": args.batch,
                "stock_code": code,
                "yahoo_ticker": p.get("yahoo_ticker", ""),
                "company_name": p.get("company_name", ""),
                "long_name": p.get("long_name", ""),
                "quote_type": p.get("quote_type", ""),
                "sector": p.get("sector", ""),
                "industry": p.get("industry", ""),
                "website": p.get("website", ""),
                "themes": " | ".join(themes.get(code, [])),
                "clusters": " | ".join(clusters.get(code, [])),
                "business_summary": p.get("business_summary", ""),
                "profile_status": p.get("profile_status", ""),
                "profile_error": p.get("profile_error", ""),
            })
    print(out_path)


if __name__ == "__main__":
    main()
