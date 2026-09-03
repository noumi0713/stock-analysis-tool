from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
from collections import defaultdict
from pathlib import Path

TRIAL_CODES = {"6654","7066","5039","6904","3653","603A","6745","8746","7864","6996"}
SOURCE_REF = "origin/research/volume-top100-overheat"
SOURCE_PATH = "research/data/theme_members_124.csv"


def norm_code(value: str) -> str:
    s = str(value or "").strip().upper()
    m = re.search(r"([0-9]{4}|[0-9]{3}[A-Z])", s)
    return m.group(1) if m else s


def load_rows() -> list[dict[str, str]]:
    raw = subprocess.check_output(
        ["git", "show", f"{SOURCE_REF}:{SOURCE_PATH}"], text=True
    )
    return list(csv.DictReader(io.StringIO(raw)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--outdir", default="research/results/theme_relevance_batches")
    args = ap.parse_args()

    rows = load_rows()
    by_code: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        code = norm_code(row.get("stock_code", ""))
        if code:
            by_code[code].append(row)

    codes = sorted(c for c in by_code if c not in TRIAL_CODES)
    start = (args.batch - 1) * args.size
    selected = codes[start:start + args.size]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    members_path = outdir / f"batch_{args.batch:03d}_memberships.csv"
    stocks_path = outdir / f"batch_{args.batch:03d}_stocks.csv"

    with stocks_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["batch","stock_code","yahoo_ticker","company_name","theme_count"])
        w.writeheader()
        for code in selected:
            rs = by_code[code]
            first = rs[0]
            w.writerow({
                "batch": args.batch,
                "stock_code": code,
                "yahoo_ticker": first.get("yahoo_ticker", ""),
                "company_name": first.get("company_name", ""),
                "theme_count": len(rs),
            })

    with members_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["batch","stock_code","yahoo_ticker","company_name","theme_name","cluster","source_url"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for code in selected:
            for row in sorted(by_code[code], key=lambda r: r.get("\ufefftheme_name", "")):
                w.writerow({
                    "batch": args.batch,
                    "stock_code": code,
                    "yahoo_ticker": row.get("yahoo_ticker", ""),
                    "company_name": row.get("company_name", ""),
                    "theme_name": row.get("\ufefftheme_name", ""),
                    "cluster": row.get("cluster", ""),
                    "source_url": row.get("source_url", ""),
                })

    print(f"batch={args.batch} size={len(selected)} start={start}")
    print(f"first={selected[0] if selected else ''} last={selected[-1] if selected else ''}")
    print(stocks_path)
    print(members_path)


if __name__ == "__main__":
    main()
