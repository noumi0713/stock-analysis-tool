from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path("research/results/theme_relevance_batches")
PATH = ROOT / "batch_001_revenue_mix_v6.csv"


def load():
    with PATH.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {(r["stock_code"], r["theme_name"]): r for r in rows}


def val(rows, code, theme):
    r = rows.get((code, theme))
    if not r or not r.get("revenue_share_pct"):
        return None
    return float(r["revenue_share_pct"])


def close(x, target, tol=0.35):
    return x is not None and abs(x-target) <= tol


def unknown_or_range(x, lo, hi):
    return x is None or lo <= x <= hi


def main():
    rows = load()
    checks = []
    # Known-good explicit sales mixes.
    checks.append(("1939 lease", close(val(rows,"1939","リース"),1.5)))
    checks.append(("1939 solar", close(val(rows,"1939","太陽光発電関連"),2.0)))
    checks.append(("2198 care", close(val(rows,"2198","介護関連"),3.0)))
    checks.append(("2198 food", close(val(rows,"2198","食品"),2.0)))
    checks.append(("1826 construction", unknown_or_range(val(rows,"1826","建設"),95,100.5)))
    # Previous v5 false positives must be rejected or corrected.
    checks.append(("1808 construction", unknown_or_range(val(rows,"1808","建設"),50,100.5)))
    checks.append(("1827 construction", unknown_or_range(val(rows,"1827","建設"),80,100.5)))
    checks.append(("1888 construction", unknown_or_range(val(rows,"1888","建設"),90,100.5)))
    checks.append(("1897 construction", unknown_or_range(val(rows,"1897","建設"),50,100.5)))
    checks.append(("2404 restaurant", unknown_or_range(val(rows,"2404","外食"),20,30)))
    # Broad food theme should aggregate directly food-related segments when a coherent table exists.
    checks.append(("2001 food", unknown_or_range(val(rows,"2001","食品"),80,100.5)))
    checks.append(("2207 food", unknown_or_range(val(rows,"2207","食品"),75,90)))
    # Ownership ratios / order mix must not become revenue exposure.
    x2413 = val(rows,"2413","IT関連")
    checks.append(("2413 ownership false positive", x2413 is None or abs(x2413-18.7)>0.5))
    x1982 = val(rows,"1982","データセンター")
    checks.append(("1982 order mix false positive", x1982 is None))
    # Very stale 2019 fallback should not be accepted as current revenue mix.
    r2353 = rows.get(("2353","不動産関連"), {})
    checks.append(("2353 stale source", not r2353.get("revenue_share_pct") or "2019" not in r2353.get("source_url", "")))

    bad = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(("PASS" if ok else "FAIL"), name)
    if bad:
        print("FAILED:", ", ".join(bad))
        return 1
    print("All v6 revenue QA checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
