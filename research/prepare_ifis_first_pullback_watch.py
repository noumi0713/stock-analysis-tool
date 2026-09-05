from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PRIMARY_CLASSES = {"CORE", "STRONG"}
REFERENCE_CLASSES = {"SUPPORT"}
REJECT_CLASSES = {"NOISE", "UNMAPPED"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_ticker(stock_code: str) -> str:
    code = str(stock_code or "").strip().upper()
    return f"{code}.T" if code else ""


def build_watch(rows: list[dict[str, str]]) -> dict:
    primary: list[dict] = []
    reference: list[dict] = []
    rejected: list[dict] = []

    for row in sorted(rows, key=lambda r: int(r.get("ifis_rank") or 999999)):
        cls = (row.get("candidate_class") or "UNMAPPED").strip()
        item = {
            "ifis_rank": int(row.get("ifis_rank") or 999999),
            "stock_code": row.get("stock_code", ""),
            "ticker": normalize_ticker(row.get("stock_code", "")),
            "company_name": row.get("company_name", ""),
            "snapshot_at": row.get("snapshot_at", ""),
            "candidate_class": cls,
            "best_theme": row.get("best_theme", ""),
            "theme_relevance_score": float(row.get("best_relevance_score") or 0.0),
            "theme_relevance_band": row.get("best_band", ""),
            "theme_relevance_confidence": row.get("best_confidence", ""),
            "theme_count": int(row.get("theme_count") or 0),
            "all_themes_sorted": row.get("all_themes_sorted", ""),
            "mapped_to_master": str(row.get("mapped_to_master", "")).lower() == "true",
        }
        if cls in PRIMARY_CLASSES:
            item["watch_role"] = "PRIMARY"
            primary.append(item)
        elif cls in REFERENCE_CLASSES:
            item["watch_role"] = "REFERENCE"
            reference.append(item)
        else:
            item["watch_role"] = "REJECTED"
            rejected.append(item)

    return {
        "status": "complete",
        "selection_rule": {
            "PRIMARY": "IFIS candidates classified CORE or STRONG by finalized 124-theme relevance master",
            "REFERENCE": "SUPPORT candidates retained for comparison only",
            "REJECTED": "NOISE or UNMAPPED candidates are not sent to primary first-pullback monitoring",
        },
        "ordering_rule": "IFIS rank remains the primary order. Theme relevance is a gate, not a trading score.",
        "active_inputs": ["IFIS manual snapshot", "completed 124-theme relevance master"],
        "excluded_inputs": ["Minkabu"],
        "primary_count": len(primary),
        "reference_count": len(reference),
        "rejected_count": len(rejected),
        "primary_watch": primary,
        "reference_watch": reference,
        "rejected": rejected,
    }


def write_flat_csv(path: Path, payload: dict) -> None:
    rows = payload["primary_watch"] + payload["reference_watch"] + payload["rejected"]
    fields = [
        "watch_role", "ifis_rank", "stock_code", "ticker", "company_name", "snapshot_at",
        "candidate_class", "best_theme", "theme_relevance_score", "theme_relevance_band",
        "theme_relevance_confidence", "theme_count", "all_themes_sorted", "mapped_to_master",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    src = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_watch(read_rows(src))
    payload["source_file"] = str(src)

    (out_dir / "first_pullback_watch_candidates.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_flat_csv(out_dir / "first_pullback_watch_candidates.csv", payload)

    if payload["primary_count"] == 0:
        raise SystemExit("No primary watch candidates were produced")

    print(json.dumps({
        "status": payload["status"],
        "primary_count": payload["primary_count"],
        "reference_count": payload["reference_count"],
        "rejected_count": payload["rejected_count"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
