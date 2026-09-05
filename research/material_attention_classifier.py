from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

TOKYO = ZoneInfo("Asia/Tokyo")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def parse_snapshot(value: str) -> datetime:
    dt = datetime.fromisoformat(str(value).strip())
    return dt if dt.tzinfo else dt.replace(tzinfo=TOKYO)


def catalyst_is_available(catalyst_date: str, snapshot_at: str) -> bool:
    c = str(catalyst_date or "").strip()
    if not c:
        return True
    snap = parse_snapshot(snapshot_at).astimezone(TOKYO)
    if "T" not in c and " " not in c:
        return datetime.fromisoformat(c).date() < snap.date()
    dt = datetime.fromisoformat(c)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TOKYO)
    return dt.astimezone(TOKYO) <= snap


def continuity_class(catalyst_type: str, material_class: str) -> str:
    t = str(catalyst_type or "").upper()
    if material_class in {"NEGATIVE", "NONE", "LOOKAHEAD_REJECT"}:
        return "NONE"
    if any(k in t for k in ("ORDER", "ADOPTION", "CONTRACT", "PARTNER", "TECH", "PRODUCT", "POLICY", "REGUL", "GX", "M&A", "TOB", "BUSINESS_RESTART")):
        return "STRUCTURAL_OR_SPECIFIC"
    if any(k in t for k in ("EARN", "CAPITAL_RETURN", "DIVIDEND", "BUYBACK")):
        return "EVENT_SPECIFIC"
    if any(k in t for k in ("THEME", "SNS", "SYMPATHY")):
        return "LOW"
    return "REVIEWED_VALID"


def classify_review(review: dict[str, str], snapshot_at: str) -> tuple[str, str]:
    gate = str(review.get("catalyst_gate") or "FAIL").upper()
    direction = str(review.get("catalyst_direction") or "").upper()
    status = str(review.get("catalyst_status") or "").upper()
    if not catalyst_is_available(review.get("catalyst_date", ""), snapshot_at):
        return "LOOKAHEAD_REJECT", "Catalyst timestamp was not safely available at the attention snapshot"
    negative = any(k in direction for k in ("NEGATIVE", "DOWN", "BAD")) or status == "NEGATIVE"
    if negative:
        return "NEGATIVE", "Attention can be caused by adverse news; exclude from positive-material candidates"
    if gate == "PASS":
        return "STRONG", "Semantically reviewed catalyst passed the positive-material gate"
    if gate == "CAUTION":
        return "WEAK", "Catalyst exists but is weak, mixed, or theme-sympathy dominated"
    if status in {"NONE", ""} or str(review.get("catalyst_type") or "").upper() == "NONE":
        return "NONE", "No company-specific catalyst was verified"
    return "NONE", "Catalyst did not pass the positive-material gate"


def build_material_stage(discovery_rows, review_rows):
    reviews = {str(r.get("stock_code") or "").strip().upper(): r for r in review_rows}
    rows, counts = [], {}
    for d in discovery_rows:
        if (d.get("attention_stage_status") or "") != "DISCOVERED":
            continue
        code = str(d.get("stock_code") or "").strip().upper(); review = reviews.get(code)
        if review is None:
            material_class, reason, review_status = "NONE", "No semantic material review exists for this discovered candidate", "MISSING_REVIEW"
            review = {}
        else:
            material_class, reason, review_status = *classify_review(review, d.get("snapshot_at", "")), "REVIEWED"
        row = {
            "ifis_rank": d.get("ifis_rank", ""), "stock_code": code, "company_name": d.get("company_name", ""),
            "snapshot_at": d.get("snapshot_at", ""), "best_theme": d.get("best_theme", ""),
            "theme_relevance_score": d.get("theme_relevance_score", ""), "review_status": review_status,
            "material_class": material_class, "material_continuity": continuity_class(review.get("catalyst_type", ""), material_class),
            "catalyst_status": review.get("catalyst_status", ""), "catalyst_direction": review.get("catalyst_direction", ""),
            "catalyst_type": review.get("catalyst_type", ""), "catalyst_date": review.get("catalyst_date", ""),
            "catalyst_gate": review.get("catalyst_gate", "FAIL"), "catalyst_confidence": review.get("confidence", ""),
            "catalyst_summary": review.get("catalyst_summary", ""), "material_reason": reason,
            "source_urls": review.get("source_urls", ""), "actionability": "RESEARCH_ONLY",
        }
        rows.append(row); counts[material_class] = counts.get(material_class, 0) + 1
    rows.sort(key=lambda x: (int(float(x["ifis_rank"] or 999999)), x["stock_code"]))
    return rows, {
        "status": "complete", "stage": "MATERIAL_BEFORE_TECHNICAL", "count": len(rows), "material_class_counts": counts,
        "rule": "Attention is not interpreted as buying demand. Positive materials are semantically reviewed before entry timing; negative, absent, weak, or lookahead-unsafe catalysts cannot qualify for BUY_NOW.",
        "lookahead_rule": "Catalyst information must be safely available by the attention snapshot; same-day date-only timestamps are rejected as ambiguous.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--discovery", required=True); ap.add_argument("--review", required=True); ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(); rows, summary = build_material_stage(read_csv(Path(args.discovery)), read_csv(Path(args.review)))
    out = Path(args.out_dir)
    fields = ["ifis_rank","stock_code","company_name","snapshot_at","best_theme","theme_relevance_score","review_status","material_class","material_continuity","catalyst_status","catalyst_direction","catalyst_type","catalyst_date","catalyst_gate","catalyst_confidence","catalyst_summary","material_reason","source_urls","actionability"]
    write_csv(out / "material_stage.csv", rows, fields); out.mkdir(parents=True, exist_ok=True)
    (out / "material_stage_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
