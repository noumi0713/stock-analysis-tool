from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TECH_REJECT = {"INVALIDATED", "EXCLUDE_RAN_AWAY"}
PRIORITY_TECH = {"FIRST_PULLBACK_SIGNAL", "BUY_NOW"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def disposition(catalyst_gate: str, technical_status: str) -> str:
    if technical_status in TECH_REJECT:
        return "TECHNICAL_REJECT"
    if catalyst_gate == "PASS":
        if technical_status in PRIORITY_TECH:
            return "PRIORITY_REVIEW"
        if technical_status == "PULLBACK_FORMING":
            return "WATCH_FORMING"
        return "WATCH_WAIT"
    if catalyst_gate == "CAUTION":
        if technical_status == "PULLBACK_FORMING":
            return "SECONDARY_FORMING"
        return "SECONDARY_WATCH"
    return "HOLD_FOR_EVIDENCE"


def combine(review_rows: list[dict[str, str]], technical: dict) -> list[dict]:
    reviews = {str(r["stock_code"]).upper(): r for r in review_rows}
    out = []
    for item in technical.get("items", []):
        code = str(item.get("stock_code", "")).upper()
        r = reviews.get(code)
        if r is None:
            raise ValueError(f"missing catalyst review for {code}")
        tech = item.get("technical_status", "UNKNOWN")
        gate = r.get("catalyst_gate", "FAIL")
        out.append({
            "ifis_rank": int(item.get("ifis_rank") or 999999),
            "stock_code": code,
            "company_name": item.get("company_name", r.get("company_name", "")),
            "best_theme": item.get("best_theme", ""),
            "theme_relevance_score": float(item.get("theme_relevance_score") or 0),
            "catalyst_status": r.get("catalyst_status", "UNKNOWN"),
            "catalyst_direction": r.get("catalyst_direction", "UNKNOWN"),
            "catalyst_type": r.get("catalyst_type", ""),
            "catalyst_gate": gate,
            "catalyst_confidence": r.get("confidence", ""),
            "catalyst_summary": r.get("catalyst_summary", ""),
            "technical_status": tech,
            "discovery_market_date": item.get("discovery_market_date", ""),
            "discovery_price": item.get("discovery_price", ""),
            "latest_market_date": item.get("latest_market_date", ""),
            "latest_close": item.get("latest_close", ""),
            "technical_reasons": " | ".join(item.get("last_reasons", [])),
            "research_disposition": disposition(gate, tech),
            "source_urls": r.get("source_urls", ""),
            "actionability": "RESEARCH_ONLY",
        })
    return sorted(out, key=lambda x: x["ifis_rank"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True)
    ap.add_argument("--technical", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    technical = json.loads(Path(args.technical).read_text(encoding="utf-8"))
    rows = combine(read_csv(Path(args.review)), technical)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "complete",
        "actionability": "RESEARCH_ONLY",
        "rule": "Catalyst review is frozen at the IFIS snapshot. Technical state is combined deterministically; no production trading rule is modified.",
        "count": len(rows),
        "disposition_counts": {},
        "items": rows,
    }
    for r in rows:
        d = r["research_disposition"]
        payload["disposition_counts"][d] = payload["disposition_counts"].get(d, 0) + 1

    (out_dir / "research_watch_final.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = list(rows[0].keys()) if rows else []
    with (out_dir / "research_watch_final.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(json.dumps({
        "status": "complete",
        "count": len(rows),
        "disposition_counts": payload["disposition_counts"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
