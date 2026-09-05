from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value, default=None):
    try:
        s = str(value).strip()
        return float(s) if s else default
    except Exception:
        return default


def as_int(value, default=999999):
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def normalize_ticker(code: str) -> str:
    return f"{str(code or '').strip().upper()}.T"


def build_watch(rows: list[dict[str, str]]) -> dict:
    primary = []
    for r in rows:
        if (r.get("attention_stage_status") or "").strip() != "DISCOVERED":
            continue
        code = (r.get("stock_code") or "").strip().upper()
        if not code:
            continue
        primary.append({
            "ifis_rank": as_int(r.get("ifis_rank")), "stock_code": code, "ticker": normalize_ticker(code),
            "company_name": r.get("company_name", ""), "snapshot_at": r.get("snapshot_at", ""),
            "candidate_class": r.get("candidate_class", ""), "best_theme": r.get("best_theme", ""),
            "theme_relevance_score": as_float(r.get("theme_relevance_score"), 0.0),
            "theme_relevance_band": r.get("theme_relevance_band", ""),
            "theme_relevance_confidence": r.get("theme_relevance_confidence", ""),
            "minkabu_attention_type": r.get("minkabu_attention_type", ""),
            "minkabu_theme_rank": as_int(r.get("minkabu_theme_rank")),
            "minkabu_relevance": as_float(r.get("minkabu_relevance"), None),
            "theme_price_change_1d_pct": as_float(r.get("theme_price_change_1d_pct"), None),
            "theme_price_change_5d_pct": as_float(r.get("theme_price_change_5d_pct"), None),
            "float_shares": as_float(r.get("float_shares"), None), "watch_role": "PRIMARY",
        })
    primary.sort(key=lambda x: (x["ifis_rank"], x["stock_code"]))
    return {
        "status": "complete", "stage": "POST_ATTENTION_DISCOVERY_TECHNICAL_WATCH",
        "selection_rule": "Only cross-source DISCOVERED candidates enter the technical watch. This is not a buy signal.",
        "active_inputs": ["IFIS top-N attention snapshot", "Minkabu POPULAR/RISING theme snapshot", "Minkabu stock-theme relation when supplied", "finalized Kabutan-derived 124-theme relevance master"],
        "primary_count": len(primary), "primary_watch": primary,
        "reference_count": 0, "reference_watch": [], "rejected_count": 0, "rejected": [],
    }


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(); payload = build_watch(read_csv(Path(args.input)))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if payload["primary_count"] == 0:
        raise SystemExit("No discovered candidates were available for technical monitoring")
    print(json.dumps({"status": "complete", "primary_count": payload["primary_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
