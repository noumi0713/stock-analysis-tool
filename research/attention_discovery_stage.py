from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def norm_code(value: str) -> str:
    s = str(value or "").strip().upper()
    s = re.sub(r"\.(T|JP|S|N|F)$", "", s)
    m = re.search(r"([0-9]{4}|[0-9]{3}[A-Z])", s)
    return m.group(1) if m else s


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def as_int(value, default=999999) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def as_float(value, default=None):
    try:
        s = str(value).strip()
        return float(s) if s else default
    except Exception:
        return default


def relevance_class(score: float | None) -> str:
    if score is None:
        return "UNMAPPED"
    if score >= 80:
        return "CORE"
    if score >= 60:
        return "STRONG"
    if score >= 40:
        return "SUPPORT"
    return "NOISE"


def load_master(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        if (r.get("status") or "finalized") != "finalized":
            continue
        code = norm_code(r.get("stock_code", ""))
        if not code:
            continue
        out.setdefault(code, []).append(r)
    for rows_for_code in out.values():
        rows_for_code.sort(
            key=lambda r: (
                -as_float(r.get("final_relevance_score"), -1.0),
                r.get("theme_name", ""),
            )
        )
    return out


def build_discovery(ifis_rows, master_rows, top_n: int = 30, min_theme_relevance: float = 40.0):
    master = load_master(master_rows)
    normalized_ifis, seen = [], set()
    for i, r in enumerate(ifis_rows, 1):
        rank = as_int(r.get("ifis_rank"), i)
        code = norm_code(r.get("stock_code", "") or r.get("code", "") or r.get("ticker", ""))
        if code and rank <= top_n and code not in seen:
            seen.add(code)
            normalized_ifis.append((rank, code, r))
    normalized_ifis.sort(key=lambda x: (x[0], x[1]))

    theme_rows, candidates = [], []
    reason_counts = Counter()
    for rank, code, ifis in normalized_ifis:
        themes = master.get(code, [])
        eligible = []
        for t in themes:
            score = as_float(t.get("final_relevance_score"), 0.0)
            is_eligible = score >= min_theme_relevance
            row = {
                "ifis_rank": rank,
                "stock_code": code,
                "company_name": (ifis.get("company_name") or t.get("company_name") or "").strip(),
                "snapshot_at": (ifis.get("snapshot_at") or "").strip(),
                "ifis_attention_score": (ifis.get("attention_score") or "").strip(),
                "theme_name": t.get("theme_name", ""),
                "theme_relevance_score": score,
                "theme_relevance_band": t.get("final_band", ""),
                "theme_relevance_confidence": t.get("final_confidence", ""),
                "eligible_by_relevance": is_eligible,
                "float_shares": as_float(ifis.get("float_shares"), None),
            }
            theme_rows.append(row)
            if is_eligible:
                eligible.append(row)
        if not themes:
            reason_counts["UNMAPPED"] += 1
            continue
        if not eligible:
            reason_counts["NOISE_ONLY"] += 1
            continue

        best = eligible[0]
        candidates.append({
            "ifis_rank": rank,
            "stock_code": code,
            "company_name": best["company_name"],
            "snapshot_at": best["snapshot_at"],
            "ifis_attention_score": best["ifis_attention_score"],
            "attention_stage_status": "DISCOVERED",
            "eligible_theme_count": len(eligible),
            "best_theme": best["theme_name"],
            "theme_relevance_score": best["theme_relevance_score"],
            "theme_relevance_band": best["theme_relevance_band"],
            "theme_relevance_confidence": best["theme_relevance_confidence"],
            "candidate_class": relevance_class(best["theme_relevance_score"]),
            "float_shares": best["float_shares"],
            "buy_decision": "NOT_EVALUATED",
        })
        reason_counts["DISCOVERED"] += 1

    theme_rows.sort(key=lambda x: (x["ifis_rank"], not x["eligible_by_relevance"], -x["theme_relevance_score"], x["theme_name"]))
    candidates.sort(key=lambda x: (x["ifis_rank"], x["stock_code"]))
    summary = {
        "status": "complete",
        "stage": "IFIS_THEME_DISCOVERY_ONLY",
        "ifis_top_n": top_n,
        "min_theme_relevance": min_theme_relevance,
        "ifis_candidate_count": len(normalized_ifis),
        "discovered_count": len(candidates),
        "not_discovered_count": len(normalized_ifis) - len(candidates),
        "theme_row_count": len(theme_rows),
        "decision_counts": dict(reason_counts),
        "rule": "Discover when an IFIS top-N stock exists in the finalized Kabutan-derived 124-theme master and has at least one finalized theme relevance score of 40 or more. Minkabu is not an input. No buy/sell decision or composite trading score is created.",
        "ranking_rule": "Preserve original IFIS rank in the discovery stage. Theme relevance is a transparent membership/directness gate, not a second attention signal.",
        "lookahead_rule": "Historical validation must use theme relevance information valid for the discovery date; do not retroactively apply a future relevance master.",
    }
    return candidates, theme_rows, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ifis", required=True)
    ap.add_argument("--master", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--min-theme-relevance", type=float, default=40.0)
    args = ap.parse_args()

    candidates, themes, summary = build_discovery(
        read_csv(Path(args.ifis)),
        read_csv(Path(args.master)),
        args.top_n,
        args.min_theme_relevance,
    )
    out = Path(args.out_dir)
    candidate_fields = [
        "ifis_rank", "stock_code", "company_name", "snapshot_at", "ifis_attention_score",
        "attention_stage_status", "eligible_theme_count", "best_theme", "theme_relevance_score",
        "theme_relevance_band", "theme_relevance_confidence", "candidate_class", "float_shares", "buy_decision",
    ]
    theme_fields = [
        "ifis_rank", "stock_code", "company_name", "snapshot_at", "ifis_attention_score",
        "theme_name", "theme_relevance_score", "theme_relevance_band", "theme_relevance_confidence",
        "eligible_by_relevance", "float_shares",
    ]
    write_csv(out / "attention_discovery_candidates.csv", candidates, candidate_fields)
    write_csv(out / "attention_discovery_themes.csv", themes, theme_fields)
    out.mkdir(parents=True, exist_ok=True)
    (out / "attention_discovery_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not candidates:
        raise SystemExit("No eligible IFIS/theme candidates were discovered")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
