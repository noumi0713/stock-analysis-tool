from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


def norm_code(value: str) -> str:
    s = str(value or "").strip().upper()
    s = re.sub(r"\.(T|JP|S|N|F)$", "", s)
    m = re.search(r"([0-9]{4}|[0-9]{3}[A-Z])", s)
    return m.group(1) if m else s


def norm_theme(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or "")).strip())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def as_int(value, default=999999) -> int:
    try: return int(float(str(value).strip()))
    except Exception: return default


def as_float(value, default=None):
    try:
        s = str(value).strip()
        return float(s) if s else default
    except Exception:
        return default


def normalize_attention_type(value: str) -> str:
    s = str(value or "").strip().upper()
    return {"人気":"POPULAR","POPULAR":"POPULAR","急上昇":"RISING","RISING":"RISING","SURGING":"RISING"}.get(s, s)


def relevance_class(score: float | None) -> str:
    if score is None: return "UNMAPPED"
    if score >= 80: return "CORE"
    if score >= 60: return "STRONG"
    if score >= 40: return "SUPPORT"
    return "NOISE"


def load_master(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    out = {}
    for r in rows:
        if (r.get("status") or "finalized") != "finalized": continue
        code, theme_key = norm_code(r.get("stock_code", "")), norm_theme(r.get("theme_name", ""))
        if not code or not theme_key: continue
        key = (code, theme_key); score = as_float(r.get("final_relevance_score"), -1.0); prev = out.get(key)
        if prev is None or score > as_float(prev.get("final_relevance_score"), -1.0): out[key] = r
    return out


def build_discovery(ifis_rows, minkabu_rows, master_rows, top_n: int = 30, min_theme_relevance: float = 40.0):
    master = load_master(master_rows)
    by_code: dict[str, list[dict[str, str]]] = {}
    for r in minkabu_rows:
        code = norm_code(r.get("stock_code", "")); attn_type = normalize_attention_type(r.get("attention_type", "") or r.get("theme_attention_type", ""))
        if not code or attn_type not in {"POPULAR", "RISING"}: continue
        x = dict(r); x["_attention_type"] = attn_type; by_code.setdefault(code, []).append(x)

    normalized_ifis, seen = [], set()
    for i, r in enumerate(ifis_rows, 1):
        rank = as_int(r.get("ifis_rank"), i); code = norm_code(r.get("stock_code", "") or r.get("code", "") or r.get("ticker", ""))
        if code and rank <= top_n and code not in seen:
            seen.add(code); normalized_ifis.append((rank, code, r))
    normalized_ifis.sort(key=lambda x: (x[0], x[1]))

    matches, candidates = [], []
    for rank, code, ifis in normalized_ifis:
        eligible = []
        for m in by_code.get(code, []):
            theme_name = str(m.get("theme_name", "")).strip(); master_row = master.get((code, norm_theme(theme_name)))
            if master_row is None: continue
            score = as_float(master_row.get("final_relevance_score"), 0.0); is_eligible = score >= min_theme_relevance
            row = {
                "ifis_rank": rank, "stock_code": code,
                "company_name": (ifis.get("company_name") or m.get("company_name") or master_row.get("company_name") or "").strip(),
                "snapshot_at": (ifis.get("snapshot_at") or "").strip(), "ifis_attention_score": (ifis.get("attention_score") or "").strip(),
                "theme_name": theme_name or master_row.get("theme_name", ""), "minkabu_attention_type": m["_attention_type"],
                "minkabu_theme_rank": as_int(m.get("theme_rank"), 999999), "minkabu_relevance": as_float(m.get("minkabu_relevance"), None),
                "theme_price_change_1d_pct": as_float(m.get("theme_price_change_1d_pct"), None), "theme_price_change_5d_pct": as_float(m.get("theme_price_change_5d_pct"), None),
                "theme_relevance_score": score, "theme_relevance_band": master_row.get("final_band", ""),
                "theme_relevance_confidence": master_row.get("final_confidence", ""), "kabutan_member": True,
                "eligible_by_relevance": is_eligible, "float_shares": as_float(ifis.get("float_shares"), None),
            }
            matches.append(row)
            if is_eligible: eligible.append(row)
        eligible.sort(key=lambda x: (0 if x["minkabu_attention_type"] == "RISING" else 1, x["minkabu_theme_rank"], -x["theme_relevance_score"], -(x["minkabu_relevance"] if x["minkabu_relevance"] is not None else -1.0), x["theme_name"]))
        if eligible:
            best = eligible[0]
            candidates.append({
                "ifis_rank": rank, "stock_code": code, "company_name": best["company_name"], "snapshot_at": best["snapshot_at"],
                "ifis_attention_score": best["ifis_attention_score"], "attention_stage_status": "DISCOVERED", "cross_source_theme_count": len(eligible),
                "best_theme": best["theme_name"], "minkabu_attention_type": best["minkabu_attention_type"], "minkabu_theme_rank": best["minkabu_theme_rank"],
                "minkabu_relevance": best["minkabu_relevance"], "theme_price_change_1d_pct": best["theme_price_change_1d_pct"], "theme_price_change_5d_pct": best["theme_price_change_5d_pct"],
                "theme_relevance_score": best["theme_relevance_score"], "theme_relevance_band": best["theme_relevance_band"],
                "theme_relevance_confidence": best["theme_relevance_confidence"], "candidate_class": relevance_class(best["theme_relevance_score"]),
                "kabutan_member": True, "float_shares": best["float_shares"], "buy_decision": "NOT_EVALUATED",
            })
    matches.sort(key=lambda x: (x["ifis_rank"], not x["eligible_by_relevance"], 0 if x["minkabu_attention_type"] == "RISING" else 1, x["minkabu_theme_rank"], -x["theme_relevance_score"]))
    candidates.sort(key=lambda x: (x["ifis_rank"], x["stock_code"]))
    summary = {
        "status":"complete", "stage":"ATTENTION_DISCOVERY_ONLY", "ifis_top_n":top_n, "min_theme_relevance":min_theme_relevance,
        "ifis_candidate_count":len(normalized_ifis), "discovered_count":len(candidates), "cross_source_match_count":len(matches),
        "eligible_cross_source_match_count":sum(1 for x in matches if x["eligible_by_relevance"]),
        "rejected_noise_match_count":sum(1 for x in matches if not x["eligible_by_relevance"]),
        "attention_type_counts":dict(Counter(x["minkabu_attention_type"] for x in matches)),
        "rule":"Discover only when an IFIS top-N stock appears in a Minkabu POPULAR/RISING theme, the same stock-theme pair exists in the finalized Kabutan-derived 124-theme master, and finalized theme relevance is at least 40. Noise matches remain in comparison logs. No buy/sell decision or composite trading score is created.",
        "ranking_rule":"Preserve IFIS rank at discovery stage; attention signals are not treated as independent buy confirmations.",
        "lookahead_rule":"Use only snapshots and theme membership/relevance information available at the discovery timestamp.",
    }
    return candidates, matches, summary


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--ifis", required=True); ap.add_argument("--minkabu", required=True); ap.add_argument("--master", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n", type=int, default=30); ap.add_argument("--min-theme-relevance", type=float, default=40.0); args = ap.parse_args()
    candidates, matches, summary = build_discovery(read_csv(Path(args.ifis)), read_csv(Path(args.minkabu)), read_csv(Path(args.master)), args.top_n, args.min_theme_relevance)
    out = Path(args.out_dir)
    candidate_fields = ["ifis_rank","stock_code","company_name","snapshot_at","ifis_attention_score","attention_stage_status","cross_source_theme_count","best_theme","minkabu_attention_type","minkabu_theme_rank","minkabu_relevance","theme_price_change_1d_pct","theme_price_change_5d_pct","theme_relevance_score","theme_relevance_band","theme_relevance_confidence","candidate_class","kabutan_member","float_shares","buy_decision"]
    match_fields = ["ifis_rank","stock_code","company_name","snapshot_at","ifis_attention_score","theme_name","minkabu_attention_type","minkabu_theme_rank","minkabu_relevance","theme_price_change_1d_pct","theme_price_change_5d_pct","theme_relevance_score","theme_relevance_band","theme_relevance_confidence","kabutan_member","eligible_by_relevance","float_shares"]
    write_csv(out / "attention_discovery_candidates.csv", candidates, candidate_fields); write_csv(out / "attention_discovery_matches.csv", matches, match_fields); out.mkdir(parents=True, exist_ok=True)
    (out / "attention_discovery_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not candidates: raise SystemExit("No eligible cross-source attention candidates were discovered")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
