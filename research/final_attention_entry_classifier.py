from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

BUY_NOW = "BUY_NOW"
WAIT_FIRST_PULLBACK = "WAIT_FIRST_PULLBACK"
OVERHEAT_SKIP = "OVERHEAT_SKIP"
LABEL_JA = {BUY_NOW: "今買う", WAIT_FIRST_PULLBACK: "初押し待ち", OVERHEAT_SKIP: "過熱・見送り"}
TECH_HARD_REJECT = {"INVALIDATED", "EXCLUDE_RAN_AWAY", "FETCH_ERROR", "NO_DISCOVERY_BAR", "INSUFFICIENT_HISTORY"}
TECH_ACTIONABLE = {"BUY_NOW", "FIRST_PULLBACK_SIGNAL"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))


def as_float(value, default=None):
    try:
        if value is None or str(value).strip() == "": return default
        return float(value)
    except Exception: return default


def metric(metrics: dict, name: str, default=None): return as_float(metrics.get(name), default)


def sequence_proxy(discovery_metrics: dict) -> str:
    r1 = metric(discovery_metrics, "return_1d_pct", 0.0); r5 = metric(discovery_metrics, "return_5d_pct", 0.0)
    dev = metric(discovery_metrics, "ma25_deviation_pct", 0.0); rsi = metric(discovery_metrics, "rsi14", 50.0); vr = metric(discovery_metrics, "volume_ratio20", 1.0)
    if r1 >= 7.0 or r5 >= 12.0 or dev >= 12.0 or rsi >= 75.0 or vr >= 5.0: return "PRICE_MOVE_PRECEDED_ATTENTION_OR_LATE"
    if r5 <= 7.0 and dev <= 8.0 and rsi <= 68.0: return "ATTENTION_BEFORE_PRICE_OVERHEAT"
    return "AMBIGUOUS"


def extreme_overheat(latest: dict) -> list[str]:
    reasons = []
    for key, threshold, label in [("rsi14",80.0,"RSI >= 80"),("ma25_deviation_pct",15.0,"25MA deviation >= 15%"),("return_10d_pct",25.0,"10-day return >= 25%"),("return_20d_pct",40.0,"20-day return >= 40%"),("volume_ratio20",8.0,"volume >= 8x 20-day average"),("upper_wick_ratio",0.50,"long upper wick")]:
        v = metric(latest, key, None)
        if v is not None and v >= threshold: reasons.append(label)
    turnover = metric(latest, "float_turnover_pct", None)
    if turnover is not None and turnover >= 100.0: reasons.append("daily float turnover >= 100%")
    return reasons


def can_buy_now(material: dict, tech_status: str, latest: dict):
    if material.get("material_class") != "STRONG": return False, ["strong positive material is required"]
    if tech_status not in TECH_ACTIONABLE: return False, [f"technical state {tech_status} is not actionable"]
    conditions = [(metric(latest,"rsi14",100.0)<=68.0,"RSI <= 68"),(metric(latest,"ma25_deviation_pct",999.0)<=8.0,"25MA deviation <= 8%"),(metric(latest,"return_5d_pct",999.0)<=10.0,"5-day return <= 10%"),(metric(latest,"return_20d_pct",999.0)<=25.0,"20-day return <= 25%"),(metric(latest,"volume_ratio20",0.0)>=1.2,"volume >= 1.2x 20-day average"),(metric(latest,"volume_ratio20",999.0)<=5.0,"volume <= 5x 20-day average"),(metric(latest,"upper_wick_ratio",1.0)<=0.35,"upper wick is controlled")]
    turnover = metric(latest, "float_turnover_pct", None)
    if turnover is not None: conditions.append((turnover <= 60.0, "float turnover <= 60%"))
    failed = [label for ok, label in conditions if not ok]
    return not failed, failed


def classify_one(discovery: dict, material: dict, tech: dict) -> dict:
    tech_status = str(tech.get("technical_status") or "UNKNOWN"); latest = tech.get("latest_metrics") or {}; discovery_metrics = tech.get("discovery_metrics") or {}
    seq = sequence_proxy(discovery_metrics); extreme = extreme_overheat(latest); reasons = []
    if material.get("review_status") != "REVIEWED": entry = OVERHEAT_SKIP; reasons.append("material review is incomplete")
    elif material.get("material_class") in {"NEGATIVE","NONE","LOOKAHEAD_REJECT","WEAK"}: entry = OVERHEAT_SKIP; reasons.append(f"material class {material.get('material_class')} does not support a long entry")
    elif tech_status in TECH_HARD_REJECT: entry = OVERHEAT_SKIP; reasons.append(f"technical state {tech_status} is a hard reject")
    elif extreme: entry = OVERHEAT_SKIP; reasons.extend(extreme)
    else:
        buyable, failed = can_buy_now(material, tech_status, latest)
        if buyable:
            entry = BUY_NOW; reasons.append("strong material + cross-source attention + limited price overheat")
            if seq == "ATTENTION_BEFORE_PRICE_OVERHEAT": reasons.append("attention/price sequence proxy is favorable")
        else:
            entry = WAIT_FIRST_PULLBACK; reasons.append("stock/material can remain valid, but current entry timing is not clean enough"); reasons.extend(failed[:4])
            if seq == "PRICE_MOVE_PRECEDED_ATTENTION_OR_LATE": reasons.append("attention may be late relative to the price move")
    overheat = metric(latest, "overheat_score", 999.0); ma_dev = metric(latest, "ma25_deviation_pct", 999.0); ret5 = metric(latest, "return_5d_pct", 999.0)
    return {"ifis_rank": int(float(discovery.get("ifis_rank") or 999999)), "stock_code": discovery.get("stock_code", ""), "company_name": discovery.get("company_name", ""), "best_theme": discovery.get("best_theme", ""), "minkabu_attention_type": discovery.get("minkabu_attention_type", ""), "minkabu_theme_rank": discovery.get("minkabu_theme_rank", ""), "minkabu_relevance": discovery.get("minkabu_relevance", ""), "theme_relevance_score": discovery.get("theme_relevance_score", ""), "material_class": material.get("material_class", ""), "material_continuity": material.get("material_continuity", ""), "catalyst_type": material.get("catalyst_type", ""), "catalyst_summary": material.get("catalyst_summary", ""), "technical_status": tech_status, "attention_price_sequence_proxy": seq, "entry_class": entry, "entry_label_ja": LABEL_JA[entry], "entry_reasons": " | ".join(reasons), "latest_market_date": tech.get("latest_market_date", ""), "latest_close": tech.get("latest_close", ""), "rsi14": metric(latest,"rsi14",None), "volume_ratio20": metric(latest,"volume_ratio20",None), "ma25_deviation_pct": ma_dev if ma_dev != 999.0 else None, "atr14_pct": metric(latest,"atr14_pct",None), "return_1d_pct": metric(latest,"return_1d_pct",None), "return_5d_pct": ret5 if ret5 != 999.0 else None, "return_10d_pct": metric(latest,"return_10d_pct",None), "return_20d_pct": metric(latest,"return_20d_pct",None), "upper_wick_ratio": metric(latest,"upper_wick_ratio",None), "distance_from_20d_high_pct": metric(latest,"distance_from_20d_high_pct",None), "float_turnover_pct": metric(latest,"float_turnover_pct",None), "overheat_score": overheat if overheat != 999.0 else None, "actionability": "RESEARCH_ONLY"}


def combine(discovery_rows, material_rows, technical: dict):
    mats = {str(x.get("stock_code") or "").upper(): x for x in material_rows}; techs = {str(x.get("stock_code") or "").upper(): x for x in technical.get("items", [])}; out = []
    for d in discovery_rows:
        if d.get("attention_stage_status") != "DISCOVERED": continue
        code = str(d.get("stock_code") or "").upper(); mat = mats.get(code, {"review_status":"MISSING_REVIEW","material_class":"NONE","material_continuity":"NONE"}); tech = techs.get(code, {"technical_status":"FETCH_ERROR"})
        out.append(classify_one(d, mat, tech))
    class_order = {BUY_NOW:0, WAIT_FIRST_PULLBACK:1, OVERHEAT_SKIP:2}
    def safe(v, fallback=999.0):
        x = as_float(v, None); return fallback if x is None else max(x, 0.0)
    out.sort(key=lambda x: (class_order[x["entry_class"]], safe(x.get("overheat_score")), safe(x.get("ma25_deviation_pct")), safe(x.get("return_5d_pct")), x["ifis_rank"]))
    for i, row in enumerate(out, 1): row["timing_rank"] = i
    return out


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--discovery", required=True); ap.add_argument("--material", required=True); ap.add_argument("--technical", required=True); ap.add_argument("--out-dir", required=True); args = ap.parse_args()
    rows = combine(read_csv(Path(args.discovery)), read_csv(Path(args.material)), json.loads(Path(args.technical).read_text(encoding="utf-8"))); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for r in rows: counts[r["entry_class"]] = counts.get(r["entry_class"], 0) + 1
    payload = {"status":"complete","actionability":"RESEARCH_ONLY","final_classes":LABEL_JA,"count":len(rows),"class_counts":counts,"ranking_rule":"No composite buy score. Rank by final class, then lower technical overheat, lower positive 25MA deviation, lower 5-day price rise, then original IFIS rank. Attention magnitude itself is not rewarded twice.","causality_warning":"attention_price_sequence_proxy is a timing proxy, not proof of causality. It flags likely late attention when the price/volume move was already large at discovery.","items":rows}
    (out / "attention_entry_final.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(rows[0].keys()) if rows else []
    with (out / "attention_entry_final.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(json.dumps({"status":"complete","count":len(rows),"class_counts":counts}, ensure_ascii=False))


if __name__ == "__main__": main()
