from __future__ import annotations

import csv
import io
import json
import random
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path("research/results/theme_relevance_batches")
SOURCE_REF = "origin/research/volume-top100-overheat"
SOURCE_PATH = "research/data/theme_members_124.csv"

TRIAL = {
    ("6654", "電力設備投資関連"): (93.0, "A", "事業の直接性と公式事業内容から主力関連"),
    ("6654", "鉄道関連"): (58.0, "B", "鉄道向け実需はあるが電力設備より中核度が低い"),
    ("7066", "デジタルトランスフォーメーション"): (76.0, "B", "DX支援を事業として展開"),
    ("7066", "生成AI"): (68.0, "B", "AIサービスを展開するが売上寄与率は非開示"),
    ("5039", "クラウドコンピューティング"): (97.0, "A", "Salesforce/クラウド支援が事業中核"),
    ("5039", "IT関連"): (94.0, "A", "ITサービスが事業中核"),
    ("5039", "デジタルトランスフォーメーション"): (93.0, "A", "クラウド導入・DX支援が事業中核"),
    ("6904", "自動車部材・部品"): (93.0, "A", "車載アンテナ等が中核事業"),
    ("6904", "自動運転車"): (66.0, "B", "CASE/自動運転関連だがテーマ単独売上比率は不明"),
    ("3653", "人工知能"): (95.0, "A", "画像処理AIが中核技術・事業"),
    ("3653", "IT関連"): (87.0, "A", "ソフトウェア・画像認識技術が中核"),
    ("3653", "デジタルトランスフォーメーション"): (82.0, "B", "画像認識AIを用いたDX用途が明確"),
    ("3653", "FA関連"): (75.0, "B", "製造・検査向け画像認識用途が明確"),
    ("3653", "自動運転車"): (69.0, "B", "車載/モビリティ向け画像認識に関連"),
    ("3653", "IoT"): (48.0, "C", "関連用途はあるが事業中核度は限定的"),
    ("3653", "ドローン"): (47.0, "C", "関連用途はあるが売上規模不明"),
    ("603A", "太陽光発電関連"): (97.0, "A", "オンサイト太陽光PPA等が中核"),
    ("603A", "再生可能エネルギー"): (97.0, "A", "再エネ事業が中核"),
    ("603A", "脱炭素"): (95.0, "A", "脱炭素ソリューションが事業中核"),
    ("603A", "電力会社"): (78.0, "B", "電力供給・エネルギーマネジメントに直接関与"),
    ("603A", "蓄電池"): (69.0, "B", "蓄電池関連事業はあるがテーマ単独比率は不明"),
    ("6745", "防災"): (96.0, "A", "火災報知・防災設備が中核事業"),
    ("8746", "金"): (90.0, "A", "金地金等の貴金属取扱いが主要事業"),
    ("7864", "機械"): (51.0, "A", "機械事業の売上構成は限定的だが公式数値で確認"),
    ("6996", "蓄電池"): (88.0, "A", "蓄電・V2H関連製品を直接展開"),
    ("6996", "電気自動車関連"): (86.0, "A", "EV充電/V2H関連製品を直接展開"),
    ("6996", "自動車部材・部品"): (75.0, "B", "車載向け電子部品に直接関与"),
    ("6996", "リチウムイオン電池"): (50.0, "B", "関連製品はあるがテーマ売上規模は限定/不明"),
    ("6996", "水素"): (12.0, "C", "テーマとの経済的直接性が弱い"),
}


def norm_code(v: str) -> str:
    s = str(v or "").strip().upper()
    m = re.search(r"([0-9]{4}|[0-9]{3}[A-Z])", s)
    return m.group(1) if m else s


def band(score: float) -> str:
    if score >= 80: return "主力テーマ"
    if score >= 60: return "有力関連"
    if score >= 40: return "補助関連"
    return "ノイズ候補"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_source() -> tuple[dict[tuple[str, str], dict[str, str]], set[str]]:
    raw = subprocess.check_output(["git", "show", f"{SOURCE_REF}:{SOURCE_PATH}"], text=True)
    rows = list(csv.DictReader(io.StringIO(raw)))
    out: dict[tuple[str, str], dict[str, str]] = {}
    stocks: set[str] = set()
    for r in rows:
        code = norm_code(r.get("stock_code", ""))
        theme = r.get("\ufefftheme_name", "") or r.get("theme_name", "")
        if not code or not theme:
            continue
        out[(code, theme)] = r
        stocks.add(code)
    return out, stocks


def load_discretionary() -> dict[tuple[str, str], dict[str, str]]:
    files = [
        ROOT / "discretionary_review_1000_final.csv",
        ROOT / "discretionary_review_1000_set2_final.csv",
        ROOT / "discretionary_review_1000_set3_final.csv",
    ]
    out: dict[tuple[str, str], dict[str, str]] = {}
    for p in files:
        for r in read_csv(p):
            out[(norm_code(r["stock_code"]), r["theme_name"])] = r
    return out


def load_explicit_qa() -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for r in read_csv(ROOT / "batch_001_final_qa_v6.csv"):
        key = (norm_code(r["stock_code"]), r["theme_name"])
        out[key] = {
            "score": r["final_relevance_score"],
            "band": r["final_band"],
            "confidence": "A" if r.get("decision") == "OVERRIDE" else "B",
            "source": "user_approved_manual_override" if r.get("decision") == "OVERRIDE" else "user_approved_structured_v6",
            "reason": r.get("reason", ""),
            "url": r.get("source_url", ""),
            "revenue_share_pct": "",
        }
    for r in read_csv(ROOT / "batch_002_final_qa.csv"):
        key = (norm_code(r["stock_code"]), r["theme_name"])
        share = "" if r.get("decision") == "MANUAL_OVERRIDE" else r.get("auto_revenue_share_pct", "")
        out[key] = {
            "score": r["final_relevance_score"],
            "band": r["final_band"],
            "confidence": r.get("confidence", "B"),
            "source": "manual_official_business_override" if r.get("decision") == "MANUAL_OVERRIDE" else "approved_structured_v6",
            "reason": r.get("decision_basis", ""),
            "url": "",
            "revenue_share_pct": share,
        }
    return out


def main() -> None:
    source, source_stocks = load_source()
    discretionary = load_discretionary()
    explicit = load_explicit_qa()

    v3: dict[tuple[str, str], dict[str, str]] = {}
    v6: dict[tuple[str, str], dict[str, str]] = {}
    key_batch: dict[tuple[str, str], str] = {}
    for b in range(1, 12):
        p3 = ROOT / f"batch_{b:03d}_scores_v3.csv"
        p6 = ROOT / f"batch_{b:03d}_scores_v6.csv"
        if not p3.exists() or not p6.exists():
            raise SystemExit(f"Missing batch {b} score files")
        for r in read_csv(p3):
            key = (norm_code(r["stock_code"]), r["theme_name"])
            v3[key] = r; key_batch[key] = str(b)
        for r in read_csv(p6):
            key = (norm_code(r["stock_code"]), r["theme_name"])
            v6[key] = r; key_batch[key] = str(b)

    trial_source_keys = {k for k in source if k[0] in {x[0] for x in TRIAL}}
    if set(TRIAL) != trial_source_keys:
        missing = sorted(trial_source_keys - set(TRIAL))
        extra = sorted(set(TRIAL) - trial_source_keys)
        raise SystemExit(f"Trial mapping mismatch missing={missing} extra={extra}")

    out: list[dict[str, str]] = []
    missing_scores: list[tuple[str, str]] = []
    for key, sr in sorted(source.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        code, theme = key
        cluster = sr.get("cluster", "")
        company = sr.get("company_name", "")
        revenue_share = ""
        source_url = sr.get("source_url", "")
        review_flag = ""
        segment_names = ""

        if key in TRIAL:
            score, conf, reason = TRIAL[key]
            final_band = band(score)
            decision_source = "trial_chatgpt_research_v0_1"
            batch_no = "0"
        elif key in explicit:
            q = explicit[key]
            score = float(q["score"]); conf = q["confidence"]; final_band = q["band"]
            decision_source = q["source"]; reason = q["reason"]
            source_url = q["url"] or source_url; revenue_share = q["revenue_share_pct"]
            batch_no = key_batch.get(key, "")
        elif key in discretionary:
            d = discretionary[key]
            score = float(d["final_relevance_score"]); conf = d.get("final_confidence", "C")
            final_band = d.get("final_band") or band(score)
            decision_source = d.get("decision_source", "chatgpt_discretion_existing_context")
            reason = d.get("decision_reason", "")
            source_url = d.get("source_url", "") or source_url
            revenue_share = d.get("revenue_share_pct", "")
            batch_no = d.get("batch", "") or key_batch.get(key, "")
        else:
            p = v3.get(key)
            s = v6.get(key)
            if p is None:
                missing_scores.append(key); continue
            try:
                prior = float(p.get("relevance_score") or "")
            except Exception:
                missing_scores.append(key); continue
            review_flag = (s or {}).get("review_flag", "")
            segment_names = (s or {}).get("segment_names", "")
            if s and s.get("relevance_score"):
                score = float(s["relevance_score"])
                final_band = s.get("band") or band(score)
                conf = "B"
                decision_source = "structured_revenue_v6"
                reason = s.get("evidence", "") or "整合性確認済みの公式セグメント売上と事業直接性・成長性を使用"
                revenue_share = s.get("revenue_share_pct", "")
                source_url = s.get("source_url", "") or source_url
            else:
                score = prior
                final_band = p.get("band") or band(score)
                conf = p.get("confidence") or "C"
                if review_flag == "macro_sensitivity_required":
                    decision_source = "chatgpt_discretion_macro_context"
                elif conf in ("A", "B"):
                    decision_source = "chatgpt_confirmed_official_context"
                else:
                    decision_source = "chatgpt_discretion_existing_context"
                reason = p.get("evidence", "") or "会社公式情報・事業概要・テーマ直接性を基に裁量確定"
                source_url = p.get("official_urls", "") or p.get("website", "") or source_url
            batch_no = key_batch.get(key, "")

        quality = "A" if conf == "A" else "B" if conf == "B" else "C"
        out.append({
            "batch": batch_no,
            "stock_code": code,
            "company_name": company,
            "theme_name": theme,
            "cluster": cluster,
            "revenue_share_pct": revenue_share,
            "final_relevance_score": f"{score:.1f}",
            "final_band": final_band,
            "final_confidence": conf,
            "decision_source": decision_source,
            "decision_reason": reason,
            "source_url": source_url,
            "original_review_flag": review_flag,
            "segment_names": segment_names,
            "quality_tier": quality,
            "status": "finalized",
        })

    if missing_scores:
        raise SystemExit(f"Missing scores for {len(missing_scores)} rows: {missing_scores[:20]}")

    final_keys = {(r["stock_code"], r["theme_name"]) for r in out}
    source_keys = set(source)
    if final_keys != source_keys:
        raise SystemExit(f"Key mismatch missing={len(source_keys-final_keys)} extra={len(final_keys-source_keys)}")
    if len(final_keys) != len(out):
        raise SystemExit("Duplicate final keys")

    out_path = ROOT / "theme_relevance_full_final.csv"
    fields = list(out[0].keys())
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)

    summary = {
        "status": "complete",
        "source_memberships": len(source_keys),
        "finalized_rows": len(out),
        "unique_stocks": len({r['stock_code'] for r in out}),
        "source_unique_stocks": len(source_stocks),
        "trial_rows": sum(r["batch"] == "0" for r in out),
        "unfinalized": sum(r["status"] != "finalized" for r in out),
        "key_match_source": final_keys == source_keys,
        "bands": dict(Counter(r["final_band"] for r in out)),
        "confidence": dict(Counter(r["final_confidence"] for r in out)),
        "decision_sources": dict(Counter(r["decision_source"] for r in out)),
        "batch_counts": dict(Counter(r["batch"] for r in out)),
        "revenue_share_disclosed_rows": sum(bool(r["revenue_share_pct"]) for r in out),
        "lower_confidence_rows": sum(r["final_confidence"] == "C" for r in out),
        "rule": "Disclosed revenue share is never fabricated. Structured official revenue takes precedence; otherwise official-company context and ChatGPT discretion finalize relevance 0-100. Existing user-approved/manual QA overrides are preserved.",
    }
    (ROOT / "theme_relevance_full_final_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # QA: all manual/explicit decisions + random 200 + random 100 lower-confidence rows.
    important = [r for r in out if r["decision_source"] in {
        "user_approved_manual_override", "manual_official_business_override", "trial_chatgpt_research_v0_1", "chatgpt_discretion_manual_web"
    }]
    rest = [r for r in out if r not in important]
    low = [r for r in rest if r["final_confidence"] == "C"]
    rng = random.Random(20260905)
    sample = important + rng.sample(rest, min(200, len(rest))) + rng.sample(low, min(100, len(low)))
    seen = set(); qa = []
    for r in sample:
        k = (r["stock_code"], r["theme_name"])
        if k not in seen:
            seen.add(k); qa.append(r)
    with (ROOT / "theme_relevance_full_qa_sample.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(qa)

    # Theme-level quality overview.
    by_theme: dict[str, list[dict[str, str]]] = {}
    for r in out: by_theme.setdefault(r["theme_name"], []).append(r)
    stat_fields = ["theme_name","rows","avg_score","main_count","strong_count","support_count","noise_count","A","B","C"]
    stats = []
    for theme, rs in sorted(by_theme.items()):
        scores = [float(r["final_relevance_score"]) for r in rs]
        bc = Counter(r["final_band"] for r in rs); cc = Counter(r["final_confidence"] for r in rs)
        stats.append({
            "theme_name": theme, "rows": len(rs), "avg_score": f"{sum(scores)/len(scores):.1f}",
            "main_count": bc["主力テーマ"], "strong_count": bc["有力関連"], "support_count": bc["補助関連"], "noise_count": bc["ノイズ候補"],
            "A": cc["A"], "B": cc["B"], "C": cc["C"],
        })
    with (ROOT / "theme_relevance_full_theme_stats.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=stat_fields); w.writeheader(); w.writerows(stats)

    print(json.dumps(summary, ensure_ascii=False))
    print("qa_rows", len(qa), "themes", len(stats))


if __name__ == "__main__":
    main()
