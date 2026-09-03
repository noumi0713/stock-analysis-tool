from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MACRO_THEMES = {"円高メリット", "円安メリット", "金利上昇メリット", "生活防衛", "インバウンド"}


def band(score: float | None) -> str:
    if score is None:
        return "未確定"
    if score >= 80:
        return "主力テーマ"
    if score >= 60:
        return "有力関連"
    if score >= 40:
        return "補助関連"
    return "ノイズ候補"


def exposure_score(share: float) -> float:
    return min(100.0, max(0.0, share / 80.0 * 100.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dir", default="research/results/theme_relevance_batches")
    args = ap.parse_args()
    root = Path(args.dir)
    rev_path = root / f"batch_{args.batch:03d}_revenue_mix_v5.csv"
    prior_path = root / f"batch_{args.batch:03d}_scores_v3.csv"
    if not prior_path.exists():
        prior_path = root / f"batch_{args.batch:03d}_scores_v2.csv"
    out_path = root / f"batch_{args.batch:03d}_scores_v5.csv"
    summary_path = root / f"batch_{args.batch:03d}_score_summary_v5.json"

    with rev_path.open(encoding="utf-8-sig") as f:
        rev_rows = list(csv.DictReader(f))
    prior = {}
    if prior_path.exists():
        with prior_path.open(encoding="utf-8-sig") as f:
            prior = {(r["stock_code"], r["theme_name"]): r for r in csv.DictReader(f)}

    out = []
    for r in rev_rows:
        key = (r["stock_code"], r["theme_name"])
        p = prior.get(key, {})
        share = None
        try:
            share = float(r.get("revenue_share_pct") or "")
        except Exception:
            pass
        direct = None
        growth = None
        try:
            direct = float(p.get("current_business_score") or "")
        except Exception:
            pass
        try:
            growth = float(p.get("growth_relevance_score") or "")
        except Exception:
            pass

        score = None
        exposure = None
        confidence = "C"
        review = "revenue_share_unknown"
        source = "unknown"
        if r.get("share_status") == "non_operating_security":
            review = "non_operating_security"
            source = "non_operating_security"
        elif r["theme_name"] in MACRO_THEMES:
            review = "macro_sensitivity_required"
            source = "macro_unknown"
        elif r.get("share_status") == "structured_segment_candidate" and share is not None:
            exposure = exposure_score(share)
            d = direct if direct is not None else 50.0
            g = growth if growth is not None else d
            score = round(0.55 * exposure + 0.30 * d + 0.15 * g, 1)
            confidence = "B"
            review = "structured_segment_needs_sample_qa"
            source = "structured_segment_v5"

        out.append({
            "batch": args.batch,
            "stock_code": r["stock_code"],
            "long_name": p.get("long_name", ""),
            "theme_name": r["theme_name"],
            "cluster": p.get("cluster", ""),
            "revenue_share_pct": r.get("revenue_share_pct", ""),
            "revenue_exposure_score": "" if exposure is None else f"{exposure:.1f}",
            "directness_score": "" if direct is None else f"{direct:.1f}",
            "growth_relevance_score": "" if growth is None else f"{growth:.1f}",
            "relevance_score": "" if score is None else f"{score:.1f}",
            "band": band(score),
            "confidence": confidence,
            "review_flag": review,
            "score_source": source,
            "segment_names": r.get("segment_names", ""),
            "share_basis": r.get("share_basis", ""),
            "source_url": r.get("source_url", ""),
            "evidence": r.get("evidence", ""),
        })

    fields = list(out[0].keys()) if out else []
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)

    bands: dict[str, int] = {}
    reviews: dict[str, int] = {}
    scored = 0
    for r in out:
        bands[r["band"]] = bands.get(r["band"], 0) + 1
        reviews[r["review_flag"]] = reviews.get(r["review_flag"], 0) + 1
        if r["relevance_score"]:
            scored += 1
    summary = {
        "batch": args.batch,
        "stock_theme_pairs": len(out),
        "scored_pairs": scored,
        "unscored_pairs": len(out) - scored,
        "band_counts": bands,
        "review_counts": reviews,
        "formula": "55% structured revenue exposure + 30% business directness + 15% growth relevance",
        "status": "research_candidate_v5",
        "warning": "v5 only scores themes mapped to disclosed revenue segments; specific themes do not inherit broad-segment revenue.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
