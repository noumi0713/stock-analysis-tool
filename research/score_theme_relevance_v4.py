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
    # 80%+ of sales is effectively maximum exposure. Preserve granularity below that.
    return min(100.0, max(0.0, share / 80.0 * 100.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dir", default="research/results/theme_relevance_batches")
    args = ap.parse_args()
    root = Path(args.dir)
    rev_path = root / f"batch_{args.batch:03d}_revenue_mix_v4.csv"
    old_path = root / f"batch_{args.batch:03d}_scores_v3.csv"
    if not old_path.exists():
        old_path = root / f"batch_{args.batch:03d}_scores_v2.csv"
    out_path = root / f"batch_{args.batch:03d}_scores_v4.csv"
    summary_path = root / f"batch_{args.batch:03d}_score_summary_v4.json"

    with rev_path.open(encoding="utf-8-sig") as f:
        revenue = {(r["stock_code"], r["theme_name"]): r for r in csv.DictReader(f)}
    old = {}
    if old_path.exists():
        with old_path.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                old[(r["stock_code"], r["theme_name"])] = r

    rows = []
    for key, r in revenue.items():
        code, theme = key
        prior = old.get(key, {})
        share_s = (r.get("revenue_share_pct") or "").strip()
        status = r.get("share_status", "")
        quote_type = prior.get("quote_type", "")

        directness = None
        growth = None
        try:
            directness = float(prior.get("current_business_score") or "")
        except Exception:
            pass
        try:
            growth = float(prior.get("growth_relevance_score") or "")
        except Exception:
            pass

        final_score = None
        confidence = "C"
        review = "revenue_share_unknown"
        score_source = "unknown"
        exposure = None

        if status == "non_operating_security":
            review = "non_operating_security"
            score_source = "non_operating_security"
        elif theme in MACRO_THEMES:
            review = "macro_sensitivity_required"
            score_source = "macro_unknown"
        elif status == "calculated_ratio_candidate":
            # Do not auto-score an inferred numerator/denominator pair until source QA confirms both values.
            review = "calculated_ratio_needs_qa"
            score_source = "calculated_candidate_only"
        elif status == "explicit_ratio_candidate" and share_s:
            try:
                share = float(share_s)
            except Exception:
                share = -1
            if 0 <= share <= 100:
                exposure = exposure_score(share)
                d = directness if directness is not None else 50.0
                g = growth if growth is not None else d
                final_score = round(0.55 * exposure + 0.30 * d + 0.15 * g, 1)
                confidence = "B"  # still candidate until source snippet is manually/arithmetic checked
                review = "explicit_ratio_needs_qa"
                score_source = "revenue_share_v4_candidate"

        rows.append({
            "batch": args.batch,
            "stock_code": code,
            "long_name": prior.get("long_name", ""),
            "theme_name": theme,
            "cluster": prior.get("cluster", ""),
            "revenue_share_pct": share_s,
            "revenue_exposure_score": "" if exposure is None else f"{exposure:.1f}",
            "directness_score": "" if directness is None else f"{directness:.1f}",
            "growth_relevance_score": "" if growth is None else f"{growth:.1f}",
            "relevance_score": "" if final_score is None else f"{final_score:.1f}",
            "band": band(final_score),
            "confidence": confidence,
            "review_flag": review,
            "score_source": score_source,
            "share_status": status,
            "share_basis": r.get("share_basis", ""),
            "source_url": r.get("source_url", ""),
            "evidence": r.get("evidence", ""),
            "quote_type": quote_type,
            "sector": prior.get("sector", ""),
            "industry": prior.get("industry", ""),
        })

    fields = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    bands: dict[str, int] = {}
    reviews: dict[str, int] = {}
    explicit = calculated = 0
    for r in rows:
        bands[r["band"]] = bands.get(r["band"], 0) + 1
        reviews[r["review_flag"]] = reviews.get(r["review_flag"], 0) + 1
        if r["share_status"] == "explicit_ratio_candidate":
            explicit += 1
        elif r["share_status"] == "calculated_ratio_candidate":
            calculated += 1
    summary = {
        "batch": args.batch,
        "stock_theme_pairs": len(rows),
        "explicit_ratio_candidates": explicit,
        "calculated_ratio_candidates": calculated,
        "unresolved_pairs": len(rows) - explicit,
        "band_counts": bands,
        "review_counts": reviews,
        "formula_after_qa": "55% revenue exposure + 30% business directness + 15% growth relevance",
        "exposure_mapping": "80%+ revenue share => 100 exposure score; linear below 80%",
        "status": "research_provisional_v4",
        "warning": "No missing revenue ratio is imputed. Calculated ratios remain unscored until QA; explicit ratios are candidate scores until source verification.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
