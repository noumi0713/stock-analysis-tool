from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def norm_code(value: str) -> str:
    s = str(value or "").strip().upper()
    s = re.sub(r"\.(T|JP)$", "", s)
    m = re.search(r"(?:^|\D)([0-9]{4}|[0-9]{3}[A-Z])(?:$|\D)", s)
    if m:
        return m.group(1)
    m = re.search(r"([0-9]{4}|[0-9]{3}[A-Z])", s)
    return m.group(1) if m else s


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def as_int(value: str, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def as_float(value: str, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return default


def candidate_class(best_score: float | None) -> str:
    if best_score is None:
        return "UNMAPPED"
    if best_score >= 80:
        return "CORE"
    if best_score >= 60:
        return "STRONG"
    if best_score >= 40:
        return "SUPPORT"
    return "NOISE"


def normalize_input(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for i, r in enumerate(rows, 1):
        code = norm_code(r.get("stock_code", "") or r.get("code", "") or r.get("ticker", ""))
        if not code or code in seen:
            continue
        seen.add(code)
        rank = as_int(r.get("ifis_rank", ""), i)
        out.append({
            "ifis_rank": str(rank),
            "stock_code": code,
            "company_name": (r.get("company_name", "") or r.get("name", "")).strip(),
            "snapshot_at": (r.get("snapshot_at", "") or r.get("snapshot_time", "")).strip(),
            "attention_score": (r.get("attention_score", "") or "").strip(),
            "input_note": (r.get("input_note", "") or r.get("note", "") or "").strip(),
        })
    return sorted(out, key=lambda x: (as_int(x["ifis_rank"], 10**9), x["stock_code"]))


def load_master(path: Path) -> dict[str, list[dict[str, str]]]:
    by_code: dict[str, list[dict[str, str]]] = {}
    for r in read_csv(path):
        code = norm_code(r.get("stock_code", ""))
        if not code:
            continue
        if r.get("status", "finalized") != "finalized":
            continue
        by_code.setdefault(code, []).append(r)
    for rows in by_code.values():
        rows.sort(
            key=lambda r: (
                -as_float(r.get("final_relevance_score", "0")),
                r.get("theme_name", ""),
            )
        )
    return by_code


def run_filter(input_path: Path, master_path: Path, out_dir: Path) -> dict[str, object]:
    candidates = normalize_input(read_csv(input_path))
    master = load_master(master_path)

    candidate_rows: list[dict[str, object]] = []
    theme_rows: list[dict[str, object]] = []
    class_counts: Counter[str] = Counter()
    snapshots: set[str] = set()

    for c in candidates:
        themes = master.get(c["stock_code"], [])
        if c["snapshot_at"]:
            snapshots.add(c["snapshot_at"])

        scores = [as_float(t.get("final_relevance_score", "0")) for t in themes]
        best = themes[0] if themes else None
        best_score = as_float(best.get("final_relevance_score", "0")) if best else None
        klass = candidate_class(best_score)
        class_counts[klass] += 1

        band_counts = Counter(t.get("final_band", "") for t in themes)
        resolved_name = c["company_name"] or (best.get("company_name", "") if best else "")
        theme_summary = " | ".join(
            f'{t.get("theme_name", "")}:{as_float(t.get("final_relevance_score", "0")):.1f}'
            for t in themes
        )

        candidate_rows.append({
            **c,
            "company_name": resolved_name,
            "candidate_class": klass,
            "best_theme": best.get("theme_name", "") if best else "",
            "best_relevance_score": f"{best_score:.1f}" if best_score is not None else "",
            "best_band": best.get("final_band", "") if best else "",
            "best_confidence": best.get("final_confidence", "") if best else "",
            "theme_count": len(themes),
            "core_theme_count": band_counts.get("主力テーマ", 0),
            "strong_theme_count": band_counts.get("有力関連", 0),
            "support_theme_count": band_counts.get("補助関連", 0),
            "noise_theme_count": band_counts.get("ノイズ候補", 0),
            "theme_score_mean": f"{sum(scores) / len(scores):.1f}" if scores else "",
            "all_themes_sorted": theme_summary,
            "mapped_to_master": "true" if themes else "false",
        })

        for position, t in enumerate(themes, 1):
            theme_rows.append({
                "ifis_rank": c["ifis_rank"],
                "stock_code": c["stock_code"],
                "company_name": resolved_name,
                "snapshot_at": c["snapshot_at"],
                "candidate_class": klass,
                "theme_rank_within_stock": position,
                "theme_name": t.get("theme_name", ""),
                "cluster": t.get("cluster", ""),
                "relevance_score": t.get("final_relevance_score", ""),
                "band": t.get("final_band", ""),
                "confidence": t.get("final_confidence", ""),
                "decision_source": t.get("decision_source", ""),
                "revenue_share_pct": t.get("revenue_share_pct", ""),
                "decision_reason": t.get("decision_reason", ""),
                "source_url": t.get("source_url", ""),
            })

    candidate_fields = [
        "ifis_rank", "stock_code", "company_name", "snapshot_at", "attention_score", "input_note",
        "candidate_class", "best_theme", "best_relevance_score", "best_band", "best_confidence",
        "theme_count", "core_theme_count", "strong_theme_count", "support_theme_count", "noise_theme_count",
        "theme_score_mean", "all_themes_sorted", "mapped_to_master",
    ]
    theme_fields = [
        "ifis_rank", "stock_code", "company_name", "snapshot_at", "candidate_class",
        "theme_rank_within_stock", "theme_name", "cluster", "relevance_score", "band", "confidence",
        "decision_source", "revenue_share_pct", "decision_reason", "source_url",
    ]
    write_csv(out_dir / "latest_candidates.csv", candidate_rows, candidate_fields)
    write_csv(out_dir / "latest_candidate_themes.csv", theme_rows, theme_fields)

    mapped = sum(1 for r in candidate_rows if r["mapped_to_master"] == "true")
    summary: dict[str, object] = {
        "status": "complete",
        "input_file": str(input_path),
        "master_file": str(master_path),
        "candidate_count": len(candidate_rows),
        "mapped_candidates": mapped,
        "unmapped_candidates": len(candidate_rows) - mapped,
        "candidate_class_counts": dict(sorted(class_counts.items())),
        "snapshot_at_values": sorted(snapshots),
        "filter_rule": {
            "CORE": "best theme relevance >= 80",
            "STRONG": "60 <= best theme relevance < 80",
            "SUPPORT": "40 <= best theme relevance < 60",
            "NOISE": "all mapped theme relevance < 40",
            "UNMAPPED": "stock code absent from completed 124-theme master",
        },
        "ordering_rule": "Preserve IFIS rank as the primary attention order; theme relevance is a transparent filter, not a trading score.",
        "ifis_collection_rule": "Manual paste/input only. This pipeline does not scrape IFIS.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Join manually supplied IFIS candidates to the finalized 124-theme relevance master.")
    ap.add_argument("--input", default="research/data/ifis_manual_input.csv")
    ap.add_argument("--master", default="research/results/theme_relevance_batches/theme_relevance_full_final.csv")
    ap.add_argument("--out-dir", default="research/results/ifis_theme_filter")
    args = ap.parse_args()
    summary = run_filter(Path(args.input), Path(args.master), Path(args.out_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
