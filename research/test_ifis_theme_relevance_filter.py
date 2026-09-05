from __future__ import annotations

import csv
import json
from pathlib import Path

from ifis_theme_relevance_filter import candidate_class, norm_code, run_filter


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def test_norm_code() -> None:
    assert norm_code("7203.T") == "7203"
    assert norm_code("603A") == "603A"
    assert norm_code("603A.T") == "603A"


def test_candidate_class() -> None:
    assert candidate_class(80.0) == "CORE"
    assert candidate_class(79.9) == "STRONG"
    assert candidate_class(60.0) == "STRONG"
    assert candidate_class(40.0) == "SUPPORT"
    assert candidate_class(39.9) == "NOISE"
    assert candidate_class(None) == "UNMAPPED"


def test_run_filter(tmp_path: Path) -> None:
    inp = tmp_path / "ifis.csv"
    master = tmp_path / "master.csv"
    out = tmp_path / "out"

    write_csv(
        inp,
        [
            {"ifis_rank": "2", "stock_code": "603A.T", "company_name": "Alpha", "snapshot_at": "2026-09-05T10:00:00+09:00"},
            {"ifis_rank": "1", "stock_code": "1111", "company_name": "Beta", "snapshot_at": "2026-09-05T10:00:00+09:00"},
            {"ifis_rank": "3", "stock_code": "9999", "company_name": "Missing", "snapshot_at": "2026-09-05T10:00:00+09:00"},
        ],
        ["ifis_rank", "stock_code", "company_name", "snapshot_at"],
    )
    write_csv(
        master,
        [
            {"stock_code": "603A", "company_name": "Alpha", "theme_name": "AI", "cluster": "IT", "final_relevance_score": "92", "final_band": "主力テーマ", "final_confidence": "A", "decision_source": "test", "revenue_share_pct": "", "decision_reason": "direct", "source_url": "", "status": "finalized"},
            {"stock_code": "603A", "company_name": "Alpha", "theme_name": "IoT", "cluster": "IT", "final_relevance_score": "55", "final_band": "補助関連", "final_confidence": "B", "decision_source": "test", "revenue_share_pct": "", "decision_reason": "related", "source_url": "", "status": "finalized"},
            {"stock_code": "1111", "company_name": "Beta", "theme_name": "機械", "cluster": "産業", "final_relevance_score": "65", "final_band": "有力関連", "final_confidence": "B", "decision_source": "test", "revenue_share_pct": "", "decision_reason": "related", "source_url": "", "status": "finalized"},
        ],
        ["stock_code", "company_name", "theme_name", "cluster", "final_relevance_score", "final_band", "final_confidence", "decision_source", "revenue_share_pct", "decision_reason", "source_url", "status"],
    )

    summary = run_filter(inp, master, out)
    assert summary["candidate_count"] == 3
    assert summary["mapped_candidates"] == 2
    assert summary["unmapped_candidates"] == 1
    assert summary["candidate_class_counts"] == {"CORE": 1, "STRONG": 1, "UNMAPPED": 1}

    rows = list(csv.DictReader((out / "latest_candidates.csv").open(encoding="utf-8-sig")))
    assert [r["stock_code"] for r in rows] == ["1111", "603A", "9999"]
    assert rows[0]["candidate_class"] == "STRONG"
    assert rows[1]["candidate_class"] == "CORE"
    assert rows[1]["best_theme"] == "AI"
    assert rows[2]["candidate_class"] == "UNMAPPED"

    themes = list(csv.DictReader((out / "latest_candidate_themes.csv").open(encoding="utf-8-sig")))
    alpha = [r for r in themes if r["stock_code"] == "603A"]
    assert [r["theme_name"] for r in alpha] == ["AI", "IoT"]

    saved = json.loads((out / "latest_summary.json").read_text(encoding="utf-8"))
    assert saved["ifis_collection_rule"].startswith("Manual")
