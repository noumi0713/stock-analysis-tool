from __future__ import annotations

import argparse
import csv
import io
import json
import os
import zipfile
from datetime import date, timedelta
from pathlib import Path

import requests

from collect_revenue_mix_v4 import calculated_ratio, score_ratio_candidate, snippets, theme_terms

BASE = "https://api.edinet-fsa.go.jp/api/v2"


def norm_sec_code(v: str) -> str:
    s = (v or "").strip().upper()
    if s.endswith(".T"):
        s = s[:-2]
    return s[:4]


def list_latest_reports(api_key: str, wanted: set[str], days: int = 430) -> dict[str, dict]:
    session = requests.Session()
    found: dict[str, dict] = {}
    today = date.today()
    for offset in range(days):
        if len(found) >= len(wanted):
            break
        d = today - timedelta(days=offset)
        try:
            r = session.get(
                f"{BASE}/documents.json",
                params={"date": d.isoformat(), "type": 2, "Subscription-Key": api_key},
                timeout=20,
            )
            if r.status_code != 200:
                continue
            data = r.json()
        except Exception:
            continue
        for row in data.get("results", []) or []:
            desc = str(row.get("docDescription") or "")
            if "有価証券報告書" not in desc or "訂正" in desc:
                continue
            code = norm_sec_code(str(row.get("secCode") or ""))
            if code in wanted and code not in found:
                found[code] = row
        if offset % 30 == 0:
            print("scan", d.isoformat(), len(found), "/", len(wanted))
    return found


def fetch_csv_text(api_key: str, doc_id: str) -> str:
    try:
        r = requests.get(
            f"{BASE}/documents/{doc_id}",
            params={"type": 5, "Subscription-Key": api_key},
            timeout=40,
        )
        if r.status_code != 200:
            return ""
        z = zipfile.ZipFile(io.BytesIO(r.content))
    except Exception:
        return ""
    chunks = []
    for name in z.namelist():
        if "XBRL_TO_CSV/" not in name or not name.lower().endswith(".csv"):
            continue
        try:
            raw = z.read(name).decode("utf-16", errors="ignore")
            reader = csv.DictReader(io.StringIO(raw), delimiter="\t")
            for row in reader:
                item = str(row.get("項目名") or "")
                value = str(row.get("値") or "")
                if not value:
                    continue
                # Keep segment/revenue text blocks and numeric segment-related rows only.
                key = item + " " + str(row.get("コンテキストID") or "")
                if any(w in key for w in ("セグメント", "売上", "収益", "Segment", "Revenue", "Sales")):
                    chunks.append(f"{item} {key} {value}")
        except Exception:
            continue
    return " ".join(chunks)[:2_000_000]


def extract_theme_ratio(text: str, theme: str) -> tuple[str, str, str]:
    for sn in snippets(text, theme_terms(theme), radius=800, limit=20):
        pct, basis = score_ratio_candidate(sn)
        if pct is not None:
            return f"{pct:.2f}", "edinet_explicit_ratio_candidate", sn[:1200]
    for sn in snippets(text, theme_terms(theme), radius=800, limit=20):
        pct, basis = calculated_ratio(sn)
        if pct is not None:
            return f"{pct:.2f}", "edinet_calculated_ratio_candidate", sn[:1200]
    return "", "unknown", ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dir", default="research/results/theme_relevance_batches")
    args = ap.parse_args()
    root = Path(args.dir)
    src = root / f"batch_{args.batch:03d}_revenue_mix_v4.csv"
    dst = root / f"batch_{args.batch:03d}_revenue_mix_v4_edinet.csv"
    summary_path = root / f"batch_{args.batch:03d}_edinet_v4_summary.json"

    api_key = os.getenv("EDINET_API_KEY", "").strip()
    with src.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not api_key:
        with dst.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
            if rows:
                w.writeheader(); w.writerows(rows)
        summary = {"status": "skipped_missing_edinet_api_key", "rows": len(rows)}
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))
        return

    unresolved_codes = {r["stock_code"] for r in rows if r.get("share_status") == "unknown"}
    reports = list_latest_reports(api_key, {norm_sec_code(c) for c in unresolved_codes})
    by_code_text = {}
    for i, (code, meta) in enumerate(reports.items(), 1):
        doc_id = str(meta.get("docID") or "")
        if not doc_id:
            continue
        by_code_text[code] = (doc_id, fetch_csv_text(api_key, doc_id))
        print("report", i, code, doc_id, len(by_code_text[code][1]))

    filled_explicit = 0
    filled_calc = 0
    for r in rows:
        if r.get("share_status") != "unknown":
            continue
        code = norm_sec_code(r["stock_code"])
        doc = by_code_text.get(code)
        if not doc or not doc[1]:
            continue
        pct, status, evidence = extract_theme_ratio(doc[1], r["theme_name"])
        if not pct:
            continue
        r["revenue_share_pct"] = pct
        r["share_status"] = status
        r["share_basis"] = "edinet_type5_csv"
        r["source_url"] = f"{BASE}/documents/{doc[0]}?type=5"
        r["evidence"] = evidence
        if "explicit" in status:
            filled_explicit += 1
        else:
            filled_calc += 1

    fields = list(rows[0].keys()) if rows else []
    with dst.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if rows:
            w.writeheader(); w.writerows(rows)
    summary = {
        "status": "completed",
        "reports_found": len(reports),
        "explicit_candidates_added": filled_explicit,
        "calculated_candidates_added": filled_calc,
        "unresolved_codes_before": len(unresolved_codes),
        "note": "EDINET type=5 CSV is used only after company-site evidence remains unknown; candidates still require QA.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
