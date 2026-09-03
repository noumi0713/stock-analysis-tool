from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "research" / "theme_catalog_124.csv"
DEFAULT_MANIFEST = ROOT / "dashboard-data" / "technical-backtest-3y" / "manifest.json"
DEFAULT_OUT = ROOT / "research" / "data" / "theme_members_124.csv"
DEFAULT_REPORT = ROOT / "research" / "data" / "theme_catalog_124_validation.json"
BASE_PATH = "https://s.kabutan.jp/themes/{}/"
CODE_RE = re.compile(r"(?:/stocks/|[?&]code=)([0-9A-Z]{4})(?:/|[&#?\"']|$)", re.I)


def read_catalog(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = [dict(row) for row in csv.DictReader(fh)]
    if len(rows) != 124:
        raise ValueError(f"Theme catalog must contain exactly 124 rows, got {len(rows)}")
    names = [str(r.get("theme_name") or "").strip() for r in rows]
    if len(set(names)) != 124 or any(not n for n in names):
        raise ValueError("theme_name must contain 124 unique non-empty values")
    clusters = {str(r.get("cluster") or "").strip() for r in rows}
    if len(clusters) != 13:
        raise ValueError(f"Expected 13 clusters, got {len(clusters)}")
    return rows


def load_manifest_names(path: Path) -> dict[str, dict[str, str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, str]] = {}
    for ticker, row in (manifest.get("stocks") or {}).items():
        code = str((row or {}).get("code") or str(ticker).removesuffix(".T")).strip()
        if not code:
            continue
        result[code] = {
            "yahoo_ticker": str(ticker),
            "company_name": str((row or {}).get("name") or ticker),
            "sector": str((row or {}).get("sector") or ""),
        }
    return result


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").replace("　", "")


def parse_theme_page(html: str) -> tuple[str | None, list[str]]:
    soup = BeautifulSoup(html, "html.parser")
    heading = None
    for tag in soup.find_all(["h1", "h2"]):
        text = tag.get_text(" ", strip=True)
        if "テーマ株一覧" in text:
            heading = text.replace("テーマ株一覧", "").strip()
            break

    codes: set[str] = set()
    for link in soup.find_all("a", href=True):
        href = str(link.get("href") or "")
        for match in CODE_RE.finditer(href):
            codes.add(match.group(1).upper())
    if not codes:
        for match in CODE_RE.finditer(html):
            codes.add(match.group(1).upper())
    return heading, sorted(codes)


def fetch_theme(
    session: requests.Session,
    kabutan_name: str,
    timeout: int,
    *,
    page_delay: float = 0.03,
    max_pages: int = 60,
) -> dict:
    encoded = quote(kabutan_name, safe="")
    base_url = BASE_PATH.format(encoded)
    all_codes: set[str] = set()
    page_heading: str | None = None
    first_status: int | None = None
    pages_with_members = 0

    try:
        for page in range(1, max_pages + 1):
            url = f"{base_url}?market=all&page={page}"
            response = session.get(url, timeout=timeout)
            if page == 1:
                first_status = int(response.status_code)
            if response.status_code == 404 and page > 1:
                break
            response.raise_for_status()
            heading, codes = parse_theme_page(response.text)
            if page == 1:
                page_heading = heading
                if not (
                    page_heading
                    and normalize_text(page_heading) == normalize_text(kabutan_name)
                ):
                    return {
                        "url": f"{base_url}?market=all",
                        "http_status": first_status,
                        "page_heading": page_heading,
                        "heading_match": False,
                        "page_count": 0,
                        "codes": [],
                        "error": None,
                    }

            code_set = set(codes)
            new_codes = code_set - all_codes
            if not code_set or (page > 1 and not new_codes):
                break
            all_codes.update(new_codes)
            pages_with_members = page

            # Kabutan currently paginates theme members at 20 rows/page. A short
            # page is the terminal page; otherwise continue until no new codes.
            if len(code_set) < 20:
                break
            if page_delay > 0:
                time.sleep(page_delay)

        return {
            "url": f"{base_url}?market=all",
            "http_status": first_status,
            "page_heading": page_heading,
            "heading_match": bool(
                page_heading
                and normalize_text(page_heading) == normalize_text(kabutan_name)
            ),
            "page_count": pages_with_members,
            "codes": sorted(all_codes),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic report must preserve failures
        return {
            "url": f"{base_url}?market=all",
            "http_status": first_status,
            "page_heading": page_heading,
            "heading_match": False,
            "page_count": pages_with_members,
            "codes": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "theme_name",
        "cluster",
        "kabutan_name",
        "stock_code",
        "yahoo_ticker",
        "company_name",
        "sector",
        "in_3y_manifest",
        "source_url",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(
    catalog_path: Path,
    manifest_path: Path,
    out_path: Path,
    report_path: Path,
    *,
    delay: float,
    page_delay: float,
    timeout: int,
    strict: bool,
) -> dict:
    catalog = read_catalog(catalog_path)
    manifest_names = load_manifest_names(manifest_path)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; stock-analysis-tool-theme-research/1.0)",
            "Accept-Language": "ja,en;q=0.7",
        }
    )

    membership_rows: list[dict] = []
    validations: list[dict] = []
    membership_counter: Counter[str] = Counter()
    eligible_counter: Counter[str] = Counter()
    unique_all: set[str] = set()
    unique_eligible: set[str] = set()

    for index, item in enumerate(catalog, 1):
        theme = str(item["theme_name"]).strip()
        cluster = str(item["cluster"]).strip()
        kabutan_name = str(item.get("kabutan_name") or theme).strip()
        fetched = fetch_theme(
            session,
            kabutan_name,
            timeout,
            page_delay=page_delay,
        )
        codes = list(fetched.pop("codes"))
        resolved = bool(fetched["heading_match"] and codes)
        eligible_codes = [code for code in codes if code in manifest_names]
        validations.append(
            {
                "theme_name": theme,
                "cluster": cluster,
                "kabutan_name": kabutan_name,
                "resolved": resolved,
                "member_count": len(codes),
                "eligible_3y_member_count": len(eligible_codes),
                **fetched,
            }
        )
        if resolved:
            membership_counter[theme] = len(codes)
            eligible_counter[theme] = len(eligible_codes)
            unique_all.update(codes)
            unique_eligible.update(eligible_codes)
            for code in codes:
                meta = manifest_names.get(code) or {}
                membership_rows.append(
                    {
                        "theme_name": theme,
                        "cluster": cluster,
                        "kabutan_name": kabutan_name,
                        "stock_code": code,
                        "yahoo_ticker": meta.get("yahoo_ticker") or f"{code}.T",
                        "company_name": meta.get("company_name") or "",
                        "sector": meta.get("sector") or "",
                        "in_3y_manifest": code in manifest_names,
                        "source_url": fetched["url"],
                    }
                )
        print(
            f"[{index:03d}/124] {theme}: resolved={resolved} "
            f"members={len(codes)} eligible={len(eligible_codes)} "
            f"pages={fetched.get('page_count', 0)}"
        )
        if delay > 0 and index < len(catalog):
            time.sleep(delay)

    membership_rows.sort(key=lambda r: (r["theme_name"], r["stock_code"]))
    write_csv(out_path, membership_rows)

    resolved_rows = [r for r in validations if r["resolved"]]
    unresolved_rows = [r for r in validations if not r["resolved"]]
    thin_rows = [r for r in resolved_rows if r["eligible_3y_member_count"] < 5]
    ranking_eligible_rows = [
        r for r in resolved_rows if r["eligible_3y_member_count"] >= 5
    ]
    by_cluster = defaultdict(
        lambda: {
            "catalog_themes": 0,
            "resolved_themes": 0,
            "ranking_eligible_themes": 0,
            "membership_rows": 0,
            "eligible_membership_rows": 0,
        }
    )
    for item in catalog:
        by_cluster[item["cluster"]]["catalog_themes"] += 1
    for row in resolved_rows:
        bucket = by_cluster[row["cluster"]]
        bucket["resolved_themes"] += 1
        if row["eligible_3y_member_count"] >= 5:
            bucket["ranking_eligible_themes"] += 1
        bucket["membership_rows"] += row["member_count"]
        bucket["eligible_membership_rows"] += row["eligible_3y_member_count"]

    catalog_ready = len(resolved_rows) == 124
    report = {
        "catalog_theme_count": len(catalog),
        "cluster_count": len(by_cluster),
        "resolved_theme_count": len(resolved_rows),
        "unresolved_theme_count": len(unresolved_rows),
        "unique_member_stock_count": len(unique_all),
        "total_membership_rows": len(membership_rows),
        "unique_3y_eligible_stock_count": len(unique_eligible),
        "total_3y_eligible_membership_rows": sum(eligible_counter.values()),
        "ranking_eligible_theme_count": len(ranking_eligible_rows),
        "themes_with_fewer_than_5_eligible_members": len(thin_rows),
        "catalog_ready": catalog_ready,
        "strict_ready": catalog_ready,
        "cluster_summary": dict(sorted(by_cluster.items())),
        "unresolved": unresolved_rows,
        "thin_themes": thin_rows,
        "themes": validations,
        "notes": [
            "This is a research-only classification layer and does not modify production trading rules.",
            "Theme membership is a current snapshot; applying it to historical prices introduces classification look-ahead bias.",
            "Overlapping stock memberships are intentionally preserved. Weekly volume analysis should apportion volume by membership count.",
            "Weekly ranking research should exclude themes with fewer than 5 valid 3-year members rather than treating them as classification failures.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    compact_keys = [
        "catalog_theme_count",
        "cluster_count",
        "resolved_theme_count",
        "unresolved_theme_count",
        "unique_member_stock_count",
        "total_membership_rows",
        "unique_3y_eligible_stock_count",
        "total_3y_eligible_membership_rows",
        "ranking_eligible_theme_count",
        "themes_with_fewer_than_5_eligible_members",
        "catalog_ready",
        "strict_ready",
    ]
    print(
        json.dumps(
            {key: report[key] for key in compact_keys},
            ensure_ascii=False,
            indent=2,
        )
    )
    if strict and not report["strict_ready"]:
        raise SystemExit(2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--delay", type=float, default=0.10)
    parser.add_argument("--page-delay", type=float, default=0.03)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    build(
        args.catalog,
        args.manifest,
        args.output,
        args.report,
        delay=args.delay,
        page_delay=args.page_delay,
        timeout=args.timeout,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
