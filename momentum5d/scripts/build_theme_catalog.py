from __future__ import annotations

import argparse
import csv
import html
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SOURCE = "https://jigyou-v2.kabu-trendpro.com/"
SECTOR_17_NAMES = {
    "1": "食品",
    "2": "エネルギー資源",
    "3": "建設・資材",
    "4": "素材・化学",
    "5": "医薬品",
    "6": "自動車・輸送機",
    "7": "鉄鋼・非鉄",
    "8": "機械",
    "9": "電機・精密",
    "10": "情報通信・サービスその他",
    "11": "電力・ガス",
    "12": "運輸・物流",
    "13": "商社・卸売",
    "14": "小売",
    "15": "銀行",
    "16": "金融（除く銀行）",
    "17": "不動産",
}
THEME_CLUSTERS: dict[str, tuple[str, ...]] = {
    "AI・半導体": (
        "人工知能",
        "生成AI",
        "フィジカルAI",
        "エッジAI",
        "半導体",
        "半導体製造装置",
        "半導体部材・部品",
        "半導体商社",
        "次世代パワー半導体",
        "車載半導体",
        "ラピダス",
        "量子コンピューター",
    ),
    "データセンター・通信": (
        "データセンター",
        "クラウド",
        "5G",
        "光通信",
        "衛星通信",
        "デジタルインフラ",
        "IoT",
        "通信機器",
        "海底ケーブル",
    ),
    "防衛・宇宙・安全保障": (
        "防衛",
        "宇宙",
        "ミサイル防衛",
        "ドローン",
        "サイバーセキュリティ",
        "経済安全保障",
        "防災",
        "監視カメラ",
        "生体認証",
    ),
    "ロボット・製造DX": (
        "ロボット",
        "AIロボット",
        "FA",
        "製造DX",
        "工場自動化",
        "産業機械",
        "工作機械",
        "3Dプリンター",
        "センサー",
    ),
    "モビリティ・自動車": (
        "EV",
        "自動運転",
        "全固体電池",
        "リチウムイオン電池",
        "車載OS",
        "自動車電子部品",
        "充電インフラ",
        "MaaS",
        "車載ソフトウェア",
    ),
    "脱炭素・エネルギー": (
        "GX",
        "再生可能エネルギー",
        "太陽光発電",
        "ペロブスカイト太陽電池",
        "洋上風力",
        "水素",
        "核融合",
        "蓄電池",
        "電力インフラ",
        "省エネルギー",
    ),
    "資源・素材": (
        "レアアース",
        "レアメタル",
        "銅",
        "リチウム",
        "ニッケル",
        "都市鉱山",
        "資源開発",
        "鉄鋼",
        "化学",
    ),
    "金融・フィンテック": (
        "地方銀行",
        "銀行",
        "証券",
        "保険",
        "フィンテック",
        "キャッシュレス",
        "仮想通貨",
        "資産運用",
        "ネット銀行",
    ),
    "医療・バイオ": (
        "バイオ",
        "ヘルスケア",
        "再生医療",
        "創薬AI",
        "医療機器",
        "遠隔医療",
        "介護",
        "がん治療",
        "ジェネリック医薬品",
        "健康経営",
    ),
    "内需・消費・観光": (
        "インバウンド",
        "旅行",
        "ホテル",
        "外食",
        "小売",
        "eコマース",
        "化粧品",
        "ゲーム",
        "コンテンツ",
        "スポーツ",
    ),
    "建設・インフラ・不動産": (
        "公共投資",
        "建設DX",
        "インフラ",
        "国土強靱化",
        "トンネル",
        "橋梁",
        "不動産",
        "スマートシティ",
        "REIT",
    ),
    "企業DX・情報サービス": (
        "SaaS",
        "DX",
        "データ分析",
        "データベース",
        "システムインテグレーション",
        "ERP",
        "BPO",
        "人材サービス",
        "広告",
        "情報配信",
    ),
    "農業・食品・物流": (
        "スマート農業",
        "農業",
        "肥料",
        "養殖",
        "食品",
        "食品ロス",
        "物流",
        "物流DX",
        "包装",
    ),
}

ALIASES = {"データ分析・解析": "データ分析"}
CARD_RE = re.compile(
    r'<div class="company-card" id="c-(?P<code>[0-9A-Z]+)">(?P<body>.*?)'
    r'(?=<div class="company-card"|<div class="pagination-container")',
    re.DOTALL,
)
NAME_RE = re.compile(r'<h2 class="company-name">.*?<a[^>]*>(?P<name>.*?)</a>', re.DOTALL)
MARKET_RE = re.compile(r'<div class="listing-info">(?P<market>.*?)</div>', re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
FUND_RE = re.compile(r"ETF|ETN|ＥＴＦ|ＥＴＮ|上場投信|ＮＥＸＴ ＦＵＮＤＳ|グローバルＸ")


def _clean(value: str) -> str:
    return " ".join(html.unescape(TAG_RE.sub(" ", value)).split())


def _fetch_theme(theme: str, retries: int = 3) -> list[dict[str, str]]:
    url = f"{SOURCE}?{urlencode({'q': theme})}"
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0 theme-catalog-research/1.0"})
            with urlopen(request, timeout=45) as response:
                page = response.read().decode("utf-8", errors="replace")
            rows: list[dict[str, str]] = []
            for match in CARD_RE.finditer(page):
                body = match.group("body")
                name_match = NAME_RE.search(body)
                market_match = MARKET_RE.search(body)
                if not name_match:
                    continue
                name = _clean(name_match.group("name"))
                if FUND_RE.search(name):
                    continue
                rows.append(
                    {
                        "stock_code": match.group("code"),
                        "company_name": name,
                        "market": _clean(market_match.group("market")) if market_match else "",
                    }
                )
            return rows[:20]
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    return []


def _existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sector_groups(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["code"].strip(): SECTOR_17_NAMES.get(row["sector_17_code"].strip(), "")
            for row in csv.DictReader(handle)
        }


def build_catalog(
    existing_path: Path,
    sector_path: Path | None,
    workers: int = 8,
    *,
    fetch: bool = True,
    only_themes: set[str] | None = None,
) -> list[dict[str, str]]:
    cluster_by_theme = {
        theme: cluster for cluster, themes in THEME_CLUSTERS.items() for theme in themes
    }
    if len(cluster_by_theme) != 124:
        raise ValueError(
            f"Theme catalog must contain 124 unique themes, got {len(cluster_by_theme)}"
        )

    existing = _existing_rows(existing_path)
    valid_groups = set(SECTOR_17_NAMES.values())
    topix_by_code = {
        row["stock_code"]: row.get("topix17_group", "")
        for row in existing
        if row.get("topix17_group", "") in valid_groups
    }
    topix_by_code.update(
        {code: group for code, group in _sector_groups(sector_path).items() if group}
    )
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for row in existing:
        theme = ALIASES.get(row["theme_name"], row["theme_name"])
        if theme not in cluster_by_theme:
            continue
        if FUND_RE.search(row.get("company_name", "")):
            continue
        normalized = dict(row)
        normalized["theme_name"] = theme
        normalized["cluster"] = cluster_by_theme[theme]
        normalized["topix17_group"] = topix_by_code.get(row["stock_code"], "")
        rows[(theme, row["stock_code"])] = normalized

    if fetch:
        fetch_themes = only_themes or set(cluster_by_theme)
        unknown = fetch_themes.difference(cluster_by_theme)
        if unknown:
            raise ValueError(f"Unknown themes requested: {sorted(unknown)}")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_fetch_theme, theme): theme for theme in fetch_themes}
            for future in as_completed(futures):
                theme = futures[future]
                for item in future.result():
                    code = item["stock_code"]
                    rows.setdefault(
                        (theme, code),
                        {
                            "theme_name": theme,
                            "cluster": cluster_by_theme[theme],
                            "topix17_group": topix_by_code.get(code, ""),
                            "stock_code": code,
                            "yahoo_ticker": f"{code}.T",
                            "company_name": item["company_name"],
                            "short_name": item["company_name"],
                            "market": item["market"],
                            "source_url": f"{SOURCE}?{urlencode({'q': theme})}",
                        },
                    )
    return sorted(rows.values(), key=lambda row: (row["theme_name"], row["stock_code"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sectors", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reuse-only", action="store_true")
    parser.add_argument("--only-theme", action="append", default=[])
    args = parser.parse_args()
    rows = build_catalog(
        args.existing,
        args.sectors,
        args.workers,
        fetch=not args.reuse_only,
        only_themes=set(args.only_theme) or None,
    )
    fields = [
        "theme_name",
        "cluster",
        "topix17_group",
        "stock_code",
        "yahoo_ticker",
        "company_name",
        "short_name",
        "market",
        "source_url",
    ]
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"themes={len({row['theme_name'] for row in rows})} "
        f"clusters={len({row['cluster'] for row in rows})} "
        f"stocks={len({row['stock_code'] for row in rows})} memberships={len(rows)}"
    )


if __name__ == "__main__":
    main()
