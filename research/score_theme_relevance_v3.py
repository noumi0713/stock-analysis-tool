from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from score_theme_relevance import band
from score_theme_relevance_v2 import score_pair as score_v2

ALIASES: dict[str, list[str]] = {
    "人工知能": ["人工知能", "AI", "機械学習", "ディープラーニング"],
    "生成AI": ["生成AI", "生成ＡＩ", "大規模言語モデル", "LLM"],
    "AIエージェント": ["AIエージェント", "ＡＩエージェント", "エージェントAI"],
    "フィジカルAI": ["フィジカルAI", "フィジカルＡＩ"],
    "半導体": ["半導体"], "半導体製造装置": ["半導体製造装置"], "半導体部材・部品": ["半導体材料", "半導体部材", "半導体部品"],
    "データセンター": ["データセンター", "データセンタ"],
    "光ファイバー": ["光ファイバー", "光ファイバ"], "電力設備投資関連": ["電力設備", "電力インフラ", "送配電"],
    "送電": ["送電", "送配電"], "スマートグリッド": ["スマートグリッド"], "受変電設備": ["受変電", "変電設備"],
    "サーバー冷却": ["サーバー冷却", "データセンター冷却", "液冷"],
    "地方銀行": ["地方銀行", "地域銀行"], "ネット銀行": ["ネット銀行", "インターネット銀行", "デジタルバンク"],
    "フィンテック": ["フィンテック", "FinTech"], "キャッシュレス決済": ["キャッシュレス", "決済サービス", "電子決済"],
    "レアアース": ["レアアース"], "レアメタル": ["レアメタル", "希少金属"], "都市鉱山": ["都市鉱山", "貴金属リサイクル", "電子廃棄物"],
    "防衛": ["防衛", "防衛装備"], "宇宙開発関連": ["宇宙開発", "衛星", "ロケット", "宇宙事業"],
    "ドローン": ["ドローン", "無人航空機"], "サイバーセキュリティ": ["サイバーセキュリティ", "情報セキュリティ"],
    "国土強靱化": ["国土強靱化", "国土強靭化", "社会インフラ補修", "社会インフラの補修", "インフラ補修", "インフラ補強"],
    "下水道": ["下水道", "下水処理", "下水"], "水道関連": ["水道", "上水道", "水処理"],
    "電線地中化": ["電線地中化", "無電柱化", "地中化"], "防災": ["防災", "災害対策"], "耐震化": ["耐震", "耐震補強"],
    "工作機械": ["工作機械", "マシニングセンタ", "NC旋盤", "ＮＣ旋盤"], "FA関連": ["FA", "ＦＡ", "ファクトリーオートメーション"],
    "サービスロボット": ["サービスロボット"], "建設機械": ["建設機械"], "IoT": ["IoT", "ＩｏＴ", "スマートファクトリー"],
    "物流テック": ["物流DX", "物流ＤＸ", "物流テック", "倉庫自動化"], "3Dプリンター": ["3Dプリンタ", "３Ｄプリンタ", "積層造形"],
    "自動運転車": ["自動運転", "ADAS", "ＡＤＡＳ"], "電気自動車関連": ["電気自動車", "EV", "ＥＶ", "V2H", "Ｖ２Ｈ"],
    "自動車電子化": ["車載電子", "車載エレクトロニクス", "ECU", "ＥＣＵ", "ADAS"], "自動車製造装置": ["自動車製造装置", "車両組立"],
    "自動車軽量化": ["自動車軽量化", "車体軽量化"], "全固体電池": ["全固体電池"], "リチウムイオン電池": ["リチウムイオン電池", "Li-ion"],
    "自動車部材・部品": ["自動車部品", "自動車部材", "車載部品"], "MaaS": ["MaaS", "ＭａａＳ"], "空飛ぶクルマ": ["空飛ぶクルマ", "eVTOL", "空のモビリティ"],
    "再生可能エネルギー": ["再生可能エネルギー", "再エネ", "グリーンエネルギー"], "太陽光発電関連": ["太陽光発電", "太陽光発電所", "ソーラー"],
    "風力発電": ["風力発電", "風力発電所"], "原子力発電": ["原子力発電", "原発"], "核融合発電": ["核融合", "フュージョン"],
    "水素": ["水素", "燃料電池"], "アンモニア": ["アンモニア"], "蓄電池": ["蓄電池", "蓄電所", "エネルギー貯蔵"],
    "省エネ関連": ["省エネ", "省エネルギー"], "脱炭素": ["脱炭素", "カーボンニュートラル", "GX", "ＧＸ"],
    "インバウンド": ["インバウンド", "訪日外国人"], "eコマース": ["EC", "ＥＣ", "eコマース", "電子商取引", "オンライン販売"],
    "バイオテクノロジー関連": ["バイオテクノロジー", "バイオ医薬"], "創薬": ["創薬", "薬剤開発"], "再生医療": ["再生医療", "細胞治療"],
    "医療機器": ["医療機器", "医療装置"], "遠隔医療": ["遠隔医療", "オンライン診療"], "ホームヘルスケア": ["在宅医療", "ホームヘルスケア"],
    "介護関連": ["介護", "介護サービス"], "認知症薬": ["認知症", "アルツハイマー"], "医薬品関連": ["医薬品", "製薬", "医薬"],
    "不動産テック": ["不動産テック", "不動産DX", "不動産ＤＸ", "不動産業界に特化", "住宅・不動産業界"],
    "不動産ファンド": ["不動産ファンド", "不動産投資ファンド", "REIT", "ＲＥＩＴ"], "再開発": ["再開発", "都市開発"], "建設DX": ["建設DX", "建設ＤＸ", "BIM", "ＢＩＭ"],
    "海運": ["海運", "海上輸送"], "倉庫": ["倉庫", "倉庫業"], "陸運": ["陸運", "トラック輸送"], "鉄道関連": ["鉄道", "軌道"],
    "航空": ["航空", "航空輸送"], "港湾運送": ["港湾運送", "港湾荷役"], "宅配": ["宅配", "定期配送", "直接配送"],
    "SaaS": ["SaaS", "ＳａａＳ", "クラウドサービス", "月額"], "クラウドコンピューティング": ["クラウド", "クラウドサービス"],
    "デジタルトランスフォーメーション": ["DX", "ＤＸ", "デジタルトランスフォーメーション", "デジタル変革"],
    "データベース": ["データベース"], "サブスクリプション": ["サブスクリプション", "月額", "継続課金"],
    "電子政府": ["電子政府", "デジタル政府", "自治体DX", "自治体ＤＸ"], "マイナンバー": ["マイナンバー", "個人番号"],
    "教育ICT": ["教育ICT", "教育ＩＣＴ", "EdTech", "エドテック"], "ゲーム関連": ["ゲーム", "ゲーム開発"],
}

BROAD = {"建設", "機械", "食品", "小売り", "物流", "銀行", "保険", "IT関連", "不動産関連", "住宅関連", "旅行", "ホテル", "外食", "証券", "リース"}
MACRO = {"円高メリット", "円安メリット", "金利上昇メリット", "生活防衛", "インバウンド"}


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).lower()


def theme_terms(theme: str) -> list[str]:
    terms = list(ALIASES.get(theme, []))
    if theme not in terms:
        terms.append(theme)
    if theme.endswith("関連"):
        terms.append(theme[:-2])
    # preserve order, remove tiny/generic tokens
    out = []
    for t in terms:
        if len(t.strip()) < 2:
            continue
        if t not in out:
            out.append(t)
    return out


def official_hits(theme: str, text: str) -> tuple[list[str], int, float | None]:
    low = normalize(text)
    found: list[str] = []
    count = 0
    first: int | None = None
    for term in theme_terms(theme):
        tl = term.lower()
        c = low.count(tl)
        if c:
            found.append(term)
            count += c
            p = low.find(tl)
            first = p if first is None else min(first, p)
    pos = None if first is None else first / max(1, len(low))
    return found, count, pos


def apply_official(theme: str, base: dict[str, object], evidence: dict[str, str]) -> dict[str, object]:
    out = dict(base)
    text = evidence.get("official_text", "") or ""
    status = evidence.get("official_status", "")
    found, count, pos = official_hits(theme, text)
    out["official_hits"] = found
    out["official_hit_count"] = count
    out["official_position"] = pos
    out["official_urls"] = evidence.get("official_urls", "")
    if status != "ok" or not found:
        return out

    current = int(out["current"])
    growth = int(out["growth"])

    if theme in MACRO:
        # An official page mentioning a macro word is not enough to prove earnings sensitivity.
        current = max(current, min(58, 42 + min(12, count * 2)))
        growth = max(growth, min(58, current))
    elif theme in BROAD:
        # Broad words in navigation/footer are common. Require repeated evidence.
        if count >= 5:
            current = max(current, 72)
            growth = max(growth, 68)
        elif count >= 2:
            current = max(current, 58)
            growth = max(growth, 54)
    else:
        if count >= 6:
            current = max(current, 92)
            growth = max(growth, 90)
        elif count >= 3:
            current = max(current, 85)
            growth = max(growth, 82)
        else:
            current = max(current, 72)
            growth = max(growth, 68)
        if pos is not None and pos <= 0.20:
            current = min(100, max(current, 90))
        if any(x in normalize(text) for x in ["重点", "成長戦略", "成長分野", "集中投資", "中期経営計画", "新設"]):
            growth = min(100, max(growth, current + 6))

    final = int(round(current * 0.70 + growth * 0.30))
    out["current"] = current
    out["growth"] = growth
    out["score"] = final
    if final >= 85:
        out["confidence"] = "A"
    elif final >= 60:
        out["confidence"] = "B"
    else:
        out["confidence"] = "C"
    base_ev = str(out.get("evidence", ""))
    off_ev = "official=" + "/".join(found[:4]) + f"; count={count}" + (f"; pos={pos:.2f}" if pos is not None else "")
    out["evidence"] = (base_ev + "; " + off_ev).strip("; ")
    if out.get("review") in {"needs_review", "qa_borderline"} and final >= 80:
        out["review"] = "official_site_supported"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dir", default="research/results/theme_relevance_batches")
    args = ap.parse_args()
    root = Path(args.dir)
    profiles_path = root / f"batch_{args.batch:03d}_profiles.csv"
    evidence_path = root / f"batch_{args.batch:03d}_official_evidence.csv"
    memberships_path = root / f"batch_{args.batch:03d}_memberships.csv"
    out_path = root / f"batch_{args.batch:03d}_scores_v3.csv"
    summary_path = root / f"batch_{args.batch:03d}_score_summary_v3.json"

    with profiles_path.open(encoding="utf-8-sig") as f:
        profiles = {r["stock_code"]: r for r in csv.DictReader(f)}
    with evidence_path.open(encoding="utf-8-sig") as f:
        evidence = {r["stock_code"]: r for r in csv.DictReader(f)}
    with memberships_path.open(encoding="utf-8-sig") as f:
        memberships = list(csv.DictReader(f))

    rows = []
    for m in memberships:
        code = m["stock_code"]
        p = profiles.get(code, {})
        b = score_v2(m["theme_name"], p)
        s = apply_official(m["theme_name"], b, evidence.get(code, {}))
        rows.append({
            "batch": args.batch,
            "stock_code": code,
            "long_name": p.get("long_name", m.get("company_name", "")),
            "theme_name": m["theme_name"],
            "cluster": m["cluster"],
            "current_business_score": s["current"],
            "growth_relevance_score": s["growth"],
            "relevance_score": s["score"],
            "band": band(int(s["score"])),
            "confidence": s["confidence"],
            "review_flag": s["review"],
            "evidence": s["evidence"],
            "official_hits": " | ".join(s.get("official_hits", [])),
            "official_hit_count": s.get("official_hit_count", 0),
            "official_position": "" if s.get("official_position") is None else f"{s['official_position']:.4f}",
            "official_urls": s.get("official_urls", ""),
            "quote_type": p.get("quote_type", ""),
            "sector": p.get("sector", ""),
            "industry": p.get("industry", ""),
            "website": p.get("website", ""),
        })

    fields = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    summary = {
        "batch": args.batch,
        "stocks": len(profiles),
        "stock_theme_pairs": len(rows),
        "official_site_available": sum(1 for r in evidence.values() if r.get("official_status") == "ok"),
        "official_site_supported_pairs": sum(1 for r in rows if int(r.get("official_hit_count") or 0) > 0),
        "bands": dict(Counter(r["band"] for r in rows)),
        "confidence": dict(Counter(r["confidence"] for r in rows)),
        "review": dict(Counter(r["review_flag"] for r in rows)),
        "status": "research_first_pass_v3",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
