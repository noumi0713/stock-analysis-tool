from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

# Current-business evidence is weighted 70%; growth/strategic evidence 30%.
# The scorer is deliberately conservative for macro/benefit themes where a company
# description alone cannot prove P&L sensitivity.

RULES: dict[str, dict[str, object]] = {
    "半導体": {"strong": ["semiconductor", "integrated circuit", "ic chip", "microcontroller", "memory chip"], "weak": ["electronic component", "electronics"]},
    "人工知能": {"strong": ["artificial intelligence", " ai ", "ai solution", "machine learning", "deep learning"], "weak": ["image recognition", "computer vision", "data science"]},
    "生成AI": {"strong": ["generative ai", "large language model", " llm", "chatgpt"], "weak": ["artificial intelligence", " ai "]},
    "AIエージェント": {"strong": ["ai agent", "autonomous agent", "agentic ai"], "weak": ["generative ai", "large language model", "artificial intelligence"]},
    "フィジカルAI": {"strong": ["physical ai", "robotics ai", "embodied ai"], "weak": ["robot", "computer vision", "artificial intelligence", "autonomous"]},
    "エッジコンピューティング": {"strong": ["edge computing", "edge device", "edge ai"], "weak": ["iot", "embedded", "gateway"]},
    "半導体製造装置": {"strong": ["semiconductor manufacturing equipment", "wafer processing", "lithography", "semiconductor equipment", "wafer inspection"], "weak": ["semiconductor", "precision equipment"]},
    "半導体部材・部品": {"strong": ["semiconductor material", "silicon wafer", "photoresist", "semiconductor component", "lead frame", "semiconductor package"], "weak": ["semiconductor", "electronic material"]},
    "半導体商社": {"strong": ["semiconductor trading", "distributes semiconductors", "semiconductor distributor", "electronic components trading"], "weak": ["semiconductor", "trading company"]},
    "量子コンピューター": {"strong": ["quantum computer", "quantum computing", "quantum technology"], "weak": ["quantum"]},
    "データ分析・解析": {"strong": ["data analytics", "data analysis", "analytics platform", "data science"], "weak": ["big data", "business intelligence", "analysis software"]},
    "ロボット": {"strong": ["robot", "robotics", "robotic"], "weak": ["automation", "autonomous machine"]},
    "データセンター": {"strong": ["data center", "datacenter"], "weak": ["server infrastructure", "cloud infrastructure", "colocation"]},
    "電力会社": {"strong": ["electric power utility", "electricity utility", "power generation and retail", "supplies electricity", "electric utility"], "weak": ["power generation", "electricity sales"]},
    "光ファイバー": {"strong": ["optical fiber", "fiber optic", "fibre optic"], "weak": ["optical cable", "telecommunication cable"]},
    "電線": {"strong": ["electric wire", "power cable", "wire and cable", "electrical cable"], "weak": ["cable", "wiring"]},
    "電力設備投資関連": {"strong": ["power transmission", "substation", "electrical construction", "power infrastructure", "electric power equipment"], "weak": ["electrical equipment", "power plant", "grid"]},
    "送電": {"strong": ["power transmission", "transmission line", "transmission grid"], "weak": ["power grid", "electric power infrastructure"]},
    "スマートグリッド": {"strong": ["smart grid", "grid management", "advanced metering"], "weak": ["power grid", "energy management", "smart meter"]},
    "受変電設備": {"strong": ["substation", "switchgear", "transformer equipment", "power receiving and distribution"], "weak": ["transformer", "electrical equipment"]},
    "サーバー冷却": {"strong": ["data center cooling", "server cooling", "liquid cooling", "cooling system"], "weak": ["air conditioning", "hvac", "thermal management"]},
    "地方銀行": {"strong": ["regional bank", "local bank"], "weak": ["banking"]},
    "ネット銀行": {"strong": ["internet bank", "online bank", "digital bank"], "weak": ["banking app", "banking"]},
    "銀行": {"strong": ["banking", "bank business", "commercial bank"], "weak": ["loans", "deposits"]},
    "金利上昇メリット": {"strong": ["banking", "life insurance", "property and casualty insurance"], "weak": ["interest income", "loans", "insurance"]},
    "証券": {"strong": ["securities brokerage", "securities business", "brokerage services", "investment banking"], "weak": ["financial instruments", "brokerage"]},
    "保険": {"strong": ["insurance business", "life insurance", "property and casualty insurance", "insurance company"], "weak": ["insurance agency", "insurance services"]},
    "リース": {"strong": ["leasing business", "lease services", "equipment leasing", "finance lease"], "weak": ["rental and leasing", "lease"]},
    "フィンテック": {"strong": ["fintech", "financial technology", "digital finance"], "weak": ["payment platform", "financial software", "blockchain"]},
    "キャッシュレス決済": {"strong": ["cashless payment", "payment processing", "digital payment", "mobile payment", "credit card payment"], "weak": ["payment service", "electronic payment"]},
    "レアアース": {"strong": ["rare earth"], "weak": ["neodymium", "dysprosium", "rare metal"]},
    "レアメタル": {"strong": ["rare metal", "critical mineral"], "weak": ["tungsten", "cobalt", "lithium", "nickel", "molybdenum"]},
    "金": {"strong": ["gold bullion", "gold mining", "gold business", "gold price", "gold bars"], "weak": ["precious metals", "gold"]},
    "銅": {"strong": ["copper mining", "copper smelting", "copper products", "copper business"], "weak": ["copper"]},
    "アルミニウム": {"strong": ["aluminum", "aluminium"], "weak": ["light metal"]},
    "鉄鋼": {"strong": ["steel products", "steel manufacturing", "steelmaker", "iron and steel"], "weak": ["steel"]},
    "非鉄": {"strong": ["non-ferrous metal", "nonferrous metal", "copper smelting", "aluminum", "zinc smelting"], "weak": ["metal products", "metals"]},
    "総合商社": {"strong": ["general trading company", "integrated trading company"], "weak": ["trading company", "trading business"]},
    "資源開発": {"strong": ["oil and gas exploration", "resource development", "mineral exploration", "mining development", "exploration and production"], "weak": ["mining", "oil and gas"]},
    "都市鉱山": {"strong": ["urban mining", "precious metal recycling", "metal recycling", "e-waste recycling"], "weak": ["recycling", "recover precious metals"]},
    "防衛": {"strong": ["defense equipment", "defence equipment", "military", "defense systems", "defence systems"], "weak": ["aerospace and defense", "security equipment"]},
    "宇宙開発関連": {"strong": ["space business", "satellite", "spacecraft", "space debris", "rocket", "space development"], "weak": ["aerospace", "orbital"]},
    "ドローン": {"strong": ["drone", "unmanned aerial", "uav"], "weak": ["aerial inspection", "unmanned aircraft"]},
    "サイバーセキュリティ": {"strong": ["cybersecurity", "cyber security", "information security", "security software"], "weak": ["network security", "endpoint security", "security service"]},
    "国土強靱化": {"strong": ["infrastructure repair", "infrastructure reinforcement", "civil infrastructure", "disaster resilience"], "weak": ["civil engineering", "bridge", "tunnel", "public works"]},
    "下水道": {"strong": ["sewerage", "sewage", "wastewater", "sewer system"], "weak": ["water treatment", "water infrastructure"]},
    "水道関連": {"strong": ["water supply", "waterworks", "water infrastructure", "water treatment"], "weak": ["water pipe", "water system"]},
    "電線地中化": {"strong": ["underground power cable", "underground cable", "utility undergrounding"], "weak": ["power cable", "civil engineering", "electrical construction"]},
    "防災": {"strong": ["disaster prevention", "fire prevention", "fire alarm", "emergency management", "flood control"], "weak": ["disaster", "seismic", "infrastructure reinforcement"]},
    "耐震化": {"strong": ["seismic reinforcement", "earthquake resistance", "seismic-resistant", "seismic retrofitting"], "weak": ["reinforcement", "earthquake"]},
    "工作機械": {"strong": ["machine tool", "machining center", "cnc machine", "lathe"], "weak": ["machining equipment", "industrial machinery"]},
    "FA関連": {"strong": ["factory automation", "industrial automation", "production automation"], "weak": ["automation", "manufacturing system", "control system"]},
    "サービスロボット": {"strong": ["service robot", "delivery robot", "cleaning robot", "communication robot"], "weak": ["robot", "robotics"]},
    "建設機械": {"strong": ["construction machinery", "construction equipment", "excavator", "crane equipment"], "weak": ["heavy machinery"]},
    "IoT": {"strong": ["internet of things", " iot", "iot solution", "iot platform"], "weak": ["connected device", "sensor network"]},
    "設備投資": {"strong": ["capital equipment", "industrial machinery", "factory equipment", "production equipment"], "weak": ["equipment", "automation", "machinery"]},
    "物流テック": {"strong": ["logistics technology", "warehouse automation", "logistics dx", "delivery optimization"], "weak": ["logistics system", "warehouse system", "supply chain software"]},
    "3Dプリンター": {"strong": ["3d printer", "3d printing", "additive manufacturing"], "weak": ["rapid prototyping"]},
    "機械": {"strong": ["industrial machinery", "machinery manufacturing", "machine manufacturer", "mechanical equipment"], "weak": ["equipment", "machine"]},
    "自動運転車": {"strong": ["autonomous driving", "self-driving", "automated driving", "adas"], "weak": ["vehicle sensing", "automotive camera", "vehicle antenna"]},
    "電気自動車関連": {"strong": ["electric vehicle", " ev ", "ev charger", "vehicle charging", "v2h"], "weak": ["automotive electrification", "battery vehicle"]},
    "自動車電子化": {"strong": ["automotive electronics", "vehicle electronics", "electronic control unit", "ecu", "adas"], "weak": ["automotive electronic", "in-vehicle"]},
    "自動車製造装置": {"strong": ["automotive manufacturing equipment", "automobile production equipment", "vehicle assembly equipment"], "weak": ["factory automation", "production line"]},
    "自動車軽量化": {"strong": ["automotive lightweight", "vehicle lightweight", "lightweight materials for vehicles"], "weak": ["aluminum automotive", "carbon fiber automotive", "resin automotive"]},
    "全固体電池": {"strong": ["solid-state battery", "all-solid-state battery"], "weak": ["battery material", "next-generation battery"]},
    "リチウムイオン電池": {"strong": ["lithium-ion battery", "lithium ion battery", "li-ion battery"], "weak": ["battery"]},
    "自動車部材・部品": {"strong": ["automotive parts", "automotive components", "vehicle components", "auto parts"], "weak": ["automotive", "vehicle"]},
    "MaaS": {"strong": ["mobility as a service", " maas", "mobility platform"], "weak": ["mobility service", "transportation platform"]},
    "空飛ぶクルマ": {"strong": ["flying car", "evtol", "air mobility"], "weak": ["urban air mobility", "electric aircraft"]},
    "再生可能エネルギー": {"strong": ["renewable energy", "renewable power", "green power"], "weak": ["solar power", "wind power", "clean energy"]},
    "太陽光発電関連": {"strong": ["solar power", "solar photovoltaic", "photovoltaic", "solar generation"], "weak": ["solar"]},
    "風力発電": {"strong": ["wind power", "wind turbine", "wind farm"], "weak": ["wind energy"]},
    "原子力発電": {"strong": ["nuclear power", "nuclear plant", "nuclear energy"], "weak": ["nuclear"]},
    "核融合発電": {"strong": ["nuclear fusion", "fusion power", "fusion energy"], "weak": ["fusion"]},
    "水素": {"strong": ["hydrogen business", "hydrogen energy", "hydrogen production", "fuel cell"], "weak": ["hydrogen"]},
    "アンモニア": {"strong": ["ammonia fuel", "ammonia energy", "green ammonia"], "weak": ["ammonia"]},
    "蓄電池": {"strong": ["storage battery", "energy storage system", "battery storage", "stationary battery"], "weak": ["battery", "energy storage"]},
    "省エネ関連": {"strong": ["energy saving", "energy efficiency", "energy conservation"], "weak": ["efficient energy", "energy management"]},
    "脱炭素": {"strong": ["decarbonization", "decarbonisation", "carbon neutral", "carbon neutrality", "low carbon"], "weak": ["renewable energy", "green energy", "co2 reduction"]},
    "インバウンド": {"strong": ["inbound tourism", "foreign tourists", "international visitors"], "weak": ["tourism", "hotel", "duty-free", "restaurant"]},
    "旅行": {"strong": ["travel agency", "travel services", "tour operator", "travel business"], "weak": ["tourism", "travel"]},
    "ホテル": {"strong": ["hotel business", "operates hotels", "hotel management", "hotels and resorts"], "weak": ["hotel", "accommodation"]},
    "外食": {"strong": ["restaurant", "food service", "dining business", "eating out"], "weak": ["franchise stores", "bar", "cafe"]},
    "小売り": {"strong": ["retail stores", "retailing", "retail business", "discount stores", "supermarket"], "weak": ["retail", "stores"]},
    "食品": {"strong": ["food products", "food manufacturing", "packaged foods", "food business", "beverage"], "weak": ["foods", "meat products", "seafood", "confectionery"]},
    "円高メリット": {"strong": ["imports raw materials", "imported raw materials", "import business"], "weak": ["imports", "purchases internationally", "import/export", "overseas procurement"], "macro": True},
    "円安メリット": {"strong": ["export business", "exports products", "export sales"], "weak": ["exports", "international sales", "overseas sales", "global sales"], "macro": True},
    "生活防衛": {"strong": ["discount store", "low price", "value retailer", "discount supermarket"], "weak": ["supermarket", "food retail", "daily necessities"], "macro": True},
    "eコマース": {"strong": ["e-commerce", "ecommerce", "online marketplace", "online retail"], "weak": ["online sales", "internet shopping"]},
    "バイオテクノロジー関連": {"strong": ["biotechnology", "biotech", "biopharmaceutical"], "weak": ["life science", "drug discovery"]},
    "創薬": {"strong": ["drug discovery", "drug development", "pharmaceutical research", "therapeutic development"], "weak": ["clinical development", "new drug"]},
    "再生医療": {"strong": ["regenerative medicine", "cell therapy", "stem cell therapy", "tissue engineering"], "weak": ["cell-based therapy", "stem cells"]},
    "医療機器": {"strong": ["medical device", "medical equipment", "diagnostic device"], "weak": ["healthcare equipment", "surgical"]},
    "遠隔医療": {"strong": ["telemedicine", "telehealth", "remote medical", "online medical"], "weak": ["online healthcare", "remote healthcare"]},
    "ホームヘルスケア": {"strong": ["home healthcare", "home health care", "home medical care", "home care equipment"], "weak": ["healthcare at home", "home nursing"]},
    "介護関連": {"strong": ["nursing care", "elderly care", "long-term care", "care facility"], "weak": ["care services", "senior care"]},
    "認知症薬": {"strong": ["dementia drug", "alzheimer", "dementia treatment"], "weak": ["dementia", "neurodegenerative"]},
    "医薬品関連": {"strong": ["pharmaceutical", "drug products", "medicines", "pharma"], "weak": ["drug discovery", "clinical development"]},
    "不動産関連": {"strong": ["real estate business", "property development", "real estate development", "real estate brokerage"], "weak": ["real estate", "property management"]},
    "建設": {"strong": ["construction business", "construction work", "civil engineering", "general contractor", "engineering and construction"], "weak": ["construction", "building work"]},
    "マンション関連": {"strong": ["condominium", "apartment development", "apartment building", "multi-family"], "weak": ["residential development", "housing"]},
    "住宅関連": {"strong": ["home construction", "housing business", "residential construction", "custom-built homes", "house builder"], "weak": ["housing", "residential"]},
    "不動産テック": {"strong": ["proptech", "real estate technology", "real estate dx", "property technology"], "weak": ["real estate software", "property app", "real estate cloud"]},
    "不動産ファンド": {"strong": ["real estate fund", "reit", "property fund", "real estate investment trust"], "weak": ["asset management", "real estate investment"]},
    "再開発": {"strong": ["redevelopment project", "urban redevelopment", "city redevelopment"], "weak": ["urban development", "property development"]},
    "建設DX": {"strong": ["construction dx", "construction digital transformation", "bim", "construction technology"], "weak": ["construction software", "construction management system"]},
    "海運": {"strong": ["shipping business", "marine transportation", "ocean shipping", "shipping company"], "weak": ["shipping", "vessel"]},
    "物流": {"strong": ["logistics business", "logistics services", "transportation and logistics", "supply chain logistics"], "weak": ["logistics", "distribution services"]},
    "倉庫": {"strong": ["warehouse business", "warehousing", "storage services", "cold storage"], "weak": ["warehouse", "storage"]},
    "陸運": {"strong": ["trucking", "road transportation", "land transportation", "truck transport"], "weak": ["transportation", "delivery"]},
    "鉄道関連": {"strong": ["railway", "railroad", "rail infrastructure", "rail construction"], "weak": ["rail", "train"]},
    "航空": {"strong": ["airline", "air transportation", "aviation business", "air cargo"], "weak": ["aviation", "aircraft"]},
    "港湾運送": {"strong": ["port transportation", "port cargo", "stevedoring", "harbor transportation"], "weak": ["port logistics", "harbor"]},
    "宅配": {"strong": ["home delivery", "parcel delivery", "courier service", "last-mile delivery"], "weak": ["delivery service", "delivery companies"]},
    "SaaS": {"strong": ["software as a service", "saas", "cloud service business", "subscription software"], "weak": ["cloud service", "software platform"]},
    "クラウドコンピューティング": {"strong": ["cloud computing", "cloud service", "cloud platform", "cloud system"], "weak": ["cloud", "saas"]},
    "デジタルトランスフォーメーション": {"strong": ["digital transformation", " dx ", "dx consulting", "digitalization"], "weak": ["digital solution", "it consulting", "cloud service"]},
    "データベース": {"strong": ["database software", "database system", "database platform", "data management platform"], "weak": ["database", "data management"]},
    "IT関連": {"strong": ["information technology", "it services", "software development", "system integration", "information systems"], "weak": ["software", "digital solution", "cloud service"]},
    "サブスクリプション": {"strong": ["subscription service", "subscription business", "recurring revenue", "monthly subscription"], "weak": ["saas", "membership service"]},
    "電子政府": {"strong": ["e-government", "digital government", "government digitalization", "public sector digital"], "weak": ["government system", "municipal system"]},
    "マイナンバー": {"strong": ["my number", "individual number card", "japanese national id"], "weak": ["identity verification", "government id"]},
    "教育ICT": {"strong": ["education ict", "edtech", "digital education", "school ict"], "weak": ["education software", "learning platform"]},
    "ゲーム関連": {"strong": ["video game", "game development", "mobile game", "online game"], "weak": ["gaming", "game content"]},
}

# Industry/sector evidence for broad themes. These are deliberately narrow.
INDUSTRY_BOOST: dict[str, list[str]] = {
    "食品": ["packaged foods", "farm products", "beverages", "confectioners", "food distribution"],
    "外食": ["restaurants"],
    "小売り": ["discount stores", "department stores", "specialty retail", "grocery stores"],
    "建設": ["engineering & construction", "building products & equipment"],
    "住宅関連": ["residential construction"],
    "不動産関連": ["real estate", "real estate services", "real estate - development"],
    "銀行": ["banks - regional", "banks - diversified"],
    "地方銀行": ["banks - regional"],
    "証券": ["capital markets"],
    "保険": ["insurance - life", "insurance - diversified", "insurance - property & casualty"],
    "半導体": ["semiconductors", "semiconductor equipment & materials"],
    "半導体製造装置": ["semiconductor equipment & materials"],
    "SaaS": ["software - application", "software - infrastructure"],
    "IT関連": ["information technology services", "software - application", "software - infrastructure"],
    "再生可能エネルギー": ["utilities - renewable"],
    "太陽光発電関連": ["solar"],
    "創薬": ["biotechnology", "drug manufacturers - specialty & generic"],
    "バイオテクノロジー関連": ["biotechnology"],
    "医薬品関連": ["drug manufacturers - specialty & generic", "drug manufacturers - general"],
    "介護関連": ["medical care facilities"],
    "物流": ["integrated freight & logistics"],
    "陸運": ["trucking"],
    "海運": ["marine shipping"],
    "航空": ["airlines", "airports & air services"],
}

GROWTH_WORDS = [
    "develops", "developing", "development", "research", "r&d", "new business",
    "expands", "expansion", "launch", "commercialization", "strategic", "focuses",
    "next-generation", "next generation", "growth business", "investment",
]

MACRO_THEMES = {"円高メリット", "円安メリット", "金利上昇メリット", "生活防衛", "インバウンド"}


def norm_text(*parts: str) -> str:
    txt = " ".join(str(p or "") for p in parts).lower()
    txt = re.sub(r"\s+", " ", txt)
    return f" {txt} "


def hits(text: str, terms: list[str]) -> list[str]:
    found = []
    for term in terms:
        t = term.lower()
        if t in text and term not in found:
            found.append(term)
    return found


def band(score: int) -> str:
    if score >= 80:
        return "主力テーマ"
    if score >= 60:
        return "有力関連"
    if score >= 40:
        return "補助関連"
    return "ノイズ候補"


def score_pair(theme: str, profile: dict[str, str]) -> dict[str, object]:
    rule = RULES.get(theme, {"strong": [], "weak": []})
    text = norm_text(profile.get("long_name", ""), profile.get("sector", ""), profile.get("industry", ""), profile.get("business_summary", ""))
    strong = hits(text, list(rule.get("strong", [])))
    weak = hits(text, list(rule.get("weak", [])))
    industry = (profile.get("industry") or "").lower()
    quote_type = (profile.get("quote_type") or "").upper()

    industry_hit = any(x in industry for x in INDUSTRY_BOOST.get(theme, []))

    # ETFs/ETNs are not operating companies, but a commodity fund can still have direct
    # exposure to a commodity theme. Keep this separate from company-business scoring.
    if quote_type in {"ETF", "MUTUALFUND"} or " etn" in text:
        if strong or weak or theme.lower() in text:
            return {
                "current": 100, "growth": 100, "score": 100, "confidence": "A",
                "evidence": "投資商品としてテーマ価格へ直接連動", "review": "non_operating_security",
                "strong_hits": strong, "weak_hits": weak,
            }
        return {
            "current": 0, "growth": 0, "score": 0, "confidence": "C",
            "evidence": "事業会社ではないため事業関連度の対象外", "review": "non_operating_security",
            "strong_hits": strong, "weak_hits": weak,
        }

    current = 8
    if strong:
        current = max(current, 68 + min(22, 8 * len(strong)))
    if weak:
        current = max(current, 38 + min(20, 6 * len(weak)))
    if industry_hit:
        current = max(current, 82)
        current = min(100, current + (5 if strong else 0))

    # Strong hits appearing early in the company description are more likely to describe
    # the core business rather than a peripheral activity.
    summary = (profile.get("business_summary") or "").lower()
    early = summary[:450]
    early_strong = [t for t in strong if t.lower() in early]
    if early_strong:
        current = min(100, current + 8)

    # Penalize a match that is only generic and has no strong or industry evidence.
    if not strong and weak and not industry_hit:
        current = min(current, 58)
    if not strong and not weak and not industry_hit:
        current = 8

    # Macro-benefit themes require explicit profit sensitivity that generic company
    # profiles seldom prove. Cap them unless a very direct phrase is present.
    if theme in MACRO_THEMES:
        if strong:
            current = min(current, 68)
        else:
            current = min(current, 52)

    growth_hits = hits(text, GROWTH_WORDS)
    growth = int(round(current * 0.72))
    if strong and growth_hits:
        growth = min(100, growth + 18 + min(12, 3 * len(growth_hits)))
    elif weak and growth_hits:
        growth = min(85, growth + 12)
    if theme in MACRO_THEMES:
        growth = min(growth, 60)

    final = int(round(current * 0.70 + growth * 0.30))

    if final >= 85 and (strong or industry_hit):
        confidence = "A"
    elif final >= 55 and (strong or weak or industry_hit):
        confidence = "B"
    else:
        confidence = "C"
    if theme in MACRO_THEMES:
        confidence = "C" if final < 65 else "B"

    evidence_parts = []
    if industry_hit:
        evidence_parts.append(f"industry={profile.get('industry','')}")
    if strong:
        evidence_parts.append("direct=" + "/".join(strong[:3]))
    if weak and not strong:
        evidence_parts.append("adjacent=" + "/".join(weak[:3]))
    if growth_hits and (strong or weak):
        evidence_parts.append("growth=" + "/".join(growth_hits[:2]))
    if not evidence_parts:
        evidence_parts.append("事業概要に直接根拠を確認できず")

    review = "ok"
    if confidence == "C" or 35 <= final < 80 or not profile.get("business_summary"):
        review = "needs_review"
    if theme in MACRO_THEMES:
        review = "needs_review_macro"

    return {
        "current": current,
        "growth": growth,
        "score": final,
        "confidence": confidence,
        "evidence": "; ".join(evidence_parts),
        "review": review,
        "strong_hits": strong,
        "weak_hits": weak,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--dir", default="research/results/theme_relevance_batches")
    args = ap.parse_args()
    root = Path(args.dir)
    profiles_path = root / f"batch_{args.batch:03d}_profiles.csv"
    memberships_path = root / f"batch_{args.batch:03d}_memberships.csv"
    out_path = root / f"batch_{args.batch:03d}_scores_auto.csv"
    summary_path = root / f"batch_{args.batch:03d}_score_summary.json"

    with profiles_path.open(encoding="utf-8-sig") as f:
        profiles = {r["stock_code"]: r for r in csv.DictReader(f)}
    with memberships_path.open(encoding="utf-8-sig") as f:
        memberships = list(csv.DictReader(f))

    fields = [
        "batch", "stock_code", "long_name", "theme_name", "cluster",
        "current_business_score", "growth_relevance_score", "relevance_score",
        "band", "confidence", "review_flag", "evidence", "strong_hits", "weak_hits",
        "quote_type", "sector", "industry", "website",
    ]
    rows_out: list[dict[str, object]] = []
    for m in memberships:
        code = m["stock_code"]
        p = profiles.get(code, {})
        s = score_pair(m["theme_name"], p)
        rows_out.append({
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
            "strong_hits": " | ".join(s["strong_hits"]),
            "weak_hits": " | ".join(s["weak_hits"]),
            "quote_type": p.get("quote_type", ""),
            "sector": p.get("sector", ""),
            "industry": p.get("industry", ""),
            "website": p.get("website", ""),
        })

    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    score_counts = defaultdict(int)
    confidence_counts = defaultdict(int)
    review_counts = defaultdict(int)
    for r in rows_out:
        score_counts[r["band"]] += 1
        confidence_counts[r["confidence"]] += 1
        review_counts[r["review_flag"]] += 1
    summary = {
        "batch": args.batch,
        "stocks": len(profiles),
        "stock_theme_pairs": len(rows_out),
        "band_counts": dict(score_counts),
        "confidence_counts": dict(confidence_counts),
        "review_counts": dict(review_counts),
        "scoring": "0.70*current_business + 0.30*growth_relevance",
        "note": "Automated first-pass. needs_review rows require manual/official-IR QA before production use.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(out_path)


if __name__ == "__main__":
    main()
