from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path('research/results/theme_relevance_batches')
CTX = ROOT / 'discretionary_review_1000_context.csv'
OVR = ROOT / 'discretionary_review_1000_manual_overrides.csv'
OUT = ROOT / 'discretionary_review_1000_final.csv'
SUMMARY = ROOT / 'discretionary_review_1000_final_summary.json'
QA = ROOT / 'discretionary_review_1000_qa_sample.csv'


def band(score: float) -> str:
    if score >= 80: return '主力テーマ'
    if score >= 60: return '有力関連'
    if score >= 40: return '補助関連'
    return 'ノイズ候補'

with CTX.open(encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
with OVR.open(encoding='utf-8-sig') as f:
    overrides = {(r['stock_code'], r['theme_name']): r for r in csv.DictReader(f)}

out = []
manual = 0
for r in rows:
    key = (r['stock_code'], r['theme_name'])
    ov = overrides.get(key)
    try:
        prior = float(r.get('prior_relevance_score') or 0)
    except ValueError:
        prior = 0.0
    if ov:
        score = float(ov['final_relevance_score'])
        conf = ov['final_confidence']
        reason = ov['reason']
        source_url = ov['source_url']
        source = 'chatgpt_discretion_manual_web'
        manual += 1
    else:
        score = prior
        conf = r.get('prior_confidence') or 'C'
        reason = r.get('evidence') or '既存の会社公式情報・事業概要を再確認し、追加の強い根拠がないため既存裁量値を採用'
        source_url = r.get('official_urls') or r.get('website') or ''
        if conf in ('A','B'):
            source = 'chatgpt_confirmed_official_context'
        else:
            source = 'chatgpt_discretion_existing_context'

    out.append({
        'batch': r['batch'],
        'stock_code': r['stock_code'],
        'long_name': r['long_name'],
        'theme_name': r['theme_name'],
        'cluster': r['cluster'],
        'revenue_share_pct': '',
        'final_relevance_score': f'{score:.1f}',
        'final_band': band(score),
        'final_confidence': conf,
        'decision_source': source,
        'decision_reason': reason,
        'source_url': source_url,
        'current_business_score': r.get('current_business_score',''),
        'growth_relevance_score': r.get('growth_relevance_score',''),
        'prior_relevance_score': r.get('prior_relevance_score',''),
        'status': 'finalized_discretionary_v1',
    })

assert len(out) == 1000, len(out)
fields = list(out[0].keys())
with OUT.open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)

summary = {
    'rows': len(out),
    'manual_web_overrides': manual,
    'bands': dict(Counter(r['final_band'] for r in out)),
    'confidence': dict(Counter(r['final_confidence'] for r in out)),
    'decision_sources': dict(Counter(r['decision_source'] for r in out)),
    'unfinalized': sum(r['status'] != 'finalized_discretionary_v1' for r in out),
    'rule': 'Revenue share remains blank unless disclosed; relevance score is finalized using official-company context and ChatGPT discretion, with manual web overrides for detected false negatives.',
}
SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

# QA: all manual overrides + deterministic 50-row sample from the rest.
manual_rows = [r for r in out if r['decision_source'] == 'chatgpt_discretion_manual_web']
rest = [r for r in out if r['decision_source'] != 'chatgpt_discretion_manual_web']
rng = random.Random(20260904)
sample = manual_rows + rng.sample(rest, min(50, len(rest)))
with QA.open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(sample)

print(json.dumps(summary, ensure_ascii=False))
print('qa_rows', len(sample))
