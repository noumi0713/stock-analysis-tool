from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('research/results/theme_relevance_batches')
LIMIT = 1000

# Exclude both previously finalized 1000-row sets.
finalized = set()
for name in ('discretionary_review_1000_final.csv', 'discretionary_review_1000_set2_final.csv'):
    p = ROOT / name
    if not p.exists():
        continue
    with p.open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            finalized.add((r['stock_code'], r['theme_name']))

rows = []
for batch in range(1, 5):
    v6 = ROOT / f'batch_{batch:03d}_scores_v6.csv'
    v3 = ROOT / f'batch_{batch:03d}_scores_v3.csv'
    if not v6.exists() or not v3.exists():
        continue
    with v3.open(encoding='utf-8-sig') as f:
        prior = {(r['stock_code'], r['theme_name']): r for r in csv.DictReader(f)}
    with v6.open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r.get('review_flag') != 'revenue_share_unknown':
                continue
            key = (r['stock_code'], r['theme_name'])
            if key in finalized:
                continue
            p = prior.get(key, {})
            rows.append({
                'batch': batch,
                'stock_code': r['stock_code'],
                'long_name': r.get('long_name','') or p.get('long_name',''),
                'theme_name': r['theme_name'],
                'cluster': r.get('cluster','') or p.get('cluster',''),
                'current_business_score': p.get('current_business_score',''),
                'growth_relevance_score': p.get('growth_relevance_score',''),
                'prior_relevance_score': p.get('relevance_score',''),
                'prior_band': p.get('band',''),
                'prior_confidence': p.get('confidence',''),
                'prior_review_flag': p.get('review_flag',''),
                'evidence': p.get('evidence',''),
                'official_hits': p.get('official_hits',''),
                'official_hit_count': p.get('official_hit_count',''),
                'official_urls': p.get('official_urls',''),
                'sector': p.get('sector',''),
                'industry': p.get('industry',''),
                'website': p.get('website',''),
            })

rows = rows[:LIMIT]
if len(rows) < LIMIT:
    raise SystemExit(f'Need {LIMIT} new rows but only found {len(rows)}')

ctx = ROOT / 'discretionary_review_1000_set3_context.csv'
fields = list(rows[0].keys())
with ctx.open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

summary = {
    'set': 3,
    'rows': len(rows),
    'excluded_previous_finalized': len(finalized),
    'batch_counts': dict(Counter(str(r['batch']) for r in rows)),
    'confidence': dict(Counter(r['prior_confidence'] for r in rows)),
    'review_flags': dict(Counter(r['prior_review_flag'] for r in rows)),
    'bands': dict(Counter(r['prior_band'] for r in rows)),
}
(ROOT / 'discretionary_review_1000_set3_context_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

low = [r for r in rows if r['prior_confidence'] == 'C']
by_company = defaultdict(list)
for r in low:
    by_company[(r['batch'], r['stock_code'], r['long_name'], r['website'])].append(r)
company_rows = []
for (batch, code, name, website), rs in sorted(by_company.items(), key=lambda x: (int(x[0][0]), x[0][1])):
    company_rows.append({
        'batch': batch, 'stock_code': code, 'long_name': name, 'website': website,
        'theme_count': len(rs),
        'themes': ' ; '.join(f"{r['theme_name']}[cur={r['current_business_score']},grow={r['growth_relevance_score']},prior={r['prior_relevance_score']}]" for r in rs),
    })
with (ROOT / 'discretionary_review_1000_set3_low_conf_companies.csv').open('w', newline='', encoding='utf-8-sig') as f:
    fs = ['batch','stock_code','long_name','website','theme_count','themes']
    w = csv.DictWriter(f, fieldnames=fs); w.writeheader(); w.writerows(company_rows)
with (ROOT / 'discretionary_review_1000_set3_low_conf_compact.csv').open('w', newline='', encoding='utf-8-sig') as f:
    fs = ['batch','stock_code','long_name','website','theme_specs']
    w = csv.DictWriter(f, fieldnames=fs); w.writeheader()
    for c in company_rows:
        w.writerow({'batch':c['batch'],'stock_code':c['stock_code'],'long_name':c['long_name'],'website':c['website'],'theme_specs':c['themes']})
(ROOT / 'discretionary_review_1000_set3_low_conf_summary.json').write_text(json.dumps({'set':3,'low_conf_rows':len(low),'unique_companies':len(company_rows),'max_themes_per_company':max((int(c['theme_count']) for c in company_rows), default=0)}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False))
print('low_conf_rows', len(low), 'companies', len(company_rows))
