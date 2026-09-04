from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path('research/results/theme_relevance_batches')
src=ROOT/'discretionary_review_1000_set2_context.csv'
with src.open(encoding='utf-8-sig') as f:
    rows=list(csv.DictReader(f))

by=defaultdict(list)
for r in rows:
    if r.get('prior_confidence')=='C':
        by[(r['batch'],r['stock_code'],r['long_name'],r.get('website',''))].append(r)

out=[]
for (batch,code,name,website), rs in by.items():
    out.append({
        'batch':batch,'stock_code':code,'long_name':name,'website':website,
        'themes':' | '.join(r['theme_name'] for r in rs),
        'theme_count':len(rs),
        'evidence':' || '.join(f"{r['theme_name']}: {r.get('evidence','')}" for r in rs),
        'official_urls':' | '.join(dict.fromkeys(u for r in rs for u in (r.get('official_urls') or '').split(' | ') if u))
    })
out.sort(key=lambda r:(int(r['batch']),r['stock_code']))
path=ROOT/'discretionary_review_1000_set2_low_conf_companies.csv'
fields=['batch','stock_code','long_name','website','themes','theme_count','evidence','official_urls']
with path.open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
summary={'set':2,'low_conf_rows':sum(len(v) for v in by.values()),'unique_companies':len(by),'max_themes_per_company':max((len(v) for v in by.values()),default=0)}
(ROOT/'discretionary_review_1000_set2_low_conf_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
