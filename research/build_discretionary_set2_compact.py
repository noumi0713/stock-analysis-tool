from __future__ import annotations

import csv
from pathlib import Path

ROOT=Path('research/results/theme_relevance_batches')
src=ROOT/'discretionary_review_1000_set2_context.csv'
out=ROOT/'discretionary_review_1000_set2_low_conf_compact.csv'
with src.open(encoding='utf-8-sig') as f:
    rows=[r for r in csv.DictReader(f) if r.get('prior_confidence')=='C']

by={}
for r in rows:
    key=(r['batch'],r['stock_code'],r['long_name'],r.get('website',''))
    by.setdefault(key,[]).append(r)

records=[]
for (batch,code,name,website),rs in by.items():
    specs=[]
    for r in rs:
        specs.append(f"{r['theme_name']}[cur={r.get('current_business_score','')},grow={r.get('growth_relevance_score','')},prior={r.get('prior_relevance_score','')}]")
    records.append({'batch':batch,'stock_code':code,'long_name':name,'website':website,'theme_specs':' ; '.join(specs)})
records.sort(key=lambda r:(int(r['batch']),r['stock_code']))
with out.open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=['batch','stock_code','long_name','website','theme_specs']);w.writeheader();w.writerows(records)
print('companies',len(records),'rows',len(rows))
