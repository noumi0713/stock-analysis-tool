from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
ROOT=Path('research/results/theme_relevance_batches')
src=ROOT/'discretionary_review_1000_context.csv'
with src.open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
by=defaultdict(list)
for r in rows:
    if r.get('prior_confidence')=='C': by[(r['batch'],r['stock_code'],r['long_name'],r['website'])].append(r)
out=[]
for (batch,code,name,website),rs in by.items():
    specs=[]
    for r in rs:
        specs.append(f"{r['theme_name']}[cur={r['current_business_score']},grow={r['growth_relevance_score']},prior={r['prior_relevance_score']}]")
    out.append({'batch':batch,'stock_code':code,'long_name':name,'website':website,'theme_specs':' ; '.join(specs)})
out.sort(key=lambda r:(int(r['batch']),r['stock_code']))
p=ROOT/'discretionary_review_1000_low_conf_compact.csv'
with p.open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0].keys()));w.writeheader();w.writerows(out)
print(len(out))
