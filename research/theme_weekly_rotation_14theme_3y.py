from __future__ import annotations
import csv, json, math, statistics
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'dashboard-data'/'technical-backtest-3y'
MEMBERS=ROOT/'dashboard-data'/'theme_members.csv'
OUT=ROOT/'research'/'results'/'theme_weekly_rotation_14theme_3y'
N=5

def ws(d): return d-timedelta(days=d.weekday())
def avg(x): return statistics.fmean(x) if x else None
def med(x): return statistics.median(x) if x else None
def pct(x): return None if x is None else round(x*100,4)
def smry(x):
    a=[float(v) for v in x if v is not None and math.isfinite(float(v))]
    return {'n':len(a),'mean_pct':pct(avg(a)) if a else None,'median_pct':pct(med(a)) if a else None,'positive_rate_pct':round(sum(v>0 for v in a)/len(a)*100,2) if a else None}

# Current theme snapshot. A stock's weekly volume is divided by its number of theme memberships.
themes_by_ticker=defaultdict(list); members=defaultdict(set)
with MEMBERS.open('r',encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        t=(r.get('yahoo_ticker') or '').strip(); th=(r.get('theme_name') or '').strip()
        if t and th:
            if th not in themes_by_ticker[t]: themes_by_ticker[t].append(th)
            members[th].add(t)

m=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
dates=[date.fromisoformat(x) for x in m['dates']]
stock=defaultdict(lambda:defaultdict(lambda:{'v':0.0,'d':None,'c':None}))
for sh in m['shards']:
    p=json.loads((DATA/sh['path']).read_text(encoding='utf-8'))
    for t,bars in p['bars'].items():
        if t not in themes_by_ticker: continue
        for b in bars:
            d=dates[int(b[0])]; w=ws(d); z=stock[t][w]
            z['v']+=float(b[5])
            if z['d'] is None or d>z['d']: z['d']=d; z['c']=float(b[4])

vol=defaultdict(float); rets=defaultdict(list)
for t,weeks in stock.items():
    ths=themes_by_ticker[t]; div=len(ths); prevw=None; prevc=None
    for w,z in sorted(weeks.items()):
        for th in ths: vol[(th,w)]+=z['v']/div
        if prevw is not None and (w-prevw).days==7 and prevc and z['c']:
            r=z['c']/prevc-1
            for th in ths: rets[(th,w)].append(r)
        prevw=w; prevc=z['c']

rows=[]; by={}
weeks=sorted({w for _,w in vol})
for th in sorted(members):
    for w in weeks:
        rr=rets.get((th,w),[]); vv=vol.get((th,w),0)
        if len(rr)<3 or vv<=0: continue
        z={'theme':th,'week_start':w,'week_end':w+timedelta(days=4),'volume':vv,'return':med(rr),'valid_members':len(rr)}
        rows.append(z); by[(th,w)]=z
for z in rows:
    prev=by.get((z['theme'],z['week_start']-timedelta(days=7)))
    nxt=by.get((z['theme'],z['week_start']+timedelta(days=7)))
    z['volume_change']=z['volume']/prev['volume']-1 if prev and prev['volume']>0 else None
    z['next_return']=nxt['return'] if nxt else None

bw=defaultdict(list)
for z in rows: bw[z['week_start']].append(z)
rankings=[]; obs=defaultdict(list); spreads=defaultdict(list); combos=defaultdict(list); yearly=defaultdict(lambda:defaultdict(list))
for w,it in sorted(bw.items()):
    vi=[z for z in it if z['volume_change'] is not None and z['next_return'] is not None]
    ri=[z for z in it if z['return'] is not None and z['next_return'] is not None]
    if len(vi)<N*2 or len(ri)<N*2: continue
    vs=sorted(vi,key=lambda z:z['volume_change'],reverse=True); rs=sorted(ri,key=lambda z:z['return'],reverse=True)
    groups={'volume_top5':vs[:N],'volume_bottom5':vs[-N:],'return_top5':rs[:N],'return_bottom5':rs[-N:]}
    for label,g in groups.items():
        ordered=g if 'top' in label else list(reversed(g))
        for rank,z in enumerate(ordered,1):
            rankings.append({'week_start':w.isoformat(),'week_end':z['week_end'].isoformat(),'ranking':label,'rank':rank,'theme':z['theme'],'volume_change_pct':pct(z['volume_change']),'weekly_return_pct':pct(z['return']),'next_week_return_pct':pct(z['next_return']),'valid_members':z['valid_members']})
            obs[label].append(z['next_return']); yearly[str(w.year)][label].append(z['next_return'])
    spreads['volume_top_minus_bottom'].append(avg([z['next_return'] for z in vs[:N]])-avg([z['next_return'] for z in vs[-N:]]))
    spreads['return_top_minus_bottom'].append(avg([z['next_return'] for z in rs[:N]])-avg([z['next_return'] for z in rs[-N:]]))
    sets={k:{z['theme'] for z in g} for k,g in groups.items()}; mp={z['theme']:z for z in it}
    for label,names in {
        'volume_top5_and_return_top5':sets['volume_top5']&sets['return_top5'],
        'volume_top5_and_return_bottom5':sets['volume_top5']&sets['return_bottom5'],
        'volume_bottom5_and_return_top5':sets['volume_bottom5']&sets['return_top5'],
        'volume_bottom5_and_return_bottom5':sets['volume_bottom5']&sets['return_bottom5']}.items():
        for th in names: combos[label].append(mp[th]['next_return'])

res={'meta':{'data_start':m['meta']['startDate'],'data_end':m['meta']['endDate'],'stock_count':m['meta']['stockCount'],'theme_count':len(members),'theme_membership_rows':sum(len(x) for x in members.values()),'top_n':N,'mode':'provisional_14_theme_test_current_membership_snapshot_applied_historically','volume_definition':'split-adjusted weekly share volume, apportioned by number of theme memberships, week-over-week change','return_definition':'median constituent weekly close-to-close return','signal_timing':'rank after week close, evaluate next week'},'baseline':smry([z['next_return'] for z in rows]),'groups':{k:smry(v) for k,v in sorted(obs.items())},'spreads':{k:smry(v) for k,v in sorted(spreads.items())},'combinations':{k:smry(v) for k,v in sorted(combos.items())},'yearly':{y:{k:smry(v) for k,v in sorted(g.items())} for y,g in sorted(yearly.items())},'ranking_week_count':len({r['week_start'] for r in rankings}),'ranking_row_count':len(rankings)}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'summary.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'weekly_rankings.csv').open('w',encoding='utf-8-sig',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=list(rankings[0].keys())); wr.writeheader(); wr.writerows(rankings)
print(json.dumps(res,ensure_ascii=False,indent=2))
