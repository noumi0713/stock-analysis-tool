from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path('dashboard-data/technical-backtest-3y')
MEMBERSHIPS = Path('research/data/theme_members_124.csv')
OUT = Path('research-output/theme-top10-abc-pullback-test.json')
MIN_VALID = 5
WEIGHTS = {
    'price_strength': 0.25,
    'turnover_inflow': 0.25,
    'breadth': 0.20,
    'relative_strength': 0.15,
    'persistence': 0.15,
}


def load_memberships():
    theme_members = defaultdict(set)
    with MEMBERSHIPS.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            t = str(row.get('yahoo_ticker') or '').strip()
            th = str(row.get('theme_name') or '').strip()
            if t and th:
                theme_members[th].add(t)
    if len(theme_members) != 124:
        raise ValueError(f'expected 124 themes, got {len(theme_members)}')
    return dict(theme_members)


def load_market(wanted):
    manifest = json.loads((DATA_DIR/'manifest.json').read_text(encoding='utf-8'))
    dates = pd.DatetimeIndex(pd.to_datetime(manifest['dates']))
    tickers = sorted(wanted)
    tcol = {t:i for i,t in enumerate(tickers)}
    close = np.full((len(dates), len(tickers)), np.nan)
    volume = np.full_like(close, np.nan)
    for shard in manifest['shards']:
        payload = json.loads((DATA_DIR/shard['path']).read_text(encoding='utf-8'))
        for t,bars in payload['bars'].items():
            j=tcol.get(t)
            if j is None or not bars: continue
            a=np.asarray(bars,float); idx=a[:,0].astype(int)
            close[idx,j]=a[:,4]; volume[idx,j]=a[:,5]
    return manifest,dates,tickers,close,volume


def pct_change(a,k):
    out=np.full_like(a,np.nan); prev=a[:-k]; cur=a[k:]
    ok=np.isfinite(prev)&np.isfinite(cur)&(prev>0)
    b=np.full_like(cur,np.nan); b[ok]=cur[ok]/prev[ok]-1
    out[k:]=b; return out


def forward_return(a,k):
    out=np.full_like(a,np.nan); cur=a[:-k]; fut=a[k:]
    ok=np.isfinite(cur)&np.isfinite(fut)&(cur>0)
    b=np.full_like(cur,np.nan); b[ok]=fut[ok]/cur[ok]-1
    out[:-k]=b; return out


def theme_median(matrix, member_idx):
    d=matrix.shape[0]; out=np.full((d,len(member_idx)),np.nan); cnt=np.zeros_like(out,int)
    for j,idx in enumerate(member_idx):
        x=matrix[:,idx]; cnt[:,j]=np.isfinite(x).sum(1)
        with np.errstate(all='ignore'): out[:,j]=np.nanmedian(x,axis=1)
        out[cnt[:,j]==0,j]=np.nan
    return out,cnt


def theme_fraction(mask,valid,member_idx):
    out=np.full((mask.shape[0],len(member_idx)),np.nan)
    for j,idx in enumerate(member_idx):
        v=valid[:,idx]; n=v.sum(1); num=(mask[:,idx]&v).sum(1); ok=n>0
        out[ok,j]=num[ok]/n[ok]
    return out


def xsec_pct(raw,eligible):
    return pd.DataFrame(np.where(eligible,raw,np.nan)).rank(axis=1,pct=True).to_numpy()*100


def summarize(vals):
    a=np.asarray(vals,float); a=a[np.isfinite(a)]
    if len(a)==0: return {'n':0}
    return {
        'n':int(len(a)),
        'mean_pct':round(a.mean()*100,4),
        'median_pct':round(np.median(a)*100,4),
        'positive_rate_pct':round((a>0).mean()*100,2),
        'plus5_rate_pct':round((a>=.05).mean()*100,2),
        'minus5_rate_pct':round((a<=-.05).mean()*100,2),
    }


def main():
    members=load_memberships(); themes=sorted(members); wanted=set().union(*members.values())
    manifest,dates,tickers,close,volume=load_market(wanted); tcol={t:i for i,t in enumerate(tickers)}
    idxs=[np.array([tcol[t] for t in sorted(members[th]) if t in tcol],int) for th in themes]

    ret1=pct_change(close,1); ret5=pct_change(close,5); ret10=pct_change(close,10)
    ma25=pd.DataFrame(close).rolling(25,min_periods=20).mean().to_numpy()
    th1,c1=theme_median(ret1,idxs); th5,c5=theme_median(ret5,idxs); th10,c10=theme_median(ret10,idxs)
    price_raw=.6*th5+.4*th10

    membership_count=np.zeros(len(tickers)); M=np.zeros((len(tickers),len(themes)))
    for j,idx in enumerate(idxs): M[idx,j]=1; membership_count[idx]+=1
    membership_count[membership_count==0]=1
    turnover=np.nan_to_num((close*volume)/membership_count[None,:],nan=0,posinf=0,neginf=0)
    th_turn=turnover@M
    prior20=pd.DataFrame(th_turn).shift(1).rolling(20,min_periods=10).median().to_numpy()
    turn_raw=np.where(prior20>0,th_turn/prior20-1,np.nan)

    pos5=theme_fraction(ret5>0,np.isfinite(ret5),idxs)
    above25=theme_fraction(close>ma25,np.isfinite(close)&np.isfinite(ma25),idxs)
    breadth=.5*pos5+.5*above25
    with np.errstate(all='ignore'): market5=np.nanmedian(ret5,axis=1)
    relative=th5-market5[:,None]
    posday=np.where(np.isfinite(th1),(th1>0).astype(float),np.nan)
    persistence=pd.DataFrame(posday).rolling(5,min_periods=3).mean().to_numpy()

    eligible=(c5>=MIN_VALID)&(c10>=MIN_VALID)&np.isfinite(price_raw)&np.isfinite(turn_raw)&np.isfinite(breadth)&np.isfinite(relative)&np.isfinite(persistence)
    factors={'price_strength':price_raw,'turnover_inflow':turn_raw,'breadth':breadth,'relative_strength':relative,'persistence':persistence}
    fp={k:xsec_pct(v,eligible) for k,v in factors.items()}
    score=sum(WEIGHTS[k]*fp[k] for k in WEIGHTS); score[~eligible]=np.nan
    rank=pd.DataFrame(score).rank(axis=1,method='first',ascending=False).to_numpy()

    fwd={}
    for h in (5,10,20):
        sf=forward_return(close,h); fwd[h],fc=theme_median(sf,idxs); fwd[h][fc<MIN_VALID]=np.nan

    top10=eligible&(rank<=10)
    prior_top10_5=np.zeros_like(top10,bool)
    for lag in range(1,6):
        prior_top10_5[lag:] |= top10[:-lag]

    prev_top10=np.vstack([np.zeros((1,top10.shape[1]),bool), top10[:-1]])
    # Daily-state definitions matching prior discussion.
    A_daily = top10 & prev_top10
    B_daily = eligible & (rank>=11) & (rank<=30) & prior_top10_5
    C_daily = eligible & (rank>=31) & prior_top10_5

    # Event definitions to avoid repeatedly counting the same state.
    streak=np.zeros_like(rank,int)
    for i in range(len(dates)):
        if i==0: streak[i]=top10[i].astype(int)
        else: streak[i]=np.where(top10[i],streak[i-1]+1,0)
    A_event = top10 & (streak==3)  # first day a theme reaches 3 consecutive Top10 days
    left_top10 = eligible & (~top10) & prev_top10
    B_event = left_top10 & (rank>=11) & (rank<=30)
    C_event = left_top10 & (rank>=31)

    states={'A_daily':A_daily,'B_daily':B_daily,'C_daily':C_daily,'A_event':A_event,'B_event':B_event,'C_event':C_event}

    out={
        'meta':{
            'start':manifest['meta']['startDate'],
            'end':manifest['meta']['endDate'],
            'themes':len(themes),
            'days':len(dates),
            'recent_window_days':5,
            'definitions':{
                'A_daily':'Current Top10 and previous trading day also Top10',
                'B_daily':'Current rank 11-30 and was Top10 at least once in prior 5 trading days',
                'C_daily':'Current rank 31+ and was Top10 at least once in prior 5 trading days',
                'A_event':'First day reaching 3 consecutive Top10 days',
                'B_event':'First day leaving Top10 directly into rank 11-30',
                'C_event':'First day leaving Top10 directly into rank 31+'
            },
            'score':'price25 + turnover25 + breadth20 + relative15 + persistence15'
        },
        'results':{},
        'by_year':{}
    }
    for name,mask in states.items():
        out['results'][name]={str(h):summarize(fwd[h][mask]) for h in (5,10,20)}
        out['by_year'][name]={}
        for year in sorted(set(dates.year)):
            ym=(dates.year==year)[:,None]
            out['by_year'][name][str(year)]={str(h):summarize(fwd[h][mask&ym]) for h in (5,10,20)}

    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
