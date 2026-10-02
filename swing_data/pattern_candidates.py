"""Causal OHLCV screening; monitoring candidates, not validated buy signals."""
from __future__ import annotations
import numpy as np
import pandas as pd

VERSION = 'triangle-bottom-v1'
RULES = {
    'history': '120営業日（日足・分割調整済み）',
    'triangle': '20/30/40/60日、確認済み高値・安値各3点、上辺下降・下辺上昇、幅20〜80%縮小、突破0.5%以上',
    'bottom': '60日高値から15%以上調整、5〜40日前の安値を維持、10日高値を0.5%以上突破、20日平均線上向き',
    'confirmation': '直近3営業日以内の突破、陽線、出来高が直前20日平均の1.3倍以上、引け位置が当日値幅の上位30%',
    'liquidity': '直前20日平均の推計売買代金1.5億円以上',
    'warning': '監視候補。勝率・期待値は未検証。底打ち確定や買い推奨を意味しません。',
}


def prepare(frame, price_date):
    need = {'date', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'close', 'volume'}
    if not need.issubset(frame.columns): return None
    f = frame.sort_values('date').drop_duplicates('date', keep='last').tail(120).copy()
    if len(f) < 80 or str(f.date.iloc[-1]) != price_date: return None
    for k in need - {'date'}: f[k] = pd.to_numeric(f[k], errors='coerce')
    if not np.isfinite(f[list(need-{'date'})].to_numpy()).all(): return None
    if (f[['adj_open','adj_high','adj_low','adj_close','close']] <= 0).any().any(): return None
    if (f.volume <= 0).any(): return None
    if ((f.adj_high < f[['adj_open','adj_close','adj_low']].max(axis=1)) | (f.adj_low > f[['adj_open','adj_close','adj_high']].min(axis=1))).any(): return None
    # Adjust historical volume to latest share units, preserving traded value across splits.
    f['v'] = f.volume * f.close / f.adj_close
    return f.reset_index(drop=True)


def pivots(values, high):
    # Last two bars only confirm previous pivots: no unconfirmed endpoint extrema.
    out = []
    for i in range(2, len(values)-2):
        neighbours = np.r_[values[i-2:i], values[i+1:i+3]]
        if (values[i] > neighbours.max() if high else values[i] < neighbours.min()): out.append(i)
    return np.array(out, dtype=int)


def triangle(f, t):
    for w in (60,40,30,20):
        if t < w: continue
        p = f.iloc[t-w:t]
        hi, lo = p.adj_high.to_numpy(), p.adj_low.to_numpy()
        hp, lp = pivots(hi,True), pivots(lo,False)
        if len(hp)<3 or len(lp)<3 or hp[-1]-hp[0]<10 or lp[-1]-lp[0]<10: continue
        hm,hb = np.polyfit(hp,hi[hp],1); lm,lb = np.polyfit(lp,lo[lp],1)
        if hm>=0 or lm<=0: continue
        start=max(hp[0],lp[0]); end=w-1
        width0=(hm*start+hb)-(lm*start+lb); width1=(hm*end+hb)-(lm*end+lb)
        if width0<=0 or not .2<=width1/width0<=.8: continue
        atr=float((p.adj_high-p.adj_low).tail(20).mean())
        if np.abs(hi[hp]-(hm*hp+hb)).max()>atr or np.abs(lo[lp]-(lm*lp+lb)).max()>atr: continue
        xs=np.arange(w); upper=hm*xs+hb; lower=lm*xs+lb
        if (p.adj_close.to_numpy()>upper*1.01).any() or (p.adj_close.to_numpy()<lower*.99).any(): continue
        resistance=float(hm*w+hb); support=float(lm*w+lb)
        if support<=0 or resistance<=support: continue
        if f.adj_close.iloc[t] <= resistance*1.005: continue
        latest_x=w+len(f)-1-t
        latest_upper=float(hm*latest_x+hb); latest_lower=float(lm*latest_x+lb)
        if f.adj_close.iloc[-1] < latest_upper or (f.adj_close.iloc[t:] < latest_lower*.99).any(): continue
        return {'kind':'triangle_breakout','label':'三角持ち合い上抜け候補','formation_start':str(p.date.iloc[0]),'formation_sessions':w,'signal_date':str(f.date.iloc[t]),'resistance':round(resistance,4),'invalidation_level':round(support,4),'width_contraction_pct':round((1-width1/width0)*100,2),'confirmed_high_pivots':len(hp),'confirmed_low_pivots':len(lp)}
    return None


def bottom(f,t):
    if t<65: return None
    p=f.iloc[t-60:t]
    low=p.adj_low.to_numpy(); idx=int(low.argmin()); age=60-idx
    if not 5<=age<=40: return None
    trough=float(low[idx]); peak=float(p.adj_high.iloc[:idx+1].max())
    if trough/peak-1 > -.15: return None
    if f.adj_low.iloc[t-5:].min() < trough*.985: return None
    level=float(f.adj_high.iloc[t-10:t].max())
    ma=float(f.adj_close.iloc[t-19:t+1].mean()); ma_old=float(f.adj_close.iloc[t-24:t-4].mean())
    if f.adj_close.iloc[t] <= max(level*1.005,ma,trough*1.03) or ma<=ma_old: return None
    if f.adj_close.iloc[-1]<level or f.adj_low.iloc[t:].min()<trough*.985: return None
    return {'kind':'bottom_reversal','label':'底打ち・戻り高値突破候補','formation_start':str(p.date.iloc[0]),'signal_date':str(f.date.iloc[t]),'bottom_date':str(p.date.iloc[idx]),'resistance':round(level,4),'invalidation_level':round(trough,4),'preceding_drawdown_pct':round((trough/peak-1)*100,2)}


def detect_triggers(frame,price_date):
    f=prepare(frame,price_date)
    if f is None: return []
    result=[]
    for t in range(len(f)-1,len(f)-4,-1):
        prior=f.iloc[t-20:t]
        liquidity=float((prior.close*prior.volume).mean())
        vol_ratio=float(f.v.iloc[t]/prior.v.mean())
        c,o,h,l=(float(f[k].iloc[t]) for k in ('adj_close','adj_open','adj_high','adj_low'))
        if liquidity<150_000_000 or vol_ratio<1.3 or c<=o or h<=l or (c-l)/(h-l)<.7: continue
        for candidate in (triangle(f,t),bottom(f,t)):
            if candidate and candidate['kind'] not in {x['kind'] for x in result}:
                candidate.update(volume_ratio=round(vol_ratio,3),average_trading_value_20d=round(liquidity),as_of=price_date,rule_version=VERSION,validated=False)
                result.append(candidate)
    return result


# The template is frozen. Similarity is descriptive, not a probability of profit.
SHAPE_VERSION = 'kioxia-120-calendar-days-v1'
def shape_similarity(frame, price_date, reference):
    f=prepare(frame,price_date)
    if f is None or not reference: return None
    start=(pd.Timestamp(price_date)-pd.Timedelta(days=120)).date().isoformat()
    p=f[f.date>=start]
    if len(p)<70 or pd.Timestamp(p.date.iloc[0])>pd.Timestamp(start)+pd.Timedelta(days=7): return None
    a=np.log(np.asarray(reference['adj_close'],dtype=float))
    b=np.log(p.adj_close.to_numpy())
    if a.std()<=1e-10 or b.std()<=1e-10: return None
    b=np.interp(np.linspace(0,1,len(a)),np.linspace(0,1,len(b)),b)
    corr=float(np.corrcoef(a,b)[0,1]); recent=float(np.corrcoef(a[-20:],b[-20:])[0,1])
    rmse=float(np.sqrt(np.mean(((a-a.mean())/a.std()-(b-b.mean())/b.std())**2)))
    ratio=float(np.ptp(b)/np.ptp(a)); peak_gap=abs(int(a.argmax())-int(b.argmax()))/len(a)
    low_gap=abs(int(a.argmin())-int(b.argmin()))/len(a)
    passed=corr>=.85 and recent>=.65 and rmse<=.55 and .5<=ratio<=2 and peak_gap<=.2 and low_gap<=.2
    return {'matches':passed,'correlation':round(corr,4),'recent_20_correlation':round(recent,4),'normalized_rmse':round(rmse,4),'amplitude_ratio':round(ratio,4),'peak_timing_gap':round(peak_gap,4),'bottom_timing_gap':round(low_gap,4),'comparison_start':str(p.date.iloc[0]),'comparison_end':price_date,'comparison_sessions':len(p)}


def detect(frame,price_date,reference=None):
    similarity=shape_similarity(frame,price_date,reference)
    if not similarity or not similarity['matches']: return []
    f=prepare(frame,price_date)
    liquidity=float((f.close*f.volume).tail(20).mean())
    if liquidity<150_000_000: return []
    triggers=detect_triggers(frame,price_date)
    return [{'kind':'kioxia_similar_shape','label':'キオクシア類似形状・監視候補','signal_date':price_date,'as_of':price_date,'rule_version':SHAPE_VERSION,'validated':False,'similarity':similarity,'trigger_labels':[x['label'] for x in triggers],'trigger_signals':triggers,'average_trading_value_20d':round(liquidity)}]
