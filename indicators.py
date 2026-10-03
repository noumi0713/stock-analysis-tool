"""On-read calculations only: no file writes, caches, or persisted signals.
Run: python -m swing_data.indicators --code 7203 (JSON to stdout).
"""
import argparse
import json
import numpy as np
import pandas as pd


def wilder(values,period=14):
    values=np.asarray(values,dtype=float)
    if len(values)<period or not np.isfinite(values).all(): return None
    average=float(np.mean(values[:period]))
    for value in values[period:]: average=(average*(period-1)+value)/period
    return average


def calculate(frame, benchmark):
    frame=frame.sort_values('date').copy()
    if len(frame)!=120 or frame.date.duplicated().any():
        raise ValueError('Exactly 120 distinct sessions required')
    cols=['adj_close','adj_high','adj_low','close','volume']
    if not np.isfinite(frame[cols].to_numpy(dtype=float)).all():
        raise ValueError('Missing or non-finite raw inputs')
    close=frame.adj_close.to_numpy(dtype=float)
    high=frame.adj_high.to_numpy(dtype=float);low=frame.adj_low.to_numpy(dtype=float)
    if (close<=0).any() or (low<=0).any() or (high<low).any() or (high<close).any() or (low>close).any() or (frame.volume<0).any():
        raise ValueError('Invalid OHLCV')
    diff=np.diff(close)
    gain=wilder(np.maximum(diff,0));loss=wilder(np.maximum(-diff,0))
    rsi=50.0 if gain==loss==0 else 100.0 if loss==0 else 100-100/(1+gain/loss)
    tr=np.maximum.reduce([high[1:]-low[1:],np.abs(high[1:]-close[:-1]),np.abs(low[1:]-close[:-1])])
    atr=wilder(tr)
    turnover=(frame.close*frame.volume).to_numpy(dtype=float)
    baseline=float(turnover[-21:-1].mean())
    result={'as_of':str(frame.date.iloc[-1]),'input_sessions':120,'rsi_14':rsi,'atr_14_adjusted':atr,
        'atr_14_current_price_units':atr*float(frame.close.iloc[-1])/close[-1],
        'ma':{str(n):float(close[-n:].mean()) for n in [5,10,20,25,60,75]},
        'return_pct':{str(n):float((close[-1]/close[-n-1]-1)*100) for n in [5,20,60]},
        'turnover_ratio_20d':float(turnover[-1]/baseline) if baseline>0 else None,
        'relative_strength_pct':{},'benchmark':'TOPIX price index (stock uses adjusted close)',
        'definition':'Wilder arithmetic seed on first 14 changes/TR; MA adjusted close. Turnover is close*volume proxy, denominator preceding 20 sessions excluding today. RS=(stock gross return / index gross return -1)*100.'}
    benchmark=benchmark.sort_values('date')
    if benchmark.date.duplicated().any(): raise ValueError('Duplicate benchmark dates')
    index=benchmark.set_index('date').close
    for n in [5,20,60]:
        dates=frame.date.iloc[-n-1:].tolist()
        values=index.reindex(dates)
        result['relative_strength_pct'][str(n)]=float(((close[-1]/close[-n-1])/(values.iloc[-1]/values.iloc[0])-1)*100) if values.notna().all() and np.isfinite(values).all() and (values>0).all() else None
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--code',required=True)
    args=parser.parse_args()
    import re
    if not re.fullmatch(r'[0-9][0-9A-Z]{3}',args.code): parser.error('Invalid code')
    base='https://raw.githubusercontent.com/noumi0713/stock-analysis-tool/swing-data-120d-latest'
    frame=pd.read_csv(f'{base}/stocks/{args.code}.csv')
    markets=pd.read_csv(base+'/markets_120d.csv')
    print(json.dumps(calculate(frame,markets[markets.ticker=='^TOPX']),ensure_ascii=False,allow_nan=False))
