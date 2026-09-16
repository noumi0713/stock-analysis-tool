from pathlib import Path
import pandas as pd,numpy as np,json
R=Path(__file__).resolve().parent/'swing-data-120d-latest';P=R/'bundle'
rank=pd.read_csv(R/'bbs_ranking_trends.csv',dtype={'stock_code':str})
markets=pd.read_csv(P/'markets_120d.csv');markets=markets[markets.date<='2026-09-14']
top=markets[markets.ticker=='^TOPX'].sort_values('date').set_index('date').close

def wilder(x,n=14):
 out=pd.Series(np.nan,index=x.index,dtype=float)
 if len(x)<n:return out
 out.iloc[n-1]=x.iloc[:n].mean()
 for i in range(n,len(x)):out.iloc[i]=(out.iloc[i-1]*(n-1)+x.iloc[i])/n
 return out
rows=[]
for s in rank.to_dict('records'):
 code=s['stock_code'];f=R/'stocks'/f'{code}.csv'
 if not f.exists():rows.append({**s,'processed':False,'reason':'no_data'});continue
 d=pd.read_csv(f).sort_values('date').drop_duplicates('date');d=d[d.date<='2026-09-14'].tail(120).reset_index(drop=True)
 valid=d[['open','high','low','close','volume']].notna().all(axis=1)&(d.close>0)&(d.volume>=0)&(d.high>=d[['open','close','low']].max(axis=1))&(d.low<=d[['open','close']].min(axis=1))
 if not len(d) or not valid.all() or d.iloc[-1].date!='2026-09-14':rows.append({**s,'processed':False,'reason':'invalid_or_stale','rows':len(d)});continue
 c=d.adj_close;h=d.adj_high;l=d.adj_low;o=d.adj_open;v=d.volume;n=len(d);last=d.iloc[-1]
 delta=c.diff();gain=wilder(delta.iloc[1:].clip(lower=0)).reindex(d.index);loss=wilder(-delta.iloc[1:].clip(upper=0)).reindex(d.index)
 rsi=(100-100/(1+gain/loss)).iloc[-1]
 tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1);atr=wilder(tr)
 tv=d.close*v;v20=v.tail(20).mean();tv20=tv.tail(20).mean() if n>=20 else np.nan
 out={**s,'processed':True,'rows':n,'date_price':last.date,'close':last.close,'open':last.open,'high':last.high,'low':last.low,'change_pct':(c.iloc[-1]/c.iloc[-2]-1)*100 if n>1 else np.nan,'volume':v.iloc[-1],'turnover_est':tv.iloc[-1],'turnover20_est':tv20,'turnover_ratio':tv.iloc[-1]/tv20,'volume_ratio':v.iloc[-1]/v20,'lot100':last.close*100,'rsi14':rsi,'atr14':atr.iloc[-1],'atr_pct':atr.iloc[-1]/c.iloc[-1]*100,'atr_change5_pct':(atr.iloc[-1]/atr.iloc[-6]-1)*100 if n>=20 else np.nan,'close_location':(c.iloc[-1]-l.iloc[-1])/(h.iloc[-1]-l.iloc[-1]) if h.iloc[-1]>l.iloc[-1] else np.nan,'upper_wick_pct':(h.iloc[-1]-max(c.iloc[-1],o.iloc[-1]))/c.iloc[-1]*100,'gap_pct':(o.iloc[-1]/c.iloc[-2]-1)*100 if n>1 else np.nan,'body_pct':(c.iloc[-1]/o.iloc[-1]-1)*100,'up_days5':int((delta.tail(5)>0).sum()),'up_with_volume5':int(((delta>0)&(v>v.shift())).tail(5).sum()),'down_with_volume5':int(((delta<0)&(v>v.shift())).tail(5).sum()),'volume5_over_prior5':v.tail(5).mean()/v.iloc[-10:-5].mean() if n>=10 else np.nan,'recent_split':bool(d.stock_splits.fillna(0).tail(30).ne(0).any())}
 for k in [5,25,75]:
  ma=c.rolling(k).mean();out['ma'+str(k)]=ma.iloc[-1];out['dev'+str(k)]=(c.iloc[-1]/ma.iloc[-1]-1)*100;out['ma'+str(k)+'_slope5']=(ma.iloc[-1]/ma.iloc[-6]-1)*100 if n>=k+5 else np.nan
 for k in [5,20,60]:
  ret=(c.iloc[-1]/c.iloc[-k-1]-1)*100 if n>k else np.nan;out['return'+str(k)]=ret
  if k!=5:
   start=d.date.iloc[-k-1] if n>k else None
   out['rs'+str(k)]=ret-(top.loc[last.date]/top.loc[start]-1)*100 if start in top.index else np.nan
 for k in [5,20,60,120]:
  out['high'+str(k)]=h.tail(k).max() if n>=k else np.nan;out['low'+str(k)]=l.tail(k).min() if n>=k else np.nan;out['distance_high'+str(k)]=(c.iloc[-1]/out['high'+str(k)]-1)*100;out['distance_low'+str(k)]=(c.iloc[-1]/out['low'+str(k)]-1)*100
 out['liquidity']='insufficient_history' if n<20 else 'priority' if tv20>=5e8 else 'eligible' if tv20>=1e8 else 'exclude'
 # Screening order only: this is not a calibrated probability or expected return.
 out['screen_score']=(2*(out['rs20']>0)+1*(out['rs60']>0)+1*(out['dev25']>0)+1*(out['ma25_slope5']>0)+1*(out['turnover_ratio']>=.8)+1*(out['close_location']>=.6)+1*(40<=rsi<=65)-2*(rsi>75)-2*(out['dev25']>15)-2*(out['return5']>15)-1*(out['lot100']>1e6)-2*(out['atr_pct']>8))
 rows.append(out)
out=pd.DataFrame(rows);out.to_csv(R/'all_ranking_metrics.csv',index=False)
print('COUNTS',out.processed.value_counts().to_dict(),'liquidity',out.liquidity.value_counts().to_dict())
print(out[out.processed & (out.liquidity!='exclude')].sort_values(['screen_score','rs20'],ascending=False)[['stock_code','stock_name','rows','close','change_pct','turnover20_est','turnover_ratio','dev25','rsi14','atr_pct','return5','rs20','rs60','close_location','screen_score']].round(2).to_string(index=False))
ms=[]
for t,g in markets.groupby('ticker'):
 g=g.sort_values('date');ms.append({'ticker':t,'date':g.date.iloc[-1],'close':g.close.iloc[-1],'return1':(g.close.iloc[-1]/g.close.iloc[-2]-1)*100,'return20':(g.close.iloc[-1]/g.close.iloc[-21]-1)*100})
print('MARKETS',json.dumps(ms,ensure_ascii=False));(R/'market_summary.json').write_text(json.dumps(ms,ensure_ascii=False,indent=2))
