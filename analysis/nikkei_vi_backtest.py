from __future__ import annotations
import io, json, math
from pathlib import Path
import pandas as pd
import requests

MARKETS_URL = "https://raw.githubusercontent.com/noumi0713/stock-analysis-tool/swing-data-120d-latest/markets_120d.csv"
NVI_URL = "https://indexes.nikkei.co.jp/nkave/historical/nikkei_stock_average_vi_daily_jp.csv"
OUT = Path("analysis/nikkei_vi_backtest_result.json")

def get(url):
    r=requests.get(url,timeout=30,headers={"User-Agent":"Mozilla/5.0"})
    r.raise_for_status()
    return r.content

def parse_vi(blob):
    last=None
    for enc in ("cp932","shift_jis","utf-8-sig","utf-8"):
        try:
            txt=blob.decode(enc)
            df=pd.read_csv(io.StringIO(txt))
            last=(enc,df)
            break
        except Exception:
            continue
    if last is None:
        raise RuntimeError("cannot decode Nikkei VI csv")
    enc,df=last
    if df.shape[1] < 5:
        raise RuntimeError(f"unexpected Nikkei VI schema: {list(df.columns)}")
    # Official daily file is date, close, open, high, low; parse by position
    x=df.iloc[:,:5].copy()
    x.columns=["date","close","open","high","low"]
    x["date"]=pd.to_datetime(x["date"],errors="coerce")
    for c in ["close","open","high","low"]:
        x[c]=pd.to_numeric(x[c].astype(str).str.replace(",","",regex=False),errors="coerce")
    x=x.dropna(subset=["date","close"]).sort_values("date").drop_duplicates("date",keep="last")
    x["date"]=x["date"].dt.strftime("%Y-%m-%d")
    return x, enc, list(df.columns)

m=pd.read_csv(io.BytesIO(get(MARKETS_URL)))
vi, enc, vi_cols=parse_vi(get(NVI_URL))
m["date"]=m["date"].astype(str)

def series(ticker):
    x=m[m.ticker==ticker].copy().sort_values("date")
    for c in ["open","high","low","close"]:
        x[c]=pd.to_numeric(x[c],errors="coerce")
    return x.dropna(subset=["close"]).reset_index(drop=True)

n=series("^N225")
us=series("^TNX")
jp=series("JP10Y")
vix=series("^VIX")
vi=vi.reset_index(drop=True)

def latest_before(df, date):
    z=df[df.date < date]
    return None if z.empty else z.iloc[-1]

def prior_row(df, row):
    hits=df.index[df.date==row["date"]].tolist()
    if not hits or hits[-1]==0:
        return None
    return df.iloc[hits[-1]-1]

rows=[]
for i,d in n.iterrows():
    u=latest_before(us,d.date)
    j=latest_before(jp,d.date)
    vv=latest_before(vix,d.date)
    nv=latest_before(vi,d.date)  # prior completed Nikkei VI: avoids 15:50 close look-ahead
    if u is None or j is None or vv is None or nv is None:
        continue
    up=prior_row(us,u); jp0=prior_row(jp,j); vp=prior_row(vix,vv); nvp=prior_row(vi,nv)
    if up is None or jp0 is None or vp is None or nvp is None:
        continue
    r={
      "i":int(i),"date":d.date,"entry":float(d.close),
      "us":float(u.close),"dus_bp":float((u.close-up.close)*100),
      "jp":float(j.close),"djp_bp":float((j.close-jp0.close)*100),
      "vix":float(vv.close),"dvix_pct":float((vv.close/vp.close-1)*100),
      "nvi":float(nv.close),"dnvi_pct":float((nv.close/nvp.close-1)*100),
      "nvi_date":nv.date,"us_date":u.date,"jp_date":j.date
    }
    for k in [1,2,3,4,5,7,10]:
        if i+k < len(n):
            future=n.iloc[i+k]
            r[f"r{k}"]=float((future.close/d.close-1)*100)
            lows=n.iloc[i+1:i+k+1].low.astype(float)
            highs=n.iloc[i+1:i+k+1].high.astype(float)
            r[f"mae{k}"]=float((lows.min()/d.close-1)*100)
            r[f"mfe{k}"]=float((highs.max()/d.close-1)*100)
    rows.append(r)
d=pd.DataFrame(rows)

def stats(x,k):
    col=f"r{k}"
    x=x[x[col].notna()].copy()
    if x.empty:return None
    vals=x[col]
    maes=x[f"mae{k}"]
    mfes=x[f"mfe{k}"]
    return {
      "n":int(len(x)),
      "avg":round(float(vals.mean()),3),
      "median":round(float(vals.median()),3),
      "win_pct":round(float((vals>0).mean()*100),1),
      "avg_mae":round(float(maes.mean()),3),
      "worst_mae":round(float(maes.min()),3),
      "avg_mfe":round(float(mfes.mean()),3),
      "p25":round(float(vals.quantile(.25)),3),
      "p75":round(float(vals.quantile(.75)),3)
    }

baseline={str(k):stats(d,k) for k in [1,2,3,4,5,7,10]}

groups={}
for name,mask in {
  "NVI<25":d.nvi<25,
  "NVI25-30":(d.nvi>=25)&(d.nvi<30),
  "NVI30-35":(d.nvi>=30)&(d.nvi<35),
  "NVI>=35":d.nvi>=35,
  "NVI_down":d.dnvi_pct<=0,
  "NVI_down5":d.dnvi_pct<=-5,
  "NVI_up":d.dnvi_pct>0,
  "NVI_up5":d.dnvi_pct>=5,
}.items():
    x=d[mask]
    groups[name]={str(k):stats(x,k) for k in [3,4,5]}

def mask_rule(us_th=-3,jp_th=3,vi_min=25,vi_move="any"):
    q=(d.dus_bp<=us_th)&(d.djp_bp<=jp_th)&(d.nvi>=vi_min)
    if vi_move=="down": q &= d.dnvi_pct<=0
    elif vi_move=="down5": q &= d.dnvi_pct<=-5
    elif vi_move=="up": q &= d.dnvi_pct>0
    elif vi_move=="up5": q &= d.dnvi_pct>=5
    return q

named={}
specs={
 "A_us-3_jp+3_nvi25":[-3,3,25,"any"],
 "B_us-3_jp+3_nvi25_down":[-3,3,25,"down"],
 "C_us-3_jp+3_nvi30":[-3,3,30,"any"],
 "D_us-3_jp+3_nvi30_down":[-3,3,30,"down"],
 "E_us0_jp+3_nvi25_down":[0,3,25,"down"],
 "F_us-5_jp+3_nvi25":[-5,3,25,"any"],
}
for name,args in specs.items():
    x=d[mask_rule(*args)]
    named[name]={"rule":{"us_bp_le":args[0],"jp_bp_le":args[1],"nvi_ge":args[2],"nvi_move":args[3]},
                 "stats":{str(k):stats(x,k) for k in [1,2,3,4,5,7,10]},
                 "dates":x.date.tolist()}

extra_masks={
 "G_us-3_jp+3_noNVI":(d.dus_bp<=-3)&(d.djp_bp<=3),
 "H_us-5_jp+3_noNVI":(d.dus_bp<=-5)&(d.djp_bp<=3),
 "I_us-3_jp+3_nvi25_30":(d.dus_bp<=-3)&(d.djp_bp<=3)&(d.nvi>=25)&(d.nvi<30),
 "J_us-5_jp+3_nvi25_30":(d.dus_bp<=-5)&(d.djp_bp<=3)&(d.nvi>=25)&(d.nvi<30),
 "K_us-3_jp+3_nvi30_35":(d.dus_bp<=-3)&(d.djp_bp<=3)&(d.nvi>=30)&(d.nvi<35),
 "L_us-3_jp+3_nvi35plus":(d.dus_bp<=-3)&(d.djp_bp<=3)&(d.nvi>=35),
}
for name,mask in extra_masks.items():
    x=d[mask]
    named[name]={"rule":name,"stats":{str(k):stats(x,k) for k in [1,2,3,4,5,7,10]},"dates":x.date.tolist()}

grid=[]
for uth in [0,-3,-5]:
  for jth in [0,3,5,999]:
    for vmin in [20,25,30,35]:
      for move in ["any","down","down5","up","up5"]:
        x=d[mask_rule(uth,jth,vmin,move)]
        s=stats(x,4)
        if s and s["n"]>=8:
            score=s["avg"]/max(.25,abs(s["avg_mae"])) * math.sqrt(s["n"]/20)
            grid.append({"us_bp_le":uth,"jp_bp_le":jth,"nvi_ge":vmin,"nvi_move":move,
                         **s,"score":round(float(score),3)})
grid=sorted(grid,key=lambda z:(z["score"],z["avg"],z["n"]),reverse=True)

# Exit diagnostics for rule A. All dynamic risk inputs are lagged/known by the Tokyo close.
sel=d[mask_rule(-3,3,25,"any")].copy()
by_i={int(r.i):r for _,r in d.iterrows()}
def dyn_trade(s,kind,maxh=5):
    end_i=min(int(s.i)+maxh,len(n)-1)
    exit_i=end_i
    for q in range(1,maxh+1):
        ii=int(s.i)+q
        if ii>=len(n):break
        z=by_i.get(ii)
        if z is None:continue
        hit=False
        if kind=="nvi_lt25": hit=z.nvi<25
        elif kind=="nvi_down10": hit=z.dnvi_pct<=-10
        elif kind=="us_up3": hit=z.dus_bp>=3
        elif kind=="nvi_lt25_or_us_up3": hit=(z.nvi<25 or z.dus_bp>=3)
        if hit:
            exit_i=ii;break
    lows=n.iloc[int(s.i)+1:exit_i+1].low.astype(float)
    return {"ret":float((n.iloc[exit_i].close/s.entry-1)*100),
            "mae":float((lows.min()/s.entry-1)*100) if len(lows) else 0.0,
            "days":exit_i-int(s.i)}
def tsum(ts):
    if not ts:return None
    r=pd.Series([x["ret"] for x in ts]); a=pd.Series([x["mae"] for x in ts]); days=pd.Series([x["days"] for x in ts])
    return {"n":len(ts),"avg":round(float(r.mean()),3),"median":round(float(r.median()),3),
            "win_pct":round(float((r>0).mean()*100),1),"avg_mae":round(float(a.mean()),3),
            "worst_mae":round(float(a.min()),3),"avg_days":round(float(days.mean()),2)}
dynamic={}
for kind in ["nvi_lt25","nvi_down10","us_up3","nvi_lt25_or_us_up3"]:
    dynamic[kind]=tsum([dyn_trade(s,kind,5) for _,s in sel.iterrows() if int(s.i)+1<len(n)])

# Conservative daily TP/SL test: if TP and SL both occur same day, count SL first.
def tp_sl(s,tp,sl,maxh=5):
    for q in range(1,maxh+1):
        ii=int(s.i)+q
        if ii>=len(n):break
        day=n.iloc[ii]
        hit_tp=day.high>=s.entry*(1+tp/100)
        hit_sl=day.low<=s.entry*(1-sl/100)
        if hit_sl:return {"ret":-sl,"days":q,"why":"SL"}
        if hit_tp:return {"ret":tp,"days":q,"why":"TP"}
    ii=min(int(s.i)+maxh,len(n)-1)
    return {"ret":float((n.iloc[ii].close/s.entry-1)*100),"days":ii-int(s.i),"why":"time"}
tpgrid=[]
for tp in [1.5,2,2.5,3,3.5]:
  for sl in [1,1.5,2,2.5,3]:
    t=[tp_sl(s,tp,sl,5) for _,s in sel.iterrows() if int(s.i)+1<len(n)]
    rr=pd.Series([x["ret"] for x in t])
    tpgrid.append({"tp":tp,"sl":sl,"n":len(t),"avg":round(float(rr.mean()),3),
                   "win_pct":round(float((rr>0).mean()*100),1),
                   "tp_hits":sum(x["why"]=="TP" for x in t),
                   "sl_hits":sum(x["why"]=="SL" for x in t),
                   "time_exits":sum(x["why"]=="time" for x in t)})
tpgrid=sorted(tpgrid,key=lambda z:z["avg"],reverse=True)

period_split={}
for name,mask in {
 "core_us-3_jp+3":(d.dus_bp<=-3)&(d.djp_bp<=3),
 "strict_us-5_jp+3":(d.dus_bp<=-5)&(d.djp_bp<=3),
 "core_nvi25_30":(d.dus_bp<=-3)&(d.djp_bp<=3)&(d.nvi>=25)&(d.nvi<30),
 "core_nvi30_35":(d.dus_bp<=-3)&(d.djp_bp<=3)&(d.nvi>=30)&(d.nvi<35),
 "core_nvi35plus":(d.dus_bp<=-3)&(d.djp_bp<=3)&(d.nvi>=35),
}.items():
    period_split[name]={
      "before_2026_07_01":{str(k):stats(d[mask & (d.date<"2026-07-01")],k) for k in [3,4,5]},
      "from_2026_07_01":{str(k):stats(d[mask & (d.date>="2026-07-01")],k) for k in [3,4,5]}
    }

result={
 "method":{
   "entry":"Nikkei 225 close on signal date",
   "nikkei_vi_timing":"uses most recent completed Nikkei VI daily close strictly before signal date",
   "us10y_timing":"uses most recent US 10Y close strictly before signal date",
   "jp10y_timing":"uses most recent JP10Y observation strictly before signal date; available by Tokyo close",
   "lookahead":"none in signal variables under close-entry convention",
   "note":"Nikkei VI official daily close is after the cash close, so same-date VI close is intentionally not used."
 },
 "source":{
   "nikkei_vi":NVI_URL,"markets":MARKETS_URL,"vi_encoding":enc,"vi_columns":vi_cols,
   "vi_rows":int(len(vi)),"vi_latest":vi.date.iloc[-1]
 },
 "range":{"first_signal_date":d.date.iloc[0],"last_signal_date":d.date.iloc[-1],"aligned_days":int(len(d)),
          "n225_first":n.date.iloc[0],"n225_last":n.date.iloc[-1]},
 "baseline":baseline,
 "nvi_groups":groups,
 "named_rules":named,
 "top_grid_4d":grid[:20],
 "rule_A_dynamic_exits_max5d":dynamic,
 "rule_A_tp_sl_max5d":tpgrid[:15],
 "period_split":period_split
}
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(result,ensure_ascii=False))
