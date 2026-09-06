from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CODES = {"285A","6976","593A","278A","5801","5803","6857","6920","9348","7013","4592"}
MAX_ABS_DAILY_RETURN = 0.45
COST_ONE_WAY = 0.001  # 0.10% each side
MAX_HOLD = 10


def scale(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return ((s - lo) / (hi - lo) * 100.0).clip(0, 100)


def rsi14(c: pd.Series) -> pd.Series:
    d = c.diff()
    gain = d.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-d.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss.ne(0), 100.0)


def load_prices(data_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    m = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    dates = pd.to_datetime(m["dates"])
    meta = m.get("stocks", {})
    out: dict[str, pd.DataFrame] = {}
    for shard in m["shards"]:
        payload = json.loads((data_dir / shard["path"]).read_text(encoding="utf-8"))
        for ticker, bars in payload["bars"].items():
            code = str(meta.get(ticker, {}).get("code") or ticker.removesuffix(".T"))[:4]
            if code not in CODES or not bars:
                continue
            a = np.asarray(bars, dtype=float)
            g = pd.DataFrame({
                "date": dates[a[:,0].astype(int)],
                "open": a[:,1], "high": a[:,2], "low": a[:,3],
                "close": a[:,4], "volume": a[:,5],
            }).sort_values("date").reset_index(drop=True)
            g["code"] = code
            g["ticker"] = ticker
            g["name"] = str(meta.get(ticker, {}).get("name") or ticker)
            out[code] = g
    return out, m


def enrich(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    c,h,l,v = g.close,g.high,g.low,g.volume
    turn = c*v
    g["history_n"] = np.arange(1,len(g)+1)
    g["ret1"] = c.pct_change(); g["ret5"] = c.pct_change(5); g["ret10"] = c.pct_change(10); g["ret20"] = c.pct_change(20)
    g["ma10"] = c.rolling(10,min_periods=10).mean(); g["ma25"] = c.rolling(25,min_periods=25).mean(); g["ma75"] = c.rolling(75,min_periods=75).mean()
    g["ma10_slope5"] = g.ma10/g.ma10.shift(5)-1; g["ma25_slope5"] = g.ma25/g.ma25.shift(5)-1
    g["rsi14"] = rsi14(c)
    g["high10"] = h.rolling(10,min_periods=10).max(); g["high20"] = h.rolling(20,min_periods=15).max(); g["prev_high20"] = h.rolling(20,min_periods=15).max().shift(1)
    g["vol_avg20"] = v.rolling(20,min_periods=10).mean().shift(1); g["vol_ratio"] = v/g.vol_avg20.replace(0,np.nan)
    g["turnover"] = turn; g["turnover_med20"] = turn.rolling(20,min_periods=10).median(); g["participation"] = turn/g.turnover_med20.replace(0,np.nan)
    g["day_pos"] = (c-l)/(h-l).replace(0,np.nan)
    ma_base = g.ma25.where(g.ma25.notna(),g.ma10); ret_base = g.ret20.where(g.ret20.notna(),g.ret10); high_base = g.high20.where(g.high20.notna(),g.high10)
    g["ma_gap"] = c/ma_base-1; g["drawdown_high"] = c/high_base-1
    signed_vol = (g.vol_ratio-1.0)*np.sign(g.ret5.fillna(0))
    g["sentiment"] = (0.40*g.rsi14 + 0.25*scale(ret_base,-0.20,0.30) + 0.20*scale(g.ma_gap,-0.10,0.15) + 0.10*scale(c/high_base,0.75,1.00) + 0.05*scale(signed_vol,-1.5,1.5)).clip(0,100)
    bad = ((g.open<=0)|(h<=0)|(l<=0)|(c<=0)|(v<0)|(h<g[["open","close"]].max(axis=1))|(l>g[["open","close"]].min(axis=1))|(g.ret1.abs()>MAX_ABS_DAILY_RETURN)).fillna(False)
    g["quality"] = ~bad.rolling(75,min_periods=1).max().astype(bool)
    g["liquid"] = g.turnover_med20 >= 500_000_000
    long_trend = ((g.ma25.notna() & g.ma75.notna() & (c>g.ma25) & (g.ma25>g.ma75) & (g.ma25_slope5>0)) | (g.ma25.notna() & g.ma75.isna() & (c>g.ma25) & (g.ma25_slope5>0)))
    ipo_trend = ((g.history_n>=15)&(g.history_n<60)&g.ma10.notna()&(c>g.ma10)&(g.ma10_slope5>0)&(g.ret10>0.05))
    breakout = (c>=g.prev_high20*0.98)&(g.ret5>0)&(g.vol_ratio>=1.0)
    pullback = g.drawdown_high.between(-0.10,-0.02)&(c>c.shift(1))&(g.day_pos>=0.60)&(g.vol_ratio>=0.70)
    ipo_breakout = ipo_trend&(c/g.high10>=0.94)&(g.vol_ratio>=1.0)
    high_part = g.participation.between(1.2,2.0,inclusive="both")
    g["blue_entry"] = g.quality & g.liquid & high_part & (long_trend|ipo_trend) & g.sentiment.between(60,80) & g.rsi14.between(52,72) & g.ma_gap.between(-0.01,0.12) & (breakout|pullback|ipo_breakout)
    g["contra_entry"] = g.quality & g.liquid & high_part & (g.sentiment<=35)&(g.rsi14<=38)&((g.ret5<=-0.08)|(g.drawdown_high<=-0.15))&(g.vol_ratio>=1.30)&(c>g.open)&(c>c.shift(1))&(g.day_pos>=0.65)
    g["euphoria"] = g.quality & g.liquid & (g.sentiment>=80)&(g.rsi14>=72)&((g.ma_gap>=0.12)|(g.ret10>=0.15))&((g.vol_ratio>=1.30)|(g.ret5>=0.10))&(g.drawdown_high>=-0.04)
    return g


def simulate(g: pd.DataFrame, mode: str) -> list[dict[str,Any]]:
    trades=[]; pos=None
    i=0
    while i < len(g)-1:
        r=g.iloc[i]
        if pos is None:
            sig = "BLUE" if bool(r.blue_entry) else ("CONTRARIAN" if bool(r.contra_entry) else None)
            if sig and (mode=="COMBINED" or (mode=="BLUE" and sig=="BLUE") or (mode=="CONTRARIAN" and sig=="CONTRARIAN")):
                er=g.iloc[i+1]
                pos={"signal":sig,"signal_date":r.date,"entry_i":i+1,"entry_date":er.date,"entry_price":float(er.open),"entry_sentiment":float(r.sentiment),"entry_participation":float(r.participation)}
                i+=1
        else:
            hold=i-pos["entry_i"]+1
            exit_reason=None
            if bool(r.euphoria): exit_reason="EUPHORIA"
            elif pos["signal"]=="BLUE" and pd.notna(r.ma25) and r.close < r.ma25: exit_reason="TREND_BREAK"
            elif pos["signal"]=="CONTRARIAN" and r.sentiment >= 60: exit_reason="SENTIMENT_NORMALIZED"
            elif hold>=MAX_HOLD: exit_reason="MAX_10D"
            if exit_reason and i < len(g)-1:
                xr=g.iloc[i+1]
                ep=pos["entry_price"]; xp=float(xr.open)
                gross=xp/ep-1; net=(xp*(1-COST_ONE_WAY))/(ep*(1+COST_ONE_WAY))-1
                path=g.iloc[pos["entry_i"]:i+2]
                mfe=float(path.high.max()/ep-1); mae=float(path.low.min()/ep-1)
                trades.append({**pos,"exit_signal_date":r.date,"exit_date":xr.date,"exit_price":xp,"exit_reason":exit_reason,"hold_days":int((i+1)-pos["entry_i"]),"gross_return":gross,"net_return":net,"mfe":mfe,"mae":mae})
                pos=None; i+=1
        i+=1
    if pos is not None:
        xr=g.iloc[-1]; ep=pos["entry_price"]; xp=float(xr.close)
        gross=xp/ep-1; net=(xp*(1-COST_ONE_WAY))/(ep*(1+COST_ONE_WAY))-1
        path=g.iloc[pos["entry_i"]:]
        trades.append({**pos,"exit_signal_date":xr.date,"exit_date":xr.date,"exit_price":xp,"exit_reason":"DATA_END","hold_days":int(len(g)-1-pos["entry_i"]),"gross_return":gross,"net_return":net,"mfe":float(path.high.max()/ep-1),"mae":float(path.low.min()/ep-1)})
    return trades


def summarize(t: pd.DataFrame) -> dict[str,Any]:
    if t.empty: return {"n":0}
    eq=(1+t.net_return).cumprod(); peak=eq.cummax(); dd=eq/peak-1
    return {"n":int(len(t)),"mean_net":float(t.net_return.mean()),"median_net":float(t.net_return.median()),"win_rate":float((t.net_return>0).mean()),"pf":float(t.loc[t.net_return>0,"net_return"].sum()/abs(t.loc[t.net_return<0,"net_return"].sum())) if (t.net_return<0).any() else None,"compound":float(eq.iloc[-1]-1),"max_dd":float(dd.min()),"avg_mfe":float(t.mfe.mean()),"avg_mae":float(t.mae.mean()),"avg_hold":float(t.hold_days.mean())}


def pct(x): return "-" if x is None else f"{100*x:.2f}%"


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data-dir",type=Path,required=True); ap.add_argument("--output-json",type=Path,required=True); ap.add_argument("--output-md",type=Path,required=True); a=ap.parse_args()
    raw,m=load_prices(a.data_dir); all_trades=[]
    for code,g in raw.items():
        eg=enrich(g)
        for mode in ["BLUE","CONTRARIAN","COMBINED"]:
            for tr in simulate(eg,mode): tr["code"]=code; tr["name"]=str(g.name.iloc[0]); tr["mode"]=mode; all_trades.append(tr)
    t=pd.DataFrame(all_trades)
    for col in ["signal_date","entry_date","exit_signal_date","exit_date"]:
        if col in t: t[col]=pd.to_datetime(t[col]).dt.strftime("%Y-%m-%d")
    out={"meta":{"data_start":m["dates"][0],"data_end":m["dates"][-1],"codes":sorted(raw),"entry_execution":"next day open","exit_execution":"next day open","cost_one_way":COST_ONE_WAY,"max_hold":MAX_HOLD,"participation":"1.2x-2.0x current turnover / 20d median"},"overall":{},"by_code":{},"trades":t.to_dict("records") if not t.empty else []}
    for mode in ["BLUE","CONTRARIAN","COMBINED"]:
        mt=t[t.mode==mode].copy() if not t.empty else pd.DataFrame(); out["overall"][mode]=summarize(mt)
    for code in sorted(raw):
        out["by_code"][code]={}
        for mode in ["BLUE","CONTRARIAN","COMBINED"]:
            mt=t[(t.code==code)&(t.mode==mode)].copy() if not t.empty else pd.DataFrame(); out["by_code"][code][mode]=summarize(mt)
    a.output_json.parent.mkdir(parents=True,exist_ok=True); a.output_json.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# 11 teaching stocks psychology round-trip test","",f"- Data: {m['dates'][0]} to {m['dates'][-1]}","- Entry at next trading day open after signal","- Exit at next trading day open after exit condition","- Cost: 0.10% each side","- Participation high-confidence zone: 1.2x–2.0x current turnover / 20d median","- Max hold: 10 trading days","", "## Overall", "", "| Mode | N | Mean net | Median | Win | PF | Compound* | Max DD* | Avg MFE | Avg MAE | Avg hold |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in ["BLUE","CONTRARIAN","COMBINED"]:
        s=out["overall"][mode]; lines.append(f"| {mode} | {s.get('n',0)} | {pct(s.get('mean_net'))} | {pct(s.get('median_net'))} | {pct(s.get('win_rate'))} | {('-' if s.get('pf') is None else f'{s.get('pf'):.2f}')} | {pct(s.get('compound'))} | {pct(s.get('max_dd'))} | {pct(s.get('avg_mfe'))} | {pct(s.get('avg_mae'))} | {('-' if s.get('avg_hold') is None else f'{s.get('avg_hold'):.1f}')} |")
    lines += ["", "* Compound/Max DD are sequential trade equity statistics, not a capital-allocation portfolio simulation.","", "## By code (COMBINED)","","| Code | N | Mean net | Win | PF | Compound | Max DD |","|---|---:|---:|---:|---:|---:|---:|"]
    for code in sorted(raw):
        s=out["by_code"][code]["COMBINED"]; lines.append(f"| {code} | {s.get('n',0)} | {pct(s.get('mean_net'))} | {pct(s.get('win_rate'))} | {('-' if s.get('pf') is None else f'{s.get('pf'):.2f}')} | {pct(s.get('compound'))} | {pct(s.get('max_dd'))} |")
    lines += ["", "## Exit rules", "", "- BLUE: euphoria -> exit; otherwise close below MA25 -> exit; otherwise max 10 days.", "- CONTRARIAN: euphoria -> exit; otherwise sentiment >=60 -> exit; otherwise max 10 days.", "- COMBINED: accepts either entry and applies the matching exit rule.", "", "## Caveat", "", "These 11 stocks were selected after observing their charts, so this is an in-sample teaching-stock audit and must not be treated as OOS evidence."]
    a.output_md.write_text("\n".join(lines),encoding="utf-8"); print("\n".join(lines))

if __name__=="__main__": main()
