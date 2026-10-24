from pathlib import Path
import re,html,urllib.request,concurrent.futures,pandas as pd,json
R=Path(__file__).resolve().parent

def parse(s):
 rows=[]
 for tr in re.findall(r'<tr\b[^>]*>(.*?)</tr>',s,re.S):
  cells=[html.unescape(re.sub('<[^>]+>','',x)) for x in re.findall(r'<t[dh]\b[^>]*>(.*?)</t[dh]>',tr,re.S)]
  if cells and re.fullmatch(r'2026/\d+/\d+',cells[0]):
   try:rows.append([pd.Timestamp(cells[0]).strftime('%Y-%m-%d')]+[float(x.replace(',','')) for x in cells[1:7]])
   except:pass
 return rows

def work(code):
 rows=parse((R/'web'/f'{code}_history.html').read_text());sources=[]
 if code!='618A':
  for page in range(2,7):
   url=f'https://finance.yahoo.co.jp/quote/{code}.T/history?page={page}'
   try:
    b=urllib.request.urlopen(url,timeout=30).read();(R/'web'/f'{code}_history_{page}.html').write_bytes(b)
    extra=parse(b.decode());sources.append(url)
    if extra==rows[:len(extra)]:print(code,'pagination not effective');break
    rows+=extra
   except Exception as e:print(code,str(e));break
 d=pd.DataFrame(rows,columns=['date','open','high','low','close','volume','adj_close']).drop_duplicates('date').sort_values('date');d=d[d.date<='2026-09-14'].tail(120)
 d['ticker']=code+'.T'
 for k in ['open','high','low']:d['adj_'+k]=d[k]*d.adj_close/d.close
 d['stock_splits']=0;d['dividends']=0
 d.to_csv(R/'swing-data-120d-latest/stocks'/f'{code}.csv',index=False)
 print(code,len(d),d.date.min(),d.date.max(),flush=True)
 return {'code':code,'rows':len(d),'source':'https://finance.yahoo.co.jp/quote/'+code+'.T/history','pages':sources,'adjustment_note':'adj_close is provider adjusted close; no split event metadata supplied'}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:a=list(ex.map(work,['1570','1357','618A']))
(R/'swing-data-120d-latest/supplement_history_sources.json').write_text(json.dumps(a,ensure_ascii=False,indent=2))
