from pathlib import Path
import ast, functools, urllib.request, urllib.parse, json, pandas as pd
ROOT=Path(__file__).resolve().parent
source=ROOT/'main/swing_data/bbs_ranking.py'
tree=ast.parse(source.read_text())
# Keep the original parser, validation, trend and immutable daily-save code.
# The HTTP adapter uses urllib because requests is not preinstalled here.
tree.body=[n for n in tree.body if not (isinstance(n,ast.Import) and any(x.name=='requests' for x in n.names)) and not (isinstance(n,ast.ImportFrom) and n.module=='swing_data.collector')]
collector=ast.parse((ROOT/'main/swing_data/collector.py').read_text())
atomic=next(n for n in collector.body if isinstance(n,ast.FunctionDef) and n.name=='atomic_json')
tree.body.append(atomic)
ns={'__name__':'ranking_original'}
exec(compile(tree,str(source),'exec'),ns)
class Response:
 def __init__(self,text):self.text=text
 def raise_for_status(self):pass
class Session:
 def get(self,url,params,headers,timeout):
  req=urllib.request.Request(url+'?'+urllib.parse.urlencode(params),headers=headers)
  with urllib.request.urlopen(req,timeout=timeout) as r:return Response(r.read().decode())
target=ROOT/'swing-data-120d-latest'
h=pd.read_csv(target/'bbs_ranking_history.csv',dtype={'stock_code':str})
for date,frame in h.groupby('date'):
 p=target/'bbs_ranking/daily'/f'{date}.csv';p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists(): frame.to_csv(p,index=False)
result=ns['collect'](target,fetcher=functools.partial(ns['fetch_snapshot'],session=Session()))
print(json.dumps(result,ensure_ascii=False,indent=2))
