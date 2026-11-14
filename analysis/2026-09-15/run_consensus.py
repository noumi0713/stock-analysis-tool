from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'main'))
from swing_data import market_consensus as m
source=(ROOT/'main/swing_data/market_consensus.py').read_text()
# Use original classification/validation logic; separate observation day and completed equity session.
source=source.replace('expected_date = manifest.get("expected_equity_date") or now.date().isoformat()', 'expected_date = now.date().isoformat()\n    equity_date = manifest.get("expected_equity_date")')
source=source.replace('local_market_data(target, code, expected_date)', 'local_market_data(target, code, equity_date)')
source=source.replace('inspect(snapshot, local, expected_date)', 'inspect(snapshot, local, equity_date)')
source=source.replace('local.get("date") == expected_date else "provider_snapshot"','local.get("date") == equity_date else "provider_snapshot"')
source=source.replace('"data_reference_date": expected_date, "retrieved_at": retrieved_at','"data_reference_date": None, "retrieved_at": retrieved_at')
# Keep original helper signatures intact; execute only the adapted collect function.
import ast
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='collect')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/'run_consensus.py'),'exec'),m.__dict__)
root=ROOT/'swing-data-120d-latest'
raw=root/'market_consensus/raw/2026-09-15';raw.mkdir(parents=True,exist_ok=True)
def fetch(code):
 s=m.yahoo_snapshot(code)
 (raw/f'{code}.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
 return s
status=m.collect(root,fetcher=fetch,max_workers=4)
audit={'observation_date':'2026-09-15','equity_date':'2026-09-14','original_rule_sha256':hashlib.sha256((ROOT/'main/swing_data/market_consensus.py').read_bytes()).hexdigest(),'adjustment':'Separate Japan observation day from completed equity date. Classification and validation functions unchanged. Provider reference timestamp unknown; left blank.','status':status}
(root/'consensus_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print(json.dumps(status,ensure_ascii=False,indent=2))
