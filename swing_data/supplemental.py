"""Public supply/demand and event observations; never a trading signal.

Every record carries its source and observation/publication dates. Missing disclosures
are not zero balances. Provider snapshots are not point-in-time historical estimates.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

from lxml import html
import pandas as pd
import requests
from swing_data.collector import atomic_json, read_json

JPX = 'https://www.jpx.co.jp'
MARGIN = JPX + '/markets/statistics-equities/margin/05.html'
SHORT = JPX + '/markets/public/short-selling/index.html'
EARNINGS = JPX + '/listing/event-schedules/financial-announcement/index.html'
TDNET = 'https://www.release.tdnet.info/inbs/'
JST = ZoneInfo('Asia/Tokyo')


def document(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content


def tree(data):
    return html.fromstring(data.decode('utf-8-sig'))


def stock_code(value):
    value = str(value).strip().removesuffix('.0')
    if re.fullmatch(r'[0-9][0-9A-Z]{3}0', value):
        value = value[:4]
    return value if re.fullmatch(r'[0-9][0-9A-Z]{3}', value) else None


def date(value):
    if value is None or pd.isna(value):
        return None
    try:
        return pd.Timestamp(value).date().isoformat()
    except (ValueError, TypeError):
        return None


def finite(value):
    try:
        x = float(value)
        return x if __import__('math').isfinite(x) else None
    except (ValueError, TypeError):
        return None


def parse_margin(text, url):
    if not all(x in text for x in ['銘柄別信用取引週末残高', 'Unit: 1 share', 'Outstanding Sales', 'Outstanding Purchases']):
        raise ValueError('JPX margin PDF schema changed')
    asof = re.search(r'As of (\d+/\d+/\d+) application based', text)
    published = re.search(r'（単位：一株）\s+(\d+/\d+/\d+)', text)
    if not asof or not published:
        raise ValueError('JPX margin dates missing')
    observations = {}
    for line in text.splitlines():
        m = re.search(r'\b([0-9][0-9A-Z]{3}0)\s+(JP[A-Z0-9]{10})\s+(.+)$', line)
        if not m:
            continue
        tokens = re.sub(r'▲\s*', '-', m[3]).replace(',', '').split()
        # Twelve columns: totals/change, general/change, standardized/change.
        if len(tokens) != 12 or not all(re.fullmatch(r'-?\d+|-', x) for x in tokens):
            continue  # Unparseable rows remain explicitly absent, not shifted.
        values = [int(x) if x != '-' else None for x in tokens]
        sell, buy, general_sell, standard_sell, general_buy, standard_buy = [values[i] for i in (0,2,4,6,8,10)]
        if any(x is None or x < 0 for x in [sell,buy,general_sell,standard_sell,general_buy,standard_buy]):
            continue
        if sell != general_sell + standard_sell or buy != general_buy + standard_buy:
            raise ValueError('JPX margin component totals disagree: ' + m[1])
        code = stock_code(m[1])
        if code in observations:
            raise ValueError('Duplicate JPX margin code: ' + code)
        observations[code] = dict(code=code, as_of=date(asof[1]), published_on=date(published[1]),
            sell_shares=sell, buy_shares=buy, general_sell_shares=general_sell,
            standard_sell_shares=standard_sell, general_buy_shares=general_buy,
            standard_buy_shares=standard_buy, source_url=url)
    if not observations:
        raise ValueError('No margin observations parsed')
    return list(observations.values())


def margin_pdf(data, url):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'source.pdf'
        path.write_bytes(data)
        text = subprocess.run(['pdftotext','-layout',str(path),'-'], check=True,
            capture_output=True, timeout=90).stdout.decode('utf-8')
    return parse_margin(text,url)


def parse_short(data, url):
    frame = pd.read_excel(BytesIO(data), header=None).fillna('')
    header = next((i for i,row in frame.iterrows() if '銘柄コード' in row.values and '空売り残高数量' in row.values), None)
    if header is None:
        raise ValueError('JPX short-position header missing')
    columns = {str(v).split('\n')[0]:i for i,v in enumerate(frame.iloc[header])}
    required = ['計算年月日','銘柄コード','商号・名称・氏名','空売り残高割合','空売り残高数量']
    if any(x not in columns for x in required):
        raise ValueError('JPX short-position schema changed')
    stamp = re.search(r'(\d{8})_Short_Positions', url)
    if not stamp:
        raise ValueError('Short disclosure date missing')
    results = []
    for _,row in frame.iloc[header+1:].iterrows():
        get = lambda key: row[columns[key]] if key in columns else None
        code, asof = stock_code(get('銘柄コード')), date(get('計算年月日'))
        if not code or not asof:
            continue
        ratio, shares = finite(get('空売り残高割合')), finite(get('空売り残高数量'))
        if ratio is None or shares is None or not 0 <= ratio <= 1 or shares < 0:
            raise ValueError('Invalid short balance')
        results.append(dict(code=code,as_of=asof,published_on=date(stamp[1]),
            holder=str(get('商号・名称・氏名')).strip(),
            investment_fund=get('信託財産・運用財産の名称'),
            discretionary_contractor=get('委託者・投資一任契約の相手方の商号・名称・氏名'),
            short_ratio=ratio,short_shares=shares,previous_as_of=date(get('直近計算年月日')),
            previous_short_ratio=finite(get('直近空売り残高割合')),notes=get('備考'),source_url=url))
    return results


def parse_earnings(data,url):
    frame = pd.read_excel(BytesIO(data),header=None).fillna('')
    header = next((i for i,row in frame.iterrows() if any('Scheduled Dates for Earnings' in str(v) for v in row)),None)
    if header is None:
        raise ValueError('Earnings schedule header missing')
    columns = {str(v).split('\n')[0]:i for i,v in enumerate(frame.iloc[header])}
    for key in ['決算発表予定日','コード','決算期末','種別']:
        if key not in columns:
            raise ValueError('Earnings schedule schema changed: '+key)
    stamp = re.search(r'As of (\d+/\d+/\d+)', ' '.join(map(str,frame.iloc[:header].values.flatten())))
    results = []
    for _,row in frame.iloc[header+1:].iterrows():
        get = lambda key: row[columns[key]]
        code = stock_code(get('コード'))
        if code:
            results.append(dict(code=code,scheduled_date=date(get('決算発表予定日')),
                schedule_text=str(get('決算発表予定日')),fiscal_year_end=date(get('決算期末')),
                fiscal_quarter=get('種別'),as_of=date(stamp[1]) if stamp else None,
                date_type='company_scheduled',source_url=url))
    return results


def parse_tdnet(data,url,day):
    doc = tree(data)
    results = []
    for row in doc.xpath('//tr[td[contains(@class,"kjTitle")]]'):
        get = lambda key: ''.join(row.xpath(f'./td[contains(@class,"{key}")]//text()')).strip()
        title, code = get('kjTitle'),stock_code(get('kjCode'))
        if not code or not re.search(r'(業績|利益|収益).*(修正|変更)|上方修正|下方修正',title):
            continue
        links = row.xpath('./td[contains(@class,"kjTitle")]//a/@href')
        xbrl = row.xpath('./td[contains(@class,"kjXbrl")]//a/@href')
        # A revision title alone cannot establish the direction or affected metric.
        direction = 'up_explicit_title' if '上方修正' in title and '下方修正' not in title else 'down_explicit_title' if '下方修正' in title and '上方修正' not in title else 'unconfirmed'
        results.append(dict(code=code,published_at=day+'T'+get('kjTime')+':00+09:00',
            title=title,direction=direction,verification='title_only; inspect disclosure before trading',
            source_url=urljoin(url,links[0]) if links else url,
            xbrl_url=urljoin(url,xbrl[0]) if xbrl else None))
    pages = {urljoin(url,a) for a in re.findall(r'I_list_\d{3}_'+day.replace('-', '')+r'\.html', data.decode('utf-8-sig'))}
    if not doc.xpath('//td[contains(@class,"kjTitle")]') and '開示された情報はありません' not in doc.text_content():
        raise ValueError('TDnet empty or unexpected page')
    return results, pages


def parse_forecast_xbrl(data, expected_code):
    from zipfile import ZipFile
    from decimal import Decimal, InvalidOperation
    from lxml import etree
    z = ZipFile(BytesIO(data))
    if sum(i.file_size for i in z.infolist()) > 30_000_000:
        raise ValueError('Oversized XBRL archive')
    docs = [n for n in z.namelist() if n.endswith(('.htm','.html'))]
    metrics = {'NetSales','SalesIFRS','OperatingIncome','OperatingIncomeIFRS',
        'OrdinaryIncome','ProfitBeforeTaxIFRS','Profit','ProfitIFRS','ProfitAttributableToOwnersOfParent',
        'ProfitAttributableToOwnersOfParentIFRS','NetIncome','NetIncomePerShare','BasicEarningsPerShareIFRS'}
    pairs = []
    for name in docs:
        doc = etree.fromstring(z.read(name), etree.XMLParser(resolve_entities=False, no_network=True))
        nodes = [n for n in doc.iter() if n.get('contextRef')]
        identity = [stock_code(''.join(n.itertext()).strip()) for n in nodes if n.get('name','').endswith(':SecuritiesCode')]
        if identity != [expected_code]:
            raise ValueError('Forecast XBRL securities code mismatch')
        facts = {}
        for node in nodes:
            metric = node.get('name','').split(':')[-1]
            context = node.get('contextRef','')
            if metric not in metrics or not context.endswith('_ForecastMember') or not any(x in context for x in ['_CurrentMember_','_PreviousMember_']): continue
            if any(k.endswith('}nil') and v in ['true','1'] for k,v in node.attrib.items()): continue
            text = ''.join(node.itertext()).strip().replace(',','').replace('△','-').replace('▲','-')
            try:
                value = Decimal(text) * Decimal(10) ** int(node.get('scale','0'))
                if node.get('sign')=='-': value = -value
                if not value.is_finite(): continue
            except (InvalidOperation,ValueError): continue
            key = (metric,context,node.get('unitRef'))
            if key in facts and facts[key]!=value: raise ValueError('Conflicting forecast facts')
            facts[key]=value
        for (metric,context,unit),current in facts.items():
            if '_CurrentMember_' not in context: continue
            previous=facts.get((metric,context.replace('_CurrentMember_','_PreviousMember_'),unit))
            if previous is None: continue
            pairs.append(dict(metric=metric,context=context,unit=unit,previous=float(previous),current=float(current),
                direction='up' if current>previous else 'down' if current<previous else 'unchanged'))
    return pairs


def unwrap(value):
    return value.get('raw') if isinstance(value,dict) else value


def parse_fundamental(payload,code,observed_at):
    result = payload.get('quoteSummary',{}).get('result')
    if not result:
        raise ValueError('No quoteSummary data')
    data = result[0]
    price, stats = data.get('price',{}),data.get('defaultKeyStatistics',{})
    if price.get('symbol') != code+'.T':
        raise ValueError('Provider symbol mismatch')
    cap, shares, floating = [finite(unwrap(x)) for x in [price.get('marketCap'),stats.get('sharesOutstanding'),stats.get('floatShares')]]
    if cap is not None and cap <= 0: cap = None
    if shares is not None and shares <= 0: shares = None
    if floating is not None and (floating < 0 or shares is not None and floating > shares): floating = None
    periods = []
    for item in data.get('earningsTrend',{}).get('trend',[]):
        if item.get('period') not in ['0q','+1q','0y','+1y']: continue
        estimate = item.get('earningsEstimate',{})
        periods.append(dict(period=item['period'],end_date=item.get('endDate'),
            currency=estimate.get('earningsCurrency'),
            number_of_analysts=finite(unwrap(estimate.get('numberOfAnalysts'))),
            estimate={k:finite(unwrap(estimate.get(k))) for k in ['avg','low','high','yearAgoEps']},
            eps_trend={k:finite(unwrap(item.get('epsTrend',{}).get(k))) for k in ['current','7daysAgo','30daysAgo','60daysAgo','90daysAgo']},
            eps_revisions={k:finite(unwrap(item.get('epsRevisions',{}).get(k))) for k in ['upLast7days','upLast30days','downLast7days','downLast30days']}))
    return dict(code=code,observed_at=observed_at,provider='Yahoo Finance',
        source_url=f'https://finance.yahoo.com/quote/{code}.T/analysis/',
        market_cap=cap,currency=price.get('currency'),shares_outstanding=shares,float_shares=floating,
        float_definition='provider floatShares; not JPX free-float weight',
        market_price_at=unwrap(price.get('regularMarketTime')),consensus_periods=periods,
        consensus_status='available' if any(p['end_date'] and p['currency'] and p['eps_trend']['current'] is not None for p in periods) else 'unavailable',
        capital_status='available' if cap is not None and floating is not None else 'partial' if cap is not None or floating is not None else 'unavailable')


def yahoo_snapshot(code, observed_at):
    # One pinned-yfinance quoteSummary request preserves period end dates, which
    # the convenience eps_trend DataFrame discards. No paid credentials used.
    import yfinance as yf
    from yfinance.scrapers.quote import _QUOTE_SUMMARY_URL_
    obj = yf.Ticker(code+'.T')
    payload = obj._data.get_raw_json(_QUOTE_SUMMARY_URL_+'/'+code+'.T',
        params={'modules':'price,defaultKeyStatistics,earningsTrend','formatted':'false'},timeout=20)
    return parse_fundamental(payload,code,observed_at)


def collect(target, *, now=None, fetch=document, fundamentals_fetch=yahoo_snapshot, max_seconds=1800):
    target = Path(target)
    now = now or datetime.now(JST)
    observed = now.isoformat()
    universe = pd.read_csv(target/'universe.csv',dtype=str)
    codes = sorted(universe.ticker.str.removesuffix('.T').tolist())
    allowed = set(codes)
    previous = read_json(target/'supplemental.json')
    status = dict(schema_version=1,run_id=read_json(target/'manifest.json').get('run_id'),
        attempted_at=observed,items={},target_count=len(codes),technical_indicators_persisted=False)
    result = dict(schema_version=1,observed_at=observed,run_id=status['run_id'])
    evidence = []
    def get(url):
        data = fetch(url)
        evidence.append(dict(url=url,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),fetched_at=observed))
        return data
    def run(name,fn):
        try:
            records = fn()
            result[name] = [r for r in records if r.get('code') in allowed]
            status['items'][name] = dict(status='fetched',record_count=len(result[name]),
                covered_codes=len({r['code'] for r in result[name]}),fetched_at=observed)
        except Exception as exc:
            result[name] = previous.get(name,[])
            status['items'][name] = dict(status='stale' if result[name] else 'fetch_failed',
                record_count=len(result[name]),error=f'{type(exc).__name__}: {exc}'[:250])
        print(json.dumps({name:status['items'][name]},ensure_ascii=False),flush=True)
    def margin():
        links = sorted({urljoin(MARGIN,a) for a in tree(get(MARGIN)).xpath('//a/@href') if re.search(r'syumatsu\d{10}\.pdf$',a)})
        if not links: raise ValueError('No weekly margin PDF links')
        records=[]
        # Initial 5 published weeks; subsequently still re-fetch the newest PDF
        # to capture corrections. Retain older observations without redating them.
        for url in links:
            old=[r for r in previous.get('margin',[]) if r.get('source_url')==url]
            records += old if old and url != links[-1] else margin_pdf(get(url),url)
        return records
    def shorts():
        links = sorted({urljoin(SHORT,a) for a in tree(get(SHORT)).xpath('//a/@href') if re.search(r'\d{8}_Short_Positions\.xls$',a)})
        if not links: raise ValueError('No short disclosure links')
        records=list(previous.get('short_positions',[]))
        for url in links:
            if any(r.get('source_url')==url for r in records) and url not in links[-2:]: continue
            new = parse_short(get(url),url)
            records = [r for r in records if r.get('source_url')!=url] + new
        cutoff=(now-timedelta(days=200)).date().isoformat()
        return [r for r in records if r['published_on']>=cutoff]
    def earnings():
        links=sorted({urljoin(EARNINGS,a) for a in tree(get(EARNINGS)).xpath('//a/@href') if re.search(r'kessan.*\.xlsx$',a)})
        if not links: raise ValueError('No earnings schedule files')
        return [r for url in links for r in parse_earnings(get(url),url)]
    def guidance():
        main=tree(get(TDNET+'I_main_00.html'))
        days=sorted(set(re.findall(r'20\d{2}/\d{2}/\d{2}',main.text_content())))
        if not days: raise ValueError('TDnet available dates missing')
        # Re-fetch two most recent available dates; keep older successful days.
        completed=set(previous.get('guidance_completed_dates',[]))
        records=list(previous.get('guidance_revisions',[]))
        for d in days:
            day=d.replace('/','-')
            if day in completed and d not in days[-2:]: continue
            initial=TDNET+'I_list_001_'+d.replace('/','')+'.html'
            todo,seen,new=[initial],set(),[]
            while todo:
                url=todo.pop()
                if url in seen: continue
                if len(seen)>30: raise ValueError('Unexpected TDnet pagination')
                rows,pages=parse_tdnet(get(url),url,day)
                seen.add(url);new+=rows;todo+=sorted(pages-seen)
            records=[r for r in records if not r['published_at'].startswith(day)] + new
            completed.add(day)
        cutoff=(now-timedelta(days=200)).date().isoformat()
        for row in records:
            if row.get('xbrl_url') and row.get('code') in allowed and 'forecast_facts' not in row:
                try:
                    pairs=parse_forecast_xbrl(get(row['xbrl_url']),row['code'])
                    row['forecast_facts']=pairs
                    moves={p['direction'] for p in pairs}-{'unchanged'}
                    if moves:
                        row['direction']='up_xbrl' if moves=={'up'} else 'down_xbrl' if moves=={'down'} else 'mixed_xbrl'
                        row['verification']='XBRL matched metric/context/unit; inspect split and consolidation changes in IR'
                except Exception as exc:
                    row['xbrl_error']=f'{type(exc).__name__}: {exc}'[:160]
        result['guidance_completed_dates']=sorted(d for d in completed if d>=cutoff)
        return list({r['source_url']:r for r in records if r['published_at'][:10]>=cutoff}.values())
    run('margin',margin)
    run('short_positions',shorts)
    run('earnings_calendar',earnings)
    run('guidance_revisions',guidance)
    result.setdefault('guidance_completed_dates',previous.get('guidance_completed_dates',[]))
    old={r['code']:r for r in previous.get('fundamentals',[])}
    stopped=threading.Event()
    started=time.monotonic()
    def one(code):
        existing=old.get(code)
        reason=None
        if existing and existing.get('observed_at','')[:10]==observed[:10]:
            return existing,dict(code=code,status='cached_today')
        if stopped.is_set(): reason='provider_rate_limited'
        elif time.monotonic()-started>max_seconds: reason='time_budget_exceeded'
        if reason is None:
            try:
                item=fundamentals_fetch(code,observed)
                return item,dict(code=code,status='fetched')
            except Exception as exc:
                reason=f'{type(exc).__name__}: {exc}'[:200]
                if '429' in reason or 'RateLimit' in reason or '401' in reason or '403' in reason:
                    stopped.set()
        return existing,dict(code=code,status='stale' if existing else 'unavailable',reason=reason)
    with ThreadPoolExecutor(max_workers=3) as pool:
        outcomes=list(pool.map(one,codes))
    result['fundamentals']=[r for r,s in outcomes if r]
    result['fundamental_status']=[s for r,s in outcomes]
    for name,field in [('capital','capital_status'),('consensus_eps','consensus_status')]:
        fresh=[r for r,s in outcomes if r and s['status'] in ['fetched','cached_today']]
        count=sum(r[field]=='available' for r in fresh)
        status['items'][name]=dict(status='complete' if count==len(codes) else 'partial' if count else 'unavailable',
            available_count=count,target_count=len(codes),attempted_count=sum(s['status']=='fetched' for r,s in outcomes))
    result['source_evidence']=evidence
    status['completed_at']=datetime.now(JST).isoformat()
    status['quality']='PASS' if all(x['status'] in ['fetched','complete'] for x in status['items'].values()) else 'PARTIAL'
    status['notes']={
        'margin':'JPX weekly share balances, five available weeks. Absent code is unknown, not zero.',
        'short_positions':'Disclosure events, not complete current holdings. Public reporting threshold generally 0.5%; below-threshold exit reports may appear. Never sum as total institutional shorts. Initial history starts at available current-month files.',
        'earnings_calendar':'Company schedules in the available JPX workbooks only; absent date does not mean no imminent earnings.',
        'guidance_revisions':'TDnet available 31-day archive initially, then retained daily. Matched XBRL previous/current forecast facts where available; otherwise title-only/unconfirmed. Read IR for corporate-action comparability. Not consensus revisions.',
        'capital':'Yahoo snapshot; floatShares is not JPX FFW. Unknown effective date stays unknown.',
        'consensus_eps':'Provider current/7/30/60/90-day EPS snapshots by fiscal end and currency, with analyst revision counts. Missing is not no change; not point-in-time backtest data.',
        'technicals':'Calculated from current 120-session raw inputs in memory on reading; no saved indicator values.'}
    atomic_json(target/'supplemental.json',result)
    atomic_json(target/'supplemental_status.json',status)
    folder=target/'supplemental';folder.mkdir(exist_ok=True)
    bycode={code:dict(code=code,run_id=status['run_id'],attempted_at=observed,coverage=status['notes'],source_status=status['items']) for code in codes}
    for name in ['margin','short_positions','earnings_calendar','guidance_revisions','fundamentals','fundamental_status']:
        for code in codes: bycode[code][name]=[]
        for row in result.get(name,[]): bycode[row['code']][name].append(row)
    for code,item in bycode.items(): atomic_json(folder/(code+'.json'),item)
    for p in folder.glob('*.json'):
        if p.stem not in allowed: p.unlink()
    manifest=read_json(target/'manifest.json');manifest['supplemental_quality']=status['quality']
    manifest['supplemental_status_url']='supplemental_status.json'
    atomic_json(target/'manifest.json',manifest)
    hashes=read_json(target/'sha256.json')
    for name in ['manifest.json','supplemental.json','supplemental_status.json']:
        hashes[name]=hashlib.sha256((target/name).read_bytes()).hexdigest()
    if (target/'indicators.py').exists():
        hashes['indicators.py']=hashlib.sha256((target/'indicators.py').read_bytes()).hexdigest()
    atomic_json(target/'sha256.json',hashes)
    # Refresh the downloadable bundle as well as the individual public files.
    from zipfile import ZipFile, ZIP_DEFLATED
    bundle=target/'chatgpt_120d.zip'
    if bundle.exists():
        replacements={'manifest.json','supplemental.json','supplemental_status.json','sha256.json','indicators.py'}
        temp=target/'chatgpt_120d.tmp.zip'
        with ZipFile(bundle) as old_zip, ZipFile(temp,'w',ZIP_DEFLATED) as new_zip:
            for entry in old_zip.infolist():
                if entry.filename not in replacements: new_zip.writestr(entry,old_zip.read(entry.filename))
            for name in sorted(replacements):
                if (target/name).exists(): new_zip.write(target/name,name)
        temp.replace(bundle)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--target',type=Path,required=True)
    args=parser.parse_args()
    try:
        print(json.dumps(collect(args.target),ensure_ascii=False),flush=True)
    except Exception as exc:
        atomic_json(args.target/'supplemental_status.json',dict(quality='FAIL',
            attempted_at=datetime.now(JST).isoformat(),error=f'{type(exc).__name__}: {exc}'[:300],
            note='Additional collection failed. Any existing observations retain their old dates.'))
        raise
