"""Typed market series: exact indexes, published yields, and explicit field quality."""
from __future__ import annotations

from datetime import datetime
from io import BytesIO
import hashlib
import re
import time

import exchange_calendars as xcals
from lxml import html
import numpy as np
import pandas as pd
import requests

MOF_CURRENT = 'https://www.mof.go.jp/jgbs/reference/interest_rate/jgbcm.csv'
MOF_HISTORY = 'https://www.mof.go.jp/jgbs/reference/interest_rate/data/jgbcm_all.csv'
CBOE_VIX = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv'


def parse_cboe_vix(content):
    data = pd.read_csv(BytesIO(content))
    if list(data.columns) != ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE']:
        raise ValueError('Cboe VIX daily history schema changed')
    data['DATE'] = pd.to_datetime(data['DATE'], format='%m/%d/%Y', errors='raise')
    return data.rename(columns={'DATE':'date', 'OPEN':'Open', 'HIGH':'High',
                                'LOW':'Low', 'CLOSE':'Close'}).set_index('date')


def fetch_cboe_vix(item, sessions, *, get=requests.get):
    if item['ticker'] != '^VIX':
        raise ValueError('Cboe VIX source cannot be used for another index')
    content = get_bytes(CBOE_VIX, get=get)
    return parse_cboe_vix(content), [source_evidence(CBOE_VIX, content)]


def get_bytes(url, *, get=requests.get):
    for attempt in range(3):
        try:
            response = get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            code = getattr(exc.response, 'status_code', None)
            # Never loop on access denials, invalid symbols, or rate limiting.
            if code in {401, 403, 404, 429} or attempt == 2:
                raise
            time.sleep(2 ** attempt)


def source_evidence(url, content):
    return {'url': url, 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}


def parse_yahoo_index(content, expected_name):
    doc = html.fromstring(content)
    title = ''.join(doc.xpath('//title/text()'))
    if not title.startswith(expected_name + '：'):
        raise ValueError(f'Index identity mismatch: {title[:100]}')
    tables = []
    for table in doc.xpath('//table'):
        rows = [[cell.text_content().strip() for cell in tr.xpath('./th|./td')]
                for tr in table.xpath('.//tr')]
        if rows and rows[0] == ['日付', '始値', '高値', '安値', '終値']:
            tables.append(rows[1:])
    if len(tables) != 1 or not tables[0]:
        raise ValueError('Index history table missing or ambiguous')
    records = []
    for row in tables[0]:
        if len(row) != 5 or not re.fullmatch(r'\d{4}/\d{1,2}/\d{1,2}', row[0]):
            raise ValueError('Unexpected index history row')
        records.append([pd.Timestamp(row[0]), *[pd.to_numeric(v.replace(',', ''), errors='coerce') for v in row[1:]]])
    return pd.DataFrame(records, columns=['date', 'Open', 'High', 'Low', 'Close']).set_index('date')


def fetch_yahoo_index(item, sessions, *, get=requests.get):
    frames, evidence = [], []
    oldest = None
    for page in range(1, 17):
        url = f"https://finance.yahoo.co.jp/quote/{item['provider_symbol']}/history?page={page}"
        content = get_bytes(url, get=get)
        frame = parse_yahoo_index(content, item['identity_name'])
        current_oldest = frame.index.min()
        if oldest is not None and current_oldest >= oldest:
            raise ValueError('Index pagination did not advance')
        oldest = current_oldest
        frames.append(frame)
        evidence.append(source_evidence(url, content))
        if oldest <= pd.Timestamp(sessions[0]):
            return pd.concat(frames).sort_index(), evidence
        time.sleep(.25)
    raise ValueError('Index history did not cover requested window')


def mof_date(value):
    m = re.fullmatch(r'([SHR])(\d+)\.(\d+)\.(\d+)', str(value).strip())
    if not m:
        return pd.NaT
    era, year, month, day = m.groups()
    return pd.Timestamp(year={'S':1925,'H':1988,'R':2018}[era]+int(year), month=int(month), day=int(day))


def parse_mof(content):
    data = pd.read_csv(BytesIO(content), encoding='cp932', skiprows=1, dtype=str)
    if not {'基準日', '10年'}.issubset(data.columns):
        raise ValueError('MOF 10-year constant-maturity column missing')
    frame = pd.DataFrame({'date': data['基準日'].map(mof_date),
                          'Close': pd.to_numeric(data['10年'], errors='coerce')})
    return frame.dropna(subset=['date']).set_index('date')


def fetch_mof(item, sessions, *, get=requests.get):
    frames, evidence = [], []
    for url in [MOF_HISTORY, MOF_CURRENT]:
        content = get_bytes(url, get=get)
        frames.append(parse_mof(content))
        evidence.append(source_evidence(url, content))
    combined = pd.concat(frames)
    overlap = combined[combined.index.duplicated(keep=False)]
    if not overlap.empty and overlap.groupby(level=0).Close.nunique(dropna=False).gt(1).any():
        raise ValueError('MOF current/history overlap values disagree')
    return combined.loc[~combined.index.duplicated()].sort_index(), evidence


def mof_sessions(now: datetime, window=120):
    # MOF publishes each observation on the NEXT business day at about 09:30 JST.
    stamp = pd.Timestamp(now).tz_convert('Asia/Tokyo')
    cal = xcals.get_calendar('XTKS')
    days = cal.sessions_in_range((stamp-pd.Timedelta(days=450)).date(), stamp.date())
    eligible = [days[i-1] for i in range(1, len(days))
                if pd.Timestamp(days[i].date()).tz_localize('Asia/Tokyo') + pd.Timedelta(hours=9, minutes=30) <= stamp]
    if len(eligible) < window:
        raise ValueError('MOF published window too short')
    return [d.date().isoformat() for d in eligible[-window:]]


def normalize_market(raw, item, sessions):
    from swing_data.collector import PRICE_COLUMNS
    frame = raw.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}).copy()
    if 'date' not in frame:
        frame['date'] = [pd.Timestamp(d).date().isoformat() for d in frame.index]
    frame = frame[frame.date.isin(sessions)].copy().reset_index(drop=True)
    report = {**item, 'expected_date':sessions[-1], 'window_start':sessions[0], 'rows':0,
              'latest_date':None, 'missing_sessions':len(sessions), 'invalid_rows':0}
    if frame.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS), {**report, 'status':'fetch_failed'}
    for col in ['open','high','low','close','volume']:
        frame[col] = pd.to_numeric(frame[col], errors='coerce') if col in frame else np.nan
    rules = {'duplicate_date':frame.date.duplicated(keep=False), 'close_not_finite':~np.isfinite(frame.close)}
    if item['kind'] != 'yield':
        rules['nonpositive_close'] = frame.close <= 0
    ohlc_bad = (~np.isfinite(frame[['open','high','low','close']]).all(axis=1)
                | (frame.high < frame[['open','close','low']].max(axis=1))
                | (frame.low > frame[['open','close','high']].min(axis=1)))
    if item['kind'] != 'yield':
        ohlc_bad |= frame[['open','high','low','close']].le(0).any(axis=1)
    if item['field_mode'] == 'ohlc':
        rules['invalid_ohlc'] = ohlc_bad
    else:
        # Do not repair bad extrema or turn a daily yield into a synthetic candle.
        report['ohlc_status'] = 'not_provided' if item['kind'] == 'yield' else 'not_used'
        if item['kind'] == 'fx':
            report['source_ohlc_invalid_rows'] = int(ohlc_bad.sum())
            report['source_ohlc_issues'] = frame.loc[ohlc_bad, ['date','open','high','low','close']].replace([np.nan, np.inf, -np.inf], None).to_dict('records')
        frame[['open','high','low']] = np.nan
    bad = pd.DataFrame(rules, index=frame.index).any(axis=1)
    report['invalid_rows'] = int(bad.sum())
    report['rejections'] = [{**row, 'reasons':[k for k, mask in rules.items() if bool(mask.iloc[pos])]}
        for pos, row in enumerate(frame[['date','open','high','low','close']].replace([np.nan, np.inf, -np.inf], None).to_dict('records')) if bad.iloc[pos]]
    frame = frame.loc[~bad].sort_values('date')
    frame['ticker'] = item['ticker']
    for col in ['close','open','high','low']:
        frame['adj_'+col] = frame[col]
    frame['dividends'], frame['stock_splits'] = np.nan, np.nan
    report['missing_dates'] = sorted(set(sessions) - set(frame.date))
    count, latest = len(frame), frame.date.iloc[-1] if len(frame) else None
    status = ('invalid_rows' if report['invalid_rows'] else 'fetch_failed' if not count else
              'stale' if latest != sessions[-1] else 'insufficient_history' if report['missing_dates'] else 'ready')
    report.update(status=status, rows=count, latest_date=latest, missing_sessions=len(report['missing_dates']))
    frame['field_mode'], frame['unit'], frame['source'] = item['field_mode'], item['unit'], item['source']
    return frame[PRICE_COLUMNS+['field_mode','unit','source']], report


def fetch_market(item, sessions, *, downloader=None, get=requests.get):
    from swing_data.collector import PRICE_COLUMNS, extract_ticker
    attempts = []
    for attempt in range(3):
        try:
            if item['source'] == 'yahoo_jp':
                raw, evidence = fetch_yahoo_index(item, sessions, get=get)
            elif item['source'] == 'mof':
                raw, evidence = fetch_mof(item, sessions, get=get)
            elif item['source'] == 'cboe':
                raw, evidence = fetch_cboe_vix(item, sessions, get=get)
            elif item['source'] == 'tradingview_public':
                from swing_data.tradingview_source import fetch_public_index
                raw, evidence = fetch_public_index(item, sessions)
            elif item['source'] == 'yfinance':
                if downloader is None:
                    import yfinance as yf
                    downloader = yf.download
                data = downloader([item['ticker']], start=sessions[0],
                    end=(pd.Timestamp(sessions[-1])+pd.Timedelta(days=1)).date().isoformat(),
                    interval='1d', auto_adjust=False, actions=True, repair=False, keepna=True,
                    progress=False, threads=False, timeout=25, group_by='ticker')
                raw = extract_ticker(data, item['ticker'])
                evidence = [{'provider':'Yahoo Finance via yfinance', 'symbol':item['ticker']}]
            else:
                raise ValueError('Unsupported market source: '+item['source'])
            frame, report = normalize_market(raw, item, sessions)
            attempts.append({'attempt':attempt+1, 'status':report['status'], 'rows':report['rows']})
            report.update(source_evidence=evidence, attempts=attempts)
            if report['status'] == 'ready' or item['source'] != 'yfinance' or attempt == 2:
                return frame, report, raw
        except Exception as exc:
            attempts.append({'attempt':attempt+1, 'error':str(exc)[:500]})
            code = getattr(getattr(exc, 'response', None), 'status_code', None) or getattr(exc, 'status_code', None)
            if code in {401,403,404,429} or isinstance(exc, (ValueError, PermissionError)) or attempt == 2:
                return pd.DataFrame(columns=PRICE_COLUMNS), {**item, 'status':'fetch_failed', 'rows':0,
                    'expected_date':sessions[-1], 'latest_date':None, 'missing_sessions':len(sessions),
                    'error':str(exc)[:500], 'attempts':attempts}, pd.DataFrame()
        time.sleep(2 ** (attempt+1))