from datetime import datetime
import json

import numpy as np
import pandas as pd
import pytest

from swing_data.market_sources import (normalize_market, parse_mof, parse_yahoo_index,
    mof_sessions, fetch_yahoo_index, get_bytes, parse_cboe_vix, fetch_cboe_vix)


def test_cboe_vix_exact_source_and_ohlc():
    raw = parse_cboe_vix(b'DATE,OPEN,HIGH,LOW,CLOSE\n09/09/2026,15.65,16.68,15.57,16.46\n')
    data, r = normalize_market(raw, item(), ['2026-09-09'])
    assert r['status'] == 'ready'
    assert data[['open','high','low','close']].iloc[0].tolist() == [15.65,16.68,15.57,16.46]
    with pytest.raises(ValueError, match='schema'):
        parse_cboe_vix(b'DATE,VIX Close\n09/09/2026,16.46\n')
    with pytest.raises(ValueError, match='another index'):
        fetch_cboe_vix({'ticker':'^TSEMOTHERS'}, ['2026-09-09'])


def item(kind='index', mode='ohlc'):
    return dict(ticker='TEST',kind=kind,field_mode=mode,source='test',unit='percent' if kind=='yield' else 'index_points')


def test_fx_valid_closes_survive_but_invalid_candles_are_not_used():
    dates = pd.bdate_range('2026-03-01', periods=120)
    raw = pd.DataFrame({'Open':151.,'High':152.,'Low':150.,'Close':151.},index=dates)
    raw.loc[dates[7],'High'] = 149.
    data, report = normalize_market(raw,item('fx','close_only'),dates.strftime('%Y-%m-%d').tolist())
    assert report['status']=='ready' and len(data)==120
    assert report['source_ohlc_invalid_rows']==1
    assert report['source_ohlc_issues'][0]['high']==149
    assert data[['open','high','low','adj_high']].isna().all().all()
    assert data.close.eq(151).all()
    raw.loc[dates[8],'Close'] = np.nan
    data, report = normalize_market(raw,item('fx','close_only'),dates.strftime('%Y-%m-%d').tolist())
    assert report['status']=='invalid_rows' and len(data)==119
    assert report['missing_dates']==[dates[8].strftime('%Y-%m-%d')]
    assert 'close_not_finite' in report['rejections'][0]['reasons']


def test_index_rejects_bad_ohlc_but_does_not_require_traded_volume():
    raw = pd.DataFrame({'Open':[101,102],'High':[103,101],'Low':[99,100],'Close':[102,103]},index=pd.to_datetime(['2026-09-08','2026-09-09']))
    data,r = normalize_market(raw,item(),['2026-09-08','2026-09-09'])
    assert r['status']=='invalid_rows' and r['invalid_rows']==1
    assert data.volume.isna().all()
    assert r['rejections'][0]['high']==101


def test_nonpositive_index_ohlc_and_nonfinite_evidence():
    raw = pd.DataFrame({'Open':[101,np.inf],'High':[103,104],'Low':[-1,100],'Close':[102,np.inf]},
                       index=pd.to_datetime(['2026-09-08','2026-09-09']))
    data,r = normalize_market(raw,item(),['2026-09-08','2026-09-09'])
    assert data.empty and r['invalid_rows']==2
    assert 'invalid_ohlc' in r['rejections'][0]['reasons']
    assert 'close_not_finite' in r['rejections'][1]['reasons']
    json.dumps(r,allow_nan=False)


def test_duplicates_stale_and_future_are_not_hidden():
    raw = pd.DataFrame({'Close':[1.,1.,2.,3.]},index=pd.to_datetime(['2026-09-07','2026-09-07','2026-09-08','2026-09-10']))
    data,r = normalize_market(raw,item('yield','close_only'),['2026-09-07','2026-09-08','2026-09-09'])
    assert len(data)==1 and r['invalid_rows']==2 and r['missing_sessions']==2
    assert r['latest_date']=='2026-09-08'


def test_mof_era_and_negative_yields_not_synthetic_ohlc():
    csv='国債金利情報\n基準日,1年,10年\nH31.4.26,0.1,-0.05\nR1.5.7,0.2,0\nR8.9.8,1.848,2.896\n※注意,,\n'
    raw=parse_mof(csv.encode('cp932'))
    data,r=normalize_market(raw,item('yield','close_only'),['2019-04-26','2019-05-07','2026-09-08'])
    assert data.close.tolist()==[-.05,0,2.896]
    assert r['status']=='ready' and data.open.isna().all()
    with pytest.raises(ValueError):parse_mof('title\n日付,1年\n'.encode('cp932'))


def test_mof_publication_clock_weekend_and_long_holiday():
    def last(stamp):return mof_sessions(datetime.fromisoformat(stamp))[-1]
    assert last('2026-09-09T09:29:00+09:00')=='2026-09-07'
    assert last('2026-09-09T09:30:00+09:00')=='2026-09-08'
    assert last('2026-09-13T18:00:00+09:00')=='2026-09-10'
    assert last('2026-09-24T09:29:00+09:00')=='2026-09-17'
    assert last('2026-09-24T09:30:00+09:00')=='2026-09-18'


def test_yahoo_identity_and_explicit_column_order():
    doc='<html><head><meta charset="utf-8"><title>TOPIX：時系列</title></head><body><table><tr><th>日付</th><th>始値</th><th>高値</th><th>安値</th><th>終値</th></tr><tr><th>2026/9/9</th><td>4,000</td><td>4,100</td><td>3,950</td><td>4,050</td></tr></table></body></html>'.encode()
    raw=parse_yahoo_index(doc,'TOPIX')
    assert raw.iloc[0].tolist()==[4000,4100,3950,4050]
    with pytest.raises(ValueError):parse_yahoo_index(doc,'東証グロース市場250指数')
    class Response:
        content=doc
        def raise_for_status(self): pass
    with pytest.raises(ValueError,match='pagination'):
        fetch_yahoo_index({'provider_symbol':'998405.T','identity_name':'TOPIX'},['2026-03-17','2026-09-09'],get=lambda *a,**k:Response())


def test_access_denial_and_rate_limit_not_retried():
    import requests
    for status in [401,403,404,429]:
        calls=[]
        def get(*a,**k):
            calls.append(1)
            r=requests.Response();r.status_code=status
            r.raise_for_status()
        with pytest.raises(requests.HTTPError):get_bytes('https://example.com',get=get)
        assert len(calls)==1
