from datetime import datetime
from zoneinfo import ZoneInfo
import json
from io import BytesIO
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pytest
from swing_data.supplemental import parse_margin,parse_tdnet,parse_fundamental,parse_forecast_xbrl,stock_code
from swing_data.indicators import calculate


def test_margin_unit_date_and_component_validation():
    header='銘柄別信用取引週末残高 Unit: 1 share Outstanding Sales Outstanding Purchases\nAs of 2026/9/4 application based （単位：一株） 2026/9/8\n'
    line='name 166A0 JP3464310006 100 ▲ 10 300 20 40 0 60 -10 200 10 100 10'
    row=parse_margin(header+line,'source')[0]
    assert row['code']=='166A' and row['as_of']=='2026-09-04' and row['published_on']=='2026-09-08'
    assert row['buy_shares']==300 and row['sell_shares']==100
    with pytest.raises(ValueError,match='disagree'): parse_margin(header+line.replace('40 0','41 0'),'source')
    with pytest.raises(ValueError): parse_margin(header.replace('Unit: 1 share','Unit: 1000 shares')+line,'source')


def test_tdnet_pagination_and_no_upward_inference():
    data=b'<html><div onclick="pagerLink(\'I_list_002_20260910.html\')"></div><table><tr><td class="kjCode">40510</td><td class="kjTime">15:00</td><td class="kjTitle"><a href="x.pdf">'+ '業績予想の修正'.encode()+b'</a></td></tr></table></html>'
    rows,pages=parse_tdnet(data,'https://www.release.tdnet.info/inbs/I_list_001_20260910.html','2026-09-10')
    assert rows[0]['direction']=='unconfirmed'
    assert len(pages)==1 and next(iter(pages)).endswith('002_20260910.html')
    assert parse_tdnet('開示された情報はありません'.encode(),'url','2026-09-10')[0]==[]
    with pytest.raises(ValueError): parse_tdnet(b'<html>error</html>','url','2026-09-10')


def test_eps_period_currency_and_missing_float():
    payload={'quoteSummary':{'result':[{'price':{'symbol':'7203.T','currency':'JPY','marketCap':{'raw':100}},'earningsTrend':{'trend':[{'period':'0y','endDate':'2027-03-31','earningsEstimate':{'earningsCurrency':'JPY','numberOfAnalysts':{'raw':5}},'epsTrend':{'current':{'raw':-10},'30daysAgo':{'raw':-12}}}]}}]}}
    row=parse_fundamental(payload,'7203','now')
    assert row['float_shares'] is None and row['capital_status']=='partial'
    p=row['consensus_periods'][0]
    assert p['eps_trend']['current']==-10 and p['end_date']=='2027-03-31' and p['number_of_analysts']==5
    with pytest.raises(ValueError): parse_fundamental(payload,'6758','now')


def test_xbrl_pairs_require_same_context_and_respect_scale():
    raw='''<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"><ix:nonNumeric name="t:SecuritiesCode" contextRef="instant">72030</ix:nonNumeric><ix:nonFraction name="t:NetSales" contextRef="Year_ConsolidatedMember_PreviousMember_ForecastMember" scale="6" unitRef="JPY">1,000</ix:nonFraction><ix:nonFraction name="t:NetSales" contextRef="Year_ConsolidatedMember_CurrentMember_ForecastMember" scale="6" unitRef="JPY">1,200</ix:nonFraction><ix:nonFraction name="t:NetSales" contextRef="Year_NonConsolidatedMember_CurrentMember_ForecastMember" scale="6" unitRef="JPY">5</ix:nonFraction></html>'''
    buf=BytesIO()
    with ZipFile(buf,'w') as z: z.writestr('x.htm',raw)
    pairs=parse_forecast_xbrl(buf.getvalue(),'7203')
    assert len(pairs)==1 and pairs[0]['previous']==1e9 and pairs[0]['direction']=='up'
    with pytest.raises(ValueError): parse_forecast_xbrl(buf.getvalue(),'6758')


def prices():
    c=np.arange(100,220,dtype=float)
    return pd.DataFrame({'date':pd.date_range('2026-01-01',periods=120).strftime('%Y-%m-%d'),'adj_close':c,'adj_high':c+2,'adj_low':c-2,'close':c,'volume':np.full(120,100.)})


def test_indicators_known_trend_and_exact_date_alignment():
    f=prices(); b=f[['date','close']].copy()
    out=calculate(f,b)
    assert out['rsi_14']==100 and out['atr_14_adjusted']==4
    assert out['ma']['20']==209.5 and out['relative_strength_pct']['20']==0
    assert out['turnover_ratio_20d']==pytest.approx(219/208.5)
    assert calculate(f,b.iloc[:-1])['relative_strength_pct']['20'] is None
    assert 'rsi_14' not in f.columns
    f['adj_close']=100;f['adj_high']=102;f['adj_low']=98
    assert calculate(f,b)['rsi_14']==50
    with pytest.raises(ValueError): calculate(f.iloc[:-1],b)
