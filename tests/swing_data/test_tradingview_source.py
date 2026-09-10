import json

import pandas as pd
import pytest

from swing_data.market_sources import normalize_market
from swing_data.tradingview_source import fetch_growth_index, fetch_public_index, unpack_messages

ITEM = dict(ticker='^TSEMOTHERS', provider_symbol='TSE:MOS', source='tradingview_public',
            kind='index', field_mode='ohlc', unit='index_points')
IDENTITY = dict(pro_name='TSE:MOS', type='index', exchange='TSE',
                description='TSE Growth Market 250 Index', timezone='Asia/Tokyo',
                bar_transform='none', delay=1200)
TOPIX_ITEM = dict(ticker='^TOPX', provider_symbol='TSE:TOPIX', source='tradingview_public',
                  kind='index', field_mode='ohlc', unit='index_points')
TOPIX_IDENTITY = dict(pro_name='TSE:TOPIX', type='index', exchange='TSE',
                      description='TOPIX Index', timezone='Asia/Tokyo',
                      bar_transform='none', delay=1200)


def wrap(payload):
    return f'~m~{len(payload)}~m~{payload}'


class Socket:
    def __init__(self, identity=None, complete=True, error=False):
        self.identity = IDENTITY if identity is None else identity
        self.complete, self.error, self.closed = complete, error, False
        self.sent, self.received = [], 0

    def send(self, message):
        self.sent.append(message)
        if 'chart_create_session' in message:
            self.session = json.loads(next(unpack_messages(message)))['p'][0]

    def recv(self):
        self.received += 1
        if self.received == 1:
            return wrap('~h~1234')
        if self.received > 2:
            return ''
        def msg(m, p): return wrap(json.dumps({'m':m,'p':[self.session, *p]}))
        if self.error:
            return msg('series_error', ['s1','s1','permission denied'])
        rows = [{'i':i, 'v':[pd.Timestamp(date,tz='UTC').timestamp(), *values]}
                for i,(date,values) in enumerate([
                    ('2026-09-08',[786.06,798.34,785.55,787.06]),
                    ('2026-09-09',[784.44,789.65,779.15,783.76]),
                    ('2026-09-10',[778.12,782.49,775.44,779.25])])]
        response = msg('symbol_resolved', ['symbol_1',self.identity])
        response += msg('timescale_update', [{'s1':{'s':rows}}])
        return response + (msg('series_completed', ['s1','s1']) if self.complete else '')

    def close(self): self.closed = True


def test_public_index_identity_dates_and_completed_bars():
    socket = Socket()
    sessions = ['2026-09-08','2026-09-09']
    raw,evidence = fetch_growth_index(ITEM,sessions,connect=lambda *a,**k:socket)
    data,report = normalize_market(raw,ITEM,sessions)
    assert socket.closed and report['status']=='ready' and len(data)==2
    assert data.close.tolist()==[787.06,783.76]
    assert data.volume.isna().all()
    assert evidence[0]['symbol']=='TSE:MOS' and evidence[0]['authentication']=='none'
    assert wrap('~h~1234') in socket.sent
    assert not any('auth_token' in s for s in socket.sent)


def test_topix_cash_index_is_allowed_but_other_topix_instruments_fail_closed():
    socket = Socket(identity=TOPIX_IDENTITY)
    sessions = ['2026-09-08','2026-09-09','2026-09-10']
    raw,evidence = fetch_public_index(TOPIX_ITEM,sessions,connect=lambda *a,**k:socket)
    data,report = normalize_market(raw,TOPIX_ITEM,sessions)
    assert socket.closed and report['status']=='ready' and len(data)==3
    assert evidence[0]['symbol']=='TSE:TOPIX' and evidence[0]['authentication']=='none'
    assert not any('auth_token' in s for s in socket.sent)

    for change in [
        {'type':'fund'},
        {'pro_name':'TSE:1306'},
        {'description':'TOPIX Futures'},
        {'description':'Japan Broad Market Index'},
        {'timezone':'UTC'},
    ]:
        bad = Socket(identity={**TOPIX_IDENTITY, **change})
        with pytest.raises(ValueError, match='identity'):
            fetch_public_index(TOPIX_ITEM,['2026-09-09'],connect=lambda *a,**k:bad)
        assert bad.closed


def test_unapproved_public_indexes_are_rejected():
    with pytest.raises(ValueError, match='approved'):
        fetch_public_index({**TOPIX_ITEM, 'ticker':'^N225', 'provider_symbol':'TSE:NKY'},
                           ['2026-09-09'])


@pytest.mark.parametrize('change',[{'type':'fund'},{'pro_name':'TSE:2516'},
    {'description':'TSE Growth Market 250 Index Futures'},{'timezone':'UTC'}])
def test_other_instruments_and_timezones_are_rejected(change):
    socket = Socket(identity={**IDENTITY,**change})
    with pytest.raises(ValueError,match='identity'):
        fetch_growth_index(ITEM,['2026-09-09'],connect=lambda *a,**k:socket)
    assert socket.closed


def test_refusal_and_incomplete_series_fail_closed():
    socket=Socket(error=True)
    with pytest.raises(PermissionError):
        fetch_growth_index(ITEM,['2026-09-09'],connect=lambda *a,**k:socket)
    assert socket.closed
    socket=Socket(complete=False)
    with pytest.raises(ValueError,match='before completion'):
        fetch_growth_index(ITEM,['2026-09-09'],connect=lambda *a,**k:socket)
    assert socket.closed
    with pytest.raises(ValueError,match='Truncated'):
        list(unpack_messages('~m~10~m~abc'))
