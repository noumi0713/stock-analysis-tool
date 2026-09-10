"""Read unauthenticated delayed TSE index charts; never substitute futures or ETFs."""
from __future__ import annotations

import json
import re
import time
import uuid
from urllib.parse import quote

import pandas as pd


def unpack_messages(packet):
    """A WebSocket message can contain multiple length-prefixed protocol messages."""
    while packet:
        header = re.match(r'~m~(\d+)~m~', packet)
        if not header:
            raise ValueError('Unexpected public chart protocol framing')
        length = int(header[1])
        payload = packet[header.end():header.end()+length]
        if len(payload) != length:
            raise ValueError('Truncated public chart protocol message')
        packet = packet[header.end()+length:]
        yield payload


def _urls(provider_symbol):
    exchange, symbol = provider_symbol.split(':', 1)
    slug = f'{exchange}-{symbol}'
    socket_url = 'wss://data.tradingview.com/socket.io/websocket?from=' + quote(f'symbols/{slug}/', safe='')
    page_url = f'https://www.tradingview.com/symbols/{slug}/'
    return socket_url, page_url


def validate_identity(meta, item):
    expected_symbol = item.get('provider_symbol')
    if (not expected_symbol or meta.get('pro_name') != expected_symbol
            or meta.get('type') != 'index' or meta.get('exchange') != 'TSE'
            or meta.get('timezone') != 'Asia/Tokyo'
            or meta.get('bar_transform') != 'none'):
        raise ValueError('Public index identity mismatch')

    description = str(meta.get('description') or '')
    if expected_symbol == 'TSE:MOS':
        if description != 'TSE Growth Market 250 Index':
            raise ValueError('Public index identity mismatch')
    elif expected_symbol == 'TSE:TOPIX':
        upper = description.upper()
        forbidden = ('FUTURE', 'ETF', 'FUND')
        if 'TOPIX' not in upper or any(word in upper for word in forbidden):
            raise ValueError('Public index identity mismatch')
    else:
        raise ValueError('Public index source is restricted to approved TSE indexes')


def fetch_public_index(item, sessions, *, connect=None):
    from swing_data.market_sources import source_evidence

    approved = {
        ('^TSEMOTHERS', 'TSE:MOS'),
        ('^TOPX', 'TSE:TOPIX'),
    }
    if (item.get('ticker'), item.get('provider_symbol')) not in approved:
        raise ValueError('Public index source is restricted to approved TSE indexes')

    socket_url, page_url = _urls(item['provider_symbol'])
    if connect is None:
        from websocket import create_connection
        connect = create_connection
    socket = connect(socket_url, timeout=15, origin='https://www.tradingview.com')
    session = 'cs_' + uuid.uuid4().hex[:12]
    transcript, rows, identity, completed = [], {}, None, False

    def send(method, params):
        payload = json.dumps({'m': method, 'p': params}, separators=(',', ':'))
        socket.send(f'~m~{len(payload)}~m~{payload}')

    try:
        # No login, credential, paid feed, or authentication token is used.
        send('chart_create_session', [session, ''])
        send('resolve_symbol', [session, 'symbol_1', '=' + json.dumps(
            {'symbol': item['provider_symbol'], 'session': 'regular'})])
        send('create_series', [session, 's1', 's1', 'symbol_1', '1D', len(sessions)+30])
        deadline = time.monotonic() + 45
        for _ in range(32):
            if time.monotonic() >= deadline:
                break
            packet = socket.recv()
            if not isinstance(packet, str) or not packet:
                raise ValueError('Public chart connection closed before completion')
            transcript.append(packet)
            for payload in unpack_messages(packet):
                if payload.startswith('~h~'):
                    socket.send(f'~m~{len(payload)}~m~{payload}')
                    continue
                message = json.loads(payload)
                method, params = message.get('m', ''), message.get('p', [])
                if 'error' in method or 'permission' in method:
                    raise PermissionError(f'Public chart refused request: {method}')
                if method not in {'symbol_resolved', 'timescale_update', 'series_completed'}:
                    continue
                if not params or params[0] != session:
                    raise ValueError('Public chart response session mismatch')
                if method == 'symbol_resolved':
                    identity = params[2]
                    validate_identity(identity, item)
                elif method == 'timescale_update':
                    for row in params[1].get('s1', {}).get('s', []):
                        values = row['v']
                        if len(values) != 5:
                            raise ValueError('Unexpected public index OHLC schema')
                        if row['i'] in rows and rows[row['i']] != values:
                            raise ValueError('Public index historical bar changed within response')
                        rows[row['i']] = values
                elif params[1] == 's1':
                    completed = True
            if completed:
                break
        if not completed or identity is None or not rows:
            raise ValueError('Public index historical series did not complete')
        raw = pd.DataFrame([rows[k] for k in sorted(rows)],
                           columns=['timestamp', 'Open', 'High', 'Low', 'Close'])
        raw.index = pd.to_datetime(raw.pop('timestamp'), unit='s', utc=True).dt.tz_convert(identity['timezone'])
        raw.index.name = 'date'
        evidence = source_evidence(socket_url, '\n'.join(transcript).encode('utf-8'))
        evidence.update(page_url=page_url, symbol=identity['pro_name'],
                        description=identity['description'], timezone=identity['timezone'],
                        delay_seconds=identity.get('delay'), authentication='none')
        return raw, [evidence]
    finally:
        socket.close()


def fetch_growth_index(item, sessions, *, connect=None):
    """Backward-compatible wrapper retained for existing tests and callers."""
    if item.get('provider_symbol') != 'TSE:MOS' or item.get('ticker') != '^TSEMOTHERS':
        raise ValueError('Public Growth 250 source is restricted to TSE:MOS')
    return fetch_public_index(item, sessions, connect=connect)
