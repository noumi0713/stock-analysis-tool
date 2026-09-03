from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, quote
import json
import urllib.request

UA = "Mozilla/5.0"


def sma(xs, n):
    return sum(xs[-n:]) / n if len(xs) >= n else None


def rsi(closes, n=14):
    if len(closes) < n + 1:
        return None
    gains, losses = [], []
    for i in range(-n, 0):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = sum(gains) / n
    al = sum(losses) / n
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


def atr(highs, lows, closes, n=14):
    if len(closes) < n + 1:
        return None
    vals = []
    for i in range(len(closes) - n, len(closes)):
        vals.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
    return sum(vals) / n


def fetch_one(code):
    symbol = code if code.endswith(".T") else f"{code}.T"
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        + quote(symbol)
        + "?range=6mo&interval=1d&events=div%2Csplits&includeAdjustedClose=true"
    )
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=12) as r:
        raw = json.load(r)
    result = raw["chart"]["result"][0]
    q = result["indicators"]["quote"][0]
    ts = result.get("timestamp", [])
    rows = []
    for i, t in enumerate(ts):
        vals = [q.get(k, [None] * len(ts))[i] for k in ("open", "high", "low", "close", "volume")]
        if any(v is None for v in vals[:4]):
            continue
        rows.append((t, *vals))
    if len(rows) < 30:
        raise ValueError("insufficient bars")

    opens = [x[1] for x in rows]
    highs = [x[2] for x in rows]
    lows = [x[3] for x in rows]
    closes = [x[4] for x in rows]
    vols = [x[5] or 0 for x in rows]
    c = closes[-1]
    ma25 = sma(closes, 25)
    ma75 = sma(closes, 75)
    rr = rsi(closes)
    aa = atr(highs, lows, closes)
    high20 = max(highs[-20:])
    avg20 = sum(vols[-20:]) / 20
    vr = vols[-1] / avg20 if avg20 else None
    ret5 = c / closes[-6] - 1
    ret20 = c / closes[-21] - 1
    ma25d = c / ma25 - 1 if ma25 else None
    close_pos = (c - lows[-1]) / (highs[-1] - lows[-1]) if highs[-1] > lows[-1] else 0.5

    score = 0.0
    if rr is not None:
        score += max(0, min(30, (rr - 55) * 1.5))
    score += max(0, min(25, ret5 * 125))
    score += max(0, min(20, ret20 * 50))
    if ma25d is not None:
        score += max(0, min(15, ma25d * 100))
    if vr is not None and vr > 2:
        score += min(10, (vr - 2) * 4)

    return {
        "code": code.replace(".T", ""),
        "symbol": symbol,
        "close": c,
        "open": opens[-1],
        "high": highs[-1],
        "low": lows[-1],
        "volume": vols[-1],
        "rsi14": rr,
        "ma25": ma25,
        "ma75": ma75,
        "distance_ma25": ma25d,
        "high20": high20,
        "drawdown20": c / high20 - 1,
        "volume_ratio20": vr,
        "return_1d": c / closes[-2] - 1,
        "return_5d": ret5,
        "return_20d": ret20,
        "atr14_pct": aa / c if aa else None,
        "close_position": close_pos,
        "overheat_score": max(0, min(100, score)),
        "trend_order": bool(ma25 and ma75 and c > ma25 > ma75),
        "bars": [
            {"t": x[0], "o": x[1], "h": x[2], "l": x[3], "c": x[4], "v": x[5]}
            for x in rows[-45:]
        ],
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            codes = [
                x.strip()
                for x in parse_qs(urlparse(self.path).query).get("codes", [""])[0].split(",")
                if x.strip()
            ][:30]
            out = []
            for code in codes:
                try:
                    out.append({"ok": True, **fetch_one(code)})
                except Exception as e:
                    out.append({"ok": False, "code": code, "error": str(e)})
            body = json.dumps({"source": "Yahoo Finance chart endpoint", "items": out}, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "s-maxage=300")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            body = json.dumps({"error": str(e)}, ensure_ascii=False).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
