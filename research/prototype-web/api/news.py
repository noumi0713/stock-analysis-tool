from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, urlencode
import json
import urllib.request


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            q = parse_qs(urlparse(self.path).query).get("q", [""])[0].strip()
            if not q:
                raise ValueError("q required")
            url = "https://api.gdeltproject.org/api/v2/doc/doc?" + urlencode({
                "query": f'"{q}"',
                "mode": "ArtList",
                "maxrecords": "12",
                "format": "json",
                "timespan": "14d",
                "sort": "HybridRel",
            })
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.load(r)
            articles = [
                {
                    "title": a.get("title"),
                    "url": a.get("url"),
                    "domain": a.get("domain"),
                    "date": a.get("seendate"),
                }
                for a in data.get("articles", [])[:12]
            ]
            body = json.dumps({"source": "GDELT DOC 2.0", "articles": articles}, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "s-maxage=600")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            body = json.dumps({"error": str(e)}, ensure_ascii=False).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
