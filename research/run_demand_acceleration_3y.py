from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import demand_acceleration_3y as study


study.THEME_FILE = Path(__file__).resolve().parent / "data" / "theme_members_124.csv"
_original_load_memberships = study.load_memberships


def load_memberships_normalized():
    raw = _original_load_memberships()
    normalized = {}
    for ticker, themes in raw.items():
        s = str(ticker).strip().upper()
        base = s[:-2] if s.endswith('.T') else s
        aliases = {s, base, f'{base}.T'}
        for alias in aliases:
            normalized[alias] = list(themes)
    return normalized


def load_bars_merged():
    manifest = json.loads((study.DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    merged = defaultdict(list)
    for shard in manifest["shards"]:
        payload = json.loads((study.DATA_DIR / shard["path"]).read_text(encoding="utf-8"))
        for ticker, bars in payload["bars"].items():
            merged[ticker].extend(bars)

    all_bars = {}
    for ticker, bars in merged.items():
        dedup = {int(bar[0]): bar for bar in bars}
        all_bars[ticker] = [dedup[idx] for idx in sorted(dedup)]
    return manifest, all_bars


study.load_memberships = load_memberships_normalized
study.load_bars = load_bars_merged
study.main()
