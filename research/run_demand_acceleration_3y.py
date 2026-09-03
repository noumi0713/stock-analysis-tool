from __future__ import annotations

import demand_acceleration_3y as study


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


study.load_memberships = load_memberships_normalized
study.main()
