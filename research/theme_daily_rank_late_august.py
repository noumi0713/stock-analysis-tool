from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path('dashboard-data/technical-backtest-3y')
MEMBERSHIPS = Path('research/data/theme_members_124.csv')
TARGET_START = '2026-08-24'
TARGET_END_EXCLUSIVE = '2026-08-29'
MIN_VALID_MEMBERS = 5
WEIGHTS = {
    'price_strength': 0.25,
    'turnover_inflow': 0.25,
    'breadth': 0.20,
    'relative_strength': 0.15,
    'persistence': 0.15,
}


def load_memberships():
    theme_members = defaultdict(set)
    theme_clusters = {}
    with MEMBERSHIPS.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            t = str(row.get('yahoo_ticker') or '').strip()
            theme = str(row.get('theme_name') or '').strip()
            cluster = str(row.get('cluster') or '').strip()
            if t and theme:
                theme_members[theme].add(t)
                theme_clusters[theme] = cluster
    if len(theme_members) != 124:
        raise ValueError(f'Expected 124 themes, got {len(theme_members)}')
    return dict(theme_members), theme_clusters


def load_history(wanted):
    manifest = json.loads((DATA_DIR / 'manifest.json').read_text(encoding='utf-8'))
    dates = pd.DatetimeIndex(pd.to_datetime(manifest['dates']))
    tickers = sorted(wanted)
    tcol = {t: j for j, t in enumerate(tickers)}
    close = np.full((len(dates), len(tickers)), np.nan)
    volume = np.full_like(close, np.nan)
    for shard in manifest['shards']:
        payload = json.loads((DATA_DIR / shard['path']).read_text(encoding='utf-8'))
        for ticker, bars in payload['bars'].items():
            j = tcol.get(ticker)
            if j is None or not bars:
                continue
            a = np.asarray(bars, dtype=float)
            idx = a[:, 0].astype(int)
            close[idx, j] = a[:, 4]
            volume[idx, j] = a[:, 5]
    return manifest, dates, tickers, close, volume


def fetch_extension(tickers):
    all_close = {}
    all_volume = {}
    batch_size = 150
    for b0 in range(0, len(tickers), batch_size):
        batch = tickers[b0:b0 + batch_size]
        print(f'fetch {b0+1}-{min(b0+batch_size, len(tickers))}/{len(tickers)}', flush=True)
        try:
            df = yf.download(
                batch,
                start=TARGET_START,
                end=TARGET_END_EXCLUSIVE,
                auto_adjust=True,
                actions=False,
                progress=False,
                threads=True,
                group_by='ticker',
                timeout=30,
            )
        except Exception as e:
            print('batch error', repr(e), flush=True)
            continue
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            level0 = set(df.columns.get_level_values(0))
            level1 = set(df.columns.get_level_values(1))
            ticker_first = any(t in level0 for t in batch)
            for t in batch:
                try:
                    sub = df[t] if ticker_first and t in level0 else df.xs(t, level=1, axis=1) if t in level1 else None
                    if sub is None or 'Close' not in sub or 'Volume' not in sub:
                        continue
                    for d, v in sub['Close'].dropna().items():
                        all_close[(pd.Timestamp(d).normalize(), t)] = float(v)
                    for d, v in sub['Volume'].dropna().items():
                        all_volume[(pd.Timestamp(d).normalize(), t)] = float(v)
                except Exception:
                    continue
        elif len(batch) == 1 and 'Close' in df and 'Volume' in df:
            t = batch[0]
            for d, v in df['Close'].dropna().items():
                all_close[(pd.Timestamp(d).normalize(), t)] = float(v)
            for d, v in df['Volume'].dropna().items():
                all_volume[(pd.Timestamp(d).normalize(), t)] = float(v)
        time.sleep(0.15)
    return all_close, all_volume


def pct_change(a, k):
    out = np.full_like(a, np.nan)
    prev, cur = a[:-k], a[k:]
    ok = np.isfinite(prev) & np.isfinite(cur) & (prev > 0)
    block = np.full_like(cur, np.nan)
    block[ok] = cur[ok] / prev[ok] - 1
    out[k:] = block
    return out


def theme_median(matrix, member_idx):
    med = np.full((matrix.shape[0], len(member_idx)), np.nan)
    cnt = np.zeros_like(med, dtype=int)
    for j, idx in enumerate(member_idx):
        if not len(idx):
            continue
        x = matrix[:, idx]
        cnt[:, j] = np.isfinite(x).sum(axis=1)
        with np.errstate(all='ignore'):
            med[:, j] = np.nanmedian(x, axis=1)
        med[cnt[:, j] == 0, j] = np.nan
    return med, cnt


def theme_fraction(mask_value, valid, member_idx):
    out = np.full((mask_value.shape[0], len(member_idx)), np.nan)
    for j, idx in enumerate(member_idx):
        if not len(idx):
            continue
        v = valid[:, idx]
        n = v.sum(axis=1)
        num = (mask_value[:, idx] & v).sum(axis=1)
        ok = n > 0
        out[ok, j] = num[ok] / n[ok]
    return out


def xsec_percentile(raw, eligible):
    x = np.where(eligible, raw, np.nan)
    return pd.DataFrame(x).rank(axis=1, method='average', pct=True).to_numpy() * 100


def main():
    theme_members, theme_clusters = load_memberships()
    themes = sorted(theme_members)
    wanted = set().union(*theme_members.values())
    manifest, dates0, tickers, close0, volume0 = load_history(wanted)
    print('history_end', dates0[-1].date(), 'tickers', len(tickers), flush=True)
    ext_close, ext_volume = fetch_extension(tickers)
    ext_dates = sorted({d for d, _ in ext_close})
    print('extension_dates', [d.strftime('%Y-%m-%d') for d in ext_dates], 'close_points', len(ext_close), flush=True)
    if not ext_dates:
        raise RuntimeError('No extension data downloaded')

    dates = dates0.append(pd.DatetimeIndex([d for d in ext_dates if d > dates0[-1]]))
    dates = pd.DatetimeIndex(sorted(set(dates)))
    dcol = {d.normalize(): i for i, d in enumerate(dates)}
    tcol = {t: j for j, t in enumerate(tickers)}
    close = np.full((len(dates), len(tickers)), np.nan)
    volume = np.full_like(close, np.nan)
    old_pos = [dcol[d.normalize()] for d in dates0]
    close[old_pos, :] = close0
    volume[old_pos, :] = volume0
    for (d, t), v in ext_close.items():
        if d in dcol and t in tcol:
            close[dcol[d], tcol[t]] = v
    for (d, t), v in ext_volume.items():
        if d in dcol and t in tcol:
            volume[dcol[d], tcol[t]] = v

    member_idx = [np.array([tcol[t] for t in sorted(theme_members[th]) if t in tcol], dtype=int) for th in themes]
    ret1 = pct_change(close, 1)
    ret5 = pct_change(close, 5)
    ret10 = pct_change(close, 10)
    ma25 = pd.DataFrame(close).rolling(25, min_periods=20).mean().to_numpy()
    th_ret1, _ = theme_median(ret1, member_idx)
    th_ret5, cnt5 = theme_median(ret5, member_idx)
    th_ret10, cnt10 = theme_median(ret10, member_idx)
    price_raw = 0.60 * th_ret5 + 0.40 * th_ret10

    membership_count = np.zeros(len(tickers))
    M = np.zeros((len(tickers), len(themes)))
    for j, idx in enumerate(member_idx):
        M[idx, j] = 1.0
        membership_count[idx] += 1.0
    membership_count[membership_count <= 0] = 1.0
    turnover = close * volume
    adjusted_turnover = np.nan_to_num(turnover / membership_count[None, :], nan=0.0, posinf=0.0, neginf=0.0)
    th_turnover = adjusted_turnover @ M
    prior20 = pd.DataFrame(th_turnover).shift(1).rolling(20, min_periods=10).median().to_numpy()
    turnover_raw = np.where(prior20 > 0, th_turnover / prior20 - 1.0, np.nan)

    pos5 = theme_fraction(ret5 > 0, np.isfinite(ret5), member_idx)
    above25 = theme_fraction(close > ma25, np.isfinite(close) & np.isfinite(ma25), member_idx)
    breadth_raw = 0.50 * pos5 + 0.50 * above25
    with np.errstate(all='ignore'):
        market_ret5 = np.nanmedian(ret5, axis=1)
    relative_raw = th_ret5 - market_ret5[:, None]
    positive_day = np.where(np.isfinite(th_ret1), (th_ret1 > 0).astype(float), np.nan)
    persistence_raw = pd.DataFrame(positive_day).rolling(5, min_periods=3).mean().to_numpy()

    eligible = ((cnt5 >= MIN_VALID_MEMBERS) & (cnt10 >= MIN_VALID_MEMBERS)
                & np.isfinite(price_raw) & np.isfinite(turnover_raw)
                & np.isfinite(breadth_raw) & np.isfinite(relative_raw) & np.isfinite(persistence_raw))
    raws = {
        'price_strength': price_raw,
        'turnover_inflow': turnover_raw,
        'breadth': breadth_raw,
        'relative_strength': relative_raw,
        'persistence': persistence_raw,
    }
    pcts = {k: xsec_percentile(v, eligible) for k, v in raws.items()}
    score = sum(WEIGHTS[k] * pcts[k] for k in WEIGHTS)
    score[~eligible] = np.nan

    rows = []
    target_dates = pd.date_range(TARGET_START, periods=5, freq='B')
    for d in target_dates:
        i = dcol.get(d.normalize())
        if i is None:
            print('missing date', d.date())
            continue
        valid_n = int(np.isfinite(score[i]).sum())
        order = np.argsort(np.where(np.isfinite(score[i]), -score[i], np.inf))[:10]
        print(f'\nDATE {d.date()} eligible_themes={valid_n}')
        for rankno, j in enumerate(order, 1):
            if not np.isfinite(score[i, j]):
                continue
            row = {
                'date': d.strftime('%Y-%m-%d'), 'rank': rankno, 'theme': themes[j],
                'cluster': theme_clusters.get(themes[j], ''), 'score': round(float(score[i, j]), 2),
                'price': round(float(pcts['price_strength'][i, j]), 1),
                'inflow': round(float(pcts['turnover_inflow'][i, j]), 1),
                'breadth': round(float(pcts['breadth'][i, j]), 1),
                'relative': round(float(pcts['relative_strength'][i, j]), 1),
                'persistence': round(float(pcts['persistence'][i, j]), 1),
            }
            rows.append(row)
            print(f"{rankno:2d}. {themes[j]} | {row['score']:.2f} | P{row['price']} I{row['inflow']} B{row['breadth']} R{row['relative']} C{row['persistence']}")

    out = Path('research-output/late-august-theme-rankings.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding='utf-8-sig')
    print('\nwrote', out, 'rows', len(rows))


if __name__ == '__main__':
    main()
