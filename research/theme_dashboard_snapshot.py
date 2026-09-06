from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path("dashboard-data/technical-backtest-3y")
MEMBERSHIPS = Path("research/data/theme_members_124.csv")
OUT_DIR = Path("theme-dashboard/data")
SNAPSHOT = OUT_DIR / "chatgpt_snapshot.json"
STATUS = OUT_DIR / "update_status.json"
FAILED = OUT_DIR / "candidate_failed.json"

MIN_VALID_MEMBERS = 5
MIN_THEME_COUNT = 120
MIN_STOCK_COVERAGE = 0.95
TOP_N = 10
RECENT_TOP_WINDOW = 5
B_MAX_RANK = 30
DRIVER_N = 5
STANDALONE_N = 10
JST = ZoneInfo("Asia/Tokyo")

WEIGHTS = {
    "price_strength": 0.25,
    "turnover_inflow": 0.25,
    "breadth": 0.20,
    "relative_strength": 0.15,
    "persistence": 0.15,
}


def json_num(value, digits=4):
    if value is None:
        return None
    try:
        x = float(value)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return round(x, digits)


def load_memberships():
    theme_members: dict[str, set[str]] = defaultdict(set)
    theme_clusters: dict[str, str] = {}
    ticker_names: dict[str, str] = {}
    ticker_codes: dict[str, str] = {}
    ticker_themes: dict[str, list[str]] = defaultdict(list)
    with MEMBERSHIPS.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            ticker = str(row.get("yahoo_ticker") or "").strip()
            theme = str(row.get("theme_name") or "").strip()
            if not ticker or not theme:
                continue
            theme_members[theme].add(ticker)
            theme_clusters[theme] = str(row.get("cluster") or "").strip()
            ticker_names[ticker] = str(row.get("company_name") or "").strip()
            ticker_codes[ticker] = str(row.get("stock_code") or ticker.removesuffix(".T")).strip()
            ticker_themes[ticker].append(theme)
    if len(theme_members) != 124:
        raise ValueError(f"Expected 124 themes, got {len(theme_members)}")
    return dict(theme_members), theme_clusters, ticker_names, ticker_codes, dict(ticker_themes)


def load_history(wanted: set[str]):
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    dates = pd.DatetimeIndex(pd.to_datetime(manifest["dates"]))
    manifest_tickers = set((manifest.get("stocks") or {}).keys())
    tickers = sorted(wanted & manifest_tickers)
    tcol = {t: j for j, t in enumerate(tickers)}
    close = np.full((len(dates), len(tickers)), np.nan, dtype=float)
    volume = np.full_like(close, np.nan)
    for shard in manifest["shards"]:
        payload = json.loads((DATA_DIR / shard["path"]).read_text(encoding="utf-8"))
        for ticker, bars in payload["bars"].items():
            j = tcol.get(ticker)
            if j is None or not bars:
                continue
            a = np.asarray(bars, dtype=float)
            idx = a[:, 0].astype(int)
            close[idx, j] = a[:, 4]
            volume[idx, j] = a[:, 5]
    return manifest, dates, tickers, close, volume


def _extract_download(df: pd.DataFrame, batch: list[str], close_map: dict, volume_map: dict):
    if df is None or df.empty:
        return
    if isinstance(df.columns, pd.MultiIndex):
        l0 = set(df.columns.get_level_values(0))
        l1 = set(df.columns.get_level_values(1))
        ticker_first = any(t in l0 for t in batch)
        for ticker in batch:
            try:
                sub = df[ticker] if ticker_first and ticker in l0 else df.xs(ticker, level=1, axis=1) if ticker in l1 else None
                if sub is None:
                    continue
                if "Close" in sub:
                    for d, v in sub["Close"].dropna().items():
                        close_map[(pd.Timestamp(d).normalize(), ticker)] = float(v)
                if "Volume" in sub:
                    for d, v in sub["Volume"].dropna().items():
                        volume_map[(pd.Timestamp(d).normalize(), ticker)] = float(v)
            except Exception:
                continue
    elif len(batch) == 1:
        ticker = batch[0]
        if "Close" in df:
            for d, v in df["Close"].dropna().items():
                close_map[(pd.Timestamp(d).normalize(), ticker)] = float(v)
        if "Volume" in df:
            for d, v in df["Volume"].dropna().items():
                volume_map[(pd.Timestamp(d).normalize(), ticker)] = float(v)


def fetch_recent(tickers: list[str], start: str, end: str):
    close_map: dict[tuple[pd.Timestamp, str], float] = {}
    volume_map: dict[tuple[pd.Timestamp, str], float] = {}
    batch_size = 120
    for b0 in range(0, len(tickers), batch_size):
        batch = tickers[b0:b0 + batch_size]
        ok = False
        for attempt in range(3):
            try:
                df = yf.download(
                    batch,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    actions=False,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                    timeout=30,
                )
                _extract_download(df, batch, close_map, volume_map)
                if df is not None and not df.empty:
                    ok = True
                    break
            except Exception as exc:
                print(f"batch {b0}-{b0+len(batch)} attempt={attempt+1} error={exc}", flush=True)
            time.sleep(10 * (attempt + 1))
        print(f"fetch {b0+1}-{min(b0+batch_size, len(tickers))}/{len(tickers)} ok={ok}", flush=True)
        time.sleep(0.1)
    return close_map, volume_map


def merge_market(dates0, tickers, close0, volume0, close_map, volume_map):
    new_dates = sorted({d for d, _ in close_map if d > dates0[-1].normalize()})
    dates = pd.DatetimeIndex(sorted(set(dates0.tolist() + new_dates)))
    dcol = {d.normalize(): i for i, d in enumerate(dates)}
    tcol = {t: j for j, t in enumerate(tickers)}
    close = np.full((len(dates), len(tickers)), np.nan)
    volume = np.full_like(close, np.nan)
    for i0, d in enumerate(dates0):
        i = dcol[d.normalize()]
        close[i] = close0[i0]
        volume[i] = volume0[i0]
    for (d, t), v in close_map.items():
        i, j = dcol.get(d), tcol.get(t)
        if i is not None and j is not None:
            close[i, j] = v
    for (d, t), v in volume_map.items():
        i, j = dcol.get(d), tcol.get(t)
        if i is not None and j is not None:
            volume[i, j] = v
    return dates, close, volume


def pct_change(a, k):
    out = np.full_like(a, np.nan)
    prev, cur = a[:-k], a[k:]
    ok = np.isfinite(prev) & np.isfinite(cur) & (prev > 0)
    b = np.full_like(cur, np.nan)
    b[ok] = cur[ok] / prev[ok] - 1
    out[k:] = b
    return out


def theme_median(matrix, member_idx):
    out = np.full((matrix.shape[0], len(member_idx)), np.nan)
    cnt = np.zeros_like(out, dtype=int)
    for j, idx in enumerate(member_idx):
        if not len(idx):
            continue
        x = matrix[:, idx]
        cnt[:, j] = np.isfinite(x).sum(axis=1)
        with np.errstate(all="ignore"):
            out[:, j] = np.nanmedian(x, axis=1)
        out[cnt[:, j] == 0, j] = np.nan
    return out, cnt


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


def xsec_pct(raw, eligible):
    return pd.DataFrame(np.where(eligible, raw, np.nan)).rank(axis=1, method="average", pct=True).to_numpy() * 100


def pct_rank(values: np.ndarray):
    return pd.Series(values).rank(method="average", pct=True).to_numpy() * 100


def main() -> int:
    now = datetime.now(JST)
    theme_members, theme_clusters, ticker_names, ticker_codes, ticker_themes = load_memberships()
    wanted = set().union(*theme_members.values())
    manifest, dates0, tickers, close0, volume0 = load_history(wanted)
    ticker_set = set(tickers)
    for theme in list(theme_members):
        theme_members[theme] = {t for t in theme_members[theme] if t in ticker_set}
    ticker_themes = {t: [th for th in themes if t in ticker_set] for t, themes in ticker_themes.items() if t in ticker_set}

    hist_end = dates0[-1].date()
    fetch_start = max(hist_end - timedelta(days=10), now.date() - timedelta(days=120))
    fetch_end = now.date() + timedelta(days=1)
    close_map, volume_map = fetch_recent(tickers, fetch_start.isoformat(), fetch_end.isoformat())
    dates, close, volume = merge_market(dates0, tickers, close0, volume0, close_map, volume_map)

    # Latest sufficiently populated trading date. This avoids selecting a sparse partial day.
    coverage_by_day = np.isfinite(close).sum(axis=1) / max(1, len(tickers))
    viable = np.where(coverage_by_day >= 0.70)[0]
    if not len(viable):
        raise RuntimeError("No sufficiently populated market date")
    i = int(viable[-1])
    market_date = dates[i].strftime("%Y-%m-%d")
    current_coverage = float(coverage_by_day[i])

    themes = sorted(theme_members)
    tcol = {t: j for j, t in enumerate(tickers)}
    member_idx = [np.array([tcol[t] for t in sorted(theme_members[th])], dtype=int) for th in themes]

    ret1 = pct_change(close, 1)
    ret5 = pct_change(close, 5)
    ret10 = pct_change(close, 10)
    ma25 = pd.DataFrame(close).rolling(25, min_periods=20).mean().to_numpy()
    th1, _ = theme_median(ret1, member_idx)
    th5, cnt5 = theme_median(ret5, member_idx)
    th10, cnt10 = theme_median(ret10, member_idx)
    price_raw = 0.60 * th5 + 0.40 * th10

    membership_count = np.zeros(len(tickers), dtype=float)
    M = np.zeros((len(tickers), len(themes)), dtype=float)
    for j, idx in enumerate(member_idx):
        M[idx, j] = 1
        membership_count[idx] += 1
    membership_count[membership_count <= 0] = 1
    turnover = close * volume
    adjusted_turnover = np.nan_to_num(turnover / membership_count[None, :], nan=0, posinf=0, neginf=0)
    th_turn = adjusted_turnover @ M
    prior20 = pd.DataFrame(th_turn).shift(1).rolling(20, min_periods=10).median().to_numpy()
    turnover_raw = np.where(prior20 > 0, th_turn / prior20 - 1, np.nan)

    pos5 = theme_fraction(ret5 > 0, np.isfinite(ret5), member_idx)
    above25 = theme_fraction(close > ma25, np.isfinite(close) & np.isfinite(ma25), member_idx)
    breadth_raw = 0.50 * pos5 + 0.50 * above25
    with np.errstate(all="ignore"):
        market5 = np.nanmedian(ret5, axis=1)
    relative_raw = th5 - market5[:, None]
    posday = np.where(np.isfinite(th1), (th1 > 0).astype(float), np.nan)
    persistence_raw = pd.DataFrame(posday).rolling(5, min_periods=3).mean().to_numpy()

    eligible = (
        (cnt5 >= MIN_VALID_MEMBERS) & (cnt10 >= MIN_VALID_MEMBERS)
        & np.isfinite(price_raw) & np.isfinite(turnover_raw)
        & np.isfinite(breadth_raw) & np.isfinite(relative_raw) & np.isfinite(persistence_raw)
    )
    raws = {
        "price_strength": price_raw,
        "turnover_inflow": turnover_raw,
        "breadth": breadth_raw,
        "relative_strength": relative_raw,
        "persistence": persistence_raw,
    }
    pcts = {k: xsec_pct(v, eligible) for k, v in raws.items()}
    score = sum(WEIGHTS[k] * pcts[k] for k in WEIGHTS)
    score[~eligible] = np.nan
    ranks = pd.DataFrame(score).rank(axis=1, method="first", ascending=False).to_numpy()

    eligible_theme_count = int(np.isfinite(score[i]).sum())
    complete = current_coverage >= MIN_STOCK_COVERAGE and eligible_theme_count >= MIN_THEME_COUNT
    previous = {}
    if SNAPSHOT.exists():
        try:
            previous = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    prev_market_date = str(previous.get("market_date") or "")
    certified = bool(complete and (not prev_market_date or market_date >= prev_market_date))

    prior20_stock_turn = pd.DataFrame(turnover).shift(1).rolling(20, min_periods=10).median().to_numpy()
    stock_turn_ratio = np.where(prior20_stock_turn > 0, turnover / prior20_stock_turn, np.nan)

    def driver_rows(theme_j: int):
        idx = member_idx[theme_j]
        if not len(idx):
            return []
        vals5, vals1, vals10, valtr = ret5[i, idx], ret1[i, idx], ret10[i, idx], stock_turn_ratio[i, idx]
        valid = np.isfinite(vals5) & np.isfinite(vals1) & np.isfinite(vals10)
        if valid.sum() == 0:
            return []
        p5 = pct_rank(np.where(valid, vals5, np.nan))
        p1 = pct_rank(np.where(valid, vals1, np.nan))
        p10 = pct_rank(np.where(valid, vals10, np.nan))
        ptr = pct_rank(np.where(np.isfinite(valtr), valtr, np.nan))
        dscore = 0.45 * p5 + 0.20 * p1 + 0.15 * p10 + 0.20 * ptr
        order = np.argsort(np.where(np.isfinite(dscore), -dscore, np.inf))[:DRIVER_N]
        rows = []
        for n, k in enumerate(order, 1):
            if not np.isfinite(dscore[k]):
                continue
            ticker = tickers[idx[k]]
            rows.append({
                "driver_rank": n,
                "ticker": ticker,
                "code": ticker_codes.get(ticker, ticker.removesuffix(".T")),
                "name": ticker_names.get(ticker) or ticker,
                "driver_score": json_num(dscore[k], 2),
                "return_1d_pct": json_num(vals1[k] * 100, 2),
                "return_5d_pct": json_num(vals5[k] * 100, 2),
                "return_10d_pct": json_num(vals10[k] * 100, 2),
                "theme_excess_5d_pct": json_num((vals5[k] - th5[i, theme_j]) * 100, 2),
                "turnover_ratio_1d_to_20d": json_num(valtr[k], 2),
            })
        return rows

    current_rank = ranks[i]
    prev_i = max(0, i - 1)
    rank5_i = max(0, i - 5)
    recent_start = max(0, i - RECENT_TOP_WINDOW)
    prior5_ranks = ranks[recent_start:i]

    theme_index = {}
    top10 = []
    a_rows = []
    b_rows = []
    c_rows = []
    for j, theme in enumerate(themes):
        if not np.isfinite(current_rank[j]):
            continue
        r = int(current_rank[j])
        prev_r = int(ranks[prev_i, j]) if np.isfinite(ranks[prev_i, j]) else None
        r5 = int(ranks[rank5_i, j]) if np.isfinite(ranks[rank5_i, j]) else None
        was_recent_top10 = bool(np.isfinite(prior5_ranks[:, j]).any() and np.nanmin(prior5_ranks[:, j]) <= TOP_N) if len(prior5_ranks) else False
        status = None
        if r <= TOP_N and prev_r is not None and prev_r <= TOP_N:
            status = "A"
        elif 11 <= r <= B_MAX_RANK and was_recent_top10:
            status = "B"
        elif r >= 31 and was_recent_top10:
            status = "C"
        elif r <= TOP_N:
            status = "NEW_TOP10"
        row = {
            "theme": theme,
            "cluster": theme_clusters.get(theme, ""),
            "rank": r,
            "prev_rank": prev_r,
            "rank_5d_ago": r5,
            "rank_change_1d": None if prev_r is None else prev_r - r,
            "score": json_num(score[i, j], 2),
            "status": status,
            "return_1d_pct": json_num(th1[i, j] * 100, 2),
            "return_5d_pct": json_num(th5[i, j] * 100, 2),
            "return_10d_pct": json_num(th10[i, j] * 100, 2),
            "breadth_positive_5d_pct": json_num(pos5[i, j] * 100, 1),
            "breadth_above_25ma_pct": json_num(above25[i, j] * 100, 1),
            "factor_percentiles": {k: json_num(pcts[k][i, j], 1) for k in WEIGHTS},
            "drivers": driver_rows(j),
        }
        theme_index[theme] = row
        if r <= TOP_N:
            top10.append(row)
        if status == "A":
            a_rows.append(row)
        elif status == "B":
            b_rows.append(row)
        elif status == "C":
            c_rows.append(row)

    top10.sort(key=lambda x: x["rank"])
    a_rows.sort(key=lambda x: x["rank"])
    b_rows.sort(key=lambda x: x["rank"])
    c_rows.sort(key=lambda x: x["rank"])

    # Standalone ranking: exclude any stock whose associated theme is broadly rising.
    theme_j = {theme: j for j, theme in enumerate(themes)}
    rising_theme = {
        theme: bool(np.isfinite(th5[i, j]) and th5[i, j] > 0 and np.isfinite(pos5[i, j]) and pos5[i, j] >= 0.60)
        for theme, j in theme_j.items()
    }
    candidates = []
    excluded_by_theme = 0
    for ticker, themes_for_stock in ticker_themes.items():
        j = tcol.get(ticker)
        if j is None or not np.isfinite(ret5[i, j]) or not np.isfinite(ret1[i, j]):
            continue
        if any(rising_theme.get(th, False) for th in themes_for_stock):
            excluded_by_theme += 1
            continue
        theme_returns = [th5[i, theme_j[th]] for th in themes_for_stock if th in theme_j and np.isfinite(th5[i, theme_j[th]])]
        benchmark = max(theme_returns) if theme_returns else 0.0
        excess = ret5[i, j] - benchmark
        if ret5[i, j] <= 0 or excess <= 0:
            continue
        candidates.append((ticker, j, benchmark, excess))

    standalone = []
    if candidates:
        v5 = np.array([ret5[i, j] for _, j, _, _ in candidates], dtype=float)
        v1 = np.array([ret1[i, j] for _, j, _, _ in candidates], dtype=float)
        vex = np.array([ex for _, _, _, ex in candidates], dtype=float)
        vtr = np.array([stock_turn_ratio[i, j] for _, j, _, _ in candidates], dtype=float)
        ss = 0.45 * pct_rank(v5) + 0.30 * pct_rank(vex) + 0.15 * pct_rank(vtr) + 0.10 * pct_rank(v1)
        order = np.argsort(np.where(np.isfinite(ss), -ss, np.inf))[:STANDALONE_N]
        for n, k in enumerate(order, 1):
            ticker, j, benchmark, excess = candidates[k]
            standalone.append({
                "rank": n,
                "ticker": ticker,
                "code": ticker_codes.get(ticker, ticker.removesuffix(".T")),
                "name": ticker_names.get(ticker) or ticker,
                "standalone_score": json_num(ss[k], 2),
                "return_1d_pct": json_num(ret1[i, j] * 100, 2),
                "return_5d_pct": json_num(ret5[i, j] * 100, 2),
                "best_theme_return_5d_pct": json_num(benchmark * 100, 2),
                "theme_excess_5d_pct": json_num(excess * 100, 2),
                "turnover_ratio_1d_to_20d": json_num(stock_turn_ratio[i, j], 2),
                "themes": themes_for_stock,
                "standalone_reason": "stock_positive_and_outperforming_all_associated_themes; broad-theme-rises excluded",
            })

    quality = {
        "complete": complete,
        "certified": certified,
        "expected_stock_count": len(tickers),
        "latest_stock_count": int(np.isfinite(close[i]).sum()),
        "latest_stock_coverage_pct": round(current_coverage * 100, 2),
        "theme_count": len(themes),
        "eligible_theme_count": eligible_theme_count,
        "min_stock_coverage_pct": MIN_STOCK_COVERAGE * 100,
        "min_eligible_theme_count": MIN_THEME_COUNT,
        "history_source_end": hist_end.isoformat(),
        "standalone_excluded_by_broad_theme_rise": excluded_by_theme,
    }

    snapshot = {
        "schema_version": "1.0",
        "market_date": market_date,
        "updated_at": now.isoformat(timespec="seconds"),
        "quality": quality,
        "definitions": {
            "TOP10": "current 124-theme score rank 1-10",
            "A": "current Top10 and previous trading day also Top10",
            "B": "current rank 11-30 and Top10 at least once in prior 5 trading days",
            "C": "current rank 31+ and Top10 at least once in prior 5 trading days",
            "driver_score": "within-theme 45% 5d return percentile + 20% 1d + 15% 10d + 20% turnover-ratio percentile",
            "standalone": "positive 5d stock return and outperformance vs all associated themes; excluded when any associated theme has positive median 5d return and >=60% positive-5d breadth",
        },
        "top10": top10,
        "A": a_rows,
        "B": b_rows,
        "C": c_rows,
        "standalone": standalone,
        "theme_index": theme_index,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status = {
        "attempted_at": now.isoformat(timespec="seconds"),
        "market_date": market_date,
        "previous_certified_market_date": prev_market_date or None,
        "result": "CERTIFIED" if certified else "FAILED_QUALITY_GATE",
        "quality": quality,
    }
    STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    if not certified:
        FAILED.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 2

    if FAILED.exists():
        FAILED.unlink()
    SNAPSHOT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, value in [("top10", top10), ("a", a_rows), ("b", b_rows), ("c", c_rows), ("standalone", standalone)]:
        payload = {
            "schema_version": "1.0",
            "market_date": market_date,
            "updated_at": snapshot["updated_at"],
            "quality": quality,
            "items": value,
        }
        (OUT_DIR / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "market_date": market_date,
        "top10": len(top10),
        "A": len(a_rows),
        "B": len(b_rows),
        "C": len(c_rows),
        "standalone": len(standalone),
        "quality": quality,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
