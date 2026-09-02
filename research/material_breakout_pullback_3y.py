from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("momentum5d/data/yahoo/processed/equities_daily.parquet")
OUTPUT_DIR = Path("research/results")
OUTPUT_JSON = OUTPUT_DIR / "material_breakout_pullback_3y.json"
OUTPUT_CSV = OUTPUT_DIR / "material_breakout_pullback_trades_3y.csv"


@dataclass(frozen=True)
class Config:
    impulse_volume_ratio: float = 3.0
    impulse_return_min: float = 0.08
    close_from_high_max: float = 0.05
    turnover_min_yen: float = 500_000_000.0
    persistence_volume_ratio: float = 1.5
    pullback_min: float = 0.05
    pullback_max: float = 0.15
    pullback_search_days: int = 3
    recovery_search_days: int = 2
    cooldown_days: int = 10


def _float(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def prepare() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise SystemExit(f"market data cache not found: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "open", "high", "low", "close", "volume"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df[(df["close"] > 0) & (df["volume"] >= 0)]
    df = df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")

    if "adjusted_close" in df.columns:
        adj = pd.to_numeric(df["adjusted_close"], errors="coerce")
        ratio = (adj / df["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        ratio = ratio.where(ratio > 0, 1.0)
    else:
        ratio = pd.Series(1.0, index=df.index)
    for col in ["open", "high", "low", "close"]:
        df[f"adj_{col}"] = df[col] * ratio

    if "turnover_value" in df.columns:
        turnover = pd.to_numeric(df["turnover_value"], errors="coerce")
        df["turnover"] = turnover.fillna(df["close"] * df["volume"])
    else:
        df["turnover"] = df["close"] * df["volume"]

    g = df.groupby("ticker", group_keys=False)
    df["prev_close"] = g["adj_close"].shift(1)
    df["return_1d"] = df["adj_close"] / df["prev_close"] - 1.0
    df["volume_ma20_prev"] = g["volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).mean()
    )
    df["volume_ratio20"] = df["volume"] / df["volume_ma20_prev"]
    df["close_from_high"] = 1.0 - (df["adj_close"] / df["adj_high"])

    max_date = df["date"].max()
    start = max_date - pd.DateOffset(years=3)
    df["in_test_window"] = df["date"] >= start
    return df.reset_index(drop=True)


def detect(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    # Vectorize the expensive whole-universe filters. Python loops only over the
    # small set of qualifying impulse events; trading rules are unchanged.
    impulse_mask = (
        df["in_test_window"]
        & df["return_1d"].notna()
        & df["volume_ratio20"].notna()
        & df["return_1d"].ge(cfg.impulse_return_min)
        & df["volume_ratio20"].ge(cfg.impulse_volume_ratio)
    )
    quality_mask = (
        impulse_mask
        & df["close_from_high"].le(cfg.close_from_high_max)
        & df["turnover"].ge(cfg.turnover_min_yen)
    )
    counts = {
        "impulse": int(impulse_mask.sum()),
        "quality_impulse": int(quality_mask.sum()),
        "persistence_and_pullback": 0,
        "recovery_confirmation": 0,
    }
    trades: list[dict] = []

    for ticker, group in df.groupby("ticker", sort=False):
        candidate_local = np.flatnonzero(quality_mask.loc[group.index].to_numpy())
        if len(candidate_local) == 0:
            continue
        x = group.reset_index(drop=True)
        last_entry_idx = -10_000
        n = len(x)

        for i in candidate_local.tolist():
            if i == 0 or i >= n - 2:
                continue
            r = x.iloc[i]
            pre_impulse_close = float(x.iloc[i - 1]["adj_close"])
            impulse_high = float(r["adj_high"])

            found_pullback = None
            found_recovery = None
            max_pullback_idx = min(n - 2, i + cfg.pullback_search_days)
            for j in range(i + 1, max_pullback_idx + 1):
                p = x.iloc[j]
                if not pd.notna(p["volume_ratio20"]):
                    continue
                interim = x.iloc[i + 1 : j + 1]["volume_ratio20"]
                if interim.empty or (interim < cfg.persistence_volume_ratio).any():
                    continue
                depth = 1.0 - float(p["adj_close"]) / impulse_high
                if not (cfg.pullback_min <= depth <= cfg.pullback_max):
                    continue
                if float(p["adj_close"]) < pre_impulse_close:
                    continue
                found_pullback = j

                max_recovery_idx = min(n - 2, j + cfg.recovery_search_days)
                for k in range(j + 1, max_recovery_idx + 1):
                    c = x.iloc[k]
                    observed = x.iloc[i + 1 : k + 1]["volume_ratio20"]
                    if observed.empty or (observed < cfg.persistence_volume_ratio).any():
                        continue
                    if not (
                        float(c["adj_close"]) > float(c["adj_open"])
                        and float(c["adj_close"]) > float(x.iloc[k - 1]["adj_close"])
                        and float(c["volume"]) > float(x.iloc[k - 1]["volume"])
                    ):
                        continue
                    found_recovery = k
                    break
                if found_recovery is not None:
                    break

            if found_pullback is None:
                continue
            counts["persistence_and_pullback"] += 1
            if found_recovery is None:
                continue
            counts["recovery_confirmation"] += 1

            k = found_recovery
            entry_idx = k + 1
            if entry_idx >= n or entry_idx - last_entry_idx <= cfg.cooldown_days:
                continue
            entry = x.iloc[entry_idx]
            if pd.isna(entry["adj_open"]) or float(entry["adj_open"]) <= 0:
                continue
            entry_price = float(entry["adj_open"])
            record = {
                "ticker": str(ticker),
                "impulse_date": str(r["date"].date()),
                "pullback_date": str(x.iloc[found_pullback]["date"].date()),
                "signal_date": str(x.iloc[k]["date"].date()),
                "entry_date": str(entry["date"].date()),
                "entry_price": entry_price,
                "impulse_return_pct": float(r["return_1d"] * 100),
                "impulse_volume_ratio": float(r["volume_ratio20"]),
                "impulse_turnover_yen": float(r["turnover"]),
                "impulse_close_from_high_pct": float(r["close_from_high"] * 100),
                "pullback_depth_pct": float(
                    (1.0 - float(x.iloc[found_pullback]["adj_close"]) / impulse_high) * 100
                ),
                "entry_gap_pct": float(
                    (entry_price / float(x.iloc[k]["adj_close"]) - 1.0) * 100
                ),
            }
            for horizon in (5, 10, 20):
                end_idx = entry_idx + horizon - 1
                if end_idx >= n:
                    record[f"ret_{horizon}d_pct"] = None
                    record[f"mfe_{horizon}d_pct"] = None
                    record[f"mae_{horizon}d_pct"] = None
                    record[f"hit_plus5_{horizon}d"] = None
                    record[f"hit_plus10_{horizon}d"] = None
                    record[f"hit_minus5_{horizon}d"] = None
                    record[f"hit_minus8_{horizon}d"] = None
                    continue
                path = x.iloc[entry_idx : end_idx + 1]
                end_close = float(x.iloc[end_idx]["adj_close"])
                max_high = float(path["adj_high"].max())
                min_low = float(path["adj_low"].min())
                record[f"ret_{horizon}d_pct"] = (end_close / entry_price - 1.0) * 100
                record[f"mfe_{horizon}d_pct"] = (max_high / entry_price - 1.0) * 100
                record[f"mae_{horizon}d_pct"] = (min_low / entry_price - 1.0) * 100
                record[f"hit_plus5_{horizon}d"] = bool(max_high >= entry_price * 1.05)
                record[f"hit_plus10_{horizon}d"] = bool(max_high >= entry_price * 1.10)
                record[f"hit_minus5_{horizon}d"] = bool(min_low <= entry_price * 0.95)
                record[f"hit_minus8_{horizon}d"] = bool(min_low <= entry_price * 0.92)
            trades.append(record)
            last_entry_idx = entry_idx

    return pd.DataFrame(trades), counts


def summarize(trades: pd.DataFrame) -> dict:
    result: dict[str, object] = {"trades": int(len(trades))}
    if trades.empty:
        return result
    for horizon in (5, 10, 20):
        col = f"ret_{horizon}d_pct"
        valid = trades.dropna(subset=[col])
        if valid.empty:
            continue
        result[f"{horizon}d"] = {
            "n": int(len(valid)),
            "mean_return_pct": _float(valid[col].mean()),
            "median_return_pct": _float(valid[col].median()),
            "win_rate_pct": _float((valid[col] > 0).mean() * 100),
            "plus5_hit_rate_pct": _float(valid[f"hit_plus5_{horizon}d"].mean() * 100),
            "plus10_hit_rate_pct": _float(valid[f"hit_plus10_{horizon}d"].mean() * 100),
            "minus5_hit_rate_pct": _float(valid[f"hit_minus5_{horizon}d"].mean() * 100),
            "minus8_hit_rate_pct": _float(valid[f"hit_minus8_{horizon}d"].mean() * 100),
            "median_mfe_pct": _float(valid[f"mfe_{horizon}d_pct"].median()),
            "median_mae_pct": _float(valid[f"mae_{horizon}d_pct"].median()),
            "p10_return_pct": _float(valid[col].quantile(0.10)),
            "p90_return_pct": _float(valid[col].quantile(0.90)),
        }
    return result


def yearly(trades: pd.DataFrame) -> list[dict]:
    if trades.empty:
        return []
    t = trades.copy()
    t["year"] = pd.to_datetime(t["entry_date"]).dt.year
    out = []
    for year, part in t.groupby("year"):
        out.append({"year": int(year), **summarize(part)})
    return out


def main() -> None:
    df = prepare()
    max_date = df["date"].max()
    test_start = max_date - pd.DateOffset(years=3)
    base = Config()
    variants = {
        "base": base,
        "volume_2x": replace(base, impulse_volume_ratio=2.0),
        "volume_4x": replace(base, impulse_volume_ratio=4.0),
        "turnover_150m": replace(base, turnover_min_yen=150_000_000.0),
        "turnover_1b": replace(base, turnover_min_yen=1_000_000_000.0),
    }

    payload = {
        "method": "material_breakout_pullback_event_study_v1",
        "status": "complete",
        "period": {"start": str(test_start.date()), "end": str(max_date.date())},
        "data": {
            "rows_total": int(len(df)),
            "tickers_total": int(df["ticker"].nunique()),
            "source": str(DATA_PATH),
        },
        "important_limitations": [
            "Historical news/material labels are not available point-in-time, so a major material is proxied by >=8% return plus abnormal volume.",
            "Historical point-in-time theme membership is not available in this branch; the theme-breadth condition is intentionally omitted rather than approximated with hindsight.",
            "This is an event study, not a portfolio backtest. No capital allocation, ranking, portfolio max drawdown, fees, or slippage are assumed.",
            "Entry is the next session open after recovery confirmation to avoid using information not known at the signal close.",
        ],
        "base_config": asdict(base),
        "variants": {},
    }

    base_trades = None
    for name, cfg in variants.items():
        trades, counts = detect(df, cfg)
        excl_tier4 = (
            trades.loc[~trades["ticker"].astype(str).str.startswith("593A")]
            if not trades.empty
            else trades
        )
        payload["variants"][name] = {
            "config": asdict(cfg),
            "stage_counts": counts,
            "summary": summarize(trades),
            "summary_excluding_593A": summarize(excl_tier4),
            "yearly": yearly(trades),
        }
        if name == "base":
            base_trades = trades

    assert base_trades is not None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    base_trades.to_csv(OUTPUT_CSV, index=False)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
