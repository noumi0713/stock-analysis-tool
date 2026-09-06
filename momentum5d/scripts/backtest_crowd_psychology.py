from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRAINING_CODES = {
    "285A", "6976", "593A", "278A", "5801", "5803", "6857", "6920", "9348", "7013", "4592"
}

LIQUIDITY_YEN = 500_000_000.0
HORIZON = 10
COOLDOWN = 5


def _scale(series: pd.Series, lo: float, hi: float) -> pd.Series:
    return ((series - lo) / (hi - lo) * 100.0).clip(0.0, 100.0)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(loss.ne(0.0), 100.0)
    return rsi


def _dedupe(mask: pd.Series, cooldown: int = COOLDOWN) -> np.ndarray:
    raw = mask.fillna(False).to_numpy(dtype=bool)
    out = np.zeros(len(raw), dtype=bool)
    last = -10_000
    for i, value in enumerate(raw):
        if value and i - last >= cooldown:
            out[i] = True
            last = i
    return out


def _period_label(date: pd.Timestamp) -> str:
    if date < pd.Timestamp("2024-08-21"):
        return "Y1_2023-08-21_2024-08-20"
    if date < pd.Timestamp("2025-08-21"):
        return "Y2_2024-08-21_2025-08-20"
    return "Y3_2025-08-21_2026-08-21"


def _sentiment_bucket(score: float) -> str:
    if score < 20:
        return "0-20 極端な悲観"
    if score < 40:
        return "20-40 悲観"
    if score < 60:
        return "40-60 中立"
    if score < 80:
        return "60-80 楽観"
    return "80-100 熱狂"


def _load_prices(data_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    dates = pd.to_datetime(manifest["dates"])
    frames: list[pd.DataFrame] = []

    for shard in manifest["shards"]:
        payload = json.loads((data_dir / shard["path"]).read_text(encoding="utf-8"))
        for ticker, bars in payload["bars"].items():
            if not bars:
                continue
            arr = np.asarray(bars, dtype=float)
            idx = arr[:, 0].astype(int)
            frame = pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates[idx],
                    "open": arr[:, 1],
                    "high": arr[:, 2],
                    "low": arr[:, 3],
                    "close": arr[:, 4],
                    "volume": arr[:, 5],
                }
            )
            frames.append(frame)

    prices = pd.concat(frames, ignore_index=True)
    prices.sort_values(["ticker", "date"], inplace=True)
    prices.reset_index(drop=True, inplace=True)
    return prices, manifest


def _metric_summary(rows: pd.DataFrame, baseline_mean: float | None = None) -> dict[str, float | int | None]:
    rows = rows.loc[rows["fwd10"].notna()].copy()
    if rows.empty:
        return {
            "count": 0,
            "mean_fwd10": None,
            "median_fwd10": None,
            "win_rate": None,
            "plus5_rate": None,
            "minus5_rate": None,
            "avg_mfe10": None,
            "avg_mae10": None,
            "edge_vs_baseline": None,
        }

    mean_ret = float(rows["fwd10"].mean())
    return {
        "count": int(len(rows)),
        "mean_fwd10": mean_ret,
        "median_fwd10": float(rows["fwd10"].median()),
        "win_rate": float((rows["fwd10"] > 0).mean()),
        "plus5_rate": float((rows["mfe10"] >= 0.05).mean()),
        "minus5_rate": float((rows["mae10"] <= -0.05).mean()),
        "avg_mfe10": float(rows["mfe10"].mean()),
        "avg_mae10": float(rows["mae10"].mean()),
        "edge_vs_baseline": None if baseline_mean is None else float(mean_ret - baseline_mean),
    }


def _fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def run_backtest(data_dir: Path) -> dict[str, Any]:
    prices, manifest = _load_prices(data_dir)
    stock_meta = manifest.get("stocks", {})

    signals: list[dict[str, Any]] = []
    baseline_rows: list[pd.DataFrame] = []
    sentiment_rows: list[pd.DataFrame] = []

    for ticker, g in prices.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True).copy()
        if len(g) < 20:
            continue

        close = g["close"]
        high = g["high"]
        low = g["low"]
        volume = g["volume"]
        turnover = close * volume

        g["history_n"] = np.arange(1, len(g) + 1)
        g["ret1"] = close.pct_change(1)
        g["ret5"] = close.pct_change(5)
        g["ret10"] = close.pct_change(10)
        g["ret20"] = close.pct_change(20)
        g["ma10"] = close.rolling(10, min_periods=10).mean()
        g["ma25"] = close.rolling(25, min_periods=25).mean()
        g["ma75"] = close.rolling(75, min_periods=75).mean()
        g["ma10_slope5"] = g["ma10"] / g["ma10"].shift(5) - 1.0
        g["ma25_slope5"] = g["ma25"] / g["ma25"].shift(5) - 1.0
        g["rsi14"] = _rsi(close, 14)
        g["high10"] = high.rolling(10, min_periods=10).max()
        g["high20"] = high.rolling(20, min_periods=15).max()
        g["prev_high20"] = high.rolling(20, min_periods=15).max().shift(1)
        g["vol_avg20"] = volume.rolling(20, min_periods=10).mean().shift(1)
        g["vol_ratio"] = volume / g["vol_avg20"].replace(0.0, np.nan)
        g["turnover_med20"] = turnover.rolling(20, min_periods=10).median()
        g["day_pos"] = (close - low) / (high - low).replace(0.0, np.nan)

        ma_base = g["ma25"].where(g["ma25"].notna(), g["ma10"])
        ret_base = g["ret20"].where(g["ret20"].notna(), g["ret10"])
        high_base = g["high20"].where(g["high20"].notna(), g["high10"])
        g["ma_gap"] = close / ma_base - 1.0
        g["drawdown_high"] = close / high_base - 1.0

        signed_volume = (g["vol_ratio"] - 1.0) * np.sign(g["ret5"].fillna(0.0))
        sentiment = (
            0.40 * g["rsi14"]
            + 0.25 * _scale(ret_base, -0.20, 0.30)
            + 0.20 * _scale(g["ma_gap"], -0.10, 0.15)
            + 0.10 * _scale(close / high_base, 0.75, 1.00)
            + 0.05 * _scale(signed_volume, -1.5, 1.5)
        )
        g["sentiment"] = sentiment.clip(0.0, 100.0)

        long_trend = (
            (g["ma25"].notna() & g["ma75"].notna() & (close > g["ma25"]) & (g["ma25"] > g["ma75"]) & (g["ma25_slope5"] > 0))
            | (g["ma25"].notna() & g["ma75"].isna() & (close > g["ma25"]) & (g["ma25_slope5"] > 0))
        )
        ipo_trend = (
            (g["history_n"] >= 15)
            & (g["history_n"] < 60)
            & g["ma10"].notna()
            & (close > g["ma10"])
            & (g["ma10_slope5"] > 0)
            & (g["ret10"] > 0.05)
        )
        trend_ok = long_trend | ipo_trend

        breakout = (
            (close >= g["prev_high20"] * 0.98)
            & (g["ret5"] > 0)
            & (g["vol_ratio"] >= 1.0)
        )
        pullback_reclaim = (
            (g["drawdown_high"] <= -0.02)
            & (g["drawdown_high"] >= -0.10)
            & (close > close.shift(1))
            & (g["day_pos"] >= 0.60)
            & (g["vol_ratio"] >= 0.70)
        )
        ipo_breakout = (
            ipo_trend
            & (close / g["high10"] >= 0.94)
            & (g["vol_ratio"] >= 1.0)
        )

        liquid = g["turnover_med20"] >= LIQUIDITY_YEN
        blue = (
            liquid
            & trend_ok
            & g["sentiment"].between(60.0, 80.0)
            & g["rsi14"].between(52.0, 72.0)
            & g["ma_gap"].between(-0.01, 0.12)
            & (breakout | pullback_reclaim | ipo_breakout)
        )

        panic_reversal = (
            liquid
            & (g["sentiment"] <= 35.0)
            & (g["rsi14"] <= 38.0)
            & ((g["ret5"] <= -0.08) | (g["drawdown_high"] <= -0.15))
            & (g["vol_ratio"] >= 1.30)
            & (close > g["open"])
            & (close > close.shift(1))
            & (g["day_pos"] >= 0.65)
        )

        euphoria = (
            liquid
            & (g["sentiment"] >= 80.0)
            & (g["rsi14"] >= 72.0)
            & ((g["ma_gap"] >= 0.12) | (g["ret10"] >= 0.15))
            & ((g["vol_ratio"] >= 1.30) | (g["ret5"] >= 0.10))
            & (g["drawdown_high"] >= -0.04)
        )

        future_high = pd.concat([high.shift(-i) for i in range(1, HORIZON + 1)], axis=1).max(axis=1)
        future_low = pd.concat([low.shift(-i) for i in range(1, HORIZON + 1)], axis=1).min(axis=1)
        g["fwd10"] = close.shift(-HORIZON) / close - 1.0
        g["mfe10"] = future_high / close - 1.0
        g["mae10"] = future_low / close - 1.0
        g.loc[g["fwd10"].isna(), ["mfe10", "mae10"]] = np.nan
        g["period"] = g["date"].map(_period_label)

        code = str(stock_meta.get(ticker, {}).get("code") or ticker.removesuffix(".T"))[:4]
        is_training = code in TRAINING_CODES

        eligible = liquid & g["sentiment"].notna() & g["fwd10"].notna()
        base = g.loc[eligible, ["ticker", "date", "fwd10", "mfe10", "mae10", "sentiment", "period"]].copy()
        base["is_training"] = is_training
        baseline_rows.append(base)
        srows = base.copy()
        srows["bucket"] = srows["sentiment"].map(_sentiment_bucket)
        sentiment_rows.append(srows)

        for signal_name, mask in (
            ("BLUE_FOLLOW", blue),
            ("RED_CONTRARIAN_BUY", panic_reversal),
            ("RED_EUPHORIA_AVOID", euphoria),
        ):
            event_mask = _dedupe(mask)
            idxs = np.flatnonzero(event_mask)
            for i in idxs:
                row = g.iloc[i]
                signals.append(
                    {
                        "ticker": ticker,
                        "code": code,
                        "name": str(stock_meta.get(ticker, {}).get("name") or ticker),
                        "date": row["date"].strftime("%Y-%m-%d"),
                        "period": row["period"],
                        "signal": signal_name,
                        "is_training": is_training,
                        "close": float(row["close"]),
                        "sentiment": float(row["sentiment"]) if pd.notna(row["sentiment"]) else None,
                        "rsi14": float(row["rsi14"]) if pd.notna(row["rsi14"]) else None,
                        "ma_gap": float(row["ma_gap"]) if pd.notna(row["ma_gap"]) else None,
                        "ret10": float(row["ret10"]) if pd.notna(row["ret10"]) else None,
                        "vol_ratio": float(row["vol_ratio"]) if pd.notna(row["vol_ratio"]) else None,
                        "fwd10": float(row["fwd10"]) if pd.notna(row["fwd10"]) else None,
                        "mfe10": float(row["mfe10"]) if pd.notna(row["mfe10"]) else None,
                        "mae10": float(row["mae10"]) if pd.notna(row["mae10"]) else None,
                    }
                )

    baseline_df = pd.concat(baseline_rows, ignore_index=True)
    sentiment_df = pd.concat(sentiment_rows, ignore_index=True)
    signal_df = pd.DataFrame(signals)
    if signal_df.empty:
        signal_df = pd.DataFrame(columns=["signal", "is_training", "period", "fwd10", "mfe10", "mae10"])

    holdout_baseline = baseline_df.loc[~baseline_df["is_training"]]
    baseline_mean = float(holdout_baseline["fwd10"].mean())

    summary: dict[str, Any] = {
        "meta": {
            "data_start": manifest["meta"]["startDate"],
            "data_end": manifest["meta"]["endDate"],
            "stock_count": manifest["meta"]["stockCount"],
            "date_count": manifest["meta"]["dateCount"],
            "training_codes": sorted(TRAINING_CODES),
            "liquidity_filter_yen_20d_median": LIQUIDITY_YEN,
            "horizon_days": HORIZON,
            "cooldown_days": COOLDOWN,
            "optimization": "none; thresholds fixed before all-TSE evaluation",
        },
        "rules": {
            "sentiment_score": "40% RSI14 + 25% 20d/10d return + 20% MA gap + 10% distance-to-high + 5% signed volume acceleration",
            "BLUE_FOLLOW": "sentiment 60-80, positive trend, RSI 52-72, MA gap -1%..+12%, breakout or first-pullback reclaim, liquidity >=5e8 yen",
            "RED_CONTRARIAN_BUY": "sentiment <=35, RSI<=38, recent selloff, volume ratio>=1.3, bullish close/reversal confirmation, liquidity >=5e8 yen",
            "RED_EUPHORIA_AVOID": "sentiment>=80, RSI>=72, stretched MA gap or 10d return, volume/price acceleration, within 4% of recent high",
        },
        "holdout_ex_training": {},
        "periods_holdout": {},
        "sentiment_buckets_holdout": {},
        "training_audit": {},
    }

    summary["holdout_ex_training"]["BASELINE_ELIGIBLE"] = _metric_summary(holdout_baseline)
    for signal_name in ["BLUE_FOLLOW", "RED_CONTRARIAN_BUY", "RED_EUPHORIA_AVOID"]:
        rows = signal_df.loc[(signal_df["signal"] == signal_name) & (~signal_df["is_training"])]
        summary["holdout_ex_training"][signal_name] = _metric_summary(rows, baseline_mean)

    for period in sorted(holdout_baseline["period"].unique()):
        pbase = holdout_baseline.loc[holdout_baseline["period"] == period]
        pmean = float(pbase["fwd10"].mean()) if not pbase.empty else 0.0
        summary["periods_holdout"][period] = {"BASELINE_ELIGIBLE": _metric_summary(pbase)}
        for signal_name in ["BLUE_FOLLOW", "RED_CONTRARIAN_BUY", "RED_EUPHORIA_AVOID"]:
            rows = signal_df.loc[
                (signal_df["signal"] == signal_name)
                & (~signal_df["is_training"])
                & (signal_df["period"] == period)
            ]
            summary["periods_holdout"][period][signal_name] = _metric_summary(rows, pmean)

    for bucket in ["0-20 極端な悲観", "20-40 悲観", "40-60 中立", "60-80 楽観", "80-100 熱狂"]:
        rows = sentiment_df.loc[(~sentiment_df["is_training"]) & (sentiment_df["bucket"] == bucket)]
        summary["sentiment_buckets_holdout"][bucket] = _metric_summary(rows, baseline_mean)

    training_signal_df = signal_df.loc[signal_df["is_training"]].copy()
    for code in sorted(TRAINING_CODES):
        rows = training_signal_df.loc[training_signal_df["code"] == code].copy()
        summary["training_audit"][code] = {
            "signal_count": int(len(rows)),
            "signals_with_full_10d_outcome": int(rows["fwd10"].notna().sum()) if not rows.empty else 0,
            "by_signal": {
                s: _metric_summary(rows.loc[rows["signal"] == s], baseline_mean)
                for s in ["BLUE_FOLLOW", "RED_CONTRARIAN_BUY", "RED_EUPHORIA_AVOID"]
            },
            "sample_events": rows.sort_values("date").tail(8).to_dict(orient="records") if not rows.empty else [],
        }

    return summary


def render_markdown(summary: dict[str, Any]) -> str:
    meta = summary["meta"]
    lines = [
        "# Crowd Psychology 3Y Backtest",
        "",
        f"- Data: {meta['data_start']} to {meta['data_end']}",
        f"- Universe: all TSE, {meta['stock_count']} stocks, {meta['date_count']} market days",
        f"- Training examples: {', '.join(meta['training_codes'])}",
        "- Primary evaluation: all TSE excluding the 11 training stocks",
        "- No threshold grid-search or parameter optimization was performed",
        "- Forward horizon: 10 trading days",
        "- Liquidity filter: 20-day median turnover >= 500 million yen",
        "",
        "## Initial rules",
        "",
        f"- Sentiment score: {summary['rules']['sentiment_score']}",
        f"- BLUE_FOLLOW: {summary['rules']['BLUE_FOLLOW']}",
        f"- RED_CONTRARIAN_BUY: {summary['rules']['RED_CONTRARIAN_BUY']}",
        f"- RED_EUPHORIA_AVOID: {summary['rules']['RED_EUPHORIA_AVOID']}",
        "",
        "## Holdout results excluding training stocks",
        "",
        "| Class | N | Mean 10d | Median 10d | Win | +5% MFE | -5% MAE | Avg MFE | Avg MAE | Edge vs baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for key in ["BASELINE_ELIGIBLE", "BLUE_FOLLOW", "RED_CONTRARIAN_BUY", "RED_EUPHORIA_AVOID"]:
        s = summary["holdout_ex_training"][key]
        lines.append(
            f"| {key} | {s['count']} | {_fmt_pct(s['mean_fwd10'])} | {_fmt_pct(s['median_fwd10'])} | {_fmt_pct(s['win_rate'])} | {_fmt_pct(s['plus5_rate'])} | {_fmt_pct(s['minus5_rate'])} | {_fmt_pct(s['avg_mfe10'])} | {_fmt_pct(s['avg_mae10'])} | {_fmt_pct(s['edge_vs_baseline'])} |"
        )

    lines += ["", "## Sentiment buckets, holdout", "", "| Sentiment | N | Mean 10d | Win | +5% MFE | -5% MAE | Edge |", "|---|---:|---:|---:|---:|---:|---:|"]
    for bucket, s in summary["sentiment_buckets_holdout"].items():
        lines.append(
            f"| {bucket} | {s['count']} | {_fmt_pct(s['mean_fwd10'])} | {_fmt_pct(s['win_rate'])} | {_fmt_pct(s['plus5_rate'])} | {_fmt_pct(s['minus5_rate'])} | {_fmt_pct(s['edge_vs_baseline'])} |"
        )

    lines += ["", "## Period stability, holdout", ""]
    for period, pdata in summary["periods_holdout"].items():
        lines += [f"### {period}", "", "| Class | N | Mean 10d | Win | +5% MFE | -5% MAE | Edge |", "|---|---:|---:|---:|---:|---:|---:|"]
        for key in ["BASELINE_ELIGIBLE", "BLUE_FOLLOW", "RED_CONTRARIAN_BUY", "RED_EUPHORIA_AVOID"]:
            s = pdata[key]
            lines.append(
                f"| {key} | {s['count']} | {_fmt_pct(s['mean_fwd10'])} | {_fmt_pct(s['win_rate'])} | {_fmt_pct(s['plus5_rate'])} | {_fmt_pct(s['minus5_rate'])} | {_fmt_pct(s['edge_vs_baseline'])} |"
            )
        lines.append("")

    lines += ["## Training-stock audit", "", "| Code | Signals | Full 10d outcomes |", "|---|---:|---:|"]
    for code, audit in summary["training_audit"].items():
        lines.append(f"| {code} | {audit['signal_count']} | {audit['signals_with_full_10d_outcome']} |")

    lines += [
        "",
        "## Important caveats",
        "",
        "- The 11 stocks were selected because they are visually instructive; that creates selection bias. They are excluded from the primary holdout metrics.",
        "- This is a technical sentiment proxy, not direct Yahoo-board/SNS natural-language sentiment yet.",
        "- The 3-year stored dataset ends at 2026-08-21, so later September examples are not part of this backtest.",
        "- Because the rule ideas were motivated by known historical examples, this is research backtest evidence, not a clean time-OOS proof.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dashboard-data/technical-backtest-3y"))
    parser.add_argument("--output-json", type=Path, default=Path("crowd-psychology-backtest.json"))
    parser.add_argument("--output-md", type=Path, default=Path("crowd-psychology-backtest.md"))
    args = parser.parse_args()

    summary = run_backtest(args.data_dir)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md = render_markdown(summary)
    args.output_md.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
