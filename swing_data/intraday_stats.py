from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MORNING_MAX_MINUTE = 150
BUCKET_MINUTES = 5


def _clock_label(minutes_from_open: int) -> str:
    total_minutes = 9 * 60 + minutes_from_open
    hour, minute = divmod(total_minutes, 60)
    return f"{hour:02d}:{minute:02d}"


def _prepare_lows(lows: pd.DataFrame) -> pd.DataFrame:
    required = {"Ticker", "MinutesFromOpen"}
    missing = sorted(required - set(lows.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    prepared = lows.copy()
    prepared["MinutesFromOpen"] = pd.to_numeric(
        prepared["MinutesFromOpen"], errors="coerce"
    )
    prepared = prepared.dropna(subset=["Ticker", "MinutesFromOpen"])
    prepared = prepared[
        prepared["MinutesFromOpen"].between(0, MORNING_MAX_MINUTE, inclusive="both")
    ].copy()
    prepared["MinutesFromOpen"] = prepared["MinutesFromOpen"].astype(int)
    prepared["Ticker"] = prepared["Ticker"].astype(str)
    return prepared


def _iter_scopes(lows: pd.DataFrame):
    yield "ALL", "ALL", lows
    for ticker, group in lows.groupby("Ticker", sort=True):
        yield "TICKER", str(ticker), group


def summarize_morning_lows(lows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build overall/per-ticker statistics and a 5-minute low-time distribution.

    One observation is one ticker x trading-date row from morning_low_summary.csv.
    The ALL scope therefore weights each ticker-date observation equally.
    """
    prepared = _prepare_lows(lows)
    if prepared.empty:
        return pd.DataFrame(), pd.DataFrame()

    stats_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []

    for scope, ticker, group in _iter_scopes(prepared):
        values = group["MinutesFromOpen"]
        observations = int(len(values))

        stats_rows.append(
            {
                "Scope": scope,
                "Ticker": ticker,
                "Observations": observations,
                "MeanMinutesFromOpen": round(float(values.mean()), 2),
                "MedianMinutesFromOpen": round(float(values.median()), 2),
                "MinMinutesFromOpen": int(values.min()),
                "MaxMinutesFromOpen": int(values.max()),
            }
        )

        for start in range(0, MORNING_MAX_MINUTE + 1, BUCKET_MINUTES):
            end = min(start + BUCKET_MINUTES - 1, MORNING_MAX_MINUTE)
            count = int(values.between(start, end, inclusive="both").sum())
            percentage = (count / observations * 100.0) if observations else 0.0
            minute_label = f"{start}-{end}" if start != end else str(start)
            time_label = (
                f"{_clock_label(start)}-{_clock_label(end)}"
                if start != end
                else _clock_label(start)
            )
            distribution_rows.append(
                {
                    "Scope": scope,
                    "Ticker": ticker,
                    "BucketStartMinute": start,
                    "BucketEndMinute": end,
                    "MinutesFromOpenRange": minute_label,
                    "ClockRangeJST": time_label,
                    "Count": count,
                    "Percentage": round(percentage, 2),
                }
            )

    return pd.DataFrame(stats_rows), pd.DataFrame(distribution_rows)


def write_statistics(input_csv: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not input_csv.exists():
        raise FileNotFoundError(f"input file not found: {input_csv}")

    lows = pd.read_csv(input_csv)
    statistics, distribution = summarize_morning_lows(lows)
    if statistics.empty:
        raise RuntimeError("no valid morning-low observations found")

    output_dir.mkdir(parents=True, exist_ok=True)
    statistics.to_csv(
        output_dir / "morning_low_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    distribution.to_csv(
        output_dir / "morning_low_5min_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return statistics, distribution


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate morning-low timing statistics from 1-minute data outputs."
    )
    parser.add_argument(
        "--input",
        default="intraday_1m_output/morning_low_summary.csv",
        help="Path to morning_low_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="intraday_1m_output",
        help="Directory for statistics CSV outputs",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    statistics, _ = write_statistics(Path(args.input), Path(args.output_dir))

    overall = statistics[statistics["Scope"] == "ALL"].iloc[0]
    print(
        "morning-low stats: "
        f"observations={int(overall['Observations'])} "
        f"mean={overall['MeanMinutesFromOpen']}min "
        f"median={overall['MedianMinutesFromOpen']}min"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
