from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

JST = "Asia/Tokyo"
MAX_DAYS = 7


@dataclass
class FetchResult:
    requested: str
    symbol: str
    rows: int = 0
    trading_dates: int = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    csv_path: str | None = None
    error: str | None = None


def normalize_symbol(raw: str) -> str:
    """Normalize a user-entered ticker for Yahoo Finance.

    Examples:
      7203   -> 7203.T
      130A   -> 130A.T
      7203.T -> 7203.T
      AAPL   -> AAPL

    This tool is intended primarily for Tokyo-listed stocks. Bare Japanese-style
    codes are automatically suffixed with .T. Other symbols are left unchanged.
    """
    value = raw.strip().upper()
    if not value:
        raise ValueError("empty ticker")

    if "." in value:
        return value

    if re.fullmatch(r"[0-9]{4}", value) or re.fullmatch(r"[0-9]{3}[A-Z]", value):
        return f"{value}.T"

    return value


def parse_tickers(values: Iterable[str]) -> list[tuple[str, str]]:
    tokens: list[str] = []
    for value in values:
        tokens.extend(x for x in re.split(r"[\s,;]+", value.strip()) if x)

    seen: set[str] = set()
    parsed: list[tuple[str, str]] = []
    for token in tokens:
        symbol = normalize_symbol(token)
        if symbol in seen:
            continue
        seen.add(symbol)
        parsed.append((token, symbol))

    if not parsed:
        raise ValueError("at least one ticker is required")
    return parsed


def _to_jst_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    result = df.copy()
    idx = pd.DatetimeIndex(result.index)
    if idx.tz is None:
        idx = idx.tz_localize(JST)
    else:
        idx = idx.tz_convert(JST)
    result.index = idx
    result.index.name = "Datetime"
    return result


def _trim_calendar_lookback(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Keep at most the requested number of JST calendar dates, including today."""
    if df.empty:
        return df
    now_jst = pd.Timestamp.now(tz=JST)
    first_date = (now_jst - pd.Timedelta(days=days - 1)).normalize()
    return df.loc[df.index >= first_date]


def filter_session(df: pd.DataFrame, session: str) -> pd.DataFrame:
    """Keep Tokyo Stock Exchange regular-session rows.

    `morning` keeps 09:00-11:30 JST.
    `full` keeps 09:00-11:30 and 12:30-15:30 JST.
    """
    if df.empty:
        return df

    morning = df.between_time("09:00", "11:30", inclusive="both")
    if session == "morning":
        return morning
    if session != "full":
        raise ValueError(f"unsupported session: {session}")

    afternoon = df.between_time("12:30", "15:30", inclusive="both")
    return pd.concat([morning, afternoon]).sort_index()


def fetch_symbol(symbol: str, days: int, session: str, retries: int = 3) -> pd.DataFrame:
    """Fetch the most recent 1-minute bars available from yfinance.

    yfinance treats 1-minute data specially: `period="max"` is internally limited
    to roughly the last 8 days. We request that bounded maximum and then trim it
    to the user's 1-7 calendar-day lookback. This avoids relying on unsupported
    custom period strings such as `7d`.
    """
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            df = yf.Ticker(symbol).history(
                period="max",
                interval="1m",
                auto_adjust=False,
                prepost=False,
                actions=False,
                repair=False,
                raise_errors=True,
            )
            if df is None or df.empty:
                raise RuntimeError("Yahoo Finance returned no 1-minute rows")

            df = _to_jst_index(df)
            wanted = [
                c
                for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
                if c in df.columns
            ]
            df = df[wanted]
            df = _trim_calendar_lookback(df, days)
            df = filter_session(df, session)
            df = df[~df.index.duplicated(keep="last")].sort_index()
            if df.empty:
                raise RuntimeError("no rows remained after lookback/session filtering")
            return df
        except Exception as exc:  # yfinance raises several backend-specific exception types
            last_error = exc
            if attempt < retries:
                time.sleep(attempt * 2)

    raise RuntimeError(str(last_error) if last_error else "unknown yfinance error")


def add_export_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = df.copy().reset_index()
    out.insert(1, "Ticker", symbol)
    out.insert(2, "Date", out["Datetime"].dt.strftime("%Y-%m-%d"))
    out.insert(3, "Time", out["Datetime"].dt.strftime("%H:%M"))
    out["Datetime"] = out["Datetime"].astype(str)
    return out


def morning_low_summary(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return one row per trading day with the first timestamp of the morning low."""
    morning = df.between_time("09:00", "11:30", inclusive="both").copy()
    if morning.empty:
        return pd.DataFrame()

    morning["TradingDate"] = morning.index.strftime("%Y-%m-%d")
    rows: list[dict[str, object]] = []

    for trading_date, group in morning.groupby("TradingDate", sort=True):
        lows = pd.to_numeric(group["Low"], errors="coerce")
        low_price = lows.min()
        hits = group[lows == low_price]
        if hits.empty or pd.isna(low_price):
            continue

        low_ts = hits.index[0]
        open_price = float(group.iloc[0]["Open"])
        close_price = float(group.iloc[-1]["Close"])
        minutes_from_open = int((low_ts.hour * 60 + low_ts.minute) - 9 * 60)
        rows.append(
            {
                "Ticker": symbol,
                "Date": trading_date,
                "MorningLow": float(low_price),
                "MorningLowTime": low_ts.strftime("%H:%M"),
                "MinutesFromOpen": minutes_from_open,
                "Open": open_price,
                "MorningClose": close_price,
                "OpenToLowPct": (float(low_price) / open_price - 1.0) * 100 if open_price else None,
                "LowToMorningClosePct": (close_price / float(low_price) - 1.0) * 100 if low_price else None,
            }
        )

    return pd.DataFrame(rows)


def write_outputs(
    parsed_tickers: list[tuple[str, str]],
    days: int,
    session: str,
    output_dir: Path,
) -> tuple[list[FetchResult], pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[FetchResult] = []
    combined_frames: list[pd.DataFrame] = []
    low_frames: list[pd.DataFrame] = []

    for requested, symbol in parsed_tickers:
        result = FetchResult(requested=requested, symbol=symbol)
        try:
            raw = fetch_symbol(symbol=symbol, days=days, session=session)
            exported = add_export_columns(raw, symbol)
            safe_name = symbol.replace(".", "_")
            csv_path = output_dir / f"{safe_name}_1m_{days}cald.csv"
            exported.to_csv(csv_path, index=False, encoding="utf-8-sig")

            result.rows = len(raw)
            result.trading_dates = raw.index.normalize().nunique()
            result.first_timestamp = raw.index.min().isoformat()
            result.last_timestamp = raw.index.max().isoformat()
            result.csv_path = str(csv_path)
            combined_frames.append(exported)

            low_summary = morning_low_summary(raw, symbol)
            if not low_summary.empty:
                low_frames.append(low_summary)
        except Exception as exc:
            result.error = str(exc)
        results.append(result)

    combined = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame()
    if not combined.empty:
        combined.to_csv(
            output_dir / "selected_tickers_1m_combined.csv",
            index=False,
            encoding="utf-8-sig",
        )

    lows = pd.concat(low_frames, ignore_index=True) if low_frames else pd.DataFrame()
    if not lows.empty:
        lows.to_csv(output_dir / "morning_low_summary.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "generated_at_jst": datetime.now(ZoneInfo(JST)).isoformat(),
        "interval": "1m",
        "calendar_days_requested": days,
        "session": session,
        "requested_count": len(parsed_tickers),
        "success_count": sum(1 for r in results if r.error is None),
        "failure_count": sum(1 for r in results if r.error is not None),
        "results": [asdict(r) for r in results],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return results, lows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch Yahoo Finance 1-minute bars only for explicitly specified tickers."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help="Ticker codes separated by spaces or commas, e.g. 7203 6758 130A",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        choices=range(1, MAX_DAYS + 1),
        metavar="1-7",
        help="JST calendar-day lookback, including today (default: 7)",
    )
    parser.add_argument(
        "--session",
        choices=["morning", "full"],
        default="morning",
        help="morning=09:00-11:30 JST, full=09:00-11:30 + 12:30-15:30 JST",
    )
    parser.add_argument(
        "--output-dir",
        default="intraday_1m_output",
        help="Directory to write CSV/JSON outputs",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        parsed = parse_tickers(args.tickers)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    results, lows = write_outputs(
        parsed_tickers=parsed,
        days=args.days,
        session=args.session,
        output_dir=Path(args.output_dir),
    )

    for result in results:
        if result.error:
            print(f"[FAIL] {result.requested} -> {result.symbol}: {result.error}")
        else:
            print(
                f"[OK] {result.requested} -> {result.symbol}: "
                f"{result.rows} rows / {result.trading_dates} trading dates "
                f"({result.first_timestamp} .. {result.last_timestamp})"
            )

    success_count = sum(1 for r in results if r.error is None)
    print(f"success={success_count}/{len(results)} morning_low_rows={len(lows)}")
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
