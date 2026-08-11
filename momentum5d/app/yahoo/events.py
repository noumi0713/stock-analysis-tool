from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from app.config import Settings
from app.storage.parquet import ParquetStore
from app.yahoo.ingestion import YahooPaths

LOGGER = logging.getLogger(__name__)

EARNINGS_HOLDING_DAYS = 7
GAP_DOWN_THRESHOLD = -0.03
GAP_DOWN_FLOOR = -0.12
EVENT_POSITION_SCALE = 0.50
EARNINGS_COLUMNS = [
    "ticker",
    "code",
    "earnings_date",
    "announcement_time",
    "source",
    "confirmed",
    "fetched_at",
]
MACRO_COLUMNS = [
    "event_date",
    "event_name",
    "event_type",
    "severity",
    "position_scale",
    "block_new_entries",
    "source",
]

CalendarFetcher = Callable[[str], Any]


@dataclass(frozen=True, slots=True)
class EventCalendarPaths:
    earnings: Path
    macro: Path


def calendar_paths(settings: Settings) -> EventCalendarPaths:
    yahoo_paths = YahooPaths(settings.data_dir / "yahoo")
    return EventCalendarPaths(
        earnings=yahoo_paths.metadata_dir / "earnings_calendar.parquet",
        macro=settings.data_dir.parent / "config" / "important_events.csv",
    )


class EarningsCalendarUpdater:
    """Refresh near-term earnings dates only for the current ranking pool."""

    def __init__(
        self,
        settings: Settings,
        *,
        fetcher: CalendarFetcher | None = None,
        max_workers: int = 6,
    ) -> None:
        self.settings = settings
        self.paths = YahooPaths(settings.data_dir / "yahoo")
        self.paths.ensure()
        self.output = calendar_paths(settings).earnings
        self.fetcher = fetcher or self._fetch_yahoo_calendar
        self.max_workers = max_workers

    def update(
        self,
        *,
        as_of: date,
        candidate_limit: int = 100,
        tickers: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        selected = list(dict.fromkeys(tickers or self._candidate_tickers(candidate_limit)))
        if not selected:
            raise RuntimeError("決算予定を取得する候補銘柄がありません")
        rows: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        fetched_at = datetime.now(UTC).isoformat()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.fetcher, ticker): ticker for ticker in selected}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    dates = _extract_earnings_dates(future.result(), as_of)
                    for earnings_date in dates:
                        rows.append(
                            {
                                "ticker": ticker,
                                "code": ticker.removesuffix(".T"),
                                "earnings_date": earnings_date,
                                "announcement_time": "unknown",
                                "source": "yfinance",
                                "confirmed": False,
                                "fetched_at": fetched_at,
                            }
                        )
                    if not dates:
                        failures.append({"ticker": ticker, "error": "future date unavailable"})
                except Exception as exc:
                    failures.append({"ticker": ticker, "error": str(exc)[:300]})
                    LOGGER.warning("earnings calendar failed ticker=%s error=%s", ticker, exc)

        existing = _read_earnings(self.output)
        incoming = pd.DataFrame(rows, columns=EARNINGS_COLUMNS)
        retained = existing.loc[~existing["ticker"].astype(str).isin(selected)].copy()
        combined = pd.concat([retained, incoming], ignore_index=True, sort=False)
        if not combined.empty:
            combined["earnings_date"] = pd.to_datetime(
                combined["earnings_date"], errors="coerce"
            ).dt.date
            combined = (
                combined.dropna(subset=["ticker", "earnings_date"])
                .drop_duplicates(["ticker", "earnings_date"], keep="last")
                .sort_values(["earnings_date", "ticker"])
                .reset_index(drop=True)
            )
        ParquetStore._atomic_parquet(combined, self.output)
        return {
            "source": "yfinance",
            "as_of": as_of.isoformat(),
            "requested_tickers": len(selected),
            "covered_tickers": int(incoming["ticker"].nunique()) if not incoming.empty else 0,
            "earnings_dates": len(incoming),
            "failures": failures,
            "output": str(self.output),
        }

    def _candidate_tickers(self, limit: int) -> list[str]:
        scores_path = self.paths.processed_dir / "analysis" / "latest_scores.parquet"
        candidates_path = self.paths.processed_dir / "analysis" / "latest_candidates.parquet"
        path = scores_path if scores_path.exists() else candidates_path
        if not path.exists():
            return []
        frame = pd.read_parquet(path)
        sort_columns = [
            column
            for column in ("monthly_target_signal", "ml_sharp_signal", "signal_score")
            if column in frame
        ]
        if sort_columns:
            frame = frame.sort_values(sort_columns, ascending=[False] * len(sort_columns))
        return frame.head(limit)["ticker"].astype(str).tolist()

    @staticmethod
    def _fetch_yahoo_calendar(ticker: str) -> Any:
        return yf.Ticker(ticker).calendar


def load_event_calendars(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = calendar_paths(settings)
    earnings = _read_earnings(paths.earnings)
    macro = _read_macro(paths.macro)
    return earnings, macro


def add_event_risk_controls(
    features: pd.DataFrame,
    earnings: pd.DataFrame,
    macro_events: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach latest-day event guards without changing historical model features."""
    frame = features.copy()
    defaults: dict[str, Any] = {
        "earnings_calendar_covered": False,
        "next_earnings_date": None,
        "earnings_days_ahead": pd.NA,
        "earnings_crossing_risk": False,
        "earnings_exit_date": None,
        "important_event_nearby": False,
        "important_event_name": "",
        "event_position_scale": 1.0,
        "event_entry_allowed": False,
        "event_trade_action": "CHECK_EARNINGS_CALENDAR",
        "event_risk_reason": "決算予定未取得のため新規買い不可",
        "earnings_gap_down": np.nan,
        "earnings_gd_reversal_signal": False,
        "earnings_gd_entry_price": np.nan,
        "earnings_gd_stop_price": np.nan,
        "earnings_gd_take_profit": np.nan,
        "earnings_gd_reason": "",
    }
    for column, value in defaults.items():
        frame[column] = value
    if frame.empty:
        return frame, _event_summary(frame, earnings, macro_events)

    latest_date = pd.Timestamp(frame["date"].max()).date()
    latest_mask = pd.to_datetime(frame["date"]).dt.date.eq(latest_date)
    latest = frame.loc[latest_mask].copy()
    previous_close = frame.groupby("ticker", sort=False)["adjusted_close"].shift(1)
    latest["_previous_close"] = previous_close.loc[latest.index]
    upcoming_macro = _nearby_macro_events(macro_events, latest_date)
    macro_nearby = not upcoming_macro.empty
    macro_name = "・".join(upcoming_macro["event_name"].astype(str).tolist())
    macro_scale = (
        float(upcoming_macro["position_scale"].min()) if macro_nearby else 1.0
    )
    macro_blocks = bool(upcoming_macro["block_new_entries"].any()) if macro_nearby else False

    earnings_work = earnings.copy()
    for column in EARNINGS_COLUMNS:
        if column not in earnings_work:
            earnings_work[column] = pd.NA
    if not earnings_work.empty:
        earnings_work["code"] = earnings_work["code"].astype(str)
        earnings_work["earnings_date"] = pd.to_datetime(
            earnings_work["earnings_date"], errors="coerce"
        ).dt.date

    for index, row in latest.iterrows():
        code = str(row["code"])
        company_events = earnings_work.loc[earnings_work["code"] == code]
        covered = not company_events.empty
        future = company_events.loc[company_events["earnings_date"] >= latest_date]
        next_date = min(future["earnings_date"]) if not future.empty else None
        days_ahead = _business_days(latest_date, next_date) if next_date else None
        crossing = days_ahead is not None and 0 <= days_ahead <= EARNINGS_HOLDING_DAYS
        reaction_today = bool((company_events["earnings_date"] == latest_date).any())
        previous = _number(row.get("_previous_close"))
        gd = _number(row.get("open")) / previous - 1.0 if previous > 0 else np.nan
        gd_signal, gd_values = _earnings_gap_down_setup(row, gd, reaction_today)

        entry_allowed = covered and not crossing and not macro_blocks
        action = "NORMAL"
        reasons: list[str] = []
        if not covered:
            action = "CHECK_EARNINGS_CALENDAR"
            reasons.append("決算予定未取得のため新規買い不可")
        elif crossing and not gd_signal:
            action = "NO_TRADE_EARNINGS"
            reasons.append(f"{next_date}の決算を保有期間中に跨ぐため新規買い不可")
        if macro_nearby:
            if macro_blocks and action == "NORMAL":
                action = "WAIT_MACRO_EVENT"
            elif action == "NORMAL":
                action = "REDUCE_50_PERCENT"
            reasons.append(f"重要指標前のため保有量を{macro_scale:.0%}へ縮小")
        if gd_signal:
            entry_allowed = not macro_blocks
            action = "BUY_GD_REVERSAL" if entry_allowed else "WAIT_MACRO_EVENT"
            reasons.append("決算反応のGD後に陽線反転を確認。翌日高値超えでのみ買い検討")

        values = {
            "earnings_calendar_covered": covered,
            "next_earnings_date": next_date,
            "earnings_days_ahead": days_ahead,
            "earnings_crossing_risk": crossing,
            "earnings_exit_date": (
                _previous_business_day(next_date)
                if crossing and days_ahead is not None and days_ahead >= 1
                else None
            ),
            "important_event_nearby": macro_nearby,
            "important_event_name": macro_name,
            "event_position_scale": (
                macro_scale
                if gd_signal
                else min(macro_scale, EVENT_POSITION_SCALE)
                if crossing
                else macro_scale
            ),
            "event_entry_allowed": entry_allowed,
            "event_trade_action": action,
            "event_risk_reason": "・".join(reasons) or "イベント制約なし",
            "earnings_gap_down": gd if np.isfinite(gd) else np.nan,
            "earnings_gd_reversal_signal": gd_signal,
            **gd_values,
        }
        for column, value in values.items():
            frame.at[index, column] = value

    frame["earnings_days_ahead"] = pd.to_numeric(
        frame["earnings_days_ahead"], errors="coerce"
    ).astype("Int64")
    return frame, _event_summary(frame.loc[latest_mask], earnings, macro_events)


def _earnings_gap_down_setup(
    row: pd.Series, gap_down: float, reaction_today: bool
) -> tuple[bool, dict[str, float | str]]:
    entry = _number(row.get("high")) * 1.002
    low = _number(row.get("low"))
    close = _number(row.get("close"))
    open_ = _number(row.get("open"))
    span = _number(row.get("high")) - low
    close_location = (close - low) / span if span > 0 else np.nan
    volume_ratio = _number(row.get("volume_ratio_1_20"))
    return_20d = _number(row.get("return_20d"))
    close_to_ma20 = _number(row.get("close_to_ma20"))
    stop = max(low * 0.995, entry * 0.97)
    stop_rate = entry / stop - 1.0 if stop > 0 else np.inf
    signal = bool(
        reaction_today
        and GAP_DOWN_FLOOR <= gap_down <= GAP_DOWN_THRESHOLD
        and close > open_
        and close_location >= 0.65
        and volume_ratio >= 1.50
        and _number(row.get("turnover_value")) >= 200_000_000
        and return_20d >= -0.05
        and close_to_ma20 >= -0.03
        and stop_rate <= 0.031
    )
    reason = (
        "GD-3〜-12%、売買代金2億円以上、出来高1.5倍、陽線かつ上ヒゲ過小"
        if signal
        else "GD後の反転・流動性・上昇基調のいずれかが未確認"
    )
    return signal, {
        "earnings_gd_entry_price": entry if signal else np.nan,
        "earnings_gd_stop_price": stop if signal else np.nan,
        "earnings_gd_take_profit": entry * 1.05 if signal else np.nan,
        "earnings_gd_reason": reason,
    }


def _nearby_macro_events(events: pd.DataFrame, as_of: date) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=MACRO_COLUMNS)
    work = events.copy()
    work["event_date"] = pd.to_datetime(work["event_date"], errors="coerce").dt.date
    days = pd.to_numeric(
        work["event_date"].map(lambda value: _business_days(as_of, value)),
        errors="coerce",
    )
    return work.loc[
        days.between(0, 1, inclusive="both") & work["severity"].eq("high")
    ]


def _read_earnings(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=EARNINGS_COLUMNS)
    frame = pd.read_parquet(path)
    for column in EARNINGS_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[EARNINGS_COLUMNS]


def _read_macro(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=MACRO_COLUMNS)
    frame = pd.read_csv(path)
    missing = set(MACRO_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"重要指標カレンダーの必須列がありません: {sorted(missing)}")
    frame["position_scale"] = pd.to_numeric(frame["position_scale"], errors="coerce").fillna(0.5)
    frame["block_new_entries"] = frame["block_new_entries"].map(_as_bool)
    return frame[MACRO_COLUMNS]


def _extract_earnings_dates(value: Any, as_of: date) -> list[date]:
    raw: Any = None
    if isinstance(value, dict):
        raw = value.get("Earnings Date") or value.get("EarningsDate")
    elif isinstance(value, pd.DataFrame):
        for key in ("Earnings Date", "EarningsDate"):
            if key in value.index:
                raw = value.loc[key].tolist()
                break
            if key in value.columns:
                raw = value[key].tolist()
                break
    if raw is None:
        return []
    values = raw if isinstance(raw, (list, tuple, pd.Series, np.ndarray)) else [raw]
    dates: set[date] = set()
    for item in values:
        if pd.isna(item):
            continue
        timestamp = pd.to_datetime(item, errors="coerce")
        if pd.isna(timestamp):
            continue
        candidate = timestamp.date()
        if candidate >= as_of:
            dates.add(candidate)
    return sorted(dates)


def _event_summary(
    latest: pd.DataFrame, earnings: pd.DataFrame, macro_events: pd.DataFrame
) -> dict[str, Any]:
    return {
        "earnings_calendar_rows": len(earnings),
        "important_event_rows": len(macro_events),
        "covered_latest_tickers": int(
            latest.get("earnings_calendar_covered", pd.Series(dtype=bool)).sum()
        ),
        "earnings_crossing_risks": int(
            latest.get("earnings_crossing_risk", pd.Series(dtype=bool)).sum()
        ),
        "gd_reversal_signals": int(
            latest.get("earnings_gd_reversal_signal", pd.Series(dtype=bool)).sum()
        ),
        "policy": {
            "earnings": "保有予定7営業日以内に決算があれば新規買い不可。保有は前営業日までに解消",
            "macro": "重要指標の1営業日前から保有量50%。指標によっては新規買い停止",
            "gap_down": "決算反応日に-3〜-12%GD後、陽線・出来高・トレンド確認後の翌日高値超えのみ",
            "missing_data": "決算予定を取得できない銘柄は自動売買不可・手動確認必須",
        },
    }


def _business_days(start: date, end: date | None) -> int | None:
    if end is None:
        return None
    if end < start:
        return -int(np.busday_count(end, start))
    return int(np.busday_count(start, end))


def _previous_business_day(value: date | None) -> date | None:
    if value is None:
        return None
    return (pd.Timestamp(value) - pd.offsets.BDay(1)).date()


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
