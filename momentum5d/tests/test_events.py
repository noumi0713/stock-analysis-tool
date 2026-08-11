from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from app.config import Settings
from app.yahoo.events import (
    EarningsCalendarUpdater,
    add_event_risk_controls,
    calendar_paths,
)


def _features(
    *,
    latest_open: float = 101.0,
    latest_high: float = 103.0,
    latest_low: float = 100.0,
    latest_close: float = 102.0,
    turnover: float = 300_000_000,
    volume_ratio: float = 2.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": date(2026, 8, 10),
                "ticker": "1111.T",
                "code": "1111",
                "open": 99.0,
                "high": 101.0,
                "low": 98.0,
                "close": 100.0,
                "adjusted_close": 100.0,
                "turnover_value": 250_000_000,
                "volume_ratio_1_20": 1.0,
                "return_20d": 0.01,
                "close_to_ma20": 0.01,
            },
            {
                "date": date(2026, 8, 11),
                "ticker": "1111.T",
                "code": "1111",
                "open": latest_open,
                "high": latest_high,
                "low": latest_low,
                "close": latest_close,
                "adjusted_close": latest_close,
                "turnover_value": turnover,
                "volume_ratio_1_20": volume_ratio,
                "return_20d": 0.01,
                "close_to_ma20": 0.01,
            },
        ]
    )


def _earnings(event_date: date) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["1111.T"],
            "code": ["1111"],
            "earnings_date": [event_date],
            "announcement_time": ["unknown"],
            "source": ["test"],
            "confirmed": [False],
            "fetched_at": ["2026-08-11T00:00:00+00:00"],
        }
    )


def _macro(*, block: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_date": [date(2026, 8, 12)],
            "event_name": ["米国CPI"],
            "event_type": ["inflation"],
            "severity": ["high"],
            "position_scale": [0.5],
            "block_new_entries": [block],
            "source": ["test"],
        }
    )


def test_upcoming_earnings_blocks_new_entry_and_sets_exit_date() -> None:
    result, _ = add_event_risk_controls(
        _features(),
        _earnings(date(2026, 8, 14)),
        pd.DataFrame(),
    )

    latest = result.iloc[-1]
    assert bool(latest["earnings_crossing_risk"]) is True
    assert bool(latest["event_entry_allowed"]) is False
    assert latest["event_trade_action"] == "NO_TRADE_EARNINGS"
    assert latest["event_position_scale"] == pytest.approx(0.5)
    assert latest["earnings_exit_date"] == date(2026, 8, 13)


def test_macro_event_reduces_position_and_can_block_entry() -> None:
    reduced, _ = add_event_risk_controls(
        _features(),
        _earnings(date(2026, 9, 30)),
        _macro(block=False),
    )
    blocked, _ = add_event_risk_controls(
        _features(),
        _earnings(date(2026, 9, 30)),
        _macro(block=True),
    )

    assert reduced.iloc[-1]["event_trade_action"] == "REDUCE_50_PERCENT"
    assert bool(reduced.iloc[-1]["event_entry_allowed"]) is True
    assert reduced.iloc[-1]["event_position_scale"] == pytest.approx(0.5)
    assert blocked.iloc[-1]["event_trade_action"] == "WAIT_MACRO_EVENT"
    assert bool(blocked.iloc[-1]["event_entry_allowed"]) is False


def test_missing_earnings_calendar_fails_closed() -> None:
    result, _ = add_event_risk_controls(_features(), pd.DataFrame(), pd.DataFrame())

    latest = result.iloc[-1]
    assert bool(latest["earnings_calendar_covered"]) is False
    assert bool(latest["event_entry_allowed"]) is False
    assert latest["event_trade_action"] == "CHECK_EARNINGS_CALENDAR"


def test_earnings_gap_down_requires_reversal_then_next_day_stop_buy() -> None:
    result, _ = add_event_risk_controls(
        _features(
            latest_open=95.0,
            latest_high=99.0,
            latest_low=94.0,
            latest_close=98.5,
        ),
        _earnings(date(2026, 8, 11)),
        pd.DataFrame(),
    )

    latest = result.iloc[-1]
    assert latest["earnings_gap_down"] == pytest.approx(-0.05)
    assert bool(latest["earnings_gd_reversal_signal"]) is True
    assert bool(latest["event_entry_allowed"]) is True
    assert latest["event_trade_action"] == "BUY_GD_REVERSAL"
    assert latest["event_position_scale"] == pytest.approx(1.0)
    assert pd.isna(latest["earnings_exit_date"])
    assert latest["earnings_gd_entry_price"] == pytest.approx(99.198)
    assert latest["earnings_gd_stop_price"] == pytest.approx(99.198 * 0.97)
    assert latest["earnings_gd_take_profit"] == pytest.approx(99.198 * 1.05)


def test_earnings_updater_writes_candidate_calendar(
    settings: Settings,
) -> None:
    updater = EarningsCalendarUpdater(
        settings,
        fetcher=lambda _ticker: {"Earnings Date": [pd.Timestamp("2026-08-20")]},
        max_workers=1,
    )

    report = updater.update(
        as_of=date(2026, 8, 11),
        tickers=["1111.T"],
    )

    stored = pd.read_parquet(calendar_paths(settings).earnings)
    assert report["covered_tickers"] == 1
    assert stored.iloc[0]["ticker"] == "1111.T"
    assert pd.Timestamp(stored.iloc[0]["earnings_date"]).date() == date(2026, 8, 20)
