import pandas as pd

from scripts.export_kabutore_replay import SIGNAL_SOURCE, build_replay_payload


def _source(live_signals: list[dict]) -> dict:
    return {
        "generated_at": "2026-08-20T04:37:01+00:00",
        "latest_date": "2026-08-20",
        "update": {"session": "morning", "session_label": "前場引け"},
        "signal_model": {"conditions": {"minimum_probability": 0.55}},
        "stocks": [
            {
                "ticker": "1234.T",
                "code": "12340",
                "company_name": "テスト社",
                "sector_17_name": "情報通信",
                "return_1d": 0.02,
            }
        ],
        "ten_day_signal_study": {
            "demo_trade_signal_study": {
                "status": "completed",
                "maximum_signals_per_day": 3,
                "historical_signals": [],
                "live_signals": live_signals,
            }
        },
    }


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-08-19",
                "ticker": "1234.T",
                "open": 100,
                "high": 105,
                "low": 99,
                "close": 102,
                "adjusted_close": 102,
                "volume": 2_000_000,
            },
            {
                "date": "2026-08-20",
                "ticker": "1234.T",
                "open": 102,
                "high": 108,
                "low": 101,
                "close": 104,
                "adjusted_close": 104,
                "volume": 2_500_000,
            },
        ]
    )


def test_exports_the_same_live_signal_and_2026_bars() -> None:
    payload = build_replay_payload(
        _source(
            [
                {
                    "signal_date": "2026-08-20",
                    "rank": 1,
                    "ticker": "1234.T",
                    "signal_close_yen": 104,
                    "daily_turnover_yen": 260_000_000,
                    "target_probability": 0.62,
                    "down_5pct_probability": 0.40,
                    "down_8pct_probability": 0.20,
                    "expected_net_return": 0.03,
                }
            ]
        ),
        _prices(),
    )

    assert payload["meta"]["signalSource"] == SIGNAL_SOURCE
    assert payload["meta"]["endDate"] == "2026-08-20"
    assert payload["signals"]["2026-08-20"][0]["targetProbability"] == 0.62
    assert payload["signals"]["2026-08-20"][0]["down8Probability"] == 0.20
    assert len(payload["bars"]["1234.T"]) == 2


def test_zero_signal_latest_date_overwrites_a_previous_candidate() -> None:
    previous = {
        "meta": {"signalSource": SIGNAL_SOURCE},
        "signals": {"2026-08-20": [{"ticker": "1234.T", "rank": 1}]},
    }

    payload = build_replay_payload(_source([]), _prices(), previous)

    assert payload["signals"]["2026-08-20"] == []
