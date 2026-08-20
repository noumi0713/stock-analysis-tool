from scripts.export_latest_signals import build_latest_signals


def _source(live_signals: list[dict]) -> dict:
    return {
        "generated_at": "2026-08-19T08:37:59+00:00",
        "latest_date": "2026-08-19",
        "update": {"market_date": "2026-08-19"},
        "metrics": {"tickers": 3_678},
        "stocks": [
            {
                "ticker": "1234.T",
                "code": "12340",
                "company_name": "テスト社",
                "close": 500.0,
                "turnover_value": 200_000_000.0,
                "rsi_14": 42.0,
                "atr_14_pct": 0.031,
            }
        ],
        "ten_day_signal_study": {
            "validation": {"selected_signals": 53},
            "validation_folds": [],
            "demo_trade_signal_study": {
                "status": "completed",
                "live_signals": live_signals,
            },
        },
    }


def test_exports_latest_market_date_and_required_signal_fields() -> None:
    payload = build_latest_signals(
        _source(
            [
                {
                    "signal_date": "2026-08-19",
                    "rank": 1,
                    "ticker": "1234.T",
                    "shape": "capitulation_reversal",
                    "target_probability": 0.61,
                    "down_5pct_probability": 0.40,
                    "down_8pct_probability": 0.20,
                    "expected_net_return": 0.03,
                }
            ]
        )
    )

    assert payload["date"] == "2026-08-19"
    assert payload["signal_count"] == 1
    assert payload["signals"][0] == {
        **payload["signals"][0],
        "date": "2026-08-19",
        "code": "12340",
        "name": "テスト社",
        "rank": 1,
        "signal": "capitulation_reversal",
        "close": 500.0,
        "trading_value": 200_000_000.0,
        "RSI": 42.0,
        "ATR": 0.031,
        "target_probability": 0.61,
        "down_5pct_probability": 0.40,
        "down_8pct_probability": 0.20,
    }


def test_exports_empty_array_for_a_zero_signal_day() -> None:
    payload = build_latest_signals(_source([]))

    assert payload["date"] == "2026-08-19"
    assert payload["signal_count"] == 0
    assert payload["signals"] == []
