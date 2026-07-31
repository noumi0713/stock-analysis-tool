from __future__ import annotations

from app.__main__ import build_parser, main
from app.config import Settings


def test_status_works_without_api_key(settings: Settings, capsys) -> None:
    no_key = Settings(
        api_key=None,
        base_url=settings.base_url,
        data_dir=settings.data_dir,
        log_dir=settings.log_dir,
        requests_per_minute=settings.requests_per_minute,
        connect_timeout_seconds=settings.connect_timeout_seconds,
        read_timeout_seconds=settings.read_timeout_seconds,
        max_retries=settings.max_retries,
        backoff_base_seconds=settings.backoff_base_seconds,
        abnormal_return_threshold=settings.abnormal_return_threshold,
        split_ratio_tolerance=settings.split_ratio_tolerance,
        universe_market_codes=settings.universe_market_codes,
    )

    exit_code = main(["status"], settings=no_key)

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "equities_daily" in output
    assert "duplicate_groups" in output
    assert no_key.duckdb_path.exists()


def test_validate_without_data_reports_failure(settings: Settings, capsys) -> None:
    exit_code = main(["validate"], settings=settings)

    error = capsys.readouterr().err
    assert exit_code == 1
    assert "equities_daily" in error


def test_backtest_cli_options_are_parsed() -> None:
    args = build_parser().parse_args(
        [
            "backtest",
            "--start",
            "2025-01-01",
            "--top-n",
            "10",
            "--transaction-cost-bps",
            "15",
        ]
    )

    assert args.command == "backtest"
    assert args.start.isoformat() == "2025-01-01"
    assert args.top_n == 10
    assert args.transaction_cost_bps == 15


def test_ingest_can_limit_datasets() -> None:
    args = build_parser().parse_args(
        [
            "ingest",
            "--start",
            "2025-01-01",
            "--datasets",
            "equities_daily",
        ]
    )

    assert args.datasets == ["equities_daily"]


def test_yahoo_commands_are_parsed() -> None:
    ingest = build_parser().parse_args(
        [
            "yahoo-ingest",
            "--as-of",
            "2026-07-29",
            "--tickers-file",
            "tickers.txt",
            "--intraday-session",
            "morning",
        ]
    )
    analyze = build_parser().parse_args(["yahoo-analyze", "--top-n", "15"])

    assert ingest.command == "yahoo-ingest"
    assert ingest.as_of.isoformat() == "2026-07-29"
    assert str(ingest.tickers_file) == "tickers.txt"
    assert ingest.intraday_session == "morning"
    assert analyze.command == "yahoo-analyze"
    assert analyze.top_n == 15
