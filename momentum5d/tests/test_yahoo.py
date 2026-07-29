from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from app.config import Settings
from app.yahoo.analysis import YahooPatternAnalyzer
from app.yahoo.dashboard import DashboardExporter
from app.yahoo.ingestion import YahooConfig, YahooFinanceIngestion, YahooPaths
from app.yahoo.quality import YahooQualityValidator


def fake_download(
    tickers: list[str],
    **_: object,
) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", "2026-01-10", freq="B")
    data: dict[tuple[str, str], object] = {}
    for number, ticker in enumerate(tickers, start=1):
        close = np.linspace(100 * number, 150 * number, len(dates))
        data[(ticker, "Open")] = close - 1
        data[(ticker, "High")] = close + 3
        data[(ticker, "Low")] = close - 2
        data[(ticker, "Close")] = close
        data[(ticker, "Adj Close")] = close
        data[(ticker, "Volume")] = np.linspace(1000, 3000, len(dates))
        data[(ticker, "Dividends")] = np.zeros(len(dates))
        data[(ticker, "Stock Splits")] = np.zeros(len(dates))
    frame = pd.DataFrame(data, index=dates)
    frame.index.name = "Date"
    return frame


def test_yahoo_ingestion_normalizes_deduplicates_and_keeps_one_year(
    settings: Settings,
    tmp_path: Path,
) -> None:
    tickers = tmp_path / "tickers.txt"
    tickers.write_text("7203.T\n6758.T\n", encoding="utf-8")
    config = YahooConfig(
        retention_days=365,
        overlap_days=10,
        batch_size=1,
        pause_seconds=0,
        max_retries=0,
        timeout_seconds=1,
    )
    ingestion = YahooFinanceIngestion(
        settings,
        config,
        downloader=fake_download,
        sleeper=lambda _: None,
    )

    first = ingestion.ingest(as_of=date(2026, 1, 10), tickers_file=tickers)
    second = ingestion.ingest(as_of=date(2026, 1, 10), tickers_file=tickers)

    saved = pd.read_parquet(ingestion.paths.prices_path)
    assert first["tickers"] == 2
    assert second["rows"] == len(saved)
    assert saved["date"].min() >= date(2025, 1, 10)
    assert saved.duplicated(["ticker", "date"]).sum() == 0
    assert set(saved["code"]) == {"72030", "67580"}
    assert set(saved["source"]) == {"yfinance"}
    assert {"dividends", "stock_splits"} <= set(saved.columns)


def test_new_ticker_gets_full_retention_window_during_update(
    settings: Settings,
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[str, ...], str]] = []

    def recording_download(tickers: list[str], **kwargs: object) -> pd.DataFrame:
        calls.append((tuple(tickers), str(kwargs["start"])))
        return fake_download(tickers, **kwargs)

    config = YahooConfig(
        retention_days=365,
        overlap_days=10,
        batch_size=1,
        pause_seconds=0,
        max_retries=0,
        timeout_seconds=1,
    )
    ingestion = YahooFinanceIngestion(
        settings,
        config,
        downloader=recording_download,
        sleeper=lambda _: None,
    )
    first_file = tmp_path / "first.txt"
    first_file.write_text("7203.T\n", encoding="utf-8")
    ingestion.ingest(as_of=date(2026, 1, 10), tickers_file=first_file)
    calls.clear()
    expanded_file = tmp_path / "expanded.txt"
    expanded_file.write_text("7203.T\n6758.T\n", encoding="utf-8")

    ingestion.ingest(as_of=date(2026, 1, 10), tickers_file=expanded_file)

    starts = {tickers[0]: start for tickers, start in calls}
    assert starts["7203.T"] == "2025-12-30"
    assert starts["6758.T"] == "2025-01-10"


def test_yahoo_analysis_writes_candidates_and_pattern_summary(
    settings: Settings,
) -> None:
    paths = YahooPaths(settings.data_dir / "yahoo")
    paths.ensure()
    dates = pd.date_range("2025-01-01", periods=80, freq="B")
    rows: list[dict[str, object]] = []
    for ticker_number, ticker in enumerate(("1111.T", "2222.T"), start=1):
        for index, day in enumerate(dates):
            close = 100 + index * ticker_number * 0.5
            if index == len(dates) - 1:
                close *= 0.99
            volume = 1000 + index * 20 * ticker_number
            rows.append(
                {
                    "date": day.date(),
                    "ticker": ticker,
                    "code": ticker.removesuffix(".T") + "0",
                    "open": close - 1,
                    "high": close * (1.07 if index % 12 == 0 else 1.01),
                    "low": close - 2,
                    "close": close,
                    "adjusted_close": close,
                    "volume": volume,
                    "turnover_value": close * volume * 100,
                    "source": "yfinance",
                }
            )
    pd.DataFrame(rows).to_parquet(paths.prices_path, index=False)

    result = YahooPatternAnalyzer(settings).run(top_n=1)

    candidates = pd.read_parquet(paths.processed_dir / "analysis" / "latest_candidates.parquet")
    assert result["tickers"] == 2
    assert 0 <= result["positive_rate"] <= 1
    assert len(candidates) <= 1
    assert "volume_change_1d" in result["patterns"]
    assert "setup_score" in candidates.columns
    assert "setup_reasons" in candidates.columns


def test_setup_ranking_excludes_already_surging_stock() -> None:
    latest_date = date(2026, 1, 9)
    common = {
        "date": latest_date,
        "close": 100.0,
        "adjusted_close": 100.0,
        "volume": 100_000,
        "turnover_value": 100_000_000,
        "return_20d": 0.05,
        "volume_change_1d": 0.10,
        "volume_ratio_5_20": 1.0,
        "close_to_ma20": 0.02,
        "breakout_20d": -0.03,
        "volatility_10d": 0.012,
        "volatility_20d": 0.018,
        "range_width_10d": 0.05,
        "up_volume_share_10d": 0.58,
        "setup_compression_score": 0.8,
        "setup_accumulation_score": 0.7,
        "setup_position_score": 0.9,
    }
    features = pd.DataFrame(
        [
            {
                **common,
                "ticker": "CALM.T",
                "code": "11110",
                "return_1d": 0.005,
                "return_5d": 0.01,
                "setup_score": 0.82,
                "signal_score": 0.82,
            },
            {
                **common,
                "ticker": "HOT.T",
                "code": "22220",
                "return_1d": 0.08,
                "return_5d": 0.28,
                "setup_score": 0.95,
                "signal_score": 0.95,
            },
        ]
    )

    candidates = YahooPatternAnalyzer._latest_candidates(features, top_n=20)

    assert candidates["ticker"].tolist() == ["CALM.T"]
    assert candidates.iloc[0]["setup_reasons"]


def test_yahoo_quality_detects_ohlc_volume_return_split_and_missing_ticker(
    settings: Settings,
) -> None:
    paths = YahooPaths(settings.data_dir / "yahoo")
    paths.ensure()
    prices = pd.DataFrame(
        [
            {
                "date": date(2026, 1, 5),
                "ticker": "1111.T",
                "code": "11110",
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 100.0,
                "adjusted_close": 50.0,
                "volume": 1000,
                "turnover_value": 100_000,
                "dividends": 0.0,
                "stock_splits": 0.0,
                "source": "yfinance",
            },
            {
                "date": date(2026, 1, 6),
                "ticker": "1111.T",
                "code": "11110",
                "open": 50.0,
                "high": 45.0,
                "low": 40.0,
                "close": 42.0,
                "adjusted_close": 80.0,
                "volume": 0,
                "turnover_value": 0,
                "dividends": 0.0,
                "stock_splits": 2.0,
                "source": "yfinance",
            },
        ]
    )
    universe = pd.DataFrame(
        {
            "ticker": ["1111.T", "2222.T"],
            "code": ["11110", "22220"],
        }
    )
    prices.to_parquet(paths.prices_path, index=False)
    universe.to_parquet(paths.universe_path, index=False)

    report = YahooQualityValidator(settings).run()

    checks = set(report.issues["check_name"])
    assert "ohlc_inconsistent" in checks
    assert "zero_volume" in checks
    assert "abnormal_adjusted_return" in checks
    assert "stock_split_event" in checks
    assert "universe_ticker_without_price" in checks
    assert report.issues_path.exists()
    with duckdb.connect(
        str(paths.metadata_dir / "market.duckdb"),
        read_only=True,
    ) as connection:
        issue_count = connection.execute("SELECT COUNT(*) FROM quality_issues").fetchone()[0]
    assert issue_count == len(report.issues)


def test_dashboard_export_contains_candidates_and_recent_candidate_charts(
    settings: Settings,
    tmp_path: Path,
) -> None:
    test_yahoo_analysis_writes_candidates_and_pattern_summary(settings)
    output = tmp_path / "latest.json"

    result = DashboardExporter(settings).export(output)

    payload = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert result["candidate_count"] <= 1
    assert payload["schema_version"] == 3
    assert payload["personal_research_only"] is True
    assert "patterns" in payload
    assert "candidates" in payload
    assert "open" not in payload["candidates"][0]
    assert "setup_score" in payload["candidates"][0]
    assert "setup_reasons" in payload["candidates"][0]
    code = str(payload["candidates"][0]["code"])
    assert code in payload["charts"]
    assert 1 <= len(payload["charts"][code]) <= 60
    assert set(payload["charts"][code][0]) == {
        "date",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    }
