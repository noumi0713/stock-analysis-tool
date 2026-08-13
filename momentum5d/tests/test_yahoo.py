from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from app.config import Settings
from app.yahoo.analysis import YahooPatternAnalyzer
from app.yahoo.corporate_actions import (
    detect_effective_split_factors,
    normalize_split_adjusted_prices,
)
from app.yahoo.dashboard import DashboardExporter
from app.yahoo.ingestion import YahooConfig, YahooFinanceIngestion, YahooPaths
from app.yahoo.quality import YahooQualityValidator
from app.yahoo.trend import (
    add_trend_features,
    analyze_sector_volume_next_week_returns,
    load_sector_map,
    weekly_sector_33_returns,
)


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


def fake_daily_and_intraday_download(
    tickers: list[str],
    **kwargs: object,
) -> pd.DataFrame:
    if kwargs.get("interval") != "5m":
        return fake_download(tickers, **kwargs)
    timestamps = pd.DatetimeIndex(
        [
            "2026-01-09 09:00:00+09:00",
            "2026-01-09 11:30:00+09:00",
            "2026-01-09 13:00:00+09:00",
            "2026-01-09 15:30:00+09:00",
        ],
        name="Datetime",
    )
    data: dict[tuple[str, str], object] = {}
    for ticker in tickers:
        data[(ticker, "Open")] = [100.0, 104.0, 110.0, 111.0]
        data[(ticker, "High")] = [105.0, 108.0, 112.0, 115.0]
        data[(ticker, "Low")] = [99.0, 103.0, 109.0, 110.0]
        data[(ticker, "Close")] = [104.0, 107.0, 111.0, 114.0]
        data[(ticker, "Adj Close")] = [104.0, 107.0, 111.0, 114.0]
        data[(ticker, "Volume")] = [1000, 2000, 3000, 4000]
    return pd.DataFrame(data, index=timestamps)


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


def test_intraday_morning_session_replaces_partial_daily_bar(
    settings: Settings,
    tmp_path: Path,
) -> None:
    tickers = tmp_path / "tickers.txt"
    tickers.write_text("7203.T\n6758.T\n", encoding="utf-8")
    ingestion = YahooFinanceIngestion(
        settings,
        YahooConfig(
            retention_days=365,
            overlap_days=10,
            batch_size=2,
            pause_seconds=0,
            max_retries=0,
            timeout_seconds=1,
            intraday_min_coverage=0.7,
        ),
        downloader=fake_daily_and_intraday_download,
        sleeper=lambda _: None,
    )

    status = ingestion.ingest(
        as_of=date(2026, 1, 9),
        tickers_file=tickers,
        intraday_session="morning",
    )

    saved = pd.read_parquet(ingestion.paths.prices_path)
    latest = saved.loc[saved["date"] == date(2026, 1, 9)].sort_values("ticker")
    assert len(latest) == 2
    assert latest["open"].tolist() == [100.0, 100.0]
    assert latest["high"].tolist() == [108.0, 108.0]
    assert latest["low"].tolist() == [99.0, 99.0]
    assert latest["close"].tolist() == [107.0, 107.0]
    assert latest["volume"].tolist() == [3000, 3000]
    assert set(latest["source"]) == {"yfinance_intraday_5m"}
    assert status["intraday"]["session"] == "morning"
    assert status["intraday"]["cutoff_time_jst"] == "11:30"
    assert status["intraday"]["successful_tickers"] == 2
    assert status["intraday"]["data_through"].endswith("+09:00")


def test_intraday_close_is_not_marked_complete_before_session_end(
    settings: Settings,
    tmp_path: Path,
) -> None:
    tickers = tmp_path / "tickers.txt"
    tickers.write_text("7203.T\n6758.T\n", encoding="utf-8")

    def early_close_download(tickers: list[str], **kwargs: object) -> pd.DataFrame:
        downloaded = fake_daily_and_intraday_download(tickers, **kwargs)
        if kwargs.get("interval") == "5m":
            return downloaded.iloc[:-1]
        return downloaded

    ingestion = YahooFinanceIngestion(
        settings,
        YahooConfig(batch_size=2, pause_seconds=0, max_retries=0),
        downloader=early_close_download,
        sleeper=lambda _: None,
    )

    with pytest.raises(RuntimeError, match="終了時刻に達していません"):
        ingestion.ingest(
            as_of=date(2026, 1, 9),
            tickers_file=tickers,
            intraday_session="close",
        )


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


def test_expanding_retention_backfills_missing_history(
    settings: Settings,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def recording_download(tickers: list[str], **kwargs: object) -> pd.DataFrame:
        calls.append(str(kwargs["start"]))
        return fake_download(tickers, **kwargs)

    tickers = tmp_path / "tickers.txt"
    tickers.write_text("7203.T\n", encoding="utf-8")
    one_year = YahooFinanceIngestion(
        settings,
        YahooConfig(
            retention_days=365,
            overlap_days=10,
            batch_size=1,
            pause_seconds=0,
            max_retries=0,
            timeout_seconds=1,
        ),
        downloader=recording_download,
        sleeper=lambda _: None,
    )
    one_year.ingest(as_of=date(2026, 1, 10), tickers_file=tickers)
    calls.clear()

    three_years = YahooFinanceIngestion(
        settings,
        YahooConfig(
            retention_days=1096,
            overlap_days=10,
            batch_size=1,
            pause_seconds=0,
            max_retries=0,
            timeout_seconds=1,
        ),
        downloader=recording_download,
        sleeper=lambda _: None,
    )
    status = three_years.ingest(as_of=date(2026, 1, 10), tickers_file=tickers)

    assert calls == ["2023-01-10"]
    assert status["request_start"] == "2023-01-10"
    assert status["history_expansion_required"] is True


def test_ingestion_rejects_inconsistent_market_rows(
    settings: Settings,
    tmp_path: Path,
) -> None:
    def inconsistent_download(tickers: list[str], **kwargs: object) -> pd.DataFrame:
        frame = fake_download(tickers, **kwargs)
        ticker = tickers[0]
        frame.loc[frame.index[-2], (ticker, "High")] = 1.0
        frame.loc[frame.index[-1], (ticker, "Volume")] = -1.0
        return frame

    tickers = tmp_path / "tickers.txt"
    tickers.write_text("7203.T\n", encoding="utf-8")
    ingestion = YahooFinanceIngestion(
        settings,
        YahooConfig(
            retention_days=365,
            overlap_days=10,
            batch_size=1,
            pause_seconds=0,
            max_retries=0,
            timeout_seconds=1,
        ),
        downloader=inconsistent_download,
        sleeper=lambda _: None,
    )

    status = ingestion.ingest(as_of=date(2026, 1, 10), tickers_file=tickers)
    report = YahooQualityValidator(settings).run()

    assert status["rejected_market_rows"] == 2
    assert report.has_errors is False
    assert report.summary["rejected_market_rows"] == 0


def test_yahoo_analysis_writes_candidates_and_pattern_summary(
    settings: Settings,
) -> None:
    paths = YahooPaths(settings.data_dir / "yahoo")
    paths.ensure()
    dates = pd.date_range("2025-01-01", periods=80, freq="B")
    rows: list[dict[str, object]] = []
    price_noise = (-0.01, 0.01, -0.005, 0.005, -0.01)
    for ticker_number, ticker in enumerate(("1111.T", "2222.T"), start=1):
        for index, day in enumerate(dates):
            close = (100 + index * ticker_number * 0.5) * (
                1 + price_noise[index % len(price_noise)]
            )
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
    price_frame = pd.DataFrame(rows)
    pattern_rows = price_frame.loc[price_frame["ticker"] == "1111.T"].tail(9).index
    pattern_candles = [
        (139.0, 140.0, 138.0, 139.0),
        (138.5, 139.0, 137.0, 138.0),
        (137.5, 138.0, 136.0, 137.0),
        (136.5, 137.0, 135.0, 136.0),
        (135.5, 136.0, 134.0, 135.0),
        (135.0, 136.0, 134.0, 135.5),
        (135.2, 137.8, 135.0, 137.5),
        (137.0, 139.8, 136.8, 139.5),
        (139.0, 141.8, 138.8, 141.5),
    ]
    for row_index, (open_, high, low, close) in zip(pattern_rows, pattern_candles, strict=True):
        price_frame.loc[row_index, ["open", "high", "low", "close", "adjusted_close"]] = [
            open_,
            high,
            low,
            close,
            close,
        ]
        price_frame.loc[row_index, "turnover_value"] = (
            close * price_frame.loc[row_index, "volume"] * 100
        )
    last_pattern_row = pattern_rows[-1]
    price_frame.loc[last_pattern_row, "volume"] = 20_000
    price_frame.loc[last_pattern_row, "turnover_value"] = (
        price_frame.loc[last_pattern_row, "close"] * 20_000 * 100
    )
    price_frame.to_parquet(paths.prices_path, index=False)
    config_dir = settings.data_dir.parent / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "prime_sectors.csv").write_text(
        "code,sector_17_code,sector_33_code\n11110,9,3650\n22220,10,5250\n",
        encoding="utf-8",
    )

    result = YahooPatternAnalyzer(settings).run(top_n=1)

    candidates = pd.read_parquet(paths.processed_dir / "analysis" / "latest_candidates.parquet")
    scores = pd.read_parquet(paths.processed_dir / "analysis" / "latest_scores.parquet")
    assert result["tickers"] == 2
    assert 0 <= result["positive_rate"] <= 1
    assert len(candidates) <= 1
    assert "volume_change_1d" in result["patterns"]
    assert result["market_regime"]["favorable"] is True
    assert result["sector_map_coverage"] == 1.0
    assert result["industry_trends"]["sector_17"]
    assert result["industry_trends"]["sector_33_weekly"]["requested_start_date"] == (
        "2025-08-12"
    )
    assert "setup_score" in candidates.columns
    assert "trend_ranking_score" in candidates.columns
    assert "sector_17_name" in candidates.columns
    assert "setup_reasons" in candidates.columns
    assert result["technical_method"] == "capitulation_reversal_10d_ml_selective_v2"
    assert "bottom_pattern_study" in result
    assert result["bottom_pattern_study"]["horizon_days"] == 5
    assert "rise_pattern_backtest" in result
    assert "retail_flow_score" in candidates.columns
    assert "observed_inflow_score" in candidates.columns
    assert len(scores) == 2
    assert scores["rsi_14"].between(0, 100).all()
    assert scores["atr_14_pct"].gt(0).all()
    assert scores["score_rank"].notna().all()


def test_effective_split_normalization_keeps_prices_and_volume_continuous() -> None:
    prices = pd.DataFrame(
        [
            {
                "date": date(2026, 7, 17),
                "ticker": "1111.T",
                "open": 1980.0,
                "high": 2020.0,
                "low": 1970.0,
                "close": 2000.0,
                "adjusted_close": 2000.0,
                "volume": 1000.0,
            },
            {
                "date": date(2026, 7, 21),
                "ticker": "1111.T",
                "open": 1000.0,
                "high": 1020.0,
                "low": 990.0,
                "close": 1005.0,
                "adjusted_close": 1005.0,
                "volume": 2200.0,
            },
        ]
    )

    factors = detect_effective_split_factors(prices)
    normalized = normalize_split_adjusted_prices(prices)

    assert factors.tolist() == [1.0, 2.0]
    assert normalized["close"].tolist() == [1000.0, 1005.0]
    assert normalized["volume"].tolist() == [2000.0, 2200.0]
    assert normalized["adjusted_close"].tolist() == [1000.0, 1005.0]


def test_trend_features_rank_stronger_industry_higher(tmp_path: Path) -> None:
    dates = pd.date_range("2025-10-01", periods=70, freq="B")
    rows: list[dict[str, object]] = []
    for ticker, code, daily_gain in (
        ("1111.T", "11110", 0.003),
        ("2222.T", "22220", -0.001),
    ):
        close = 100.0
        for day in dates:
            close *= 1 + daily_gain
            rows.append(
                {
                    "date": day.date(),
                    "ticker": ticker,
                    "code": code,
                    "adjusted_close": close,
                    "return_5d": np.nan,
                    "return_20d": np.nan,
                    "setup_score": 0.70,
                }
            )
    features = pd.DataFrame(rows).sort_values(["ticker", "date"])
    grouped = features.groupby("ticker", sort=False)
    features["return_5d"] = features["adjusted_close"] / grouped["adjusted_close"].shift(5) - 1
    features["return_20d"] = features["adjusted_close"] / grouped["adjusted_close"].shift(20) - 1
    sector_path = tmp_path / "prime_sectors.csv"
    sector_path.write_text(
        "code,sector_17_code,sector_33_code\n11110,9,3650\n22220,10,5250\n",
        encoding="utf-8",
    )

    trended = add_trend_features(features, load_sector_map(sector_path))
    latest = trended.loc[trended["date"] == trended["date"].max()]
    strong = latest.loc[latest["ticker"] == "1111.T"].iloc[0]
    weak = latest.loc[latest["ticker"] == "2222.T"].iloc[0]

    assert strong["sector_17_name"] == "電機・精密"
    assert strong["sector_17_trend_score"] > weak["sector_17_trend_score"]
    assert strong["trend_ranking_score"] > weak["trend_ranking_score"]


def test_weekly_sector_returns_start_on_requested_date_and_include_all_sectors() -> None:
    dates = pd.date_range("2025-08-12", periods=12, freq="B")
    rows: list[dict[str, object]] = []
    for ticker, sector_code, sector_name, daily_gain in (
        ("1111.T", "3650", "電気機器", 0.01),
        ("2222.T", "5250", "情報・通信業", -0.005),
    ):
        close = 100.0
        ticker_rows: list[dict[str, object]] = []
        for day in dates:
            close *= 1 + daily_gain
            ticker_rows.append(
                {
                    "date": day.date(),
                    "ticker": ticker,
                    "return_5d": np.nan,
                    "sector_33_code": sector_code,
                    "sector_33_name": sector_name,
                    "adjusted_close": close,
                }
            )
        ticker_frame = pd.DataFrame(ticker_rows)
        ticker_frame["return_5d"] = (
            ticker_frame["adjusted_close"] / ticker_frame["adjusted_close"].shift(5) - 1
        )
        rows.extend(ticker_frame.to_dict("records"))

    history = weekly_sector_33_returns(pd.DataFrame(rows))

    assert history["requested_start_date"] == "2025-08-12"
    assert history["first_as_of"] == "2025-08-22"
    assert history["week_count"] == 2
    assert all(len(week["sectors"]) == 33 for week in history["weeks"])
    first_week = history["weeks"][0]
    assert first_week["sector_count"] == 2
    assert first_week["sectors"][0]["sector_33_name"] == "電気機器"
    assert first_week["sectors"][0]["rank"] == 1
    missing = next(
        row for row in first_week["sectors"] if row["sector_33_name"] == "水産・農林業"
    )
    assert missing["median_return_5d"] is None
    assert missing["stock_count"] == 0


def test_sector_volume_change_is_compared_with_following_week_return() -> None:
    dates = pd.date_range("2025-08-04", periods=40, freq="B")
    rows: list[dict[str, object]] = []
    for ticker_number, (ticker, sector_code, sector_name) in enumerate(
        (
            ("1111.T", "3650", "電気機器"),
            ("2222.T", "3650", "電気機器"),
            ("3333.T", "5250", "情報・通信業"),
            ("4444.T", "5250", "情報・通信業"),
        ),
        start=1,
    ):
        close = 100.0
        for index, day in enumerate(dates):
            week_number = index // 5
            volume_multiplier = 1 + 0.25 * ((week_number + ticker_number) % 4)
            next_return_driver = 0.004 * ((week_number + ticker_number) % 4)
            close *= 1 + next_return_driver
            rows.append(
                {
                    "date": day.date(),
                    "ticker": ticker,
                    "adjusted_close": close,
                    "volume": 100_000 * ticker_number * volume_multiplier,
                    "sector_33_code": sector_code,
                    "sector_33_name": sector_name,
                }
            )

    study = analyze_sector_volume_next_week_returns(pd.DataFrame(rows))

    assert study["requested_start_date"] == "2025-08-12"
    assert study["period_count"] == 6
    assert study["sector_week_observation_count"] == 12
    assert study["overall"]["sample_count"] == 12
    assert len(study["direction_groups"]) == 2
    assert len(study["volume_change_quintiles"]) == 5
    assert len(study["by_sector"]) == 2


def test_sector_volume_study_excludes_incomplete_one_day_target_week() -> None:
    dates = list(pd.date_range("2025-08-04", periods=15, freq="B"))
    dates.append(pd.Timestamp("2025-08-25"))
    rows = []
    for index, day in enumerate(dates):
        rows.append(
            {
                "date": day.date(),
                "ticker": "1111.T",
                "adjusted_close": 100.0 + index,
                "volume": 100_000 + index * 1_000,
                "sector_33_code": "3650",
                "sector_33_name": "電気機器",
            }
        )

    study = analyze_sector_volume_next_week_returns(pd.DataFrame(rows))

    assert study["period_count"] == 1
    assert study["last_next_as_of"] == "2025-08-22"


def test_candidate_ranking_requires_strong_shape_ml_and_returns_at_most_one() -> None:
    latest_date = date(2026, 1, 9)
    common = {
        "date": latest_date,
        "close": 100.0,
        "adjusted_close": 100.0,
        "volume": 100_000,
        "turnover_value": 300_000_000,
        "return_20d": 0.05,
        "intraday_return": 0.01,
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
        "rsi_14": 62.0,
        "sakata_pattern": "赤三兵",
        "sakata_reasons": "赤三兵",
        "sakata_score": 0.90,
        "sakata_buy_signal": True,
        "sakata_sell_signal": False,
        "sakata_bullish_count": 1,
        "sakata_bearish_count": 0,
        "turnover_ratio_5_20": 1.4,
        "retail_volume_attention_rank": 0.8,
        "retail_return_attention_rank": 0.7,
        "retail_turnover_rank": 0.8,
        "retail_relative_strength_rank": 0.7,
        "retail_attention_acceleration_score": 0.65,
        "retail_discovery_score": 0.78,
        "retail_understanding_proxy_score": 0.65,
        "retail_expectation_score": 0.68,
        "retail_safety_score": 0.70,
        "retail_action_score": 0.72,
        "retail_overheat_penalty": 0.10,
        "retail_loss_anxiety_penalty": 0.05,
        "retail_flow_score": 0.76,
        "retail_attention_hybrid_score": 0.78,
        "volume_ratio_1_20": 1.60,
        "turnover_ratio_1_20": 1.65,
        "observed_volume_ratio_rank": 0.90,
        "observed_turnover_ratio_rank": 0.92,
        "observed_volume_intensity_score": 0.75,
        "observed_turnover_intensity_score": 0.78,
        "observed_price_confirmation_score": 0.70,
        "observed_inflow_score": 0.79,
        "observed_inflow_confirmed": True,
        "legacy_setup_score": 0.80,
        "ml_sharp_probability": 0.0,
        "ml_sharp_down_5pct_probability": 1.0,
        "ml_sharp_down_8pct_probability": 1.0,
        "ml_sharp_expected_net_return": -1.0,
        "ml_sharp_model_samples": 0,
        "ml_sharp_signal": False,
        "ml_sharp_rank": pd.NA,
        "ml_sharp_reason": "",
        "ml_sharp_entry_rule": "翌営業日寄付きが前日終値比+3%以下の場合のみ有効",
        "ml_ten_day_probability": 0.0,
        "ml_ten_day_down_5pct_probability": 1.0,
        "ml_ten_day_down_8pct_probability": 1.0,
        "ml_ten_day_expected_net_return": -1.0,
        "ml_ten_day_model_samples": 0,
        "ml_ten_day_signal": False,
        "ml_ten_day_rank": pd.NA,
        "ml_ten_day_reason": "",
        "ml_ten_day_entry_rule": "翌営業日寄付き条件待ち",
    }
    features = pd.DataFrame(
        [
            {
                **common,
                "ticker": "INFLOW.T",
                "code": "11110",
                "return_1d": 0.005,
                "return_5d": 0.01,
                "setup_score": 0.82,
                "signal_score": 0.82,
            },
            {
                **common,
                "ticker": "QUIET.T",
                "code": "22220",
                "return_1d": 0.01,
                "return_5d": 0.02,
                "observed_inflow_score": 0.90,
                "observed_inflow_confirmed": False,
                "setup_score": 0.95,
                "signal_score": 0.95,
            },
            {
                **common,
                "ticker": "PATTERN.T",
                "code": "33330",
                "return_1d": -0.03,
                "return_5d": -0.08,
                "observed_inflow_score": 0.30,
                "observed_inflow_confirmed": False,
                "rise_pattern_probability": 0.95,
                "rise_pattern_samples": 120,
                "rise_pattern_signal": True,
                "rise_pattern_shape": "sharp_selloff",
                "rise_pattern_reason": "急落継続・過去類似120件・補正+5%率95%",
                "ml_sharp_probability": 0.62,
                "ml_sharp_down_5pct_probability": 0.30,
                "ml_sharp_down_8pct_probability": 0.15,
                "ml_sharp_expected_net_return": 0.01,
                "ml_sharp_model_samples": 500,
                "ml_sharp_signal": True,
                "ml_sharp_rank": 1,
                "ml_sharp_reason": "急落継続・売買代金2億円以上・強形状ML参考率62%",
                "ml_ten_day_probability": 0.68,
                "ml_ten_day_down_5pct_probability": 0.25,
                "ml_ten_day_down_8pct_probability": 0.10,
                "ml_ten_day_expected_net_return": 0.02,
                "ml_ten_day_model_samples": 500,
                "ml_ten_day_signal": True,
                "ml_ten_day_rank": 1,
                "ml_ten_day_reason": "急落継続・10営業日+5%参考率68%",
                "setup_score": 0.95,
                "signal_score": 0.95,
            },
        ]
    )

    candidates = YahooPatternAnalyzer._latest_candidates(features, top_n=20)

    assert candidates["ticker"].tolist() == ["PATTERN.T"]
    assert candidates.iloc[0]["ml_ten_day_signal"]
    assert "急落継続" in candidates.iloc[0]["setup_reasons"]
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
                "high": 55.0,
                "low": 40.0,
                "close": 42.0,
                "adjusted_close": 80.0,
                "volume": 0,
                "turnover_value": 0,
                "dividends": 0.0,
                "stock_splits": 2.0,
                "source": "yfinance",
            },
            {
                "date": date(2026, 1, 7),
                "ticker": "1111.T",
                "code": "11110",
                "open": 50.0,
                "high": 45.0,
                "low": 40.0,
                "close": 50.0,
                "adjusted_close": 50.0,
                "volume": 1000,
                "turnover_value": 50_000,
                "dividends": 0.0,
                "stock_splits": 0.0,
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
    assert "invalid_market_row_removed" in checks
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
    paths = YahooPaths(settings.data_dir / "yahoo")
    pd.DataFrame(
        {
            "ticker": ["1111.T", "2222.T"],
            "code": ["11110", "22220"],
        }
    ).to_parquet(paths.universe_path, index=False)
    scores_path = paths.processed_dir / "analysis" / "latest_scores.parquet"
    candidate_path = paths.processed_dir / "analysis" / "latest_candidates.parquet"
    candidate = pd.read_parquet(scores_path).loc[lambda frame: frame["code"] == "11110"].head(1)
    candidate.to_parquet(candidate_path, index=False)
    output = tmp_path / "latest.json"

    result = DashboardExporter(settings).export(output)

    payload = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert result["candidate_count"] <= 1
    assert payload["schema_version"] == 13
    assert payload["personal_research_only"] is True
    assert payload["update"]["session"] == "daily"
    assert payload["update"]["status"] == "complete"
    assert payload["market_regime"]["favorable"] is True
    assert "patterns" in payload
    assert "bottom_pattern_study" in payload
    assert "rise_pattern_backtest" in payload
    assert "golden_cross_volume_study" in payload
    assert "perfect_order_pullback_study" in payload
    assert "candidates" in payload
    assert len(payload["stocks"]) == 2
    assert "rsi_14" in payload["stocks"][0]
    assert "atr_14_pct" in payload["stocks"][0]
    assert len(payload["indicator_notes"]) == 5
    assert payload["technical_method"]["label"] == "10営業日+5% ML厳選"
    assert payload["signal_model"]["conditions"]["target_holding_days"] == 10
    assert payload["signal_model"]["conditions"]["minimum_turnover_yen"] == 200_000_000
    assert "ten_day_signal_study" in payload
    assert "open" not in payload["candidates"][0]
    assert "setup_score" in payload["candidates"][0]
    assert "trend_ranking_score" in payload["candidates"][0]
    assert payload["candidates"][0]["sector_17_name"]
    assert "setup_reasons" in payload["candidates"][0]
    assert "retail_flow_score" in payload["candidates"][0]
    assert "retail_attention_hybrid_score" in payload["candidates"][0]
    assert "observed_inflow_score" in payload["candidates"][0]
    assert "sakata_pattern" in payload["candidates"][0]
    assert "event_entry_allowed" in payload["candidates"][0]
    assert "event_trade_action" in payload["candidates"][0]
    assert "event_risk_summary" in payload
    assert payload["candidates"][0]["company_name"] is None
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


def test_ten_day_study_accepts_three_year_evaluation_window() -> None:
    import inspect

    from app.yahoo.rise_pattern import add_ten_day_signal_and_study

    signature = inspect.signature(add_ten_day_signal_and_study)
    assert signature.parameters["evaluation_days"].default == 240


def test_dashboard_export_fills_company_name_from_bundled_master(
    settings: Settings,
    tmp_path: Path,
) -> None:
    test_yahoo_analysis_writes_candidates_and_pattern_summary(settings)
    paths = YahooPaths(settings.data_dir / "yahoo")
    pd.DataFrame(
        {
            "ticker": ["1111.T", "2222.T"],
            "code": ["11110", "22220"],
        }
    ).to_parquet(paths.universe_path, index=False)
    scores_path = paths.processed_dir / "analysis" / "latest_scores.parquet"
    candidate_path = paths.processed_dir / "analysis" / "latest_candidates.parquet"
    candidate = pd.read_parquet(scores_path).loc[lambda frame: frame["code"] == "11110"].head(1)
    candidate.to_parquet(candidate_path, index=False)
    config_dir = settings.data_dir.parent / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "prime_names.csv").write_text(
        "code,company_name\n11110,テスト工業\n22220,サンプル商事\n",
        encoding="utf-8",
    )
    output = tmp_path / "latest.json"

    DashboardExporter(settings).export(output)

    payload = __import__("json").loads(output.read_text(encoding="utf-8"))
    expected_names = {"11110": "テスト工業", "22220": "サンプル商事"}
    candidate = payload["candidates"][0]
    assert candidate["company_name"] == expected_names[str(candidate["code"])]
