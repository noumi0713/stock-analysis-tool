# Market environment history data

This pipeline is intentionally isolated from the live strategy and backtest calculation path. It does **not** modify signal, ranking, entry, exit, holding-period, position-count, allocation, fee, slippage, or existing backtest settings.

## Datasets

- `raw/equities/year=YYYY/part.parquet`: discovered TSE tickers, 5-year OHLC/Adj Close/Volume plus dividends and stock splits from yfinance.
- `raw/indexes/year=YYYY/part.parquet`: exact configured index tickers, 10 years. Unavailable requested indexes are reported; no substitute series is silently used.
- `raw/external/year=YYYY/part.parquet`: USD/JPY, US yields where an exact configured series exists, S&P 500, NASDAQ, SOX, VIX and Dow.
- `raw/universe.parquet`: ticker/name/sector/source snapshot used for the run.
- `raw/ticker_mapping.json`: requested series ↔ ticker ↔ formal-name manifest.
- `features/market_internals/year=YYYY/part.parquet`: advance/decline counts, 25-day advance-decline ratio, 20-day and 52-week new highs/lows, trading-value estimates, trading-value change and advancing trading-value ratio.
- `features/sectors/year=YYYY/part.parquet`: equal-weight sector return and trading value when sector classification is available.
- `features/metadata.json`: calculation definitions and bias notes.
- `quality/report.json`, `quality/completion_report.csv`, `quality/failed_equities.txt`: publication gate evidence.

## Acquisition rules

- Equity history is requested in batches of 150 tickers (configurable, requirement range 100–200).
- Failed downloads use exponential backoff for at least three attempts and are then retried individually.
- Bootstrap builds into a temporary staging tree. Production `raw/features/quality` is replaced only after critical checks pass.
- Daily update restores the last certified artifact and requests only the latest overlap window, then de-duplicates by `(Ticker, Date)`.
- Dates are normalized to daily dates; operational timezone is Asia/Tokyo.
- Raw source data is retained separately from computed features so all features can be rebuilt.

## Quality gate

FAIL publication when critical equity/external coverage, missingness, duplicate dates, OHLC consistency, non-positive prices, negative volume, or future-date checks fail. Stock-split anomalies are listed for inspection. Requested indexes that cannot be obtained with the exact configured series remain explicitly unavailable rather than being substituted.

## Known residual risks

1. **Survivorship bias:** the repository's discovered/current universe cannot prove complete delisted coverage for the entire five-year history. JPX notes that its Data Portal can expose recent delisted issues, but a full historical point-in-time universe must be supplied before this bias can be eliminated.
2. **Historical industry membership:** sector classification may be a current snapshot applied backward. Such sector aggregates are marked as potentially look-ahead biased.
3. **Market-cap weighting:** sector return defaults to equal weight because point-in-time market capitalization is not fabricated. A verified historical market-cap source is required before market-cap-weighted sector returns are enabled.
4. **US 2-year yield:** no proxy is silently substituted. The manifest keeps it unavailable until an exact verified source is configured.
5. **ETF/REIT/delisted handling:** any symbols present in the supplied/discovered universe are fetched and recorded. Missing historical constituents are reported rather than inferred.

## Run

```bash
cd momentum5d
python scripts/build_market_history.py --mode bootstrap --batch-size 150
python scripts/build_market_history.py --mode update --batch-size 150
```

For full point-in-time TSE coverage, pass a curated CSV/TXT with `--universe-file`. CSV supports `ticker`/`symbol`/`code`, optional `name`, and optional `sector`/`industry`.
