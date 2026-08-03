from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from app.api.client import JQuantsClient
from app.config import Settings
from app.ingestion.pipeline import IngestionPipeline
from app.logging_config import configure_logging
from app.modeling.backtest import BacktestConfig, WalkForwardBacktester
from app.modeling.features import FeatureBuilder, LabelConfig
from app.modeling.store import BacktestStore
from app.quality.checks import QualityValidator
from app.storage.catalog import DuckDBCatalog
from app.storage.checkpoints import CheckpointStore
from app.storage.parquet import ParquetStore
from app.yahoo.analysis import YahooPatternAnalyzer
from app.yahoo.dashboard import DashboardExporter
from app.yahoo.ingestion import YahooFinanceIngestion, YahooPaths
from app.yahoo.quality import YahooQualityValidator
from app.yahoo.sakata_backtest import SakataBacktestConfig, SakataBacktester

LOGGER = logging.getLogger(__name__)


def iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"日付は YYYY-MM-DD 形式で指定してください: {value}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description="日本株短期上昇シグナル分析基盤",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="環境変数ファイル（既定: .env）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="デバッグログを表示",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="指定期間を初回・再開取得")
    ingest.add_argument("--start", type=iso_date, required=True)
    ingest.add_argument(
        "--end",
        type=iso_date,
        default=None,
        help="終了日（既定: 日本時間の当日）",
    )
    ingest.add_argument(
        "--datasets",
        nargs="+",
        choices=[
            "listed_master",
            "equities_daily",
            "topix_daily",
            "indices_daily",
        ],
        default=None,
        help="取得対象を限定（既定: すべて）",
    )

    update = commands.add_parser("update", help="最終保存日の翌日から差分更新")
    update.add_argument(
        "--end",
        type=iso_date,
        default=None,
        help="終了日（既定: 日本時間の当日）",
    )

    commands.add_parser("validate", help="processedデータの品質検査")
    commands.add_parser("status", help="保存件数、期間、重複、再開状態を表示")

    yahoo_ingest = commands.add_parser(
        "yahoo-ingest",
        help="yfinanceからPrime銘柄の直近1年を取得・差分更新",
    )
    yahoo_ingest.add_argument(
        "--as-of",
        type=iso_date,
        default=None,
        help="基準日（既定: 日本時間の当日）",
    )
    yahoo_ingest.add_argument(
        "--tickers-file",
        type=Path,
        default=None,
        help="任意のYahooティッカー一覧。省略時は保存済みPrime銘柄一覧を使用",
    )
    yahoo_ingest.add_argument(
        "--full-refresh",
        action="store_true",
        help="既存の最終日を無視し、直近1年を全銘柄再取得",
    )
    yahoo_ingest.add_argument(
        "--intraday-session",
        choices=["morning", "close"],
        default=None,
        help="当日の5分足を集計して前場引けまたは大引けの日足へ置換",
    )
    yahoo_analyze = commands.add_parser(
        "yahoo-analyze",
        help="+5%到達前の値動き・出来高を集計し最新候補を生成",
    )
    yahoo_analyze.add_argument("--top-n", type=int, default=20)
    yahoo_export = commands.add_parser(
        "yahoo-export-dashboard",
        help="スマホ画面へ送る集計済みJSONを生成",
    )
    yahoo_export.add_argument("--output", type=Path, default=Path("work/latest.json"))
    yahoo_sakata_backtest = commands.add_parser(
        "yahoo-backtest-sakata",
        help="酒田五法と従来仕込みスコアを同一条件で比較",
    )
    yahoo_sakata_backtest.add_argument("--start", type=iso_date, default=date(2026, 4, 1))
    yahoo_sakata_backtest.add_argument("--end", type=iso_date, default=None)
    yahoo_sakata_backtest.add_argument("--initial-capital", type=float, default=1_000_000)
    yahoo_sakata_backtest.add_argument("--top-n", type=int, default=10)
    yahoo_sakata_backtest.add_argument(
        "--output-dir", type=Path, default=Path("outputs/sakata_backtest")
    )
    yahoo_retail_backtest = commands.add_parser(
        "yahoo-backtest-retail",
        help="個人投資家フロー、酒田五法、従来方式を同一条件で比較",
    )
    yahoo_retail_backtest.add_argument("--start", type=iso_date, default=date(2026, 4, 1))
    yahoo_retail_backtest.add_argument("--end", type=iso_date, default=None)
    yahoo_retail_backtest.add_argument("--initial-capital", type=float, default=1_000_000)
    yahoo_retail_backtest.add_argument("--top-n", type=int, default=10)
    yahoo_retail_backtest.add_argument(
        "--output-dir", type=Path, default=Path("outputs/retail_flow_backtest")
    )
    commands.add_parser("yahoo-validate", help="Yahoo Finance日足の品質を検査")
    commands.add_parser("yahoo-status", help="Yahoo Finance取得・分析状態を表示")

    backtest = commands.add_parser(
        "backtest",
        help="特徴量を生成してウォークフォワード・バックテスト",
    )
    backtest.add_argument("--start", type=iso_date, default=None)
    backtest.add_argument("--end", type=iso_date, default=None)
    backtest.add_argument("--horizon-days", type=int, default=5)
    backtest.add_argument("--threshold", type=float, default=0.05)
    backtest.add_argument("--min-train-days", type=int, default=252)
    backtest.add_argument("--retrain-every-days", type=int, default=20)
    backtest.add_argument("--top-n", type=int, default=20)
    backtest.add_argument("--min-turnover", type=float, default=10_000_000)
    backtest.add_argument("--transaction-cost-bps", type=float, default=20.0)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    current_settings = settings or Settings.load(args.env_file)
    current_settings.ensure_directories()
    log_path = configure_logging(current_settings.log_dir, args.verbose)

    try:
        if args.command == "ingest":
            end = args.end or datetime.now(ZoneInfo("Asia/Tokyo")).date()
            with JQuantsClient(current_settings) as client:
                result = IngestionPipeline(current_settings, client).ingest(
                    args.start,
                    end,
                    datasets=tuple(args.datasets) if args.datasets else None,
                )
            _print_json(result)
        elif args.command == "update":
            with JQuantsClient(current_settings) as client:
                result = IngestionPipeline(current_settings, client).update(args.end)
            _print_json(result)
        elif args.command == "validate":
            report = QualityValidator(current_settings).run()
            _print_json(report.summary)
            print(f"issues: {report.issues_path}")
            print(f"summary: {report.summary_path}")
            if report.has_errors:
                LOGGER.error("品質検査でerrorを検出しました")
                return 2
        elif args.command == "status":
            _print_status(current_settings)
        elif args.command == "yahoo-ingest":
            result = YahooFinanceIngestion(current_settings).ingest(
                as_of=args.as_of or datetime.now(ZoneInfo("Asia/Tokyo")).date(),
                tickers_file=args.tickers_file,
                full_refresh=args.full_refresh,
                intraday_session=args.intraday_session,
            )
            _print_json(result)
        elif args.command == "yahoo-analyze":
            if args.top_n < 1:
                raise ValueError("--top-nは1以上で指定してください")
            result = YahooPatternAnalyzer(current_settings).run(top_n=args.top_n)
            _print_json(result)
        elif args.command == "yahoo-validate":
            report = YahooQualityValidator(current_settings).run()
            _print_json(report.summary)
            print(f"issues: {report.issues_path}")
            print(f"summary: {report.summary_path}")
            if report.has_errors:
                return 2
        elif args.command == "yahoo-export-dashboard":
            result = DashboardExporter(current_settings).export(args.output)
            _print_json(result)
        elif args.command in {"yahoo-backtest-sakata", "yahoo-backtest-retail"}:
            if args.initial_capital <= 0 or args.top_n < 1:
                raise ValueError("--initial-capitalと--top-nは正数で指定してください")
            result = SakataBacktester(current_settings).run(
                SakataBacktestConfig(
                    start=args.start,
                    end=args.end,
                    initial_capital=args.initial_capital,
                    top_n=args.top_n,
                ),
                output_dir=args.output_dir,
            )
            _print_json(result)
        elif args.command == "yahoo-status":
            _print_yahoo_status(current_settings)
        elif args.command == "backtest":
            parquet = ParquetStore(
                current_settings.raw_dir,
                current_settings.processed_dir,
            )
            equities = parquet.read_processed("equities_daily")
            calendar = parquet.read_processed("trading_calendar")
            modeling_data = FeatureBuilder(
                LabelConfig(
                    horizon_days=args.horizon_days,
                    threshold=args.threshold,
                )
            ).build(equities, calendar)
            result = WalkForwardBacktester(
                BacktestConfig(
                    start=args.start,
                    end=args.end,
                    min_train_days=args.min_train_days,
                    retrain_every_days=args.retrain_every_days,
                    horizon_days=args.horizon_days,
                    top_n=args.top_n,
                    min_turnover=args.min_turnover,
                    transaction_cost_bps=args.transaction_cost_bps,
                )
            ).run(modeling_data)
            paths = BacktestStore(
                current_settings.processed_dir,
                current_settings.metadata_dir,
            ).save(result)
            DuckDBCatalog(
                current_settings.duckdb_path,
                current_settings.processed_dir,
            ).refresh()
            _print_json(result.summary)
            for name, path in paths.items():
                print(f"{name}: {path}")
        else:
            parser.error(f"未対応のコマンドです: {args.command}")
        LOGGER.info("Command completed command=%s log=%s", args.command, log_path)
        return 0
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        LOGGER.exception("Command failed command=%s error=%s", args.command, exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"log: {log_path}", file=sys.stderr)
        return 1


def _print_status(settings: Settings) -> None:
    catalog = DuckDBCatalog(settings.duckdb_path, settings.processed_dir)
    rows = catalog.status()
    print("dataset         rows       min_date    max_date    duplicate_groups")
    for row in rows:
        print(
            f"{row['dataset']:<15} "
            f"{row['rows']:>10} "
            f"{str(row['min_date'] or '-'):>12} "
            f"{str(row['max_date'] or '-'):>12} "
            f"{row['duplicate_key_groups']:>17}"
        )

    state = CheckpointStore(settings.checkpoint_path).snapshot()
    status_counts: dict[str, int] = {}
    for unit in state["units"].values():
        status = str(unit.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    print("\ncheckpoints:", json.dumps(status_counts, ensure_ascii=False, sort_keys=True))
    if state["datasets"]:
        print("dataset_access:")
        for dataset, value in sorted(state["datasets"].items()):
            print(f"  {dataset}: {value.get('status')}")
    print(f"duckdb: {settings.duckdb_path}")


def _print_yahoo_status(settings: Settings) -> None:
    paths = YahooPaths(settings.data_dir / "yahoo")
    if not paths.prices_path.exists():
        print("Yahoo Finance日足は未取得です")
        return
    frame = pd.read_parquet(paths.prices_path, columns=["date", "ticker"])
    print("source: yfinance (personal research only)")
    print(f"rows: {len(frame)}")
    print(f"tickers: {frame['ticker'].nunique()}")
    print(f"min_date: {frame['date'].min()}")
    print(f"max_date: {frame['date'].max()}")
    analysis_path = paths.metadata_dir / "analysis_latest.json"
    print(f"analysis: {'ready' if analysis_path.exists() else 'not created'}")
    quality_path = paths.metadata_dir / "quality_latest.json"
    print(f"quality: {'ready' if quality_path.exists() else 'not checked'}")
    print(f"duckdb: {paths.metadata_dir / 'market.duckdb'}")


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
