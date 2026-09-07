from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb


JST = ZoneInfo("Asia/Tokyo")
HORIZONS = (5, 10, 20, 40, 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STEP2 forward returns, MFE and MAE")
    parser.add_argument("--root", default="data/market_history")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--memory-limit", default="2GB")
    return parser.parse_args()


def parquet_glob(path: Path) -> str:
    return str(path.resolve() / "**" / "*.parquet").replace("'", "''")


def fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(path.parents[2])).encode())
        digest.update(str(path.stat().st_size).encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def target_sql() -> str:
    windows = []
    selections = []
    for horizon in HORIZONS:
        frame = (
            f"PARTITION BY Ticker ORDER BY Date "
            f"ROWS BETWEEN 1 FOLLOWING AND {horizon} FOLLOWING"
        )
        windows.extend(
            [
                f"count(adj_close) OVER ({frame}) AS future_count_{horizon}d",
                f"lead(adj_close,{horizon}) OVER (PARTITION BY Ticker ORDER BY Date) "
                f"AS exit_close_{horizon}d",
                f"max(adj_high) OVER ({frame}) AS max_high_{horizon}d",
                f"min(adj_low) OVER ({frame}) AS min_low_{horizon}d",
            ]
        )
        selections.extend(
            [
                f"CASE WHEN future_count_{horizon}d={horizon} THEN "
                f"exit_close_{horizon}d/nullif(adj_close,0)-1 END "
                f"AS forward_return_{horizon}d",
                f"CASE WHEN future_count_{horizon}d={horizon} THEN "
                f"max_high_{horizon}d/nullif(adj_close,0)-1 END AS mfe_{horizon}d",
                f"CASE WHEN future_count_{horizon}d={horizon} THEN "
                f"min_low_{horizon}d/nullif(adj_close,0)-1 END AS mae_{horizon}d",
            ]
        )
    return f"""
    WITH sequenced AS (
      SELECT CAST(Date AS DATE) AS Date,Ticker,Close,High,Low,"Adj Close",Volume,
        lag("Adj Close") OVER(PARTITION BY Ticker ORDER BY Date) AS prior_adj_close
      FROM step1
    ), validated AS (
      SELECT *,(Close>0 AND High>0 AND Low>0 AND "Adj Close">0 AND Volume>0
          AND Close<=1e8 AND High<=1e8 AND Low<=1e8 AND "Adj Close"<=1e8
          AND "Adj Close"/nullif(Close,0) BETWEEN 0.001 AND 1000
          AND (prior_adj_close IS NULL OR "Adj Close"/prior_adj_close BETWEEN 0.2 AND 5)
          ) AS price_valid
      FROM sequenced
    ), adjusted AS (
      SELECT Date,Ticker,
        CASE WHEN price_valid THEN "Adj Close" END AS adj_close,
        CASE WHEN price_valid THEN High*("Adj Close"/Close) END AS adj_high,
        CASE WHEN price_valid THEN Low*("Adj Close"/Close) END AS adj_low
      FROM validated
    ), future AS (
      SELECT *,{','.join(windows)} FROM adjusted
    )
    SELECT Date,Ticker,{','.join(selections)} FROM future
    """


def write_step3_prompt(path: Path) -> None:
    path.write_text(
        """# STEP3 only: large-move event extraction

STEP1の認証済み特徴量マスタとSTEP2の認証済み目的変数
`data/market_history/targets/equity_daily_forward_targets/` を入力にし、
STEP1・STEP2を再計算せずSTEP3「大相場イベント抽出」だけを実行してください。

40営業日以内に調整後終値ベースで+30%以上へ到達した事象を主定義として全件抽出し、
20日・60日窓と+20%・+50%の感度分析用定義も別列で保持してください。
同一銘柄でイベント期間が重なる場合は勝手に削除せず、元の候補行と、重複をまとめた
独立イベントIDの両方を保存してください。イベント定義に使用する未来情報をSTEP1特徴量へ
混ぜないでください。現行シグナル・売買戦略・閾値は使用または変更しないでください。

新規Parquetへ保存し、定義別件数、銘柄数、期間別件数、重複イベント数、欠損、異常値、
容量、品質、残存リスクを報告してください。完了時はSTEP3成果物を再計算不要と明記し、
STEP4「大相場前20営業日から当日までの共通特徴分析」だけの次回プロンプトを保存してください。
""",
        encoding="utf-8",
    )


def main() -> int:
    options = parse_args()
    repository = Path(__file__).resolve().parents[2]
    root = (repository / "momentum5d" / options.root).resolve()
    source = root / "features/equity_daily_features"
    step1_report_path = root / "quality/step1_report.json"
    source_files = sorted(source.rglob("*.parquet"))
    if not source_files or not step1_report_path.exists():
        raise RuntimeError("certified STEP1 inputs are incomplete")
    step1_report = json.loads(step1_report_path.read_text(encoding="utf-8"))
    if step1_report.get("quality") != "PASS" or step1_report.get("step") != 1:
        raise RuntimeError("STEP1 report is not certified PASS")

    source_hash = fingerprint(source_files)
    target = root / "targets/equity_daily_forward_targets"
    report_path = root / "quality/step2_report.json"
    prompt_path = root / "quality/STEP3_PROMPT.md"
    if report_path.exists() and target.exists() and not options.force:
        prior = json.loads(report_path.read_text(encoding="utf-8"))
        if prior.get("input_fingerprint") == source_hash and prior.get("quality") == "PASS":
            print(json.dumps({"status": "PASS", "reused": True}, ensure_ascii=False))
            return 0

    work = Path(tempfile.mkdtemp(prefix="step2_", dir=root.parent))
    stage = work / "output"
    stage.mkdir()
    database = work / "work.duckdb"
    connection = duckdb.connect(str(database))
    connection.execute(f"SET memory_limit='{options.memory_limit}'")
    connection.execute("SET threads=2")
    spill = str(work / "spill").replace("'", "''")
    connection.execute(f"SET temp_directory='{spill}'")
    connection.execute(f"CREATE VIEW step1 AS FROM read_parquet('{parquet_glob(source)}')")
    input_rows = connection.execute("SELECT count(*) FROM step1").fetchone()[0]
    input_tickers = connection.execute("SELECT count(DISTINCT Ticker) FROM step1").fetchone()[0]
    (
        source_invalid_rows,
        source_invalid_tickers,
        zero_volume_rows,
        invalid_price_scale_rows,
        discontinuity_rows,
    ) = connection.execute(
        """WITH sequenced AS (
          SELECT *,lag("Adj Close") OVER(PARTITION BY Ticker ORDER BY Date) prior_adj_close
          FROM step1
        ), flags AS (
          SELECT *,Volume<=0 AS zero_volume,
            (Close<=0 OR High<=0 OR Low<=0 OR "Adj Close"<=0
             OR Close>1e8 OR High>1e8 OR Low>1e8 OR "Adj Close">1e8
             OR "Adj Close"/nullif(Close,0) NOT BETWEEN 0.001 AND 1000) AS invalid_scale,
            (prior_adj_close IS NOT NULL
             AND "Adj Close"/prior_adj_close NOT BETWEEN 0.2 AND 5) AS discontinuity
          FROM sequenced
        ) SELECT count(*) FILTER(WHERE zero_volume OR invalid_scale OR discontinuity),
          count(DISTINCT Ticker) FILTER(WHERE zero_volume OR invalid_scale OR discontinuity),
          count(*) FILTER(WHERE zero_volume),count(*) FILTER(WHERE invalid_scale),
          count(*) FILTER(WHERE discontinuity) FROM flags"""
    ).fetchone()
    output_file = str(stage / "part.parquet").replace("'", "''")
    connection.execute(
        f"COPY ({target_sql()}) TO '{output_file}' (FORMAT PARQUET,COMPRESSION ZSTD)"
    )
    connection.close()

    connection = duckdb.connect()
    connection.execute(f"SET memory_limit='{options.memory_limit}'")
    connection.execute(f"CREATE VIEW output AS FROM read_parquet('{parquet_glob(stage)}')")
    summary = connection.execute(
        """SELECT count(*),count(DISTINCT Ticker),count(DISTINCT Date),min(Date),max(Date),
          count(*)-count(DISTINCT (Ticker,Date)),
          sum((Ticker IS NULL OR Date IS NULL)::INT)
        FROM output"""
    ).fetchone()
    rows, tickers, days, oldest, latest, duplicates, key_missing = summary

    period_metrics: dict[str, dict[str, Any]] = {}
    integrity_errors = 0
    nonfinite = 0
    extreme_rows: set[tuple[str, str]] = set()
    for horizon in HORIZONS:
        forward = f"forward_return_{horizon}d"
        mfe = f"mfe_{horizon}d"
        mae = f"mae_{horizon}d"
        metrics = connection.execute(
            f"""SELECT
              count({forward}),count(*)-count({forward}),
              count(*) FILTER(WHERE ({forward} IS NULL)!=( {mfe} IS NULL)
                                   OR ({forward} IS NULL)!=( {mae} IS NULL)),
              count(*) FILTER(WHERE {mae}>{forward}+1e-5 OR {forward}>{mfe}+1e-5
                                   OR {mae}>{mfe}+1e-5),
              count(*) FILTER(WHERE NOT isfinite({forward}) OR NOT isfinite({mfe})
                                   OR NOT isfinite({mae})),
              count(*) FILTER(WHERE abs({forward})>3 OR {mfe}>5 OR {mae}<-0.95),
              min({forward}),max({forward}),min({mae}),max({mfe})
            FROM output"""
        ).fetchone()
        calculable, missing, mismatch, ordering, invalid_finite, extremes, min_ret, max_ret, min_mae, max_mfe = metrics
        expected_tail_missing = connection.execute(
            f"""SELECT sum(least({horizon},ticker_rows)) FROM
            (SELECT count(*) AS ticker_rows FROM output GROUP BY Ticker)"""
        ).fetchone()[0]
        integrity_errors += int(mismatch or 0) + int(ordering or 0)
        nonfinite += int(invalid_finite or 0)
        period_metrics[str(horizon)] = {
            "calculable_rows": int(calculable),
            "missing_rows": int(missing),
            "missing_rate": float(missing / rows),
            "expected_tail_missing_rows": int(expected_tail_missing),
            "invalid_source_affected_rows": int(missing - expected_tail_missing),
            "triplet_missing_mismatch_rows": int(mismatch or 0),
            "ordering_error_rows": int(ordering or 0),
            "nonfinite_rows": int(invalid_finite or 0),
            "extreme_rows": int(extremes or 0),
            "forward_return_min": float(min_ret) if min_ret is not None else None,
            "forward_return_max": float(max_ret) if max_ret is not None else None,
            "mae_min": float(min_mae) if min_mae is not None else None,
            "mfe_max": float(max_mfe) if max_mfe is not None else None,
        }
        for ticker, date in connection.execute(
            f"SELECT Ticker,Date FROM output WHERE abs({forward})>3 OR {mfe}>5 OR {mae}<-0.95"
        ).fetchall():
            extreme_rows.add((str(ticker), str(date)))

    expected_columns = 2 + len(HORIZONS) * 3
    actual_columns = len(connection.execute("DESCRIBE output").fetchall())
    connection.close()

    quality = "PASS"
    if (
        rows != input_rows
        or tickers != input_tickers
        or duplicates
        or key_missing
        or integrity_errors
        or nonfinite
        or actual_columns != expected_columns
    ):
        quality = "FAIL"

    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(target)
    shutil.rmtree(work, ignore_errors=True)

    report: dict[str, Any] = {
        "step": 2,
        "status": "STEP2 complete" if quality == "PASS" else "STEP2 FAIL",
        "quality": quality,
        "created_at_jst": datetime.now(JST).isoformat(),
        "input_fingerprint": source_hash,
        "step1_input_fingerprint": step1_report.get("input_fingerprint"),
        "step1_recalculated": False,
        "step1_recalculation_forbidden": True,
        "future_values_in_feature_columns": False,
        "target_columns_only": True,
        "calculation_basis": "per-ticker trading-row shift, not calendar-day shift",
        "rows_deleted": 0,
        "source_rows_retained": rows == input_rows,
        "insufficient_future_rows_imputed": False,
        "split_delisted_missing_rows_deleted": False,
        "period": {"oldest": str(oldest), "latest": str(latest)},
        "tickers": int(tickers),
        "trading_days": int(days),
        "rows": int(rows),
        "target_count": len(HORIZONS) * 3,
        "period_metrics": period_metrics,
        "duplicate_rows": int(duplicates),
        "key_missing_rows": int(key_missing or 0),
        "invalid_source_rows": int(source_invalid_rows),
        "invalid_source_tickers": int(source_invalid_tickers),
        "invalid_source_reason_counts": {
            "zero_volume_rows": int(zero_volume_rows),
            "invalid_price_or_adjustment_scale_rows": int(invalid_price_scale_rows),
            "one_day_discontinuity_rows": int(discontinuity_rows),
        },
        "integrity_error_rows": int(integrity_errors),
        "nonfinite_value_count": int(nonfinite),
        "extreme_unique_rows": len(extreme_rows),
        "input_output_row_parity": rows == input_rows,
        "input_output_ticker_parity": tickers == input_tickers,
        "save_path": str(target),
        "file_size_bytes": directory_size(target),
        "definitions": {
            "entry_price": "current adjusted close",
            "forward_return_Nd": "adjusted close at t+N divided by adjusted close at t minus 1",
            "mfe_Nd": "maximum split-adjusted intraday high over t+1..t+N divided by entry minus 1",
            "mae_Nd": "minimum split-adjusted intraday low over t+1..t+N divided by entry minus 1",
            "availability": "all N future trading observations must exist; otherwise all three labels are NULL",
            "source_validity": "positive finite-scale adjusted OHLC, positive volume, and no >5x/<0.2x one-day discontinuity",
        },
        "label_only_schema": actual_columns == expected_columns,
        "feature_columns_modified": [],
        "strategy_files_modified": [],
        "signal_conditions_created": False,
        "recalculation_required": quality != "PASS",
        "completion_statement": (
            "STEP2で作成した成果物は今後再計算不要" if quality == "PASS" else "STEP2再実行が必要"
        ),
        "created_files": [str(target), str(report_path), str(prompt_path)],
        "residual_risks": [
            "the last 5/10/20/40/60 observations per ticker are intentionally unavailable by horizon",
            "delisted securities absent from the certified STEP1 universe cannot receive targets",
            "MFE and MAE use daily adjusted OHLC and do not reconstruct intraday execution order",
            "extreme adjusted returns are reported for review and are not silently removed",
            "invalid certified source-price rows are retained by key but their affected targets are NULL",
        ],
    }
    root.joinpath("quality").mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_step3_prompt(prompt_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if quality == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
