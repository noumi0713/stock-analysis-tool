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
HORIZONS = (20, 40, 60)
THRESHOLDS = (20, 30, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STEP3 large-move event extraction")
    parser.add_argument("--root", default="data/market_history")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--memory-limit", default="2GB")
    return parser.parse_args()


def parquet_glob(path: Path) -> str:
    return str(path.resolve() / "**" / "*.parquet").replace("'", "''")


def fingerprint(groups: list[tuple[str, list[Path]]]) -> str:
    digest = hashlib.sha256()
    for label, files in groups:
        digest.update(label.encode())
        for path in sorted(files):
            digest.update(path.name.encode())
            digest.update(str(path.stat().st_size).encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def event_label_sql() -> str:
    windows = []
    for horizon in HORIZONS:
        frame = (
            f"PARTITION BY s.Ticker ORDER BY s.Date "
            f"ROWS BETWEEN 1 FOLLOWING AND {horizon} FOLLOWING"
        )
        windows.extend(
            [
                f'max(s."Adj Close") OVER ({frame}) AS future_close_max_{horizon}d',
                f'max_by(s.Date,s."Adj Close") OVER ({frame}) AS future_close_peak_date_{horizon}d',
            ]
        )
    returns = []
    flags = []
    for horizon in HORIZONS:
        returns.extend(
            [
                f"CASE WHEN mfe_{horizon}d IS NOT NULL THEN "
                f"future_close_max_{horizon}d/nullif(entry_adjusted_close,0)-1 END "
                f"AS max_forward_close_return_{horizon}d",
                f"CASE WHEN mfe_{horizon}d IS NOT NULL THEN "
                f"future_close_peak_date_{horizon}d END AS peak_date_{horizon}d",
            ]
        )
        for threshold in THRESHOLDS:
            flags.append(
                f"CASE WHEN max_forward_close_return_{horizon}d IS NULL THEN NULL ELSE "
                f"max_forward_close_return_{horizon}d>={threshold / 100} END "
                f"AS event_{horizon}d_ge_{threshold}pct"
            )
    return f"""
    WITH windowed AS (
      SELECT s.Date,s.Ticker,s."Adj Close" AS entry_adjusted_close,
        {','.join(windows)},
        t.mfe_20d,t.mfe_40d,t.mfe_60d
      FROM step1 s JOIN step2 t USING(Date,Ticker)
    ), returns AS (
      SELECT Date,Ticker,entry_adjusted_close,{','.join(returns)} FROM windowed s
    )
    SELECT *,{','.join(flags)} FROM returns
    """


def candidate_sql() -> str:
    flag_columns = ",".join(
        f"event_{horizon}d_ge_{threshold}pct"
        for horizon in HORIZONS
        for threshold in THRESHOLDS
    )
    return f"""
    WITH main AS (
      SELECT Date,Ticker,entry_adjusted_close,
        max_forward_close_return_20d,max_forward_close_return_40d,
        max_forward_close_return_60d,peak_date_20d,peak_date_40d,peak_date_60d,
        {flag_columns},
        Date AS candidate_start_date,peak_date_40d AS candidate_end_date
      FROM labels WHERE event_40d_ge_30pct
    ), running AS (
      SELECT *,max(candidate_end_date) OVER(
        PARTITION BY Ticker ORDER BY candidate_start_date,candidate_end_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
      ) AS prior_running_end
      FROM main
    ), islands AS (
      SELECT *,CASE WHEN prior_running_end IS NULL OR candidate_start_date>prior_running_end
        THEN 1 ELSE 0 END AS new_event
      FROM running
    ), numbered AS (
      SELECT *,sum(new_event) OVER(
        PARTITION BY Ticker ORDER BY candidate_start_date,candidate_end_date
        ROWS UNBOUNDED PRECEDING
      ) AS event_sequence
      FROM islands
    )
    SELECT Date,Ticker,entry_adjusted_close,
      max_forward_close_return_20d,max_forward_close_return_40d,
      max_forward_close_return_60d,peak_date_20d,peak_date_40d,peak_date_60d,
      {flag_columns},candidate_start_date,candidate_end_date,
      Ticker||'-'||lpad(event_sequence::VARCHAR,6,'0') AS event_id
    FROM numbered
    """


def independent_event_sql() -> str:
    return """
    SELECT event_id,Ticker,min(candidate_start_date) AS event_start_date,
      max(candidate_end_date) AS event_end_date,count(*) AS candidate_row_count,
      arg_max(Date,max_forward_close_return_40d) AS strongest_candidate_date,
      max(max_forward_close_return_40d) AS max_forward_close_return_40d,
      max(max_forward_close_return_20d) AS max_forward_close_return_20d,
      max(max_forward_close_return_60d) AS max_forward_close_return_60d,
      bool_or(event_20d_ge_20pct) AS any_20d_ge_20pct,
      bool_or(event_20d_ge_30pct) AS any_20d_ge_30pct,
      bool_or(event_20d_ge_50pct) AS any_20d_ge_50pct,
      bool_or(event_40d_ge_20pct) AS any_40d_ge_20pct,
      bool_or(event_40d_ge_30pct) AS any_40d_ge_30pct,
      bool_or(event_40d_ge_50pct) AS any_40d_ge_50pct,
      bool_or(event_60d_ge_20pct) AS any_60d_ge_20pct,
      bool_or(event_60d_ge_30pct) AS any_60d_ge_30pct,
      bool_or(event_60d_ge_50pct) AS any_60d_ge_50pct
    FROM candidates GROUP BY event_id,Ticker
    """


def write_step4_prompt(path: Path) -> None:
    path.write_text(
        """# STEP4 only: pre-event common-feature analysis

STEP1の認証済み特徴量マスタとSTEP3の認証済み独立イベント表を入力にし、
STEP1〜STEP3を再計算せず、STEP4「大相場前の共通特徴分析」だけを実行してください。

各独立イベントの `event_start_date` を基準日0として、銘柄ごとの営業日順で
-20営業日から当日までのSTEP1特徴量を結合してください。基準日より後の情報、
STEP2・STEP3の未来リターン、MFE、MAE、イベント到達後の情報を説明変数へ混ぜないでください。

日次位置（-20〜0）ごと、および-20〜-11、-10〜-6、-5〜-1、当日の区間ごとに、
株価・出来高・業種・市場内部・外部環境の分布、中央値、四分位範囲、欠損率、
イベント間の一貫性を集計してください。今回は共通特徴の記述と可視化だけを行い、
シグナル条件、閾値、売買ルールを作成しないでください。

新規Parquetと集計レポートを保存し、イベント数、結合可能件数、期間別件数、欠損、
異常値、容量、品質、残存リスクを報告してください。完了時はSTEP4成果物を再計算不要と
明記し、STEP5「非大相場との対照比較」だけの次回プロンプトを保存してください。
""",
        encoding="utf-8",
    )


def main() -> int:
    options = parse_args()
    repository = Path(__file__).resolve().parents[2]
    root = (repository / "momentum5d" / options.root).resolve()
    step1_dir = root / "features/equity_daily_features"
    step2_dir = root / "targets/equity_daily_forward_targets"
    step1_report_path = root / "quality/step1_report.json"
    step2_report_path = root / "quality/step2_report.json"
    step1_files = sorted(step1_dir.rglob("*.parquet"))
    step2_files = sorted(step2_dir.rglob("*.parquet"))
    if not step1_files or not step2_files or not step1_report_path.exists() or not step2_report_path.exists():
        raise RuntimeError("certified STEP1/STEP2 inputs are incomplete")
    step1_report = json.loads(step1_report_path.read_text(encoding="utf-8"))
    step2_report = json.loads(step2_report_path.read_text(encoding="utf-8"))
    if step1_report.get("quality") != "PASS" or step2_report.get("quality") != "PASS":
        raise RuntimeError("STEP1 and STEP2 reports must both be certified PASS")

    input_hash = fingerprint([("step1", step1_files), ("step2", step2_files)])
    target = root / "targets/large_move_events"
    report_path = root / "quality/step3_report.json"
    prompt_path = root / "quality/STEP4_PROMPT.md"
    if report_path.exists() and target.exists() and not options.force:
        prior = json.loads(report_path.read_text(encoding="utf-8"))
        if prior.get("input_fingerprint") == input_hash and prior.get("quality") == "PASS":
            print(json.dumps({"status": "PASS", "reused": True}, ensure_ascii=False))
            return 0

    work = Path(tempfile.mkdtemp(prefix="step3_", dir=root.parent))
    stage = work / "large_move_events"
    labels_dir = stage / "event_labels"
    candidates_dir = stage / "main_candidates"
    events_dir = stage / "independent_events"
    for directory in (labels_dir, candidates_dir, events_dir):
        directory.mkdir(parents=True)

    connection = duckdb.connect(str(work / "work.duckdb"))
    connection.execute(f"SET memory_limit='{options.memory_limit}'")
    connection.execute("SET threads=2")
    spill = str(work / "spill").replace("'", "''")
    connection.execute(f"SET temp_directory='{spill}'")
    connection.execute(f"CREATE VIEW step1 AS FROM read_parquet('{parquet_glob(step1_dir)}')")
    connection.execute(f"CREATE VIEW step2 AS FROM read_parquet('{parquet_glob(step2_dir)}')")
    input_summary = connection.execute(
        """SELECT (SELECT count(*) FROM step1),(SELECT count(*) FROM step2),
          (SELECT count(DISTINCT (Ticker,Date)) FROM step1),
          (SELECT count(DISTINCT (Ticker,Date)) FROM step2)"""
    ).fetchone()

    labels_file = str(labels_dir / "part.parquet").replace("'", "''")
    connection.execute(
        f"COPY ({event_label_sql()}) TO '{labels_file}' (FORMAT PARQUET,COMPRESSION ZSTD)"
    )
    connection.execute(f"CREATE VIEW labels AS FROM read_parquet('{parquet_glob(labels_dir)}')")
    candidates_file = str(candidates_dir / "part.parquet").replace("'", "''")
    connection.execute(
        f"COPY ({candidate_sql()}) TO '{candidates_file}' (FORMAT PARQUET,COMPRESSION ZSTD)"
    )
    connection.execute(
        f"CREATE VIEW candidates AS FROM read_parquet('{parquet_glob(candidates_dir)}')"
    )
    events_file = str(events_dir / "part.parquet").replace("'", "''")
    connection.execute(
        f"COPY ({independent_event_sql()}) TO '{events_file}' (FORMAT PARQUET,COMPRESSION ZSTD)"
    )
    connection.close()

    connection = duckdb.connect()
    connection.execute(f"SET memory_limit='{options.memory_limit}'")
    connection.execute(f"CREATE VIEW labels AS FROM read_parquet('{parquet_glob(labels_dir)}')")
    connection.execute(
        f"CREATE VIEW candidates AS FROM read_parquet('{parquet_glob(candidates_dir)}')"
    )
    connection.execute(f"CREATE VIEW events AS FROM read_parquet('{parquet_glob(events_dir)}')")
    label_summary = connection.execute(
        """SELECT count(*),count(DISTINCT Ticker),count(DISTINCT Date),min(Date),max(Date),
          count(*)-count(DISTINCT (Ticker,Date)) FROM labels"""
    ).fetchone()
    candidate_summary = connection.execute(
        """SELECT count(*),count(DISTINCT Ticker),count(DISTINCT event_id),
          count(*)-count(DISTINCT (Ticker,Date)),
          count(*) FILTER(WHERE event_id IS NULL OR candidate_end_date<candidate_start_date)
        FROM candidates"""
    ).fetchone()
    event_summary = connection.execute(
        """SELECT count(*),count(DISTINCT Ticker),count(*)-count(DISTINCT event_id),
          count(*) FILTER(WHERE event_end_date<event_start_date),
          count(*) FILTER(WHERE candidate_row_count>1),
          coalesce(sum(candidate_row_count-1),0)
        FROM events"""
    ).fetchone()

    definitions: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS:
        missing = connection.execute(
            f"SELECT count(*) FILTER(WHERE max_forward_close_return_{horizon}d IS NULL) FROM labels"
        ).fetchone()[0]
        for threshold in THRESHOLDS:
            flag = f"event_{horizon}d_ge_{threshold}pct"
            count, tickers = connection.execute(
                f"SELECT count(*) FILTER(WHERE {flag}),count(DISTINCT Ticker) FILTER(WHERE {flag}) FROM labels"
            ).fetchone()
            definitions[f"{horizon}d_ge_{threshold}pct"] = {
                "candidate_rows": int(count),
                "tickers": int(tickers),
                "missing_rows": int(missing),
                "missing_rate": float(missing / label_summary[0]),
            }

    main_by_year = {
        str(year): int(count)
        for year, count in connection.execute(
            "SELECT year(Date),count(*) FROM candidates GROUP BY 1 ORDER BY 1"
        ).fetchall()
    }
    independent_by_year = {
        str(year): int(count)
        for year, count in connection.execute(
            "SELECT year(event_start_date),count(*) FROM events GROUP BY 1 ORDER BY 1"
        ).fetchall()
    }
    nonfinite = connection.execute(
        """SELECT count(*) FROM labels WHERE
          NOT isfinite(max_forward_close_return_20d) OR
          NOT isfinite(max_forward_close_return_40d) OR
          NOT isfinite(max_forward_close_return_60d)"""
    ).fetchone()[0]
    monotonic_errors = connection.execute(
        """SELECT count(*) FROM labels WHERE
          max_forward_close_return_20d>max_forward_close_return_40d+1e-5 OR
          max_forward_close_return_40d>max_forward_close_return_60d+1e-5"""
    ).fetchone()[0]
    flag_errors = 0
    for horizon in HORIZONS:
        for threshold in THRESHOLDS:
            flag_errors += connection.execute(
                f"""SELECT count(*) FROM labels WHERE
                  event_{horizon}d_ge_{threshold}pct IS DISTINCT FROM
                  CASE WHEN max_forward_close_return_{horizon}d IS NULL THEN NULL ELSE
                    max_forward_close_return_{horizon}d>={threshold / 100} END"""
            ).fetchone()[0]
    extreme_rows, maximum_return = connection.execute(
        """SELECT count(*) FILTER(WHERE max_forward_close_return_60d>5),
          max(max_forward_close_return_60d) FROM labels"""
    ).fetchone()
    connection.close()

    rows, tickers, days, oldest, latest, label_duplicates = label_summary
    candidate_rows, candidate_tickers, independent_ids, candidate_duplicates, candidate_errors = candidate_summary
    event_rows, event_tickers, event_id_duplicates, event_interval_errors, multi_candidate_events, overlapping_rows = event_summary
    quality = "PASS"
    if (
        input_summary[0] != input_summary[1]
        or input_summary[0] != input_summary[2]
        or input_summary[1] != input_summary[3]
        or rows != input_summary[0]
        or label_duplicates
        or candidate_duplicates
        or candidate_errors
        or event_id_duplicates
        or event_interval_errors
        or independent_ids != event_rows
        or nonfinite
        or monotonic_errors
        or flag_errors
    ):
        quality = "FAIL"

    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(target)
    shutil.rmtree(work, ignore_errors=True)

    report: dict[str, Any] = {
        "step": 3,
        "status": "STEP3 complete" if quality == "PASS" else "STEP3 FAIL",
        "quality": quality,
        "created_at_jst": datetime.now(JST).isoformat(),
        "input_fingerprint": input_hash,
        "step1_recalculated": False,
        "step2_recalculated": False,
        "future_values_in_step1_features": False,
        "strategy_files_modified": [],
        "signal_conditions_created": False,
        "period": {"oldest": str(oldest), "latest": str(latest)},
        "tickers": int(tickers),
        "trading_days": int(days),
        "label_rows": int(rows),
        "definition_counts": definitions,
        "main_definition": "40 trading observations, maximum adjusted close return >= 30%",
        "main_candidate_rows": int(candidate_rows),
        "main_candidate_tickers": int(candidate_tickers),
        "independent_event_rows": int(event_rows),
        "independent_event_tickers": int(event_tickers),
        "events_with_overlapping_candidates": int(multi_candidate_events),
        "overlapping_candidate_rows": int(overlapping_rows),
        "main_candidates_by_year": main_by_year,
        "independent_events_by_start_year": independent_by_year,
        "input_row_parity": input_summary[0] == input_summary[1] == rows,
        "label_duplicate_rows": int(label_duplicates),
        "candidate_duplicate_rows": int(candidate_duplicates),
        "event_id_duplicate_rows": int(event_id_duplicates),
        "candidate_interval_errors": int(candidate_errors),
        "event_interval_errors": int(event_interval_errors),
        "nonfinite_rows": int(nonfinite),
        "horizon_monotonic_error_rows": int(monotonic_errors),
        "flag_consistency_errors": int(flag_errors),
        "extreme_60d_over_500pct_rows": int(extreme_rows),
        "maximum_60d_adjusted_close_return": float(maximum_return) if maximum_return is not None else None,
        "rows_deleted": 0,
        "save_path": str(target),
        "file_size_bytes": directory_size(target),
        "definitions": {
            "entry_price": "current adjusted close",
            "future_window": "per-ticker trading-row positions t+1 through t+N, not calendar days",
            "candidate_interval": "candidate date through the date of maximum adjusted close in t+1..t+40",
            "overlap_merge": "same-ticker candidate intervals are merged by interval overlap using a running maximum end date",
            "missing": "NULL inherited when STEP2 marks the full horizon unavailable; no future value is imputed",
        },
        "completion_statement": (
            "STEP3で作成した成果物は今後再計算不要" if quality == "PASS" else "STEP3再実行が必要"
        ),
        "recalculation_required": quality != "PASS",
        "created_files": [
            str(target / "event_labels"),
            str(target / "main_candidates"),
            str(target / "independent_events"),
            str(report_path),
            str(prompt_path),
        ],
        "residual_risks": [
            "extreme returns are retained and reported; corporate actions and ticker reuse still require review",
            "delisted securities absent from certified STEP1 cannot be reconstructed in STEP3",
            "overlap consolidation depends on the documented candidate-to-40d-peak interval definition",
            "the event labels intentionally use future adjusted closes and must never be used as live features",
        ],
    }
    root.joinpath("quality").mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_step4_prompt(prompt_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if quality == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
