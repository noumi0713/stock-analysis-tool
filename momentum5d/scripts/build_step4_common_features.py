from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


JST = ZoneInfo("Asia/Tokyo")
BUCKETS = ((-20, -11, "d-20_to_d-11"), (-10, -6, "d-10_to_d-6"), (-5, -1, "d-5_to_d-1"), (0, 0, "d0"))
FORBIDDEN_TOKENS = ("future", "forward", "mfe", "mae", "target", "label", "event", "peak_date")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STEP4 pre-event common-feature analysis")
    parser.add_argument("--root", default="data/market_history")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--memory-limit", default="3GB")
    return parser.parse_args()


def parquet_glob(path: Path) -> str:
    return str(path.resolve() / "**" / "*.parquet").replace("'", "''")


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def fingerprint(groups: list[tuple[str, list[Path]]]) -> str:
    digest = hashlib.sha256()
    for label, files in groups:
        digest.update(label.encode())
        for path in sorted(files):
            digest.update(str(path.relative_to(path.parents[3]) if len(path.parents) > 3 else path.name).encode())
            digest.update(str(path.stat().st_size).encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def safe_float(value: Any) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def baseline_for(column: str, is_boolean: bool) -> float | None:
    name = column.lower()
    if is_boolean:
        return 0.5
    if "advancer_ratio" in name or "decliner_ratio" in name or "close_location" in name:
        return 0.5
    if any(token in name for token in ("volume_ratio", "trading_value_ratio", "contraction_ratio")):
        return 1.0
    if any(token in name for token in ("return", "change", "deviation", "slope", "distance", "vs_ma20", "strength", "gap_rate")):
        return 0.0
    return None


def distribution(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    finite = numeric[np.isfinite(numeric)]
    if not len(finite):
        return {key: None for key in ("mean", "std", "min", "p05", "p25", "median", "p75", "p95", "max", "iqr")}
    q05, q25, q50, q75, q95 = np.quantile(finite, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "mean": safe_float(np.mean(finite)),
        "std": safe_float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "min": safe_float(np.min(finite)),
        "p05": safe_float(q05),
        "p25": safe_float(q25),
        "median": safe_float(q50),
        "p75": safe_float(q75),
        "p95": safe_float(q95),
        "max": safe_float(np.max(finite)),
        "iqr": safe_float(q75 - q25),
    }


def summarize_daily(frame: pd.DataFrame, features: list[str], boolean_columns: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    event_total = frame["event_id"].nunique()
    for relative_day, daily in frame.groupby("relative_day", sort=True):
        for feature in features:
            present = int(daily[feature].notna().sum())
            row = {
                "relative_day": int(relative_day),
                "feature": feature,
                "feature_type": "boolean" if feature in boolean_columns else "numeric",
                "row_count": int(len(daily)),
                "non_null_count": present,
                "missing_count": int(len(daily) - present),
                "missing_rate": float(1 - present / len(daily)) if len(daily) else None,
                "event_coverage_rate": float(daily.loc[daily[feature].notna(), "event_id"].nunique() / event_total),
            }
            row.update(distribution(daily[feature]))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_buckets(frame: pd.DataFrame, features: list[str], boolean_columns: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_medians: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    event_total = frame["event_id"].nunique()
    for start, end, label in BUCKETS:
        subset = frame.loc[frame["relative_day"].between(start, end), ["event_id", *features]].copy()
        medians = subset.groupby("event_id", sort=False)[features].median(numeric_only=True).reset_index()
        medians.insert(1, "bucket", label)
        event_medians.append(medians)
        for feature in features:
            present = int(medians[feature].notna().sum())
            stats = distribution(medians[feature])
            baseline = baseline_for(feature, feature in boolean_columns)
            consistency = None
            if baseline is not None and present:
                values = pd.to_numeric(medians[feature], errors="coerce").dropna()
                median = stats["median"]
                if median is not None:
                    consistency = float((values >= baseline).mean() if median >= baseline else (values <= baseline).mean())
            summary_rows.append({
                "bucket": label,
                "feature": feature,
                "feature_type": "boolean" if feature in boolean_columns else "numeric",
                "event_count": event_total,
                "events_with_value": present,
                "missing_events": int(event_total - present),
                "missing_rate": float(1 - present / event_total),
                "direction_reference": baseline,
                "directional_consistency_rate": consistency,
                **stats,
            })
    return pd.DataFrame(summary_rows), pd.concat(event_medians, ignore_index=True)


def categorical_summaries(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in columns:
        for relative_day, daily in frame.groupby("relative_day", sort=True):
            values = daily[column].fillna("<MISSING>").astype(str)
            counts = values.value_counts(dropna=False)
            total = int(len(values))
            for rank, (category, count) in enumerate(counts.items(), start=1):
                rows.append({"scope": "relative_day", "position": str(int(relative_day)), "feature": column,
                             "category": category, "rank": rank, "count": int(count), "share": float(count / total)})
        for start, end, label in BUCKETS:
            values = frame.loc[frame["relative_day"].between(start, end)].groupby("event_id", sort=False)[column].first()
            values = values.fillna("<MISSING>").astype(str)
            counts = values.value_counts(dropna=False)
            total = int(len(values))
            for rank, (category, count) in enumerate(counts.items(), start=1):
                rows.append({"scope": "bucket", "position": label, "feature": column,
                             "category": category, "rank": rank, "count": int(count), "share": float(count / total)})
    return pd.DataFrame(rows)


def make_visualizations(frame: pd.DataFrame, output: Path) -> list[str]:
    plt.style.use("seaborn-v0_8-whitegrid")
    created: list[str] = []
    coverage = frame.groupby("relative_day")["event_id"].nunique().reindex(range(-20, 1), fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(coverage.index, coverage.values, marker="o", linewidth=2)
    ax.set(title="Event coverage before large-move start", xlabel="Trading-day position", ylabel="Events")
    ax.set_xticks(range(-20, 1, 2))
    fig.tight_layout()
    path = output / "event_coverage_by_relative_day.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    created.append(str(path))

    groups = {
        "price_trend": ["return_20d", "ma_25_deviation", "distance_from_52w_high", "rsi_14"],
        "volume_sector": ["volume_ratio_20d", "trading_value_ratio_20d", "sector_return_20d", "equity_vs_sector_strength_20d"],
        "market_breadth": ["market_advancer_ratio", "market_new_high_20d_ratio", "advancing_trading_value_ratio", "index_topix_vs_ma20"],
        "external_environment": ["index_tse_growth_250_vs_ma20", "external_sp500_vs_ma20", "external_vix_vs_ma20", "external_us10y_change_20d"],
    }
    for group, columns in groups.items():
        available = [column for column in columns if column in frame and frame[column].notna().any()]
        if not available:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
        for ax, column in zip(axes.flat, available, strict=False):
            grouped = frame.groupby("relative_day")[column]
            median = grouped.median().reindex(range(-20, 1))
            q25 = grouped.quantile(0.25).reindex(range(-20, 1))
            q75 = grouped.quantile(0.75).reindex(range(-20, 1))
            x = median.index.to_numpy(dtype=float)
            ax.plot(x, median.to_numpy(dtype=float), linewidth=2, label="Median")
            ax.fill_between(x, q25.to_numpy(dtype=float), q75.to_numpy(dtype=float), alpha=0.2, label="IQR")
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(column)
            ax.set_xticks(range(-20, 1, 5))
        for ax in axes.flat[len(available):]:
            ax.axis("off")
        fig.suptitle(f"Pre-event profiles: {group.replace('_', ' ')}")
        fig.supxlabel("Trading-day position")
        fig.tight_layout()
        path = output / f"pre_event_{group}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        created.append(str(path))
    return created


def write_step5_prompt(path: Path) -> None:
    path.write_text("""# STEP5 only: compare large-move events with controls

STEP1の認証済み特徴量、STEP3の認証済みイベントラベル、STEP4の認証済みイベント前特徴を入力にし、
STEP1〜STEP4を再計算せず、STEP5「非大相場との対照比較」だけを実行してください。

主定義40営業日+30%以上の独立イベントに対し、同一銘柄・近接時期を優先した非イベント日を、
将来ラベルを対照群選定のみに使用してマッチングしてください。対照日は主イベントと十分離し、
同じ対照日の重複使用、マッチ不能件数、時期差、銘柄差を報告してください。

基準日前20営業日〜当日のSTEP1特徴量について、イベント群と対照群の中央値差、標準化差、
効果量、欠損差、年別再現性を比較してください。未来リターン・MFE・MAE・イベントラベルを
説明変数へ混ぜず、シグナル条件、閾値、売買ルールは作成しないでください。

新規Parquetとレポートを保存し、品質と残存リスクを報告してください。完了時はSTEP5成果物を
再計算不要と明記し、STEP6「単変量分析」だけの次回プロンプトを保存してください。
""", encoding="utf-8")


def main() -> int:
    options = parse_args()
    repository = Path(__file__).resolve().parents[2]
    root = (repository / "momentum5d" / options.root).resolve()
    features_dir = root / "features/equity_daily_features"
    events_dir = root / "targets/large_move_events/independent_events"
    step1_report_path = root / "quality/step1_report.json"
    step3_report_path = root / "quality/step3_report.json"
    feature_files = sorted(features_dir.rglob("*.parquet"))
    event_files = sorted(events_dir.rglob("*.parquet"))
    if not feature_files or not event_files or not step1_report_path.exists() or not step3_report_path.exists():
        raise RuntimeError("certified STEP1/STEP3 inputs are incomplete")
    step1_report = json.loads(step1_report_path.read_text(encoding="utf-8"))
    step3_report = json.loads(step3_report_path.read_text(encoding="utf-8"))
    if step1_report.get("quality") != "PASS" or step3_report.get("quality") != "PASS":
        raise RuntimeError("STEP1 and STEP3 reports must both be certified PASS")

    input_hash = fingerprint([("step1", feature_files), ("step3", event_files)])
    target = root / "analysis/step4_pre_event_common_features"
    report_path = root / "quality/step4_report.json"
    prompt_path = root / "quality/STEP5_PROMPT.md"
    if report_path.exists() and target.exists() and not options.force:
        prior = json.loads(report_path.read_text(encoding="utf-8"))
        if prior.get("input_fingerprint") == input_hash and prior.get("quality") == "PASS":
            print(json.dumps({"status": "PASS", "reused": True}, ensure_ascii=False))
            return 0

    work = Path(tempfile.mkdtemp(prefix="step4_", dir=root.parent))
    stage = work / "step4_pre_event_common_features"
    windows_dir = stage / "pre_event_windows"
    daily_dir = stage / "daily_feature_summary"
    bucket_dir = stage / "bucket_feature_summary"
    event_bucket_dir = stage / "event_bucket_medians"
    categorical_dir = stage / "categorical_summary"
    visuals_dir = stage / "visualizations"
    for directory in (windows_dir, daily_dir, bucket_dir, event_bucket_dir, categorical_dir, visuals_dir):
        directory.mkdir(parents=True)

    connection = duckdb.connect(str(work / "work.duckdb"))
    connection.execute(f"SET memory_limit='{options.memory_limit}'")
    connection.execute("SET threads=2")
    schema = connection.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_glob(features_dir)}')").fetchall()
    columns = {name: dtype for name, dtype, *_ in schema}
    feature_columns = [name for name in columns if name not in {"Date", "Ticker"}]
    forbidden = [name for name in feature_columns if any(token in name.lower() for token in FORBIDDEN_TOKENS)]
    if forbidden:
        raise RuntimeError(f"future/target-like columns detected in STEP1: {forbidden}")
    numeric_columns = [name for name in feature_columns if any(token in columns[name] for token in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "BOOLEAN"))]
    boolean_columns = {name for name in feature_columns if columns[name] == "BOOLEAN"}
    categorical_columns = [name for name in feature_columns if columns[name] == "VARCHAR"]
    selected = ",".join(f"f.{quote(name)}" for name in feature_columns)
    connection.execute(f"CREATE VIEW features AS SELECT *,row_number() OVER(PARTITION BY Ticker ORDER BY Date)-1 AS row_position FROM read_parquet('{parquet_glob(features_dir)}')")
    connection.execute(f"CREATE VIEW events AS FROM read_parquet('{parquet_glob(events_dir)}')")
    query = f"""
      WITH anchors AS (
        SELECT e.event_id,e.Ticker,e.event_start_date,f.row_position AS anchor_position
        FROM events e LEFT JOIN features f ON e.Ticker=f.Ticker AND e.event_start_date=f.Date
      )
      SELECT a.event_id,a.Ticker,a.event_start_date,f.Date AS observation_date,
        CAST(f.row_position-a.anchor_position AS SMALLINT) AS relative_day,{selected}
      FROM anchors a JOIN features f ON a.Ticker=f.Ticker
        AND f.row_position BETWEEN a.anchor_position-20 AND a.anchor_position
      ORDER BY a.event_id,relative_day
    """
    windows_file = str(windows_dir / "part.parquet").replace("'", "''")
    connection.execute(f"COPY ({query}) TO '{windows_file}' (FORMAT PARQUET,COMPRESSION ZSTD)")
    input_events, duplicate_events = connection.execute("SELECT count(*),count(*)-count(DISTINCT event_id) FROM events").fetchone()
    anchor_missing = connection.execute("SELECT count(*) FROM events e LEFT JOIN features f ON e.Ticker=f.Ticker AND e.event_start_date=f.Date WHERE f.Date IS NULL").fetchone()[0]
    connection.close()

    frame = pd.read_parquet(windows_dir)
    for column in boolean_columns:
        frame[column] = frame[column].astype("Float64")
    daily_summary = summarize_daily(frame, numeric_columns, boolean_columns)
    bucket_summary, event_bucket = summarize_buckets(frame, numeric_columns, boolean_columns)
    categorical_summary = categorical_summaries(frame, categorical_columns)
    daily_summary.to_parquet(daily_dir / "part.parquet", index=False, compression="zstd")
    bucket_summary.to_parquet(bucket_dir / "part.parquet", index=False, compression="zstd")
    event_bucket.to_parquet(event_bucket_dir / "part.parquet", index=False, compression="zstd")
    categorical_summary.to_parquet(categorical_dir / "part.parquet", index=False, compression="zstd")
    visualizations = make_visualizations(frame, visuals_dir)

    counts = frame.groupby("event_id")["relative_day"].agg(["count", "min", "max"])
    joinable_events = int(frame["event_id"].nunique())
    full_window_events = int(((counts["count"] == 21) & (counts["min"] == -20) & (counts["max"] == 0)).sum())
    duplicate_window_rows = int(frame.duplicated(["event_id", "relative_day"]).sum())
    post_event_rows = int((frame["relative_day"] > 0).sum())
    date_order_errors = int((pd.to_datetime(frame["observation_date"]) > pd.to_datetime(frame["event_start_date"])).sum())
    numeric_values = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    nonfinite_cells = int(np.isinf(numeric_values.to_numpy(dtype=float, na_value=np.nan)).sum())
    missing_cells = int(numeric_values.isna().sum().sum())
    total_numeric_cells = int(numeric_values.shape[0] * numeric_values.shape[1])
    missing_by_feature = numeric_values.isna().mean().sort_values(ascending=False)
    all_null_features = [name for name, value in missing_by_feature.items() if value == 1.0]
    top_missing = {name: float(value) for name, value in missing_by_feature.head(20).items()}
    outliers: dict[str, int] = {}
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna().astype(float)
        if values.empty:
            continue
        q25, q75 = values.quantile([0.25, 0.75])
        iqr = q75 - q25
        if iqr > 0:
            count = int(((values < q25 - 10 * iqr) | (values > q75 + 10 * iqr)).sum())
            if count:
                outliers[column] = count
    top_outliers = dict(sorted(outliers.items(), key=lambda item: item[1], reverse=True)[:20])
    events_by_year = {str(int(year)): int(count) for year, count in frame.drop_duplicates("event_id").groupby(pd.to_datetime(frame.drop_duplicates("event_id")["event_start_date"]).dt.year).size().items()}

    quality = "PASS"
    if duplicate_events or anchor_missing or joinable_events != input_events or duplicate_window_rows or post_event_rows or date_order_errors or nonfinite_cells:
        quality = "FAIL"

    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(target)
    shutil.rmtree(work, ignore_errors=True)
    root.joinpath("quality").mkdir(parents=True, exist_ok=True)
    write_step5_prompt(prompt_path)
    report = {
        "step": 4,
        "status": "STEP4 complete" if quality == "PASS" else "STEP4 FAIL",
        "quality": quality,
        "created_at_jst": datetime.now(JST).isoformat(),
        "input_fingerprint": input_hash,
        "step1_recalculated": False,
        "step2_recalculated": False,
        "step3_recalculated": False,
        "future_or_target_columns_in_features": forbidden,
        "post_event_rows": post_event_rows,
        "strategy_files_modified": [],
        "signal_conditions_created": False,
        "input_events": int(input_events),
        "joinable_events": joinable_events,
        "unjoinable_events": int(input_events - joinable_events),
        "full_21day_window_events": full_window_events,
        "partial_window_events": int(joinable_events - full_window_events),
        "pre_event_rows": int(len(frame)),
        "relative_day_range": [int(frame.relative_day.min()), int(frame.relative_day.max())],
        "event_start_period": {"oldest": str(frame.event_start_date.min()), "latest": str(frame.event_start_date.max())},
        "events_by_start_year": events_by_year,
        "step1_feature_columns": len(feature_columns),
        "numeric_boolean_features_summarized": len(numeric_columns),
        "categorical_features_summarized": categorical_columns,
        "daily_summary_rows": int(len(daily_summary)),
        "bucket_summary_rows": int(len(bucket_summary)),
        "event_bucket_rows": int(len(event_bucket)),
        "missing_numeric_cells": missing_cells,
        "numeric_cell_count": total_numeric_cells,
        "numeric_missing_rate": float(missing_cells / total_numeric_cells),
        "top_feature_missing_rates": top_missing,
        "all_null_numeric_boolean_features": all_null_features,
        "analytical_completeness": "PARTIAL" if all_null_features else "COMPLETE",
        "duplicate_input_event_ids": int(duplicate_events),
        "duplicate_event_relative_day_rows": duplicate_window_rows,
        "date_order_errors": date_order_errors,
        "nonfinite_cells": nonfinite_cells,
        "extreme_outlier_definition": "outside Q1 +/- 10*IQR; retained and not a quality failure",
        "features_with_extreme_outliers": len(outliers),
        "top_extreme_outlier_counts": top_outliers,
        "consistency_definition": "share of event-level bucket medians on the same side of a documented neutral reference as the cross-event median; null when no defensible neutral reference exists",
        "save_path": str(target),
        "file_size_bytes": directory_size(target),
        "visualizations": [str(target / "visualizations" / Path(path).name) for path in visualizations],
        "completion_statement": "STEP4で作成した成果物は今後再計算不要" if quality == "PASS" else "STEP4再実行が必要",
        "recalculation_required": quality != "PASS",
        "created_files": [str(target / name) for name in ("pre_event_windows", "daily_feature_summary", "bucket_feature_summary", "event_bucket_medians", "categorical_summary", "visualizations")] + [str(report_path), str(prompt_path)],
        "residual_risks": [
            "STEP4 is event-only descriptive analysis; predictive value cannot be inferred before STEP5 controls",
            "survivorship and historical-sector-classification biases inherited from certified STEP1 remain",
            "partial windows occur for events near the beginning of ticker histories and are retained",
            "extreme values are retained; split adjustments, ticker reuse, and provider errors require later sensitivity checks",
            "multiple events from the same ticker are not statistically independent",
        ],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if quality == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
