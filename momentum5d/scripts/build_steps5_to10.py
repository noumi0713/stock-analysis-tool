from __future__ import annotations

import argparse
import bisect
import hashlib
import itertools
import json
import math
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score


JST = ZoneInfo("Asia/Tokyo")
FORMATION_END = pd.Timestamp("2023-12-31")
VALIDATION_END = pd.Timestamp("2025-09-07")
OOS_START = pd.Timestamp("2025-09-08")
BUCKET = "d-5_to_d-1"
ROUND_TRIP_COST = 0.004
STEP9_IMPLEMENTATION_VERSION = 2
STEP10_IMPLEMENTATION_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sequential research STEP5 through STEP10")
    parser.add_argument("--root", default="data/market_history")
    parser.add_argument("--to-step", type=int, default=10, choices=range(5, 11))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--memory-limit", default="4GB")
    return parser.parse_args()


def parquet_glob(path: Path) -> str:
    return str(path.resolve() / "**" / "*.parquet").replace("'", "''")


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def hash_inputs(groups: list[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for label, path in groups:
        digest.update(label.encode())
        for item in sorted(path.rglob("*.parquet") if path.is_dir() else [path]):
            digest.update(str(item).encode())
            digest.update(str(item.stat().st_size).encode())
            with item.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def report_ok(path: Path, step: int) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"STEP{step} report missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("quality") != "PASS":
        raise RuntimeError(f"STEP{step} is not certified PASS")
    return report


def finish_stage(stage: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(target)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_parquet_verified(frame: pd.DataFrame, path: Path) -> None:
    """Write inside an unpublished staging tree and read back before certification."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for stale in path.parent.glob("*.tmp"):
        stale.unlink()
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression="zstd")
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    parquet.close()
    if metadata.num_rows != len(frame):
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Parquet row-count mismatch for {path}")


def write_wide_parquet_verified(frame: pd.DataFrame, path: Path) -> None:
    """Use DuckDB for wide nullable frames; PyArrow showed unstable wide-file finalization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.register("wide_frame", frame)
    escaped = str(path.resolve()).replace("'", "''")
    connection.execute(f"COPY wide_frame TO '{escaped}' (FORMAT PARQUET,COMPRESSION ZSTD)")
    connection.unregister("wide_frame")
    connection.close()
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    parquet.close()
    if metadata.num_rows != len(frame):
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Parquet row-count mismatch for {path}")


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"event_id", "Ticker", "event_start_date", "observation_date", "relative_day", "control_start_date", "bucket"}
    return [column for column in frame.columns if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])]


def bucket_medians(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    definitions = ((-20, -11, "d-20_to_d-11"), (-10, -6, "d-10_to_d-6"), (-5, -1, BUCKET), (0, 0, "d0"))
    for start, end, label in definitions:
        values = frame.loc[frame.relative_day.between(start, end), ["event_id", *features]]
        medians = values.groupby("event_id", sort=False)[features].median(numeric_only=True).reset_index()
        medians.insert(1, "bucket", label)
        rows.append(medians)
    return pd.concat(rows, ignore_index=True)


def greedy_control_matches(events: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pools = {ticker: group.sort_values("row_position") for ticker, group in pool.groupby("Ticker", sort=False)}
    for ticker, ticker_events in events.groupby("Ticker", sort=False):
        candidates = pools.get(ticker)
        if candidates is None or candidates.empty:
            continue
        positions = candidates.row_position.astype(int).tolist()
        dates = pd.to_datetime(candidates.Date).tolist()
        years = [date.year for date in dates]
        used: set[int] = set()
        for event in ticker_events.sort_values("event_start_date").itertuples(index=False):
            event_pos = int(event.row_position)
            event_date = pd.Timestamp(event.event_start_date)
            order = sorted(range(len(positions)), key=lambda i: (years[i] != event_date.year, abs(positions[i] - event_pos), dates[i]))
            selected = next((index for index in order if index not in used), None)
            if selected is None:
                continue
            used.add(selected)
            rows.append({
                "event_id": event.event_id,
                "Ticker": ticker,
                "event_start_date": event_date,
                "event_row_position": event_pos,
                "control_start_date": dates[selected],
                "control_row_position": positions[selected],
                "trading_day_distance": abs(positions[selected] - event_pos),
                "same_calendar_year": years[selected] == event_date.year,
            })
    return pd.DataFrame(rows)


def comparison_summary(event_bucket: pd.DataFrame, control_bucket: pd.DataFrame, matches: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    match_dates = matches[["event_id", "event_start_date"]].copy()
    match_dates["event_year"] = pd.to_datetime(match_dates.event_start_date).dt.year
    rows: list[dict[str, Any]] = []
    for bucket in event_bucket.bucket.unique():
        event_part = event_bucket[event_bucket.bucket == bucket].merge(match_dates, on="event_id")
        control_part = control_bucket[control_bucket.bucket == bucket].merge(match_dates, on="event_id")
        for year in [None, *sorted(match_dates.event_year.unique())]:
            event_slice = event_part if year is None else event_part[event_part.event_year == year]
            control_slice = control_part if year is None else control_part[control_part.event_year == year]
            joined = event_slice[["event_id", *features]].merge(control_slice[["event_id", *features]], on="event_id", suffixes=("_event", "_control"))
            for feature in features:
                left = pd.to_numeric(joined[f"{feature}_event"], errors="coerce")
                right = pd.to_numeric(joined[f"{feature}_control"], errors="coerce")
                valid = left.notna() & right.notna() & np.isfinite(left) & np.isfinite(right)
                diff = (left[valid] - right[valid]).astype(float)
                if diff.empty:
                    mean_diff = median_diff = paired_d = sign_rate = None
                else:
                    mean_diff = finite(diff.mean())
                    median_diff = finite(diff.median())
                    sd = diff.std(ddof=1)
                    paired_d = finite(diff.mean() / sd) if sd and math.isfinite(sd) else None
                    nonzero = diff[diff != 0]
                    sign_rate = finite(max((nonzero > 0).mean(), (nonzero < 0).mean())) if len(nonzero) else None
                rows.append({
                    "bucket": bucket, "scope_year": "ALL" if year is None else str(int(year)), "feature": feature,
                    "matched_pairs": int(valid.sum()), "event_missing_rate": float(left.isna().mean()),
                    "control_missing_rate": float(right.isna().mean()), "event_median": finite(left[valid].median()) if valid.any() else None,
                    "control_median": finite(right[valid].median()) if valid.any() else None,
                    "mean_paired_difference": mean_diff, "median_paired_difference": median_diff,
                    "paired_standardized_effect": paired_d, "paired_direction_consistency": sign_rate,
                })
    return pd.DataFrame(rows)


def prompt_text(step: int) -> str:
    prompts = {
        6: """# STEP6 only: univariate analysis\n\nSTEP5の認証済みマッチド比較表を入力にし、STEP1〜STEP5を再計算せず、STEP6「単変量分析」だけを実行してください。各特徴を単独で評価し、形成期間と検証期間を分離してAUC、効果量、欠損、年別方向再現性、粗い分位別イベント率を保存してください。2025-09-08以降はSTEP11用OOSとして使用しないでください。閾値最適化やシグナル採用は行わず、新規Parquetと品質レポートを保存してください。完了時はSTEP6成果物を再計算不要と明記し、STEP7「2〜3条件組み合わせ検証」だけの次回プロンプトを保存してください。\n""",
        7: """# STEP7 only: coarse 2-3 condition combinations\n\nSTEP6の認証済み単変量分析を入力にし、STEP1〜STEP6を再計算せず、STEP7だけを実行してください。形成期間で固定した30/70分位の粗い条件のみを使い、相関の高い特徴を重複採用せず2〜3条件を比較してください。検証期間の件数、イベント率、リフト、再現性を評価し、2025-09-08以降のOOSには触れないでください。候補であって採用シグナルではないと明記し、新規Parquetと品質レポートを保存してください。完了時はSTEP7成果物を再計算不要と明記し、STEP8「相場環境別検証」だけの次回プロンプトを保存してください。\n""",
        8: """# STEP8 only: regime analysis\n\nSTEP7の認証済み粗条件候補を入力にし、STEP1〜STEP7を再計算せず、STEP8「相場環境別検証」だけを実行してください。日経平均トレンド、VIX方向、米10年金利方向、市場騰落幅など利用可能な事前情報だけで固定レジームを作り、候補の件数、イベント率、リフト、レジーム間格差を比較してください。欠損レジームを隠さず、OOSと売買ルールには触れないでください。完了時はSTEP8成果物を再計算不要と明記し、STEP9「エントリータイミング比較」だけの次回プロンプトを保存してください。\n""",
        9: """# STEP9 only: entry timing comparison\n\nSTEP7・STEP8の認証済み候補を入力にし、STEP1〜STEP8を再計算せず、STEP9だけを実行してください。形成期間で固定済みの候補を検証期間の全銘柄日へ適用し、当日終値（非実行可能ベンチマーク）、翌日始値、翌日終値、5日以内の指値押し目を同一の40営業日出口・同一コストで比較してください。約定率、件数、平均・中央値、勝率、PF、年別成績を保存し、OOSは使用しないでください。完了時はSTEP9成果物を再計算不要と明記し、STEP10「出口検証」だけの次回プロンプトを保存してください。\n""",
        10: """# STEP10 only: exit comparison\n\nSTEP9の認証済み実行可能エントリー候補を入力にし、STEP1〜STEP9を再計算せず、STEP10だけを実行してください。固定10/20/40日、固定利確・損切り、25MA割れ、3ATRトレールを同一シグナル・同一コストで比較し、期待値、勝率、PF、MFE、MAE、逐次取引DD、年別再現性を保存してください。日中に利確と損切りが同時到達した場合は損切り優先とし、OOSは使用しないでください。完了時はSTEP10成果物を再計算不要と明記し、STEP11「完全未使用期間でのOOS検証」だけの次回プロンプトを保存してください。\n""",
        11: """# STEP11 only: untouched out-of-sample validation\n\nSTEP10までに固定した候補定義・エントリー・出口を変更せず、2025-09-08以降の完全未使用期間だけでSTEP11 OOS検証を実行してください。OOS結果を見て閾値や条件を変更せず、期待値、PF、最大DD、取引回数、年・レジーム別成績、形成・検証期間からの劣化率を報告してください。結果が不十分なら不採用とし、再最適化しないでください。完了時はSTEP11成果物を再計算不要と明記し、STEP12「最終シグナル採否判定」だけの次回プロンプトを保存してください。\n""",
    }
    return prompts[step]


def build_step5(root: Path, memory_limit: str, force: bool) -> dict[str, Any]:
    report_path = root / "quality/step5_report.json"
    target = root / "analysis/step5_control_comparison"
    inputs = [("step1", root / "features/equity_daily_features"), ("step3_labels", root / "targets/large_move_events/event_labels"), ("step3_events", root / "targets/large_move_events/independent_events"), ("step4", root / "analysis/step4_pre_event_common_features")]
    input_hash = hash_inputs(inputs)
    if not force and report_path.exists() and target.exists():
        prior = json.loads(report_path.read_text())
        if prior.get("quality") == "PASS" and prior.get("input_fingerprint") == input_hash:
            return prior
    work = Path(tempfile.mkdtemp(prefix="step5_", dir=root.parent))
    stage = work / target.name
    for name in ("matched_pairs", "control_pre_event_windows", "control_bucket_medians", "feature_comparison"):
        (stage / name).mkdir(parents=True)
    features_dir = root / "features/equity_daily_features"
    labels_dir = root / "targets/large_move_events/event_labels"
    events_dir = root / "targets/large_move_events/independent_events"
    event_windows = pd.read_parquet(root / "analysis/step4_pre_event_common_features/pre_event_windows")
    for column in event_windows.select_dtypes(include="boolean").columns:
        event_windows[column] = event_windows[column].astype("Float64")
    features = numeric_feature_columns(event_windows)
    con = duckdb.connect(str(work / "work.duckdb"))
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute("SET threads=2")
    con.execute(f"CREATE VIEW f0 AS FROM read_parquet('{parquet_glob(features_dir)}')")
    con.execute(f"CREATE VIEW labels AS FROM read_parquet('{parquet_glob(labels_dir)}')")
    con.execute(f"CREATE VIEW events AS FROM read_parquet('{parquet_glob(events_dir)}')")
    con.execute("CREATE TABLE positions AS SELECT Date,Ticker,row_number() OVER(PARTITION BY Ticker ORDER BY Date)-1 AS row_position FROM f0")
    con.execute("CREATE TABLE anchors AS SELECT e.event_id,e.Ticker,e.event_start_date,p.row_position FROM events e JOIN positions p ON e.Ticker=p.Ticker AND e.event_start_date=p.Date")
    event_anchors = con.execute("SELECT * FROM anchors ORDER BY Ticker,event_start_date").df()
    pool = con.execute("""
      WITH marked AS (
        SELECT p.*,max(CASE WHEN a.event_id IS NULL THEN 0 ELSE 1 END) OVER(
          PARTITION BY p.Ticker ORDER BY p.row_position ROWS BETWEEN 60 PRECEDING AND 60 FOLLOWING
        ) AS near_event
        FROM positions p LEFT JOIN anchors a ON p.Ticker=a.Ticker AND p.Date=a.event_start_date
      )
      SELECT m.Date,m.Ticker,m.row_position FROM marked m JOIN labels l USING(Date,Ticker)
      WHERE m.near_event=0 AND l.event_40d_ge_30pct=false AND l.max_forward_close_return_40d IS NOT NULL
    """).df()
    matches = greedy_control_matches(event_anchors, pool)
    write_parquet_verified(matches, stage / "matched_pairs/part.parquet")
    con.register("matches", matches)
    schema = con.execute("DESCRIBE f0").fetchall()
    feature_columns = [name for name, *_ in schema if name not in {"Date", "Ticker"}]
    selected = ",".join(f"f.{quote(name)}" for name in feature_columns)
    control_file = str(stage / "control_pre_event_windows/part.parquet").replace("'", "''")
    con.execute(f"""COPY (
      SELECT m.event_id,m.Ticker,m.control_start_date,f.Date AS observation_date,
        CAST(p.row_position-m.control_row_position AS SMALLINT) AS relative_day,{selected}
      FROM matches m JOIN positions p ON m.Ticker=p.Ticker AND p.row_position BETWEEN m.control_row_position-20 AND m.control_row_position
      JOIN f0 f ON p.Ticker=f.Ticker AND p.Date=f.Date ORDER BY m.event_id,relative_day
    ) TO '{control_file}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
    con.close()
    control_windows = pd.read_parquet(stage / "control_pre_event_windows")
    for column in control_windows.select_dtypes(include="boolean").columns:
        control_windows[column] = control_windows[column].astype("Float64")
    control_bucket = bucket_medians(control_windows, features)
    write_wide_parquet_verified(control_bucket, stage / "control_bucket_medians/part.parquet")
    event_bucket = pd.read_parquet(root / "analysis/step4_pre_event_common_features/event_bucket_medians")
    comparison = comparison_summary(event_bucket, control_bucket, matches, features)
    write_parquet_verified(comparison, stage / "feature_comparison/part.parquet")
    duplicate_controls = int(matches.duplicated(["Ticker", "control_start_date"]).sum())
    post_control_rows = int((control_windows.relative_day > 0).sum())
    quality = "PASS" if len(matches) > 0 and duplicate_controls == 0 and post_control_rows == 0 else "FAIL"
    finish_stage(stage, target)
    shutil.rmtree(work, ignore_errors=True)
    report = {
        "step": 5, "status": "STEP5 complete" if quality == "PASS" else "STEP5 FAIL", "quality": quality,
        "created_at_jst": datetime.now(JST).isoformat(), "input_fingerprint": input_hash,
        "steps1_to_4_recalculated": False, "future_labels_used_only_for_control_assignment": True,
        "strategy_files_modified": [], "signal_conditions_created": False,
        "input_events": int(len(event_anchors)), "matched_events": int(len(matches)), "unmatched_events": int(len(event_anchors) - len(matches)),
        "match_rate": float(len(matches) / len(event_anchors)), "unique_control_dates_by_ticker": int(len(matches) - duplicate_controls),
        "reused_control_rows": duplicate_controls, "same_year_match_rate": float(matches.same_calendar_year.mean()),
        "median_control_distance_trading_days": finite(matches.trading_day_distance.median()),
        "control_exclusion_radius": "60 ticker trading rows around every main event start",
        "control_definition": "40d +30% flag false with complete 40d label; same ticker; nearest unused date, same year preferred",
        "control_window_rows": int(len(control_windows)), "post_control_rows": post_control_rows,
        "features_compared": len(features), "comparison_rows": int(len(comparison)),
        "file_size_bytes": directory_size(target), "save_path": str(target),
        "completion_statement": "STEP5で作成した成果物は今後再計算不要" if quality == "PASS" else "STEP5再実行が必要",
        "recalculation_required": quality != "PASS",
        "residual_risks": ["matching controls the ticker but not every latent time-varying confounder", "event observations from one ticker remain correlated", "control selection uses future labels only for class assignment", "all-null STEP1 feature families remain unavailable"],
    }
    write_json(report_path, report)
    (root / "quality/STEP6_PROMPT.md").write_text(prompt_text(6), encoding="utf-8")
    return report


def eligible_predictor(column: str) -> bool:
    name = column.lower()
    raw = {"open", "high", "low", "close", "adj close", "volume", "trading_value", "atr_14"}
    if name in raw or name.startswith("ma_") and not any(x in name for x in ("deviation", "slope", "rank")):
        return False
    if name.endswith("_close") or "_mean_" in name or name in {"market_trading_value_total", "sector_trading_value"}:
        return False
    return any(token in name for token in ("return", "deviation", "slope", "rank", "perfect", "distance", "rsi", "atr_14_pct", "volatility", "gap", "ratio", "location", "price_up", "price_flat", "new_high", "strength", "change", "vs_ma20"))


def sample_frame(root: Path) -> tuple[pd.DataFrame, list[str]]:
    event_bucket = pd.read_parquet(root / "analysis/step4_pre_event_common_features/event_bucket_medians")
    control_bucket = pd.read_parquet(root / "analysis/step5_control_comparison/control_bucket_medians")
    matches = pd.read_parquet(root / "analysis/step5_control_comparison/matched_pairs")
    event = event_bucket[event_bucket.bucket == BUCKET].merge(matches[["event_id", "event_start_date"]], on="event_id")
    control = control_bucket[control_bucket.bucket == BUCKET].merge(matches[["event_id", "event_start_date"]], on="event_id")
    event["outcome"] = 1
    control["outcome"] = 0
    event["sample_id"] = event.event_id + "-event"
    control["sample_id"] = control.event_id + "-control"
    frame = pd.concat([event, control], ignore_index=True)
    frame["event_start_date"] = pd.to_datetime(frame.event_start_date)
    frame["period"] = np.select([frame.event_start_date <= FORMATION_END, frame.event_start_date <= VALIDATION_END], ["formation", "validation"], default="reserved_oos")
    features = [column for column in numeric_feature_columns(frame) if eligible_predictor(column) and frame[column].notna().any()]
    return frame, features


def auc_metric(y: pd.Series, x: pd.Series) -> tuple[float | None, str | None, int]:
    valid = y.notna() & x.notna() & np.isfinite(x)
    if valid.sum() < 20 or y[valid].nunique() < 2 or x[valid].nunique() < 2:
        return None, None, int(valid.sum())
    auc = float(roc_auc_score(y[valid], x[valid]))
    return (auc if auc >= 0.5 else 1 - auc), ("high" if auc >= 0.5 else "low"), int(valid.sum())


def build_step6(root: Path, force: bool) -> dict[str, Any]:
    report_path = root / "quality/step6_report.json"
    target = root / "analysis/step6_univariate"
    input_hash = hash_inputs([("step5", root / "analysis/step5_control_comparison")])
    if not force and report_path.exists() and target.exists():
        prior = json.loads(report_path.read_text())
        if prior.get("quality") == "PASS" and prior.get("input_fingerprint") == input_hash:
            return prior
    frame, features = sample_frame(root)
    stage = Path(tempfile.mkdtemp(prefix="step6_", dir=root.parent)) / target.name
    (stage / "univariate_metrics").mkdir(parents=True)
    (stage / "quantile_bins").mkdir(parents=True)
    metrics: list[dict[str, Any]] = []
    scopes = [("formation", frame.period == "formation"), ("validation", frame.period == "validation")]
    scopes += [(str(year), frame.event_start_date.dt.year == year) for year in sorted(frame.event_start_date.dt.year.unique()) if year <= 2025]
    for feature in features:
        for scope, mask in scopes:
            subset = frame.loc[mask, ["outcome", feature]]
            auc, direction, count = auc_metric(subset.outcome, pd.to_numeric(subset[feature], errors="coerce"))
            events = pd.to_numeric(subset.loc[subset.outcome == 1, feature], errors="coerce")
            controls = pd.to_numeric(subset.loc[subset.outcome == 0, feature], errors="coerce")
            pooled = math.sqrt((events.var(ddof=1) + controls.var(ddof=1)) / 2) if len(events.dropna()) > 1 and len(controls.dropna()) > 1 else np.nan
            metrics.append({"feature": feature, "scope": scope, "auc_oriented": auc, "direction": direction, "valid_rows": count,
                            "event_median": finite(events.median()), "control_median": finite(controls.median()),
                            "standardized_mean_difference": finite((events.mean() - controls.mean()) / pooled) if pooled and math.isfinite(pooled) else None,
                            "missing_rate": float(subset[feature].isna().mean())})
    metrics_df = pd.DataFrame(metrics)
    bins: list[dict[str, Any]] = []
    formation = frame[frame.period == "formation"]
    validation = frame[frame.period == "validation"]
    for feature in features:
        values = pd.to_numeric(formation[feature], errors="coerce").dropna()
        if values.nunique() < 5:
            continue
        edges = np.unique(values.quantile([0, .2, .4, .6, .8, 1]).to_numpy(dtype=float))
        if len(edges) < 3:
            continue
        for scope, subset in (("formation", formation), ("validation", validation)):
            x = pd.to_numeric(subset[feature], errors="coerce")
            bucket = pd.cut(x, bins=edges, labels=False, include_lowest=True, duplicates="drop")
            for number, group in subset.assign(_bucket=bucket).dropna(subset=["_bucket"]).groupby("_bucket"):
                bins.append({"feature": feature, "scope": scope, "bin": int(number) + 1, "lower": float(edges[int(number)]), "upper": float(edges[int(number) + 1]),
                             "rows": int(len(group)), "events": int(group.outcome.sum()), "event_rate": float(group.outcome.mean())})
    bins_df = pd.DataFrame(bins)
    write_parquet_verified(metrics_df, stage / "univariate_metrics/part.parquet")
    write_parquet_verified(bins_df, stage / "quantile_bins/part.parquet")
    reserved_rows = int((frame.period == "reserved_oos").sum())
    quality = "PASS" if reserved_rows > 0 and (metrics_df.scope == "validation").any() else "FAIL"
    finish_stage(stage, target)
    report = {"step": 6, "status": "STEP6 complete" if quality == "PASS" else "STEP6 FAIL", "quality": quality,
              "created_at_jst": datetime.now(JST).isoformat(), "input_fingerprint": input_hash, "steps1_to_5_recalculated": False,
              "analysis_bucket": BUCKET, "formation_end": str(FORMATION_END.date()), "validation_end": str(VALIDATION_END.date()),
              "reserved_oos_start": str(OOS_START.date()), "reserved_oos_rows_not_used": reserved_rows,
              "features_tested": len(features), "metric_rows": int(len(metrics_df)), "quantile_rows": int(len(bins_df)),
              "threshold_optimization_performed": False, "file_size_bytes": directory_size(target), "save_path": str(target),
              "completion_statement": "STEP6で作成した成果物は今後再計算不要" if quality == "PASS" else "STEP6再実行が必要",
              "recalculation_required": quality != "PASS", "residual_risks": ["matched case-control AUC is not an unconditional market event probability", "multiple-testing remains substantial", "year-level dependence and repeated tickers reduce effective sample size"]}
    write_json(report_path, report)
    (root / "quality/STEP7_PROMPT.md").write_text(prompt_text(7), encoding="utf-8")
    return report


def condition_mask(frame: pd.DataFrame, definition: dict[str, Any]) -> pd.Series:
    values = pd.to_numeric(frame[definition["feature"]], errors="coerce")
    return values.ge(definition["threshold"]) if definition["direction"] == "high" else values.le(definition["threshold"])


def combo_metrics(frame: pd.DataFrame, definitions: list[dict[str, Any]], combo: tuple[int, ...], period: str) -> dict[str, Any]:
    subset = frame[frame.period == period]
    mask = pd.Series(True, index=subset.index)
    for index in combo:
        mask &= condition_mask(subset, definitions[index]).fillna(False)
    selected = subset[mask]
    base = float(subset.outcome.mean()) if len(subset) else np.nan
    rate = float(selected.outcome.mean()) if len(selected) else np.nan
    event_total = int(subset.outcome.sum())
    return {"period": period, "rows": int(len(selected)), "events": int(selected.outcome.sum()) if len(selected) else 0,
            "event_rate": finite(rate), "base_event_rate": finite(base), "lift": finite(rate / base) if base and len(selected) else None,
            "recall": finite(selected.outcome.sum() / event_total) if event_total else None}


def build_step7(root: Path, force: bool) -> dict[str, Any]:
    report_path = root / "quality/step7_report.json"
    target = root / "analysis/step7_condition_combinations"
    input_hash = hash_inputs([("step6", root / "analysis/step6_univariate")])
    if not force and report_path.exists() and target.exists():
        prior = json.loads(report_path.read_text())
        if prior.get("quality") == "PASS" and prior.get("input_fingerprint") == input_hash:
            return prior
    frame, features = sample_frame(root)
    metrics = pd.read_parquet(root / "analysis/step6_univariate/univariate_metrics")
    formation_metrics = metrics[metrics.scope == "formation"].set_index("feature")
    validation_metrics = metrics[metrics.scope == "validation"].set_index("feature")
    ranking: list[tuple[str, float]] = []
    for feature in features:
        if feature not in formation_metrics.index or feature not in validation_metrics.index:
            continue
        fm, vm = formation_metrics.loc[feature], validation_metrics.loc[feature]
        if pd.notna(fm.auc_oriented) and pd.notna(vm.auc_oriented) and fm.direction == vm.direction and fm.auc_oriented >= .53 and vm.auc_oriented >= .52:
            ranking.append((feature, float(min(fm.auc_oriented, vm.auc_oriented))))
    ranking.sort(key=lambda item: item[1], reverse=True)
    formation = frame[frame.period == "formation"]
    selected_features: list[str] = []
    for feature, _ in ranking:
        if len(selected_features) >= 8:
            break
        correlations = [abs(pd.to_numeric(formation[feature], errors="coerce").corr(pd.to_numeric(formation[chosen], errors="coerce"), method="spearman")) for chosen in selected_features]
        if not correlations or max(value for value in correlations if pd.notna(value)) < .8:
            selected_features.append(feature)
    definitions: list[dict[str, Any]] = []
    for feature in selected_features:
        direction = str(formation_metrics.loc[feature, "direction"])
        quantile = .7 if direction == "high" else .3
        threshold = float(pd.to_numeric(formation[feature], errors="coerce").quantile(quantile))
        definitions.append({"feature": feature, "direction": direction, "quantile": quantile, "threshold": threshold})
    combo_rows: list[dict[str, Any]] = []
    for size in (2, 3):
        for combo in itertools.combinations(range(len(definitions)), size):
            combo_id = "__AND__".join(definitions[index]["feature"] for index in combo)
            for period in ("formation", "validation"):
                combo_rows.append({"combo_id": combo_id, "condition_count": size,
                                   "conditions_json": json.dumps([definitions[index] for index in combo], ensure_ascii=False),
                                   **combo_metrics(frame, definitions, combo, period)})
    combo_df = pd.DataFrame(combo_rows)
    valid = combo_df[(combo_df.period == "validation") & (combo_df.rows >= 100) & combo_df.lift.notna()].sort_values(["lift", "rows"], ascending=[False, False])
    candidate_ids = valid.combo_id.head(10).tolist()
    combo_df["top10_validation_candidate"] = combo_df.combo_id.isin(candidate_ids)
    stage = Path(tempfile.mkdtemp(prefix="step7_", dir=root.parent)) / target.name
    (stage / "combination_metrics").mkdir(parents=True)
    write_parquet_verified(combo_df, stage / "combination_metrics/part.parquet")
    write_json(stage / "condition_definitions.json", {"definitions": definitions, "top10_candidate_combo_ids": candidate_ids})
    quality = "PASS" if definitions and candidate_ids else "FAIL"
    finish_stage(stage, target)
    report = {"step": 7, "status": "STEP7 complete" if quality == "PASS" else "STEP7 FAIL", "quality": quality,
              "created_at_jst": datetime.now(JST).isoformat(), "input_fingerprint": input_hash, "steps1_to_6_recalculated": False,
              "candidate_feature_count": len(definitions), "candidate_features": selected_features, "fixed_quantiles": [0.3, 0.7],
              "combination_count": int(combo_df.combo_id.nunique()) if len(combo_df) else 0, "top_candidate_count": len(candidate_ids),
              "minimum_validation_rows": 100, "reserved_oos_used": False, "fine_threshold_optimization": False,
              "signals_adopted": False, "file_size_bytes": directory_size(target), "save_path": str(target),
              "completion_statement": "STEP7で作成した成果物は今後再計算不要" if quality == "PASS" else "STEP7再実行が必要",
              "recalculation_required": quality != "PASS", "residual_risks": ["candidate ranking uses the validation period and is not final evidence", "multiple combinations create selection bias", "case-control lift does not equal live-market precision"]}
    write_json(report_path, report)
    (root / "quality/STEP8_PROMPT.md").write_text(prompt_text(8), encoding="utf-8")
    return report


def build_step8(root: Path, force: bool) -> dict[str, Any]:
    report_path = root / "quality/step8_report.json"
    target = root / "analysis/step8_regime_analysis"
    input_hash = hash_inputs([("step7", root / "analysis/step7_condition_combinations")])
    if not force and report_path.exists() and target.exists():
        prior = json.loads(report_path.read_text())
        if prior.get("quality") == "PASS" and prior.get("input_fingerprint") == input_hash:
            return prior
    frame, _ = sample_frame(root)
    event_d0 = pd.read_parquet(root / "analysis/step4_pre_event_common_features/event_bucket_medians")
    control_d0 = pd.read_parquet(root / "analysis/step5_control_comparison/control_bucket_medians")
    matches = pd.read_parquet(root / "analysis/step5_control_comparison/matched_pairs")
    event_d0 = event_d0[event_d0.bucket == "d0"].assign(sample_id=lambda x: x.event_id + "-event")
    control_d0 = control_d0[control_d0.bucket == "d0"].assign(sample_id=lambda x: x.event_id + "-control")
    d0 = pd.concat([event_d0, control_d0], ignore_index=True)
    frame = frame.merge(d0.drop(columns=["event_id", "bucket"], errors="ignore"), on="sample_id", suffixes=("", "_d0"))
    definitions_data = json.loads((root / "analysis/step7_condition_combinations/condition_definitions.json").read_text())
    combo_metrics_df = pd.read_parquet(root / "analysis/step7_condition_combinations/combination_metrics")
    candidate_ids = definitions_data["top10_candidate_combo_ids"]
    condition_lookup = {row.combo_id: json.loads(row.conditions_json) for row in combo_metrics_df.drop_duplicates("combo_id").itertuples() if row.combo_id in candidate_ids}
    validation = frame[frame.period == "validation"].copy()
    regime_specs = {
        "nikkei_trend": ("index_nikkei225_vs_ma20_d0", 0.0, "above_ma20", "below_ma20"),
        "vix_direction": ("external_vix_vs_ma20_d0", 0.0, "above_ma20", "below_ma20"),
        "us10y_direction": ("external_us10y_change_20d_d0", 0.0, "rising", "falling"),
        "market_breadth": ("market_advancer_ratio_d0", 0.5, "broad", "narrow"),
    }
    rows: list[dict[str, Any]] = []
    unavailable: list[str] = []
    for regime, (column, split, high_label, low_label) in regime_specs.items():
        if column not in validation or not validation[column].notna().any():
            unavailable.append(regime)
            continue
        state = np.where(validation[column].isna(), "missing", np.where(validation[column] >= split, high_label, low_label))
        for combo_id, definitions in condition_lookup.items():
            signal = pd.Series(True, index=validation.index)
            for definition in definitions:
                signal &= condition_mask(validation, definition).fillna(False)
            for label in sorted(set(state)):
                subset = validation[state == label]
                selected = validation[(state == label) & signal]
                base = float(subset.outcome.mean()) if len(subset) else np.nan
                rate = float(selected.outcome.mean()) if len(selected) else np.nan
                rows.append({"combo_id": combo_id, "regime": regime, "state": label, "rows": int(len(selected)), "events": int(selected.outcome.sum()),
                             "event_rate": finite(rate), "regime_base_rate": finite(base), "lift": finite(rate / base) if base and len(selected) else None})
    result = pd.DataFrame(rows)
    stage = Path(tempfile.mkdtemp(prefix="step8_", dir=root.parent)) / target.name
    (stage / "regime_metrics").mkdir(parents=True)
    write_parquet_verified(result, stage / "regime_metrics/part.parquet")
    quality = "PASS" if len(result) and len(unavailable) < len(regime_specs) else "FAIL"
    finish_stage(stage, target)
    report = {"step": 8, "status": "STEP8 complete" if quality == "PASS" else "STEP8 FAIL", "quality": quality,
              "created_at_jst": datetime.now(JST).isoformat(), "input_fingerprint": input_hash, "steps1_to_7_recalculated": False,
              "candidate_combos": len(candidate_ids), "regimes_requested": list(regime_specs), "unavailable_regimes": unavailable,
              "metric_rows": int(len(result)), "reserved_oos_used": False, "file_size_bytes": directory_size(target), "save_path": str(target),
              "completion_statement": "STEP8で作成した成果物は今後再計算不要" if quality == "PASS" else "STEP8再実行が必要",
              "recalculation_required": quality != "PASS", "residual_risks": ["regime cells can be small and unstable", "regime labels are fixed descriptive splits, not optimized", "unavailable STEP1 macro series cannot be evaluated"]}
    write_json(report_path, report)
    (root / "quality/STEP9_PROMPT.md").write_text(prompt_text(9), encoding="utf-8")
    return report


def trade_summary(trades: pd.DataFrame, group: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, subset in trades.groupby(group, dropna=False):
        returns = pd.to_numeric(subset.net_return, errors="coerce").dropna()
        gains = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
        rows.append({group: key, "trades": int(len(returns)), "signals": int(subset.signal_id.nunique()), "fill_rate": float(len(returns) / subset.signal_id.nunique()) if subset.signal_id.nunique() else None,
                     "mean_net_return": finite(returns.mean()), "median_net_return": finite(returns.median()), "win_rate": finite((returns > 0).mean()),
                     "profit_factor": finite(gains / losses) if losses > 0 else None})
    return pd.DataFrame(rows)


def cooldown_signals(frame: pd.DataFrame, mask: pd.Series, cooldown: int = 20) -> pd.DataFrame:
    selected: list[int] = []
    for _, group in frame[mask].groupby("Ticker", sort=False):
        last = -10**9
        for index, position in zip(group.index, group.row_position.astype(int), strict=False):
            if position - last > cooldown:
                selected.append(index)
                last = position
    return frame.loc[selected].copy()


def load_price_features(root: Path, combo_definitions: list[dict[str, Any]], memory_limit: str) -> pd.DataFrame:
    features = sorted({definition["feature"] for definition in combo_definitions})
    rolling = ",".join(f"median({quote(feature)}) OVER(PARTITION BY Ticker ORDER BY Date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS {quote(feature)}" for feature in features)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    result = con.execute(f"""
      SELECT Date,Ticker,row_number() OVER(PARTITION BY Ticker ORDER BY Date)-1 AS row_position,
        "Adj Close" AS adj_close,Open*"Adj Close"/nullif(Close,0) AS adj_open,
        High*"Adj Close"/nullif(Close,0) AS adj_high,Low*"Adj Close"/nullif(Close,0) AS adj_low,
        ma_25*"Adj Close"/nullif(Close,0) AS adj_ma25,atr_14*"Adj Close"/nullif(Close,0) AS adj_atr14,
        {rolling}
      FROM read_parquet('{parquet_glob(root / 'features/equity_daily_features')}') ORDER BY Ticker,Date
    """).df()
    con.close()
    result["Date"] = pd.to_datetime(result.Date)
    return result


def simulate_entries(prices: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    groups = {ticker: group.reset_index(drop=True) for ticker, group in prices.groupby("Ticker", sort=False)}
    for signal in signals.itertuples(index=False):
        group = groups[signal.Ticker]
        p = int(signal.row_position)
        signal_id = f"{signal.Ticker}-{pd.Timestamp(signal.Date).date()}"
        modes: list[tuple[str, int | None, float | None]] = [
            ("open_t", p, group.at[p, "adj_open"]),
            ("close_t_benchmark", p, group.at[p, "adj_close"]),
        ]
        if p + 1 < len(group):
            modes += [("next_open", p + 1, group.at[p + 1, "adj_open"]), ("next_close", p + 1, group.at[p + 1, "adj_close"])]
        limit_pos = limit_price = None
        signal_close = float(group.at[p, "adj_close"])
        for q in range(p + 1, min(p + 6, len(group))):
            if group.at[q, "adj_low"] <= signal_close:
                limit_pos = q
                limit_price = min(float(group.at[q, "adj_open"]), signal_close)
                break
        modes.append(("pullback_limit_5d", limit_pos, limit_price))
        for mode, entry_pos, entry_price in modes:
            if entry_pos is None or entry_price is None or not math.isfinite(float(entry_price)) or entry_pos + 40 >= len(group):
                records.append({"signal_id": signal_id, "Ticker": signal.Ticker, "signal_date": signal.Date, "mode": mode, "filled": False})
                continue
            exit_price = float(group.at[entry_pos + 40, "adj_close"])
            gross = exit_price / float(entry_price) - 1
            records.append({"signal_id": signal_id, "Ticker": signal.Ticker, "signal_date": signal.Date, "mode": mode, "filled": True,
                            "entry_date": group.at[entry_pos, "Date"], "entry_position": entry_pos, "entry_price": float(entry_price),
                            "exit_date": group.at[entry_pos + 40, "Date"], "exit_price": exit_price, "gross_return": gross, "net_return": gross - ROUND_TRIP_COST})
    return pd.DataFrame(records)


def build_step9(root: Path, memory_limit: str, force: bool) -> tuple[dict[str, Any], pd.DataFrame | None]:
    report_path = root / "quality/step9_report.json"
    target = root / "analysis/step9_entry_timing"
    input_hash = hash_inputs([("step7", root / "analysis/step7_condition_combinations"), ("step8", root / "analysis/step8_regime_analysis")])
    if not force and report_path.exists() and target.exists():
        prior = json.loads(report_path.read_text())
        if prior.get("quality") == "PASS" and prior.get("input_fingerprint") == input_hash and prior.get("implementation_version") == STEP9_IMPLEMENTATION_VERSION:
            return prior, None
    combos = pd.read_parquet(root / "analysis/step7_condition_combinations/combination_metrics")
    validation = combos[(combos.period == "validation") & combos.top10_validation_candidate & (combos.rows >= 100)].sort_values(["lift", "rows"], ascending=[False, False])
    best = validation.iloc[0]
    definitions = json.loads(best.conditions_json)
    prices = load_price_features(root, definitions, memory_limit)
    mask = (prices.Date > FORMATION_END) & (prices.Date <= VALIDATION_END)
    for definition in definitions:
        mask &= condition_mask(prices, definition).fillna(False)
    signals = cooldown_signals(prices, mask, 20)
    trades = simulate_entries(prices, signals)
    filled = trades[trades.filled].copy()
    summary = trade_summary(trades, "mode")
    annual_rows: list[dict[str, Any]] = []
    trades["year"] = pd.to_datetime(trades.signal_date).dt.year
    for (mode, year), subset in trades.groupby(["mode", "year"]):
        returns = pd.to_numeric(subset.loc[subset.filled, "net_return"], errors="coerce").dropna()
        gains, losses = returns[returns > 0].sum(), -returns[returns < 0].sum()
        annual_rows.append({"mode": mode, "year": int(year), "trades": int(len(returns)),
                            "signals": int(subset.signal_id.nunique()), "fill_rate": float(subset.filled.mean()),
                            "mean_net_return": finite(returns.mean()), "median_net_return": finite(returns.median()),
                            "win_rate": finite((returns > 0).mean()), "profit_factor": finite(gains / losses) if losses > 0 else None})
    annual = pd.DataFrame(annual_rows)
    executable = summary[(summary["mode"] != "close_t_benchmark") & (summary.fill_rate >= .7) & summary.mean_net_return.notna()].sort_values(["mean_net_return", "profit_factor"], ascending=False)
    selected_mode = str(executable.iloc[0]["mode"]) if len(executable) else None
    stage = Path(tempfile.mkdtemp(prefix="step9_", dir=root.parent)) / target.name
    for name in ("entry_trades", "entry_summary", "annual_summary", "signals"):
        (stage / name).mkdir(parents=True)
    write_parquet_verified(trades, stage / "entry_trades/part.parquet")
    write_parquet_verified(summary, stage / "entry_summary/part.parquet")
    write_parquet_verified(annual, stage / "annual_summary/part.parquet")
    write_parquet_verified(signals[["Date", "Ticker", "row_position"]], stage / "signals/part.parquet")
    write_json(stage / "selected_research_candidate.json", {"combo_id": best.combo_id, "conditions": definitions, "entry_mode_for_step10": selected_mode})
    oos_used = bool((signals.Date >= OOS_START).any())
    quality = "PASS" if len(signals) and selected_mode and not oos_used else "FAIL"
    finish_stage(stage, target)
    report = {"step": 9, "status": "STEP9 complete" if quality == "PASS" else "STEP9 FAIL", "quality": quality,
              "created_at_jst": datetime.now(JST).isoformat(), "input_fingerprint": input_hash, "implementation_version": STEP9_IMPLEMENTATION_VERSION, "steps1_to_8_recalculated": False,
              "research_combo": str(best.combo_id), "signal_count_after_20d_cooldown": int(len(signals)), "trade_rows": int(len(trades)),
              "transaction_cost_round_trip": ROUND_TRIP_COST, "fixed_exit_horizon": 40,
              "selected_executable_mode_for_step10": selected_mode, "close_t_is_execution_sensitive_benchmark": True,
              "reserved_oos_used": oos_used, "file_size_bytes": directory_size(target), "save_path": str(target),
              "completion_statement": "STEP9で作成した成果物は今後再計算不要" if quality == "PASS" else "STEP9再実行が必要",
              "recalculation_required": quality != "PASS", "residual_risks": ["selection used validation results and must be confirmed only in STEP11 OOS", "signals are throttled with a documented 20-trading-day ticker cooldown", "limit-order fills use daily bars and cannot resolve intraday path"]}
    write_json(report_path, report)
    (root / "quality/STEP10_PROMPT.md").write_text(prompt_text(10), encoding="utf-8")
    return report, prices


def exit_trade(group: pd.DataFrame, entry_pos: int, entry_price: float, rule: str) -> dict[str, Any] | None:
    if entry_pos >= len(group) - 1:
        return None
    end = min(entry_pos + 60, len(group) - 1)
    exit_pos = end
    exit_price = float(group.at[end, "adj_close"])
    reason = "max_60d"
    if rule.startswith("fixed_"):
        horizon = int(rule.split("_")[1][:-1])
        if entry_pos + horizon >= len(group):
            return None
        exit_pos = entry_pos + horizon
        exit_price = float(group.at[exit_pos, "adj_close"])
        reason = rule
    elif rule in {"tp20_sl8", "tp30_sl10"}:
        tp, sl = ((.20, .08) if rule == "tp20_sl8" else (.30, .10))
        for q in range(entry_pos + 1, end + 1):
            stop_price, target_price = entry_price * (1 - sl), entry_price * (1 + tp)
            if group.at[q, "adj_low"] <= stop_price:
                exit_pos, exit_price, reason = q, stop_price, "stop_first"
                break
            if group.at[q, "adj_high"] >= target_price:
                exit_pos, exit_price, reason = q, target_price, "take_profit"
                break
    elif rule == "ma25_break":
        for q in range(entry_pos + 1, end):
            if pd.notna(group.at[q, "adj_ma25"]) and group.at[q, "adj_close"] < group.at[q, "adj_ma25"]:
                exit_pos, exit_price, reason = q + 1, float(group.at[q + 1, "adj_open"]), "next_open_after_ma25_break"
                break
    elif rule == "atr3_trail":
        highest = entry_price
        for q in range(entry_pos + 1, end):
            highest = max(highest, float(group.at[q, "adj_close"]))
            atr = group.at[q, "adj_atr14"]
            if pd.notna(atr) and group.at[q, "adj_close"] < highest - 3 * atr:
                exit_pos, exit_price, reason = q + 1, float(group.at[q + 1, "adj_open"]), "next_open_after_atr3_breach"
                break
    path = group.iloc[entry_pos + 1:exit_pos + 1]
    mfe = float(path.adj_high.max() / entry_price - 1) if len(path) else 0.0
    mae = float(path.adj_low.min() / entry_price - 1) if len(path) else 0.0
    gross = exit_price / entry_price - 1
    return {"exit_position": exit_pos, "exit_date": group.at[exit_pos, "Date"], "exit_price": exit_price, "exit_reason": reason,
            "holding_days": exit_pos - entry_pos, "gross_return": gross, "net_return": gross - ROUND_TRIP_COST, "mfe": mfe, "mae": mae}


def max_equal_notional_drawdown_points(returns: pd.Series) -> float | None:
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values):
        return None
    cumulative_pnl = np.r_[0.0, np.cumsum(values)]
    peaks = np.maximum.accumulate(cumulative_pnl)
    return float(np.min(cumulative_pnl - peaks))


def build_step10(root: Path, memory_limit: str, force: bool, prices: pd.DataFrame | None) -> dict[str, Any]:
    report_path = root / "quality/step10_report.json"
    target = root / "analysis/step10_exit_analysis"
    input_hash = hash_inputs([("step9", root / "analysis/step9_entry_timing")])
    if not force and report_path.exists() and target.exists():
        prior = json.loads(report_path.read_text())
        if prior.get("quality") == "PASS" and prior.get("input_fingerprint") == input_hash and prior.get("implementation_version") == STEP10_IMPLEMENTATION_VERSION:
            return prior
    selected = json.loads((root / "analysis/step9_entry_timing/selected_research_candidate.json").read_text())
    mode = selected["entry_mode_for_step10"]
    entries = pd.read_parquet(root / "analysis/step9_entry_timing/entry_trades")
    entries = entries[(entries["mode"] == mode) & entries.filled].copy()
    if prices is None:
        prices = load_price_features(root, selected["conditions"], memory_limit)
    groups = {ticker: group.reset_index(drop=True) for ticker, group in prices.groupby("Ticker", sort=False)}
    rules = ("fixed_10d", "fixed_20d", "fixed_40d", "tp20_sl8", "tp30_sl10", "ma25_break", "atr3_trail")
    records: list[dict[str, Any]] = []
    for entry in entries.itertuples(index=False):
        group = groups[entry.Ticker]
        for rule in rules:
            outcome = exit_trade(group, int(entry.entry_position), float(entry.entry_price), rule)
            if outcome is not None:
                records.append({"signal_id": entry.signal_id, "Ticker": entry.Ticker, "signal_date": entry.signal_date,
                                "entry_date": entry.entry_date, "entry_price": entry.entry_price, "entry_mode": mode, "exit_rule": rule, **outcome})
    trades = pd.DataFrame(records)
    summary_rows: list[dict[str, Any]] = []
    for rule, subset in trades.groupby("exit_rule"):
        returns = subset.net_return
        gains, losses = returns[returns > 0].sum(), -returns[returns < 0].sum()
        summary_rows.append({"exit_rule": rule, "trades": int(len(subset)), "mean_net_return": finite(returns.mean()), "median_net_return": finite(returns.median()),
                             "win_rate": finite((returns > 0).mean()), "profit_factor": finite(gains / losses) if losses > 0 else None,
                             "mean_mfe": finite(subset.mfe.mean()), "mean_mae": finite(subset.mae.mean()), "mean_holding_days": finite(subset.holding_days.mean()),
                             "equal_notional_max_drawdown_return_points": max_equal_notional_drawdown_points(subset.sort_values("signal_date").net_return)})
    summary = pd.DataFrame(summary_rows)
    annual_rows: list[dict[str, Any]] = []
    trades["year"] = pd.to_datetime(trades.signal_date).dt.year
    for (rule, year), subset in trades.groupby(["exit_rule", "year"]):
        returns = subset.net_return
        gains, losses = returns[returns > 0].sum(), -returns[returns < 0].sum()
        annual_rows.append({"exit_rule": rule, "year": int(year), "trades": int(len(subset)), "mean_net_return": finite(returns.mean()),
                            "win_rate": finite((returns > 0).mean()), "profit_factor": finite(gains / losses) if losses > 0 else None})
    annual = pd.DataFrame(annual_rows)
    stage = Path(tempfile.mkdtemp(prefix="step10_", dir=root.parent)) / target.name
    for name in ("exit_trades", "exit_summary", "annual_summary"):
        (stage / name).mkdir(parents=True)
    write_parquet_verified(trades, stage / "exit_trades/part.parquet")
    write_parquet_verified(summary, stage / "exit_summary/part.parquet")
    write_parquet_verified(annual, stage / "annual_summary/part.parquet")
    quality = "PASS" if len(summary) == len(rules) and not np.isinf(trades[["net_return", "mfe", "mae"]]).any().any() else "FAIL"
    finish_stage(stage, target)
    report = {"step": 10, "status": "STEP10 complete" if quality == "PASS" else "STEP10 FAIL", "quality": quality,
              "created_at_jst": datetime.now(JST).isoformat(), "input_fingerprint": input_hash, "implementation_version": STEP10_IMPLEMENTATION_VERSION, "steps1_to_9_recalculated": False,
              "entry_mode": mode, "exit_rules": list(rules), "signals": int(entries.signal_id.nunique()), "trade_rows": int(len(trades)),
              "transaction_cost_round_trip": ROUND_TRIP_COST, "intraday_double_hit_assumption": "stop loss first",
              "portfolio_max_drawdown_calculated": False, "drawdown_metric": "peak-to-trough decline in cumulative equal-notional trade return points; not a capital-constrained portfolio percentage",
              "reserved_oos_used": False, "final_rule_adopted": False, "file_size_bytes": directory_size(target), "save_path": str(target),
              "completion_statement": "STEP10で作成した成果物は今後再計算不要" if quality == "PASS" else "STEP10再実行が必要",
              "recalculation_required": quality != "PASS", "residual_risks": ["daily OHLC cannot establish intraday path; double hits are conservatively stop-first", "overlapping trades are not a portfolio simulation", "exit comparison remains validation data and requires untouched STEP11 OOS", "taxes, borrow constraints, limit-up/down, and order-book liquidity are not modeled"]}
    write_json(report_path, report)
    (root / "quality/STEP11_PROMPT.md").write_text(prompt_text(11), encoding="utf-8")
    return report


def main() -> int:
    options = parse_args()
    repository = Path(__file__).resolve().parents[2]
    root = (repository / "momentum5d" / options.root).resolve()
    for step in (1, 3, 4):
        report_ok(root / f"quality/step{step}_report.json", step)
    builders = [(5, lambda: build_step5(root, options.memory_limit, options.force)),
                (6, lambda: build_step6(root, options.force)),
                (7, lambda: build_step7(root, options.force)),
                (8, lambda: build_step8(root, options.force))]
    reports: list[dict[str, Any]] = []
    for step, builder in builders:
        if step > options.to_step:
            break
        report = builder()
        reports.append(report)
        if report["quality"] != "PASS":
            print(json.dumps(reports, ensure_ascii=False, indent=2))
            return 2
    prices: pd.DataFrame | None = None
    if options.to_step >= 9:
        report, prices = build_step9(root, options.memory_limit, options.force)
        reports.append(report)
        if report["quality"] != "PASS":
            print(json.dumps(reports, ensure_ascii=False, indent=2))
            return 2
    if options.to_step >= 10:
        report = build_step10(root, options.memory_limit, options.force, prices)
        reports.append(report)
        if report["quality"] != "PASS":
            print(json.dumps(reports, ensure_ascii=False, indent=2))
            return 2
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
