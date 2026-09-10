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
import pandas as pd

JST = ZoneInfo("Asia/Tokyo")


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STEP1 point-in-time feature master")
    p.add_argument("--root", default="data/market_history")
    p.add_argument("--force", action="store_true")
    p.add_argument("--memory-limit", default="2GB")
    return p.parse_args()


def pg(path: Path) -> str:
    return str(path.resolve() / "**" / "*.parquet").replace("'", "''")


def quoted(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(str(path.stat().st_size).encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def named_features(path: Path, prefix: str, lag_one: bool) -> pd.DataFrame:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["Date"])
    frame = pd.concat(
        [pd.read_parquet(f, columns=["Date", "Series", "Close"]) for f in files],
        ignore_index=True,
    ).sort_values(["Series", "Date"])
    if lag_one:
        # Prevent Tokyo-close look-ahead from US/FX same-calendar-date candles.
        frame["Close"] = frame.groupby("Series", sort=False)["Close"].shift(1)
    pieces = []
    for name, group in frame.groupby("Series", sort=False):
        group = group.sort_values("Date").drop_duplicates("Date", keep="last")
        out = pd.DataFrame({"Date": group["Date"], f"{prefix}_{name}_close": group["Close"]})
        for n in [1, 5, 20]:
            out[f"{prefix}_{name}_change_{n}d"] = group["Close"].pct_change(n, fill_method=None).to_numpy()
        out[f"{prefix}_{name}_vs_ma20"] = (
            group["Close"] / group["Close"].rolling(20, min_periods=20).mean() - 1
        ).to_numpy()
        pieces.append(out.set_index("Date"))
    # Carry the last already-known observation across source-market holidays.
    # The return itself remains based on source observations, not calendar rows.
    return pd.concat(pieces, axis=1).sort_index().ffill().reset_index()


def feature_sql(context_columns: list[str]) -> str:
    context = "".join(f", c.{quoted(col)}" for col in context_columns)
    return f"""
    WITH w AS (
      SELECT CAST(Date AS DATE) Date, Ticker, Open, High, Low, Close, "Adj Close", Volume,
        Close*Volume trading_value,
        lag(Close,1) OVER z c1, lag(Close,5) OVER z c5,
        lag(Close,10) OVER z c10, lag(Close,20) OVER z c20,
        lag(Close,60) OVER z c60, lag(Close,120) OVER z c120,
        avg(Close) OVER(PARTITION BY Ticker ORDER BY Date ROWS 4 PRECEDING) ma5_raw,
        avg(Close) OVER(PARTITION BY Ticker ORDER BY Date ROWS 9 PRECEDING) ma10_raw,
        avg(Close) OVER(PARTITION BY Ticker ORDER BY Date ROWS 24 PRECEDING) ma25_raw,
        avg(Close) OVER(PARTITION BY Ticker ORDER BY Date ROWS 74 PRECEDING) ma75_raw,
        avg(Close) OVER(PARTITION BY Ticker ORDER BY Date ROWS 199 PRECEDING) ma200_raw,
        row_number() OVER(PARTITION BY Ticker ORDER BY Date) hist_n,
        max(High) OVER(PARTITION BY Ticker ORDER BY Date ROWS 19 PRECEDING) hi20,
        max(High) OVER(PARTITION BY Ticker ORDER BY Date ROWS 59 PRECEDING) hi60,
        max(High) OVER(PARTITION BY Ticker ORDER BY Date ROWS 251 PRECEDING) hi252,
        min(Low) OVER(PARTITION BY Ticker ORDER BY Date ROWS 19 PRECEDING) lo20,
        min(Low) OVER(PARTITION BY Ticker ORDER BY Date ROWS 251 PRECEDING) lo252,
        max(High) OVER(PARTITION BY Ticker ORDER BY Date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) prior_hi20,
        max(High) OVER(PARTITION BY Ticker ORDER BY Date ROWS BETWEEN 252 PRECEDING AND 1 PRECEDING) prior_hi252,
        avg(Volume) OVER(PARTITION BY Ticker ORDER BY Date ROWS 4 PRECEDING) vol5,
        avg(Volume) OVER(PARTITION BY Ticker ORDER BY Date ROWS 19 PRECEDING) vol20,
        avg(Close*Volume) OVER(PARTITION BY Ticker ORDER BY Date ROWS 4 PRECEDING) value5,
        avg(Close*Volume) OVER(PARTITION BY Ticker ORDER BY Date ROWS 19 PRECEDING) value20
      FROM equities WINDOW z AS(PARTITION BY Ticker ORDER BY Date)
    ), r AS (
      SELECT * EXCLUDE(ma5_raw,ma10_raw,ma25_raw,ma75_raw,ma200_raw),
        CASE WHEN hist_n>=5 THEN ma5_raw END ma5,
        CASE WHEN hist_n>=10 THEN ma10_raw END ma10,
        CASE WHEN hist_n>=25 THEN ma25_raw END ma25,
        CASE WHEN hist_n>=75 THEN ma75_raw END ma75,
        CASE WHEN hist_n>=200 THEN ma200_raw END ma200,
        Close/nullif(c1,0)-1 ret1, Close/nullif(c5,0)-1 ret5,
        Close/nullif(c10,0)-1 ret10, Close/nullif(c20,0)-1 ret20,
        Close/nullif(c60,0)-1 ret60, Close/nullif(c120,0)-1 ret120,
        greatest(High-Low,abs(High-c1),abs(Low-c1)) tr,
        lag(trading_value,1) OVER z value_lag,
        lag(ma5,5) OVER z ma5_lag, lag(ma10,5) OVER z ma10_lag,
        lag(ma25,5) OVER z ma25_lag, lag(ma75,5) OVER z ma75_lag,
        lag(ma200,5) OVER z ma200_lag
      FROM w WINDOW z AS(PARTITION BY Ticker ORDER BY Date)
    ), t AS (
      SELECT *,
        CASE WHEN hist_n>=15 THEN 100-100/(1+avg(greatest(Close-c1,0)) OVER q/
          nullif(avg(greatest(c1-Close,0)) OVER q,0)) END rsi14,
        CASE WHEN hist_n>=15 THEN avg(tr) OVER q END atr14,
        CASE WHEN hist_n>=21 THEN stddev_samp(ret1) OVER(PARTITION BY Ticker ORDER BY Date ROWS 19 PRECEDING)*sqrt(252) END rv20,
        CASE WHEN hist_n>=61 THEN stddev_samp(ret1) OVER(PARTITION BY Ticker ORDER BY Date ROWS 59 PRECEDING)*sqrt(252) END rv60
      FROM r WINDOW q AS(PARTITION BY Ticker ORDER BY Date ROWS 13 PRECEDING)
    ), e AS (
      SELECT Date,Ticker,Open,High,Low,Close,"Adj Close",Volume,trading_value,
        ret1 return_1d,ret5 return_5d,ret10 return_10d,ret20 return_20d,
        ret60 return_60d,ret120 return_120d,
        ma5 ma_5,ma10 ma_10,ma25 ma_25,ma75 ma_75,ma200 ma_200,
        Close/nullif(ma5,0)-1 ma_5_deviation,Close/nullif(ma10,0)-1 ma_10_deviation,
        Close/nullif(ma25,0)-1 ma_25_deviation,Close/nullif(ma75,0)-1 ma_75_deviation,
        Close/nullif(ma200,0)-1 ma_200_deviation,
        ma5/nullif(ma5_lag,0)-1 ma_5_slope_5d,ma10/nullif(ma10_lag,0)-1 ma_10_slope_5d,
        ma25/nullif(ma25_lag,0)-1 ma_25_slope_5d,ma75/nullif(ma75_lag,0)-1 ma_75_slope_5d,
        ma200/nullif(ma200_lag,0)-1 ma_200_slope_5d,
        1+(ma5<ma10)::INT+(ma5<ma25)::INT+(ma5<ma75)::INT+(ma5<ma200)::INT ma_5_rank_desc,
        1+(ma10<ma5)::INT+(ma10<ma25)::INT+(ma10<ma75)::INT+(ma10<ma200)::INT ma_10_rank_desc,
        1+(ma25<ma5)::INT+(ma25<ma10)::INT+(ma25<ma75)::INT+(ma25<ma200)::INT ma_25_rank_desc,
        1+(ma75<ma5)::INT+(ma75<ma10)::INT+(ma75<ma25)::INT+(ma75<ma200)::INT ma_75_rank_desc,
        1+(ma200<ma5)::INT+(ma200<ma10)::INT+(ma200<ma25)::INT+(ma200<ma75)::INT ma_200_rank_desc,
        hist_n>=200 AND ma5>ma10 AND ma10>ma25 AND ma25>ma75 AND ma75>ma200 perfect_order_bull,
        hist_n>=200 AND ma5<ma10 AND ma10<ma25 AND ma25<ma75 AND ma75<ma200 perfect_order_bear,
        CASE WHEN hist_n>=20 THEN Close/nullif(hi20,0)-1 END distance_from_20d_high,
        CASE WHEN hist_n>=60 THEN Close/nullif(hi60,0)-1 END distance_from_60d_high,
        CASE WHEN hist_n>=252 THEN Close/nullif(hi252,0)-1 END distance_from_52w_high,
        CASE WHEN hist_n>=20 THEN Close/nullif(lo20,0)-1 END distance_from_20d_low,
        CASE WHEN hist_n>=252 THEN Close/nullif(lo252,0)-1 END distance_from_52w_low,
        rsi14 rsi_14,atr14 atr_14,atr14/nullif(Close,0) atr_14_pct,
        rv20 realized_volatility_20d,rv20/nullif(rv60,0) volatility_contraction_ratio,
        Open/nullif(c1,0)-1 gap_rate,abs(Close-Open)/nullif(High-Low,0) body_ratio,
        (High-greatest(Open,Close))/nullif(High-Low,0) upper_wick_ratio,
        (least(Open,Close)-Low)/nullif(High-Low,0) lower_wick_ratio,
        (Close-Low)/nullif(High-Low,0) close_location,
        CASE WHEN hist_n>=5 THEN vol5 END volume_mean_5d,
        CASE WHEN hist_n>=20 THEN vol20 END volume_mean_20d,
        CASE WHEN hist_n>=5 THEN Volume/nullif(vol5,0) END volume_ratio_5d,
        CASE WHEN hist_n>=20 THEN Volume/nullif(vol20,0) END volume_ratio_20d,
        CASE WHEN hist_n>=5 THEN value5 END trading_value_mean_5d,
        CASE WHEN hist_n>=20 THEN value20 END trading_value_mean_20d,
        CASE WHEN hist_n>=5 THEN trading_value/nullif(value5,0) END trading_value_ratio_5d,
        CASE WHEN hist_n>=20 THEN trading_value/nullif(value20,0) END trading_value_ratio_20d,
        trading_value/nullif(value_lag,0)-1 trading_value_change_1d,
        CASE WHEN hist_n>=20 THEN ret1>0 AND Volume/nullif(vol20,0)>1 END price_up_volume_up,
        CASE WHEN hist_n>=20 THEN abs(ret1)<=0.005 AND Volume/nullif(vol20,0)>1 END price_flat_volume_up,
        CASE WHEN hist_n>=21 THEN High>prior_hi20 AND Volume/nullif(vol20,0)>1 END new_high_volume_up,
        CASE WHEN hist_n>=21 THEN High>prior_hi20 END is_new_high_20d,
        CASE WHEN hist_n>=253 THEN High>prior_hi252 END is_new_high_52w
      FROM t
    ), eu AS (
      SELECT e.*,nullif(trim(u.sector),'') sector
      FROM e LEFT JOIN universe u ON e.Ticker=u.ticker
    ), sector_base AS (
      SELECT Date,sector,avg(return_5d) sector_return_5d,
        avg(return_20d) sector_return_20d,avg(return_60d) sector_return_60d,
        sum(trading_value) sector_trading_value,
        avg((return_1d>0)::INT) sector_advancer_ratio,
        avg(is_new_high_20d::INT) sector_new_high_20d_ratio,
        avg(is_new_high_52w::INT) sector_new_high_52w_ratio
      FROM eu WHERE sector IS NOT NULL GROUP BY Date,sector
    ), sector_context AS (
      SELECT *,sector_trading_value/nullif(lag(sector_trading_value,5)
        OVER(PARTITION BY sector ORDER BY Date),0)-1 sector_trading_value_change_5d
      FROM sector_base
    )
    SELECT eu.* EXCLUDE(sector),eu.sector,
      s.sector_return_5d,s.sector_return_20d,s.sector_return_60d,
      s.sector_trading_value,s.sector_trading_value_change_5d,s.sector_advancer_ratio,
      s.sector_new_high_20d_ratio,s.sector_new_high_52w_ratio,
      eu.return_20d-s.sector_return_20d equity_vs_sector_strength_20d,
      CASE WHEN eu.sector IS NULL THEN 'sector classification unavailable'
        ELSE 'current classification applied historically; look-ahead bias recorded' END sector_missing_reason,
      m.* EXCLUDE(Date){context}
    FROM eu LEFT JOIN sector_context s USING(Date,sector)
      LEFT JOIN market_context m USING(Date) LEFT JOIN named_context c USING(Date)
    """


def directory_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> int:
    a = args()
    repo = Path(__file__).resolve().parents[2]
    root = (repo / "momentum5d" / a.root).resolve()
    raw = root / "raw"
    sources: list[Path] = []
    for name in ["equities", "indexes", "external"]:
        sources += sorted((raw / name).rglob("*.parquet"))
    sources += sorted((root / "features/market_internals").rglob("*.parquet"))
    sources.append(raw / "universe.parquet")
    if not sources or any(not f.exists() for f in sources):
        raise RuntimeError("certified market-history inputs are incomplete")
    source_hash = fingerprint(sources)
    target = root / "features/equity_daily_features"
    report_path = root / "quality/step1_report.json"
    if report_path.exists() and target.exists() and not a.force:
        prior = json.loads(report_path.read_text(encoding="utf-8"))
        if prior.get("input_fingerprint") == source_hash and prior.get("quality") == "PASS":
            print(json.dumps({"status": "PASS", "reused": True}, ensure_ascii=False))
            return 0

    indexes = named_features(raw / "indexes", "index", False)
    external = named_features(raw / "external", "external", True)
    named = indexes.merge(external, on="Date", how="outer")
    # Keep a stable schema for explicitly requested but unavailable exact
    # series. Missingness remains visible and no proxy is silently substituted.
    for series in ["index_topix", "index_tse_growth_250", "external_us2y"]:
        for suffix in ["close", "change_1d", "change_5d", "change_20d", "vs_ma20"]:
            column = f"{series}_{suffix}"
            if column not in named:
                named[column] = float("nan")
    named["Date"] = pd.to_datetime(named["Date"]).dt.date
    context_columns = [c for c in named.columns if c != "Date"]

    work = Path(tempfile.mkdtemp(prefix="step1_", dir=root.parent))
    stage = work / "output"
    stage.mkdir()
    con = duckdb.connect(str(work / "work.duckdb"))
    con.execute(f"SET memory_limit='{a.memory_limit}'")
    con.execute("SET threads=2")
    spill = str(work / "spill").replace("'", "''")
    con.execute(f"SET temp_directory='{spill}'")
    con.execute(f"CREATE VIEW equities AS FROM read_parquet('{pg(raw / 'equities')}')")
    universe_path = str((raw / "universe.parquet").resolve()).replace("'", "''")
    con.execute(f"CREATE VIEW universe AS FROM read_parquet('{universe_path}')")
    con.execute(f"CREATE VIEW internals AS FROM read_parquet('{pg(root / 'features/market_internals')}')")
    con.register("named_df", named)
    con.execute("CREATE TABLE named_context AS SELECT Date::DATE Date,* EXCLUDE(Date) FROM named_df")
    con.execute("""CREATE TABLE market_context AS SELECT Date::DATE Date,
      advancers/nullif(advancers+decliners+unchanged,0) market_advancer_ratio,
      decliners/nullif(advancers+decliners+unchanged,0) market_decliner_ratio,
      advance_decline_ratio_25d market_advance_decline_ratio_25d,
      new_high_20d/nullif(advancers+decliners+unchanged,0) market_new_high_20d_ratio,
      new_low_20d/nullif(advancers+decliners+unchanged,0) market_new_low_20d_ratio,
      new_high_52w/nullif(advancers+decliners+unchanged,0) market_new_high_52w_ratio,
      new_low_52w/nullif(advancers+decliners+unchanged,0) market_new_low_52w_ratio,
      market_trading_value market_trading_value_total,
      trading_value_change market_trading_value_change_1d,
      advancing_trading_value_ratio FROM internals""")
    sql = feature_sql(context_columns)
    con.execute(f"CREATE VIEW feature_source AS {sql}")
    source_schema = con.execute("DESCRIBE feature_source").fetchdf()
    input_rows = con.execute("SELECT count(*) FROM equities").fetchone()[0]
    preserve_double = {
        "Open", "High", "Low", "Close", "Adj Close", "trading_value",
        "market_trading_value_total", "sector_trading_value",
    }
    selections = []
    for row in source_schema.itertuples(index=False):
        name = row.column_name
        data_type = str(row.column_type).upper()
        if data_type == "DOUBLE" and name not in preserve_double:
            selections.append(f"{quoted(name)}::FLOAT AS {quoted(name)}")
        else:
            selections.append(quoted(name))
    compact_select = ",".join(selections)
    output_file = str(stage / "part.parquet").replace("'", "''")
    con.execute(f"COPY (SELECT {compact_select} FROM feature_source) TO '{output_file}' "
                "(FORMAT PARQUET,COMPRESSION ZSTD)")
    # Closing finalizes all partition writers before the quality scan opens
    # the Parquet files. This avoids reading a footer while it is still being
    # atomically renamed by DuckDB.
    con.close()
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{a.memory_limit}'")
    con.execute(f"CREATE VIEW output AS SELECT * FROM read_parquet('{pg(stage)}')")
    summary = con.execute("""SELECT count(*),count(DISTINCT Ticker),count(DISTINCT Date),min(Date),max(Date),
      count(*)-count(DISTINCT (Ticker,Date)),
      sum((Ticker IS NULL OR Date IS NULL OR Open IS NULL OR High IS NULL OR Low IS NULL OR Close IS NULL OR Volume IS NULL)::INT),
      sum((Date>current_date)::INT) FROM output""").fetchone()
    sector_coverage = con.execute("SELECT avg((sector IS NOT NULL)::INT) FROM output").fetchone()[0]
    columns = con.execute("DESCRIBE output").fetchdf()["column_name"].tolist()
    null_sql = ",".join(f"avg(({quoted(c)} IS NULL)::INT)" for c in columns)
    null_values = con.execute(f"SELECT {null_sql} FROM output").fetchone()
    con.close()
    missing = dict(zip(columns, map(float, null_values), strict=True))
    available_missing = [rate for rate in missing.values() if rate < 1.0]
    rows,tickers,days,oldest,latest,duplicates,core_missing,future_rows = summary
    quality = "PASS" if rows == input_rows and not duplicates and not core_missing and not future_rows else "FAIL"
    if target.exists():
        shutil.rmtree(target)
    stage.rename(target)
    shutil.rmtree(work, ignore_errors=True)
    universe = pd.read_parquet(raw / "universe.parquet", columns=["ticker"])
    report: dict[str, Any] = {
        "step": 1, "status": "STEP1 complete" if quality == "PASS" else "STEP1 FAIL",
        "quality": quality, "created_at_jst": datetime.now(JST).isoformat(),
        "input_fingerprint": source_hash, "period": {"oldest": str(oldest), "latest": str(latest)},
        "tickers": int(tickers), "trading_days": int(days), "rows": int(rows),
        "feature_count": len(columns)-2,
        "overall_missing_rate": sum(missing.values())/len(missing),
        "available_feature_missing_rate": sum(available_missing)/len(available_missing),
        "missing_rate_by_feature": missing,
        "excluded_ticker_count": int(universe.ticker.nunique()-tickers),
        "duplicate_rows": int(duplicates), "future_rows": int(future_rows),
        "core_missing_rows": int(core_missing), "output_rows_match_input": rows == input_rows,
        "save_path": str(target), "file_size_bytes": directory_size(target),
        "recalculation_required": quality != "PASS",
        "completion_statement": "STEP1で作成した成果物は今後再計算不要" if quality == "PASS" else "STEP1再実行が必要",
        "created_files": [str(target), str(report_path), str(root / "quality/STEP2_PROMPT.md")],
        "sector_coverage_rate": float(sector_coverage or 0),
        "definitions": {"rsi_14": "Cutler trailing-14 RSI", "atr_14": "simple trailing-14 ATR",
                        "external_lag": "one source observation", "ma_order": "ma_*_rank_desc"},
        "residual_risks": [
            "TOPIX and TSE Growth 250 unavailable from certified exact-ticker source",
            "US 2-year yield unavailable; no proxy substituted",
            ("current sector classification is applied historically and can introduce look-ahead bias"
             if sector_coverage else "sector classification unavailable; sector features are NA"),
            "complete historical delisted membership unavailable; survivorship bias remains",
            "moving windows have expected warm-up NA; recent IPOs have shorter histories",
        ],
        "strategy_files_modified": [], "signal_conditions_created": False,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "quality/STEP2_PROMPT.md").write_text(
        """# STEP2 only: future returns, MFE and MAE\n\nSTEP1の認証済み成果物 `data/market_history/features/equity_daily_features/` を入力にし、STEP1を再計算せずSTEP2だけを実行してください。各銘柄・各営業日の5/10/20/40/60営業日先終値リターン、期間内MFE、MAEを作成してください。目的変数以外へ未来情報を混ぜず、現行戦略は使用・変更しないでください。新規Parquetへ保存し、期間別計算可能件数、欠損、重複、異常値、容量、品質、残存リスクを報告してください。完了時はSTEP2成果物を再計算不要と明記し、STEP3「大相場イベント抽出」だけの次回プロンプトを保存してください。\n""",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if quality == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
