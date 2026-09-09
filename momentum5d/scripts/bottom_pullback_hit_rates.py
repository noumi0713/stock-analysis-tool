"""Fixed causal bottom/pullback detection; two chronological halves, no fitting.

The five stages are specification, detection, forward labels, quality, hit rates.
Daily high target hits are hypothetical opportunities, not realised trade wins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import subprocess
import sys
import zipfile
import importlib.metadata
from collections import deque
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = "noumi0713/stock-analysis-tool"
ARTIFACT = 10020110334
RUN = 34125934863
ZIP_SHA = "5b5dc2a0903d9c37daddb67030bfaa9df749409efd44a9851f48735fb6ba3a4f"
SPEC = {
    "version": "bottom_pullback_hit_rates_v2",
    "targets": [0.05, 0.10], "target_operator": ">= (relative floating tolerance 1e-12)",
    "entry": "next_ticker_session_adjusted_open",
    "window": "entry row e through e+9 inclusive (10 sessions)",
    "stop_or_take_profit_exit": False, "cost_in_hit_rate": False,
    "ma_windows": [5, 25, 75], "ma75_slope_lag": 10,
    "atr": "Wilder14, seed first14 true ranges with prior close",
    "bottom_start": "prior C<MA75 & MA25<MA75 & MA75<lag10; current L<prior20 minL",
    "pullback_start": "prior C>MA75 & MA25>MA75 & MA75>lag10; last prior20 maxH age1..10; P-C>=priorATR14; C>MA75",
    "confirmation": "next1..20 rows: C>prior5 maxH & C>MA5 & L>=prior candidate minimum",
    "pullback_failure": "C<=MA75 before confirmation (failure wins ties)",
    "cooldown": "20 rows after confirmation, 5 after other termination",
    "confirmation_volume": "positive volume required; zero-volume row cannot confirm",
    "split": "median distinct source date; core ownership by signal date; state carried",
    "training": "none; identical fixed rules both halves; no parameter search",
    "period_policy": "entire original archive descriptively, previously used periods are NOT OOS",
    "session_calendar": "union of source equity dates; gaps treated unknown, not silently compressed",
    "five_stages": ["freeze_and_split", "causal_detection", "ten_session_targets", "quality", "pooled_hit_rates"],
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def parquet(path, rows, keys):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if not len(df):
        df = pd.DataFrame({k: pd.Series(dtype="str") for k in keys})
    df = df.sort_values(keys, kind="stable").reset_index(drop=True)
    if df.duplicated(keys).any():
        raise ValueError(f"duplicate output key: {path}")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="zstd")
    pd.testing.assert_frame_equal(df, pd.read_parquet(path), check_dtype=False)


def manifest(root, suffix="*.parquet"):
    return {str(p.relative_to(root)): sha(p) for p in sorted(Path(root).rglob(suffix)) if p.is_file()}


def period(date):
    return "formation_used" if date <= "2023-12-31" else ("validation_used" if date < "2025-09-08" else "contaminated_validation")


def positive(v):
    return v is not None and math.isfinite(float(v)) and v > 0


class Detector:
    """Only current/past bars enter this state machine; no outcome references."""
    def __init__(self, ticker):
        self.ticker = ticker
        self.i = -1
        self.last_session = None
        self.history = deque(maxlen=85)
        self.atr = None
        self.tr_seed = []
        self.candidate = None
        self.blocked_until = -1
        self.events = []
        self.audit = []

    def finish(self, status, date, reason=None):
        row = dict(self.candidate)
        row.update(end_date=date, status=status, reason=reason)
        self.events.append(row)
        self.candidate = None
        self.blocked_until = self.i + (20 if status == "confirmed" else 5)
        return row

    def step(self, bar):
        self.i += 1
        date = bar["Date"]
        gap = self.last_session is not None and bar["session"] != self.last_session + 1
        self.last_session = bar["session"]
        if gap or not bar["valid"]:
            reason = "missing_source_session" if gap else "invalid_ohlc"
            self.audit.append({"Ticker": self.ticker, "Date": date, "reason": reason})
            if self.candidate:
                self.finish("unknown", date, reason)
            self.history.clear()
            self.atr, self.tr_seed = None, []
            if not bar["valid"]:
                return None
        prev = list(self.history)
        prior_atr = self.atr
        if prev:
            pc = prev[-1]["c"]
            tr = max(bar["h"] - bar["l"], abs(bar["h"] - pc), abs(bar["l"] - pc))
            if self.atr is None:
                self.tr_seed.append(tr)
                if len(self.tr_seed) == 14:
                    self.atr = sum(self.tr_seed) / 14
            else:
                self.atr = (13 * self.atr + tr) / 14
        current = dict(bar, i=self.i)
        self.history.append(current)
        hist = list(self.history)
        if len(prev) < 85:
            return None
        prior_ma75 = sum(x["c"] for x in prev[-75:]) / 75
        prior_ma25 = sum(x["c"] for x in prev[-25:]) / 25
        prior_ma75_lag10 = sum(x["c"] for x in prev[-85:-10]) / 75
        ma75 = sum(x["c"] for x in hist[-75:]) / 75
        ma5 = sum(x["c"] for x in hist[-5:]) / 5
        pc = prev[-1]["c"]
        if self.candidate:
            c = self.candidate
            age = self.i - c["start_i"]
            old_min = c["min_low"]
            c["min_low"] = min(old_min, bar["l"])
            if c["kind"] == "pullback" and bar["c"] <= ma75:
                self.finish("invalidated", date, "close_below_ma75")
            elif (1 <= age <= 20 and bar["c"] > max(x["h"] for x in prev[-5:])
                  and bar["c"] > ma5 and bar["l"] >= old_min and positive(bar["v"])):
                row = self.finish("confirmed", date)
                return {"signal_id": row["event_id"], "Ticker": self.ticker, "kind": row["kind"],
                        "candidate_date": row["start_date"], "signal_date": date,
                        "signal_session": bar["session"], "confirmation_i": self.i,
                        "signal_half": bar["half"], "period": period(date),
                        "structural_low": row["min_low"], "atr14": self.atr}
            elif age >= 20:
                self.finish("expired", date, "twenty_sessions_no_confirmation")
            return None
        if self.i <= self.blocked_until:
            return None
        bottom = (pc < prior_ma75 and prior_ma25 < prior_ma75 and prior_ma75 < prior_ma75_lag10
                  and bar["l"] < min(x["l"] for x in prev[-20:]))
        p = max(prev[-20:], key=lambda x: (x["h"], x["i"]))
        pullback = (pc > prior_ma75 and prior_ma25 > prior_ma75 and prior_ma75 > prior_ma75_lag10
                    and 1 <= self.i - p["i"] <= 10 and positive(prior_atr)
                    and p["h"] - bar["c"] >= prior_atr and bar["c"] > ma75)
        if bottom or pullback:
            kind = "bottom" if bottom else "pullback"
            self.candidate = {"event_id": f"BP2|{self.ticker}|{date}|{kind}", "Ticker": self.ticker,
                              "kind": kind, "start_date": date, "start_i": self.i,
                              "start_half": bar["half"], "min_low": bar["l"],
                              "prior_ma25": prior_ma25, "prior_ma75": prior_ma75,
                              "prior_ma75_lag10": prior_ma75_lag10}
        return None


class Outcome:
    """Evaluation is a separate consumer, never read by Detector."""
    def __init__(self, signal):
        self.row = dict(signal, entry_date=None, entry_price=None, n_observed=0,
                        eval_end_date=None, status="pending", reason=None, mfe=None, mae=None,
                        hit_5=None, hit_10=None, hit_5_date=None, hit_10_date=None,
                        hit_5_session=None, hit_10_session=None, close_return=None,
                        net_close_return=None, crosses_historical_boundary=False)
        self.last_session = signal["signal_session"]
        self.hi, self.lo = -math.inf, math.inf

    def step(self, bar):
        r = self.row
        if r["status"] != "pending":
            return
        if bar["session"] != self.last_session + 1:
            r.update(status="unknown", reason="missing_source_session_in_window")
            return
        self.last_session = bar["session"]
        if not bar["valid"]:
            r.update(status="unknown", reason="invalid_ohlc_in_window")
            return
        if r["entry_date"] is None:
            # Daily positive volume does NOT prove that an opening auction fill occurred.
            if not positive(bar["v"]):
                r.update(status="unfilled", reason="entry_volume_missing_or_zero")
                return
            r.update(entry_date=bar["Date"], entry_price=bar["o"], hit_5=False, hit_10=False)
        elif not positive(bar["v"]):
            r.update(status="unknown", reason="zero_or_missing_volume_in_window")
            return
        r["n_observed"] += 1
        r["eval_end_date"] = bar["Date"]
        r["crosses_historical_boundary"] |= period(bar["Date"]) != r["period"]
        self.hi, self.lo = max(self.hi, bar["h"]), min(self.lo, bar["l"])
        for pct in (5, 10):
            barrier = r["entry_price"] * (1 + pct / 100)
            reached = bar["h"] >= barrier or math.isclose(bar["h"], barrier, rel_tol=1e-12, abs_tol=0)
            if not r[f"hit_{pct}"] and reached:
                r[f"hit_{pct}"] = True
                r[f"hit_{pct}_date"] = bar["Date"]
                r[f"hit_{pct}_session"] = r["n_observed"]
        if r["n_observed"] == 10:
            r.update(status="complete", mfe=self.hi / r["entry_price"] - 1,
                     mae=self.lo / r["entry_price"] - 1,
                     close_return=bar["c"] / r["entry_price"] - 1,
                     net_close_return=bar["c"] / r["entry_price"] - 1 - 0.004)

    def close(self):
        if self.row["status"] == "pending":
            self.row.update(status="right_censored", reason="ten_session_window_unavailable")


def stats(df):
    eligible = df[df.status.eq("complete")]
    n = len(eligible)
    result = {"signals": len(df), "unique_tickers": df.Ticker.nunique(), "evaluable": n,
              "unknown_or_unfilled": len(df) - n}
    for pct in (5, 10):
        k = int(eligible[f"hit_{pct}"].sum())
        result[f"success_{pct}"] = k
        result[f"hit_rate_{pct}"] = k / n if n else None
        result[f"all_signals_lower_bound_{pct}"] = k / len(df) if len(df) else None
        result[f"all_signals_upper_bound_{pct}"] = (k + len(df) - n) / len(df) if len(df) else None
    result.update(mean_mfe=float(eligible.mfe.mean()) if n else None,
                  median_mae=float(eligible.mae.median()) if n else None,
                  close_net_win_rate=float((eligible.net_close_return > 0).mean()) if n else None,
                  mean_net_close_return=float(eligible.net_close_return.mean()) if n else None)
    return result


def summarise(rows):
    columns = ["signal_id", "Ticker", "kind", "signal_half", "signal_date", "period", "status", "hit_5", "hit_10", "mfe", "mae", "net_close_return", "crosses_historical_boundary"]
    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    result = []
    for scope in ["all", "half_1", "half_2", "formation_used", "validation_used", "contaminated_validation"]:
        base = df if scope == "all" else (df[df.signal_half.eq(int(scope[-1]))] if scope.startswith("half") else df[df.period.eq(scope) & ~df.crosses_historical_boundary.astype(bool)])
        for kind in ["all", "bottom", "pullback"]:
            g = base if kind == "all" else base[base.kind.eq(kind)]
            result.append({"scope": scope, "kind": kind, **stats(g)})
    return pd.DataFrame(result)


def restore(root):
    """Only normal Actions GITHUB_TOKEN via gh; never extract other credentials."""
    destination = root / "certified_input"
    if destination.exists():
        raise RuntimeError("Refusing to overwrite certified input directory")
    meta = json.loads(subprocess.check_output(["gh", "api", f"repos/{REPO}/actions/artifacts/{ARTIFACT}"]))
    if (meta["id"] != ARTIFACT or meta["name"] != "step1-equity-features" or meta["expired"]
            or meta["workflow_run"]["id"] != RUN or meta["digest"] != f"sha256:{ZIP_SHA}"):
        raise RuntimeError("Pinned artifact identity mismatch")
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "certified_step1.zip"
    with archive.open("wb") as f:
        subprocess.run(["gh", "api", f"repos/{REPO}/actions/artifacts/{ARTIFACT}/zip"], stdout=f, check=True)
    if sha(archive) != ZIP_SHA:
        raise RuntimeError("Pinned archive SHA256 mismatch")
    destination.mkdir()
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP CRC failure")
        for info in z.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise RuntimeError("Unsafe archive path")
        z.extractall(destination)
    provenance = {"repository": REPO, "run_id": RUN, "artifact_id": ARTIFACT,
                  "archive_sha256": ZIP_SHA, "archive_bytes": archive.stat().st_size,
                  "extracted_manifest": manifest(destination, "*"), "metadata": meta}
    dump(root / "quality/input_provenance.json", provenance)


def sources(root):
    matches = sorted((root / "certified_input").rglob("equity_daily_features"))
    if len(matches) != 1:
        raise RuntimeError("Exactly one authenticated STEP1 feature root required")
    return matches[0]


def split(root):
    source = sources(root)
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=2")
    glob = str(source / "**/*.parquet").replace("'", "''")
    con.execute(f"CREATE VIEW src AS SELECT * FROM read_parquet('{glob}')")
    missing_keys, dupes = con.execute("SELECT count(*) FILTER (WHERE Ticker IS NULL OR Date IS NULL),count(*)-count(DISTINCT (Ticker,Date)) FROM src").fetchone()
    if missing_keys or dupes:
        raise RuntimeError(f"Input keys invalid: missing={missing_keys}, duplicate={dupes}")
    dates = [str(r[0]) for r in con.execute("SELECT DISTINCT CAST(Date AS DATE) FROM src ORDER BY 1").fetchall()]
    if len(dates) < 2:
        raise RuntimeError("Need at least two source dates")
    mid = dates[(len(dates) - 1) // 2]
    # Only identifiers/dates used to choose the split. No target values or results.
    session = pd.DataFrame({"Date": pd.to_datetime(dates), "session": range(len(dates))})
    con.register("sessions", session)
    halves = []
    (root / "halves").mkdir()
    for half, op in [(1, "<="), (2, ">")]:
        path = root / f"halves/half_{half}.parquet"
        sql = f'''SELECT CAST(s.Ticker AS VARCHAR) Ticker, CAST(s.Date AS DATE) Date,
             CAST(s.Open AS DOUBLE) Open, CAST(s.High AS DOUBLE) High,
             CAST(s.Low AS DOUBLE) Low, CAST(s.Close AS DOUBLE) Close,
             CAST(s."Adj Close" AS DOUBLE) "Adj Close", CAST(s.Volume AS DOUBLE) Volume,
             d.session, {half} half FROM src s JOIN sessions d ON CAST(s.Date AS DATE)=d.Date
             WHERE CAST(s.Date AS DATE) {op} DATE '{mid}' ORDER BY Ticker,Date'''
        con.execute(f"COPY ({sql}) TO '{str(path)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        n = pq.ParquetFile(path).metadata.num_rows
        halves.append({"half": half, "rows": n, "bytes": path.stat().st_size, "sha256": sha(path)})
    total = con.execute("SELECT count(*) FROM src").fetchone()[0]
    assert sum(h["rows"] for h in halves) == total
    con.close()
    report = {"split_date": mid, "first_date": dates[0], "last_date": dates[-1],
              "source_dates": len(dates), "rows": total, "halves": halves,
              "split_purpose": "memory only, NOT train/test or OOS separation"}
    dump(root / "quality/split_manifest.json", report)
    return report


def bars(path):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=8192):
        names = batch.schema.names
        arrays = [batch.column(i).to_pylist() for i in range(len(names))]
        for values in zip(*arrays):
            row = dict(zip(names, values))
            valid = all(positive(row[k]) for k in ["Open", "High", "Low", "Close", "Adj Close"])
            valid = valid and row["Low"] <= min(row["Open"], row["Close"]) <= max(row["Open"], row["Close"]) <= row["High"]
            f = row["Adj Close"] / row["Close"] if valid else 1.0
            yield {"Ticker": row["Ticker"], "Date": str(row["Date"]),
                   "session": row["session"], "half": row["half"], "valid": valid,
                   "o": row["Open"] * f if valid else None,
                   "h": row["High"] * f if valid else None,
                   "l": row["Low"] * f if valid else None,
                   "c": row["Adj Close"] if valid else None, "v": row["Volume"]}


def run_stream(paths, out):
    detectors, pending = {}, {}
    signals, outcomes = [], []
    counts = []
    for number, path in enumerate(paths, 1):
        n = 0
        for bar in bars(path):
            ticker = bar["Ticker"]
            if ticker in pending:
                pending[ticker].step(bar)
                if pending[ticker].row["status"] != "pending":
                    del pending[ticker]
            detector = detectors.setdefault(ticker, Detector(ticker))
            signal = detector.step(bar)
            if signal:
                if ticker in pending:
                    raise RuntimeError("Overlapping pending signals despite fixed cooldown")
                signals.append(signal)
                target = Outcome(signal)
                outcomes.append(target)
                pending[ticker] = target
            n += 1
        # Serialized history includes Wilder ATR and candidate state, preserving exact continuity.
        checkpoint = out / f"checkpoint_half_{number}.pkl"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("wb") as f:
            pickle.dump((detectors, pending, signals, outcomes), f, protocol=5)
        with checkpoint.open("rb") as f:
            detectors, pending, signals, outcomes = pickle.load(f)
        counts.append({"half": number, "core_rows": n, "signals_cumulative": len(signals),
                       "pending_evaluation": len(pending), "checkpoint_sha256": sha(checkpoint)})
        print(json.dumps(counts[-1]), flush=True)
    for target in outcomes:
        target.close()
    event_rows, audit = [], []
    for detector in detectors.values():
        event_rows.extend(detector.events)
        if detector.candidate:
            event_rows.append(dict(detector.candidate, end_date=None, status="right_censored", reason="candidate_window_unavailable"))
        audit.extend(detector.audit)
    rows = [x.row for x in outcomes]
    summary = summarise(rows)
    for half in (1, 2):
        folder = out / f"half_{half}"
        parquet(folder / "signals.parquet", [r for r in signals if r["signal_half"] == half], ["signal_id"])
        parquet(folder / "outcomes.parquet", [r for r in rows if r["signal_half"] == half], ["signal_id"])
        parquet(folder / "candidates.parquet", [r for r in event_rows if r["start_half"] == half], ["event_id"])
    parquet(out / "audit.parquet", audit, ["Ticker", "Date", "reason"])
    parquet(out / "summary.parquet", summary, ["scope", "kind"])
    df = pd.DataFrame(rows)
    annual = []
    if len(df):
        for (year, kind), g in df.groupby([df.signal_date.str[:4], "kind"], sort=True):
            annual.append({"year": year, "kind": kind, **stats(g)})
    parquet(out / "annual.parquet", annual, ["year", "kind"])
    ticker_metrics = []
    if len(df):
        for ticker, g in df.groupby("Ticker", sort=True):
            ticker_metrics.append({"Ticker": ticker, **stats(g)})
    parquet(out / "per_ticker.parquet", ticker_metrics, ["Ticker"])
    assert len(signals) == len(rows) == len({r["signal_id"] for r in rows})
    for r in rows:
        if r["status"] == "complete":
            assert r["n_observed"] == 10 and r["entry_date"] > r["signal_date"]
            assert not r["hit_10"] or r["hit_5"]
            assert math.isfinite(r["mfe"]) and math.isfinite(r["mae"])
    # Make missing/status counts explicit; don't turn missing targets into failures.
    status_counts = df.status.value_counts().to_dict() if len(df) else {}
    return {"summary": json.loads(summary.to_json(orient="records")), "status_counts": status_counts,
            "candidates": len(event_rows), "signals": len(signals), "tickers": len(detectors),
            "audit_rows": len(audit), "core_processing": counts,
            "cross_historical_boundary": sum(r["crosses_historical_boundary"] for r in rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/bottom_pullback_hit_rates_v2")
    p.add_argument("--restore", action="store_true")
    args = p.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    dump(root / "quality/frozen_specification.json", SPEC)
    dump(root / "quality/frozen_specification_sha256.json", {"sha256": sha(root / "quality/frozen_specification.json")})
    if args.restore:
        restore(root)
    if not (root / "certified_step1.zip").is_file() or sha(root / "certified_step1.zip") != ZIP_SHA:
        raise RuntimeError("Authenticated original archive required; reports alone insufficient")
    before = manifest(root / "certified_input", "*")
    provenance = json.loads((root / "quality/input_provenance.json").read_text())
    if before != provenance["extracted_manifest"]:
        raise RuntimeError("Extracted files differ from pinned archive restoration")
    split_report = split(root)
    paths = [root / f"halves/half_{i}.parquet" for i in (1, 2)]
    first = run_stream(paths, root / "run_1")
    second = run_stream(paths, root / "run_2")
    a, b = manifest(root / "run_1"), manifest(root / "run_2")
    if a != b or first != second:
        raise RuntimeError("Independent execution reproducibility mismatch")
    if before != manifest(root / "certified_input", "*"):
        raise RuntimeError("Certified inputs changed")
    report = {"quality": "PASS", "execution": "completed", "untouched_oos": False,
              "library_versions": {k: importlib.metadata.version(k) for k in ["numpy", "pandas", "pyarrow", "duckdb"]},
              "selection_or_optimization_performed": False,
              "specification_sha256": sha(root / "quality/frozen_specification.json"),
              "split": split_report, **first, "output_parquet_sha256": a,
              "independent_two_run_reproducibility": "PASS", "inputs_changed": False,
              "duplicate_signal_keys": 0, "future_outcomes_used_by_detector": False,
              "execution_model": "hypothetical next open; 10 bars including entry; no stop",
              "risks": ["Fixed baseline rules, not validated adoption signals", "All dates previously used: not OOS",
                        "Daily high is not guaranteed executable profit; positive daily volume does not certify opening liquidity",
                        "Source session calendar inferred from union of equity dates", "Universe may omit delisted securities",
                        "Correlated signals: hit rate is descriptive, not an independent Bernoulli probability",
                        "Adjusted prices reflect source corporate actions policy; cost excluded from hit labels",
                        "No portfolio, realised-profit or ruin claim; 10-day net close return is supplemental"]}
    dump(root / "quality/report.json", report)
    md = "# 底打ち・押し目 10営業日到達率\n\n品質: PASS。完全未使用OOSではありません。\n\n"
    md += pd.DataFrame(first["summary"]).to_markdown(index=False)
    md += "\n\n+5%/+10%は日中高値による含み益到達率。利確・損切りを適用した取引勝率ではない。分母は完全な10営業日の評価可能シグナル。未約定・欠損・右打ち切りは別件数。\n"
    md += "\nリスク:\n" + "\n".join("- " + r for r in report["risks"])
    (root / "quality/REPORT.md").write_text(md + "\n")
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        root_arg = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else "data/bottom_pullback_hit_rates_v2"
        dump(Path(root_arg) / "quality/FAIL.json", {"quality": "FAIL", "reason": str(exc), "untouched_oos": False})
        raise
