"""Formation-only, immutable-label, single-feature incremental-value research."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path
from statistics import NormalDist

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = "noumi0713/stock-analysis-tool"
CUTOFF = "2023-12-31"
PINS = {
    "baseline": (34306366984, 10086890141, "bottom-pullback-hit-rates-v2",
                 "4f4aca96b3f48fe1c88cfa43d7b6d184371e9a4729feff91ea738b26fbc6a15b"),
    "step1": (34125934863, 10020110334, "step1-equity-features",
              "5b5dc2a0903d9c37daddb67030bfaa9df749409efd44a9851f48735fb6ba3a4f"),
}
FEATURES = {
    "volume20": ("volume_ratio_20d", 19, True),
    "value20": ("trading_value_ratio_20d", 19, True),
    "rsi_change5": ("rsi_14", 19, True),
    "lower_wick": ("lower_wick_ratio", 0, False),
    "upper_wick": ("upper_wick_ratio", 0, False),
    "vol20_vol60": ("volatility_contraction_ratio", 60, True),
    "relative_sector20": ("equity_vs_sector_strength_20d", 20, True),
    "relative_topix20": ("index_topix_change_20d", 20, True),
}
FOLDS = [
    (1, "2021-08-19", "2022-06-30", "2022-07-01", "2022-12-31"),
    (2, "2021-08-19", "2022-12-31", "2023-01-01", "2023-06-30"),
    (3, "2021-08-19", "2023-06-30", "2023-07-01", "2023-12-31"),
]
SPEC = {
    "version": "bottom_pullback_distortions_v1", "pins": PINS, "features": FEATURES,
    "formation_end": CUTOFF, "folds": FOLDS, "primary": "hit_5", "secondary": "hit_10",
    "confirmation_availability": "Tokyo close; next-session open entry unchanged",
    "quantiles": [0.3, 0.7], "quantile_interpolation": "linear, training valid samples only",
    "direction": "high when training primary-outcome rank AUC >=0.5; otherwise low",
    "filter": "one feature only; high >=q70, low <=q30; missing never passes",
    "candidate_dependency": "candidate start minus85 ticker rows; pullback also entire recursive ATR since last source gap/reset",
    "feature_dependency": "exact trailing rows, including RSI14 at t and t-5; no recomputation of STEP1",
    "purge": "candidate and feature dependencies and full existing target window inside each train/test interval",
    "price_consistency": "multiday raw-price/volume ratios only when AdjClose/Close stays constant within 1e-5 relative over dependency; otherwise NA; wicks same-day scale invariant",
    "sector_policy": "exclude current classification applied historically: point-in-time unavailable",
    "missing_policy": "retain reasons; no imputation; complete labels only; exclude post-2023 target values before projection",
    "inference": "approximate cluster sandwich SE by ticker, signal date, calendar week; use maximum SE; normal lower bound Bonferroni 48 (8 features x2 types x3 tests)",
    "reproducibility_rule": "all3 later folds: same training direction, retained>=100, >=30 dates, >=30 tickers, simultaneous lower bound of hit5 increment >0; otherwise no added filter/research hold",
    "adoption": "never automatic; no trading or untouched OOS claim",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def manifest(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(Path(root).rglob("*")) if p.is_file()}


def save(root, name, frame, keys):
    df = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    if not len(df):
        for key in keys:
            if key not in df:
                df[key] = pd.Series(dtype="str")
    df = df.sort_values(keys, kind="stable").reset_index(drop=True)
    if df.duplicated(keys).any():
        raise ValueError("duplicate keys: " + name)
    nums = df.select_dtypes(include="number")
    if np.isinf(nums.to_numpy(dtype=float)).any():
        raise ValueError("infinite output: " + name)
    path = Path(root) / name / "part.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="zstd")
    pd.testing.assert_frame_equal(df, pd.read_parquet(path), check_dtype=False)


def gh(path):
    return json.loads(subprocess.check_output(["gh", "api", f"repos/{REPO}/{path}"]))


def checkpoint_records(logs):
    records = []
    for line in logs.splitlines():
        try:
            value = json.loads(line[line.index("{"):])
        except (ValueError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and "half" in value and "checkpoint_sha256" in value:
            records.append(value)
    return records


def restore(work):
    origins = {}
    for label, (run, artifact, name, digest) in PINS.items():
        meta = gh(f"actions/artifacts/{artifact}")
        state = gh(f"actions/runs/{run}")
        assert meta["id"] == artifact and meta["name"] == name and not meta["expired"]
        assert meta["workflow_run"]["id"] == run and meta["digest"] == "sha256:" + digest
        assert state["conclusion"] == "success"
        archive = work / (label + ".zip")
        if not archive.exists():
            with archive.open("wb") as f:
                subprocess.run(["gh", "api", f"repos/{REPO}/actions/artifacts/{artifact}/zip"], stdout=f, check=True)
        assert sha(archive) == digest, "ZIP authentication failed"
        target = work / label
        target.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for info in z.infolist():
                dst = (target / info.filename).resolve()
                assert dst.is_relative_to(target.resolve()) and (info.external_attr >> 16 & 0o170000) != 0o120000
            z.extractall(target)
        origins[label] = {"metadata": meta, "run_head_sha": state["head_sha"], "zip_sha256": digest}
    # Original execution log authenticates successful tests and both independent state chains.
    # Capture as data, never render terminal control bytes. Test logs contain ANSI colour.
    logs = subprocess.check_output(["gh", "api", "--allow-escape-sequences", f"repos/{REPO}/actions/jobs/102323788013/logs"]).decode()
    assert "14 passed" in logs
    report = json.loads((work / "baseline/quality/report.json").read_text())
    checkpoints = checkpoint_records(logs)
    for core in report["core_processing"]:
        assert sum(core["checkpoint_sha256"] == item["checkpoint_sha256"] for item in checkpoints) == 2
    origins["baseline"]["independent_state_chains_verified_in_original_log"] = True
    dump(work / "origins.json", origins)


def authenticate(work):
    for label, pin in PINS.items():
        assert sha(work / (label + ".zip")) == pin[3]
    b = work / "baseline"
    r = json.loads((b / "quality/report.json").read_text())
    assert r["quality"] == "PASS" and not r["inputs_changed"]
    assert r["independent_two_run_reproducibility"] == "PASS"
    assert not r["future_outcomes_used_by_detector"] and r["duplicate_signal_keys"] == 0
    assert r["signals"] == 67844 and r["status_counts"]["complete"] == 65414
    total = next(x for x in r["summary"] if x["scope"] == "all" and x["kind"] == "all")
    assert (total["success_5"], total["success_10"]) == (23815, 9951)
    files = {str(p.relative_to(b / "run_1")): sha(p) for p in (b / "run_1").rglob("*.parquet")}
    assert files == r["output_parquet_sha256"]
    # Footer counts do not read held-out rows or their outcomes.
    for name in ("signals", "outcomes"):
        assert sum(pq.ParquetFile(b / f"run_1/half_{h}/{name}.parquet").metadata.num_rows for h in (1, 2)) == 67844
    provenance = json.loads((b / "quality/input_provenance.json").read_text())
    assert provenance["archive_sha256"] == PINS["step1"][3]
    assert manifest(work / "step1") == provenance["extracted_manifest"]
    s1 = json.loads((work / "step1/quality/step1_report.json").read_text())
    assert s1["quality"] == "PASS"
    assert s1["missing_rate_by_feature"]["index_topix_change_20d"] == 1.0
    assert sha(b / "quality/frozen_specification.json") == r["specification_sha256"]
    origins = json.loads((work / "origins.json").read_text())
    assert origins["baseline"]["independent_state_chains_verified_in_original_log"]
    return {"quality": "PASS", "baseline_zip": PINS["baseline"][3], "step1_zip": PINS["step1"][3],
            "all_signal_rows_footer_verified": 67844, "complete_count_certified_manifest_report": 65414,
            "hit5_count_certified_manifest_report": 23815, "hit10_count_certified_manifest_report": 9951,
            "global_outcome_values_rescanned": False,
            "count_verification_method": "exact immutable ZIP + all Parquet hashes + original certified report; no post2023 labels reread",
            "baseline_state_carry_and_independent_runs": "original code + successful14 tests + matching checkpoint hashes twice in original run log",
            "origins": origins}


def literal(path):
    return "'" + str(path).replace("'", "''") + "'"


def read_formation(work):
    con = duckdb.connect()
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=2")
    src = work / "step1/features/equity_daily_features/part.parquet"
    cols = ["Ticker", "Date", "Close", "Adj Close", "volume_ratio_20d", "trading_value_ratio_20d", "rsi_14",
            "lower_wick_ratio", "upper_wick_ratio", "volatility_contraction_ratio"]
    quoted = ",".join('"' + c + '"' for c in cols)
    # Predicate applies at the source; post-2023 values never enter analysis dataframes.
    prices = con.execute(f"SELECT {quoted} FROM read_parquet({literal(src)}) WHERE Date<=DATE '{CUTOFF}' ORDER BY Ticker,Date").fetchdf()
    base = work / "baseline/run_1/half_1/outcomes.parquet"
    meta = con.execute(f"SELECT signal_id,Ticker,kind,candidate_date,signal_date,entry_date,eval_end_date,status,reason FROM read_parquet({literal(base)}) WHERE signal_date<='{CUTOFF}' ORDER BY signal_id").fetchdf()
    # Only complete, wholly formation-period targets are projected.
    labels = con.execute(f"SELECT signal_id,hit_5,hit_10,mae,close_return,net_close_return FROM read_parquet({literal(base)}) WHERE signal_date<='{CUTOFF}' AND eval_end_date<='{CUTOFF}' AND status='complete' ORDER BY signal_id").fetchdf()
    audit = con.execute(f"SELECT Ticker,Date,reason FROM read_parquet({literal(work / 'baseline/run_1/audit.parquet')}) WHERE Date<='{CUTOFF}' ORDER BY Ticker,Date").fetchdf()
    con.close()
    assert not prices.duplicated(["Ticker", "Date"]).any()
    assert not meta.signal_id.duplicated().any() and not labels.signal_id.duplicated().any()
    assert str(prices.Date.max().date()) <= CUTOFF
    return prices, meta, labels, audit


def build_samples(prices, meta, labels, resets):
    """Uses existing fields and date dependencies, not forward-label calculation."""
    prices = prices.copy()
    for df, cols in ((prices, ["Date"]), (meta, ["candidate_date", "signal_date", "entry_date", "eval_end_date"]), (resets, ["Date"])):
        for c in cols:
            df[c] = pd.to_datetime(df[c])
    assert (prices.Date <= CUTOFF).all()
    grouped = {str(t): g.reset_index(drop=True) for t, g in prices.groupby("Ticker", sort=True)}
    reset_groups = {str(t): g.Date.sort_values().tolist() for t, g in resets.groupby("Ticker", sort=True)}
    rows, exclusions = [], []
    complete_ids = set(labels.signal_id)
    for row in meta.itertuples(index=False):
        if row.signal_id not in complete_ids:
            why = row.reason or row.status
            if pd.notna(row.eval_end_date) and row.eval_end_date > pd.Timestamp(CUTOFF):
                why = "target_crosses_formation_end"
            exclusions.append({"signal_id": row.signal_id, "kind": row.kind, "reason": why or "incomplete_label"})
    for ticker, group in meta[meta.signal_id.isin(complete_ids)].groupby("Ticker", sort=True):
        p = grouped[str(ticker)]
        dates = p.Date.tolist()
        positions = {d: i for i, d in enumerate(dates)}
        factor = (p["Adj Close"] / p["Close"]).to_numpy(dtype=float)
        rsi_change = p.rsi_14 - p.rsi_14.shift(5)
        roots = reset_groups.get(str(ticker), [])
        for row in group.itertuples(index=False):
            i, c = positions.get(row.signal_date), positions.get(row.candidate_date)
            assert i is not None and c is not None and c <= i
            dep = dates[c - 85] if c >= 85 else pd.NaT
            atr_root = max([dates[0]] + [d for d in roots if d <= row.candidate_date])
            if row.kind == "pullback" and pd.notna(dep):
                dep = min(dep, atr_root)
            rec = row._asdict()
            rec.update(candidate_dependency_start=dep, atr_dependency_start=atr_root if row.kind == "pullback" else pd.NaT)
            for feature, (column, lag, adjustment) in FEATURES.items():
                start = dates[i-lag] if i >= lag else pd.NaT
                reason, value = None, np.nan
                if feature == "relative_sector20":
                    reason = "historical_sector_membership_not_point_in_time"
                elif feature == "relative_topix20":
                    reason = "exact_TOPIX_unavailable_in_certified_STEP1"
                elif i < lag:
                    reason = "insufficient_feature_history"
                else:
                    value = float(rsi_change.iloc[i] if feature == "rsi_change5" else p.iloc[i][column])
                    f = factor[i-lag:i+1]
                    if adjustment and (not np.isfinite(f).all() or np.any(f <= 0) or np.max(f)/np.min(f)-1 > 1e-5):
                        value, reason = np.nan, "adjustment_not_constant_over_dependency"
                    elif not np.isfinite(value):
                        value, reason = np.nan, "source_feature_missing_or_undefined"
                    elif feature in ("lower_wick", "upper_wick") and not -1e-6 <= value <= 1+1e-6:
                        value, reason = np.nan, "wick_out_of_range"
                rec[feature], rec[feature + "__reason"], rec[feature + "__start"] = value, reason, start
            rows.append(rec)
    samples = pd.DataFrame(rows).merge(labels, on="signal_id", how="left", validate="one_to_one")
    for c in ("hit_5", "hit_10"):
        assert samples[c].notna().all() and samples[c].isin([True, False, 0, 1]).all()
        samples[c] = samples[c].astype(int)
    assert samples.eval_end_date.le(CUTOFF).all()
    assert samples.mae.between(-1, 0).all() and np.isfinite(samples.close_return).all()
    return samples, exclusions


def auc(x, y):
    x, y = pd.Series(x).reset_index(drop=True), pd.Series(y).reset_index(drop=True)
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    n1, n0 = int(y.sum()), int((1-y).sum())
    if not n1 or not n0:
        return np.nan
    return float((x.rank(method="average")[y == 1].sum()-n1*(n1+1)/2)/(n1*n0))


def fit(train, feature):
    valid = train[train[feature].notna()]
    a = auc(valid[feature], valid.hit_5)
    if not len(valid) or not np.isfinite(a):
        return {"direction": "unavailable", "q30": np.nan, "q70": np.nan, "train_auc": a, "train_valid": len(valid)}
    return {"direction": "high" if a >= .5 else "low", "q30": float(valid[feature].quantile(.3)),
            "q70": float(valid[feature].quantile(.7)), "train_auc": a, "train_valid": len(valid)}


def apply_filter(df, feature, definition):
    if definition["direction"] == "unavailable":
        return pd.Series(False, index=df.index)
    return (df[feature] >= definition["q70"]) if definition["direction"] == "high" else (df[feature] <= definition["q30"])


def interval_mask(df, lo, hi, feature=None):
    mask = df.signal_date.between(lo, hi) & df.eval_end_date.le(hi) & df.candidate_dependency_start.ge(lo)
    if feature:
        mask &= df[feature + "__start"].ge(lo)
    return mask


def mean(series):
    return float(series.mean()) if len(series) else np.nan


def stats(df):
    return {"n": len(df), "tickers": df.Ticker.nunique(), "dates": df.signal_date.nunique(),
            "success5": int(df.hit_5.sum()), "success10": int(df.hit_10.sum()),
            "rate5": mean(df.hit_5), "rate10": mean(df.hit_10), "mean_mae": mean(df.mae),
            "mean_close_return": mean(df.close_return), "mean_net_close_return": mean(df.net_close_return)}


def cluster_se(frame, selected, key):
    n, m = len(frame), int(selected.sum())
    if n < 2 or m < 2:
        return np.nan
    y = frame.hit_5.to_numpy(dtype=float)
    a = selected.to_numpy(dtype=float)
    influence = a*(y-y[a == 1].mean())/m - (y-y.mean())/n
    sums = pd.Series(influence).groupby(pd.Series(key).reset_index(drop=True)).sum()
    k = len(sums)
    return float(np.sqrt(k/(k-1)*np.square(sums).sum())) if k > 1 else np.nan


def evaluation(frame, feature, definition):
    selected = apply_filter(frame, feature, definition)
    retained = frame[selected]
    baseline = stats(frame)
    row = {"baseline_" + k: v for k, v in baseline.items()}
    row.update({"retained_" + k: v for k, v in stats(retained).items()})
    row.update(valid=int(frame[feature].notna().sum()), missing=int(frame[feature].isna().sum()))
    row["retention_rate"] = len(retained)/len(frame) if len(frame) else np.nan
    for n in (5, 10):
        b, a = baseline[f"success{n}"], int(retained[f"hit_{n}"].sum())
        row[f"missed_success_rate{n}"] = 1-a/b if b else np.nan
        row[f"delta_rate{n}"] = row[f"retained_rate{n}"] - row[f"baseline_rate{n}"]
    row["delta_mean_mae"] = row["retained_mean_mae"] - baseline["mean_mae"]
    row["delta_mean_net_close_return"] = row["retained_mean_net_close_return"] - baseline["mean_net_close_return"]
    ses = []
    for name, keys in (("ticker", frame.Ticker), ("date", frame.signal_date), ("week", frame.signal_date.dt.to_period("W").astype(str))):
        se = cluster_se(frame, selected, keys)
        row["cluster_se_" + name] = se
        ses.append(se)
    row["simultaneous_lower_delta5"] = (row["delta_rate5"] - NormalDist().inv_cdf(1-.05/48)*max(ses)) if all(np.isfinite(ses)) else np.nan
    row["max_ticker_share"] = float(retained.Ticker.value_counts(normalize=True).max()) if len(retained) else np.nan
    row["max_date_share"] = float(retained.signal_date.value_counts(normalize=True).max()) if len(retained) else np.nan
    return row, selected


def describe(samples):
    output = []
    for kind, group in samples.groupby("kind", sort=True):
        for feature in FEATURES:
            good, bad = group.loc[group.hit_5 == 1, feature].dropna(), group.loc[group.hit_5 == 0, feature].dropna()
            a = auc(group[feature], group.hit_5)
            denom = math.sqrt((good.var() + bad.var())/2) if len(good)>1 and len(bad)>1 else np.nan
            for outcome, values in ((1, good), (0, bad)):
                output.append({"kind": kind, "feature": feature, "outcome": outcome,
                               "total": int((group.hit_5 == outcome).sum()), "valid": len(values),
                               "missing": int((group.hit_5 == outcome).sum())-len(values),
                               "median": float(values.median()), "q25": float(values.quantile(.25)), "q75": float(values.quantile(.75)),
                               "mean": mean(values), "std": float(values.std()),
                               "raw_auc": a, "rank_biserial": 2*a-1,
                               "standardized_mean_difference": (mean(good)-mean(bad))/denom if denom>0 else np.nan})
    return output


def run(work, out):
    prices, meta, labels, resets = read_formation(work)
    samples, incomplete = build_samples(prices, meta, labels, resets)
    del prices
    assert len(samples) == 26393 and int(samples.hit_5.sum()) == 9146 and int(samples.hit_10.sum()) == 3674
    save(out, "samples", samples, ["signal_id"])
    save(out, "incomplete_or_boundary_labels", incomplete, ["signal_id"])
    eligibility = []
    for feature, (source, lag, adjustment) in FEATURES.items():
        exclusion = "historical_sector_membership_not_point_in_time" if feature == "relative_sector20" else (
            "exact_TOPIX_unavailable_in_certified_STEP1" if feature == "relative_topix20" else None)
        eligibility.append({"feature": feature, "source": source, "eligible": exclusion is None,
                            "dependency_rows": lag, "availability": "confirmation_close",
                            "price_policy": "constant_adjustment_factor_in_window" if adjustment else "same_day_scale_invariant",
                            "excluded_reason": exclusion, "valid": int(samples[feature].notna().sum()), "missing": int(samples[feature].isna().sum())})
    save(out, "feature_eligibility", eligibility, ["feature"])
    save(out, "univariate_distributions", describe(samples), ["kind", "feature", "outcome"])
    missing = []
    for kind, df in samples.groupby("kind", sort=True):
        for f in FEATURES:
            for reason, count in df[f+"__reason"].fillna("available").value_counts().items():
                missing.append({"kind": kind, "feature": f, "reason": reason, "n": int(count)})
    save(out, "missingness", missing, ["kind", "feature", "reason"])
    fold_rows, trial_rows, membership, purges, dependency = [], [], [], [], []
    eligible = [r["feature"] for r in eligibility if r["eligible"]]
    for fold, start, end, test_start, test_end in FOLDS:
        fold_rows.append({"fold": fold, "train_start": start, "train_end": end, "test_start": test_start, "test_end": test_end})
        for kind, kdf in samples.groupby("kind", sort=True):
            for feature in eligible:
                tr = kdf[interval_mask(kdf, start, end, feature)]
                definition = fit(tr, feature)
                for phase, lo, hi in (("train", start, end), ("test", test_start, test_end)):
                    scope = kdf[kdf.signal_date.between(lo, hi)]
                    validmask = interval_mask(scope, lo, hi, feature)
                    frame = scope[validmask]
                    trial_id = f"f{fold}|{kind}|{feature}|{phase}"
                    metrics, kept = evaluation(frame, feature, definition)
                    trial_rows.append({"trial_id": trial_id, "fold": fold, "kind": kind, "feature": feature, "phase": phase,
                                       "lo": lo, "hi": hi, **definition, **metrics, "purged": int((~validmask).sum())})
                    for row in scope[~validmask].itertuples(index=False):
                        rec = row._asdict()
                        reasons = []
                        if pd.isna(row.candidate_dependency_start) or row.candidate_dependency_start < pd.Timestamp(lo):
                            reasons.append("candidate_or_recursive_ATR_dependency_crosses_start")
                        if pd.isna(rec[feature+"__start"]) or rec[feature+"__start"] < pd.Timestamp(lo):
                            reasons.append("feature_dependency_crosses_start")
                        if row.eval_end_date > pd.Timestamp(hi):
                            reasons.append("target_crosses_end")
                        purges.append({"trial_id": trial_id, "signal_id": row.signal_id, "reason": ";".join(reasons),
                                       "signal_date": row.signal_date, "candidate_dependency_start": row.candidate_dependency_start,
                                       "feature_dependency_start": rec[feature+"__start"], "target_end": row.eval_end_date})
                    for sid, flag, missing_value in zip(frame.signal_id, kept, frame[feature].isna(), strict=True):
                        membership.append({"trial_id": trial_id, "signal_id": sid, "passes": bool(flag), "missing_feature": bool(missing_value)})
                    if phase == "test":
                        for axis in ("Ticker", "signal_date"):
                            for key, group in frame.assign(passes=kept).groupby(axis, sort=True):
                                dependency.append({"trial_id": trial_id, "axis": axis, "cluster": str(key), "n": len(group),
                                                   "retained": int(group.passes.sum()), "success5": int(group.hit_5.sum()),
                                                   "retained_success5": int(group.loc[group.passes, "hit_5"].sum())})
    trials = pd.DataFrame(trial_rows)
    save(out, "fold_definitions", fold_rows, ["fold"])
    save(out, "all_trials", trials, ["trial_id"])
    save(out, "trial_membership", membership, ["trial_id", "signal_id"])
    save(out, "boundary_exclusions", purges, ["trial_id", "signal_id"])
    save(out, "cluster_dependency", dependency, ["trial_id", "axis", "cluster"])
    save(out, "baseline", [{"kind": k, **stats(g)} for k, g in samples.groupby("kind", sort=True)], ["kind"])
    hypotheses = []
    for (kind, feature), group in trials[trials.phase == "test"].groupby(["kind", "feature"], sort=True):
        same = group.direction.nunique() == 1 and not group.direction.eq("unavailable").any()
        adequate = (group.retained_n.ge(100) & group.retained_dates.ge(30) & group.retained_tickers.ge(30)).all()
        benefit = group.simultaneous_lower_delta5.gt(0).all()
        fixed = fit(samples[samples.kind == kind], feature)
        status = "fixed_hypothesis_for_used_validation_only" if same and adequate and benefit else "no_additional_filter_research_hold"
        hypotheses.append({"kind": kind, "feature": feature, "status": status, "consistent_direction": same,
                           "adequate_three_folds": bool(adequate), "positive_simultaneous_lower_all_folds": bool(benefit),
                           "positive_point_estimate_folds": int(group.delta_rate5.gt(0).sum()), **fixed})
    save(out, "frozen_hypotheses", hypotheses, ["kind", "feature"])
    audit_rows = [
        {"check": "post2023_values_in_samples", "count": int(samples.signal_date.gt(CUTOFF).sum() + samples.eval_end_date.gt(CUTOFF).sum())},
        {"check": "sample_key_duplicates", "count": int(samples.signal_id.duplicated().sum())},
        {"check": "future_fields_in_predictors", "count": len(set(FEATURES) & {"hit_5", "hit_10", "mae", "close_return", "net_close_return"})},
        {"check": "retained_trials_boundary_crossings", "count": 0},
    ]
    # Re-check retained membership against actual dates, independently of saved counts.
    mf = pd.DataFrame(membership).merge(samples, on="signal_id", validate="many_to_one")
    by_trial = {r["trial_id"]: r for r in trial_rows}
    crossing = 0
    for tid, group in mf.groupby("trial_id", sort=True):
        r = by_trial[tid]
        crossing += int((~interval_mask(group, r["lo"], r["hi"], r["feature"])).sum())
        assert len(group) == r["baseline_n"] and int(group.passes.sum()) == r["retained_n"]
    audit_rows[-1]["count"] = crossing
    assert not any(r["count"] for r in audit_rows)
    save(out, "period_audit", audit_rows, ["check"])
    return {"samples": len(samples), "tickers": int(samples.Ticker.nunique()), "source_confirmation_signals_formation": len(meta),
            "incomplete_or_boundary_labels": len(incomplete), "eligible_features": len(eligible), "excluded_features": 8-len(eligible),
            "feature_sample_missing_rate": float(samples[eligible].isna().to_numpy().mean()),
            "trial_rows": len(trials), "test_trials": int(trials.phase.eq("test").sum()), "purged_trial_sample_rows": len(purges),
            "hypotheses_for_used_validation": sum(r["status"] == "fixed_hypothesis_for_used_validation_only" for r in hypotheses),
            "source_oldest": "2021-08-19", "sample_first": str(samples.signal_date.min().date()), "sample_last": str(samples.signal_date.max().date()),
            "input_tickers": int(meta.Ticker.nunique()), "post2023_value_rows": 0,
            "audit": audit_rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="data/distortions_authenticated_inputs")
    parser.add_argument("--root", default="data/market_history/analysis/bottom_pullback_distortions_v1")
    parser.add_argument("--restore", action="store_true")
    a = parser.parse_args()
    work, root = Path(a.work).resolve(), Path(a.root).resolve()
    work.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.rglob("*.parquet")):
        raise RuntimeError("Existing research output must not be overwritten")
    quality = root / "quality"
    dump(quality / "frozen_specification.json", SPEC)
    dump(quality / "specification_hash.json", {"sha256": sha(quality / "frozen_specification.json")})
    if a.restore:
        restore(work)
    auth = authenticate(work)
    print("Pinned inputs authenticated; beginning formation-only run 1", flush=True)
    dump(quality / "input_authentication.json", auth)
    before = {label: manifest(work / label) for label in PINS}
    dump(quality / "input_hashes_before.json", before)
    first = run(work, root / "run_1")
    print("Run 1 finished; beginning independent formation-only run 2", flush=True)
    second = run(work, root / "run_2")
    m1, m2 = manifest(root / "run_1"), manifest(root / "run_2")
    after = {label: manifest(work / label) for label in PINS}
    dump(quality / "input_hashes_after.json", after)
    assert before == after, "input modified"
    assert first == second and m1 == m2, "independent rerun mismatch"
    dump(quality / "reproducibility.json", {"quality": "PASS", "independent_runs": 2, "run_1": m1, "run_2": m2})
    report = {"quality": "PASS", "execution": "complete", **first, "inputs_changed": False,
              "independent_two_run_reproducibility": "PASS", "output_parquet_sha256": m1,
              "parquet_bytes_run1": sum(p.stat().st_size for p in (root / "run_1").rglob("*.parquet")),
              "adoption": "none", "untouched_oos": False,
              "risks": ["virtual next-open entry; daily high hit is not realised profit or verified opening execution",
                        "all post2023 periods previously viewed and not untouched OOS; no such sample values loaded here",
                        "strict recursive ATR dependency purge can leave pullback later folds underpowered",
                        "complete-window conditioning creates selection bias; unknown outcomes never recoded failure",
                        "survivorship and retrospectively adjusted source prices remain; no vintage market-data archive",
                        "approximate maximum single-axis clustered uncertainty, not proof against all joint or serial dependence",
                        "48-comparison normal bounds are conservative diagnostics; no automatic signal adoption"]}
    dump(quality / "report.json", report)
    t = pd.read_parquet(root / "run_1/all_trials/part.parquet")
    cols = ["fold", "kind", "feature", "direction", "baseline_n", "retained_n", "baseline_rate5", "retained_rate5", "retained_rate10", "missed_success_rate5", "delta_mean_mae", "delta_mean_net_close_return", "simultaneous_lower_delta5"]
    md = "# 底打ち・押し目の歪み：形成期内の追加価値\n\n品質PASS。採用判断とは別です。完全未使用OOSではありません。\n\n"
    md += "形成期の完全評価可能シグナルだけを使用。2024年以降の行の価格・特徴量・目的変数は分析データフレームへ読み込んでいません。全期間件数は固定ZIP・実Parquetハッシュ・既存認証レポートで照合し、期間外目的変数の再走査はしていません。\n\n"
    md += json.dumps(first, ensure_ascii=False, indent=2) + "\n\n"
    md += t.loc[t.phase == "test", cols].to_markdown(index=False) + "\n\n"
    md += "純10日終値損益は原本の往復0.4%控除済み値。到達率の分母は各区間の境界条件を満たす完全評価可能な基準シグナル全件で、特徴量欠損も基準分母に保持。フィルター欠損は不通過とし、有効件数を別記。\n\n"
    md += "候補形成は85行の遡及窓を含み、押し目はWilder ATRの起点まで含めてパージ。正の点推定だけで追加フィルターを採用しません。全試行・除外行・銘柄/日付別件数を保存。\n\n"
    md += "\n".join("- " + risk for risk in report["risks"])
    (quality / "REPORT.md").write_text(md + "\n")
    prompt_source = Path(__file__).resolve().parents[1] / "research/BOTTOM_PULLBACK_DISTORTIONS_NEXT_PROMPT.md"
    (quality / "NEXT_PROMPT.md").write_text(prompt_source.read_text())
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        root = Path(sys.argv[sys.argv.index("--root")+1]) if "--root" in sys.argv else Path("data/market_history/analysis/bottom_pullback_distortions_v1")
        dump(root / "quality/FAIL.json", {"quality": "FAIL", "reason": str(exc), "not_completed": True})
        raise
