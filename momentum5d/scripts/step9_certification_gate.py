"""Read-only, fail-closed provenance checks before STEP9 analysis.

Never infer input authenticity from a PASS string, matching counts, or a hash
computed for the first time from a local test fixture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

STEP1_FINGERPRINT = "e533988050b88ec77faf8e7b9b0f6b34fabbc8445efde31a05113448a09c8fe3"
STEP2_PARQUET_SHA256 = "27ec65f1fec449f2910d11bb43a31ad3dfd54dea0858b17f4720f32df65fcfcb"
STEP7_REPORT_SHA256 = "5b32a250ac673bbfa60c7e6dcd818ada28832f4fa7a2381c8f906775a3f2f14a"
PINNED_ARTIFACTS = {
    "step1": {"run_id": 34125934863, "artifact_id": 10020110334, "name": "step1-equity-features", "archive_sha256": "5b5dc2a0903d9c37daddb67030bfaa9df749409efd44a9851f48735fb6ba3a4f"},
    "step2": {"run_id": 34130393964, "artifact_id": 10021871736, "name": "step2-forward-targets", "archive_sha256": "d00fd69adfdf3886fc4c45ada334ccb4b0a6f339c0f32a72084b1b4089c35eaf"},
    "step6_v2": {"run_id": 34183481958, "name": "step6-v2-univariate"},
    "step7_v2": {"run_id": 34185770982, "name": "step7-v2-condition-combinations"},
    "step8_v2": {"run_id": 34198158546, "name": "step8-v2-regime-analysis"},
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def step1_fingerprint(files: list[Path]) -> str:
    # Reproduce only the STEP2 source-byte fingerprint, never any targets.
    result = hashlib.sha256()
    for path in sorted(files):
        result.update(str(path.relative_to(path.parents[2])).encode())
        result.update(str(path.stat().st_size).encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                result.update(chunk)
    return result.hexdigest()


def inspect_inputs(root: Path, reproducibility_requested: bool) -> dict:
    root = root.resolve()
    errors = []
    observed = {}
    reports = {}
    if not reproducibility_requested:
        errors.append("two_independent_runs_not_requested")
    provenance_path = root / "quality/step9_input_provenance.json"
    if not provenance_path.is_file():
        errors.append("pinned_artifact_provenance_missing")
    else:
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            provenance = {}
            errors.append("pinned_artifact_provenance_unreadable")
        observed[str(provenance_path.relative_to(root))] = digest(provenance_path)
        if provenance.get("repository") != "noumi0713/stock-analysis-tool":
            errors.append("artifact_repository_mismatch")
        for name, expected in PINNED_ARTIFACTS.items():
            actual = provenance.get("artifacts", {}).get(name, {})
            for key, value in expected.items():
                if actual.get(key) != value:
                    errors.append(f"artifact_provenance_mismatch:{name}:{key}")
            if actual.get("download_verified") is not True:
                errors.append(f"artifact_download_not_verified:{name}")
    for name in ("step1", "step2", "step7_v2", "step8_v2"):
        path = root / "quality" / f"{name}_report.json"
        if not path.is_file():
            errors.append(f"{name}_certification_report_missing")
            continue
        try:
            reports[name] = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            errors.append(f"{name}_certification_report_unreadable")
            continue
        observed[str(path.relative_to(root))] = digest(path)
        if reports[name].get("quality") != "PASS":
            errors.append(f"{name}_certification_not_PASS")
    step1_files = sorted((root / "features/equity_daily_features").rglob("*.parquet"))
    if not step1_files:
        errors.append("certified_step1_parquet_missing")
    else:
        actual = step1_fingerprint(step1_files)
        observed["step1_legacy_byte_fingerprint"] = actual
        if actual != STEP1_FINGERPRINT:
            errors.append("step1_bytes_do_not_match_certified_step2_source")
    target = root / "targets/equity_daily_forward_targets/part.parquet"
    if not target.is_file():
        errors.append("certified_step2_parquet_missing")
    else:
        actual = digest(target)
        observed[str(target.relative_to(root))] = actual
        if actual != STEP2_PARQUET_SHA256:
            errors.append("step2_bytes_do_not_match_pinned_artifact")
    if reports.get("step2", {}).get("input_fingerprint") != STEP1_FINGERPRINT:
        errors.append("step2_source_fingerprint_mismatch")
    if observed.get("quality/step7_v2_report.json") != STEP7_REPORT_SHA256:
        errors.append("step7_report_bytes_mismatch")
    for name, folder in (("step7_v2", "step7_condition_combinations_v2"),
                         ("step8_v2", "step8_regime_analysis_v2")):
        report = reports.get(name, {})
        certified = report.get("first_output_manifest", {})
        second = report.get("second_output_manifest", {})
        if not certified or certified != second or report.get("reproducibility_passed") is not True:
            errors.append(f"{name}_certified_reproducibility_missing")
        base = root / "analysis" / folder
        actual_paths = {str(p.relative_to(base)) for p in base.rglob("*.parquet")}
        if actual_paths != set(certified):
            errors.append(f"{name}_artifact_file_set_mismatch")
        for relative, expected in sorted(certified.items()):
            path = base / relative
            if not path.is_file():
                continue
            actual = digest(path)
            observed[str(path.relative_to(root))] = actual
            if actual != expected:
                errors.append(f"{name}_artifact_hash_mismatch:{relative}")
    sample = root / "analysis/step6_univariate_v2/analysis_samples/part.parquet"
    certified_sample = reports.get("step7_v2", {}).get("input_manifest_before", {}).get(
        "step6_v2/analysis_samples/part.parquet", {}).get("sha256")
    if not sample.is_file() or not certified_sample:
        errors.append("certified_fixed_definition_sample_values_missing")
    else:
        actual = digest(sample)
        observed[str(sample.relative_to(root))] = actual
        if actual != certified_sample:
            errors.append("step6_sample_bytes_mismatch_with_step7_input")
    return {
        "step": 9, "version": 2, "stage": "input_certification_only",
        "quality": "PASS" if not errors else "FAIL",
        "analysis_executed": False,
        "errors": errors, "observed_input_hashes": observed,
        "steps1_to_8_recalculated": False,
        "step2_target_values_recalculated": False,
        "reproducibility_requested": reproducibility_requested,
        "reproducibility_passed": None,
        "completion_statement": None,
    }


def require_certified_inputs(root: Path, reproducibility_requested: bool) -> None:
    report = inspect_inputs(root, reproducibility_requested)
    if report["quality"] != "PASS":
        raise RuntimeError("STEP9 input certification FAIL: " + "; ".join(report["errors"]))


def save_failure_report(root: Path, report: dict, destination: Path) -> None:
    if report["quality"] != "FAIL":
        raise RuntimeError("A preflight pass alone must not certify STEP9 outputs")
    after = inspect_inputs(root, report["reproducibility_requested"])
    unchanged = report["observed_input_hashes"] == after["observed_input_hashes"]
    result = {
        **report,
        "status": "STEP9 V2 FAIL - stopped before performance evaluation",
        "input_manifest_before": report["observed_input_hashes"],
        "input_manifest_after": after["observed_input_hashes"],
        "inputs_unchanged_during_this_audit": unchanged,
        "source_directory_audited": str(root.resolve()),
        "formation": {"start": "data_start", "end": "2023-12-31"},
        "validation": {"start": "2024-01-01", "end": "2025-09-07"},
        "contaminated_holdout_start": "2025-09-08",
        "new_untouched_oos_start": None,
        "upstream_reported_counts_not_step9_results": {
            "input_pairs": 8611, "events": 8611, "controls": 8611,
            "tickers": 2637, "fixed_features": 24, "fixed_combinations": 2300,
            "step8_environment_variables": 9,
        },
        "performance_metrics": {
            "signals": None, "fills": None, "unfilled": None,
            "fill_rate": None, "formation_results": None,
            "validation_results": None, "missing_rate": None,
            "boundary_violations": None, "future_information_leaks": None,
            "duplicate_keys": None, "parquet_bytes": None,
        },
        "performance_checks_status": "not_run_due_to_input_certification_failure",
        "step9_analysis_parquet_created": False,
        "step10_executed": False,
        "step1_original_artifact": {
            "run_id": 34125934863, "artifact_id": 10020110334,
            "name": "step1-equity-features", "bytes": 764317136,
            "archive_sha256": "5b5dc2a0903d9c37daddb67030bfaa9df749409efd44a9851f48735fb6ba3a4f",
            "local_download_blocker": "connector_archive_size_limit_536870912_bytes",
        },
        "step2_original_zip_verified": {
            "archive_sha256": "d00fd69adfdf3886fc4c45ada334ccb4b0a6f339c0f32a72084b1b4089c35eaf",
            "parquet_sha256": STEP2_PARQUET_SHA256,
            "zip_crc_check": "PASS",
        },
        "residual_risks": [
            "Local test fixtures are not certified STEP1/STEP2 inputs and must never be reused.",
            "Full STEP9 implementation and two-run reproducibility remain unverified.",
            "The draft requires review of holdout price loading, signal membership outputs and entry tradability before a production run.",
            "A first-pullback condition known only at the closing print does not establish executable same-close fills; the requested rule is at most an idealized comparison unless execution feasibility is established.",
            "Case-control returns cannot be interpreted as unconditional strategy performance.",
        ],
    }
    if not unchanged:
        result["errors"].append("input_changed_during_preflight_audit")
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "step9_v2_report.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    markdown = """# STEP9 V2：FAIL（入力認証で停止）

STEP9の成績比較は未実施です。これは戦略成績のFAILではなく入力認証のFAILです。

## 確認した問題

- ローカル試験用STEP1は原本と不一致で、認証レポートもありません。
- ローカル試験用STEP2は認証済み原本とハッシュが一致しません。
- STEP7 V2とSTEP8 V2のローカル成果物は各認証マニフェストと一致しました。
- STEP2の元ZIPはSHA256とZIP整合性を確認済みです。
- STEP1原本は764,317,136 bytes。取得経路の536,870,912 bytes上限を超え、未復元です。

試験用フォルダはstep9-test-fixtures-UNTRUSTEDへ名前を変えて隔離しました。
削除していません。試験用に生成された価格・目的変数の結果は分析へ使用禁止です。

## 対象と未実施項目

形成期：データ開始日〜2023-12-31。
検証期：2024-01-01〜2025-09-07。
2025-09-08以降はcontaminated_holdoutであり、使用禁止です。

上流認証レポート記載：8,611ペア、イベント8,611件、対照8,611件、2,637銘柄、
24固定特徴量、2,300固定組み合わせ、9環境変数。STEP9の実績ではありません。

STEP9のシグナル数、約定数、未約定数、約定率、平均・中央値、MFE/MAE、
欠損・除外件数、期間越境、未来情報混入、重複件数、成績の2回実行再現性は未測定です。
未測定値を0件やPASSへ置き換えていません。分析Parquetは作成していません。

今回の認証監査では読み取り前後の入力ハッシュを照合しています。
STEP1〜STEP8原本の再計算・変更、STEP10の実行は行っていません。

## 残存リスク・再開条件

認証済みSTEP1・STEP2原本を復元して照合し、STEP9実装レビューを完了してから
本計算を2回実行する必要があります。試験データでは代用できません。
初押し当日終値で条件確定して同終値約定する想定は、実際の約定可能性が未証明です。
ユーザー指定定義は勝手に翌日始値へ変更せず、理想化した比較と実約定を区別してください。
ケース・コントロール標本の成績は実市場の無条件成績ではありません。

STEP9が認証PASSになるまでSTEP10へ進めません。
"""
    with (destination / "STEP9_V2_REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write(markdown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--verify-reproducibility", action="store_true")
    parser.add_argument("--failure-report-directory")
    args = parser.parse_args()
    result = inspect_inputs(Path(args.root), args.verify_reproducibility)
    if args.failure_report_directory:
        save_failure_report(Path(args.root), result, Path(args.failure_report_directory))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["quality"] == "PASS" else 1)
