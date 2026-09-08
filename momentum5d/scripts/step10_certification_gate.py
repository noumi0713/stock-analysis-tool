# ruff: noqa: E501
"""Fail-closed certification checks for STEP10 V2 inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

STEP1_FINGERPRINT = "e533988050b88ec77faf8e7b9b0f6b34fabbc8445efde31a05113448a09c8fe3"
STEP7_REPORT_SHA256 = "5b32a250ac673bbfa60c7e6dcd818ada28832f4fa7a2381c8f906775a3f2f14a"
STEP9_REPORT_SHA256 = "8b4fb535dd8fcc616be785e0b40b5984be8e4234d6f566d3d0f2cfca1c449625"
PINNED_ARTIFACTS = {
    "step1": {
        "run_id": 34125934863,
        "artifact_id": 10020110334,
        "name": "step1-equity-features",
        "archive_sha256": "5b5dc2a0903d9c37daddb67030bfaa9df749409efd44a9851f48735fb6ba3a4f",
    },
    "step7_v2": {
        "run_id": 34185770982,
        "artifact_id": 10040503513,
        "name": "step7-v2-condition-combinations",
        "archive_sha256": "27cd624d2b3ecc9a199f02bff76389653820b1f062ef83145c82bddc7973dd43",
    },
    "step9_v2": {
        "run_id": 34216003204,
        "artifact_id": 10053710484,
        "name": "step9-v2-entry-timing",
        "archive_sha256": "9f6432cb6048bd67a8ca95e3b3bb8968f9965503caa7f7a4d165b387897d1d37",
    },
}
EXPECTED_STEP9_PARQUETS = {
    "annual_metrics/part.parquet": "438ee603cbf3e015c27f8633c3e201bf774bd0142798d7276b7414d82b3bbbfe",
    "boundary_exclusions/part.parquet": "cdc36dc3b73426e8a59c8b2f65e2a85cad34a2b5c43eab3ce3cab0351d1b085a",
    "entry_definitions/part.parquet": "1dae06b08ff401187b45d951a33e920f457b4c48679bc99d60885f49ec840254",
    "entry_observations/part.parquet": "1b5cf41712d6ae43d65bb48139ec87226c4ccb662c4e779aaf9f94b2f40afc8f",
    "entry_timing_metrics/part.parquet": "944ba56e6d4e51beae0fe29c6ae4489bbac1402c157766e6e0c5d3c94522b1c7",
    "paired_timing_comparison/part.parquet": "911b1895b54375b54f42a34bc7f535e7cebe2389c2c1aaa4a787386fe0ac175a",
    "period_assignment_audit/part.parquet": "1ef622f15310b0e5dd3adce089c07fdbc4bf9424349ffb81fe6269f11acf5f69",
    "reference_summary/part.parquet": "fd553dcaec43bb4515410dd93718153da21bc85e7f44d4645556c3ea0ad42387",
    "signal_membership/part.parquet": "aa5984436feb4a42f9c31433c34ca2ec457d7b188a27fd311b54833439008fbc",
    "timing_stability/part.parquet": "2d5d9b1ef4e396341ccc76cf4f4c76f5c0a4ba59db78c1ed111656e48abdf4f0",
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def step1_fingerprint(files: list[Path]) -> str:
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
    errors: list[str] = []
    observed: dict[str, str] = {}
    if not reproducibility_requested:
        errors.append("two_independent_runs_not_requested")

    provenance_path = root / "quality/step10_input_provenance.json"
    provenance: dict = {}
    if not provenance_path.is_file():
        errors.append("pinned_artifact_provenance_missing")
    else:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        observed[str(provenance_path.relative_to(root))] = digest(provenance_path)
        if provenance.get("repository") != "noumi0713/stock-analysis-tool":
            errors.append("artifact_repository_mismatch")
        for name, expected in PINNED_ARTIFACTS.items():
            actual = provenance.get("artifacts", {}).get(name, {})
            for key, value in expected.items():
                if actual.get(key) != value:
                    errors.append(f"artifact_provenance_mismatch:{name}:{key}")
            if actual.get("download_verified") is not True or actual.get("zip_crc_check") != "PASS":
                errors.append(f"artifact_not_verified:{name}")

    step1_files = sorted((root / "features/equity_daily_features").rglob("*.parquet"))
    if not step1_files:
        errors.append("certified_step1_parquet_missing")
    else:
        actual = step1_fingerprint(step1_files)
        observed["step1_legacy_byte_fingerprint"] = actual
        if actual != STEP1_FINGERPRINT:
            errors.append("step1_bytes_do_not_match_certified_source")

    step7_report_path = root / "quality/step7_v2_report.json"
    step9_report_path = root / "quality/step9_v2_report.json"
    reports: dict[str, dict] = {}
    for name, path, expected_hash in (
        ("step7", step7_report_path, STEP7_REPORT_SHA256),
        ("step9", step9_report_path, STEP9_REPORT_SHA256),
    ):
        if not path.is_file():
            errors.append(f"{name}_report_missing")
            continue
        reports[name] = json.loads(path.read_text(encoding="utf-8"))
        actual_hash = digest(path)
        observed[str(path.relative_to(root))] = actual_hash
        if actual_hash != expected_hash:
            errors.append(f"{name}_report_hash_mismatch")
        if reports[name].get("quality") != "PASS" or not reports[name].get("reproducibility_passed"):
            errors.append(f"{name}_not_PASS_or_reproducible")

    step7_base = root / "analysis/step7_condition_combinations_v2"
    step7_report = reports.get("step7", {})
    certified7 = step7_report.get("first_output_manifest", {})
    if not certified7 or certified7 != step7_report.get("second_output_manifest"):
        errors.append("step7_manifest_not_reproducible")
    actual7_paths = {str(path.relative_to(step7_base)) for path in step7_base.rglob("*.parquet")}
    if actual7_paths != set(certified7):
        errors.append("step7_artifact_file_set_mismatch")
    for relative, expected in sorted(certified7.items()):
        path = step7_base / relative
        if not path.is_file() or digest(path) != expected:
            errors.append(f"step7_artifact_hash_mismatch:{relative}")
        elif path.is_file():
            observed[str(path.relative_to(root))] = expected

    step9_base = root / "analysis/step9_entry_timing_v2"
    step9_report = reports.get("step9", {})
    certified9 = step9_report.get("first_output_manifest", {})
    if certified9 != EXPECTED_STEP9_PARQUETS or certified9 != step9_report.get("second_output_manifest"):
        errors.append("step9_certified_manifest_mismatch")
    actual_paths = {str(path.relative_to(step9_base)) for path in step9_base.rglob("*.parquet")}
    if actual_paths != set(EXPECTED_STEP9_PARQUETS):
        errors.append("step9_artifact_file_set_mismatch")
    for relative, expected in sorted(EXPECTED_STEP9_PARQUETS.items()):
        path = step9_base / relative
        if not path.is_file() or digest(path) != expected:
            errors.append(f"step9_artifact_hash_mismatch:{relative}")
        elif path.is_file():
            observed[str(path.relative_to(root))] = expected

    metrics = step9_report.get("metrics", {})
    expected_metrics = {
        "input_pairs": 8611,
        "input_events": 8611,
        "input_controls": 8611,
        "input_tickers": 2637,
        "fixed_candidate_features": 24,
        "fixed_combinations": 2300,
        "entry_definitions": 3,
        "period_or_boundary_errors": 0,
        "contaminated_values_used": 0,
        "future_target_condition_leaks": 0,
        "major_key_duplicates": 0,
        "independent_processes_executed": 2,
        "all_fixed_combinations_preserved": True,
    }
    for key, expected in expected_metrics.items():
        if metrics.get(key) != expected:
            errors.append(f"step9_metric_mismatch:{key}")
    if any(step9_report.get("hard_failures", {}).values()):
        errors.append("step9_has_hard_failures")
    if not step9_report.get("inputs_unchanged"):
        errors.append("step9_input_change_recorded")

    try:
        entries = pd.read_parquet(step9_base / "entry_observations")
        membership = pd.read_parquet(step9_base / "signal_membership")
        definitions = pd.read_parquet(step7_base / "combination_definitions")
        candidates = pd.read_parquet(step7_base / "candidate_features")
        if len(entries) != 51666 or entries.sample_id.nunique() != 17222:
            errors.append("step9_entry_row_count_mismatch")
        unique_samples = entries.drop_duplicates("sample_id")
        if (
            unique_samples.event_id.nunique() != 8611
            or int(unique_samples.outcome.eq(1).sum()) != 8611
            or int(unique_samples.outcome.eq(0).sum()) != 8611
            or unique_samples.Ticker.nunique() != 2637
        ):
            errors.append("step9_sample_population_mismatch")
        if entries.duplicated(["sample_id", "entry_definition"]).any():
            errors.append("step9_entry_key_duplicates")
        if set(entries.entry_definition) != {"anchor_close", "next_session_open", "first_pullback"}:
            errors.append("step9_entry_definition_set_mismatch")
        if len(membership) != 2847366 or membership.combination_id.nunique() != 2300:
            errors.append("step9_membership_count_mismatch")
        if membership.duplicated(["combination_id", "sample_id"]).any():
            errors.append("step9_membership_key_duplicates")
        if len(definitions) != 2300 or set(definitions.combination_id) != set(membership.combination_id):
            errors.append("step7_step9_combination_set_mismatch")
        if int(candidates.selected_candidate.fillna(False).sum()) != 24:
            errors.append("step7_candidate_count_mismatch")
    except Exception as error:
        errors.append(f"parquet_content_check_failed:{type(error).__name__}:{error}")

    return {
        "step": 10,
        "version": 2,
        "stage": "input_certification_only",
        "quality": "PASS" if not errors else "FAIL",
        "analysis_executed": False,
        "errors": sorted(set(errors)),
        "observed_input_hashes": observed,
        "steps1_to_9_recalculated": False,
        "reproducibility_requested": reproducibility_requested,
        "completion_statement": None,
    }


def require_certified_inputs(root: Path, reproducibility_requested: bool) -> dict:
    result = inspect_inputs(root, reproducibility_requested)
    if result["quality"] != "PASS":
        raise RuntimeError("STEP10 input certification FAIL: " + "; ".join(result["errors"]))
    return result
