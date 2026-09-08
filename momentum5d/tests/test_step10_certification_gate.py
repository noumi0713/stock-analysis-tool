from pathlib import Path

from scripts.step10_certification_gate import inspect_inputs


def test_missing_inputs_fail_closed(tmp_path: Path) -> None:
    report = inspect_inputs(tmp_path, reproducibility_requested=True)
    assert report["quality"] == "FAIL"
    assert "pinned_artifact_provenance_missing" in report["errors"]
    assert "certified_step1_parquet_missing" in report["errors"]


def test_two_independent_runs_are_required(tmp_path: Path) -> None:
    report = inspect_inputs(tmp_path, reproducibility_requested=False)
    assert "two_independent_runs_not_requested" in report["errors"]
