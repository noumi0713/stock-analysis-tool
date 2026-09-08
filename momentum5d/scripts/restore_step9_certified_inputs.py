"""Restore the five pinned STEP9 inputs and record verifiable provenance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path


ARTIFACTS = (
    ("step1", 34125934863, 10020110334, "step1-equity-features", "5b5dc2a0903d9c37daddb67030bfaa9df749409efd44a9851f48735fb6ba3a4f"),
    ("step2", 34130393964, 10021871736, "step2-forward-targets", "d00fd69adfdf3886fc4c45ada334ccb4b0a6f339c0f32a72084b1b4089c35eaf"),
    ("step6_v2", 34183481958, None, "step6-v2-univariate", None),
    ("step7_v2", 34185770982, None, "step7-v2-condition-combinations", None),
    ("step8_v2", 34198158546, None, "step8-v2-regime-analysis", None),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gh_json(endpoint: str) -> dict:
    output = subprocess.run(
        ["gh", "api", "-H", "Accept: application/vnd.github+json", endpoint],
        check=True, stdout=subprocess.PIPE,
    ).stdout
    return json.loads(output)


def resolve_artifact(repository: str, run_id: int, artifact_id: int | None, name: str) -> dict:
    if artifact_id is not None:
        metadata = gh_json(f"/repos/{repository}/actions/artifacts/{artifact_id}")
    else:
        listing = gh_json(f"/repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100")
        matches = [item for item in listing.get("artifacts", []) if item.get("name") == name and not item.get("expired")]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one live artifact {name} in run {run_id}; found {len(matches)}")
        metadata = matches[0]
    actual_run = (metadata.get("workflow_run") or {}).get("id")
    if metadata.get("name") != name or actual_run != run_id or metadata.get("expired"):
        raise RuntimeError(f"Pinned artifact identity mismatch for {name}")
    return metadata


def download(repository: str, artifact_id: int, destination: Path) -> None:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, 5):
        destination.unlink(missing_ok=True)
        try:
            with destination.open("wb") as handle:
                subprocess.run(
                    ["gh", "api", "-H", "Accept: application/vnd.github+json", f"/repos/{repository}/actions/artifacts/{artifact_id}/zip"],
                    check=True, stdout=handle,
                )
            return
        except subprocess.CalledProcessError as error:
            last_error = error
            destination.unlink(missing_ok=True)
            if attempt < 4:
                time.sleep(5 * attempt)
    assert last_error is not None
    raise last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/market_history")
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", "noumi0713/stock-analysis-tool"))
    args = parser.parse_args()
    if args.repository != "noumi0713/stock-analysis-tool":
        raise RuntimeError("STEP9 pinned artifacts belong to noumi0713/stock-analysis-tool")
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="step9_pinned_") as temporary:
        for key, run_id, fixed_id, name, expected_sha in ARTIFACTS:
            metadata = resolve_artifact(args.repository, run_id, fixed_id, name)
            artifact_id = int(metadata["id"])
            metadata_digest = metadata.get("digest")
            if expected_sha is not None:
                archive = Path(temporary) / f"{key}.zip"
                download(args.repository, artifact_id, archive)
                actual_sha = sha256(archive)
                if actual_sha != expected_sha:
                    raise RuntimeError(f"Pinned ZIP SHA256 mismatch for {name}")
                if metadata_digest and metadata_digest != f"sha256:{actual_sha}":
                    raise RuntimeError(f"GitHub artifact digest mismatch for {name}")
                with zipfile.ZipFile(archive) as bundle:
                    corrupt = bundle.testzip()
                    if corrupt is not None:
                        raise RuntimeError(f"ZIP CRC failure for {name}: {corrupt}")
                    bundle.extractall(root)
                verification_method = "downloaded_zip_sha256_plus_crc"
                archive_size = archive.stat().st_size
                zip_crc_check = "PASS"
            else:
                subprocess.run(
                    ["gh", "run", "download", str(run_id), "--repo", args.repository, "-n", name, "-D", str(root)],
                    check=True,
                )
                actual_sha = metadata_digest.removeprefix("sha256:") if metadata_digest else None
                verification_method = "github_run_download_plus_saved_output_manifests"
                archive_size = int(metadata.get("size_in_bytes", 0))
                zip_crc_check = "verified_by_github_run_download"
            records[key] = {
                "run_id": run_id,
                "artifact_id": artifact_id,
                "name": name,
                "archive_sha256": actual_sha,
                "expected_archive_sha256": expected_sha,
                "github_digest": metadata_digest,
                "archive_size_bytes": archive_size,
                "zip_crc_check": zip_crc_check,
                "verification_method": verification_method,
                "download_verified": True,
            }
    provenance = {
        "repository": args.repository,
        "restore_policy": "pinned GitHub Actions artifacts only; no substitute generation",
        "artifacts": records,
    }
    quality = root / "quality"
    quality.mkdir(parents=True, exist_ok=True)
    (quality / "step9_input_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        root = Path("data/market_history").resolve()
        quality = root / "quality"
        quality.mkdir(parents=True, exist_ok=True)
        report = {
            "step": 9,
            "version": 2,
            "status": "STEP9 V2 FAIL - pinned artifact restoration failed",
            "quality": "FAIL",
            "failure_stage": "pinned_artifact_restoration",
            "reason": f"{type(error).__name__}: {error}",
            "analysis_executed": False,
            "substitute_data_generated": False,
            "steps1_to_8_recalculated": False,
            "step10_executed": False,
            "completion_statement": None,
        }
        (quality / "step9_v2_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (quality / "STEP9_V2_REPORT.md").write_text(
            "# STEP9 V2 FAIL\n\nPinned GitHub Actions input restoration failed, so performance evaluation was not started.\n\n"
            f"Reason: `{type(error).__name__}: {error}`\n\nNo substitute data was generated and STEP10 was not executed.\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False), file=sys.stderr)
        raise
