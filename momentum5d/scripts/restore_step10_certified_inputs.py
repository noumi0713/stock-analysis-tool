# ruff: noqa: E501
"""Restore only the pinned inputs required by STEP10 V2.

The archives are identified by immutable Actions run/artifact ids and verified
before extraction.  Failure is closed: no substitute data is generated.
"""
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
    (
        "step1",
        34125934863,
        10020110334,
        "step1-equity-features",
        "5b5dc2a0903d9c37daddb67030bfaa9df749409efd44a9851f48735fb6ba3a4f",
    ),
    (
        "step7_v2",
        34185770982,
        10040503513,
        "step7-v2-condition-combinations",
        "27cd624d2b3ecc9a199f02bff76389653820b1f062ef83145c82bddc7973dd43",
    ),
    (
        "step9_v2",
        34216003204,
        10053710484,
        "step9-v2-entry-timing",
        "9f6432cb6048bd67a8ca95e3b3bb8968f9965503caa7f7a4d165b387897d1d37",
    ),
)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def gh_json(endpoint: str) -> dict:
    output = subprocess.run(
        ["gh", "api", "-H", "Accept: application/vnd.github+json", endpoint],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return json.loads(output)


def resolve_artifact(repository: str, run_id: int, artifact_id: int, name: str) -> dict:
    metadata = gh_json(f"/repos/{repository}/actions/artifacts/{artifact_id}")
    actual_run = (metadata.get("workflow_run") or {}).get("id")
    if metadata.get("name") != name or actual_run != run_id or metadata.get("expired"):
        raise RuntimeError(f"Pinned artifact identity mismatch for {name}")
    return metadata


def download(repository: str, artifact_id: int, destination: Path) -> None:
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise RuntimeError("GH_TOKEN is required for pinned artifact restoration")
    api_url = f"https://api.github.com/repos/{repository}/actions/artifacts/{artifact_id}/zip"
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, 7):
        try:
            subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--location",
                    "--silent",
                    "--show-error",
                    "--retry",
                    "8",
                    "--retry-all-errors",
                    "--retry-delay",
                    "3",
                    "--continue-at",
                    "-",
                    "--output",
                    str(destination),
                    "--header",
                    f"Authorization: Bearer {token}",
                    "--header",
                    "Accept: application/vnd.github+json",
                    "--header",
                    "X-GitHub-Api-Version: 2022-11-28",
                    api_url,
                ],
                check=True,
            )
            return
        except subprocess.CalledProcessError as error:
            last_error = error
            if attempt < 6:
                time.sleep(5 * attempt)
    assert last_error is not None
    raise last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/market_history")
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY", "noumi0713/stock-analysis-tool"),
    )
    args = parser.parse_args()
    if args.repository != "noumi0713/stock-analysis-tool":
        raise RuntimeError("STEP10 pinned artifacts belong to noumi0713/stock-analysis-tool")
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="step10_pinned_") as temporary:
        for key, run_id, artifact_id, name, expected_sha in ARTIFACTS:
            metadata = resolve_artifact(args.repository, run_id, artifact_id, name)
            archive = Path(temporary) / f"{key}.zip"
            download(args.repository, artifact_id, archive)
            actual_sha = sha256(archive)
            if actual_sha != expected_sha:
                raise RuntimeError(f"Pinned ZIP SHA256 mismatch for {name}")
            metadata_digest = metadata.get("digest")
            if metadata_digest and metadata_digest != f"sha256:{actual_sha}":
                raise RuntimeError(f"GitHub artifact digest mismatch for {name}")
            with zipfile.ZipFile(archive) as bundle:
                corrupt = bundle.testzip()
                if corrupt is not None:
                    raise RuntimeError(f"ZIP CRC failure for {name}: {corrupt}")
                bundle.extractall(root)
            records[key] = {
                "run_id": run_id,
                "artifact_id": artifact_id,
                "name": name,
                "archive_sha256": actual_sha,
                "expected_archive_sha256": expected_sha,
                "github_digest": metadata_digest,
                "archive_size_bytes": archive.stat().st_size,
                "zip_crc_check": "PASS",
                "verification_method": "resumable_downloaded_zip_sha256_plus_crc",
                "download_verified": True,
            }
    provenance = {
        "repository": args.repository,
        "restore_policy": "pinned GitHub Actions artifacts only; no substitute generation",
        "artifacts": records,
    }
    quality = root / "quality"
    quality.mkdir(parents=True, exist_ok=True)
    (quality / "step10_input_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
            "step": 10,
            "version": 2,
            "status": "STEP10 V2 FAIL - pinned artifact restoration failed",
            "quality": "FAIL",
            "failure_stage": "pinned_artifact_restoration",
            "reason": f"{type(error).__name__}: {error}",
            "analysis_executed": False,
            "substitute_data_generated": False,
            "steps1_to_9_recalculated": False,
            "step11_executed": False,
            "completion_statement": None,
        }
        (quality / "step10_v2_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (quality / "STEP10_V2_REPORT.md").write_text(
            "# STEP10 V2 FAIL\n\n"
            "Pinned input restoration failed, so exit evaluation was not started.\n\n"
            f"Reason: `{type(error).__name__}: {error}`\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False), file=sys.stderr)
        raise
