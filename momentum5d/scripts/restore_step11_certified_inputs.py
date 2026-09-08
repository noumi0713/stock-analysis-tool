"""Restore the immutable inputs required by STEP11 V2.

This restoration is deliberately fail-closed.  The 2025-09-08 onward slice is
not an untouched OOS sample; the downstream builder labels it
``contaminated_validation`` throughout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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
        "step10_v2",
        34255527308,
        10067805811,
        "step10-v2-exit-analysis",
        "a149ea5622f144c4cc9f130f51b11ee416e6be3aa3ca6ea7bdc1fe5ce4abeb00",
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


def download(repository: str, artifact_id: int, destination: Path) -> None:
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise RuntimeError("GH_TOKEN is required for pinned artifact restoration")
    url = f"https://api.github.com/repos/{repository}/actions/artifacts/{artifact_id}/zip"
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
                    url,
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
        raise RuntimeError("Pinned STEP11 inputs belong to noumi0713/stock-analysis-tool")

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="step11_pinned_") as temporary:
        for key, run_id, artifact_id, name, expected_sha in ARTIFACTS:
            metadata = gh_json(f"/repos/{args.repository}/actions/artifacts/{artifact_id}")
            if (
                metadata.get("name") != name
                or (metadata.get("workflow_run") or {}).get("id") != run_id
                or metadata.get("expired")
            ):
                raise RuntimeError(f"Pinned artifact identity mismatch for {name}")
            archive = Path(temporary) / f"{key}.zip"
            download(args.repository, artifact_id, archive)
            actual_sha = sha256(archive)
            if actual_sha != expected_sha:
                raise RuntimeError(f"Pinned ZIP SHA256 mismatch for {name}")
            if metadata.get("digest") not in (None, f"sha256:{actual_sha}"):
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
                "github_digest": metadata.get("digest"),
                "archive_size_bytes": archive.stat().st_size,
                "zip_crc_check": "PASS",
                "download_verified": True,
            }

    quality = root / "quality"
    quality.mkdir(parents=True, exist_ok=True)
    provenance = {
        "repository": args.repository,
        "restore_policy": "immutable Actions artifacts only; no substitute generation",
        "analysis_label": "contaminated_validation",
        "untouched_oos_claimed": False,
        "artifacts": records,
    }
    (quality / "step11_input_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        quality = Path("data/market_history/quality")
        quality.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": 11,
            "version": 2,
            "quality": "FAIL",
            "status": "STEP11 V2 FAIL - pinned input restoration failed",
            "reason": f"{type(error).__name__}: {error}",
            "analysis_executed": False,
            "substitute_data_generated": False,
            "untouched_oos_claimed": False,
        }
        (quality / "step11_v2_report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise
