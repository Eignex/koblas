#!/usr/bin/env python3
"""Download a run artifact through GitHub CLI and verify its digest before extracting it."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile


def extract_verified_archive(archive, digest, destination):
    if not digest or not digest.startswith("sha256:"):
        raise ValueError("artifact has no SHA-256 digest")
    with open(archive, "rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != f"sha256:{actual}":
        raise ValueError("artifact digest mismatch")
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in entry.filename:
                raise ValueError(f"unsafe artifact path: {entry.filename}")
        bundle.extractall(destination)


def download_artifact(repository, run_id, name, destination):
    endpoint = f"repos/{repository}/actions/runs/{run_id}/artifacts"
    result = subprocess.run(
        ["gh", "api", "--paginate", "--slurp", endpoint],
        check=True, capture_output=True, text=True,
    )
    artifacts = [item for page in json.loads(result.stdout) for item in page["artifacts"]
                 if item["name"] == name and not item["expired"]]
    if len(artifacts) != 1:
        raise ValueError(f"expected one unexpired artifact named {name}, found {len(artifacts)}")
    artifact = artifacts[0]
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "artifact.zip"
        with archive.open("wb") as stream:
            subprocess.run(
                ["gh", "api", f"repos/{repository}/actions/artifacts/{artifact['id']}/zip"],
                check=True, stdout=stream,
            )
        extract_verified_archive(archive, artifact.get("digest"), destination)
    print(f"Downloaded and verified {name}: {artifact['digest']}")


if __name__ == "__main__":
    try:
        download_artifact(*sys.argv[1:])
    except (ValueError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        detail = error.stderr if isinstance(error, subprocess.CalledProcessError) and error.stderr else str(error)
        sys.exit(f"::error::{detail}")
